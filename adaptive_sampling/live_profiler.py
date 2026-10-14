"""Live adaptive profiling on real GPUs.

--strategy doptimal_cost (default): each round picks the engine
(TP, PP, token budget) with the best D-optimal gain per second, keeps it
loaded, and measures its 5 most informative (max_tokens, batch) rows.
--strategy random: 50 random rows, measured grouped by engine (the baseline).
Rows are written in the normal profiler CSV format to
live_results/<model>/<strategy>_s<seed>/.

    python live_profiler.py --model Qwen/Qwen2.5-14B-Instruct
    python live_profiler.py --model Qwen/Qwen2.5-14B-Instruct --strategy random --seed 7
    python live_profiler.py --model Qwen/Qwen2.5-14B-Instruct --n 80 --time-budget 900 \\
        --reference ../nvidia/results_zeus/run_text_only_Qwen2.5-14B-Instruct_fp16/res_native/profiling_metrics.csv
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.join(HERE, "..", "nvidia")
sys.path.insert(0, NVIDIA)
import main_profiler  # noqa: E402  (sets the vLLM/torch environment first)

import argparse  # noqa: E402
import csv  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from source.model import Model  # noqa: E402
from source.gpu_guard import assert_gpus_exclusive  # noqa: E402
from source.profiling import (  # noqa: E402
    Modality, load_model_meta, build_token_batches, build_workload, extract_item_data,
    init_csv_fields, init_csv_file, _build_model_message, _process_batch_and_write,
)
from simulator import prepare_df, fit_unified, unified_predict, ranking_metrics  # noqa: E402
from strategies import DOptimal, K, _pilot, _by_cost  # noqa: E402

ENGINE = ["tp_degree", "pp_degree", "budget"]


class _Recorder:
    """csv.writer that also keeps the last row written."""
    def __init__(self, writer):
        self.writer, self.last = writer, None

    def writerow(self, row):
        self.last = row
        self.writer.writerow(row)


class LiveSession:
    """Same interface as simulator.Simulator, but `measure` runs the model."""

    def __init__(self, args):
        self.args = args
        self.modality = Modality[args.modality.upper()]
        self.ds = main_profiler.load_modality_dataset(args.modality, args.n_items)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self.meta = load_model_meta(args.model, True, 0)
        self._build_pool()
        self.measured, self.cost, self.reloads, self.loaded = [], 0.0, 0, None
        self.reload_s, self.load_times, self.dead = 60.0, [], set()
        self.model, self.nan_streak, self.t0 = None, 0, None
        self.log = []

    def _build_pool(self):
        heads, gpus = self.meta["num_attention_heads"], torch.cuda.device_count()
        layouts = [(tp, pp) for tp in self.args.tp for pp in self.args.pp
                   if tp * pp <= gpus and (not self.meta["has_attention"] or heads % tp == 0)]
        self.batches, rows = {}, []
        for b in self.args.budgets:
            self.batches[b] = []
            for items, tokens in build_token_batches(self.ds, self.tokenizer, b, self.modality,
                                                     model_config=self.meta["model_config"]):
                self.batches[b].append((items, tokens))
                if len(self.batches[b]) == self.args.batches:
                    break
            for tp, pp in layouts:
                for m in self.args.max_tokens:
                    for j, (items, tokens) in enumerate(self.batches[b]):
                        rows.append(dict(tp_degree=tp, pp_degree=pp, budget=b, max_tokens=m, batch=j,
                                         decode_tokens=float(m),
                                         batch_size=len(items), total_input=float(tokens),
                                         measured_energy=np.nan, measured_latency=np.nan))
        self.pool = pd.DataFrame(rows)
        self.engine = list(zip(*(self.pool[c].tolist() for c in ENGINE)))
        self.latency = np.full(len(self.pool), 5.0)
        print(f"[live] {len(layouts)} layouts x {len(self.args.budgets)} budgets = "
              f"{len(set(self.engine))} engines, {len(self.pool)} candidate rows")

    def available(self):
        taken = set(self.measured)
        return [i for i in range(len(self.pool)) if i not in taken and self.engine[i][:2] not in self.dead]

    def observed(self):
        return self.pool.iloc[self.measured].dropna(subset=["measured_energy"])

    def elapsed(self):
        return time.time() - self.t0

    def _unload(self):
        if self.model is not None:
            try:
                self.model.close()
            except Exception:
                pass
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(10)

    def _load(self, e):
        self._unload()
        self.loaded, self.nan_streak = e, 0
        tp, pp, b = e
        self.heavy = {"tensor_parallel_size": tp, "pipeline_parallel_size": pp,
                      "max_num_batched_tokens": b, "dtype": "bfloat16"}
        assert_gpus_exclusive(f"before loading {e}")
        start = time.time()
        try:
            self.model = Model(name=self.args.model, max_model_len=self.args.max_model_len,
                               trust_remote_code=True, **self.heavy)
            item = extract_item_data(self.ds[0], self.modality, 0, self.meta["model_config"])
            for _ in range(2):
                self.model.generate(messages=_build_model_message(item, self.modality),
                                    sampling_params={"max_tokens": self.args.max_tokens[0]})
        except Exception as exc:
            print(f"[live] engine {e} failed to load ({type(exc).__name__}: {exc}); layout {e[:2]} excluded")
            self.dead.add(e[:2])
            self.model = None
            self._unload()
            return False
        self.load_times.append(time.time() - start)
        self.reload_s = float(np.median(self.load_times)) + 10.0
        self.reloads += 1
        return True

    def measure(self, rows, limit=np.inf):
        first = {}
        for i in rows:
            first.setdefault(self.engine[i], len(first))
        rows = sorted(rows, key=lambda i: (self.engine[i] != self.loaded, first[self.engine[i]]))
        for i in rows:
            if self.elapsed() > limit:
                return False
            e = self.engine[i]
            if e[:2] in self.dead:
                continue
            if e != self.loaded and not self._load(e):
                continue
            self._measure_row(i)
        self.cost = self.elapsed()
        return True

    def _measure_row(self, i):
        r = self.pool.iloc[i]
        items, tokens = self.batches[int(r.budget)][int(r.batch)]
        light = {"max_tokens": int(r.max_tokens)}
        workload = build_workload(prompt=items[0]["text"], heavy_cfg=self.heavy, light_cfg=light,
                                  modality=self.modality, batch_size=len(items),
                                  total_input_tokens=tokens, tokenizer=self.tokenizer)
        assert_gpus_exclusive(f"row {i}")
        _process_batch_and_write(self.model, items, self.modality, light, self.heavy, workload,
                                 self.meta["model_info"], self.fields, self.rec, self.fh)
        energy = self.rec.last[self.fields["all"].index("energy")]
        latency = self.rec.last[self.fields["all"].index("latency")]
        ok = energy not in ("", "none", None)
        self.pool.loc[i, ["measured_energy", "measured_latency"]] = (
            (float(energy), float(latency)) if ok else (np.nan, np.nan))
        self.measured.append(i)
        self.nan_streak = 0 if ok else self.nan_streak + 1
        if self.nan_streak >= 2:
            print(f"[live] engine {self.loaded} keeps failing to generate; layout {self.loaded[:2]} excluded")
            self.dead.add(self.loaded[:2])
            self._unload()
            self.loaded = None
        self._update_latency()

    def _update_latency(self):
        obs = self.observed()
        if len(obs):
            per_m = obs.groupby("max_tokens")["measured_latency"].median()
            fallback = obs["measured_latency"].median()
            self.latency = self.pool["max_tokens"].map(per_m).fillna(fallback).to_numpy(float)

    def run(self):
        os.makedirs(self.args.out, exist_ok=True)
        csv_path = os.path.join(self.args.out, "profiling_metrics.csv")
        heavy_keys = [{"tensor_parallel_size": 0, "pipeline_parallel_size": 0, "max_num_batched_tokens": 0}]
        self.fields = init_csv_fields(heavy_keys, [{"max_tokens": 0}], {})
        init_csv_file(csv_path, self.fields)
        rng = np.random.RandomState(self.args.seed)
        limit = self.args.time_budget or np.inf
        self.t0 = time.time()
        with open(csv_path, "a", newline="") as self.fh:
            self.rec = _Recorder(csv.writer(self.fh))
            for rnd, rows in enumerate(_rounds(self, rng)):
                rows = rows[: self.args.n - len(self.observed())]
                done = self.measure(rows, limit)
                engines = sorted({self.engine[i] for i in rows})
                self.log.append({"round": rnd, "engines": [list(e) for e in engines],
                                 "rows": [int(i) for i in rows], "valid": len(self.observed()),
                                 "reloads": self.reloads, "elapsed_s": round(self.elapsed(), 1)})
                print(f"[live] round {rnd}: {len(rows)} rows on {len(engines)} engine(s)  valid rows "
                      f"{len(self.observed())}/{self.args.n}  reloads {self.reloads}  {self.elapsed()/60:.1f} min")
                if not done or len(self.observed()) >= self.args.n or not self.available():
                    break
        self._unload()


def _rounds(sess, rng):
    if sess.args.strategy == "random":
        while True:
            avail = sess.available()
            need = min(sess.args.n - len(sess.observed()), len(avail))
            yield [int(i) for i in rng.choice(avail, need, replace=False)]
    yield from _pilot(sess, rng, "cost")
    scorer = DOptimal()
    while True:
        scorer.fit(sess)
        yield _by_cost(scorer, sess, K)


def evaluate_against(sess, reference):
    raw = pd.read_csv(reference).dropna(subset=["latency", "power", "energy"]).reset_index(drop=True)
    # Live runs use the max_tokens limit as decode length (a candidate row's output
    # length is unknown before it is measured), so the reference must too.
    ref = prepare_df(raw, reference).assign(decode_tokens=lambda d: d.max_tokens.astype(float))
    seen = set(zip(sess.observed()["max_tokens"], sess.observed()["batch_size"]))
    test = ref[[s not in seen for s in zip(ref["max_tokens"], ref["batch_size"])]]
    theta = fit_unified(sess.observed(), restarts=20)
    pred = unified_predict(theta, test)
    y = test["measured_energy"].to_numpy(float)
    return {"test_rows": len(test), "mape": 100 * np.mean(np.abs(pred - y) / y),
            **ranking_metrics(test, pred)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--strategy", default="doptimal_cost", choices=["doptimal_cost", "random"])
    ap.add_argument("--modality", default="text_only", choices=["text_only"])
    ap.add_argument("--n", type=int, default=50, help="valid rows to collect")
    ap.add_argument("--time-budget", type=float, default=None, help="wall-clock seconds")
    ap.add_argument("--tp", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--pp", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--budgets", type=int, nargs="+", default=main_profiler.TEXT_BUDGETS)
    ap.add_argument("--max-tokens", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--batches", type=int, default=20, help="candidate batches per budget")
    ap.add_argument("--n-items", type=int, default=1000)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reference", default=None, help="full-grid CSV to evaluate the fit against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or os.path.join(HERE, "live_results", args.model.split("/")[-1],
                                        f"{args.strategy}_s{args.seed}")

    main_profiler._acquire_output_lock(args.out)
    sess = LiveSession(args)
    sess.run()

    summary = {"model": args.model, "strategy": args.strategy, "seed": args.seed,
               "reference": os.path.abspath(args.reference) if args.reference else None, "valid_rows": len(sess.observed()), "measured_rows": len(sess.measured),
               "reloads": sess.reloads, "minutes": round(sess.elapsed() / 60, 2),
               "engine_load_s_median": round(float(np.median(sess.load_times)), 1) if sess.load_times else None,
               "excluded_layouts": [list(e) for e in sorted(sess.dead)]}
    def save():
        with open(os.path.join(args.out, "selection_log.json"), "w") as f:
            json.dump({"summary": summary, "rounds": sess.log}, f, indent=2)
    save()   # before the reference fit, so a failure there cannot lose the run
    if args.reference:
        summary["vs_reference"] = {k: round(float(v), 3) for k, v in evaluate_against(sess, args.reference).items()}
        save()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
