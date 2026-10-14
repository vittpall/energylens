"""Sampling strategies for choosing which pool rows to profile.

A strategy is a generator: it yields a round of pool indices, the simulator
measures them, and the next round can use everything measured so far.

Adaptive strategies combine a scorer with a selection mode:
  scorers  doptimal     information gain on EnergyLens's 12 parameters (D-optimal)
           uncertainty  disagreement of bootstrap refits on the row's energy
           decision     how contested the best configuration of the row's scenario is
  modes    rows         best K rows anywhere
           cost         best engine by score per second (reload + batch time), then K rows on it
"""
import numpy as np
import pandas as pd

from simulator import fit_unified, unified_predict

K = 5           # rows per round
PILOT_ROWS = 10  # random rows before the first fit (rows mode)
PILOT_ENGINES, PILOT_PER_ENGINE = 3, 4  # cost-mode pilot


def random_rows(sim, n, rng):
    yield list(rng.choice(sim.available(), n, replace=False))


def random_engines(sim, n, rng):
    engines = sorted(set(sim.engine))
    for e in rng.permutation(len(engines)):
        rows = [i for i in sim.available() if sim.engine[i] == engines[e]]
        yield list(rng.choice(rows, min(K, len(rows)), replace=False))


def _pilot(sim, rng, mode):
    if mode == "rows":
        yield list(rng.choice(sim.available(), PILOT_ROWS, replace=False))
        return
    layouts = {}
    for e in sim.engine:
        layouts.setdefault(e[:2], set()).add(e)
    keys = sorted(layouts)
    for j in rng.permutation(len(keys))[:PILOT_ENGINES]:
        options = sorted(layouts[keys[j]])
        e = options[rng.randint(len(options))]
        rows = [i for i in sim.available() if sim.engine[i] == e]
        yield list(rng.choice(rows, min(PILOT_PER_ENGINE, len(rows)), replace=False))


def _rel_jacobian(theta, df):
    f0 = unified_predict(theta, df)
    J = np.empty((len(df), len(theta)))
    for k in range(len(theta)):
        h = 1e-4 * max(abs(theta[k]), 1e-3)
        up, dn = theta.copy(), theta.copy()
        up[k] += h
        dn[k] -= h
        J[:, k] = (unified_predict(up, df) - unified_predict(dn, df)) / (2 * h)
    return J / np.maximum(np.abs(f0), 1e-9)[:, None]


class DOptimal:
    def fit(self, sim):
        obs = sim.observed()
        theta = fit_unified(obs, restarts=5)
        self.G = _rel_jacobian(theta, sim.pool)
        M = self.G[obs.index].T @ self.G[obs.index]
        M += np.diag(1e-6 * np.diag(M) + 1e-12)
        self.Minv = np.linalg.inv(M)

    def scores(self, idx):
        g = self.G[idx]
        return np.log1p(np.einsum("ij,jk,ik->i", g, self.Minv, g))

    def commit(self, i):
        g = self.G[i]
        v = self.Minv @ g
        self.Minv -= np.outer(v, v) / (1 + g @ v)


class Uncertainty:
    B = 8

    def fit(self, sim):
        obs = sim.observed()
        rng = np.random.RandomState(len(obs))
        self.P = np.array([
            unified_predict(fit_unified(obs.iloc[rng.randint(0, len(obs), len(obs))], restarts=3), sim.pool)
            for _ in range(self.B)
        ])
        self.rel_std = self.P.std(0) / np.maximum(np.abs(self.P.mean(0)), 1e-9)

    def scores(self, idx):
        return self.rel_std[idx]

    def commit(self, i):
        pass


class Decision(Uncertainty):
    def fit(self, sim):
        super().fit(sim)
        keys = ["max_tokens", "batch_size", "tp_degree", "pp_degree"]
        pred = pd.concat([sim.pool[keys], pd.DataFrame(self.P.T, index=sim.pool.index)], axis=1)
        best = pred.groupby(keys).mean().groupby(level=[0, 1]).idxmin()
        share, top = {}, {}
        for scen, winners in best.iterrows():
            counts = winners.map(lambda k: k[2:]).value_counts() / self.B
            top[scen] = counts.max()
            share.update({(scen, c): v for c, v in counts.items()})
        rows = zip(zip(sim.pool["max_tokens"], sim.pool["batch_size"]),
                   zip(sim.pool["tp_degree"], sim.pool["pp_degree"]))
        self.score = np.array([(1 - top[sc]) * share.get((sc, c), 0.0) for sc, c in rows])
        self.score += 1e-3 * self.rel_std

    def scores(self, idx):
        return self.score[idx]


def _by_rows(scorer, sim, k):
    cand, picks = sim.available(), []
    for _ in range(min(k, len(cand))):
        j = int(np.argmax(scorer.scores(cand)))
        picks.append(cand.pop(j))
        scorer.commit(picks[-1])
    return picks


def _by_cost(scorer, sim, k):
    cand = np.array(sim.available())
    s = scorer.scores(cand)
    groups = {}
    for j, i in enumerate(cand):
        groups.setdefault(sim.engine[i], []).append(j)
    def value(e):
        top = sorted(groups[e], key=lambda j: s[j])[-k:]
        secs = sim.latency[cand[top]].sum() + (0.0 if e == sim.loaded else sim.reload_s)
        return s[top].sum() / secs
    best = max(groups, key=value)
    sub = [int(i) for i in cand[groups[best]]]
    picks = []
    for _ in range(min(k, len(sub))):
        j = int(np.argmax(scorer.scores(sub)))
        picks.append(sub.pop(j))
        scorer.commit(picks[-1])
    return picks


def adaptive(scorer_cls, mode):
    def strategy(sim, n, rng):
        yield from _pilot(sim, rng, mode)
        scorer = scorer_cls()
        while True:
            scorer.fit(sim)
            yield (_by_rows if mode == "rows" else _by_cost)(scorer, sim, K)
    return strategy


STRATEGIES = {
    "random": random_rows,
    "random_engines": random_engines,
    "doptimal": adaptive(DOptimal, "rows"),
    "doptimal_cost": adaptive(DOptimal, "cost"),
    "uncertainty": adaptive(Uncertainty, "rows"),
    "uncertainty_cost": adaptive(Uncertainty, "cost"),
    "decision": adaptive(Decision, "rows"),
    "decision_cost": adaptive(Decision, "cost"),
}
