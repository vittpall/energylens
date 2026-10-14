"""The Zeus-measured datasets the paper figures draw from."""
import os
import sys
from dataclasses import dataclass

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NVIDIA = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(NVIDIA, "test"))
from compare_cross_model import prepare_df  # noqa: E402

Z = "results_zeus/run_{}/res_native/profiling_metrics.csv"
ZM = "results_zeus_maxgen/run_{}/res_native/profiling_metrics.csv"


@dataclass(frozen=True)
class Dataset:
    model: str          # short model name, as in the paper
    family: str         # key of style.FAMILY
    modality: str       # text | image | video | video-only
    path: str           # relative to nvidia/
    color: str          # fixed per model, the same in every figure
    precision: str = "fp16"
    variant: str = ""   # e.g. the video resolution, when a model/modality has several runs

    @property
    def label(self):
        extra = [self.modality] if self.modality != "text" else []
        extra += [self.variant] if self.variant else []
        extra += [self.precision] if self.precision != "fp16" else []
        return f"{self.model} ({', '.join(extra)})" if extra else self.model

    def exists(self):
        """CSV present with at least one measured row (a run that stopped before its first
        batch leaves a header-only file)."""
        p = os.path.join(NVIDIA, self.path)
        if not os.path.exists(p):
            return False
        with open(p) as f:
            return sum(1 for _ in zip(range(2), f)) > 1

    def load(self):
        p = os.path.join(NVIDIA, self.path)
        return prepare_df(pd.read_csv(p), p)


# tab10 + extras: one colour per model, kept across figures
C = {"Mistral-7B": "#2ca02c", "Llama-3.1-8B": "#9467bd", "Qwen2.5-14B": "#8c564b",
     "Mistral-Small-24B": "#17becf", "Qwen2.5-32B": "#7f7f7f", "Qwen2-VL-7B": "#e377c2",
     "LLaVA-1.5-7B": "#bcbd22", "Qwen1.5-MoE": "#1f77b4", "Mixtral-8x7B": "#393b79",
     "Mamba-130M": "#ff7f0e", "Falcon-Mamba-7B": "#d6616b", "Nemotron-Nano-9B": "#d62728",
     "InternVL2-76B": "#637939"}

DATASETS = [
    # text-only, one per model: the TP/PP characterization set
    Dataset("Mistral-7B", "dense", "text", Z.format("text_only_Mistral-7B-Instruct-v0.3_fp16"), C["Mistral-7B"]),
    Dataset("Llama-3.1-8B", "dense", "text", Z.format("text_only_Llama-3.1-8B-Instruct_fp16"), C["Llama-3.1-8B"]),
    Dataset("Qwen2.5-14B", "dense", "text", Z.format("text_only_Qwen2.5-14B-Instruct_fp16"), C["Qwen2.5-14B"]),
    Dataset("Mistral-Small-24B", "dense", "text", Z.format("text_only_Mistral-Small-24B-Instruct-2501_fp16"),
            C["Mistral-Small-24B"]),
    Dataset("Qwen2.5-32B", "dense", "text", Z.format("text_only_Qwen2.5-32B-Instruct_fp16"), C["Qwen2.5-32B"]),
    Dataset("Qwen1.5-MoE", "moe", "text", Z.format("text_only_Qwen1.5-MoE-A2.7B_fp16"), C["Qwen1.5-MoE"]),
    Dataset("Mixtral-8x7B", "moe", "text", Z.format("text_only_Mixtral-8x7B-Instruct-v0.1_fp16"), C["Mixtral-8x7B"]),
    Dataset("Mamba-130M", "ssm", "text", Z.format("text_only_mamba-130m-hf_fp16"), C["Mamba-130M"]),
    Dataset("Falcon-Mamba-7B", "ssm", "text", Z.format("text_only_falcon-mamba-7b_fp16"), C["Falcon-Mamba-7B"]),
    Dataset("Nemotron-Nano-9B", "hybrid", "text", Z.format("text_only_NVIDIA-Nemotron-Nano-9B-v2_fp16"),
            C["Nemotron-Nano-9B"]),
    # multimodal models (dense transformers with a vision encoder)
    Dataset("Qwen2-VL-7B", "dense", "text", ZM.format("text_only_Qwen2-VL-7B-Instruct_fp16"), C["Qwen2-VL-7B"]),
    Dataset("Qwen2-VL-7B", "dense", "image", ZM.format("text_image_Qwen2-VL-7B-Instruct_fp16"), C["Qwen2-VL-7B"]),
    *[Dataset("Qwen2-VL-7B", "dense", "video-only",
              f"results_zeus/run_video_only_Qwen2-VL-7B-Instruct_fp16/res{r}p/profiling_metrics.csv",
              C["Qwen2-VL-7B"], variant=f"{r}p") for r in (240, 360, 480, 720)],
    *[Dataset("Qwen2-VL-7B", "dense", "video",
              f"results_zeus/run_text_video_Qwen2-VL-7B-Instruct_fp16/res{r}p/profiling_metrics.csv",
              C["Qwen2-VL-7B"], variant=f"{r}p") for r in (240, 360, 480, 720)],
    Dataset("LLaVA-1.5-7B", "dense", "text", Z.format("text_only_llava-1.5-7b-hf_fp16"), C["LLaVA-1.5-7B"]),
    Dataset("LLaVA-1.5-7B", "dense", "image", ZM.format("text_image_llava-1.5-7b-hf_fp16"), C["LLaVA-1.5-7B"]),
    *[Dataset("LLaVA-1.5-7B", "dense", "video",
              f"results_zeus/run_text_video_llava-1.5-7b-hf_fp16/res{r}p/profiling_metrics.csv",
              C["LLaVA-1.5-7B"], variant=f"{r}p") for r in (240, 360, 480, 720)],
    Dataset("InternVL2-76B", "dense", "image", Z.format("text_image_InternVL2-Llama3-76B_fp16"), C["InternVL2-76B"]),
    # 4-bit (bitsandbytes) variants
    Dataset("Mistral-7B", "dense", "text", Z.format("text_only_Mistral-7B-Instruct-v0.3_bnb4bit"), C["Mistral-7B"],
            "bnb4bit"),
    Dataset("Qwen2.5-14B", "dense", "text", Z.format("text_only_Qwen2.5-14B-Instruct_bnb4bit"), C["Qwen2.5-14B"],
            "bnb4bit"),
]


def select(modality=None, precision="fp16", models=None, family=None):
    """Datasets matching the filters whose CSV exists; missing ones are reported."""
    out = []
    for d in DATASETS:
        if (modality and d.modality != modality) or (precision and d.precision != precision) \
                or (models and d.model not in models) or (family and d.family != family):
            continue
        if d.exists():
            out.append(d)
        else:
            print(f"[skip] {d.label}: {d.path} not found (run not finished?)")
    return out


def get(model, modality="text", precision="fp16"):
    found = select(modality, precision, [model])
    return found[0] if found else None
