---
pretty_name: "EnergyLens: LLM Inference Energy across Parallelism Layouts"
license: cc-by-4.0
task_categories:
- tabular-regression
tags:
- energy
- llm-inference
- vllm
- parallelism
configs:
- config_name: grid
  data_files:
  - split: all
    path: "grid.parquet"
  default: true
- config_name: online_day
  data_files:
  - split: all
    path: "online_day.parquet"
---

# EnergyLens dataset

Measured GPU energy of LLM and multimodal inference served with vLLM, across
tensor-parallel (TP) and pipeline-parallel (PP) layouts, batch sizes, output lengths,
token budgets, modalities and precisions. It is the data behind the EnergyLens paper.

## Subsets

| Config | What it holds |
|---|---|
| `grid` | Offline batches on a full grid: every layout (TP x PP) x token budget x batch size x output length, one row per measured batch. NVIDIA A100-40GB (1-8 GPUs) and Intel GPU Max 1550 (1-12 tiles). |
| `online_day` | One day of Azure LLM inference traffic replayed live against every layout and serving setting, 90 s windows at six hours of the day; one row per (window, configuration). |

## How energy was measured

- **NVIDIA:** [Zeus](https://ml.energy/zeus) reads the NVML energy counter of every GPU the engine uses, over a window around each batch's generation call.
- **Intel:** Zeus has no Intel backend, so the profiler reads the Level Zero Sysman energy counters over the same window. They count whole cards (two tiles each).
- **Power** is sampled independently over the same window. It is never computed as energy divided by latency.

## Main columns

| Column | Meaning |
|---|---|
| `hardware`, `energy_counter` | GPU and the counter that measured it |
| `model`, `modality`, `precision`, `variant` | Model checkpoint, input modality, weight format (`fp16`, `bnb4bit`, `awq4bit`, `gptq8bit`, `w8a8`), video resolution where relevant |
| `tensor_parallel_size`, `pipeline_parallel_size`, `n_devices` | Layout |
| `batch_size`, `max_tokens`, `input_length`, `generated_tokens` | Workload of the batch |
| `max_num_batched_tokens`, `max_num_seqs` | vLLM token budget and concurrency limit |
| `latency` | Wall-clock seconds of the batch |
| `energy` | `grid`: Joules **per request** (the batch's energy divided by `batch_size`); `online_day`: Joules over the whole window |
| `batch_energy`, `energy_per_output_token` | Joules for the whole batch, and per generated token |
| `power` | Mean power in W over all measured devices |
| `valid` | False when the row fails a validity check (missing measurement, wrong device count, power outside the device range, energy inconsistent with power x latency) |

## Usage

```python
import pandas as pd
grid = pd.read_parquet("hf://datasets/<org>/energylens/grid.parquet")
grid = grid[grid.valid]
a100 = grid[grid.hardware.str.contains("A100")]
```

## Known limitations

- On the A100 node, some TP=2 layouts lose batches to a vLLM crash. Those layouts have fewer rows.
- Intel counters cover whole cards. A layout using an odd number of tiles is charged for the idle tile of its last card.
- The `online_day` traffic is the public Azure LLM inference trace (2024), scaled to 0.06x the production rate.

## License

CC BY 4.0.
