#!/bin/bash
# Profile the models shared with the A100 and Max 1550 datasets on 2x AMD Instinct MI100.
# Run inside the container (see README_amd.md):
#   docker exec -d vllm bash /work/<repo>/nvidia/amd_setup/run_amd.sh
# Layouts: TP=1 and TP=2. PP=2 is not possible with vLLM 0.6.3's synchronous LLM engine
# ("Pipeline parallelism is only supported through AsyncLLMEngine").
# Qwen2.5-14B in fp16 (~29.5 GB of weights) does not fit on one 32 GB GPU: TP=2 only.
# Llama-3.1-8B-Instruct comes from the ungated unsloth mirror (same weights as meta-llama).
export PYTHONPATH=/work/build/vllm ENERGYLENS_BACKEND=amd
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p logs_amd
python test_amd_energy.py > logs_amd/energy_selftest.log 2>&1 || { echo "energy self-test failed"; exit 1; }
python main_profiler.py --model mistralai/Mistral-7B-Instruct-v0.3 --tp 1 2 --pp 1 --output-root ./results_amd > logs_amd/mistral7b.log 2>&1
python main_profiler.py --model unsloth/Llama-3.1-8B-Instruct --tp 1 2 --pp 1 --output-root ./results_amd > logs_amd/llama8b.log 2>&1
python main_profiler.py --model Qwen/Qwen2.5-14B-Instruct --tp 2 --pp 1 --output-root ./results_amd > logs_amd/qwen14b.log 2>&1
echo ALL_DONE > logs_amd/done
