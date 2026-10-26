#!/bin/bash
# Rerun after the energy counters stalled (fixed by a reboot): Llama-3.1-8B and Qwen2.5-14B.
export PYTHONPATH=/work/build/vllm ENERGYLENS_BACKEND=amd
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p logs_amd
python test_amd_energy.py > logs_amd/energy_selftest.log 2>&1 || { echo "energy self-test failed" > logs_amd/done; exit 1; }
python main_profiler.py --model unsloth/Llama-3.1-8B-Instruct --tp 1 2 --pp 1 --output-root ./results_amd > logs_amd/llama8b.log 2>&1
echo "llama exit $?" >> logs_amd/status
python test_amd_energy.py >> logs_amd/energy_selftest.log 2>&1 || { echo "energy self-test failed after llama" > logs_amd/done; exit 1; }
python main_profiler.py --model Qwen/Qwen2.5-14B-Instruct --tp 2 --pp 1 --output-root ./results_amd > logs_amd/qwen14b.log 2>&1
echo "qwen exit $?" >> logs_amd/status
python test_amd_energy.py >> logs_amd/energy_selftest.log 2>&1
echo "ALL_DONE selftest_exit=$?" > logs_amd/done
