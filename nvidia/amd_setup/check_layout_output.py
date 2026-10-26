"""Print generations for one layout so garbage output can be spotted by eye before
trusting its energy: python check_layout_output.py MODEL TP PP"""
import os, sys
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from vllm import LLM, SamplingParams
def main():
    model, tp, pp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    llm = LLM(model, tensor_parallel_size=tp, pipeline_parallel_size=pp, max_model_len=4096, dtype="float16",
              enable_chunked_prefill=True, max_num_batched_tokens=2048, disable_custom_all_reduce=True,
              enforce_eager=os.environ.get("EAGER") == "1")
    outs = llm.generate(["The capital of France is", "Explain in one sentence why the sky is blue.",
                         "def fibonacci(n):"], SamplingParams(temperature=0, max_tokens=40))
    for o in outs:
        print("PROMPT:", o.prompt, "\nOUTPUT:", repr(o.outputs[0].text))


if __name__ == "__main__":
    main()
