"""Two-GPU RCCL all-reduce, to tell a communication fault apart from a vLLM one."""
import os, torch, torch.distributed as dist, torch.multiprocessing as mp
def w(r):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29511")
    torch.cuda.set_device(r); dist.init_process_group("nccl", rank=r, world_size=2)
    x = torch.ones(1024, device=f"cuda:{r}") * (r + 1); dist.all_reduce(x); torch.cuda.synchronize()
    print("rank", r, "allreduce", x[0].item(), flush=True); dist.destroy_process_group()
if __name__ == "__main__": mp.spawn(w, nprocs=2)
