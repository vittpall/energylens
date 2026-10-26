"""Check RCCL fp16 all-reduce numerics on large tensors (sum of both ranks vs local reference)."""
import os, torch, torch.distributed as dist, torch.multiprocessing as mp
def w(r):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29512")
    torch.cuda.set_device(r); dist.init_process_group("nccl", rank=r, world_size=2)
    for n in (4096, 4096 * 4096):
        g = torch.Generator().manual_seed(0)
        a, b = torch.randn(n, generator=g), torch.randn(n, generator=g)
        x = (a if r == 0 else b).half().cuda(); dist.all_reduce(x); torch.cuda.synchronize()
        ref = (a.half().float() + b.half().float())
        print("rank", r, "n", n, "max abs err", (x.float().cpu() - ref).abs().max().item(), flush=True)
    dist.destroy_process_group()
if __name__ == "__main__": mp.spawn(w, nprocs=2)
