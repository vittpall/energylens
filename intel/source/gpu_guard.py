"""Detect other jobs running on this job's GPUs."""

import os
import time

try:
    import pynvml
except ImportError:          # no NVIDIA stack (e.g. AMD node): guard disabled
    pynvml = None


class GPUContentionError(RuntimeError):
    """Another job's process is running on one of this job's GPUs."""


_JOB = os.environ.get("PBS_JOBID")
_nvml_ok = None
_warned = False


def _nvml_ready():
    global _nvml_ok
    if _nvml_ok is None:
        try:
            pynvml.nvmlInit()
            _nvml_ok = True
        except Exception:
            _nvml_ok = False
    return _nvml_ok


def _proc_jobid(pid):
    """PBS_JOBID from a process's environment; None if unreadable or gone."""
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            env = f.read().split(b"\0")
    except OSError:          # other user's process, or not visible from here
        return None
    for kv in env:
        if kv.startswith(b"PBS_JOBID="):
            return kv[len(b"PBS_JOBID="):].decode(errors="replace")
    return ""


def _descendants(root):
    """root plus every process below it, from /proc parent links."""
    children = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d))
    out, stack = {root}, [root]
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def _describe(pid):
    """user, parent pid and command line of a process, for the stop message."""
    try:
        with open(f"/proc/{pid}/status") as f:
            st = dict(l.split(":", 1) for l in f if ":" in l)
        uid = st.get("Uid", "?").split()[0]
        ppid = st.get("PPid", "?").strip()
        try:
            import pwd
            user = pwd.getpwuid(int(uid)).pw_name
        except (KeyError, ValueError, ImportError):
            user = f"uid {uid}"
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()[:120]
        return f"user {user}, parent {ppid}, cmd: {cmd or '?'}"
    except OSError:
        return "not visible from this job (other user or other node namespace)"


def foreign_gpu_processes():
    """
    [(pid, gpu_index, gpu_uuid, used_MiB), ...] for processes on this job's
    GPUs that are not ours, or None if NVML is unavailable.
    """
    if pynvml is None or not _nvml_ready():
        return None
    mine = _descendants(os.getpid())
    found = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        h = pynvml.nvmlDeviceGetHandleByIndex(i)
        uuid = pynvml.nvmlDeviceGetUUID(h)
        uuid = uuid.decode() if isinstance(uuid, bytes) else uuid
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
            if p.pid in mine:
                continue
            if _JOB and _proc_jobid(p.pid) == _JOB:
                continue
            used = (p.usedGpuMemory or 0) / 2**20
            found.append((p.pid, i, uuid, round(used)))
    return found


def assert_gpus_exclusive(where, retry_after=5.0):
    """
    Raise GPUContentionError if another job's process is on our GPUs.

    Looks twice, `retry_after` seconds apart, so a process that is just
    exiting (NVML can list it for a moment) does not stop the run.
    """
    global _warned
    found = foreign_gpu_processes()
    if found is None:
        if not _warned:
            print("[gpu_guard] NVML unavailable — cannot check for other jobs "
                  "on these GPUs; continuing unguarded.", flush=True)
            _warned = True
        return
    if found:
        time.sleep(retry_after)
        found = foreign_gpu_processes() or []
    if found:
        lines = "\n".join(f"    pid {pid} on GPU {i} ({uuid}), {mib} MiB — {_describe(pid)}"
                          for pid, i, uuid, mib in found)
        raise GPUContentionError(
            f"another job is running on this job's GPUs ({where}):\n{lines}\n"
            f"  Energy measured now would include its work. Stopping without "
            f"marking the config done; resubmit to resume once the GPUs are free.")
