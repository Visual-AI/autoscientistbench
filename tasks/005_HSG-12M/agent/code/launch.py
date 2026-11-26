"""Launch precisely two local DDP ranks within the supplied CPU budget."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

out = Path(sys.argv[1]).resolve(); out.mkdir(parents=True, exist_ok=True)
threads = min(20, max(1, int(os.environ.get('BENCH_CPUS', '128'))//2))
with socket.socket() as s:
    s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
processes = []; handles = []
try:
    for rank in (0, 1):
        env = os.environ.copy()
        env.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), WORLD_SIZE='2', LOCAL_WORLD_SIZE='2',
                   RANK=str(rank), LOCAL_RANK=str(rank), GROUP_RANK='0', TORCHELASTIC_RUN_ID='hsg-fit')
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
            env[name] = str(threads)
        log = (out/f'rank{rank}.log').open('w'); handles.append(log)
        processes.append(subprocess.Popen([sys.executable, str(Path(__file__).with_name('bench_train.py')),
                                          '--out', str(out)], env=env, stdout=log, stderr=subprocess.STDOUT))
    # A failed peer must terminate the other rank promptly.
    while any(p.poll() is None for p in processes):
        failed = [p.returncode for p in processes if p.poll() not in (None, 0)]
        if failed:
            for p in processes:
                if p.poll() is None: p.terminate()
            break
        time.sleep(1)
    codes = [p.wait() for p in processes]
    if any(codes):
        for rank in (0, 1): print((out/f'rank{rank}.log').read_text()[-12000:], file=sys.stderr)
        raise RuntimeError(f'DDP worker failure: {codes}')
finally:
    for p in processes:
        if p.poll() is None: p.terminate()
    for p in processes: p.wait()
    for f in handles: f.close()
