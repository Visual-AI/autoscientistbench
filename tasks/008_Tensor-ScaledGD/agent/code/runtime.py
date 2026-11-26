"""Restore the immutable offline Octave runtime into the model directory."""
from pathlib import Path
import os,tarfile

def prepare(model):
    model=Path(model); runtime=model/'runtime'
    if not (runtime/'octave/bin/octave-cli').exists():
        runtime.mkdir(parents=True,exist_ok=True)
        with tarfile.open(Path(os.environ['BENCH_DATA_DIR'])/'train/runtime.tar.gz') as tar:tar.extractall(runtime)
    return runtime

def environment(runtime,threads=1,parallel=False):
    runtime=Path(runtime).resolve();env=dict(os.environ)
    env.update(OCTAVE_HOME=str(runtime/'octave'),LD_LIBRARY_PATH=str(runtime/'lib'),
               OMP_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
               REPRO_WORKER_THREADS=str(threads),MKL_DYNAMIC='FALSE',OMP_DYNAMIC='FALSE',KMP_BLOCKTIME='0',OMP_PROC_BIND='false')
    if parallel:env['LD_PRELOAD']=':'.join(str(runtime/'lib'/x) for x in ['libmkl_intel_thread.so.2','libiomp5.so'])
    return env
