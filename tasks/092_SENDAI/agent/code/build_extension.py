"""Build the reproduction's exact CPU RNG helper in the model output directory."""
import os,sys
from pathlib import Path
from torch.utils.cpp_extension import load
build=Path(sys.argv[1]).resolve();build.mkdir(parents=True,exist_ok=True)
os.environ['MAX_JOBS']=str(min(4,int(os.environ.get('BENCH_CPUS','4'))))
load(name='sendai_rng',sources=[str(Path(__file__).resolve().parent/'exact_rng.cpp')],extra_cflags=['-O3','-march=native','-mprefer-vector-width=512','-fopenmp'],extra_ldflags=['-fopenmp'],build_directory=str(build),is_python_module=False,verbose=True)
