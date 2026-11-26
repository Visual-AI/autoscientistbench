import argparse,os,sys
sys.path.insert(0,os.environ['BENCH_BENCH_DIR'])
from private_eval import export_artifact
p=argparse.ArgumentParser()
for key in ('input','model','output'):p.add_argument('--'+key,required=True)
a=p.parse_args()
# Runtime is the immutable task-owned R snapshot, not Python packages.
export_artifact(os.environ['BENCH_CODE_DIR'],a.model,a.output,'Sta-33',include_dependencies=False)
