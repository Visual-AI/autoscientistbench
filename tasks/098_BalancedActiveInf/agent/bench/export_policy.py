import argparse,os,sys
sys.path.insert(0,os.environ['BENCH_BENCH_DIR'])
from private_eval import export_artifact
p=argparse.ArgumentParser()
for key in ('model','input','output'):p.add_argument('--'+key,required=True)
a=p.parse_args()
# Exact immutable Python/R snapshot is already supplied to isolated workers.
export_artifact(os.environ['BENCH_CODE_DIR'],a.model,a.output,'imp-104-BalancedActiveInf',include_dependencies=False)
