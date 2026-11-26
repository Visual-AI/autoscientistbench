"""One Figure-4 solver run; original arithmetic, counted batched traversals."""
import argparse,json,os,random
from pathlib import Path
import numpy as np
import graph_ir as LiteEFG
from algorithm.asymp_dgda import _AsymPDGDA
import leduc_env
p=argparse.ArgumentParser()
for name in ('model','input','output'):p.add_argument('--'+name,required=True)
a=p.parse_args();config=json.loads((Path(a.model)/'config.json').read_text());row=json.loads(Path(a.input).read_text())
seed=int(os.environ.get('BENCH_SEED','1'));random.seed(seed);np.random.seed(seed);LiteEFG.set_seed(seed)
mus=[0.,0.];mus[row['run_id']]=config['mu'];graph=_AsymPDGDA(eta=config['eta'],mus=mus)
env=leduc_env.open_run(a.input,row['id'],graph,row['run_id'])
steps=min(int(config['steps_per_run']),int(row['steps_per_run']))
if steps<0:raise ValueError('steps must be nonnegative')
for start in range(0,steps,1000):env.update_many([(graph.u,1,[0]),(graph.u,2,[1])],repeats=min(1000,steps-start))
state=env.export(graph.u);Path(a.output).write_text(json.dumps(state,separators=(',',':'))+'\n')
