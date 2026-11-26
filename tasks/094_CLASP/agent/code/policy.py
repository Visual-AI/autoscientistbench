"""One independent episode; the released CLASP-I optimization is unchanged."""
import argparse,json,os
from pathlib import Path
import numpy as np
from main import load_env
from clasp import CLASP_I,T,F,G,D
p=argparse.ArgumentParser()
for name in ('model','input','output'):p.add_argument('--'+name,required=True)
a=p.parse_args();json.loads((Path(a.model)/'model.json').read_text());row=json.loads(Path(a.input).read_text())
np.random.seed(int(os.environ.get('BENCH_SEED','1')))
env=load_env();ep=env.make_episode(row['id']);algorithm=CLASP_I(ep.initial(),T,(F,G,D));final_x,_=algorithm.run(ep)
env.write_output(a.output,[(row['id'],ep.finish(final_x))])
