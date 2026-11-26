"""ATC on a judge-owned stream. No future samples are present in this worker."""
import argparse
import json
from pathlib import Path
from private_rpc import call
from icml.atc_scaling_scenarios_dense_adver import atc_run, candidate_splits_all, gamma_paper

class RemoteEpisode:
    def __init__(self, horizon): self.horizon = horizon
    def observe(self): return call('observe')
    def submit(self, estimate): return call('submit', float(estimate))
    def finish(self): return call('finish')

p=argparse.ArgumentParser()
for name in ('model','input','output'): p.add_argument('--'+name, required=True)
a=p.parse_args()
json.loads((Path(a.model)/'model.json').read_text())
request=json.loads(Path(a.input).read_text())
atc_run(RemoteEpisode(int(request['horizon'])), sigma=1., alpha=.05,
        split_fn=candidate_splits_all, gamma_fn=gamma_paper)
Path(a.output).write_text('{"complete":true}\n')
