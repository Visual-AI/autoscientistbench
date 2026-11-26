import collections
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract import read_predictions

def score(pred,labels):
    truth=labels['observations']
    if set(pred)!=set(truth): raise ValueError('prediction IDs do not match labels')
    cells=collections.defaultdict(list)
    for key,label in truth.items():
        p=pred[key]
        cells[label['cell']].append((p['hi']-p['lo'],round(label['truth'],12) in p['values']))
    if len(cells)!=36 or any(len(v)!=labels['replications_per_cell'] for v in cells.values()):
        raise ValueError('incomplete Table 2 grid')
    detail={key:{'width':statistics.mean(x[0] for x in values),'coverage':statistics.mean(x[1] for x in values)} for key,values in sorted(cells.items())}
    width=statistics.mean(v['width'] for v in detail.values()); coverage=statistics.mean(v['coverage'] for v in detail.values())
    minimum=min(v['coverage'] for v in detail.values()); valid=minimum>=.91 and coverage>=.94
    return {'mean_width':width if valid else 1.,'unconstrained_mean_width':width,'mean_coverage':coverage,'minimum_cell_coverage':minimum,'coverage_pass':valid,'tie_break_higher_mean_coverage':coverage,'cells':detail}

if __name__=='__main__':
    print(json.dumps(score(read_predictions(sys.argv[1]),json.load(open(sys.argv[2])))))
