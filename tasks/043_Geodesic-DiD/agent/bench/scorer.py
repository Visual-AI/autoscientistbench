import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from io_tools import read_predictions


def remaining_time():
    deadline = os.environ.get('BENCH_EVAL_DEADLINE_TS')
    if deadline is None:
        return None
    remaining = float(deadline) - time.time()
    if remaining <= 0:
        raise TimeoutError('overall evaluation deadline expired')
    return remaining


if __name__ == '__main__':
    labels = Path(sys.argv[2])
    spec = json.loads((labels / 'labels.json').read_text())
    read_predictions(sys.argv[1], spec['ids'])
    bench = Path(__file__).resolve().parent
    proc = subprocess.run(['Rscript', '--vanilla', str(bench/'score.R'), sys.argv[1],
                           str(labels/'truth.csv'), str(bench)],
                          text=True, capture_output=True, check=True, timeout=remaining_time())
    value = float(proc.stdout.strip().splitlines()[-1])
    if not math.isfinite(value) or value < 0:
        raise ValueError('invalid geodesic distance')
    print(json.dumps({'mean_geodesic_error': value, 'replications': len(spec['ids'])}))
