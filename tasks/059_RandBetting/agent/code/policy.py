"""Editable stateful policy example; the default certificate uses the fast path.

To use this entry point, train writes execution=stateful and the fixed global
B/theta/trunc_scale/combine values to model/betting.json. Replace this code freely.
"""
import argparse
import json
import math
from pathlib import Path

from private_rpc import call


def main():
    parser = argparse.ArgumentParser()
    for name in ('model', 'input', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.input).read_text())
    method = json.loads((Path(args.model) / 'betting.json').read_text())
    theta = method['theta']
    total = 0.0
    centered_squares = 0.0
    predmix_squares = 0.0
    for t in range(1, request['n'] + 1):
        mean = (0.5 + total) / t
        variance = (0.25 if t == 1 else centered_squares) / (t + 1)
        gap = mean - 0.5
        positive = gap / (0.5 * abs(gap) + variance + gap * gap)
        previous_variance = 0.25 if t == 1 else (0.25 + predmix_squares) / t
        negative = (math.sqrt(2 * math.log(1 / (0.05 * (1 - theta))) /
                              (t * math.log(1 + t) * previous_variance))
                    if theta < 1 else 0.0)
        response = call('bet', {'positive': positive, 'negative': negative})
        x = response['x']
        total += x
        centered_squares += (x - 0.5) ** 2
        predmix_squares += (x - (0.5 + total) / (t + 1)) ** 2
    Path(args.output).write_text(json.dumps({'complete': True}))


if __name__ == '__main__':
    main()
