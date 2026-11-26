"""Baseline policy: the authors' RPB-OM with POPU, one instance per cache set, driven as cache/cache.py drives it."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import random

import paging_env
from cache.evict.algorithms import PredictAlgorithmFactory, PredictiveRPBOnlineMinAlgorithm


def main():
    parser = argparse.ArgumentParser()
    for name in ('model', 'input', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.model, 'config.json').read_text())
    random.seed(int(os.environ['BENCH_SEED']))   # the authors' algorithms draw from the global random module
    factory = PredictAlgorithmFactory.generate_predictive_algorithm(
        partial(PredictiveRPBOnlineMinAlgorithm, pred_budget=config['tau'],
                max_support_factor=config['max_support_factor']), config['predictor'])
    algorithms = [factory(paging_env.WAYS) for _ in range(paging_env.NUM_SETS)]
    victim = None
    while True:
        batch = paging_env.step(victim)
        victim = None
        for pc, address in zip(batch.pcs, batch.addresses):
            line = paging_env.line_of(address)
            algorithm = algorithms[paging_env.set_of(line)]
            before = list(algorithm.cache)
            algorithm.access(pc, line)
        if batch.needs_victim:
            # The authors' access() has replaced the victim; only the last access of a batch can evict.
            victim, = set(before) - set(algorithm.cache)
        if batch.done:
            break
    paging_env.write_output(args.output, paging_env.finish())


if __name__ == '__main__':
    main()
