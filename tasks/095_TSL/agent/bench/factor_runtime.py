"""Execute an arbitrary univariate factor on one value in a fresh private process."""
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

from private_eval import Evaluator


class FactorRuntime:
    def __init__(self, model_dir, card):
        self.model_dir = Path(model_dir)
        self.card = card
        self.evaluator = None
        self.cache = {}
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.evaluator is not None:
            self.evaluator.__exit__(*args)

    def evaluate(self, factor, values):
        name = factor.get('name')
        if not isinstance(name, str) or not name or len(name) > 1024:
            raise ValueError('executable factor needs a nonempty function name')
        if self.evaluator is None:
            self.evaluator = Evaluator(self.model_dir / 'factors.zip', 'icml-6321-TSL',
                                       cpus=1, timeout_s=None)
            frozen = json.loads((self.evaluator.package / 'model/model_card.json').read_text())
            if frozen != self.card:
                raise ValueError('factor artifact differs from the trained model card')
            if 'BENCH_SEED' in os.environ and self.evaluator.manifest['algorithm_seed'] != int(os.environ['BENCH_SEED']):
                raise ValueError('factor artifact seed differs from BENCH_SEED')
        result = []
        for value in values:
            value = float(value)
            key = (factor['feature'], name, value.hex())
            if key not in self.cache:
                request = self.evaluator.root / 'factor-input.json'
                request.write_text(json.dumps({'feature': factor['feature'], 'factor': name, 'x': value}))
                outcome = self.evaluator.run(request)
                try:
                    output = json.loads(Path(outcome['output_path']).read_text())
                    if not isinstance(output, dict) or set(output) != {'value'}:
                        raise ValueError('factor output requires exactly value')
                    prediction = output['value']
                    if type(prediction) not in (int, float) or not math.isfinite(prediction):
                        raise ValueError('factor output must be a finite scalar')
                    self.cache[key] = float(prediction)
                    self.calls += 1
                finally:
                    shutil.rmtree(Path(outcome['output_path']).parent)
            result.append(self.cache[key])
        return np.asarray(result, dtype=np.float64)
