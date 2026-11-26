"""Replay candidate preparation before solving; never trust submitted timings.

The outer train result is deliberately not used as the evaluated model. The
submitted train program recreates its model in an isolated, timed worker.
"""
import json
import os
import shutil
import stat
from pathlib import Path

from private_eval import Evaluator


class PreparedEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A crafted archive must not expose an unmeasured outer model to train.
        shutil.rmtree(self.package / 'model')
        (self.package / 'model').mkdir()
        code = self.package / 'code'
        submitted = code / '.tensor-submitted-policy.sh'
        if submitted.exists():
            raise ValueError('reserved replay entry already exists')
        (code / 'policy.sh').rename(submitted)
        (code / 'policy.sh').write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\n'
            'exec python -I /ws/bench/tensor_replay_entry.py "$@"\n'
        )
        self.prepared_model = None

    def prepare(self, affinity, timeout_s=None):
        request = self.root / 'prepare-request.json'
        request.write_text('{"stage":"prepare"}\n')
        result = self.run(request, env={'TENSOR_REPLAY_STAGE': 'prepare'},
                          affinity=affinity, timeout_s=timeout_s)
        if json.loads(Path(result['output_path']).read_text()) != {'prepared': True}:
            raise ValueError('preparation did not complete')
        model = Path(result['output_path']).parent / 'output' / 'replayed-model'
        info = model.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError('prepared model must be a directory')
        # Worker descendants are gone before run returns. Nested symlinks resolve
        # within the next worker namespace; the scorer never executes this tree.
        self.prepared_model = model
        self.runtime_mounts.append((str(model), '/ws/model'))
        return result

    def configuration_text(self):
        """Copy only bounded, regular JSON metadata for legacy serialization."""
        if self.prepared_model is None:
            raise ValueError('preparation is required')
        fd = os.open(self.prepared_model / 'config.json', os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
                raise ValueError('invalid prepared configuration')
            value = source.read(65537)
        config = json.loads(value)
        if not isinstance(config, dict):
            raise ValueError('prepared configuration must be a JSON object')
        return json.dumps(config)
