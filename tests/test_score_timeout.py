import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench.evalrun import one_seed


class ScoreTimeoutTests(unittest.TestCase):
    def test_explicit_operator_probe_cap_without_implicit_stage_default(self):
        for configured, expected in [(7200, 7200), (None, None)]:
            with self.subTest(configured=configured), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "pred.out").write_text("{}")
                caps = {"train": 10, "predict": 10}
                if configured is not None:
                    caps["score"] = configured
                result = {"elapsed_s": 0, "timed_out": False, "rc": 0}
                with patch("bench.evalrun.train", return_value=result), patch("bench.evalrun.predict", return_value=result), \
                     patch("bench.evalrun.validate", return_value=(True, "")), \
                     patch("bench.evalrun.score", return_value={"cost": 0.03}) as scorer:
                    outcome = one_seed(seed=1, code_dir=root, data_dir=root, out_root=root,
                                       input_path=root / "input", labels_path=root / "labels",
                                       validator_py=root / "validate.py", scorer_py=root / "scorer.py",
                                       exec_env={}, tool_env={}, tool_python="python3", caps=caps)
                    self.assertEqual(outcome["status"], "ok")
                    self.assertEqual(scorer.call_args.kwargs["timeout"], expected)
