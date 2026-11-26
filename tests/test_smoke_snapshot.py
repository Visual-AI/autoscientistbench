import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench.patch import git
from bench.smoke import smoke
from bench.sandbox import probe_proc_flags


class SmokeSnapshotTests(unittest.TestCase):
    def test_real_sandbox_pipeline_without_reference(self):
        try:
            probe_proc_flags()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp).chmod(0o755)   # the sandbox runs as a uid of its own (sandbox_user.py) and must reach its binds
            task = Path(tmp) / "task"
            for name in ("agent/code", "agent/data/train", "agent/data/val", "agent/bench", "judge/test", "judge/bench", "runtime"):
                (task / name).mkdir(parents=True)
            code = task / "agent/code"
            (task / "agent/README.md").write_text("Integration fixture: doubles observations.\n")
            (code / "env.lock").write_text("")
            (code / "train.sh").write_text('#!/bin/sh\nmkdir -p "$2"\nprintf ready > "$2/model"\n')
            (code / "predict.sh").write_text('#!/bin/sh\nexec python3 /ws/code/predict.py "$@"\n')
            (code / "predict.py").write_text('''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--input');p.add_argument('--output');a=p.parse_args()
assert Path(a.model,'model').read_text()=='ready'
Path(a.output).write_text(json.dumps([2*x for x in json.loads(Path(a.input).read_text())]))
''')
            validator = '''import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text());x=json.loads(Path(sys.argv[2]).read_text())
assert isinstance(p,list) and len(p)==len(x) and all(isinstance(v,(int,float)) for v in p)
'''
            scorer = '''import json,sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text());y=json.loads(Path(sys.argv[2]).read_text())
print(json.dumps({'mae':sum(abs(a-b) for a,b in zip(p,y))/len(y)}))
'''
            for root in (task / "agent/bench", task / "judge/bench"):
                (root / "validate.py").write_text(validator)
                (root / "scorer.py").write_text(scorer)
            for root in (task / "agent/data/val", task / "judge/test"):
                (root / "input").write_text("[1,2]")
                (root / "labels").write_text("[2,4]")
            (task / "runtime/runtime.json").write_text(json.dumps({"version": 1, "mounts": [], "python": "/usr/bin/python3"}))
            identity = {"GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@local", "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@local"}
            git(["init", "-q"], code)
            git(["add", "."], code)
            git(["commit", "-qm", "fixture"], code, env=identity)
            head = git(["rev-parse", "HEAD"], code).stdout.strip()
            (task / "task.json").write_text(json.dumps({"pid": "snapshot-smoke-test", "metric": {"name": "mae", "higher_is_better": False},
                                                       "adapt": {"adapt_commit": head}, "env": {"mode": "snapshot"}}))
            with patch("bench.smoke.LOCAL_ROOT", Path(tmp) / "local"), patch("bench.runtime.LOCAL_ROOT", Path(tmp) / "local"):
                result = smoke(str(task), cpus=1, cap_s=15)
            self.assertEqual(result["status"], "ok", result)
            self.assertEqual(result["metrics"]["mae"], 0)
            self.assertTrue(result["invalid_prediction_rejected"])
            self.assertFalse((task / "judge/reference/result.json").exists())


if __name__ == "__main__":
    unittest.main()
