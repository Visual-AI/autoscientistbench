"""Tests for GitHub-release mechanics: unbundled code repos, HF asset fetching, and 052_AnisoBivFunc plan loading."""
import csv
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from bench import assets, patch  # noqa: E402
from bench.task import DEFAULTS, Task  # noqa: E402
import generate_052_AnisoBivFunc  # noqa: E402


class MaterializeCodeRepoTests(unittest.TestCase):
    def test_plain_dir_roundtrip_without_nested_git(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            src = td / "agent_code"
            src.mkdir()
            (src / "train.sh").write_text("#!/bin/sh\necho train\n")
            (src / "predict.sh").write_text("#!/bin/sh\necho predict\n")
            (src / "model.py").write_text("ALPHA = 1.0\n")
            (src / "env.lock").write_text("numpy==2.2.6\n")
            (src / ".gitignore").write_text("*.pt\n")

            ws = td / "ws_code"
            base1 = patch.materialize_code_repo(src, ws)
            # Must be deterministic across separate checkouts
            judge_code = td / "judge_code"
            base2 = patch.materialize_code_repo(src, judge_code)
            self.assertEqual(base1, base2)

            # Simulate agent edits inside ws
            (ws / "model.py").write_text("ALPHA = 0.25\n")
            (ws / "helper.py").write_text("VALUE = 42\n")
            (ws / "ignored.pt").write_bytes(b"\x00\x01")
            pb = patch.make_patch(ws, base1)
            ok, probs, files = patch.check_patch(pb, DEFAULTS["patch_limits"])
            self.assertTrue(ok, probs)
            self.assertEqual(sorted(f["path"] for f in files), ["helper.py", "model.py"])

            # Judge applies patch onto fresh materialized checkout
            pp = td / "final.patch"
            pp.write_bytes(pb)
            patch.apply_patch(judge_code, pp)
            self.assertEqual((judge_code / "model.py").read_text(), "ALPHA = 0.25\n")
            self.assertEqual((judge_code / "helper.py").read_text(), "VALUE = 42\n")
            self.assertFalse((judge_code / "ignored.pt").exists())

    def test_task_problems_allows_plain_code_dir(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for rel in ("agent/code", "agent/bench", "agent/data/val", "judge/test"):
                (td / rel).mkdir(parents=True)
            (td / "agent/README.md").write_text("# task\n")
            (td / "agent/code/train.sh").write_text("#!/bin/sh\n")
            (td / "agent/code/predict.sh").write_text("#!/bin/sh\n")
            (td / "agent/code/env.lock").write_text("numpy==2.2.6\n")
            (td / "agent/bench/validate.py").write_text("")
            (td / "agent/bench/scorer.py").write_text("")
            (td / "agent/data/val/input").write_text("x")
            (td / "agent/data/val/labels").write_text("y")
            (td / "judge/test/input").write_text("x")
            (td / "judge/test/labels").write_text("y")
            t = Task(td, {"pid": "demo", "metric": {"name": "mse", "higher_is_better": False}})
            self.assertEqual(t.problems(deep=True), [])


class AssetFetchTests(unittest.TestCase):
    def _make_tar_bytes(self, files: dict[str, bytes]) -> bytes:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tf:
            for name, payload in files.items():
                ti = tarfile.TarInfo(name)
                ti.size = len(payload)
                tf.addfile(ti, io.BytesIO(payload))
        return buf.getvalue()

    def test_multipart_fetch_and_sha_verification(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            staging = td / "staging"
            tasks_dir = td / "tasks"
            (tasks_dir / "demo").mkdir(parents=True)

            data_tar = self._make_tar_bytes({"train/a.txt": b"hello world" * 100, "val/b.txt": b"val"})
            mid = len(data_tar) // 2
            p0, p1 = data_tar[:mid], data_tar[mid:]
            (staging / "data/demo").mkdir(parents=True)
            (staging / "data/demo/agent_data.tar.part00").write_bytes(p0)
            (staging / "data/demo/agent_data.tar.part01").write_bytes(p1)

            test_tar = self._make_tar_bytes({"input.txt": b"test_in", "labels.txt": b"test_lab"})
            (staging / "data/demo/judge_test.tar").write_bytes(test_tar)

            rt_tar = self._make_tar_bytes({"bin/tool": b"#!/bin/sh\necho ok\n"})
            (staging / "runtimes/demo").mkdir(parents=True)
            (staging / "runtimes/demo/r.tar").write_bytes(rt_tar)

            manifest = {
                "schema_version": 1,
                "benchmark": "AutoScientistBench",
                "hf_repo_default": "",
                "tasks": {
                    "demo": {
                        "env_mode": "snapshot",
                        "data": {
                            "mode": "hf_tar_multipart",
                            "target_dir": "agent/data",
                            "parts": [
                                {"path": "data/demo/agent_data.tar.part00", "bytes": len(p0),
                                 "sha256": hashlib.sha256(p0).hexdigest()},
                                {"path": "data/demo/agent_data.tar.part01", "bytes": len(p1),
                                 "sha256": hashlib.sha256(p1).hexdigest()},
                            ],
                        },
                        "test": {
                            "mode": "hf_tar",
                            "target_dir": "judge/test",
                            "parts": [
                                {"path": "data/demo/judge_test.tar", "bytes": len(test_tar),
                                 "sha256": hashlib.sha256(test_tar).hexdigest()},
                            ],
                        },
                        "runtimes": [
                            {"name": "r", "target_path": "runtime/r.tar", "hf_path": "runtimes/demo/r.tar",
                             "bytes": len(rt_tar), "sha256": hashlib.sha256(rt_tar).hexdigest()},
                        ],
                    }
                },
            }
            mp = td / "assets_manifest.json"
            mp.write_text(json.dumps(manifest))

            res = assets.fetch_assets("demo", local_dir=staging, manifest_path=mp, tasks_dir=tasks_dir)
            self.assertEqual(res["fetched"], 3)
            self.assertEqual((tasks_dir / "demo/agent/data/train/a.txt").read_bytes(), b"hello world" * 100)
            self.assertEqual((tasks_dir / "demo/judge/test/labels.txt").read_bytes(), b"test_lab")
            self.assertEqual((tasks_dir / "demo/runtime/r.tar").read_bytes(), rt_tar)

            # Second fetch should skip all 3
            res2 = assets.fetch_assets("demo", local_dir=staging, manifest_path=mp, tasks_dir=tasks_dir)
            self.assertEqual(res2["fetched"], 0)
            self.assertEqual(res2["skipped"], 3)

            # Verify-only should succeed
            res3 = assets.fetch_assets("demo", verify_only=True, manifest_path=mp, tasks_dir=tasks_dir)
            self.assertEqual(res3["verified"], 3)

    def test_traversal_archive_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            bad_tar = td / "bad.tar"
            bad_tar.write_bytes(self._make_tar_bytes({"../escape.txt": b"boom"}))
            with self.assertRaises(ValueError):
                assets._safe_extract_tar_parts([bad_tar], td / "out")
            self.assertFalse((td / "escape.txt").exists())


class AnisoBivFuncPlanTests(unittest.TestCase):
    def test_load_and_verify_plan(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            plan = td / "plan.csv"
            with plan.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["id", "N", "side", "sigma", "alpha", "seed", "cell", "rep", "split", "path", "file", "offset"])
                w.writeheader()
                w.writerow({"id": "a", "N": 2, "side": 5, "sigma": 0.1, "alpha": 0.5, "seed": 1, "cell": 1, "rep": 1,
                            "split": "val", "path": "agent/data/val/input/s1.bin", "file": "s1.bin", "offset": 0})
                w.writerow({"id": "b", "N": 3, "side": 4, "sigma": 0.1, "alpha": 0.5, "seed": 2, "cell": 1, "rep": 2,
                            "split": "test", "path": "judge/test/input/s2.bin", "file": "s2.bin", "offset": 0})
            _, rows, exp = generate_052_AnisoBivFunc.load_plan(plan, splits={"val"})
            self.assertEqual(len(rows), 1)
            self.assertEqual(exp, {"agent/data/val/input/s1.bin": 2 * 25 * 8})
            ok, bad = generate_052_AnisoBivFunc.verify_shards(td, exp)
            self.assertEqual(len(ok), 0)
            self.assertEqual(len(bad), 1)
            (td / "agent/data/val/input").mkdir(parents=True)
            (td / "agent/data/val/input/s1.bin").write_bytes(b"\x00" * (2 * 25 * 8))
            ok, bad = generate_052_AnisoBivFunc.verify_shards(td, exp)
            self.assertEqual(len(ok), 1)
            self.assertEqual(len(bad), 0)


if __name__ == "__main__":
    unittest.main()
