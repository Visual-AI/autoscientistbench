import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bench.runtime import bind_runtime, build_environment, load_snapshot, materialize, runtime_env
from bench.sandbox import Sandbox
from bench.task import Task


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.taskdir = self.root / "task"
        (self.taskdir / "runtime" / "r" / "bin").mkdir(parents=True)
        (self.taskdir / "runtime" / "r" / "bin" / "Rscript").write_text("copied executable")
        (self.taskdir / "agent" / "code").mkdir(parents=True)
        (self.taskdir / "agent" / "code" / "env.lock").write_text("")
        self.target = "/old/reproduction/.Renv"
        self.manifest = {"version": 1, "python": "/usr/bin/python3",
                         "mounts": [{"source": "runtime/r", "target": self.target}],
                         "env": {"PATH": self.target + "/bin:/usr/bin:/bin", "R_HOME": self.target + "/lib/R"}}
        self.task = Task(self.taskdir, {"pid": "test-snapshot", "env": {"mode": "snapshot"}})
        self.save()

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        (self.taskdir / "runtime" / "runtime.json").write_text(json.dumps(self.manifest))

    def test_original_prefix_binds_the_copy(self):
        sandbox = Sandbox()
        bind_runtime(sandbox, self.task)
        self.assertIn(("ro", str(self.taskdir / "runtime" / "r"), self.target), sandbox.binds)
        self.assertNotIn(self.target, [source for _, source, _ in sandbox.binds])
        self.assertEqual(runtime_env(self.task)["R_HOME"], self.target + "/lib/R")
        self.assertEqual(runtime_env(self.task, host=True)["R_HOME"], str(self.taskdir / "runtime/r/lib/R"))

    def test_archived_runtime_and_escape_rejected(self):
        path = self.taskdir / "runtime" / "r.tar"
        with tarfile.open(path, "w") as archive:
            entry = tarfile.TarInfo("bin/Rscript")
            entry.size = 3
            entry.mode = 0o755
            archive.addfile(entry, io.BytesIO(b"abc"))
        self.manifest["mounts"] = [{"source": "runtime/r.tar", "target": self.target, "format": "tar"}]
        self.save()
        with patch("bench.runtime.LOCAL_ROOT", self.root / "local"):
            data = materialize(self.task)
            self.assertEqual((data["mounts"][0]["host"] / "bin/Rscript").read_bytes(), b"abc")
        with tarfile.open(path, "w") as archive:
            entry = tarfile.TarInfo("../escape")
            entry.size = 1
            archive.addfile(entry, io.BytesIO(b"x"))
        with patch("bench.runtime.LOCAL_ROOT", self.root / "local"), self.assertRaises(ValueError):
            materialize(self.task)
        self.assertFalse((self.root / "escape").exists())

    def test_snapshot_cannot_mount_data_or_override_boundaries(self):
        self.manifest["mounts"][0]["source"] = "agent/code"
        self.save()
        with self.assertRaises(ValueError):
            load_snapshot(self.task)
        self.manifest["mounts"][0]["source"] = "runtime/r"
        self.manifest["env"]["BENCH_DATA_DIR"] = "/old/data"
        self.save()
        with self.assertRaises(ValueError):
            load_snapshot(self.task)

    def test_changed_lock_is_rejected_without_installing(self):
        candidate = self.root / "candidate.lock"
        candidate.write_text("numpy==1.0\n")
        result = build_environment(self.task, self.root / "venv", candidate, [], self.root, self.root / "build.log")
        self.assertFalse(result["ok"])
        self.assertIn("immutable", (self.root / "build.log").read_text())

    def test_pip_default_still_builds_venv(self):
        task = Task(self.taskdir, {"pid": "pip-task"})
        with patch("bench.envlock.build_venv", return_value={"ok": True}) as build:
            self.assertTrue(build_environment(task, "venv", "lock", [], "cache", "log")["ok"])
            build.assert_called_once()

    def test_tool_bind_does_not_expose_task_judges(self):
        from bench import ROOT
        sandbox = Sandbox()
        sandbox.add_tools({})
        self.assertNotIn(str(ROOT), [source for _, source, _ in sandbox.binds])
        self.assertIn(("ro", str(ROOT / "bench"), "/opt/arb/src/bench"), sandbox.binds)


if __name__ == "__main__":
    unittest.main()
