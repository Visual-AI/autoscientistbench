"""Environment preparation uses one budget, without building a real environment."""
import hashlib
import io
import json
import os
import shlex
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from bench.envlock import build_venv
from bench.runtime import build_environment, materialize
from bench.task import Task
from bench.util import capture


class EnvironmentBudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.lock = self.root / "env.lock"
        self.lock.write_text("")

    def tearDown(self):
        self.tmp.cleanup()

    def venv(self, cap_s):
        return build_venv(self.root / "venv", "3.11.15", self.lock, [], self.root / "cache",
                          self.root / "build.log", cap_s=cap_s)

    def snapshot(self):
        taskdir = self.root / "task"
        (taskdir / "agent/code").mkdir(parents=True)
        (taskdir / "agent/code/env.lock").write_text("")
        (taskdir / "runtime").mkdir()
        archive_path = taskdir / "runtime/tiny.tar"
        with tarfile.open(archive_path, "w") as archive:
            member = tarfile.TarInfo("bin/example")
            member.mode = 0o755
            member.size = 4
            archive.addfile(member, io.BytesIO(b"tiny"))
        manifest = {"version": 1, "python": "/usr/bin/python3", "mounts": [
            {"name": "tiny", "source": "runtime/tiny.tar", "target": "/opt/example/runtime",
             "format": "tar", "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest()}]}
        (taskdir / "runtime/runtime.json").write_text(json.dumps(manifest))
        return Task(taskdir, {"pid": "environment-budget-test", "env": {"mode": "snapshot"}})

    def build_snapshot(self, task, cap_s):
        return build_environment(task, self.root / "venv", self.lock, [], self.root / "cache",
                                 self.root / "build.log", cap_s=cap_s)

    def test_pip_receives_only_remaining_budget_without_a_floor(self):
        clock = [100.0]
        limits = []

        def fake_run(argv, **kwargs):
            limits.append(kwargs["timeout"])
            clock[0] += .3 if len(limits) == 1 else .1
            return {"rc": 0, "timed_out": False}

        with patch("bench.envlock.time.time", side_effect=lambda: clock[0]), \
                patch("bench.envlock.run", side_effect=fake_run):
            result = self.venv(.5)
        self.assertTrue(result["ok"])
        self.assertAlmostEqual(limits[0], .5)
        self.assertAlmostEqual(limits[1], .2)

    def test_exhausted_venv_budget_does_not_start_pip(self):
        clock = [100.0]

        def fake_run(*args, **kwargs):
            clock[0] += .6
            return {"rc": 0, "timed_out": False}

        with patch("bench.envlock.time.time", side_effect=lambda: clock[0]), \
                patch("bench.envlock.run", side_effect=fake_run) as run:
            result = self.venv(.5)
        self.assertEqual(run.call_count, 1)
        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertEqual(result["stage"], "sync")

    def test_operator_mirror_preserves_submission_pins_and_special_indexes(self):
        self.lock.write_text("torch==2.6.0+cpu\nnumpy==2.1.0\n")
        original = self.lock.read_bytes()
        calls = []
        def record(argv, **kwargs):
            calls.append(argv)
            return {"rc": 0, "timed_out": False}
        mirror = "https://mirror.example/simple/"
        with patch("bench.envlock.run", side_effect=record):
            result = build_venv(self.root / "venv", "3.11.15", self.lock,
                                ["https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"],
                                self.root / "cache", self.root / "build.log",
                                env={"ARB_PYPI_MIRROR": mirror})
        self.assertTrue(result["ok"])
        self.assertEqual(self.lock.read_bytes(), original)
        self.assertEqual(calls[0][calls[0].index("--index-url") + 1], mirror)
        self.assertEqual(calls[1][calls[1].index("--index-url") + 1], mirror)
        self.assertEqual(calls[1][calls[1].index("--extra-index-url") + 1],
                         "https://download.pytorch.org/whl/cpu")

    def test_verified_wheel_cache_can_replace_unreachable_cpu_index(self):
        cache = self.root / "wheels"
        cache.mkdir()
        calls = []
        def record(argv, **kwargs):
            calls.append(argv)
            return {"rc": 0, "timed_out": False}
        with patch("bench.envlock.run", side_effect=record):
            result = build_venv(self.root / "venv", "3.11.15", self.lock,
                ["https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"],
                self.root / "cache", self.root / "build.log", env={
                    "ARB_PYPI_MIRROR":"https://mirror.example/simple/",
                    "ARB_BUILD_WHEELHOUSE":str(cache), "ARB_CACHED_PYTORCH_CPU_INDEX":"1"})
        self.assertTrue(result["ok"])
        self.assertEqual(calls[1][calls[1].index("--find-links") + 1], str(cache))
        self.assertNotIn("https://download.pytorch.org/whl/cpu", calls[1])
        self.assertIn("https://mirror.example/simple/", calls[1])

    def test_exhausted_budget_does_not_remove_an_existing_venv(self):
        (self.root / "venv").mkdir()
        marker = self.root / "venv/keep"
        marker.write_text("keep")
        with patch("bench.envlock.run") as run:
            result = self.venv(0)
        run.assert_not_called()
        self.assertTrue(result["timed_out"])
        self.assertTrue(marker.is_file())

    def test_real_delayed_pip_stops_at_shared_deadline(self):
        fake_uv = self.root / "fake-uv"
        started = self.root / "pip-started"
        finished = self.root / "pip-finished"
        fake_uv.write_text(
            '#!/bin/sh\nif [ "$1" = venv ]; then\n sleep .08\nelse\n'
            f' printf "%s" "$$" > {shlex.quote(str(started))}\n'
            " sleep 2\n"
            f" printf finished > {shlex.quote(str(finished))}\nfi\n")
        fake_uv.chmod(0o755)
        start = time.monotonic()
        with patch.dict(os.environ, {"ARB_UV": str(fake_uv)}):
            result = self.venv(1)
        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertTrue(started.exists())
        self.assertFalse(finished.exists())
        self.assertLess(time.monotonic() - start, 1.8)
        self.assertFalse(Path(f"/proc/{int(started.read_text())}").exists())
        # Wait beyond the fixture's full sleep to detect delayed writes as well
        # as checking that its original process has already been reaped.
        time.sleep(2.1)
        self.assertFalse(finished.exists())

    def test_tiny_snapshot_builds_in_bounded_worker_and_reuses_cache(self):
        task = self.snapshot()
        with patch("bench.runtime.LOCAL_ROOT", self.root / "local"):
            result = self.build_snapshot(task, 5)
            data = materialize(task)
            again = self.build_snapshot(task, 5)
        self.assertTrue(result["ok"], result)
        self.assertFalse(result["timed_out"])
        self.assertTrue(again["ok"], again)
        self.assertEqual((data["mounts"][0]["host"] / "bin/example").read_bytes(), b"tiny")
        self.assertIn("/usr/bin/python3", (self.root / "venv/bin/python").read_text())

    def test_snapshot_with_exhausted_budget_does_not_spawn(self):
        task = self.snapshot()
        with patch("bench.runtime.capture") as spawn:
            result = self.build_snapshot(task, 0)
        spawn.assert_not_called()
        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertFalse((self.root / "venv/bin/python").exists())

    def test_blocked_snapshot_worker_is_stopped_without_publishing_launcher(self):
        task = self.snapshot()
        finished = self.root / "snapshot-finished"
        started = self.root / "snapshot-started"
        program = ("import os,pathlib,time; "
                   f"pathlib.Path({str(started)!r}).write_text(str(os.getpid())); "
                   "time.sleep(2); "
                   f"pathlib.Path({str(finished)!r}).write_text('finished')")
        requested_limits = []

        def delayed_worker(argv, **kwargs):
            self.assertEqual(argv[1:], ["-m", "bench.runtime"])
            requested_limits.append(kwargs["timeout"])
            return capture([sys.executable, "-c", program], **kwargs)

        start = time.monotonic()
        with patch("bench.runtime.capture", side_effect=delayed_worker):
            result = self.build_snapshot(task, 1)
        elapsed = time.monotonic() - start
        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])
        self.assertTrue(started.exists())
        self.assertLess(elapsed, 1.8)
        self.assertGreater(requested_limits[0], 0)
        self.assertLessEqual(requested_limits[0], 1)
        self.assertFalse((self.root / "venv/bin/python").exists())
        self.assertFalse(Path(f"/proc/{int(started.read_text())}").exists())
        time.sleep(2.1)
        self.assertFalse(finished.exists())

    def test_snapshot_cooperatively_rejects_expired_deadline(self):
        task = self.snapshot()
        with patch("bench.runtime._extract") as extract, self.assertRaises(TimeoutError):
            materialize(task, deadline_ts=time.time() - 1)
        extract.assert_not_called()


class ExactPythonTests(unittest.TestCase):
    """A venv must link its interpreter's real directory: see bench.envlock.exact_python."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.calls = self.root / "calls"
        self.installed = self.root / "installed"
        self.lock = self.root / "env.lock"
        self.lock.write_text("")
        python = self.root / "fake-python"
        python.write_text("#!/bin/sh\necho 3.11.15\n")
        python.chmod(0o755)
        self.uv = self.root / "fake-uv"
        self.write_uv(f'touch {shlex.quote(str(self.installed))}')
        self.python = python

    def tearDown(self):
        self.tmp.cleanup()

    def write_uv(self, install_action):
        self.uv.write_text(
            f'#!/bin/sh\necho "$@" >> {shlex.quote(str(self.calls))}\ncase "$1 $2" in\n'
            f'  "python find") [ -e {shlex.quote(str(self.installed))} ] || exit 1; echo {shlex.quote(str(self.root / "fake-python"))};;\n'
            f'  "python install") {install_action};;\nesac\n')
        self.uv.chmod(0o755)

    def build(self, python):
        with patch.dict(os.environ, {"ARB_UV": str(self.uv)}):
            return build_venv(self.root / "venv", python, self.lock, [], self.root / "cache", self.root / "build.log", cap_s=30)

    def logged(self):
        return self.calls.read_text().splitlines()

    def test_bare_minor_builds_from_the_exact_patch(self):
        self.installed.touch()
        self.assertTrue(self.build("3.11")["ok"])
        self.assertIn("venv --quiet --python 3.11.15 --seed --no-project " + str(self.root / "venv"), self.logged())
        self.assertFalse(any(line.startswith("python install") for line in self.logged()))
        self.assertIn("python 3.11 -> 3.11.15", (self.root / "build.log").read_text())

    def test_missing_interpreter_is_installed_without_touching_the_bin_dir(self):
        self.assertTrue(self.build("3.11")["ok"])
        self.assertIn("python install --no-bin 3.11", self.logged())
        self.assertIn("venv --quiet --python 3.11.15 --seed --no-project " + str(self.root / "venv"), self.logged())

    def test_exact_request_is_not_resolved(self):
        self.assertTrue(self.build("3.10.22")["ok"])
        self.assertFalse(any(line.startswith("python ") for line in self.logged()))
        self.assertIn("venv --quiet --python 3.10.22 --seed --no-project " + str(self.root / "venv"), self.logged())

    def test_unresolvable_interpreter_fails_instead_of_building_a_symlinked_venv(self):
        self.write_uv("echo no such python >&2; exit 1")
        result = self.build("3.11")
        self.assertFalse(result["ok"])
        self.assertFalse(result["timed_out"])
        self.assertEqual(result["stage"], "venv")
        self.assertFalse(any(line.startswith("venv") for line in self.logged()))
        self.assertIn("no such python", (self.root / "build.log").read_text())

    def test_expired_budget_while_resolving_reports_a_timeout(self):
        self.uv.write_text("#!/bin/sh\nsleep 5\n")
        with patch.dict(os.environ, {"ARB_UV": str(self.uv)}):
            result = build_venv(self.root / "venv", "3.11", self.lock, [], self.root / "cache", self.root / "build.log", cap_s=.3)
        self.assertFalse(result["ok"])
        self.assertTrue(result["timed_out"])


if __name__ == "__main__":
    unittest.main()
