"""A frozen submission must retain installed runtime/build dependencies on replay."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from bench import envlock


@unittest.skipUnless(shutil.which("uv"), "requires uv for an offline environment round trip")
class EnvironmentReplayTests(unittest.TestCase):
    def test_packaging_dependency_imports_survive_freeze_and_fresh_sync(self):
        # Tiny local wheels exercise the real installer, metadata discovery,
        # submission freeze and clean-environment sync without network access.
        with tempfile.TemporaryDirectory(prefix="arb-envlock-replay-") as td:
            root = Path(td)
            wheels = root / "wheels"
            wheels.mkdir()
            for name, module in (("setuptools", "pkg_resources"), ("wheel", "wheel")):
                dist = f"{name}-1.0.dist-info"
                entries = {
                    f"{module}/__init__.py": "replay_marker = 'available'\n",
                    f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n",
                    f"{dist}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
                }
                entries[f"{dist}/RECORD"] = "".join(f"{p},,\n" for p in [*entries, f"{dist}/RECORD"])
                with zipfile.ZipFile(wheels / f"{name}-1.0-py3-none-any.whl", "w") as z:
                    for path, content in entries.items():
                        z.writestr(path, content)
            env = dict(os.environ, UV_CACHE_DIR=str(root / "cache"), UV_OFFLINE="1",
                       UV_NO_PROGRESS="1", UV_PYTHON_DOWNLOADS="never")
            uv = envlock.uv_bin()

            def run(*args):
                return subprocess.run(args, env=env, check=True, text=True, capture_output=True, timeout=30)

            agent, judge = root / "agent", root / "judge"
            for venv in (agent, judge):
                run(uv, "venv", "--no-project", "--python", sys.executable, str(venv))
            original = root / "original.lock"
            original.write_text("setuptools==1.0\nwheel==1.0\n")
            run(uv, "pip", "sync", "--no-index", "--find-links", str(wheels),
                "--python", str(agent / "bin/python"), str(original))
            lock = root / "submission.lock"
            lock.write_text(envlock.freeze(agent, env=env))
            run(uv, "pip", "sync", "--no-index", "--find-links", str(wheels),
                "--python", str(judge / "bin/python"), str(lock))
            probe = run(str(judge / "bin/python"), "-I", "-c",
                        "import pkg_resources, wheel; "
                        "print(pkg_resources.replay_marker, wheel.replay_marker)")
            self.assertEqual(probe.stdout.strip(), "available available")
            self.assertEqual(envlock.freeze(judge, env=env), lock.read_text())


if __name__ == "__main__":
    unittest.main()
