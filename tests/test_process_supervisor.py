import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from bench import process_supervisor


SCRIPT = Path(process_supervisor.__file__).resolve()


class ProcessSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream:
                    stream.close()
        self.tmp.cleanup()

    def launch(self, code, *args, **kwargs):
        process = subprocess.Popen([sys.executable, str(SCRIPT), "--", sys.executable, "-c", code, *args],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   start_new_session=True, **kwargs)
        self.processes.append(process)
        return process

    def wait_file(self, path):
        deadline = time.monotonic() + 3
        while not path.exists():
            self.assertLess(time.monotonic(), deadline, f"child did not create {path}")
            time.sleep(.01)

    def detached_child(self, name="child"):
        pidfile, late = self.root / (name + ".pid"), self.root / (name + ".late")
        code = ("import os,pathlib,signal,time; "
                "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); "
                "time.sleep(.7); "
                f"pathlib.Path({str(late)!r}).write_text('late')")
        return pidfile, late, code

    def parent(self, child, pidfile, *, delay=0, exitcode=0):
        return ("import pathlib,subprocess,sys,time\n"
                f"subprocess.Popen([sys.executable,'-c',{child!r}],start_new_session=True)\n"
                f"while not pathlib.Path({str(pidfile)!r}).exists(): time.sleep(.005)\n"
                f"time.sleep({delay})\nraise SystemExit({exitcode})\n")

    def assert_reaped(self, pidfile):
        pid = int(pidfile.read_text())
        self.assertFalse(Path(f"/proc/{pid}").exists(), f"descendant {pid} was not reaped")

    def test_normal_exit_kills_and_reaps_detached_background(self):
        pidfile, late, child = self.detached_child()
        process = self.launch(self.parent(child, pidfile, exitcode=7))
        stdout, stderr = process.communicate(timeout=3)
        self.assertEqual(process.returncode, 7, stderr)
        self.assertEqual(stdout, "")
        self.assert_reaped(pidfile)
        self.assertFalse(late.exists())

    def test_timeout_cleans_setsid_child_holding_output_pipe(self):
        pidfile, late, child = self.detached_child()
        process = self.launch(self.parent(child, pidfile, delay=10))
        self.wait_file(pidfile)
        with self.assertRaises(subprocess.TimeoutExpired):
            process.communicate(timeout=.1)
        start = time.monotonic()
        process.terminate()  # Only signal the supervisor, not its process group.
        _, stderr = process.communicate(timeout=3)
        self.assertEqual(process.returncode, -signal.SIGTERM, stderr)
        self.assertLess(time.monotonic() - start, 1.5)
        self.assert_reaped(pidfile)
        self.assertFalse(late.exists())

    def test_double_fork_grandchild_is_adopted_and_reaped(self):
        pidfile, late = self.root / "grandchild.pid", self.root / "grandchild.late"
        code = (
            "import os,pathlib,signal,time\n"
            "first=os.fork()\n"
            "if first == 0:\n"
            " os.setsid()\n"
            " if os.fork(): os._exit(0)\n"
            " signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
            f" pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()))\n"
            " time.sleep(.7)\n"
            f" pathlib.Path({str(late)!r}).write_text('late')\n"
            " os._exit(0)\n"
            "os.waitpid(first,0)\n"
            f"while not pathlib.Path({str(pidfile)!r}).exists(): time.sleep(.005)\n")
        process = self.launch(code)
        _, stderr = process.communicate(timeout=3)
        self.assertEqual(process.returncode, 0, stderr)
        self.assert_reaped(pidfile)
        self.assertFalse(late.exists())

    def test_parallel_callers_do_not_kill_each_others_processes(self):
        other_started, other_done = self.root / "other.started", self.root / "other.done"
        other = ("import pathlib,time; "
                 f"pathlib.Path({str(other_started)!r}).write_text('started'); "
                 "time.sleep(.6); "
                 f"pathlib.Path({str(other_done)!r}).write_text('completed')")
        pidfile, late, child = self.detached_child()

        def execute(code):
            process = self.launch(code)
            stdout, stderr = process.communicate(timeout=3)
            return process.returncode, stdout, stderr

        with ThreadPoolExecutor(max_workers=2) as pool:
            long = pool.submit(execute, other)
            self.wait_file(other_started)
            short = pool.submit(execute, self.parent(child, pidfile))
            self.assertEqual(short.result(timeout=3)[0], 0)
            self.assertEqual(long.result(timeout=3)[0], 0)
        self.assertTrue(other_done.exists())
        self.assert_reaped(pidfile)
        self.assertFalse(late.exists())

    def test_preserves_cwd_environment_stdio_arguments_and_normal_exit_code(self):
        code = ("import json,os,sys; "
                "print(json.dumps({'cwd':os.getcwd(),'env':os.environ['SUPERVISOR_TEST_VALUE'],"
                "'stdin':sys.stdin.read(),'args':sys.argv[1:]})); "
                "print('stderr unchanged',file=sys.stderr); sys.exit(23)")
        value = "literal $HOME ; -- spaces"
        env = {**os.environ, "SUPERVISOR_TEST_VALUE": value}
        process = self.launch(code, value, cwd=self.root, env=env, stdin=subprocess.PIPE)
        stdout, stderr = process.communicate(input="input unchanged", timeout=3)
        self.assertEqual(process.returncode, 23)
        self.assertEqual(json.loads(stdout), {"cwd": str(self.root), "env": value,
                                             "stdin": "input unchanged", "args": [value]})
        self.assertEqual(stderr, "stderr unchanged\n")

    def test_preserves_signal_exit_status(self):
        process = self.launch("import os,signal; os.kill(os.getpid(),signal.SIGUSR1)")
        process.communicate(timeout=3)
        self.assertEqual(process.returncode, -signal.SIGUSR1)

    def test_unavailable_subreaper_fails_before_starting_command(self):
        with patch.object(process_supervisor, "_enable_subreaper", side_effect=PermissionError("blocked")), \
                patch.object(process_supervisor.subprocess, "Popen") as spawn, \
                contextlib.redirect_stderr(io.StringIO()):
            result = process_supervisor.main(["--", "/bin/true"])
        self.assertEqual(result, 125)
        spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
