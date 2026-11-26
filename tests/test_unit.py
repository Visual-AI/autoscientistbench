"""Unit tests that need no model and no sandbox: patch, envlock, score, proxy allowlist, task defaults.
Run: python3 -m unittest discover -s tests -v   (from autoresearch-bench/)"""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import agentcli, envlock, evalrun, patch, score  # noqa: E402
from bench.proxy import IMAGE_NOTE, ProxyConfig, strip_images  # noqa: E402
from bench.task import DEFAULTS, Task  # noqa: E402

GIT_ENV = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@x")


def git(args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV, check=True, capture_output=True, text=True).stdout


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.repo = Path(self.td.name) / "code"
        self.repo.mkdir()
        (self.repo / "model.py").write_text("LAMBDA = 1.0\n")
        (self.repo / "env.lock").write_text("numpy==2.2.6\n")
        (self.repo / ".gitignore").write_text("*.pt\n__pycache__/\n")
        git(["init", "-q"], self.repo)
        git(["add", "-A"], self.repo)
        git(["commit", "-q", "-m", "base"], self.repo)
        self.base = git(["rev-parse", "HEAD"], self.repo).strip()

    def tearDown(self):
        self.td.cleanup()

    def test_make_check_apply_roundtrip(self):
        (self.repo / "model.py").write_text("LAMBDA = 0.1\n")
        (self.repo / "new_module.py").write_text("print('new')\n")      # untracked, must be included
        (self.repo / "weights.pt").write_bytes(b"\x00\x01")             # gitignored, must be excluded
        pb = patch.make_patch(self.repo, self.base)
        ok, problems, files = patch.check_patch(pb, DEFAULTS["patch_limits"])
        self.assertTrue(ok, problems)
        self.assertEqual(sorted(f["path"] for f in files), ["model.py", "new_module.py"])
        clone = Path(self.td.name) / "clone"
        patch.clean_clone(self.repo, clone, self.base)
        pp = Path(self.td.name) / "p.patch"
        pp.write_bytes(pb)
        patch.apply_patch(clone, pp)
        self.assertEqual((clone / "model.py").read_text(), "LAMBDA = 0.1\n")
        self.assertTrue((clone / "new_module.py").exists())
        self.assertFalse((clone / "weights.pt").exists())

    def test_agent_commits_do_not_matter(self):
        (self.repo / "model.py").write_text("LAMBDA = 0.5\n")
        git(["commit", "-q", "-am", "agent commit"], self.repo)
        (self.repo / "model.py").write_text("LAMBDA = 0.25\n")
        pb = patch.make_patch(self.repo, self.base)
        self.assertIn("+LAMBDA = 0.25", pb.decode())
        self.assertNotIn("+LAMBDA = 0.5\n", pb.decode())

    def test_rejects_binary_and_data_files(self):
        (self.repo / "cache.npy").write_bytes(b"\x93NUMPY\x00\x00\x01")
        (self.repo / "blob.bin").write_bytes(bytes(range(256)))
        pb = patch.make_patch(self.repo, self.base)
        ok, problems, files = patch.check_patch(pb, DEFAULTS["patch_limits"])
        self.assertFalse(ok)
        self.assertTrue(any("cache.npy" in p for p in problems), problems)
        self.assertTrue(any("binary" in p for p in problems), problems)

    def test_size_limit(self):
        (self.repo / "big.py").write_text("x = 1\n" * 200000)
        pb = patch.make_patch(self.repo, self.base)
        ok, problems, _ = patch.check_patch(pb, {"max_bytes": 1000, "forbidden_new_ext": []})
        self.assertFalse(ok)

    def test_empty_patch(self):
        pb = patch.make_patch(self.repo, self.base)
        self.assertEqual(pb.strip(), b"")
        ok, problems, files = patch.check_patch(pb, DEFAULTS["patch_limits"])
        self.assertTrue(ok)
        self.assertEqual(files, [])


class EnvLockTests(unittest.TestCase):
    def test_validate(self):
        ok, probs, pins = envlock.validate_lock_text("numpy==2.2.6\ntorch==2.5.1+cpu\n# comment\nscikit_learn[extra]==1.5.0\n")
        self.assertTrue(ok, probs)
        self.assertEqual(pins[0], ("numpy", "2.2.6"))
        self.assertEqual(pins[2][0], "scikit-learn")
        for bad in ("-e /ws/code/mylib", "mylib @ file:///ws/code/mylib", "foo @ git+https://x/y.git", "requests>=2", "--index-url x"):
            ok, probs, _ = envlock.validate_lock_text(bad + "\n")
            self.assertFalse(ok, bad)

    def test_sync_args(self):
        self.assertEqual(envlock.sync_args(["https://pypi.org/simple"]), ["--index-url", "https://pypi.org/simple"])
        a = envlock.sync_args(["https://pypi.org/simple", "https://download.pytorch.org/whl/cpu"])
        self.assertIn("--extra-index-url", a)
        self.assertIn("unsafe-best-match", a)


class ScoreTests(unittest.TestCase):
    def test_gain_direction(self):
        self.assertAlmostEqual(score.gain(0.9, 1.0, higher_is_better=False), 0.1)
        self.assertAlmostEqual(score.gain(0.9, 1.0, higher_is_better=True), -0.1)

    def test_decide(self):
        d = score.decide(0.95, 1.0, 0.0, False, 0.02)
        self.assertTrue(d["pass"])
        d = score.decide(0.99, 1.0, 0.0, False, 0.02)
        self.assertFalse(d["pass"])
        d = score.decide(0.95, 1.0, 0.05, False, 0.02)   # noisy reference: threshold 2 sigma = 10 %
        self.assertFalse(d["pass"])
        self.assertAlmostEqual(d["threshold"], 0.10)
        d = score.decide(2.0, 1.0, 0.0, False, 0.02)     # much worse -> clipped at -10 %
        self.assertEqual(d["gain_clipped"], score.FAIL_GAIN)

    def test_aggregate_and_parity(self):
        a = score.aggregate([1.0, 1.1, 0.9])
        self.assertAlmostEqual(a["mean"], 1.0)
        self.assertEqual(a["n"], 3)
        ok, info = score.parity_ok(1.02, 1.0, 0.1)
        self.assertTrue(ok)
        ok, info = score.parity_ok(1.2, 1.0, 0.1)
        self.assertFalse(ok)
        ok, _ = score.parity_ok(1.0, 1.0, 0.0)
        self.assertTrue(ok)


class ProxyTests(unittest.TestCase):
    def test_allowlist(self):
        cfg = ProxyConfig(["pypi.org", "*.pythonhosted.org", "api.example.org"])
        self.assertTrue(cfg.host_allowed("pypi.org"))
        self.assertTrue(cfg.host_allowed("files.pythonhosted.org"))
        self.assertTrue(cfg.host_allowed("PYPI.ORG"))
        self.assertFalse(cfg.host_allowed("example.com"))
        self.assertFalse(cfg.host_allowed("evilpypi.org"))
        self.assertFalse(cfg.host_allowed("pypi.org.evil.com"))

    def test_strip_images(self):
        req = {"input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "a"}, {"type": "input_image", "image_url": "data:x"}]},
            {"type": "function_call_output", "call_id": "c", "output": [{"type": "input_image", "image_url": "data:y"}]},
            {"type": "function_call_output", "call_id": "d", "output": "plain"}]}
        self.assertEqual(strip_images(req), 2)
        self.assertEqual(req["input"][0]["content"], [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": IMAGE_NOTE}])
        self.assertEqual(req["input"][1]["output"], IMAGE_NOTE)
        self.assertEqual(req["input"][2]["output"], "plain")
        self.assertEqual(strip_images({"input": "text"}), 0)


class CodexConfigTests(unittest.TestCase):
    def test_toml_layout_both_auth_modes(self):
        import tomllib
        from bench.harness.codex import Codex
        from bench.harness.base import RunContext
        ctx = RunContext(run_id="r", pid="p", budget_s=10, deadline_ts=0, cpus=1, port=18080, ws_inside="/ws", home_inside="/h",
                         run_dir_inside="/run/arb", prompt_text="", last_message_inside="/run/arb/last.txt", model=None)
        with tempfile.TemporaryDirectory() as td:
            api = Codex("codex-api", {"auth": "api", "model": "m", "upstream_base_url": "http://127.0.0.1:1/v1"}, td)
            api.prepare_home(Path(td), ctx)
            d = tomllib.loads((Path(td) / ".codex" / "config.toml").read_text())
            self.assertEqual(d["web_search"], "disabled")
            self.assertEqual(d["model_provider"], "bench")
            self.assertEqual(d["model_providers"]["bench"]["wire_api"], "responses")
            self.assertIn("model_providers.bench.base_url", " ".join(api.command(ctx)))
        with tempfile.TemporaryDirectory() as td:
            fake_home = Path(td) / "codexhome"
            fake_home.mkdir()
            (fake_home / "auth.json").write_text("{}")
            sub = Codex("codex", {"auth": "subscription", "model": "m", "codex_home": str(fake_home)}, td)
            sub.prepare_home(Path(td) / "home", ctx)
            d = tomllib.loads((Path(td) / "home" / ".codex" / "config.toml").read_text())
            self.assertEqual(d["web_search"], "disabled")
            self.assertNotIn("model_provider", d)
            self.assertTrue((Path(td) / "home" / ".codex" / "auth.json").exists())
            self.assertNotIn("model_providers.bench.base_url", " ".join(sub.command(ctx)))
            self.assertIn("chatgpt.com", sub.allow_hosts())


class TaskTests(unittest.TestCase):
    def test_defaults_and_overall_budget_policy(self):
        t = Task("/nonexistent/x", {"pid": "x", "metric": {"name": "mse", "higher_is_better": False},
                                    "baseline": {"train_min": 10, "eval_min": 2},
                                    "judge": {"train_wall_cap_s": 1, "predict_wall_cap_s": 2, "score_wall_cap_s": 3}})
        self.assertIsNone(t.train_cap_s)
        self.assertIsNone(t.predict_cap_s)
        self.assertIsNone(t.score_cap_s)
        self.assertEqual(t.seeds, [1, 2, 3])
        self.assertIn("files.pythonhosted.org", t.index_hosts())
        self.assertIn("agent/README.md missing", t.problems())


class AgentCliTests(unittest.TestCase):
    def test_tools_import_siblings_but_stages_stay_safe(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "common.py").write_text("OK = True\n")
            (d / "validate.py").write_text("from common import OK\nprint(OK)\n")
            st = {"venv": sys.prefix, "cpus": 1, "data": str(d), "bench": str(d), "code": str(d)}
            ok, msg = evalrun.validate(d / "validate.py", "p", "i", sys.executable, agentcli._tool_env(st, 0))
            self.assertTrue(ok, msg)
            self.assertEqual(agentcli._exec_env(st, 0)["PYTHONSAFEPATH"], "1")
            ok, msg = evalrun.validate(d / "validate.py", "p", "i", sys.executable, agentcli._exec_env(st, 0))
            self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
