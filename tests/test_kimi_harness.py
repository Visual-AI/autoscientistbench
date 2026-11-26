"""Kimi adapter contract tests, using fake credentials and native wire records."""
import json
import tempfile
import unittest
from pathlib import Path

from bench.harness import load_harness
from bench.harness.base import RunContext
from bench.harness.kimi import Kimi, _read_config, _write_config


class KimiHarnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "operator"
        self.source.mkdir()
        self.pkg = self.root / "package"
        (self.pkg / "dist").mkdir(parents=True)
        (self.pkg / "dist/main.mjs").write_text("// fake CLI\n")
        (self.pkg / "package.json").write_text('{"version":"2.6.2"}')
        (self.root / "rg").write_text("#!/bin/sh\nexit 0\n")
        self.model = {"provider": "managed:kimi-code", "model": "k3.1-pro",
                      "max_context_size": 1048576, "support_efforts": ["low", "high", "max"],
                      "default_effort": "high", "capabilities": ["thinking", "tool_use"]}
        self.config = {
            "default_model": "kimi-code/k3.1-pro", "thinking": {"enabled": True, "effort": "max"},
            "models": {"kimi-code/k3.1-pro": self.model,
                       "kimi-code/fast": {"provider": "managed:kimi-code", "model": "fast"}},
            "providers": {"managed:kimi-code": {"type": "kimi", "api_key": "",
                "base_url": "https://api.kimi.com/coding/v1",
                "oauth": {"storage": "file", "key": "oauth/kimi-code"}},
                "unrelated": {"type": "openai", "api_key": "must-not-copy"}},
            "hooks": [{"command": "must-not-run"}], "services": {"secret": "must-not-copy"},
            "mcp": {"servers": {"private": {"command": "must-not-run"}}},
        }
        _write_config(self.source / "config.toml", self.config)
        self.credential = self.source / "credentials/kimi-code.json"
        self.credential.parent.mkdir()
        self.credential.write_text(json.dumps(self.token("original", 100)))
        (self.source / "sessions").mkdir()
        (self.source / "sessions/operator-secret.txt").write_text("private history")
        (self.source / "server.token").write_text("must-not-copy")
        self.cfg = {"kimi_home": str(self.source), "package_dir": str(self.pkg), "rg_path": str(self.root / "rg")}

    @staticmethod
    def token(label, expires):
        return {"access_token": "fake-access-" + label, "refresh_token": "fake-refresh-" + label,
                "expires_at": expires}

    def prepare(self, model=None, **cfg):
        home = self.root / ("home-" + str(len(list(self.root.glob("home-*")))))
        home.mkdir()
        ctx = RunContext(run_id="test", pid="toy-ridge", budget_s=600, deadline_ts=0, cpus=2, port=18080,
                         ws_inside="/ws", home_inside="/home/agent", run_dir_inside="/run/arb",
                         prompt_text="Inspect the code; submit an improvement.",
                         last_message_inside="/run/arb/last_message.txt", model=model)
        h = Kimi("kimi", {**self.cfg, **cfg}, self.root)
        h.prepare_home(home, ctx)
        return h, home, ctx

    def test_registered_and_config_isolated(self):
        self.assertIsInstance(load_harness("kimi", self.cfg), Kimi)
        h, home, ctx = self.prepare()
        actual = _read_config(home / ".kimi-code/config.toml")
        self.assertEqual(list(actual["models"]), ["kimi-code/k3.1-pro"])
        self.assertEqual(list(actual["providers"]), ["managed:kimi-code"])
        self.assertEqual(actual["thinking"]["effort"], "max")
        self.assertFalse(actual["telemetry"])
        for key in ("mcp", "hooks", "services"):
            self.assertNotIn(key, actual)
        for name in ("sessions", "server.token"):
            self.assertFalse((home / ".kimi-code" / name).exists())
        self.assertEqual((home / ".kimi-code/credentials/kimi-code.json").stat().st_mode & 0o777, 0o600)
        self.assertNotIn("Agent", actual["tools"]["enabled"])
        self.assertIn("Bash", actual["tools"]["enabled"])
        self.assertEqual(h.allow_hosts(), ["api.kimi.com", "auth.kimi.com", "code.kimi.com"])
        self.assertTrue((home / ".kimi-code-pkg/bin/rg").stat().st_mode & 0o111)
        self.assertIn("/home/agent/.kimi-code-pkg/bin", h.env(ctx)["PATH"])

    def test_headless_flags_and_model_override(self):
        h, home, ctx = self.prepare("kimi-code/fast")
        args = h.command(ctx)
        self.assertIn("--use-env-proxy", args)
        self.assertEqual(args[args.index("--model") + 1], "kimi-code/fast")
        self.assertEqual(args[args.index("--prompt") + 1], ctx.prompt_text)
        self.assertIn("stream-json", args)
        self.assertFalse({"--auto", "--yolo", "--continue", "--session"}.intersection(args))
        self.assertEqual(h.stdin(ctx), "")
        self.assertNotIn("effort", _read_config(home / ".kimi-code/config.toml")["thinking"])

    def test_unknown_model_or_effort_fails_before_agent_start(self):
        for model, cfg in [("absent", {}), ("kimi-code/k3.1-pro", {"reasoning_effort": "xhigh"}),
                           ("kimi-code/fast", {"reasoning_effort": "high"})]:
            with self.subTest(model=model, cfg=cfg), self.assertRaises(ValueError):
                self.prepare(model, **cfg)

    def test_refresh_syncs_atomically_and_preserves_mode(self):
        h, home, ctx = self.prepare()
        refreshed = self.token("refreshed", 200)
        (home / ".kimi-code/credentials/kimi-code.json").write_text(json.dumps(refreshed))
        h.after_run(home, ctx)
        self.assertEqual(json.loads(self.credential.read_text()), refreshed)
        self.assertEqual(self.credential.stat().st_mode & 0o777, 0o600)
        self.assertEqual(h._auth_sync, "refreshed")

    def test_concurrent_run_cannot_replace_newer_login(self):
        first, home1, ctx1 = self.prepare()
        second, home2, ctx2 = self.prepare()
        newer = self.token("newer", 300)
        (home1 / ".kimi-code/credentials/kimi-code.json").write_text(json.dumps(newer))
        (home2 / ".kimi-code/credentials/kimi-code.json").write_text(json.dumps(self.token("older", 200)))
        first.after_run(home1, ctx1)
        second.after_run(home2, ctx2)
        self.assertEqual(json.loads(self.credential.read_text()), newer)
        self.assertEqual(second._auth_sync, "source_changed_preserved")

    def test_invalid_or_older_refresh_is_ignored(self):
        for token in ({"access_token": "incomplete"}, self.token("stale", 50)):
            h, home, ctx = self.prepare()
            (home / ".kimi-code/credentials/kimi-code.json").write_text(json.dumps(token))
            h.after_run(home, ctx)
            self.assertEqual(json.loads(self.credential.read_text()), self.token("original", 100))

    def test_native_usage_archive_and_partial_record(self):
        h, home, ctx = self.prepare()
        wire = home / ".kimi-code/sessions/workspace/session/agents/main/wire.jsonl"
        wire.parent.mkdir(parents=True)
        usage = {"inputOther": 100, "inputCacheRead": 50, "inputCacheCreation": 10, "output": 30}
        rows = [{"type": "metadata", "protocol_version": "1.5"},
                {"type": "event", "event": {"type": "usage.record", "model": "k3.1-pro", "usage": usage}},
                {"type": "usage.record", "model": "k3.1-pro", "usage": usage},
                {"type": "state.snapshot", "usage": usage},  # NOT another model call
                {"type": "turn.ended", "outcome": "completed"}]
        wire.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"interrupted":')
        rec = self.root / "rec"
        rec.mkdir()
        events = [{"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "Bash"}}]},
                  {"role": "tool", "tool_call_id": "c1", "content": "done"},
                  {"role": "assistant", "content": "Submitted."}]
        (rec / "agent_events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in events))
        result = h.collect(rec)
        self.assertEqual(result["tokens_in"], 320)
        self.assertEqual(result["tokens_out"], 60)
        self.assertEqual(result["tokens_cached"], 100)
        self.assertEqual(result["model_calls"], 2)
        self.assertEqual(result["commands"], 1)
        self.assertEqual(result["turns"], 1)
        self.assertIsNone(result["tokens_reasoning"])
        self.assertEqual((rec / "last_message.txt").read_text(), "Submitted.")
        self.assertTrue(list((rec / "trace/kimi").rglob("wire.jsonl")))
        self.assertFalse(list((rec / "trace/kimi").rglob("kimi-code.json")))
        # Re-analysis after bench deletes the sandbox must retain model provenance.
        recovered = Kimi("kimi", self.cfg, self.root).collect(rec)
        self.assertEqual(recovered["cli_version"], "2.6.2")
        self.assertEqual(recovered["model"], "kimi-code/k3.1-pro")
        self.assertEqual(recovered["tokens_in"], result["tokens_in"])

    def test_partially_reported_cache_usage_is_not_zero(self):
        h, home, ctx = self.prepare()
        wire = home / ".kimi-code/sessions/workspace/session/agents/main/wire.jsonl"
        wire.parent.mkdir(parents=True)
        rows = [
            {"type": "usage.record", "model": "k3.1-pro", "usage":
             {"inputOther": 6910, "output": 92, "inputCacheCreation": 0}},
            {"type": "usage.record", "model": "k3.1-pro", "usage":
             {"inputOther": 134, "output": 38, "inputCacheRead": 6912, "inputCacheCreation": 0}},
        ]
        wire.write_text("".join(json.dumps(r) + "\n" for r in rows))
        result = h.collect(self.root / "partial-rec")
        self.assertEqual(result["tokens_out"], 130)
        self.assertEqual(result["usage"]["inputCacheRead"], 6912)
        self.assertFalse(result["cache_usage_complete"])
        self.assertIsNone(result["tokens_cached"])
        self.assertIsNone(result["tokens_in"])

    def test_missing_usage_is_unknown_not_zero(self):
        h, home, ctx = self.prepare()
        result = h.collect(self.root / "empty-rec")
        self.assertIsNone(result["tokens_in"])
        self.assertIsNone(result["tokens_out"])
        self.assertEqual(result["usage_source"], "missing")


if __name__ == "__main__":
    unittest.main()
