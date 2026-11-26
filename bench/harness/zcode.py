"""ZCode (Z.AI's coding CLI) inside our bwrap, talking to a GLM endpoint over the Anthropic protocol.

Why not the codex harness: codex dropped `wire_api = "chat"` ("no longer supported", it insists on the Responses
API) and BigModel serves neither /responses nor the Responses protocol, so driving GLM through codex would need a
Responses->chat translator in the middle of the measurement path. ZCode speaks GLM natively, so it is the smaller
and more honest piece of machinery.

Packaging: `zcode` on PATH is an npm shim that only manages versions and OAuth; the agent itself is pure JS at
<package>/vendor/zcode.cjs, run as `node zcode.cjs ...`. The sandbox already ro_binds /usr, so node is there; the
67 MB package is copied into each run's HOME instead (the Harness interface can only add files through HOME).

Credentials: like traecli, the key sits in the sandbox HOME (~/.zcode/cli/config.json) and the agent could read
it, so the config written here keeps ONLY the provider this run needs -- the other providers' keys from the
operator's own ~/.zcode never enter the sandbox. The proxy allowlists the one upstream host by CONNECT.

Thinking: GLM-5.3 and GLM-5.3-Flash always reason and cannot be turned off; the depth is `reasoning_effort` with
low/high/max and **max is the model's default**, so an unset effort already means max (docs.bigmodel.cn, checked
2026-11-09). `reasoning_effort` in the config below is recorded for provenance and sent when the runtime accepts it.
"""
import json
import os
import shutil
from pathlib import Path

from . import register
from .base import Harness, read_jsonl

PKG_IN_HOME = ".zcode-pkg"


@register("zcode")
class ZCode(Harness):
    type_name = "zcode"

    def tools(self):
        return ["node"]

    def allow_hosts(self):
        return list(self.cfg.get("allow_hosts") or ["open.bigmodel.cn"])

    def _pkg_src(self):
        p = self.cfg.get("package_dir") or os.environ.get("ARB_ZCODE_PKG")
        if not p:
            shim = shutil.which("zcode")
            if not shim:
                raise ValueError("zcode harness: set package_dir, or put the zcode shim on PATH")
            p = Path(os.path.realpath(shim)).parents[1]      # <pkg>/bin/zcode.js -> <pkg>
        p = Path(os.path.expanduser(str(p)))
        if not (p / "vendor" / "zcode.cjs").is_file():
            raise ValueError(f"zcode harness: {p}/vendor/zcode.cjs not found")
        return p

    def prepare_home(self, home_host: Path, ctx):
        shutil.copytree(self._pkg_src(), home_host / PKG_IN_HOME, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.map", "__pycache__"))
        # Start from the operator's own working config and only PRUNE it, rather than synthesising one: a minimal
        # config of our own left out `options.apiKeyRequired` and `model.lite`, and the runtime then sent the whole
        # session to zcode.z.ai (2.2 MB over 590 s) instead of the configured provider -- the eval would have
        # measured whatever Z.AI's own gateway routes to. Measured 2026-11-09.
        model = ctx.model or self.cfg.get("model", "glm-5.3-flash")
        provider = self.cfg.get("provider_name", "bigmodel")
        src = Path(os.path.expanduser(self.cfg.get("source_config") or "~/.zcode/cli/config.json"))
        base = json.loads(src.read_text()) if src.is_file() else {}
        providers = base.get("provider") or {}
        if provider not in providers:
            raise ValueError(f"zcode harness: provider {provider!r} is not in {src} (have {sorted(providers)})")
        # Only the provider this run needs: the other providers' keys never enter a sandbox.
        keep = {provider: json.loads(json.dumps(providers[provider]))}
        if model not in (keep[provider].get("models") or {}):
            raise ValueError(f"zcode harness: {provider} has no model {model!r} in {src}")
        opts = keep[provider].setdefault("options", {})
        if self.cfg.get("base_url"):
            opts["baseURL"] = self.cfg["base_url"]
        # The context window and output cap the runtime plans against come from the provider's own model entry, so a
        # self-hosted endpoint must say what it really serves: against BigModel's 1M/128k entry the runtime reported
        # contextWindow 1000000 and asked for max_tokens 128000 (measured 2026-11-10), which on a 256K vLLM would
        # both delay compaction and overrun the server's limit.
        if self.cfg.get("model_limits"):
            keep[provider]["models"][model] = {**(keep[provider]["models"][model] or {}),
                                               **self.cfg["model_limits"]}
        key_env = self.cfg.get("api_key_env")
        key = os.environ.get(key_env) if key_env else None
        if key_env and not key and not opts.get("apiKey"):
            raise ValueError(f"zcode harness: {key_env} is not set and {src} has no key for {provider}")
        if key:
            opts["apiKey"] = key
        cfg = dict(base)
        cfg["provider"] = keep
        cfg["model"] = {"main": f"{provider}/{model}"}
        lite = self.cfg.get("lite_model")
        if lite is None:                      # keep the operator's lite model when it is this provider's
            was = (base.get("model") or {}).get("lite") or ""
            lite = was.split("/", 1)[1] if was.startswith(provider + "/") else None
        if lite:
            cfg["model"]["lite"] = f"{provider}/{lite}"
        cfg.update(self.cfg.get("config_extra") or {})
        d = home_host / ".zcode" / "cli"
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps(cfg, indent=2))
        # Tell the shim-less runtime not to look for an update or a newer runtime to unpack.
        (d / "version.json").write_text(json.dumps({"checkedVersion": "pinned", "latestVersion": "pinned",
                                                    "lastCheckedAt": "2026-01-01T00:00:00.000Z"}))

    def command(self, ctx):
        # The prompt goes in argv: `--prompt <text>` is the runtime's documented headless form, while `--print`
        # (which its own --help lists) wants a positional and rejects the flag -- "Unknown option '--print'",
        # measured 2026-11-09. Prompts here are 14-70 KB, far below the ~2 MB argv limit.
        # --use-env-proxy: the sandbox has no DNS and reaches the network only through the proxy bench exports as
        # HTTP(S)_PROXY, but node's built-in fetch ignores those variables unless this flag is given (node 24+).
        # Without it the first request died with `getaddrinfo EAI_AGAIN open.bigmodel.cn`, measured 2026-11-09.
        # --output-format stream-json, not --json: `--json` prints ONE end-of-session object, so the first real
        # GLM pass (28 agents, 19 model requests and 8302 internal events each) left no transcript at all and
        # nothing to explain why every one of them stopped at ~5% of its budget. The stream carries typed events
        # (turn.started, model.streaming, session.updated, turn.completed with toolCallCount/stopReason) and ends
        # with the very same summary object as `--json`, as `type: "result"` -- strictly more, measured against the
        # local stub 2026-11-10.
        cmd = ["node", "--use-env-proxy", f"{ctx.home_inside}/{PKG_IN_HOME}/vendor/zcode.cjs",
               "--prompt", ctx.prompt_text, "--output-format", "stream-json",
               "--mode", self.cfg.get("mode", "yolo"), "--cwd", ctx.ws_inside,
               "--surface", "terminal", "--no-color"]
        for t in self.cfg.get("disallowed_tools") or []:
            cmd += ["--disallowed-tools", t]
        cmd += list(self.cfg.get("extra_args") or [])
        return cmd

    def stdin(self, ctx):
        # Nothing: the prompt is in argv, and feeding an unread pipe would block once it passed the 64 KB buffer.
        return ""

    def env(self, ctx):
        return {"NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
                "ZCODE_DISABLE_AUTOUPDATE": "1", "CI": "1", **(self.cfg.get("env") or {})}

    def collect(self, rec_dir: Path):
        """ZCode's own event schema, which the codex-shaped default reads as all zeros.

        Its names differ throughout: usage is camelCase (inputTokens, not input_tokens), a turn ends with
        `turn.completed` carrying `toolCallCount` rather than one `item.completed` per command, and the stream's
        last line is `type: "result"` holding the session total. `end_reason` and `stop_reason` are recorded
        because this route's open question is why a session goes idle with most of its budget left.
        """
        evs = read_jsonl(rec_dir / "agent_events.jsonl")
        tail = next((e for e in reversed(evs) if e.get("type") == "result"), {})
        u = tail.get("usage") or {}
        turns = [e for e in evs if e.get("type") == "turn.completed"]
        commands = sum(int((e.get("payload") or {}).get("toolCallCount") or 0) for e in turns)
        last = (turns[-1].get("payload") if turns else {}) or {}
        return {"events": int(tail.get("eventCount") or len(evs)), "turns": len(turns),
                "items": {"tool_calls": commands}, "commands": commands,
                "errors": sum(1 for e in evs if "error" in str(e.get("type") or "").lower()),
                "tokens_in": int(u.get("inputTokens") or 0), "tokens_out": int(u.get("outputTokens") or 0),
                "tokens_cached": int(u.get("cacheReadTokens") or 0),
                "tokens_reasoning": int(u.get("reasoningTokens") or 0),
                "model_calls": int(u.get("modelRequestCount") or 0),
                "end_status": (tail.get("projection") or {}).get("status"),
                "end_reason": last.get("resultType"), "stop_reason": last.get("stopReason"),
                "last_message": str(tail.get("response") or "")[:2000], "usage": u}
