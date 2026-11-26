"""codex exec inside our bwrap; its own sandbox is off (-s danger-full-access) because the closure is ours.

Two auth modes (config "auth"):
  subscription  ChatGPT login. The operator's $CODEX_HOME/auth.json is copied into the sandbox HOME; codex talks TLS
                end-to-end to chatgpt.com / auth.openai.com through the proxy's CONNECT allowlist. Nothing to inject,
                nothing to record (token usage comes from the --json events). Refreshed tokens are copied back after
                the run (auth_sync_back) so the operator's login does not go stale.
  api           An OpenAI-compatible endpoint with a key: codex -> 127.0.0.1:<port>/t/<run>/main/v1 -> bridge ->
                proxy (injects the key from upstream_api_key_env, records every exchange) -> upstream_base_url.
                The key never enters the sandbox.
"""
import os
import shutil
from pathlib import Path

from . import register
from .base import Harness

DEFAULT_OPENAI_HOSTS = ["chatgpt.com", "*.chatgpt.com", "auth.openai.com", "api.openai.com", "*.openai.com"]

CONFIG_TOML = """# written by bench-run for one sandboxed run (top-level keys first: TOML puts later keys into the last table)
model = "{model}"
{provider_key}model_reasoning_effort = "{effort}"
model_reasoning_summary = "none"
web_search = "disabled"
{extra}
[agents]
enabled = false
{provider_table}
[projects."/ws"]
trust_level = "trusted"
"""

PROVIDER_TABLE = """
[model_providers.bench]
name = "bench proxy"
base_url = "http://127.0.0.1:9/v1"
wire_api = "{wire_api}"
requires_openai_auth = false
request_max_retries = 4
stream_max_retries = 5
stream_idle_timeout_ms = {idle_ms}
"""


@register("codex")
class Codex(Harness):
    type_name = "codex"

    @property
    def auth(self):
        a = self.cfg.get("auth", "subscription")
        if a not in ("subscription", "api"):
            raise ValueError(f"codex auth must be subscription|api, got {a!r}")
        return a

    def tools(self):
        # codex-code-mode-host lives beside the codex binary, as traex-code-mode-host does beside traecli. codex
        # 0.167.0 runs every command through it, and without it each tool call fails with "failed to spawn code-mode
        # host /opt/arb/bin/codex-code-mode-host" -- the agent then ends its own turn in 30 s having run nothing
        # (measured 2026-11-10). tool_paths() simply leaves out a name it cannot find, so 0.146.0, which ships no
        # such helper, is unaffected.
        return ["codex", "codex-code-mode-host"]

    def allow_hosts(self):
        hosts = list(self.cfg.get("allow_hosts") or [])
        if self.auth == "subscription" and not hosts:
            hosts = list(DEFAULT_OPENAI_HOSTS)
        return hosts

    def model_upstream(self):
        if self.auth != "api":
            return None
        base = self.cfg.get("upstream_base_url")
        if not base:
            raise ValueError("codex auth=api needs upstream_base_url")
        key_env = self.cfg.get("upstream_api_key_env")
        key = os.environ.get(key_env) if key_env else None
        if key_env and not key:
            raise ValueError(f"codex auth=api: environment variable {key_env} is not set on the operator side")
        return {"base_url": base, "api_key": key, "headers": self.cfg.get("upstream_headers") or {},
                "strip_images": bool(self.cfg.get("strip_images"))}

    def _src_home(self):
        return Path(os.path.expanduser(self.cfg.get("codex_home") or os.environ.get("CODEX_HOME") or "~/.codex"))

    def describe(self):
        d = super().describe()
        d["auth"] = self.auth
        if self.auth == "api":
            d["upstream"] = self.cfg.get("upstream_base_url")
        return d

    def prepare_home(self, home_host: Path, ctx):
        ch = home_host / ".codex"
        ch.mkdir(parents=True, exist_ok=True)
        extra = ""
        if self.cfg.get("context_window"):
            extra += f"model_context_window = {int(self.cfg['context_window'])}\n"
        if self.cfg.get("compact_limit"):
            extra += f"model_auto_compact_token_limit = {int(self.cfg['compact_limit'])}\n"
        api = self.auth == "api"
        text = CONFIG_TOML.format(
            model=ctx.model or self.cfg.get("model", "gpt-5"), effort=self.cfg.get("reasoning_effort", "high"), extra=extra,
            provider_key='model_provider = "bench"\n' if api else "",
            provider_table=PROVIDER_TABLE.format(wire_api=self.cfg.get("wire_api", "responses"),
                                                 idle_ms=int(self.cfg.get("stream_idle_timeout_ms", 1800000))) if api else "")
        if not api:
            auth = self._src_home() / "auth.json"
            if not auth.exists():
                raise FileNotFoundError(f"codex auth=subscription: {auth} not found; run `codex login` on this machine first "
                                        f"(or point harnesses/<name>/config.json codex_home at a logged-in CODEX_HOME)")
            shutil.copy2(auth, ch / "auth.json")
        (ch / "config.toml").write_text(text)

    def command(self, ctx):
        cmd = ["codex", "exec", "--skip-git-repo-check", "-s", "danger-full-access", "-C", ctx.ws_inside,
               "-o", ctx.last_message_inside, "--json", "-m", ctx.model or self.cfg.get("model", "gpt-5"),
               "-c", 'approval_policy="never"']
        if self.auth == "api":
            cmd += ["-c", f'model_providers.bench.base_url="http://127.0.0.1:{ctx.port}/t/{ctx.run_id}/main/v1"']
        for c in self.cfg.get("extra_config") or []:
            cmd += ["-c", c]
        cmd.append("-")
        return cmd

    def env(self, ctx):
        # api mode talks to 127.0.0.1 directly (NO_PROXY); subscription mode goes through HTTPS_PROXY like everything else
        return {"CODEX_HOME": f"{ctx.home_inside}/.codex",
                "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}

    def after_run(self, home_host: Path, ctx):
        if self.auth != "subscription" or not self.cfg.get("auth_sync_back", True):
            return
        new = home_host / ".codex" / "auth.json"
        src = self._src_home() / "auth.json"
        try:
            if new.exists() and src.exists() and new.read_bytes() != src.read_bytes():
                shutil.copy2(src, src.with_suffix(".json.bak"))
                shutil.copy2(new, src)
        except OSError:
            pass
