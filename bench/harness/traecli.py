"""traecli (TraeCode CLI) inside our bwrap sandbox.

Configure `api_base_url` and `allow_hosts` in `harnesses/<name>/config.json`.
The proxy allowlists the configured HTTPS hosts via CONNECT.
"""
import os
import shutil
from pathlib import Path

from . import register
from .base import Harness

TRAE_TOML = """model = "{model}"
model_provider = "trae"
{extra}
[projects."/ws"]
trust_level = "trusted"
"""


@register("traecli")
class TraeCLI(Harness):
    type_name = "traecli"

    def tools(self):
        return ["traecli", "traex-code-mode-host"]

    def allow_hosts(self):
        return list(self.cfg.get("allow_hosts") or [])

    def prepare_home(self, home_host: Path, ctx):
        src = Path(os.path.expanduser(self.cfg.get("trae_home") or "~/.trae"))
        dst = home_host / ".trae"
        (dst / "cli").mkdir(parents=True, exist_ok=True)
        extra = "".join(f"{k} = {v}\n" for k, v in (self.cfg.get("toml_extra") or {}).items())
        (dst / "traecli.toml").write_text(TRAE_TOML.format(model=ctx.model or self.cfg.get("model", "gpt-5.5"), extra=extra))
        for rel in ("cli/auth.json", "cli/models_cache.json", "installation_id", "version.json", "minimum_supported_version.json"):
            if (src / rel).exists():
                shutil.copy2(src / rel, dst / rel)
        if (src / "model-provider").exists():
            shutil.copytree(src / "model-provider", dst / "model-provider", dirs_exist_ok=True)

    def command(self, ctx):
        cmd = ["traecli", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "danger-full-access", "-C", ctx.ws_inside,
               "-o", ctx.last_message_inside, "--json", "-m", ctx.model or self.cfg.get("model", "gpt-5.5"),
               "-c", 'approval_policy="never"']
        for c in self.cfg.get("extra_config") or []:
            cmd += ["-c", c]
        cmd.append("-")
        return cmd

    def env(self, ctx):
        out = {"TRAE_SURFACE_FUNCTION": self.cfg.get("surface_function", "solo_agent"),
               "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost", **(self.cfg.get("env") or {})}
        if self.cfg.get("api_base_url"):
            out["TRAE_API_BASE_URL"] = self.cfg["api_base_url"]
        return out
