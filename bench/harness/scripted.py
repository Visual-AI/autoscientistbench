"""A test double: runs a shell script inside the sandbox instead of an LLM agent (no model calls).
Used by the end-to-end tests and to probe the closure (ls /mnt, curl example.com, ...)."""
import shutil
from pathlib import Path

from . import register
from .base import Harness


@register("scripted")
class Scripted(Harness):
    type_name = "scripted"

    def prepare_home(self, home_host: Path, ctx):
        script = self.cfg.get("script")
        text = self.cfg.get("script_text")
        if not script and not text:
            raise ValueError("scripted harness needs cfg.script (host path) or cfg.script_text")
        (home_host / "script.sh").write_text(text if text else Path(script).read_text())

    def command(self, ctx):
        return ["bash", f"{ctx.home_inside}/script.sh"]

    def stdin(self, ctx):
        return None

    def collect(self, rec_dir: Path):
        log = rec_dir / "agent_events.jsonl"
        n = sum(1 for _ in open(log, encoding="utf-8", errors="replace")) if log.exists() else 0
        return {"events": n, "turns": 0, "tokens_in": 0, "tokens_out": 0, "commands": 0, "errors": 0}
