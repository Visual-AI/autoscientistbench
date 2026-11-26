import json
from pathlib import Path

from ..util import read_jsonl


class RunContext:
    """What an adapter needs to build its command. All *_inside paths are as seen from the sandbox."""

    def __init__(self, **kw):
        self.run_id = kw["run_id"]
        self.pid = kw["pid"]
        self.budget_s = kw["budget_s"]
        self.deadline_ts = kw["deadline_ts"]
        self.cpus = kw["cpus"]
        self.port = kw["port"]
        self.ws_inside = kw["ws_inside"]
        self.home_inside = kw["home_inside"]
        self.run_dir_inside = kw["run_dir_inside"]
        self.prompt_text = kw["prompt_text"]
        self.last_message_inside = kw["last_message_inside"]
        self.model = kw.get("model")


class Harness:
    type_name = "base"

    def __init__(self, name, cfg, cfg_dir):
        self.name = name
        self.cfg = cfg
        self.cfg_dir = Path(cfg_dir)

    # what must be visible inside the sandbox
    def tools(self):
        return []

    def allow_hosts(self):
        return list(self.cfg.get("allow_hosts") or [])

    def model_upstream(self):
        """{"base_url", "api_key", "headers"} for the proxy's recorded model route, or None."""
        return None

    def prepare_home(self, home_host: Path, ctx: RunContext):
        pass

    def command(self, ctx: RunContext):
        raise NotImplementedError

    def env(self, ctx: RunContext):
        return {}

    def stdin(self, ctx: RunContext):
        return ctx.prompt_text

    def after_run(self, home_host: Path, ctx: RunContext):
        """Hook after the harness exited (e.g. copy refreshed login tokens back)."""
        pass

    def describe(self):
        return {"name": self.name, "type": self.type_name, "model": self.cfg.get("model")}

    # reading the log afterwards (codex-style JSONL events by default)
    def collect(self, rec_dir: Path):
        return collect_codex_events(rec_dir / "agent_events.jsonl")


def collect_codex_events(path):
    """Sum usage over turn.completed events, count item types. Works for codex and traecli (same lineage)."""
    evs = read_jsonl(path)
    usage = {}
    items = {}
    turns = 0
    commands = 0
    errors = 0
    for e in evs:
        t = e.get("type")
        if t == "turn.completed":
            turns += 1
            for k, v in (e.get("usage") or {}).items():
                if isinstance(v, (int, float)):
                    usage[k] = usage.get(k, 0) + int(v)
        elif t == "item.completed":
            it = e.get("item") or {}
            k = it.get("type") or it.get("item_type") or "?"
            items[k] = items.get(k, 0) + 1
            if k == "command_execution":
                commands += 1
        elif t in ("error", "turn.failed"):
            errors += 1
    return {"events": len(evs), "turns": turns, "items": items, "commands": commands, "errors": errors,
            "tokens_in": usage.get("input_tokens", 0), "tokens_out": usage.get("output_tokens", 0),
            "tokens_cached": usage.get("cached_input_tokens", 0), "tokens_reasoning": usage.get("reasoning_output_tokens", 0),
            "usage": usage}
