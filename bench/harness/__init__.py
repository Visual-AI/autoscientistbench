"""Adapters for the harnesses under test. One adapter = how to launch it inside our sandbox and how to read its log."""
import json
from pathlib import Path

from .. import HARNESSES_DIR

TYPES = {}


def register(name):
    def deco(cls):
        TYPES[name] = cls
        return cls
    return deco


def load_harness(name, overrides=None):
    from . import codex, traecli, scripted, zcode, kimi, agy, claude_code  # noqa: F401  (registers the types)
    cfg_path = HARNESSES_DIR / name / "config.json"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    typ = cfg.get("type", name)
    if typ not in TYPES:
        raise KeyError(f"unknown harness type {typ!r} (have {sorted(TYPES)}); add harnesses/{name}/config.json")
    cfg = dict(cfg)
    cfg.update(overrides or {})
    return TYPES[typ](name, cfg, HARNESSES_DIR / name)
