"""Kimi Code's native headless agent, using an isolated copy of the local login.

The JSON stdout stream omits thinking and token usage. Preserve the native session
wire log as well, and count only usage.record deltas (not repeated state snapshots).
No operator history, hooks, MCP servers or skills are copied into a benchmark run.
"""
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from . import register
from .base import Harness, read_jsonl

PKG_IN_HOME = ".kimi-code-pkg"
STATE_IN_HOME = ".kimi-code"
CODING_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep",
                "TaskList", "TaskOutput", "TaskStop", "WaitFor"]


def _read_config(path):
    # Lazy import keeps the other harnesses usable on Python 3.9/3.10.
    try:
        import tomllib
    except ImportError as exc:
        raise RuntimeError("The kimi harness requires Python >= 3.11 to read config.toml") from exc
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def _toml(value):
    """The selected provider/model are ordinary JSON values; inline TOML needs no dependency."""
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{json.dumps(k, ensure_ascii=False)} = {_toml(v)}"
                                for k, v in value.items() if v is not None) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_toml(v) for v in value) + "]"
    if isinstance(value, (str, bool, int, float)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False)
    raise ValueError(f"Unsupported Kimi config value: {type(value).__name__}")


def _write_config(path, config):
    path.write_text("\n".join(f"{json.dumps(k)} = {_toml(v)}" for k, v in config.items()) + "\n",
                    encoding="utf-8")
    path.chmod(0o600)


def _private_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    shutil.copyfile(src, dst)
    dst.chmod(0o600)


def _event(row):
    # The native wire has an envelope; tolerate direct records in older versions.
    for key in ("event", "payload", "message"):
        nested = row.get(key)
        if isinstance(nested, dict) and "type" in nested:
            return nested
    return row


@register("kimi")
class Kimi(Harness):
    type_name = "kimi"

    def _source_home(self):
        return Path(self.cfg.get("kimi_home") or os.environ.get("KIMI_CODE_HOME") or
                    "~/.kimi-code").expanduser()

    def _selection(self, model=None):
        source = self._source_home() / "config.toml"
        if not source.is_file():
            raise ValueError(f"Kimi config missing: {source}; log in with kimi first")
        base = _read_config(source)
        alias = model or self.cfg.get("model") or base.get("default_model")
        models = base.get("models") or {}
        if alias not in models:
            raise ValueError(f"Unknown Kimi model {alias!r}; available aliases: {', '.join(models)}")
        entry = models[alias]
        provider_name = entry["provider"]
        provider = (base.get("providers") or {}).get(provider_name)
        if not provider or provider.get("type") != "kimi":
            raise ValueError(f"Kimi model {alias!r} needs a configured Kimi provider")
        efforts = entry.get("support_efforts") or []
        effort = self.cfg.get("reasoning_effort")
        if effort is not None and effort not in efforts:
            raise ValueError(f"Model {alias!r} does not advertise effort {effort!r}; supported: {efforts}")
        if effort is None and efforts:
            preferred = (base.get("thinking") or {}).get("effort")
            effort = preferred if preferred in efforts else entry.get("default_effort")
        return alias, entry, provider_name, provider, effort

    def _package(self):
        src = self.cfg.get("package_dir") or os.environ.get("ARB_KIMI_PKG")
        if not src:
            binary = shutil.which("kimi")
            if not binary:
                raise ValueError("kimi not on PATH; set package_dir or ARB_KIMI_PKG")
            src = Path(binary).resolve().parents[1]
        src = Path(src).expanduser()
        if not (src / "dist/main.mjs").is_file():
            raise ValueError(f"Kimi Code npm entrypoint missing: {src}/dist/main.mjs")
        return src

    def tools(self):
        return ["node"]

    def allow_hosts(self):
        # Compute before prepare_home: bench creates its proxy before the run context.
        hosts = {"api.kimi.com", "auth.kimi.com", "code.kimi.com"}
        base = _read_config(self._source_home() / "config.toml")
        for provider in (base.get("providers") or {}).values():
            if provider.get("type") == "kimi":
                host = urlparse(provider.get("base_url", "")).hostname
                if host:
                    hosts.add(host)
        return sorted(hosts | set(self.cfg.get("allow_hosts") or []))

    def prepare_home(self, home_host, ctx):
        alias, entry, provider_name, provider, effort = self._selection(ctx.model)
        self._home_host = Path(home_host)
        source_home = self._source_home()
        home = self._home_host / STATE_IN_HOME
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        (home / "empty-skills").mkdir(exist_ok=True)

        # Explicitly construct this config. Copying an operator config would enable
        # unrelated providers, remote tools, personal instructions and startup hooks.
        thinking = {"enabled": True}
        if effort is not None:
            thinking["effort"] = effort
        config = {"default_model": alias, "providers": {provider_name: provider},
                  "models": {alias: entry}, "thinking": thinking,
                  "telemetry": False,
                  "default_plan_mode": False, "merge_all_available_skills": False,
                  "tools": {"enabled": CODING_TOOLS}, "workspace": {"require_trust": False}}
        _write_config(home / "config.toml", config)
        _write_config(home / "tui.toml", {"auto_session_title": False,
                                         "upgrade": {"auto_install": False}})

        self._credential_source = None
        self._auth_sync = "not_applicable"
        oauth = provider.get("oauth") or {}
        if oauth:
            # Kimi Code's built-in file OAuth store maps oauth/kimi-code to this file.
            if oauth != {"storage": "file", "key": "oauth/kimi-code"}:
                raise ValueError("kimi harness currently supports the built-in file OAuth login only")
            credential = source_home / "credentials/kimi-code.json"
            if not credential.is_file():
                raise ValueError("Kimi OAuth login is missing; log in with kimi first")
            raw = credential.read_bytes()
            if not json.loads(raw).get("refresh_token"):
                raise ValueError("Kimi OAuth login has no refresh token; log in with kimi again")
            self._credential_source = credential
            self._credential_before = hashlib.sha256(raw).digest()
            dst = home / "credentials/kimi-code.json"
            dst.parent.mkdir(mode=0o700, exist_ok=True)
            dst.write_bytes(raw)
            dst.chmod(0o600)
            self._auth_sync = "unchanged"
        elif not provider.get("api_key"):
            raise ValueError("Selected Kimi provider has neither OAuth login nor an API key")
        for name in ("device_id", "region"):
            if (source_home / name).is_file():
                _private_copy(source_home / name, home / name)

        package = self._package()
        target = self._home_host / PKG_IN_HOME
        target.mkdir(exist_ok=True)
        # rg on this pod lives below the operator's private Codex home. A direct
        # bwrap bind cannot traverse that directory as the sandbox uid. Stage only
        # the executable in the run HOME instead of exposing the operator home.
        rg = self.cfg.get("rg_path") or os.environ.get("ARB_TOOL_RG") or shutil.which("rg")
        if not rg:
            raise ValueError("kimi harness needs rg on PATH (or set ARB_TOOL_RG)")
        (target / "bin").mkdir(exist_ok=True)
        shutil.copyfile(Path(rg).expanduser().resolve(), target / "bin/rg")
        (target / "bin/rg").chmod(0o755)
        # Desktop assets and other OS binaries are not needed by the headless agent.
        for rel in ("dist", "node_modules", "native/linux"):
            if (package / rel).is_dir():
                shutil.copytree(package / rel, target / rel, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("*.map"))
        shutil.copy2(package / "package.json", target / "package.json")
        self._resolved = {"model": alias, "backend_model": entry["model"],
                          "display_name": entry.get("display_name"), "reasoning_effort": effort,
                          "max_context_size": entry.get("max_context_size"),
                          "cli_version": json.loads((package / "package.json").read_text())["version"]}

    def command(self, ctx):
        alias = self._selection(ctx.model)[0]
        resume_args = []
        if getattr(ctx, 'resuming', False):
            sessions = self.sessions(self._home_host / STATE_IN_HOME / 'sessions')
            if not sessions:
                raise RuntimeError('No saved Kimi session found; refusing to restart the task with an empty context')
            resume_args = ['--session', sessions[-1]['id']]
        return ["node", "--use-env-proxy", f"{ctx.home_inside}/{PKG_IN_HOME}/dist/main.mjs", *resume_args,
                "--model", alias, "--skills-dir", f"{ctx.home_inside}/{STATE_IN_HOME}/empty-skills",
                "--output-format", "stream-json", "--prompt", ctx.prompt_text]

    @staticmethod
    def sessions(root):
        result = []
        for path in Path(root).glob('*/*/state.json'):
            try:
                row = json.loads(path.read_text())
                if row.get('id') == path.parent.name and 'main' in row.get('agents', {}):
                    result.append({k:row.get(k) for k in ('id','cwd','createdAt','updatedAt','lastTurnReason')})
            except (OSError, ValueError):
                continue
        return sorted(result, key=lambda r:r.get('updatedAt') or 0)

    def pause_reason(self, rec_dir, since):
        """Only provider error events trigger quota pauses, never ordinary agent text."""
        paths = [Path(rec_dir)/'agent_events.jsonl']
        # During execution HOME belongs to the sandbox uid. The trusted recorder
        # mirrors native events into the host-owned archive for live inspection.
        paths += list((Path(rec_dir)/'trace/kimi/sessions').rglob('wire.jsonl'))
        offsets = getattr(self, '_pause_offsets', {})
        self._pause_offsets = offsets
        for path in paths:
            if not path.is_file():continue
            with path.open('rb') as stream:
                stream.seek(offsets.get(str(path), 0))
                while True:
                    position = stream.tell();line = stream.readline()
                    if not line or not line.endswith(b'\n'):
                        offsets[str(path)] = position
                        break
                    offsets[str(path)] = stream.tell()
                    try:row = _event(json.loads(line))
                    except ValueError:continue
                    stamp = row.get('time', row.get('timestamp'))
                    if isinstance(stamp, (int,float)) and stamp < since*1000:continue
                    kind = str(row.get('type',''))
                    if not (kind in ('turn.ended','turn.step.retrying','error','turn.failed') or
                            'failed' in kind or 'error' in kind):continue
                    error = row.get('error') or row
                    if not isinstance(error, dict):continue
                    status = error.get('statusCode', error.get('status'))
                    name = error.get('errorName', error.get('name', error.get('kind','')))
                    message = str(error.get('errorMessage', error.get('message','')))
                    if status == 429 or name in ('quota_exhausted','rate_limit','APIProviderQuotaExhaustedError','APIProviderRateLimitError'):
                        return {'kind':'provider_quota_or_rate_limit','error_type':name,'status_code':status,
                                'message':message[:1000], 'retry_after_ms':error.get('retryAfterMs',error.get('delayMs'))}
        return None

    def begin_attempt(self, rec_dir):
        paths = [Path(rec_dir)/'agent_events.jsonl']
        paths += list((Path(rec_dir)/'trace/kimi/sessions').rglob('wire.jsonl'))
        self._pause_offsets = {str(p):p.stat().st_size for p in paths if p.is_file()}

    def stdin(self, ctx):
        # Headless prompt is an argument; an unread stdin pipe can deadlock large prompts.
        return ""

    def env(self, ctx):
        path = f"/venv/bin:{ctx.home_inside}/{PKG_IN_HOME}/bin:/opt/arb/bin:/usr/local/bin:/usr/bin:/bin"
        return {"KIMI_CODE_HOME": f"{ctx.home_inside}/{STATE_IN_HOME}",
                "KIMI_CODE_TRUST_WORKSPACE": "1", "CI": "1",
                "PATH": path, "BENCH_PATH": path,
                "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}

    def after_run(self, home_host, ctx):
        """Keep refreshed OAuth tokens, without overwriting a newer concurrent login."""
        src = getattr(self, "_credential_source", None)
        if src is None or not self.cfg.get("auth_sync_back", True):
            return
        candidate = Path(home_host) / STATE_IN_HOME / "credentials/kimi-code.json"
        if not candidate.is_file():
            return
        raw = candidate.read_bytes()
        if hashlib.sha256(raw).digest() == self._credential_before:
            return
        token = json.loads(raw)
        if not token.get("access_token") or not token.get("refresh_token"):
            self._auth_sync = "invalid_refresh_ignored"
            return
        # This lock coordinates benchmark writers. Compare the original bytes as well
        # so another run or an operator re-login cannot be replaced by our older copy.
        with open(src.with_suffix(".bench.lock"), "a") as lock:
            os.chmod(lock.name, 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not src.is_file() or hashlib.sha256(src.read_bytes()).digest() != self._credential_before:
                self._auth_sync = "source_changed_preserved"
                return
            old = json.loads(src.read_bytes())
            if float(token.get("expires_at", 0)) < float(old.get("expires_at", 0)):
                self._auth_sync = "older_refresh_ignored"
                return
            fd, tmp = tempfile.mkstemp(prefix=".bench-refresh-", dir=src.parent)
            try:
                with os.fdopen(fd, "wb") as out:
                    out.write(raw)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(tmp, src)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            self._auth_sync = "refreshed"

    def collect(self, rec_dir):
        rec_dir = Path(rec_dir)
        trace = rec_dir / "trace/kimi"
        trace.mkdir(parents=True, exist_ok=True)
        home = getattr(self, "_home_host", None)
        if home is not None:
            sessions = home / STATE_IN_HOME / "sessions"
            # Only native session records. In particular, never archive credentials.
            for path in sessions.rglob("*"):
                if path.is_file() and not path.is_symlink() and (
                        path.name in {"wire.jsonl", "state.json"} or path.suffix == ".log"):
                    dst = trace / "sessions" / path.relative_to(sessions)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dst)
        manifest = trace / "manifest.json"
        metadata = json.loads(manifest.read_text()) if manifest.exists() else {}
        metadata.update(getattr(self, "_resolved", {}))
        if hasattr(self, "_auth_sync"):
            metadata["auth_sync"] = self._auth_sync
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")

        stdout = read_jsonl(rec_dir / "agent_events.jsonl")
        native = [_event(row) for path in sorted((trace / "sessions").rglob("wire.jsonl"))
                  for row in read_jsonl(path)]
        records = [row for row in native if row.get("type") == "usage.record"]
        usage = Counter()
        by_model = {}
        for row in records:
            delta = {k: int(v) for k, v in (row.get("usage") or {}).items()
                     if isinstance(v, (int, float)) and not isinstance(v, bool)}
            usage.update(delta)
            by_model.setdefault(row.get("model", "unknown"), Counter()).update(delta)
        calls = [call for row in stdout if row.get("role") == "assistant"
                 for call in row.get("tool_calls", [])]
        items = Counter((call.get("function") or {}).get("name", "unknown") for call in calls)
        messages = [row["content"] for row in stdout if row.get("role") == "assistant"
                    and isinstance(row.get("content"), str) and row["content"]]
        last = messages[-1] if messages else ""
        (rec_dir / "last_message.txt").write_text(last, encoding="utf-8")
        ended = [row for row in native if row.get("type") == "turn.ended"]
        cached_known = bool(records) and all("inputCacheRead" in row.get("usage", {}) for row in records)
        return {**metadata, "sessions":self.sessions(trace/'sessions'),
                "events": len(stdout), "native_events": len(native), "turns": len(ended),
                "items": dict(items), "commands": items["Bash"], "tool_calls": len(calls),
                "errors": sum(row.get("type") in ("error", "turn.failed") for row in stdout),
                "retries": sum(row.get("type") == "turn.step.retrying" for row in stdout),
                "tokens_in": (usage["inputOther"] + usage["inputCacheRead"] + usage["inputCacheCreation"]
                              if cached_known else None),
                "tokens_out": usage["output"] if records else None,
                "tokens_cached": usage["inputCacheRead"] if cached_known else None,
                "tokens_cache_creation": usage["inputCacheCreation"] if records else None,
                "tokens_reasoning": None,  # Provider does not split reasoning from output tokens.
                "model_calls": len(records), "usage_source": "native_session_wire" if records else "missing",
                "cache_usage_complete": cached_known,
                "usage": dict(usage), "usage_by_model": {k: dict(v) for k, v in by_model.items()},
                "last_message": last[:2000], "last_turn": ended[-1] if ended else None,
                "trace": "trace/kimi"}
