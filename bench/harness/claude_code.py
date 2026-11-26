"""Claude Code with an isolated HOME and the recorded Anthropic model route."""
import json
from pathlib import Path
import subprocess
import sys
import threading
import uuid

from . import register
from .base import Harness, read_jsonl


@register('claude-code')
class ClaudeCode(Harness):
    type_name = 'claude-code'

    def tools(self):
        return ['claude', 'rg']

    def model_upstream(self):
        import os
        base = self.cfg.get('upstream_base_url')
        if not base:
            raise ValueError('Claude Code requires an explicit upstream_base_url')
        key_env = self.cfg.get('upstream_api_key_env')
        if key_env:
            key = os.environ.get(key_env)
            if not key:
                raise ValueError(f'Claude Code: environment variable {key_env} is not set on the operator side')
            return {'base_url': base, 'headers': {'x-api-key': key}}
        return {'base_url': base, 'headers': {'x-api-key': 'local-endpoint'}}

    def prepare_home(self, home_host, ctx):
        self.home = Path(home_host)
        state = self.home / '.claude'
        state.mkdir(parents=True, exist_ok=True)
        session_file = self.home / '.benchmark_claude_session'
        if getattr(ctx, 'resuming', False):
            self.session = str(uuid.UUID(session_file.read_text().strip()))
            if not list((state/'projects').rglob(self.session+'.jsonl')):
                raise RuntimeError('Saved Claude Code conversation is missing; cannot resume')
        else:
            self.session = str(uuid.uuid4())
            session_file.write_text(self.session+'\n')
        # No operator credentials, memories, plugins or project hooks are copied.
        (state/'settings.json').write_text(json.dumps({
            'effortLevel': self.cfg.get('reasoning_effort', 'xhigh'),
            'autoCompactWindow': self.cfg.get('auto_compact_window', 200000),
            'autoMemoryEnabled': False, 'permissions': {'defaultMode': 'bypassPermissions'},
        })+'\n')

    def command(self, ctx):
        cmd = ['claude', '--print', '--verbose', '--output-format', 'stream-json',
            '--include-partial-messages', '--dangerously-skip-permissions',
            '--disable-slash-commands', '--setting-sources', 'user',
            '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
            '--tools', 'Bash,Read,Write,Edit,Grep,Glob,NotebookEdit,TaskOutput,TaskStop',
            '--model', ctx.model or self.cfg['model'],
            '--effort', self.cfg.get('reasoning_effort', 'xhigh')]
        cmd += ['--resume' if getattr(ctx, 'resuming', False) else '--session-id', self.session]
        return cmd

    def env(self, ctx):
        model = ctx.model or self.cfg['model']
        capabilities = 'effort,xhigh_effort,thinking,adaptive_thinking,interleaved_thinking'
        env = {
            'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{ctx.port}/t/{ctx.run_id}/main',
            'ANTHROPIC_API_KEY': 'local-endpoint', 'ANTHROPIC_MODEL': model,
            'CLAUDE_CONFIG_DIR': f'{ctx.home_inside}/.claude',
            'CLAUDE_CODE_EFFORT_LEVEL': self.cfg.get('reasoning_effort', 'xhigh'),
            'CLAUDE_CODE_MAX_CONTEXT_TOKENS': str(self.cfg.get('context_window', 262144)),
            'CLAUDE_CODE_AUTO_COMPACT_WINDOW': str(self.cfg.get('auto_compact_window', 200000)),
            'CLAUDE_CODE_MAX_OUTPUT_TOKENS': str(self.cfg.get('max_output_tokens', 32768)),
            'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
            'CLAUDE_CODE_ATTRIBUTION_HEADER': '0', 'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1',
            'DISABLE_AUTOUPDATER': '1', 'DISABLE_TELEMETRY': '1', 'DISABLE_ERROR_REPORTING': '1',
            'DISABLE_PROMPT_CACHING': '1', 'API_TIMEOUT_MS': '1800000',
            'NO_PROXY': '127.0.0.1,localhost', 'no_proxy': '127.0.0.1,localhost',
        }
        for alias in ['ANTHROPIC_DEFAULT_OPUS_MODEL', 'ANTHROPIC_DEFAULT_SONNET_MODEL',
                      'ANTHROPIC_DEFAULT_HAIKU_MODEL', 'ANTHROPIC_DEFAULT_FABLE_MODEL',
                      'ANTHROPIC_CUSTOM_MODEL_OPTION']:
            env[alias] = model
            env[alias+'_SUPPORTED_CAPABILITIES'] = capabilities
        env['CLAUDE_CODE_SUBAGENT_MODEL'] = model
        return env

    def _snapshot(self):
        helper = Path(__file__).resolve().parents[1]/'claude_archive.py'
        subprocess.run(['sudo', '-n', sys.executable, '-I', '-S', str(helper),
            str(self.home), str(self.record)], check=True, capture_output=True, timeout=90)

    def begin_attempt(self, rec_dir):
        self.record = Path(rec_dir)
        events = self.record/'agent_events.jsonl'
        self.offset = events.stat().st_size if events.exists() else 0
        self.archive_errors = []
        self.stop_archive = threading.Event()
        def mirror():
            while not self.stop_archive.wait(30):
                try: self._snapshot()
                except Exception as exc: self.archive_errors.append(str(exc)[:500])
        self.archive_thread = threading.Thread(target=mirror, daemon=True)
        self.archive_thread.start()

    def after_run(self, home_host, ctx):
        self.stop_archive.set()
        self.archive_thread.join(95)
        self._snapshot()
        (self.record/'native_archive_status.json').write_text(json.dumps({
            'final_snapshot': 'complete', 'transient_errors': self.archive_errors})+'\n')

    def pause_reason(self, rec_dir, since):
        path = Path(rec_dir)/'agent_events.jsonl'
        if not path.exists(): return None
        with path.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                position = stream.tell(); line = stream.readline()
                if not line or not line.endswith(b'\n'):
                    self.offset = position; break
                self.offset = stream.tell()
                try: event = json.loads(line)
                except ValueError: continue
                if event.get('type') == 'result' and event.get('is_error'):
                    return {'kind': 'provider_error', 'message': str(event.get('errors') or event.get('result') or event.get('subtype'))[:1500]}
        return None

    def collect(self, rec_dir):
        events = read_jsonl(Path(rec_dir)/'agent_events.jsonl')
        messages = {}; models = set(); compactions = 0
        for event in events:
            if event.get('type') == 'assistant':
                message = event.get('message') or {}
                if message.get('id'):
                    messages[(event.get('session_id'), message['id'])] = message
                if message.get('model'): models.add(message['model'])
            if event.get('type') == 'system' and event.get('subtype') == 'compact_boundary':
                compactions += 1
        usage = {}; commands = 0
        for message in messages.values():
            for key, value in (message.get('usage') or {}).items():
                if isinstance(value, (float, int)): usage[key] = usage.get(key, 0)+value
            commands += sum(block.get('type') == 'tool_use' for block in message.get('content', []))
        return {'events': len(events), 'model_calls': len(messages), 'commands': commands,
            'models': sorted(models), 'usage': usage, 'compactions': compactions,
            'tokens_in': sum(usage.get(k, 0) for k in ['input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens']),
            'tokens_out': usage.get('output_tokens', 0), 'tokens_cached': usage.get('cache_read_input_tokens', 0),
            'errors': sum(e.get('type') == 'result' and bool(e.get('is_error')) for e in events),
            'usage_source': 'unique_assistant_messages'}
