"""Antigravity CLI with an isolated login, native conversations and exact model selection."""
import json
import os
import shutil
from pathlib import Path

from . import register
from .base import Harness, read_jsonl

STATE = '.gemini/antigravity-cli'
DISABLED_TOOLS = ['search_web', 'read_url_content', 'generate_image', 'schedule',
    'send_message', 'search_marketplace', 'list_plugin_accounts', 'call_mcp_tool',
    'list_resources', 'read_resource', 'run_workflow', 'define_subagent',
    'invoke_subagent', 'manage_subagents', 'manage_inbox', 'delete_knowledge',
    'browser_click_element', 'browser_drag_pixel_to_pixel', 'browser_get_dom',
    'browser_get_network_request', 'browser_input', 'browser_list_network_requests',
    'browser_mouse_down', 'browser_mouse_up', 'browser_move_mouse', 'browser_press_key',
    'browser_refresh_page', 'browser_resize_window', 'browser_scroll', 'browser_scroll_dom',
    'browser_select_option', 'browser_subagent', 'capture_browser_console_logs',
    'capture_browser_screenshot', 'click_browser_pixel', 'execute_browser_javascript',
    'list_browser_pages', 'open_browser_url', 'read_browser_page']


@register('agy')
class Agy(Harness):
    type_name = 'agy'

    def tools(self):
        return ['agy', 'rg']

    def allow_hosts(self):
        return list(self.cfg.get('allow_hosts') or [
            'oauth2.googleapis.com', 'accounts.google.com', 'www.googleapis.com',
            'daily-cloudcode-pa.googleapis.com', 'cloudcode-pa.googleapis.com',
            'aicode.googleapis.com', 'businessaicode.googleapis.com'])

    def prepare_home(self, home_host, ctx):
        self._home = Path(home_host)
        source = Path(self.cfg.get('agy_home') or '~/.gemini/antigravity-cli').expanduser()
        dest = self._home / STATE
        dest.mkdir(parents=True, exist_ok=True, mode=0o700)
        credential = source / 'antigravity-oauth-token'
        if not credential.is_file():
            raise RuntimeError('Agy login missing; authenticate with agy first')
        # Never copy operator history, memories, project settings, hooks or plugins.
        token = dest / credential.name
        if not token.exists() or not getattr(ctx, 'resuming', False):
            shutil.copyfile(credential, token)
            token.chmod(0o600)
        (dest / 'settings.json').write_text(json.dumps({'trustedWorkspaces': [ctx.ws_inside],
            'artifactReviewPolicy': 'always-proceed', 'allowNonWorkspaceAccess': True,
            'permissions': {'deny': ['read_url(*)', 'execute_url(*)', 'mcp(*)']}})+'\n')
        config = self._home / '.gemini/config'
        config.mkdir(parents=True, exist_ok=True)
        (config / 'config.json').write_text(json.dumps({'plugins': {'swarm': {'enabled': False}}})+'\n')
        (config / 'mcp_config.json').write_text('{}\n')
        # Native CLI hooks enforce the existing benchmark's offline tool policy.
        # Tool names may remain advertised; verify the actual denial in the gate.
        (config / 'hooks.json').write_text(json.dumps({'benchmark-tool-policy': {'PreToolUse': [{
            'matcher': '^('+'|'.join(DISABLED_TOOLS)+')$',
            'hooks': [{'type':'command','command':'/usr/bin/python3 -I -S /opt/arb/src/bench/harness/agy_policy.py',
                       'timeout':10}]}]}})+'\n')

    def command(self, ctx):
        model = ctx.model or self.cfg.get('model', 'gemini-4-neon-xhigh')
        cmd = ['agy', '--model', model, '--effort', self.cfg.get('reasoning_effort', 'xhigh'),
               '--dangerously-skip-permissions', '--disable-slash-commands',
               '--output-format', 'stream-json', '--print-timeout', '0',
               '--log-file', f'{ctx.home_inside}/{STATE}/cli.log']
        if getattr(ctx, 'resuming', False):
            sessions = list((self._home / STATE / 'conversations').glob('*.db'))
            if len(sessions) != 1:
                raise RuntimeError('Expected exactly one saved Agy conversation; refusing an empty restart')
            cmd += ['--conversation', sessions[0].stem]
        return cmd + ['--print', ctx.prompt_text]

    def env(self, ctx):
        return {'AGY_CLI_DISABLE_AUTO_UPDATE': '1', 'ANTIGRAVITY_BROWSER_TOOLS_ENABLED': 'false',
                'NO_PROXY': '127.0.0.1,localhost', 'no_proxy': '127.0.0.1,localhost'}

    def stdin(self, ctx):
        return ''

    def begin_attempt(self, rec_dir):
        path=Path(rec_dir)/'agent_events.jsonl'
        self._event_offset=path.stat().st_size if path.exists() else 0

    def pause_reason(self, rec_dir, since):
        path=Path(rec_dir)/'agent_events.jsonl'
        if not path.exists():return None
        with path.open('rb') as stream:
            stream.seek(getattr(self, '_event_offset', 0))
            while True:
                pos=stream.tell();line=stream.readline()
                if not line or not line.endswith(b'\n'):
                    self._event_offset=pos;break
                self._event_offset=stream.tell()
                try:event=json.loads(line)
                except ValueError:continue
                if event.get('event')=='result' and event.get('result',{}).get('status')=='ERROR':
                    error=event['result'].get('error','Unknown provider error')
                    return {'kind':'provider_error','message':error[:1200]}
        return None

    def collect(self, rec_dir):
        events = read_jsonl(Path(rec_dir)/'agent_events.jsonl')
        kinds, steps, models, commands = {}, {}, set(), set()
        errors=0;turns=0
        for event in events:
            kind = event.get('event', '?')
            kinds[kind] = kinds.get(kind, 0)+1
            if kind=='init':models.add(event.get('init',{}).get('model'))
            step=event.get('step_update',{})
            if kind=='step_update' and step.get('state')=='DONE':
                key=(step.get('conversation_id'),step.get('step_index'))
                if step.get('usage'):steps[key]=step['usage']
                if step.get('tool_name')=='run_command':commands.add(key)
            if kind=='result':
                result=event.get('result',{});turns+=1
                if result.get('status')=='ERROR':errors+=1
                if result.get('response'):
                    (Path(rec_dir)/'last_message.txt').write_text(result['response'])
        usage={}
        for item in steps.values():
            for key,value in item.items():
                if isinstance(value,(int,float)):usage[key]=usage.get(key,0)+value
        return {'events':len(events),'event_types':kinds,'usage':usage,'model_calls':len(steps),
            'models':sorted(m for m in models if m),'commands':len(commands),'errors':errors,'turns':turns,
            'tokens_in':usage.get('input_tokens',0)+usage.get('cache_read_tokens',0) if steps else None,
            'tokens_cached':usage.get('cache_read_tokens') if steps else None,
            'tokens_out':usage.get('output_tokens') if steps else None,
            'tokens_reasoning':usage.get('thinking_tokens') if steps else None,
            'usage_source':'unique_completed_step_usage' if steps else 'missing'}
