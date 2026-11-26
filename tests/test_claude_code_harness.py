import json
from pathlib import Path
import tempfile
import unittest
from bench.harness.claude_code import ClaudeCode
from bench.harness.base import RunContext
from bench.claude_archive import snapshot
from bench.proxy import Proxy,ProxyConfig

class ClaudeCodeTests(unittest.TestCase):
 def test_proxy_resume_never_overwrites_prior_trace(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);a=Proxy(ProxyConfig(trace_dir=root));a._record('run','main',{'attempt':1})
   b=Proxy(ProxyConfig(trace_dir=root));b._record('run','main',{'attempt':2})
   self.assertEqual(json.loads((root/'run/main/req_0001.json').read_text()),{'attempt':1})
   self.assertEqual(json.loads((root/'run/main/req_0002.json').read_text()),{'attempt':2})
 def test_native_archive_excludes_credentials_and_outside_symlinks(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);home=root/'home';p=home/'.claude/projects/ws';p.mkdir(parents=True)
   (home/'.claude/.credentials.json').write_text('private');(root/'outside.jsonl').write_text('private')
   (p/'real.jsonl').write_text('{"event":"saved"}\n');(p/'link.jsonl').symlink_to(root/'outside.jsonl')
   self.assertEqual(snapshot(home,root/'record'),1)
   self.assertEqual(len(list((root/'record').rglob('*.jsonl'))),1)
   self.assertFalse(any('private' in x.read_text() for x in (root/'record').rglob('*') if x.is_file()))
 def test_resume_uses_original_session_and_accounting_deduplicates_stream(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);home=root/'home';home.mkdir();h=ClaudeCode('test',{'model':'Qwen3.8-27B'},root)
   ctx=RunContext(run_id='r',pid='t',budget_s=100,deadline_ts=None,cpus=1,port=9,ws_inside='/ws',home_inside='/home/agent',run_dir_inside='/run/arb',last_message_inside='/tmp/last',prompt_text='task',model='Qwen3.8-27B')
   h.prepare_home(home,ctx);session=h.session;ctx.resuming=True
   with self.assertRaises(RuntimeError):h.prepare_home(home,ctx)
   native=home/'.claude/projects/ws';native.mkdir(parents=True);(native/(session+'.jsonl')).write_text('{}\n')
   h.prepare_home(home,ctx);cmd=h.command(ctx);self.assertEqual(cmd[cmd.index('--resume')+1],session)
   self.assertEqual(h.env(ctx)['CLAUDE_CODE_MAX_CONTEXT_TOKENS'],'262144')
   msg={'type':'assistant','session_id':session,'message':{'id':'m','model':'Qwen3.8-27B','content':[{'type':'tool_use','name':'Bash'}],'usage':{'input_tokens':10,'output_tokens':4,'cache_read_input_tokens':2}}}
   record=root/'record';record.mkdir();(record/'agent_events.jsonl').write_text('\n'.join(json.dumps(e) for e in [msg,msg,{'type':'result','usage':{'input_tokens':9999}}])+'\n')
   result=h.collect(record);self.assertEqual(result['tokens_in'],12);self.assertEqual(result['tokens_out'],4);self.assertEqual(result['commands'],1)

if __name__=='__main__':unittest.main()
