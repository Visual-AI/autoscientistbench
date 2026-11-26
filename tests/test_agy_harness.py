import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from bench.harness.agy import Agy, STATE
from bench.harness.base import RunContext

class AgyTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.source=self.root/'operator';self.source.mkdir()
        (self.source/'antigravity-oauth-token').write_text('{"token":"fake"}')
        (self.source/'history.jsonl').write_text('private operator history')
        self.home=self.root/'home';self.home.mkdir()
        self.ctx=RunContext(run_id='test',pid='toy',budget_s=60,deadline_ts=60,cpus=2,port=18080,
            ws_inside='/ws',home_inside='/home/agent',run_dir_inside='/run/arb',
            last_message_inside='/run/arb/last_message.txt',prompt_text='test',model='gemini-4-neon-xhigh')
        self.h=Agy('agy',{'agy_home':str(self.source)},self.root)

    def test_isolated_login_and_resume_preserve_refreshed_credential(self):
        self.h.prepare_home(self.home,self.ctx)
        state=self.home/STATE
        self.assertFalse((state/'history.jsonl').exists())
        token=state/'antigravity-oauth-token'
        self.assertEqual(token.stat().st_mode&0o777,0o600)
        token.write_text('refreshed token')
        self.ctx.resuming=True
        self.h.prepare_home(self.home,self.ctx)
        self.assertEqual(token.read_text(),'refreshed token')
        with self.assertRaises(RuntimeError):self.h.command(self.ctx)
        (state/'conversations').mkdir()
        (state/'conversations/uuid.db').touch()
        cmd=self.h.command(self.ctx)
        self.assertEqual(cmd[cmd.index('--conversation')+1],'uuid')
        (state/'conversations/ambiguous.db').touch()
        with self.assertRaises(RuntimeError):self.h.command(self.ctx)

    def test_usage_deduplicates_resume_replay_and_ignores_cumulative_result(self):
        rec=self.root/'record';rec.mkdir()
        step={'event':'step_update','step_update':{'conversation_id':'a','step_index':1,
            'state':'DONE','usage':{'input_tokens':100,'cache_read_tokens':30,'output_tokens':20,'thinking_tokens':12}}}
        rows=[{'event':'init','init':{'model':'gemini-4-neon-xhigh'}},step,step,
            {'event':'result','result':{'status':'SUCCESS','usage':{'input_tokens':9999},'response':'done'}}]
        (rec/'agent_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        result=self.h.collect(rec)
        self.assertEqual(result['tokens_in'],130)
        self.assertEqual(result['tokens_out'],20)
        self.assertEqual(result['tokens_reasoning'],12)
        self.assertEqual(result['model_calls'],1)
        self.h.begin_attempt(rec)
        with (rec/'agent_events.jsonl').open('a') as out:
            out.write(json.dumps({'event':'result','result':{'status':'ERROR','error':'quota exceeded'}})+'\n')
        self.assertEqual(self.h.pause_reason(rec,0)['kind'],'provider_error')
        self.assertIsNone(self.h.pause_reason(rec,0))

if __name__=='__main__':unittest.main()
