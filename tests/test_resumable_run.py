import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from bench.patch import git
from bench.run import bench_run
from bench.sandbox import probe_proc_flags
from bench.util import free_cpus


class ResumableRunTests(unittest.TestCase):
    def test_pause_resume_preserves_workspace_submissions_and_budget(self):
        try:probe_proc_flags()
        except RuntimeError as exc:self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);root.chmod(0o755);task=root/'task';code=task/'agent/code'
            for rel in ('agent/code','agent/data','agent/bench','judge/bench','runtime'):
                (task/rel).mkdir(parents=True)
            (task/'agent/README.md').write_text('Resume fixture.\n')
            for name in ('train.sh','predict.sh'):(code/name).write_text('exit 0\n')
            (code/'env.lock').write_text('')
            for name in ('scorer.py','validate.py'):(task/'agent/bench'/name).write_text('print("{}")\n')
            (task/'runtime/runtime.json').write_text(json.dumps({'version':1,'mounts':[],'python':'/usr/bin/python3'}))
            identity={'GIT_AUTHOR_NAME':'test','GIT_AUTHOR_EMAIL':'test@local','GIT_COMMITTER_NAME':'test','GIT_COMMITTER_EMAIL':'test@local'}
            git(['init','-q'],code);git(['add','.'],code);git(['commit','-qm','fixture'],code,env=identity)
            head=git(['rev-parse','HEAD'],code).stdout.strip()
            (task/'task.json').write_text(json.dumps({'pid':'resume-test','metric':{'name':'score','higher_is_better':True},
                'adapt':{'base_commit':head,'adapt_commit':head},'env':{'mode':'snapshot'}}))
            rec=root/'records/run';local=root/'work/run'
            script=('set -eu\nif test -f /ws/code/phase; then\n'
                    '  test "$(cat /ws/code/phase)" = first\n'
                    '  printf second > /ws/code/phase\n  bench submit\n'
                    'else\n  printf first > /ws/code/phase\n  bench submit --final\n  sleep 60\nfi\n')
            cfg={'script_text':script,'resumable':True,'record_solution_versions':True,'submission_selection':'last_valid'}
            def pause():
                deadline=time.monotonic()+20
                while time.monotonic()<deadline:
                    if (rec/'submissions/001.json').exists():
                        (rec/'PAUSE.json').write_text('{"kind":"test_quota_pause"}')
                        return
                    time.sleep(.1)
            worker=threading.Thread(target=pause);worker.start()
            with patch('bench.run.RUNS_DIR',root/'records'),patch('bench.runtime.LOCAL_ROOT',root/'runtime'):
                args=dict(cpus=1,cpuset=str(free_cpus()[0]),run_id='run',local_root=root/'work',overrides=cfg)
                first=bench_run(str(task),'scripted',.02,**args)
                worker.join(timeout=1)
                self.assertEqual(first['status'],'paused')
                self.assertFalse((rec/'final.patch').exists())
                self.assertTrue((rec/'checkpoint/workspace/code/phase').exists())
                self.assertTrue((local/'venv').is_dir())
                with self.assertRaises(ValueError):bench_run(str(task),'scripted',.03,resume=True,**args)
                time.sleep(1)
                second=bench_run(str(task),'scripted',.02,resume=True,**args)
            self.assertEqual(second['status'],'finished')
            self.assertEqual(len(second['attempts']),2)
            self.assertGreater(second['wall_elapsed_s'],second['active_elapsed_s'])
            self.assertLess(second['remaining_s'],first['remaining_s'])
            self.assertAlmostEqual(second['active_elapsed_s']+second['remaining_s'],72)
            final=json.loads((rec/'final.json').read_text())
            self.assertEqual(final['selected_submission_n'],2)
            self.assertEqual(len(list((rec/'submissions').glob('*.patch'))),2)
            self.assertIn(b'second',(rec/'final.patch').read_bytes())


if __name__=='__main__':unittest.main()
