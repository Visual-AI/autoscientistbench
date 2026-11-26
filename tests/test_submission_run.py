import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from bench.patch import git
from bench.run import bench_run
from bench.sandbox import probe_proc_flags
from bench.util import free_cpus

class SubmissionRunTests(unittest.TestCase):
    def test_real_run_requires_explicit_submission(self):
        try:
            probe_proc_flags()
        except RuntimeError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); task = root / 'task'
            root.chmod(0o755)   # the sandbox runs as a uid of its own (sandbox_user.py) and must reach its binds
            for rel in ('agent/code', 'agent/data', 'agent/bench', 'judge/bench', 'runtime'):
                (task / rel).mkdir(parents=True)
            code = task / 'agent/code'
            (task / 'agent/README.md').write_text('A tiny submission fixture.\n')
            (code / 'env.lock').write_text('')
            (code / 'train.sh').write_text('exit 0\n')
            (code / 'predict.sh').write_text('exit 0\n')
            for name in ('scorer.py', 'validate.py'):
                (task / 'agent/bench' / name).write_text('print("{}")\n')
            (task / 'runtime/runtime.json').write_text(json.dumps({'version': 1, 'mounts': [], 'python': '/usr/bin/python3'}))
            identity = {'GIT_AUTHOR_NAME': 'test', 'GIT_AUTHOR_EMAIL': 'test@local', 'GIT_COMMITTER_NAME': 'test', 'GIT_COMMITTER_EMAIL': 'test@local'}
            git(['init', '-q'], code); git(['add', '.'], code)
            git(['commit', '-qm', 'fixture'], code, env=identity)
            head = git(['rev-parse', 'HEAD'], code).stdout.strip()
            (task / 'task.json').write_text(json.dumps({'pid': 'submission-test', 'metric': {'name': 'score', 'higher_is_better': True},
                'adapt': {'base_commit': head, 'adapt_commit': head}, 'env': {'mode': 'snapshot'}}))
            for explicit in (False, True):
                name = 'explicit' if explicit else 'missing'
                script = ('set -eu\nprintf changed > /ws/code/changed.txt\n'
                          'test -n "$BENCH_DEADLINE_TS"\n'
                          'printf "%s" "$BENCH_DEADLINE_TS" > /ws/code/deadline.txt\n'
                          'test -r "$BENCH_RUN_DIR/state.json"\n'
                          'if printf fake > "$BENCH_RUN_DIR/state.json"; then exit 88; fi\n'
                          'if printf fake > "$BENCH_RUN_DIR/submissions/999.json"; then exit 89; fi\n')
                if explicit: script += 'bench submit --final\n'
                with patch('bench.run.RUNS_DIR', root / 'records'), patch('bench.runtime.LOCAL_ROOT', root / 'runtime'), \
                     patch.dict('os.environ', {'ARB_DURABLE_RUNS_DIR':str(root/'durable')}):
                    result = bench_run(str(task), 'scripted', .003, cpus=1, cpuset=str(free_cpus()[0]),
                                       local_root=root / 'work', run_id=name, overrides={'script_text': script,
                                           'record_solution_versions':True, 'submission_selection':'last_valid'},
                                       judge_budget_hours=.005)
                rec = root / 'records' / name
                final = json.loads((rec / 'final.json').read_text())
                self.assertEqual(result['solution_recording']['status'], 'complete')
                self.assertEqual(final['selection_policy'], 'last_valid')
                self.assertTrue((rec/'submissions').is_symlink())
                self.assertTrue((root/'durable'/name/'solutions/manifest.json').is_file())
                self.assertEqual(result['judge_budget_s'], 18)
                argv = json.loads((rec / 'sandbox_argv.json').read_text())
                self.assertIn('BENCH_DEADLINE_TS', argv)
                self.assertEqual(float(argv[argv.index('BENCH_DEADLINE_TS') + 1]), result['deadline_ts'])
                if explicit:
                    self.assertEqual(result['status'], 'finished')
                    self.assertTrue(final['ok'])
                    self.assertEqual(final['source'], 'final')
                    self.assertIn(b'changed.txt', (rec / 'final.patch').read_bytes())
                else:
                    self.assertEqual(result['status'], 'no_submission')
                    self.assertFalse(final['ok'])
                    self.assertEqual((rec / 'final.patch').read_bytes(), b'')
