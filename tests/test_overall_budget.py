import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from bench.agentcli import make_submission
from bench.budget import remaining_time
from bench.evalrun import one_seed
from bench.judge import bench_judge, run_seeds
from bench.run import pick_final, render_prompt
from bench.task import load_task
from bench.util import capture, run, sha256_bytes


class OverallBudgetTests(unittest.TestCase):
    def test_stages_share_one_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'train.sh').write_text('sleep 0.5\nprintf trained > "$2/model"\n')
            (root / 'predict.sh').write_text('sleep 1\nprintf prediction > "$6"\n')
            (root / 'validate.py').write_text('raise SystemExit(0)\n')
            (root / 'scorer.py').write_text('print(\'{"metric": 1}\')\n')
            start = time.time()
            result = one_seed(seed=1, code_dir=root, data_dir=root, out_root=root / 'out',
                              input_path=root / 'input', labels_path=root / 'labels',
                              validator_py=root / 'validate.py', scorer_py=root / 'scorer.py',
                              exec_env=dict(os.environ), tool_env=dict(os.environ), tool_python=sys.executable,
                              caps={'train': None, 'predict': None, 'score': None}, deadline_ts=start + 1.4)
            self.assertTrue((root / 'out/model/model').exists())
            self.assertEqual(result['status'], 'budget_exhausted')
            self.assertEqual(result['failure']['stage'], 'predict')
            self.assertFalse((root / 'out/pred.out').exists())
            self.assertLess(time.time() - start, 3)

    def test_missing_submission_fails_before_build_or_training(self):
        from bench.task import Task
        fake_task = Task('/tmp/fake-toy', {'pid': 'toy-ridge', 'metric': {'name': 'mse', 'higher_is_better': False}})
        for source in ('no_submission', 'deadline_snapshot'):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); run = root / 'run'; run.mkdir()
                (run / 'run.json').write_text(json.dumps({'task': 'toy-ridge', 'run_id': 'missing', 'budget_s': 60}))
                (run / 'final.json').write_text(json.dumps({'source': source, 'ok': source == 'deadline_snapshot'}))
                with patch('bench.judge.load_task', return_value=fake_task), patch('bench.judge.build_environment') as build:
                    result = bench_judge(run_dir=run, local_root=root / 'work')
                build.assert_not_called()
                self.assertEqual(result['status'], 'no_submission')
                self.assertFalse(result['pass'])

    def test_expired_deadline_and_no_implicit_phase_limit(self):
        self.assertIsNone(remaining_time())
        with self.assertRaises(TimeoutError): remaining_time(time.time() - 1)

    def test_explicit_submission_selection_and_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(pick_final(root, 100)[0])
            records = [dict(n=1, ts=80, ok=True, final=False, source='submit'),
                       dict(n=2, ts=90, ok=True, final=True, source='submit'),
                       dict(n=3, ts=101, ok=True, final=True, source='submit'),
                       dict(n=4, ts=95, ok=False, final=True, source='submit')]
            for item in records:
                pb = str(item['n']).encode()
                item.update(patch_sha256=sha256_bytes(pb), bytes=len(pb))
                (root / f'{item["n"]:03d}.patch').write_bytes(pb)
                (root / f'{item["n"]}.json').write_text(json.dumps(item))
            self.assertEqual(pick_final(root, 100)[0]['n'], 2)
            (root / '2.json').unlink()
            self.assertEqual(pick_final(root, 100)[0]['n'], 1)
            (root / '001.patch').write_bytes(b'changed after submission')
            self.assertIsNone(pick_final(root, 100)[0])
        with patch('bench.agentcli.envlock.freeze') as freeze:
            result = make_submission({'deadline_ts': time.time() - 1})
        freeze.assert_not_called()
        self.assertFalse(result['ok'])

    def test_seed_budget_does_not_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = SimpleNamespace(test_input=root / 'input', test_labels=root / 'labels', data_dir=root)
            stage = SimpleNamespace(code=root, path_map=lambda *args: str, prefix=lambda *args: None)
            deadlines = []
            deadline = time.time() + 10
            def run_seed(**kw):
                deadlines.append(kw['deadline_ts'])
                return {'status': 'ok', 'metrics': {'metric': 1}, 'timing': {}, 'failure': None}
            with patch('bench.judge.one_seed', side_effect=run_seed):
                result = run_seeds(task, stage, [1, 2, 3], root, {}, sys.executable,
                                   root / 'validator', root / 'scorer', {}, val_too=False, deadline_ts=deadline)
            self.assertEqual(len(result), 3)
            self.assertEqual(deadlines, [deadline] * 3)

    def test_prompt_has_single_task_text_and_explicit_submission_requirement(self):
        from bench.task import Task
        task = Task('/tmp/fake-toy', {'pid': 'toy-ridge', 'metric': {'name': 'mse', 'higher_is_better': False, 'delta': 0.02}})
        with tempfile.TemporaryDirectory() as tmp:
            text = render_prompt(task, SimpleNamespace(cfg_dir=Path(tmp), cfg={}), 7200, 2, 'UNIQUE TASK STATEMENT')
        self.assertEqual(text.count('UNIQUE TASK STATEMENT'), 1)
        self.assertNotIn('Read it first', text)
        self.assertNotIn('training must finish within', text)
        self.assertNotIn('train_cap_min', text)
        self.assertNotIn('snapshot of `code/`', text)
        self.assertIn('If no valid submission is recorded', text)
        self.assertIn('whole method', text)
        self.assertIn('fixed research objects', text)
        self.assertIn('editable only when the task permits', text)
        self.assertIn('task-prescribed starting state', text)
        self.assertNotIn('anything in it: model, loss', text)

    def test_capture_timeout_stops_children_that_ignore_term(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / 'escaped'
            child = 'import signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(1); pathlib.Path(' + repr(str(marker)) + ').write_text("bad")'
            parent = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",' + repr(child) + ']); time.sleep(10)'
            with self.assertRaises(subprocess.TimeoutExpired):
                capture([sys.executable, '-c', parent], timeout=.3)
            time.sleep(1.1)
            self.assertFalse(marker.exists())

    def test_successful_parent_cannot_leave_workers_running(self):
        for runner in (run, capture):
            with self.subTest(runner=runner.__name__), tempfile.TemporaryDirectory() as tmp:
                marker = Path(tmp) / 'escaped'
                child = ('import signal,time,pathlib; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                         'time.sleep(1.2); pathlib.Path(' + repr(str(marker)) + ').write_text("bad")')
                parent = ('import subprocess,sys; subprocess.Popen([sys.executable,"-c",' + repr(child)
                          + '],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)')
                result = runner([sys.executable, '-c', parent], timeout=3)
                self.assertEqual(result['rc'] if runner is run else result.returncode, 0)
                time.sleep(1.3)
                self.assertFalse(marker.exists())

    def test_scorer_receives_deadline_and_successful_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'train.sh').write_text('printf trained > "$2/model"\n')
            (root / 'predict.sh').write_text('printf prediction > "$6"\n')
            (root / 'validate.py').write_text('raise SystemExit(0)\n')
            (root / 'scorer.py').write_text(
                'import json,os,time\n'
                'assert float(os.environ["BENCH_EVAL_DEADLINE_TS"]) > time.time()\n'
                'print(json.dumps({"metric": 1}))\n')
            result = one_seed(seed=1, code_dir=root, data_dir=root, out_root=root / 'out',
                              input_path=root / 'input', labels_path=root / 'labels',
                              validator_py=root / 'validate.py', scorer_py=root / 'scorer.py',
                              exec_env=dict(os.environ), tool_env=dict(os.environ), tool_python=sys.executable,
                              caps={'train': None, 'predict': None, 'score': None}, deadline_ts=time.time() + 5)
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['metrics'], {'metric': 1})

    def test_snapshot_scorer_receives_trusted_overall_deadline(self):
        from bench.sandbox import probe_proc_flags
        try: probe_proc_flags()
        except RuntimeError as exc: self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as tmp:
            task = Path(tmp) / 'task'
            (task / 'runtime').mkdir(parents=True)
            (task / 'agent/bench').mkdir(parents=True)
            (task / 'runtime/runtime.json').write_text(json.dumps({'version': 1, 'mounts': [], 'python': '/usr/bin/python3'}))
            (task / 'task.json').write_text(json.dumps({'pid': 'deadline-fixture', 'env': {'mode': 'snapshot'}}))
            scorer = task / 'agent/bench/scorer.py'
            scorer.write_text('import os,json\nprint(json.dumps({"deadline": float(os.environ["BENCH_EVAL_DEADLINE_TS"])}))\n')
            deadline = time.time() + 10
            env = dict(os.environ, BENCH_EVAL_DEADLINE_TS=str(deadline))
            helper = Path(__file__).resolve().parents[1] / 'bench/runtime_tool.py'
            result = capture([sys.executable, str(helper), str(task), '1', str(scorer)], env=env, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['deadline'], deadline)
