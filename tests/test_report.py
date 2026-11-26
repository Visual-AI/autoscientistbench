"""Unit tests for `bench.report` official leaderboard scoring & CLI."""
import json
import tempfile
import unittest
from pathlib import Path

from bench import report


class LeaderboardReportTests(unittest.TestCase):
    def test_compute_task_improvement_score_both_directions_and_clipping(self):
        # Higher is better: baseline=0.80, ceiling=1.00
        s, raw = report.compute_task_improvement_score(0.90, 0.80, 1.00, status="scored")
        self.assertAlmostEqual(s, 50.0)
        self.assertAlmostEqual(raw, 50.0)

        # Uncapped above 100%
        s_above, raw_above = report.compute_task_improvement_score(1.05, 0.80, 1.00, status="scored")
        self.assertAlmostEqual(s_above, 125.0)
        self.assertAlmostEqual(raw_above, 125.0)

        # Regression clipped to 0%
        s_reg, raw_reg = report.compute_task_improvement_score(0.75, 0.80, 1.00, status="scored")
        self.assertEqual(s_reg, 0.0)
        self.assertAlmostEqual(raw_reg, -25.0)

        # Lower is better: baseline=16.2, ceiling=0.0 -> x=8.1 gives +50%
        s_low, raw_low = report.compute_task_improvement_score(8.1, 16.2, 0.0, status="scored")
        self.assertAlmostEqual(s_low, 50.0)
        self.assertAlmostEqual(raw_low, 50.0)

        # Lower is better regression: x=20.0 -> clipped to 0%
        s_low_bad, raw_low_bad = report.compute_task_improvement_score(20.0, 16.2, 0.0, status="scored")
        self.assertEqual(s_low_bad, 0.0)
        self.assertLess(raw_low_bad, 0.0)

        # Non-scored / missing / non-finite -> 0.0
        self.assertEqual(report.compute_task_improvement_score(8.1, 16.2, 0.0, status="no_submission"), (0.0, None))
        self.assertEqual(report.compute_task_improvement_score(None, 16.2, 0.0, status="scored"), (0.0, None))
        self.assertEqual(report.compute_task_improvement_score(float("inf"), 16.2, 0.0, status="scored"), (0.0, None))

    def test_build_report_from_runs_and_csv_export(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            runs_dir = td / "runs"

            # Run 1: 001_InertialGenome using real bench_run run.json dict schema (harness.describe() + requested_model)
            r1 = runs_dir / "r1"
            r1.mkdir(parents=True)
            (r1 / "run.json").write_text(json.dumps({
                "run_id": "r1",
                "task": "001_InertialGenome",
                "harness": {"name": "codex-api", "type": "codex", "model": "gpt-5.4"},
                "requested_model": "demo-v1",
                "status": "finished",
            }))
            (r1 / "result.json").write_text(json.dumps({
                "status": "scored", "pass": True, "gain": 0.07,
                "candidate": {"test": {"mean": 0.877382058137856}},
            }))

            # Run 2: 004_GradNormSmooth using --set model override in harness dict (requested_model=None)
            r2 = runs_dir / "r2"
            r2.mkdir(parents=True)
            (r2 / "run.json").write_text(json.dumps({
                "run_id": "r2",
                "task": "004_GradNormSmooth",
                "harness": {"name": "codex-api", "type": "codex", "model": "demo-v1"},
                "requested_model": None,
                "status": "finished",
            }))
            (r2 / "result.json").write_text(json.dumps({
                "status": "scored", "pass": True, "gain": 0.5,
                "candidate": {"test": {"mean": 8.1}},
            }))

            # Run 3: 002_CLEAR failed with no_submission -> contributes 0
            r3 = runs_dir / "r3"
            r3.mkdir(parents=True)
            (r3 / "run.json").write_text(json.dumps({
                "run_id": "r3",
                "task": "002_CLEAR",
                "harness": {"name": "codex-api", "type": "codex", "model": "demo-v1"},
                "status": "no_submission",
            }))

            rep = report.build_report(runs_dir=runs_dir)
            self.assertEqual(rep["denominator"], 104)
            self.assertEqual(len(rep["groups"]), 1)
            g = rep["groups"][0]
            self.assertEqual(g["group"], "codex-api:demo-v1")
            self.assertEqual(g["harness"], "codex-api")
            self.assertEqual(g["model"], "demo-v1")
            self.assertEqual(g["evaluated"], 3)
            self.assertEqual(g["scored"], 2)
            self.assertEqual(g["passed"], 2)
            # Both scored tasks are exactly halfway between baseline and ceiling (50% + 50% + 0%) / 104
            self.assertAlmostEqual(g["improvement_score"], 100.0 / 104.0, places=5)
            self.assertAlmostEqual(g["evaluated_only_improvement_score"], 100.0 / 3.0, places=5)

            # Also verify suite mode (`results/<suite>/per_run/*.json`)
            suite_dir = td / "suite_demo"
            (suite_dir / "per_run").mkdir(parents=True)
            (suite_dir / "per_run" / "004_GradNormSmooth__codex-api__r1.json").write_text(json.dumps({
                "task": "004_GradNormSmooth",
                "harness": "codex-api",
                "repeat": 1,
                "run_id": "r2",
                "status": "scored",
                "pass": True,
                "gain": 0.5,
                "candidate_test_mean": 8.1,
            }))
            rep_suite = report.build_report(suite=str(suite_dir), runs_dir=runs_dir, domain_filter="mathematics")
            self.assertEqual(rep_suite["domain_filter"], "mathematics")
            self.assertEqual(rep_suite["groups"][0]["group"], "codex-api:demo-v1")
            self.assertEqual(rep_suite["groups"][0]["passed"], 1)

            txt = report.format_report_text(rep)
            self.assertIn("codex-api:demo-v1", txt)
            csv_p = td / "out.csv"
            report.write_report_csv(rep, csv_p)
            lines = csv_p.read_text().splitlines()
            self.assertEqual(len(lines), 105)  # header + 104 tasks


if __name__ == "__main__":
    unittest.main()
