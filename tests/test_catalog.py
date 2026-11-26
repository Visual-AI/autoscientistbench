"""Retirement removes active membership while retaining historical task files."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bench import catalog, task
from bench.run import _bench_run


class CatalogTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.catalog = self.root / "benchmark_tasks.json"
        self.catalog.write_text(json.dumps({"schema_version": 1, "task_ids": ["active"],
            "retired": [{"task_id": "retired"}]}))
        self.root_patch = patch.object(catalog, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_list_omits_retired_but_preserves_artifacts(self):
        tasks = self.root / "tasks"
        for pid in ("active", "retired", "toy-ridge"):
            (tasks / pid).mkdir(parents=True)
            (tasks / pid / "task.json").write_text(json.dumps({"pid": pid}))
        with patch.object(task, "TASKS_DIR", tasks):
            self.assertEqual(task.list_tasks(), ["active", "toy-ridge"])
            self.assertEqual(task.load_task("retired").pid, "retired")
        self.assertTrue((tasks / "retired/task.json").exists())

    def test_retired_run_stops_before_harness_or_resources(self):
        with patch("bench.run.load_task", return_value=SimpleNamespace(pid="retired")), \
                patch("bench.run.load_harness") as harness:
            with self.assertRaisesRegex(ValueError, "removed from the benchmark"):
                _bench_run("retired", "unused", 1, dry=True)
            harness.assert_not_called()

    def test_conflicting_membership_is_rejected(self):
        self.catalog.write_text(json.dumps({"schema_version": 1, "task_ids": ["active", "retired"],
            "retired": [{"task_id": "retired"}]}))
        with self.assertRaises(ValueError):
            catalog.read_catalog()

    def test_older_checkout_without_catalog_is_supported(self):
        self.catalog.unlink()
        self.assertIsNone(catalog.read_catalog())
        catalog.require_active_task("active")


if __name__ == "__main__":
    unittest.main()
