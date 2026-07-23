"""Tests for the Job abstraction: lifecycle, structured events, persistence."""

import tempfile
import unittest
from pathlib import Path

from aion.workspace.jobs import JobStatus, JobStore, run_job
from aion.workspace.store import ProjectStore


class JobTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.project = ProjectStore(Path(self._tmp.name)).create("Proj")

    def tearDown(self):
        self._tmp.cleanup()

    def test_success_lifecycle_and_events(self):
        job = run_job(self.project, "demo", {"x": 1},
                      lambda progress: (progress(0.5, "half"), {"ok": True})[1])
        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertEqual(job.result, {"ok": True})
        self.assertEqual(job.progress, 1.0)
        types = [e["type"] for e in job.events]
        self.assertEqual(types[0], "queued")
        self.assertIn("started", types)
        self.assertIn("progress", types)
        self.assertEqual(types[-1], "completed")

    def test_failure_is_captured_not_raised(self):
        def boom(progress):
            raise RuntimeError("nope")

        job = run_job(self.project, "demo", {}, boom)
        self.assertEqual(job.status, JobStatus.FAILED)
        self.assertIn("nope", job.error)
        self.assertEqual(job.events[-1]["type"], "failed")

    def test_non_dict_result_is_wrapped(self):
        job = run_job(self.project, "demo", {}, lambda progress: 7)
        self.assertEqual(job.result, {"value": 7})

    def test_persisted_and_listable(self):
        job = run_job(self.project, "demo", {}, lambda progress: {"ok": 1})
        store = JobStore(self.project.logs_dir())
        loaded = store.get(job.id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.status, JobStatus.COMPLETED)
        self.assertEqual([j.id for j in store.list()], [job.id])


if __name__ == "__main__":
    unittest.main()
