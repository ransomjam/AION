"""Tests for the app API: scratch labs, projects, datasets, jobs, registry."""

import tempfile
import unittest
from pathlib import Path

from aion.app import api


class ScratchLabTest(unittest.TestCase):
    def test_tokenize_report(self):
        r = api.tokenize_report("Café ﬁle!!")
        self.assertEqual(r["stages"][1]["value"], "cafe file!!")
        self.assertEqual(r["tokens"].count("!"), 2)

    def test_vocabulary_report_probe(self):
        r = api.vocabulary_report(["the cat", "the dog"], probe="the fox")
        self.assertEqual(r["probe"]["unknown"], ["fox"])

    def test_labs_registry_lists_home_and_labs(self):
        reg = api.labs_registry()
        ids = {l["id"] for l in reg["labs"]}
        self.assertIn("projects", ids)
        self.assertIn("data", ids)
        data = next(l for l in reg["labs"] if l["id"] == "data")
        self.assertEqual(data["scope"], "project")


class ProjectDataFlowTest(unittest.TestCase):
    """Exercises the full project → dataset → import → analyze → search flow."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        api.set_workspace_root(Path(self._tmp.name))

    def tearDown(self):
        api.set_workspace_root(None)
        self._tmp.cleanup()

    def test_project_lifecycle(self):
        api.create_project("TinyGPT", description="d", language="en")
        self.assertEqual(api.list_projects()["projects"][0]["id"], "tinygpt")
        self.assertEqual(api.get_project("tinygpt")["project"]["language"], "en")
        api.update_project("tinygpt", {"description": "updated"})
        self.assertEqual(api.get_project("tinygpt")["project"]["description"], "updated")
        self.assertEqual(api.archive_project("tinygpt")["project"]["status"], "archived")

    def test_import_analyze_search(self):
        api.create_project("Proj")
        api.create_dataset("proj", "Corpus")
        imp = api.import_text("proj", "corpus", "the cat\nthe dog\nthe the the",
                              split="lines")
        self.assertEqual(imp["job"]["status"], "completed")
        self.assertEqual(imp["job"]["result"]["imported"], 3)

        got = api.get_dataset("proj", "corpus")
        self.assertEqual(len(got["document_ids"]), 3)

        analysis = api.analyze_dataset("proj", "corpus")
        self.assertEqual(analysis["job"]["status"], "completed")
        self.assertEqual(analysis["statistics"]["documents"], 3)
        self.assertEqual(analysis["statistics"]["top_tokens"][0], ["the", 5])
        self.assertIn("issues", analysis["quality"])

        found = api.search_documents("proj", "corpus", "cat")
        self.assertEqual(found["match_count"], 1)

    def test_document_edit_and_delete(self):
        api.create_project("Proj")
        api.create_dataset("proj", "C")
        api.import_text("proj", "c", "hello", split="one")
        self.assertEqual(api.get_document("proj", "c", 1)["text"], "hello")
        api.edit_document("proj", "c", 1, "changed")
        self.assertEqual(api.get_document("proj", "c", 1)["text"], "changed")
        api.delete_document("proj", "c", 1)
        self.assertEqual(api.get_dataset("proj", "c")["document_ids"], [])

    def test_jobs_recorded(self):
        api.create_project("Proj")
        api.create_dataset("proj", "C")
        api.import_text("proj", "c", "a\nb", split="lines")
        jobs = api.list_jobs("proj")["jobs"]
        self.assertTrue(jobs)
        job_id = jobs[0]["id"]
        self.assertEqual(api.get_job("proj", job_id)["job"]["id"], job_id)


if __name__ == "__main__":
    unittest.main()
