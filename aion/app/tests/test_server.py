"""Live smoke test: boot the real server and hit its routes end to end."""

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from aion.app import api
from aion.app.server import build_server


class ServerSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        api.set_workspace_root(Path(cls._tmp.name))
        cls.httpd = build_server("127.0.0.1", 0)
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)
        api.set_workspace_root(None)
        cls._tmp.cleanup()

    def _url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def _get(self, path):
        with urllib.request.urlopen(self._url(path), timeout=5) as resp:
            return resp.status, resp.read(), resp.headers.get("Content-Type", "")

    def _post(self, path, payload):
        req = urllib.request.Request(
            self._url(path), data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read())
            finally:
                e.close()

    def test_health_and_labs(self):
        status, body, _ = self._get("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "ok")
        status, body, _ = self._get("/api/labs")
        self.assertEqual(status, 200)
        self.assertIn("projects", {l["id"] for l in json.loads(body)["labs"]})

    def test_index_and_static(self):
        status, body, ctype = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)
        _s, _b, ctype = self._get("/static/app.js")
        self.assertIn("javascript", ctype)

    def test_scratch_tokenize(self):
        status, body = self._post("/api/tokenize", {"text": "Hello WORLD"})
        self.assertEqual(status, 200)
        self.assertEqual(body["tokens"], ["hello", "world"])

    def test_project_and_dataset_flow(self):
        self.assertEqual(self._post("/api/projects/create", {"name": "SrvProj"})[0], 200)
        self._post("/api/datasets/create", {"project_id": "srvproj", "name": "C"})
        imp = self._post("/api/datasets/import_text",
                         {"project_id": "srvproj", "dataset_id": "c",
                          "text": "the cat\nthe dog", "split": "lines"})
        self.assertEqual(imp[0], 200)
        self.assertEqual(imp[1]["job"]["result"]["imported"], 2)
        an = self._post("/api/datasets/analyze",
                        {"project_id": "srvproj", "dataset_id": "c"})
        self.assertEqual(an[1]["statistics"]["documents"], 2)

    def test_duplicate_project_conflicts(self):
        self._post("/api/projects/create", {"name": "Dup"})
        status, body = self._post("/api/projects/create", {"name": "Dup"})
        self.assertEqual(status, 409)

    def test_unknown_project_404(self):
        status, _ = self._post("/api/projects/get", {"id": "ghost"})
        self.assertEqual(status, 404)

    def test_bad_input_400(self):
        status, _ = self._post("/api/projects/create", {"name": ""})
        self.assertEqual(status, 400)

    def test_unknown_endpoint_404(self):
        status, _ = self._post("/api/nope", {})
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
