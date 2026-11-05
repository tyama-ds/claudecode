import io
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from app import ROOT, LocalHTTPServer, PilotState, make_handler, preflight, read_json


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.state = PilotState(cls.temp.name)
        cls.server = LocalHTTPServer(('127.0.0.1', 0), make_handler(cls.state))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.server.server_address[1]}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def fetch(self, path, data=None, headers=None):
        h = dict(headers or {})
        body = None
        if data is not None:
            body = json.dumps(data).encode()
            h.setdefault('Content-Type', 'application/json')
            h.setdefault('X-Pilot-Token', self.state.token)
        req = urllib.request.Request(self.base + path, body, h)
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read()
            return json.loads(content) if 'application/json' in response.headers['Content-Type'] else content

    def payload(self):
        return {'reports': read_json(ROOT / 'samples/reports.json'),
                'gold': read_json(ROOT / 'samples/gold.json'),
                'config': read_json(ROOT / 'samples/models.demo.json')}

    def test_static_and_bootstrap(self):
        self.assertIn('数値ファクトチェック'.encode(), self.fetch('/'))
        data = self.fetch('/api/bootstrap')
        self.assertEqual(data['token'], self.state.token)
        self.assertEqual(len(data['samples']['reports']['reports']), 2)

    def test_port_cannot_be_shared_with_another_server(self):
        with self.assertRaises(OSError):
            LocalHTTPServer(self.server.server_address, make_handler(self.state))

    def test_reject_remote_origin_and_token(self):
        for headers in ({'Origin': 'https://example.com'}, {'X-Pilot-Token': 'wrong'}, {'Host': 'evil.example'}):
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self.fetch('/api/validate', self.payload(), headers)
            self.assertEqual(raised.exception.code, 403)

    def test_malformed_request(self):
        with self.assertRaises(urllib.error.HTTPError) as raised:
            self.fetch('/api/run', {})
        self.assertEqual(raised.exception.code, 400)

    def test_validation_and_full_demo_download(self):
        data = self.payload()
        validation = self.fetch('/api/validate', data)
        self.assertTrue(validation['demo'])
        self.assertEqual(validation['fill_calls'], 12)
        job = self.fetch('/api/run', data)
        until = time.monotonic() + 15
        while True:
            status = self.fetch('/api/jobs/' + job['id'])
            if status['status'] != 'running':
                break
            self.assertLess(time.monotonic(), until)
            time.sleep(.03)
        self.assertEqual(status['status'], 'completed', status)
        self.assertEqual(len(status['result']['fill_metrics']), 2)
        self.assertEqual(len(status['result']['pipeline_metrics']), 4)
        summary = self.fetch('/api/runs/' + job['id'])
        self.assertTrue(summary['demo'])
        archive = self.fetch('/api/runs/' + job['id'] + '/download')
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            self.assertIn('summary.json', zipped.namelist())
            self.assertIn('calls.jsonl', zipped.namelist())

    def test_unknown_path_and_traversal(self):
        for path in ('/README.md', '/api/runs/../README', '/api/runs/missing'):
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self.fetch(path)
            self.assertEqual(raised.exception.code, 404)


if __name__ == '__main__':
    unittest.main()
