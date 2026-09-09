"""
The HTML/JS Web UI is THE GUI (no Tkinter available):
- launch_gui / launch_fermi_gui / `deep-research gui` start the Web UI and
  open a browser (never import tkinter)
- the settings the Tk GUI used to offer reach create_config from the Web UI
- Fermi estimation runs as a Web UI job (one LLM call, offline here)
"""

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from deep_research_tool.webui.server import (
    JobManager,
    WebUIHandler,
    build_config_kwargs,
)


class TestGuiLaunchers:

    def test_launch_gui_runs_the_web_ui_not_tk(self):
        import deep_research_tool
        calls = []
        with patch("deep_research_tool.webui.server.run_server",
                   side_effect=lambda **kw: calls.append(kw)):
            deep_research_tool.launch_gui(port=9999, output_dir="/tmp/o")
            deep_research_tool.launch_fermi_gui(port=9998)
        assert calls[0]["port"] == 9999 and calls[0]["open_browser"] is True
        assert calls[1]["fragment"] == "fermi"

    def test_cli_gui_command_opens_browser_on_web_ui(self):
        from click.testing import CliRunner
        from deep_research_tool.cli import cli
        calls = []
        with patch("deep_research_tool.webui.server.run_server",
                   side_effect=lambda **kw: calls.append(kw)):
            result = CliRunner().invoke(cli, ["gui", "--port", "8123", "--fermi"])
        assert result.exit_code == 0, result.output
        assert calls[0]["port"] == 8123
        assert calls[0]["open_browser"] is True
        assert calls[0]["fragment"] == "fermi"

    def test_run_server_opens_browser_when_asked(self, tmp_path):
        from deep_research_tool.webui import server as srv
        opened = []
        fake_server = SimpleNamespace(serve_forever=lambda: None,
                                      server_close=lambda: None)
        with patch.object(srv, "ThreadingHTTPServer",
                          return_value=fake_server), \
                patch("webbrowser.open", side_effect=opened.append), \
                patch.object(srv.threading, "Timer",
                             side_effect=lambda d, fn: SimpleNamespace(start=fn)):
            srv.run_server("127.0.0.1", 8555, str(tmp_path),
                           open_browser=True, fragment="fermi")
        assert opened == ["http://127.0.0.1:8555/#fermi"]


class TestTkParityParams:

    def test_former_tk_settings_reach_create_config(self):
        kwargs = build_config_kwargs({
            "temperature": 0.3, "max_tokens": 2048, "max_results": 20,
            "target_pages": 12, "search_region": "jp-jp",
            "include_images": False, "include_citations": True,
            "include_toc": False,
        })
        assert kwargs["temperature"] == 0.3
        assert kwargs["max_tokens"] == 2048
        assert kwargs["max_results"] == 20
        assert kwargs["target_pages"] == 12
        assert kwargs["search_region"] == "jp-jp"
        assert kwargs["include_images"] is False
        assert kwargs["include_toc"] is False
        # and create_config accepts them without warnings
        import warnings
        from deep_research_tool.config import create_config
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cfg = create_config(provider="openai", openai_api_key="sk-t",
                                **kwargs)
        assert cfg.api.temperature == 0.3
        assert cfg.api.max_tokens == 2048
        assert cfg.search.max_results == 20
        assert cfg.search.region == "jp-jp"
        assert cfg.report.target_pages == 12
        assert cfg.report.include_images is False

    def test_invalid_parity_values_are_field_errors(self):
        from deep_research_tool.webui.server import FieldError
        for key, bad in (("temperature", 2.5), ("max_tokens", 0),
                         ("max_results", "abc"), ("target_pages", -1)):
            with pytest.raises(FieldError) as ei:
                build_config_kwargs({key: bad})
            assert ei.value.field == key


FERMI_JSON = json.dumps({
    "factors": [
        {"name": "人口", "description": "日本の人口", "low": 120e6, "mid": 125e6,
         "high": 126e6, "unit": "人", "operation": "multiply", "basis": "known_value"},
        {"name": "1人あたり台数", "description": "", "low": 0.4, "mid": 0.5,
         "high": 0.6, "unit": "台/人", "operation": "multiply", "basis": "assumption"},
    ],
    "unit": "台", "formula": "人口 × 1人あたり台数",
    "reasoning": "人口に保有率を掛ける", "assumptions": ["保有率は横ばい"],
}, ensure_ascii=False)


class FermiFakeLLM:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, system_prompt=None, **kw):
        self.calls += 1
        return SimpleNamespace(content=FERMI_JSON)


@pytest.fixture
def server(tmp_path):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = JobManager(output_dir=str(tmp_path))
    httpd.output_dir = str(tmp_path)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd
    httpd.shutdown()


def _post(server, path, payload):
    port = server.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestFermiWebJob:

    def test_fermi_endpoint_validates_input(self, server):
        status, data = _post(server, "/api/fermi", {"question": ""})
        assert status == 400 and data["field"] == "fermi_question"
        status, data = _post(server, "/api/fermi",
                             {"question": "q", "known_values": {"人口": "abc"}})
        assert status == 400 and data["field"] == "fermi_known"

    def test_fermi_job_runs_offline_and_saves_files(self, server, tmp_path):
        llm = FermiFakeLLM()
        with patch("deep_research_tool.api.get_client", return_value=llm):
            status, data = _post(server, "/api/fermi", {
                "question": "日本の乗用車保有台数は？",
                "known_values": {"日本の人口": "125,000,000"},
                "provider": "openai", "openai_api_key": "sk-test",
            })
            assert status == 202
            job = server.job_manager.get(data["job_id"])
            for _ in range(100):
                if job.is_terminal:
                    break
                time.sleep(0.1)
        assert job.state == "completed", job.error
        assert llm.calls == 1                       # exactly one LLM call
        r = job.result
        assert r["kind"] == "fermi"
        assert r["fermi"]["value"] == pytest.approx(62.5e6)
        assert "人口" in r["fermi_markdown"]
        assert r["status"] == {"process": "completed", "verification": "skipped",
                               "quality": "unverified"}
        saved = r["saved_artifacts"]
        assert saved["fermi_md"].endswith(".md") and saved["fermi_json"].endswith(".json")
        from pathlib import Path
        assert Path(saved["fermi_md"]).is_file() and Path(saved["fermi_json"]).is_file()
        assert str(tmp_path) in saved["fermi_md"]   # under the job's output dir
        # never a secret in the ledger
        ledger = (tmp_path / "jobs_ledger.json").read_text("utf-8")
        assert "sk-test" not in ledger and "フェルミ推定" in ledger
        # status endpoint exposes the result for the panel's polling
        port = server.server_address[1]
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/status?job_id={job.job_id}") as res:
            st = json.loads(res.read())
        assert st["result"]["fermi_markdown"]
