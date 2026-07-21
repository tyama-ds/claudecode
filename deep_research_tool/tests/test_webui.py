"""
Tests for the Web UI server (parameter mapping and HTTP endpoints).
"""

import json
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer

import pytest

from deep_research_tool.webui.server import (
    JobManager,
    ResearchJob,
    WebUIHandler,
    build_config_kwargs,
)


class TestBuildConfigKwargs:
    def test_basic_mapping(self):
        kwargs = build_config_kwargs({
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "crawl_mode": "aicrawl",
            "ai_crawl_max_pages": 20,
            "ai_crawl_site_depth": 3,
            "gap_fill_rounds": 2,
            "report_version": "v2",
            "chart_library": "seaborn",
            "auto_figures": True,
        })
        assert kwargs["provider"] == "anthropic"
        assert kwargs["crawl_mode"] == "aicrawl"
        assert kwargs["ai_crawl_max_total_pages"] == 20
        assert kwargs["ai_crawl_site_depth"] == 3
        assert kwargs["max_gap_fill_rounds"] == 2
        assert kwargs["report_generator_version"] == "v2"
        assert kwargs["chart_library"] == "seaborn"
        assert kwargs["auto_figures"] is True

    def test_empty_values_dropped(self):
        kwargs = build_config_kwargs({"model": "", "provider": "openai"})
        assert "model" not in kwargs
        assert kwargs["provider"] == "openai"

    def test_unknown_keys_ignored(self):
        kwargs = build_config_kwargs({"query": "x", "evil_param": "y"})
        assert "query" not in kwargs  # query is not a config kwarg
        assert "evil_param" not in kwargs

    def test_stage_llm_cleaned(self):
        kwargs = build_config_kwargs({
            "stage_llm": {
                "writing": {"provider": "anthropic", "model": "claude-x", "api_key": ""},
                "planning": {"provider": "", "model": ""},  # empty -> dropped
            }
        })
        assert kwargs["stage_llm"] == {
            "writing": {"provider": "anthropic", "model": "claude-x"}
        }

    def test_no_stage_llm_key_when_all_empty(self):
        kwargs = build_config_kwargs({"stage_llm": {"writing": {"provider": ""}}})
        assert "stage_llm" not in kwargs

    def test_gap_fill_rounds_zero_preserved(self):
        # 0 is a meaningful value (disable) and must not be dropped
        kwargs = build_config_kwargs({"gap_fill_rounds": 0})
        assert kwargs["max_gap_fill_rounds"] == 0


@pytest.fixture
def server(tmp_path):
    """Start the Web UI server on an ephemeral port."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = JobManager()
    httpd.output_dir = str(tmp_path)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()


def get(server, path):
    port = server.server_address[1]
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as res:
        return res.status, res.read()


def post(server, path, payload):
    port = server.server_address[1]
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


class TestEndpoints:
    def test_index_served(self, server):
        status, body = get(server, "/")
        assert status == 200
        text = body.decode("utf-8")
        assert "Deep Research Tool" in text
        assert "調査を開始" in text

    def test_status_idle(self, server):
        status, body = get(server, "/api/status")
        assert status == 200
        assert json.loads(body)["state"] == "idle"

    def test_research_requires_query(self, server):
        status, data = post(server, "/api/research", {"query": ""})
        assert status == 400
        assert "query" in data["error"]

    def test_reports_empty_list(self, server):
        status, body = get(server, "/api/reports")
        assert status == 200
        assert json.loads(body)["reports"] == []

    def test_live_report_unavailable_without_job(self, server):
        status, body = get(server, "/api/live-report")
        assert status == 200
        assert json.loads(body)["available"] is False

    def test_live_report_serves_job_snapshot(self, server):
        from deep_research_tool.report.live_report import WebUILiveSink

        manager = server.job_manager
        job = ResearchJob("job-live", "テーマ")
        manager.jobs[job.job_id] = job
        job.live_sink = WebUILiveSink()
        job.live_sink.on_plan("ライブタイトル", [{"section": "1", "title": "章"}])
        job.live_sink.on_section("1", "章", "草稿本文", draft=True)

        status, body = get(server, "/api/live-report?job_id=job-live")
        assert status == 200
        data = json.loads(body)
        assert data["available"] is True
        assert data["title"] == "ライブタイトル"
        assert data["sections"]["1"]["text"] == "草稿本文"
        assert data["sections"]["1"]["draft"] is True
        assert data["rev"] >= 2

    def test_reports_lists_files(self, server, tmp_path):
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        (reports_dir / "report_1.md").write_text("# r", encoding="utf-8")
        status, body = get(server, "/api/reports")
        data = json.loads(body)
        assert status == 200
        assert data["reports"][0]["name"] == "report_1.md"

    def test_report_file_download(self, server, tmp_path):
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        f = reports_dir / "report_2.md"
        f.write_text("# report body", encoding="utf-8")
        status, body = get(
            server, f"/api/report-file?path={urllib.request.quote(str(f))}",
        )
        assert status == 200
        assert b"report body" in body

    def test_report_file_rejects_path_traversal(self, server, tmp_path):
        outside = tmp_path.parent / "secret.txt"
        outside.write_text("secret", encoding="utf-8")
        try:
            port = server.server_address[1]
            url = (f"http://127.0.0.1:{port}/api/report-file"
                   f"?path={urllib.request.quote(str(outside))}")
            try:
                with urllib.request.urlopen(url) as res:
                    status = res.status
            except urllib.error.HTTPError as e:
                status = e.code
            assert status == 400
        finally:
            outside.unlink()

    def test_running_job_status_shape(self, server):
        job = ResearchJob("job-1", "テストクエリ")
        job.update("調査中...", 42.0)
        server.job_manager.jobs[job.job_id] = job   # parallel-jobs registry
        status, body = get(server, "/api/status")
        data = json.loads(body)
        assert status == 200
        assert data["state"] == "running"
        assert data["progress"] == 42.0
        assert data["message"] == "調査中..."
        assert data["log"][-1]["msg"] == "調査中..."
        # per-job targeting
        status, body = get(server, f"/api/status?job_id={job.job_id}")
        assert json.loads(body)["job_id"] == job.job_id
        # jobs listing
        status, body = get(server, "/api/jobs")
        listing = json.loads(body)
        assert listing["max_concurrent"] >= 1
        assert any(j["job_id"] == job.job_id for j in listing["jobs"])

    def test_job_rejected_at_max_concurrent(self, server):
        manager = server.job_manager
        for i in range(manager.MAX_CONCURRENT):
            job = ResearchJob(f"job-{i+1}", f"q{i+1}")   # state=running
            manager.jobs[job.job_id] = job
        status, data = post(server, "/api/research", {"query": "one too many"})
        assert status == 409
