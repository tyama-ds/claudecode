"""
The evidence locker is viewable from the Web UI in its own tab:
- GET /evidence serves the viewer page
- GET /api/evidence?job_id=… returns the LIVE locker while the run is
  going, otherwise the evidence JSON the run wrote (history too); files
  outside the output directory are never served; unknown jobs are 404
- EvidenceLocker.to_export_dict is what export_to_json writes
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from deep_research_tool.evidence.locker import EvidenceLocker


@pytest.fixture
def web(tmp_path):
    from deep_research_tool.webui.server import JobManager, WebUIHandler
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = JobManager(output_dir=str(tmp_path))
    httpd.output_dir = str(tmp_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd, tmp_path
    httpd.shutdown()


def _get(server, path):
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=30) as res:
            ctype = res.headers.get("Content-Type", "")
            body = res.read()
            return res.status, (json.loads(body) if "json" in ctype else body.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, body.decode("utf-8", "replace")


def _locker(tmp, n=2):
    locker = EvidenceLocker(research_id="r1", output_dir=tmp)
    for i in range(n):
        locker.add_evidence(url=f"https://example.com/{i}", title=f"出典 {i}",
                            content_excerpt=f"抜粋 {i} 市場規模", section_reference="1")
    return locker


class TestLocker:

    def test_export_dict_matches_the_json_export(self, tmp_path):
        locker = _locker(tmp_path, 3)
        d = locker.to_export_dict()
        path = locker.export_to_json()
        on_disk = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("research_id", "total_evidence", "evidence", "section_evidence", "sections"):
            assert on_disk[key] == d[key]
        assert d["total_evidence"] == 3 and len(d["section_evidence"]["1"]) == 3
        assert "sections" not in locker.to_export_dict(include_sections=False)


class TestEvidenceEndpoint:

    def test_page_is_served(self, web):
        server, _ = web
        status, body = _get(server, "/evidence?job_id=x")
        assert status == 200 and "根拠ロッカー" in body and "/api/evidence" in body

    def test_unknown_job_is_404(self, web):
        server, _ = web
        status, d = _get(server, "/api/evidence?job_id=nope")
        assert status == 404 and d["error"] == "unknown job"

    def test_live_locker_is_served_while_running(self, web):
        server, tmp = web
        from deep_research_tool.webui.server import ResearchJob
        job = ResearchJob("job-live", "炭素繊維", params={"query": "炭素繊維"})
        server.job_manager.jobs["job-live"] = job
        locker = _locker(tmp, 2)
        job.evidence_source = lambda: locker
        job.plan_source = lambda: {"table_of_contents": {"items": [
            {"section": "1", "title": "市場規模", "subsections": [{"section": "1.1", "title": "国内"}]}]}}
        status, d = _get(server, "/api/evidence?job_id=job-live")
        assert status == 200 and d["available"] and d["source"] == "live"
        assert d["total"] == 2 and d["query"] == "炭素繊維" and d["state"] == "running"
        assert d["section_evidence"]["1"] == [e["id"] for e in d["evidence"]]
        assert d["section_titles"] == {"1": "市場規模", "1.1": "国内"}
        assert "sections" not in d
        # a third item added by the run shows up on the next request
        locker.add_evidence(url="https://example.com/3", title="出典 3", section_reference="2")
        assert _get(server, "/api/evidence?job_id=job-live")[1]["total"] == 3
        # nothing yet: available=false (never an error page)
        job.evidence_source = lambda: None
        status, d = _get(server, "/api/evidence?job_id=job-live")
        assert status == 200 and d["available"] is False and d["evidence"] == []

    def test_finished_job_reads_its_evidence_file(self, web):
        server, tmp = web
        from deep_research_tool.webui.server import ResearchJob
        locker = _locker(tmp, 4)
        path = locker.export_to_json()
        job = ResearchJob("job-done", "q", params={"query": "q"})
        job.result = {"evidence_json": str(path), "evidence_csv": str(locker.export_to_csv()),
                      "plan": {"table_of_contents": {"items": [{"section": "1", "title": "T"}]}}}
        job.progress = 100.0
        job.finish("completed")
        server.job_manager.jobs["job-done"] = job
        status, d = _get(server, "/api/evidence?job_id=job-done")
        assert status == 200 and d["source"] == "file" and d["total"] == 4
        assert d["files"]["evidence_json"] == str(path)
        assert d["section_titles"] == {"1": "T"}
        # a long full text is capped for the viewer
        assert all(len(e.get("extracted_text") or "") <= 4000 for e in d["evidence"])

    def test_files_outside_the_output_dir_are_never_served(self, web, tmp_path_factory):
        server, tmp = web
        from deep_research_tool.webui.server import ResearchJob
        elsewhere = tmp_path_factory.mktemp("elsewhere")
        path = _locker(elsewhere, 1).export_to_json()
        job = ResearchJob("job-x", "q", params={"query": "q"})
        job.result = {"evidence_json": str(path)}
        job.finish("completed")
        server.job_manager.jobs["job-x"] = job
        status, d = _get(server, "/api/evidence?job_id=job-x")
        assert status == 200 and d["available"] is False
