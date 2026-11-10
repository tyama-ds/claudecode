"""Web UI: the measured body length next to the length setting (v1.12.0).

The UI used to echo only the target (「固定 約4,000字」), so a report many
times over it looked normal. The result view and the history now show the
measured body size and whether it is within the target.
"""

import json
import threading
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from deep_research_tool.tests.test_webui_state import _browser, _load, _url
from deep_research_tool.webui.server import JobManager, WebUIHandler

OVER = {"mode": "fixed", "body_chars": 13200, "target": 4000, "lower": 3200,
        "upper": 4800, "within": False, "over": True, "under": False}
WITHIN = {"mode": "fixed", "body_chars": 4120, "target": 4000, "lower": 3200,
          "upper": 4800, "within": True, "over": False, "under": False}


def _server(tmp_path, jobs):
    (tmp_path / "jobs_ledger.json").write_text(json.dumps({"version": 1, "jobs": jobs}),
                                               encoding="utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = JobManager(output_dir=str(tmp_path))
    httpd.output_dir = str(tmp_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def _job(jid, length, started):
    return {"job_id": jid, "query": "q", "state": "completed", "started_at": started,
            "params_summary": {"deliverable": "summary", "length_mode": "fixed",
                               "preferred_body_chars": 4000},
            "result": {"status": {"verification": "performed"}, "length": length}}


def test_the_server_keeps_the_measured_length():
    from deep_research_tool.webui.server import ResearchJob
    job = ResearchJob("job-l", "q", {"query": "q"})
    JobManager._finish_from_result(job, {}, {"report_path": "r.md", "length": WITHIN,
                                             "run_status": "completed"},
                                   SimpleNamespace(to_dict_list=lambda: [], count=lambda: 0))
    assert job.result["length"] == WITHIN


@pytest.mark.parametrize("length,want,absent", [
    (WITHIN, ["固定 約4,000字 → 実績 4,120字"], ["超過"]),
    (OVER, ["実績 13,200字（超過）"], []),
    (None, ["固定 約4,000字"], ["実績"]),
])
def test_history_shows_the_measured_length(tmp_path, length, want, absent):
    httpd = _server(tmp_path, {"job-x": _job("job-x", length, 1.0)})
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(httpd, "/#history"), None)
        text = page.inner_text("#historyList .hist-params")
        for w in want:
            assert w in text, (w, text)
        for a in absent:
            assert a not in text, (a, text)
        assert errors == []
    finally:
        browser.close()
        pw.stop()
        httpd.shutdown()
        httpd.server_close()


CEILING = {"mode": "adaptive", "body_chars": 31000, "target": None,
           "hard_min": None, "hard_max": 20000}


def test_the_user_set_ceiling_is_checked_in_adaptive_mode(tmp_path):
    # review C15: an adaptive body over 本文の上限 showed a green 'ok' box
    job = _job("job-c", CEILING, 1.0)
    job["params_summary"] = {"deliverable": "standard", "length_mode": "adaptive"}
    httpd = _server(tmp_path, {"job-c": job})
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(httpd, "/#history"), None)
        assert "実績 31,000字（超過）" in page.inner_text("#historyList .hist-params")
        page.evaluate("showResults(historyCache.find(r => r.job_id === 'job-c').result, 'job-c', true); setView('result')")
        page.wait_for_selector("#lengthBox")
        box = page.inner_text("#lengthBox")
        assert "上限 20,000字を超過" in box, box
        assert "over" in page.get_attribute("#lengthBox", "class")
        assert errors == []
    finally:
        browser.close()
        pw.stop()
        httpd.shutdown()
        httpd.server_close()


def test_verify_runs_show_the_measured_length(tmp_path):
    # review C16: verify-only runs compress a fixed body too
    job = _job("job-v", OVER, 1.0)
    job["params_summary"] = {"verify": True, "length_mode": "fixed",
                             "preferred_body_chars": 4000}
    job["result"]["kind"] = "verify"
    httpd = _server(tmp_path, {"job-v": job})
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(httpd, "/#history"), None)
        text = page.inner_text("#historyList .hist-params")
        assert "検証のみ" in text and "実績 13,200字（超過）" in text, text
        assert errors == []
    finally:
        browser.close()
        pw.stop()
        httpd.shutdown()
        httpd.server_close()


def test_result_view_says_whether_the_body_fits(tmp_path):
    httpd = _server(tmp_path, {"job-over": _job("job-over", OVER, 2.0),
                               "job-ok": _job("job-ok", WITHIN, 1.0)})
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(httpd, "/#history"), None)
        page.evaluate("showResults(historyCache.find(r => r.job_id === 'job-over').result, 'job-over', true); setView('result')")
        page.wait_for_selector("#lengthBox")
        box = page.inner_text("#lengthBox")
        assert "13,200字" in box and "目標 約4,000字" in box and "超過" in box, box
        assert "参考文献" in box          # the count covers the body only
        assert "over" in page.get_attribute("#lengthBox", "class")
        page.evaluate("showResults(historyCache.find(r => r.job_id === 'job-ok').result, 'job-ok', true)")
        page.wait_for_timeout(200)
        box = page.inner_text("#lengthBox")
        assert "4,120字" in box and "範囲内" in box, box
        assert "over" not in page.get_attribute("#lengthBox", "class")
        assert errors == []
    finally:
        browser.close()
        pw.stop()
        httpd.shutdown()
        httpd.server_close()
