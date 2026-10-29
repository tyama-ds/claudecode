"""
The Web UI with state an upgraded user has (v1.11.1):
- a saved 機能 → 検証 OFF, a broken saved value or a browser that blocks
  site data never stops the page's start-up (the buttons did nothing, 履歴
  stayed empty and 調査を開始 sent nothing)
- a past result opened from 履歴 after a restart shows its 本文
- two servers on one output folder keep each other's history; an
  unreadable ledger is kept aside; a request that raises answers 500
"""

import json
import os
import shutil
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from deep_research_tool.webui import server as srv
from deep_research_tool.webui.server import JobManager, ResearchJob, WebUIHandler


@pytest.fixture
def ui_server(tmp_path):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = JobManager(output_dir=str(tmp_path))
    httpd.output_dir = str(tmp_path)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def _url(httpd, path):
    return f"http://127.0.0.1:{httpd.server_address[1]}{path}"


# ---------------------------------------------------------------------------
# the page in a real browser (skipped where Playwright / Chromium are missing)
# ---------------------------------------------------------------------------

def _browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    exe = "/opt/pw-browsers/chromium"
    pw = sync_api.sync_playwright().start()
    try:
        browser = pw.chromium.launch(**({"executable_path": exe} if os.path.exists(exe) else {}),
                                     args=["--no-sandbox"])
    except Exception as e:                       # no browser installed here
        pw.stop()
        pytest.skip(f"chromium not available: {e}")
    return pw, browser


def _load(browser, url, init_script):
    page = browser.new_page()
    errors, posts = [], []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("request", lambda r: posts.append(r.url) if r.method == "POST" else None)
    if init_script:
        page.add_init_script(init_script)
    page.goto(url, wait_until="networkidle")
    page.wait_for_timeout(800)
    return page, errors, posts


@pytest.mark.parametrize("init_script", [
    # saved 検証 OFF (調査のみ): stopped the page from v1.7.0 on
    "localStorage.setItem('drt_settings', JSON.stringify({enable_verification: false, chart_style: 'enhanced'}))",
    # a saved value no version writes
    "localStorage.setItem('drt_settings', 'null')",
    # site data blocked by the browser
    "Object.defineProperty(window, 'localStorage', {get() { throw new DOMException('denied', 'SecurityError'); }})",
])
def test_saved_state_never_stops_the_page(ui_server, init_script):
    pw, browser = _browser()
    try:
        page, errors, posts = _load(browser, _url(ui_server, "/"), init_script)
        assert errors == [], errors
        assert page.evaluate("pageReady") is True
        assert page.text_content("#appVersion").startswith("v")
        assert "履歴はまだありません" in page.text_content("#historyList")
        assert not page.query_selector("#initErrorBanner")
        page.fill("#query", "炭素繊維の市場調査")
        if page.is_checked("#plan_review"):
            page.uncheck("#plan_review")
        page.click("#startBtn")
        page.wait_for_timeout(800)
        assert any(u.endswith("/api/research") for u in posts), posts
    finally:
        browser.close()
        pw.stop()


def test_saved_verification_off_shows_research_only(ui_server):
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(ui_server, "/"),
                                "localStorage.setItem('drt_settings', JSON.stringify({enable_verification: false}))")
        assert errors == []
        assert page.evaluate("radioValue('run_mode')") == "research_only"
        assert "検証なし" in page.text_content("#startBtn")
    finally:
        browser.close()
        pw.stop()


def test_an_error_while_starting_shows_a_way_out(ui_server):
    pw, browser = _browser()
    try:
        page = browser.new_page()
        page.route("**/", lambda route: route.fulfill(
            status=200, content_type="text/html; charset=utf-8",
            body=(srv.STATIC_DIR / "index.html").read_text(encoding="utf-8").replace(
                "const $ = (id) => document.getElementById(id);\n",
                "const $ = (id) => document.getElementById(id);\n", 1).replace(
                "// initial load: pick up any jobs already running on the server",
                "throw new Error('boom at start');\n// initial load", 1)))
        page.goto(_url(ui_server, "/"), wait_until="networkidle")
        page.wait_for_timeout(500)
        banner = page.text_content("#initErrorBanner")
        assert "boom at start" in banner and "リセット" in banner
    finally:
        browser.close()
        pw.stop()


def test_past_result_body_after_a_restart(tmp_path):
    report = tmp_path / "job-old" / "reports" / "report.md"
    report.parent.mkdir(parents=True)
    report.write_text("# 炭素繊維市場\n\n国内市場は拡大している。", encoding="utf-8")
    (tmp_path / "jobs_ledger.json").write_text(json.dumps({"version": 1, "jobs": {"job-old": {
        "job_id": "job-old", "query": "炭素繊維", "state": "completed", "started_at": 1.0,
        "params_summary": {}, "result": {"report_path": str(report), "status": {}}}}}), encoding="utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = JobManager(output_dir=str(tmp_path))       # a fresh process: no live jobs
    httpd.output_dir = str(tmp_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(httpd, "/#history"), None)
        page.wait_for_timeout(500)
        page.evaluate("showResults(historyCache[0].result, historyCache[0].job_id, true)")
        page.wait_for_timeout(1200)
        assert "国内市場は拡大している" in page.text_content("#resultBody")
        assert errors == []
    finally:
        browser.close()
        pw.stop()
        httpd.shutdown()
        httpd.server_close()


# ---------------------------------------------------------------------------
# the server: ledger, errors, downloads
# ---------------------------------------------------------------------------

def _job(job_id, state="completed"):
    job = ResearchJob(job_id, "q", params={"query": "q"})
    job.state = state
    return job


class TestLedger:

    def test_two_servers_on_one_folder_keep_each_others_records(self, tmp_path):
        a = JobManager(output_dir=str(tmp_path))
        b = JobManager(output_dir=str(tmp_path))
        a.record(_job("job-a1"))
        b.record(_job("job-b1"))
        a.record(_job("job-a2"))                         # A's stale copy has no job-b1
        on_disk = json.loads((tmp_path / "jobs_ledger.json").read_text("utf-8"))["jobs"]
        assert set(on_disk) == {"job-a1", "job-b1", "job-a2"}
        assert {r["job_id"] for r in b.history()} == {"job-a1", "job-b1", "job-a2"}
        # this process's own newer record wins over the copy on disk
        a.record(_job("job-a1", state="error"))
        assert {r["job_id"]: r["state"] for r in a.history()}["job-a1"] == "error"

    def test_forget_is_not_undone_by_the_merge(self, tmp_path):
        a = JobManager(output_dir=str(tmp_path))
        a.record(_job("job-x"))
        a.record(_job("job-y"))
        assert a.forget("job-x")
        assert {r["job_id"] for r in a.history()} == {"job-y"}
        a.record(_job("job-z"))
        assert "job-x" not in json.loads((tmp_path / "jobs_ledger.json").read_text("utf-8"))["jobs"]

    def test_a_second_server_does_not_mark_live_jobs_interrupted(self, tmp_path):
        (tmp_path / "jobs_ledger.json").write_text(json.dumps({"version": 1, "jobs": {
            "job-run": {"job_id": "job-run", "query": "q", "state": "running", "started_at": 1.0}}}),
            encoding="utf-8")
        lock, shared = srv._lock_output_folder(str(tmp_path))
        try:
            assert lock is not None and shared is False
            _lock2, shared2 = srv._lock_output_folder(str(tmp_path))
            assert shared2 is True                     # another live server holds the folder
            m = JobManager(output_dir=str(tmp_path), mark_interrupted=not shared2)
            assert m.history()[0]["state"] == "running"
        finally:
            lock.close()
        lock3, shared3 = srv._lock_output_folder(str(tmp_path))
        assert shared3 is False                         # released with its holder
        lock3.close()

    def test_bom_and_list_ledgers_are_read_and_a_broken_one_is_kept(self, tmp_path):
        ledger = tmp_path / "jobs_ledger.json"
        ledger.write_text(json.dumps({"version": 1, "jobs": {"job-a": {"query": "q", "state": "completed",
                                                                      "started_at": 1.0}}}),
                          encoding="utf-8-sig")
        assert [r["job_id"] for r in JobManager(output_dir=str(tmp_path)).history()] == ["job-a"]
        ledger.write_text(json.dumps([{"job_id": "job-l", "state": "completed", "started_at": 2.0}]),
                          encoding="utf-8")
        assert [r["job_id"] for r in JobManager(output_dir=str(tmp_path)).history()] == ["job-l"]
        ledger.write_text("{not json", encoding="utf-8")
        m = JobManager(output_dir=str(tmp_path))
        assert m.history() == []
        kept = list(tmp_path.glob("jobs_ledger.unreadable-*.json"))
        assert kept and kept[0].read_text(encoding="utf-8") == "{not json"

    def test_odd_started_at_never_breaks_the_history(self, tmp_path):
        (tmp_path / "jobs_ledger.json").write_text(json.dumps({"version": 1, "jobs": {
            "job-a": {"job_id": "job-a", "state": "completed", "started_at": 5.0},
            "job-b": {"job_id": "job-b", "state": "completed", "started_at": "2026-01-01T00:00:00"},
            "job-c": "not a record"}}), encoding="utf-8")
        assert [r["job_id"] for r in JobManager(output_dir=str(tmp_path)).history()] == ["job-a", "job-b"]


class TestRequests:

    def test_a_request_that_raises_answers_500_with_the_reason(self, ui_server, monkeypatch):
        def boom():
            raise RuntimeError("ledger exploded")
        monkeypatch.setattr(ui_server.job_manager, "history", boom)
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(_url(ui_server, "/api/history"), timeout=10)
        assert err.value.code == 500
        assert "ledger exploded" in json.loads(err.value.read().decode("utf-8"))["error"]

    def test_download_of_a_japanese_file_name(self, ui_server, tmp_path):
        f = tmp_path / "uploads" / "市場調査レポート.md"
        f.parent.mkdir(parents=True)
        f.write_text("本文", encoding="utf-8")
        from urllib.parse import quote
        with urllib.request.urlopen(_url(ui_server, f"/api/report-file?path={quote(str(f))}&download=1"),
                                    timeout=10) as res:
            assert res.status == 200
            assert "filename*=UTF-8''" + quote("市場調査レポート.md") in res.headers["Content-Disposition"]
            assert res.read().decode("utf-8") == "本文"


def test_console_gone_never_stops_a_run():
    from deep_research_tool.utils.helpers import _SafeTextStream

    class Gone:
        encoding = "utf-8"

        def write(self, text):
            raise BrokenPipeError(32, "Broken pipe")

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")
    stream = _SafeTextStream(Gone())
    assert stream.write("進捗 50%\n") == len("進捗 50%\n")
    stream.flush()


def test_windows_never_shares_a_port():
    assert srv._UIServer.allow_reuse_address == (os.name != "nt")
