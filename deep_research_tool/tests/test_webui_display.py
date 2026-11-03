"""
Web UI display fixes (v1.11.2):
- engine cards never overflow; the long reason / setup command is a tooltip
- 開く saves a file under its own name (a Word report lost its extension)
- the 結果 view is not redrawn every poll (the 根拠 scroll jumped to the top)
- each source says how the verification used it, and its reliability is
  rated by its domain (every source showed 'unverified' after a verification)
- the history shows what each run was set to (depth, length, figures…)
"""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import quote

import pytest

from deep_research_tool.evidence.locker import EvidenceLocker, QualityCategory
from deep_research_tool.tests.test_webui_state import _browser, _load, _url
from deep_research_tool.webui.server import JobManager, WebUIHandler


@pytest.fixture
def ui_server(tmp_path):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = JobManager(output_dir=str(tmp_path))
    httpd.output_dir = str(tmp_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

def test_open_names_the_file_with_its_extension(ui_server, tmp_path):
    docx = tmp_path / "job-1" / "reports" / "research_report_ab12.docx"
    docx.parent.mkdir(parents=True)
    docx.write_bytes(b"PK\x03\x04 fake docx")
    with urllib.request.urlopen(_url(ui_server, f"/api/report-file?path={quote(str(docx))}"), timeout=10) as res:
        assert res.headers["Content-Type"].startswith(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        assert 'inline; filename="research_report_ab12.docx"' in res.headers["Content-Disposition"]
    pdf = docx.with_suffix(".pdf")
    pdf.write_bytes(b"%PDF-1.4")
    with urllib.request.urlopen(_url(ui_server, f"/api/report-file?path={quote(str(pdf))}"), timeout=10) as res:
        assert res.headers["Content-Type"] == "application/pdf"
    with urllib.request.urlopen(_url(ui_server, "/"), timeout=10) as res:          # the page itself
        assert res.headers.get("Content-Disposition") is None


# ---------------------------------------------------------------------------
# evidence: verification use and reliability
# ---------------------------------------------------------------------------

def test_runner_summarizes_how_the_verification_used_each_source():
    from deep_research_tool.report.finalization_runner import FinalizationRunner
    claims = [SimpleNamespace(status="supported", supporting_source_ids=["e1"], evidence_provenance=[("e1", 0), ("e2", 5)]),
              SimpleNamespace(status="supported", supporting_source_ids=["e1"], evidence_provenance=[("e1", 9)]),
              SimpleNamespace(status="contradicted", supporting_source_ids=[], evidence_provenance=[("e3", 0)])]
    runner = SimpleNamespace(claim_verifier=SimpleNamespace(last_claims=claims))
    support = FinalizationRunner._evidence_support(runner)
    assert support["e1"] == {"supported": 2, "judged": 2, "contradicted": 0}
    assert support["e2"] == {"supported": 0, "judged": 1, "contradicted": 0}
    assert support["e3"] == {"supported": 0, "judged": 1, "contradicted": 1}


def test_sources_are_rated_by_domain_and_carry_the_verification(tmp_path):
    from deep_research_tool.main import DeepResearchTool
    locker = EvidenceLocker(research_id="t", output_dir=tmp_path)
    gov = locker.add_evidence(url="https://www.meti.go.jp/statistics/a.html", title="商業動態統計",
                              content_excerpt="短い抜粋")
    news = locker.add_evidence(url="https://www.nikkei.com/article/x", title="記事", content_excerpt="短い抜粋")
    blog = locker.add_evidence(url="https://example.com/a", title="ブログ", content_excerpt="短い抜粋")
    tool = SimpleNamespace(config=SimpleNamespace(research=SimpleNamespace(language="ja")),
                           finalization_outcome={"evidence_support": {gov.id: {"supported": 2, "judged": 2, "contradicted": 0}}})
    DeepResearchTool._annotate_evidence(tool, locker)
    assert gov.quality_category == QualityCategory.AUTHORITATIVE           # not 'low' from a short excerpt
    assert news.quality_category == QualityCategory.HIGH
    assert blog.quality_category == QualityCategory.UNVERIFIED             # unknown domain: 未評価
    assert gov.metadata["verification"] == {"supported": 2, "judged": 2, "contradicted": 0}
    assert blog.metadata["verification"] == {"supported": 0, "judged": 0, "contradicted": 0}
    exported = json.loads(locker.export_to_json().read_text(encoding="utf-8"))
    by_url = {e["url"]: e for e in exported["evidence"]}
    assert by_url[gov.url]["metadata"]["verification"]["supported"] == 2
    # without a verification nothing is claimed about it
    tool2 = SimpleNamespace(config=tool.config, finalization_outcome=None)
    locker2 = EvidenceLocker(research_id="t2", output_dir=tmp_path / "2")
    ev = locker2.add_evidence(url="https://www.stat.go.jp/x", title="統計", content_excerpt="x")
    DeepResearchTool._annotate_evidence(tool2, locker2)
    assert "verification" not in ev.metadata and ev.quality_category == QualityCategory.AUTHORITATIVE


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

LONG_REASON = (r"設定された Python（C:\Users\someone\Documents\claudecode\engines_venv\gpt-researcher\Scripts"
               r"\python.exe）が見つかりません。Python 3.12 以上で `py -3.12 deep_research_tool/scripts/"
               r"setup_engine_venv.py gpt-researcher` を実行してください")


@pytest.mark.parametrize("width", [1280, 390])
def test_engine_cards_never_overflow_and_explain_on_hover(ui_server, width):
    pw, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": width, "height": 900})

        def engines(route):
            data = json.loads(urllib.request.urlopen(_url(ui_server, "/api/engines")).read())
            for e in data["engines"]:
                if e["id"] != "builtin":
                    e.update(health="needs_setup", reason=LONG_REASON,
                             setup_command="py -3.12 deep_research_tool/scripts/setup_engine_venv.py " + e["id"])
            route.fulfill(status=200, content_type="application/json", body=json.dumps(data))
        page.route("**/api/engines*", engines)
        page.goto(_url(ui_server, "/#new"), wait_until="networkidle")
        page.wait_for_timeout(800)
        state = page.evaluate("""() => ({
            overflow: Array.from(document.querySelectorAll('#engineModes .engine-card'))
                .some(c => c.scrollWidth > c.clientWidth + 1),
            sideways: document.documentElement.scrollWidth > window.innerWidth,
            reasonInCard: document.getElementById('engineModes').innerText.includes('python.exe')})""")
        assert state == {"overflow": False, "sideways": False, "reasonInCard": False}, state
        card = page.query_selector("#engineModes .engine-card[data-engine='gpt-researcher']")
        assert "python.exe" in card.get_attribute("data-tip")
        card.scroll_into_view_if_needed()
        card.hover()
        page.wait_for_timeout(200)
        tip = page.evaluate("getComputedStyle(document.querySelector(\"#engineModes .engine-card[data-engine='gpt-researcher']\"), '::after')")
        assert tip is not None
        shown = page.evaluate("getComputedStyle(document.querySelector(\"#engineModes .engine-card[data-engine='gpt-researcher']\"), '::after').display")
        assert shown == "block"
    finally:
        browser.close()
        pw.stop()


def _past_job(tmp_path):
    out = tmp_path / "job-old"
    (out / "evidence").mkdir(parents=True)
    ev = out / "evidence" / "evidence_x.json"
    ev.write_text(json.dumps({"evidence": [
        {"id": "e1", "url": "https://www.meti.go.jp/a", "title": "商業動態統計", "quality_category": "authoritative",
         "metadata": {"verification": {"supported": 2, "judged": 2, "contradicted": 0}}},
        {"id": "e2", "url": "https://example.com/b", "title": "ブログ", "quality_category": "unverified",
         "metadata": {"verification": {"supported": 0, "judged": 0, "contradicted": 0}}}] +
        [{"id": f"x{i}", "url": f"https://example.com/{i}", "title": f"資料{i}", "quality_category": "unverified"}
         for i in range(40)],
        "section_evidence": {"1": ["e1", "e2"] + [f"x{i}" for i in range(40)]}}), encoding="utf-8")
    (tmp_path / "jobs_ledger.json").write_text(json.dumps({"version": 1, "jobs": {"job-old": {
        "job_id": "job-old", "query": "炭素繊維の市場調査", "state": "completed", "started_at": 1.0,
        "params_summary": {"research_preset": "thorough", "deliverable": "detailed", "iterations": 3,
                           "max_iterations": 6, "length_mode": "adaptive", "preferred_body_chars": 20000,
                           "auto_figures": True, "chart_max_per_section": 6, "chart_style": "enhanced",
                           "verification_profile": "strict", "source_mode": "web", "crawl_mode": "standard",
                           "output_format": "docx", "report_version": "v1", "language": "ja",
                           "deep_think": True, "provider": "openai", "model": "gpt-x"},
        "result": {"evidence_json": str(ev), "status": {"verification": "performed"},
                   "figures": {"charts": 5, "tables": 2, "images": 1}}}}}), encoding="utf-8")


def test_history_shows_what_the_run_emphasised(tmp_path):
    _past_job(tmp_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = JobManager(output_dir=str(tmp_path))
    httpd.output_dir = str(tmp_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(httpd, "/#history"), None)
        text = page.inner_text("#historyList .hist-params")
        for want in ("調査の深さ", "詳細", "仕上がり", "詳細調査", "約20,000字", "グラフ最大 6/章", "改良版",
                     "グラフ 5・表 2・画像 1", "Strict", "Word・V1", "DeepThink", "gpt-x"):
            assert want in text, (want, text)
        assert errors == []
    finally:
        browser.close()
        pw.stop()
        httpd.shutdown()
        httpd.server_close()


def test_evidence_tab_keeps_its_scroll_and_explains_each_source(tmp_path):
    _past_job(tmp_path)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = JobManager(output_dir=str(tmp_path))
    httpd.output_dir = str(tmp_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    pw, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 600})
        page.goto(_url(httpd, "/#history"), wait_until="networkidle")
        page.wait_for_timeout(600)
        page.evaluate("showResults(historyCache[0].result, historyCache[0].job_id, true); setView('result'); selectResultTab('evidence')")
        page.wait_for_timeout(800)
        text = page.inner_text("#resultEvidence")
        assert "✔ 検証: 主張 2 件を裏付け" in text and "出典の信頼度: 公的・学術" in text
        assert "出典の信頼度: 未評価" in text and "unverified" not in text
        page.evaluate("window.scrollTo(0, 500)")
        before = page.evaluate("window.scrollY")
        # the same result arriving again (the 1.5 s poll) must not redraw it
        page.evaluate("for (let i = 0; i < 3; i++) showResults(historyCache[0].result, historyCache[0].job_id, true)")
        page.wait_for_timeout(1500)
        assert before > 0 and page.evaluate("window.scrollY") == before
        assert "✔ 検証" in page.inner_text("#resultEvidence")
    finally:
        browser.close()
        pw.stop()
        httpd.shutdown()
        httpd.server_close()
