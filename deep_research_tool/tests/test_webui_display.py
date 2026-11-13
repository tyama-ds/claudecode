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
from pathlib import Path
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
              # cited e3, and e3 disagrees with it
              SimpleNamespace(status="contradicted", citation_mismatch=True, supporting_source_ids=[],
                              evidence_provenance=[("e3", 0)]),
              # judged against chunks picked from the whole locker (an
              # executive summary): which one disagrees is not known
              SimpleNamespace(status="contradicted", citation_mismatch=False, supporting_source_ids=[],
                              evidence_provenance=[("e1", 0), ("e2", 0), ("e4", 0)])]
    runner = SimpleNamespace(claim_verifier=SimpleNamespace(last_claims=claims))
    support = FinalizationRunner._evidence_support(runner)
    assert support["e1"] == {"supported": 2, "judged": 3, "contradicted": 0}
    assert support["e2"] == {"supported": 0, "judged": 2, "contradicted": 0}
    assert support["e3"] == {"supported": 0, "judged": 1, "contradicted": 1}
    assert support["e4"] == {"supported": 0, "judged": 1, "contradicted": 0}


class _ScriptedLLM:
    def __init__(self, route):
        self.route = route

    def generate(self, prompt, **kwargs):
        out = self.route(prompt)
        return SimpleNamespace(content=out if isinstance(out, str) else json.dumps(out, ensure_ascii=False))


def test_a_contradiction_is_put_only_on_the_source_the_claim_cited():
    """Real ClaimVerifier: an executive-summary claim (judged against the
    whole locker) that is contradicted must not badge every source shown
    to the judge; a body claim whose cited source disagrees does."""
    import re
    from deep_research_tool.evidence.locker import Evidence
    from deep_research_tool.report.citations import CitationManager
    from deep_research_tool.report.finalization_runner import FinalizationRunner
    from deep_research_tool.verification.claim_verifier import ClaimVerifier
    backed = "A社の売上高は2025年に100億円である"
    wrong_cited = "A社の従業員数は2025年に9000人である"
    summary = "A社の売上高は2025年に500億円に達した"
    body = {"1": f"{backed} [SOURCE 1]。{wrong_cited} [SOURCE 2]。", "要旨": summary + "。"}

    def route(prompt):
        if "検証が必要な事実主張" in prompt:
            if summary in prompt:
                return {"claims": [{"claim": summary, "importance": "important", "source_numbers": []}]}
            return {"claims": [{"claim": backed, "importance": "important", "source_numbers": [1]},
                               {"claim": wrong_cited, "importance": "important", "source_numbers": [2]}]}
        verdicts = []
        for cid, text in re.findall(r"◆ 主張 (\S+?)（[^）]*）\s*\n?(.*)", prompt):
            if summary in text or (summary in prompt and len(re.findall("◆ 主張", prompt)) == 1):
                verdicts.append({"id": cid, "status": "contradicted", "reason": "120億円", "supporting_source_ids": []})
            elif wrong_cited in text:
                verdicts.append({"id": cid, "status": "contradicted", "reason": "3000人", "supporting_source_ids": []})
            else:
                verdicts.append({"id": cid, "status": "supported", "reason": "ok", "supporting_source_ids": ["E1"]})
        if verdicts:
            return {"verdicts": verdicts}
        if summary in prompt.split("【エビデンス】")[0]:
            return {"status": "contradicted", "reason": "120億円", "supporting_source_ids": []}
        if wrong_cited in prompt.split("【エビデンス】")[0]:
            return {"status": "contradicted", "reason": "3000人", "supporting_source_ids": []}
        return {"status": "supported", "reason": "ok", "supporting_source_ids": ["E1"]}

    ev = [Evidence(id="E1", extracted_text="A社の売上高は2025年に100億円である。"),
          Evidence(id="E2", extracted_text="A社の従業員数は2025年に3000人である。A社の売上高は2025年の決算で公表された。"),
          Evidence(id="E3", extracted_text="A社の売上高は2025年に120億円だった。")]
    mgr = CitationManager(lambda e: e in {"E1", "E2", "E3"})
    mgr.register_section("1", ["E1", "E2"])
    mgr.register_section("要旨", [])
    verifier = ClaimVerifier(_ScriptedLLM(route))
    verifier.verify_report(body, ev, mgr, citation_exempt_sections={"要旨"})
    by_text = {c.text: c for c in verifier.last_claims}
    assert by_text[summary].status == "contradicted"
    assert len({e for e, _ in by_text[summary].evidence_provenance}) >= 2      # several sources shown
    assert by_text[wrong_cited].status == "contradicted"
    support = FinalizationRunner._evidence_support(SimpleNamespace(claim_verifier=verifier))
    assert support["E1"]["supported"] == 1 and support["E1"]["contradicted"] == 0
    assert support["E2"]["contradicted"] == 1                                 # the cited source that disagrees
    assert support.get("E3", {}).get("contradicted", 0) == 0


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
        page.add_init_script("localStorage.setItem('drt_ui_mode', 'pro')")   # the engine cards are pro items

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
        # the text of a card stays inside it (the selected card's orbit
        # ring sticks out 1.5 px by design, so the card box is not measured)
        state = page.evaluate("""() => ({
            overflow: Array.from(document.querySelectorAll('#engineModes .engine-card')).some(c => {
                const r = c.getBoundingClientRect();
                return Array.from(c.querySelectorAll('.t, .d, .r, .more')).some(el => {
                    const e = el.getBoundingClientRect();
                    return e.right > r.right + 1 || el.scrollWidth > el.clientWidth + 1; }); }),
            sideways: document.documentElement.scrollWidth > window.innerWidth,
            reasonInCard: document.getElementById('engineModes').innerText.includes('python.exe')})""")
        assert state == {"overflow": False, "sideways": False, "reasonInCard": False}, state
        card = page.query_selector("#engineModes .engine-card[data-engine='gpt-researcher']")
        assert not card.query_selector(".tip").is_visible()
        card.scroll_into_view_if_needed()
        card.hover()
        page.wait_for_timeout(200)
        tip = card.query_selector(".tip")
        assert tip.is_visible() and "python.exe" in tip.inner_text()
        # the SELECTED card too (its ::after is the orbit ring, which stays)
        builtin = page.query_selector("#engineModes .engine-card.sel")
        builtin.hover()
        page.wait_for_timeout(200)
        assert builtin.query_selector(".tip").is_visible()
        ring = page.evaluate("getComputedStyle(document.querySelector('#engineModes .engine-card.sel'), '::after').display")
        assert ring == "block"
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
        # the same result arriving again (the 1.5 s poll) must not redraw it:
        # nothing in the tab changes (timing-free), and a layout after each
        # call would show a redraw's placeholder as a jump to the top
        page.evaluate("""() => {
            window.__muts = 0;
            new MutationObserver(m => { window.__muts += m.length; })
                .observe(document.getElementById('resultEvidence'), {childList: true, subtree: true});
            for (let i = 0; i < 3; i++) {
                showResults(historyCache[0].result, historyCache[0].job_id, true);
                void document.documentElement.scrollHeight;
            }
        }""")
        page.wait_for_timeout(1500)
        assert page.evaluate("window.__muts") == 0
        assert before > 0 and page.evaluate("window.scrollY") == before
        assert "✔ 検証" in page.inner_text("#resultEvidence")
    finally:
        browser.close()
        pw.stop()
        httpd.shutdown()
        httpd.server_close()


def _records(evidence_json):
    """url -> metadata.verification of an exported evidence file."""
    data = json.loads(Path(evidence_json).read_text(encoding="utf-8"))
    return {e["url"]: (e.get("metadata") or {}).get("verification") for e in data["evidence"]}


def test_a_verified_run_writes_each_sources_record_into_its_evidence_file(tmp_path):
    """The whole pipeline (fake LLM): verifier -> runner -> annotation ->
    export. Each link is needed for the 根拠 tab to show 検証."""
    from deep_research_tool.tests.test_finalization_integration import (
        SUPPORTED_EVIDENCE, CLAIM_SENTENCE, FakeSearch, build_tool, long_body, make_llm, make_locker,
        make_plan, make_session, run_tool)
    session = make_session(make_plan([("1", "市場シェア", "a")]), {"1": {
        "title": "市場シェア", "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
        "evidence": SUPPORTED_EVIDENCE}})
    tool = build_tool(tmp_path, make_llm(), FakeSearch())
    result = run_tool(tool, session, make_locker(tmp_path, SUPPORTED_EVIDENCE))
    assert result["status"]["verification"] == "performed"
    assert tool.finalization_outcome["evidence_support"]
    rec = _records(result["evidence_json"])[SUPPORTED_EVIDENCE[0][0]]
    assert rec and rec["supported"] >= 1 and rec["contradicted"] == 0


def test_a_failed_verification_leaves_no_record(tmp_path):
    """No claim could be extracted from a factual body: the check failed,
    so no source may read 'not checked / nobody cited it'."""
    from deep_research_tool.tests.test_finalization_integration import (
        SUPPORTED_EVIDENCE, CLAIM_SENTENCE, FakeSearch, RoutedLLM, all_answered_coverage, build_tool, j,
        long_body, make_locker, make_plan, make_session, run_tool)
    llm = RoutedLLM().on("extract", j({"claims": []})).on("coverage", all_answered_coverage)
    session = make_session(make_plan([("1", "市場シェア", "a")]), {"1": {
        "title": "市場シェア", "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
        "evidence": SUPPORTED_EVIDENCE}})
    locker = make_locker(tmp_path, SUPPORTED_EVIDENCE)
    for ev in locker.get_all_evidence():          # even one carried over is removed
        ev.metadata["verification"] = {"supported": 4, "judged": 4, "contradicted": 0}
    tool = build_tool(tmp_path, llm, FakeSearch())
    result = run_tool(tool, session, locker)
    assert result["status"]["verification"] == "failed"
    assert tool.finalization_outcome["evidence_support"] is None
    assert set(_records(result["evidence_json"]).values()) == {None}


@pytest.mark.parametrize("stop", ["cancel", "timeout"])
def test_a_verification_stopped_after_a_pass_leaves_no_record(tmp_path, stop):
    """One verification pass ran (the verifier holds its claims), then the
    run was cancelled / timed out: the stopped check has no record."""
    from deep_research_tool.report import finalization as fz
    from deep_research_tool.tests.test_finalization_integration import (
        SUPPORTED_EVIDENCE, CLAIM_SENTENCE, FakeSearch, build_tool, long_body, make_llm, make_locker,
        make_plan, make_session)
    from deep_research_tool.verification.runtime import VerificationCancelled, VerificationTimeout
    tool = build_tool(tmp_path, make_llm(), FakeSearch())
    tool.verification_progress = None
    session = make_session(make_plan([("1", "市場シェア", "a")]), {"1": {
        "title": "市場シェア", "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
        "evidence": SUPPORTED_EVIDENCE}})
    locker = make_locker(tmp_path, SUPPORTED_EVIDENCE)
    runner = tool._build_finalization_runner(session, locker, "q")
    error = VerificationCancelled if stop == "cancel" else VerificationTimeout

    def one_pass_then_stop(self, current):
        self.verify_fn(current)
        raise error()
    orig = fz.FinalizationController.run
    fz.FinalizationController.run = one_pass_then_stop
    try:
        outcome = runner.run({"1": long_body("1", "市場シェア", CLAIM_SENTENCE)})
    finally:
        fz.FinalizationController.run = orig
    assert outcome["decision"] == ("cancelled" if stop == "cancel" else "timeout")
    assert runner.claim_verifier.last_claims                    # a pass did run
    assert outcome["evidence_support"] is None
    tool.finalization_outcome = outcome
    tool._annotate_evidence(locker)
    assert all("verification" not in ev.metadata for ev in locker.get_all_evidence())


def test_a_resumed_run_gets_its_record_back(tmp_path):
    """Resumed after the verification finished: no verification call, the
    record comes back from final_body_<id>.json."""
    from deep_research_tool.tests.test_resume_stages import _first_run, _resume
    tool, fake, result, _llm = _first_run(tmp_path)
    url = next(iter(_records(result["evidence_json"])))
    assert _records(result["evidence_json"])[url]["supported"] >= 1
    fake.locker.drop_verification_records()       # nothing left in memory either
    tool2, res2, llm2 = _resume(tmp_path, fake, result["session_id"])
    assert not llm2.prompts("judge")
    assert tool2.finalization_outcome.get("restored") is True
    assert _records(res2["evidence_json"])[url]["supported"] >= 1


def test_a_resumed_run_keeps_a_failed_verification_failed(tmp_path):
    """The check failed (no claims), then the run was resumed with the
    verification stage reused: still 'failed', still no record."""
    from deep_research_tool.tests.test_finalization_integration import RoutedLLM, all_answered_coverage, j
    from deep_research_tool.tests.test_resume_stages import _first_run, _resume

    def failing():
        return RoutedLLM().on("extract", j({"claims": []})).on("coverage", all_answered_coverage)
    tool, fake, result, _llm = _first_run(tmp_path, failing())
    assert result["status"]["verification"] == "failed"
    tool2, res2, llm2 = _resume(tmp_path, fake, result["session_id"], failing())
    assert tool2.finalization_outcome.get("restored") is True and not llm2.prompts("extract")
    assert res2["status"]["verification"] == "failed" and res2["status"]["quality"] == "unverified"
    assert set(_records(res2["evidence_json"]).values()) == {None}


def _figures_run(tmp_path, fmt, missing_png=False, insertion_fails=False):
    """run() with a frozen figure collection of one chart and one table."""
    pytest.importorskip("reportlab")
    from unittest.mock import patch
    from PIL import Image
    from deep_research_tool.main import DeepResearchTool
    from deep_research_tool.report.figure_table_generator import (
        Figure, FigureTableCollection, FigureTableGenerator, FigureType, TableData)
    from deep_research_tool.tests import test_finalization_integration as T
    from deep_research_tool.utils.helpers import ResearchWarnings
    ResearchWarnings.reset()
    png = tmp_path / "figs" / "chart.png"
    png.parent.mkdir(parents=True)
    if not missing_png:
        Image.new("RGB", (800, 500), "white").save(png)
    coll = FigureTableCollection(
        charts=[Figure(figure_id="c1", figure_type=FigureType.CHART, title="シェアの推移",
                       caption="A社のシェアの推移を示す図。出所: 統計", section_id="1", image_path=png)],
        tables=[TableData(table_id="t1", title="主要企業のシェア", caption="出所: 各社IR",
                          headers=["社名", "シェア"], rows=[["A社", "52%"], ["B社", "0"]], section_id="1")])
    llm = T.make_llm()
    tool = T.build_tool(tmp_path, llm, T.FakeSearch(), auto_figures=True, output_format=fmt,
                        report_generator_version="v1")
    gen = FigureTableGenerator(llm_client=llm, output_dir=tmp_path / "figs")
    base = T.Base()
    base.tmp = tmp_path

    def boom(self, **kw):
        raise RuntimeError("insertion down")
    with patch.object(DeepResearchTool, "_generate_figure_collection", lambda self, **kw: (coll, gen)):
        if insertion_fails:
            with patch.object(DeepResearchTool, "_insert_figures_into_report", boom):
                return T.run_tool(tool, base.default_session(), T.make_locker(tmp_path, T.SUPPORTED_EVIDENCE))
        return T.run_tool(tool, base.default_session(), T.make_locker(tmp_path, T.SUPPORTED_EVIDENCE))


def test_figure_counts_say_how_many_reached_the_report(tmp_path):
    # the PDF renderer places the table but not the chart whose image is missing
    res = _figures_run(tmp_path / "pdf", "pdf", missing_png=True)
    assert res["figures"] == {"charts": 1, "tables": 1, "images": 0, "placed": 1, "total": 2}
    # the post-render insertion failed: the report has none of them
    res = _figures_run(tmp_path / "md-fail", "markdown", insertion_fails=True)
    assert res["figures"] == {"charts": 1, "tables": 1, "images": 0, "placed": 0, "total": 2}
    # inserted after rendering: generated counts only (placement not measured there)
    res = _figures_run(tmp_path / "md", "markdown")
    assert res["figures"] == {"charts": 1, "tables": 1, "images": 0}


def test_a_feedback_re_research_does_not_start_with_the_parents_record(tmp_path):
    from deep_research_tool.revision import seed_locker
    parent = EvidenceLocker(research_id="parent", output_dir=tmp_path / "parent")
    ev = parent.add_evidence(url="https://example.com/a", title="親の出典", content_excerpt="x")
    ev.metadata["verification"] = {"supported": 3, "judged": 3, "contradicted": 0}
    ev.metadata["date_provenance"] = {"kept": True}
    path = parent.export_to_json()
    child = seed_locker(path, "child", tmp_path / "child")
    (cev,) = child.get_all_evidence()
    assert "verification" not in cev.metadata and cev.metadata.get("date_provenance") == {"kept": True}
    # the child's first checkpoint (what a cancelled child would show)
    assert set(_records(child.export_to_json()).values()) == {None}


def test_a_verify_only_run_does_not_start_with_the_loaded_runs_record(tmp_path):
    from deep_research_tool.main import DeepResearchTool
    from deep_research_tool.tests.test_finalization_integration import FakeSearch, build_tool, make_llm
    from deep_research_tool.tests.test_verify_only import _saved_research
    session_path = _saved_research(tmp_path)
    ev_path = tmp_path / "evidence" / "evidence_itest.json"
    data = json.loads(ev_path.read_text(encoding="utf-8"))
    for e in data["evidence"]:
        e.setdefault("metadata", {})["verification"] = {"supported": 9, "judged": 9, "contradicted": 0}
    ev_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    seen = []
    orig = DeepResearchTool._build_finalization_runner

    def spy(self, session, locker, *a, **kw):
        seen.append([dict(ev.metadata).get("verification") for ev in locker.get_all_evidence()])
        return orig(self, session, locker, *a, **kw)
    DeepResearchTool._build_finalization_runner = spy
    try:
        tool = build_tool(tmp_path / "verify-job", make_llm(), FakeSearch())
        result = tool.verify_report(session_path=str(session_path))
    finally:
        DeepResearchTool._build_finalization_runner = orig
    assert seen and seen[0] == [None]                  # the loaded run's 9s are gone before verifying
    assert result["status"]["verification"] == "performed"
    (rec,) = _records(result["evidence_json"]).values()
    assert rec and rec["supported"] != 9               # this run's own record


def test_the_server_keeps_the_figure_counts_for_the_history():
    from deep_research_tool.webui.server import ResearchJob
    job = ResearchJob("job-f", "q", {"query": "q"})
    figures = {"charts": 5, "tables": 2, "images": 1, "placed": 6, "total": 8}
    JobManager._finish_from_result(job, {}, {"report_path": "r.md", "figures": figures,
                                             "run_status": "completed"},
                                   SimpleNamespace(to_dict_list=lambda: [], count=lambda: 0))
    assert job.result["figures"] == figures


@pytest.mark.parametrize("ps,res,want,absent", [
    ({"verify": True, "run_mode": "research_only", "verification_profile": "balanced"},
     {"kind": "verify", "status": {"verification": "performed"}}, ["検証のみ", "Balanced"], ["なし（調査のみ）", "調査の深さ"]),
    ({"query": "q", "regenerate": {"output_format": "docx"}}, {"regenerated_from": "final_body"},
     ["再出力", "Word"], ["label:検証"]),
    ({"enable_verification": False}, {"kind": "fermi"}, ["フェルミ推定"], ["label:検証"]),
    ({"research_preset": "standard", "verification_profile": "strict"}, {"status": {"verification": "cancelled"}},
     ["中止"], ["Strict"]),
    # a verification that failed / a run that errored: Japanese, never the raw value
    ({"research_preset": "standard"}, {"status": {"verification": "failed"}}, ["実行できず"], ["failed"]),
    ({"verify": True, "verification_profile": "balanced"}, {"kind": "verify", "status": {"verification": "unknown"}},
     ["Balanced"], ["unknown"]),
    # an external engine: its own options, not the built-in loop's rounds / crawl
    ({"research_engine": "gpt-researcher", "research_preset": "standard", "iterations": 2, "max_iterations": 4,
      "crawl_mode": "standard", "source_mode": "web",
      "engine_options": {"max_iterations": 5, "report_type": "detailed_report"}}, {},
     ["GPT Researcher", "検索イテレーション数5", "レポートの種類detailed_report"], ["label:調査回数/章", "label:クロール"]),
    # figures: the generated counts and how many did not reach the body
    ({"auto_figures": True}, {"figures": {"charts": 5, "tables": 2, "images": 1, "placed": 6, "total": 8}},
     ["生成: グラフ 5・表 2・画像 1", "本文に入らなかったもの 2 件"], ["実際"]),
    ({"auto_figures": True}, {"figures": {"charts": 3, "tables": 0, "images": 0, "placed": 3, "total": 3}},
     ["生成: グラフ 3・表 0・画像 0"], ["本文に入らなかった"]),
])
def test_history_chips_follow_the_run_kind(tmp_path, ps, res, want, absent):
    (tmp_path / "jobs_ledger.json").write_text(json.dumps({"version": 1, "jobs": {"job-x": {
        "job_id": "job-x", "query": "q", "state": "completed", "started_at": 1.0,
        "params_summary": ps, "result": res}}}), encoding="utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = JobManager(output_dir=str(tmp_path))
    httpd.output_dir = str(tmp_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(httpd, "/#history"), None)
        text = page.inner_text("#historyList .hist-params")
        labels = page.eval_on_selector_all("#historyList .hist-params .pchip i", "els => els.map(e => e.textContent)")
        for w in want:
            assert w in text, (w, text)
        for a in absent:
            if a.startswith("label:"):
                assert a[6:] not in labels, (a, labels)
            else:
                assert a not in text, (a, text)
        assert errors == []
    finally:
        browser.close()
        pw.stop()
        httpd.shutdown()
        httpd.server_close()


def _engines_route(ui_server, mocks):
    def route(r):
        req = urllib.request.Request(_url(ui_server, "/api/engines"), method="POST",
                                     data=json.dumps({"mocks": mocks}).encode(),
                                     headers={"Content-Type": "application/json"})
        data = json.loads(urllib.request.urlopen(req, timeout=30).read())
        for e in data["engines"]:
            if e["id"] != "builtin":     # long text in every tip
                e["description"] = (e.get("description") or "") + " " + LONG_REASON
        r.fulfill(status=200, content_type="application/json", body=json.dumps(data))
    return route


def test_engine_tip_closes_after_a_click_and_opens_on_keyboard_focus(ui_server):
    pw, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.add_init_script("localStorage.setItem('drt_ui_mode', 'pro')")   # the engine cards are pro items
        page.route("**/api/engines*", _engines_route(ui_server, False))
        page.goto(_url(ui_server, "/#new"), wait_until="networkidle")
        page.wait_for_timeout(800)
        card = page.query_selector("#engineModes .engine-card[data-engine='builtin']")
        card.scroll_into_view_if_needed()
        card.click()
        page.mouse.move(5, 5)
        page.wait_for_timeout(200)
        # the radio is focused by the click, but the tip does not stay over the next card
        assert page.evaluate("document.activeElement.name") == "research_engine"
        assert not card.query_selector(".tip").is_visible()
        # keyboard: Tab onto an engine card shows its tip
        page.evaluate("document.activeElement.blur(); document.getElementById('engineModes').scrollIntoView()")
        found = False
        page.keyboard.press("Shift+Tab")
        for _ in range(80):
            page.keyboard.press("Tab")
            inside = page.evaluate("""() => { const c = document.activeElement.closest('.engine-card');
                return c ? c.dataset.engine : null; }""")
            if inside:
                found = True
                break
        assert found
        assert page.query_selector(f"#engineModes .engine-card[data-engine='{inside}'] .tip").is_visible()
        # screen readers get the tip as the radio's description
        assert page.evaluate("""() => { const i = document.querySelector('#engineModes input[value="gpt-researcher"]');
            const t = document.getElementById(i.getAttribute('aria-describedby')); return !!t && t.textContent.includes('python.exe'); }""")
    finally:
        browser.close()
        pw.stop()


@pytest.mark.parametrize("width", [700, 1280, 390])
def test_engine_tips_stay_inside_the_window_with_many_engines(ui_server, width):
    """More than three engines (auto-fill grid): any card can be at an
    edge, not only the last one."""
    pw, browser = _browser()
    try:
        page = browser.new_page(viewport={"width": width, "height": 900})
        page.add_init_script("localStorage.setItem('drt_ui_mode', 'pro')")   # the engine cards are pro items
        page.route("**/api/engines*", _engines_route(ui_server, True))
        page.goto(_url(ui_server, "/#new"), wait_until="networkidle")
        page.wait_for_timeout(800)
        ids = page.eval_on_selector_all("#engineModes .engine-card", "els => els.map(e => e.dataset.engine)")
        assert len(ids) > 3 and page.evaluate("document.getElementById('engineModes').classList.contains('mode-cards-auto')")
        for eid in ids:
            card = page.query_selector(f"#engineModes .engine-card[data-engine='{eid}']")
            card.scroll_into_view_if_needed()
            card.hover()
            page.wait_for_timeout(120)
            box = page.evaluate(f"""() => {{ const t = document.querySelector('#engineModes .engine-card[data-engine="{eid}"] .tip');
                const r = t.getBoundingClientRect();
                return {{left: r.left, right: r.right, vw: document.documentElement.clientWidth,
                         sideways: document.documentElement.scrollWidth > window.innerWidth}}; }}""")
            assert box["left"] >= 0 and box["right"] <= box["vw"] and not box["sideways"], (eid, box)
    finally:
        browser.close()
        pw.stop()


def test_badges_follow_the_runs_own_verification(tmp_path):
    """Both counts show; a run whose verification did not complete shows no
    record at all (one there could only be another run's)."""
    _past_job(tmp_path)
    ev = json.loads((tmp_path / "job-old" / "evidence" / "evidence_x.json").read_text(encoding="utf-8"))
    ev["evidence"].insert(0, {"id": "e0", "url": "https://www.stat.go.jp/c", "title": "両方", "quality_category": "authoritative",
                              "metadata": {"verification": {"supported": 2, "judged": 3, "contradicted": 1}}})
    ev["section_evidence"]["1"].insert(0, "e0")
    (tmp_path / "job-old" / "evidence" / "evidence_x.json").write_text(json.dumps(ev), encoding="utf-8")
    ledger = json.loads((tmp_path / "jobs_ledger.json").read_text(encoding="utf-8"))
    failed = json.loads(json.dumps(ledger["jobs"]["job-old"]))
    failed.update(job_id="job-fail", started_at=0.5)
    failed["result"]["status"] = {"verification": "failed"}
    ledger["jobs"]["job-fail"] = failed
    (tmp_path / "jobs_ledger.json").write_text(json.dumps(ledger), encoding="utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), WebUIHandler)
    httpd.job_manager = JobManager(output_dir=str(tmp_path))
    httpd.output_dir = str(tmp_path)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    pw, browser = _browser()
    try:
        page, errors, _ = _load(browser, _url(httpd, "/#history"), None)
        page.evaluate("showResults(historyCache[0].result, historyCache[0].job_id, true); setView('result'); selectResultTab('evidence')")
        page.wait_for_timeout(800)
        text = page.inner_text("#resultEvidence")
        row = page.evaluate("""() => Array.from(document.querySelectorAll('#resultEvidence .warn-row'))
            .find(r => r.innerText.includes('両方')).innerText""")
        assert "✔ 検証: 主張 2 件を裏付け" in row and "⚠ 検証: 主張 1 件と食い違い" in row     # one row, both counts
        assert "食い違うと判定された出典: 1 件" in text
        # the same files under a cancelled verification
        page.evaluate("""async () => { const r = JSON.parse(JSON.stringify(historyCache[0].result));
            r.status = {verification: 'cancelled'}; await loadResultEvidence(r); }""")
        page.wait_for_timeout(300)
        text = page.inner_text("#resultEvidence")
        assert "✔ 検証" not in text and "⚠ 検証" not in text and "検証が最後まで行えなかった" in text
        # 根拠ロッカー: both badges; none under a failed verification
        page2 = browser.new_page()
        page2.goto(_url(httpd, "/evidence.html?job_id=job-old"), wait_until="networkidle")
        page2.wait_for_timeout(600)
        body = page2.inner_text("body")
        assert "✔ 裏付け 2" in body and "⚠ 食い違い 1" in body
        page2.goto(_url(httpd, "/evidence.html?job_id=job-fail"), wait_until="networkidle")
        page2.wait_for_timeout(600)
        body = page2.inner_text("body")
        assert "両方" in body and "✔ 裏付け" not in body and "⚠ 食い違い" not in body
        assert errors == []
    finally:
        browser.close()
        pw.stop()
        httpd.shutdown()
        httpd.server_close()


def test_download_name_never_breaks_the_header(ui_server, tmp_path):
    f = tmp_path / "up" / 'a"b;c.md'
    f.parent.mkdir(parents=True)
    f.write_text("x", encoding="utf-8")
    with urllib.request.urlopen(_url(ui_server, f"/api/report-file?path={quote(str(f))}&download=1"), timeout=10) as res:
        cd = res.headers["Content-Disposition"]
        assert cd.startswith('attachment; filename="a_b_c.md"') and "filename*=UTF-8''a%22b%3Bc.md" in cd
