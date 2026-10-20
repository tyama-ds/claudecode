"""Pre-merge review fixes for v1.5.0 / v1.6.0 (chartkit, parallel
sections, run modes, resume, retries). Each test pins one defect the
review found so it cannot come back. No network, no real LLM."""
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_research_tool.tests.test_finalization_integration import (
    CLAIM_SENTENCE, SUPPORTED_EVIDENCE, FakeSearch, build_tool, long_body,
    make_llm, make_locker, make_plan, make_session)
from deep_research_tool.tests.test_resume_stages import _ResumableResearcher
from deep_research_tool.utils.run_state import RunState


# ----------------------------------------------------------------------
# chartkit
# ----------------------------------------------------------------------

def _dp(subject, metric, year, value, unit, url="https://example.com/a", **kw):
    from deep_research_tool.evidence.numerical_extractor import NumericalDataPoint
    kw.setdefault("raw_text", f"{value}{unit}")
    return NumericalDataPoint(value=value, unit=unit, metric_name=metric, subject=subject,
                              year=year, source_url=url, source_title="t", **kw)


def _data():
    return {"years": ["2023", "2024", "2025"], "other": ["2022", "2023", "2024"],
            "ents": ["toray", "teijin"],
            "a": {"unit": "億円", "labels_ref": "years", "values": [50, 60, 70], "basis": ["公表"] * 3},
            "b": {"unit": "億円", "labels_ref": "other", "values": [100, 110, 120], "basis": ["公表"] * 3},
            "pa": {"unit": "%", "labels_ref": "ents", "values": [60, 30], "basis": ["公表", "公表"]},
            "pc": {"unit": "%", "labels_ref": "ents", "values": [None, 25], "basis": [None, "公表"]}}


def _panel(**kw):
    p = {"id": "p1", "form": "bar", "title": "売上高（億円）", "labels_ref": "years",
         "series": [{"name": "A", "values_ref": "a", "role": "emphasis"}],
         "unit": "億円", "caption": "c", "judgment": "j"}
    p.update(kw)
    return p


class TestChartkitSpec:
    def test_series_on_another_label_array_is_rejected(self):
        from deep_research_tool.chartkit import validate_panels
        p = _panel(series=[{"name": "A", "values_ref": "a", "role": "context"},
                           {"name": "B", "values_ref": "b", "role": "emphasis"}])
        valid, rejected = validate_panels([p], _data())
        assert not valid and "labels_ref" in rejected[0][1]

    def test_shared_label_arrays_get_one_labels_ref(self):
        from deep_research_tool.chartkit import build_section_catalog
        pts = [_dp("Toray", "売上高", y, v, "億円", section_id="1") for y, v in ((2023, 100), (2024, 110), (2025, 120))]
        pts += [_dp("Teijin", "売上高", y, v, "億円", section_id="1") for y, v in ((2023, 50), (2024, 60), (2025, 70))]
        cat = build_section_catalog(pts, "1")
        refs = {s.labels_ref for s in cat.series.values() if s.kind == "time"}
        assert len(refs) == 1                       # both series share the year array
        data = cat.to_data()
        assert all("labels_ref" in v for k, v in data.items() if isinstance(v, dict) and k != "_meta")

    def test_unit_comes_from_the_data_and_y_max_never_clips(self):
        from deep_research_tool.chartkit import validate_panels
        valid, rejected = validate_panels([_panel(unit="%", y_max=60)], _data())
        assert valid and not rejected
        assert valid[0]["unit"] == "億円" and valid[0]["series"][0]["unit"] == "億円"
        assert valid[0]["y_max"] == 70.0            # raised to the data max
        valid2, _ = validate_panels([_panel(y_max="big", height="300px")], _data())
        assert "y_max" not in valid2[0] and valid2[0]["height"] == 300

    def test_duplicate_ids_are_rejected_and_main_suffixes_them(self):
        from deep_research_tool.chartkit import validate_panels
        p2 = _panel(form="hbar", series=[{"name": "A", "values_ref": "a", "role": "context"}])
        valid, rejected = validate_panels([_panel(), p2], _data())
        assert [p["id"] for p in valid] == ["p1"] and "重複" in rejected[0][1]

    def test_line_xy_plots_numeric_labels(self):
        from deep_research_tool.chartkit import validate_panels
        valid, rejected = validate_panels([_panel(form="line_xy")], _data())
        assert valid, rejected
        assert valid[0]["series"][0]["values"] == [{"x": 2023.0, "y": 50.0}, {"x": 2024.0, "y": 60.0},
                                                   {"x": 2025.0, "y": 70.0}]
        bad, rej = validate_panels([_panel(form="line_xy", labels_ref="ents",
                                           series=[{"name": "A", "values_ref": "pa", "role": "emphasis"}])],
                                   _data())
        assert not bad and "line_xy" in rej[0][1]

    def test_stacked_hbar_draws_one_row_per_label(self, tmp_path):
        from deep_research_tool.chartkit import render_png, validate_panels
        pytest.importorskip("matplotlib")
        p = _panel(id="comp", form="stacked_hbar", labels_ref="ents", unit="%",
                   series=[{"name": "A", "values_ref": "pa", "role": "part"},
                           {"name": "C", "values_ref": "pc", "role": "part"}])
        valid, rejected = validate_panels([p], _data())
        assert valid, rejected
        out = render_png(valid[0], tmp_path / "comp.png")
        assert Path(out).stat().st_size > 1000
        # the stack total per label, not a composition of different labels
        from deep_research_tool.chartkit.spec import _data_max
        assert _data_max(valid[0]) == 60.0

    def test_html_page_escapes_script_breakers_and_placeholders(self):
        from deep_research_tool.chartkit import render_html, validate_panels
        valid, _ = validate_panels([_panel(title="</script><!--{{SPEC}}")], _data())
        page = render_html(valid, title="{{PANELS}}")
        assert "</script><!--" not in page                 # neither in the JSON nor the HTML
        # placeholder-like text stays literal (HTML-escaped), it is never
        # substituted by a later replacement
        assert "&lt;!--{{SPEC}}" in page and "<h1>{{PANELS}}" in page.replace("\n", "")[:len(page)] or "{{PANELS}}" in page
        assert page.count("\\u003c/script") == 1            # the JSON escape, once (the title)

    def test_primary_domain_needs_a_label_boundary(self):
        from deep_research_tool.chartkit.data_layer import (BASIS_PUBLISHED, BASIS_SECONDARY,
                                                            classify_source_basis)
        assert classify_source_basis("https://www.oecd.org/x", ["oecd.org"]) == BASIS_PUBLISHED
        assert classify_source_basis("https://notoecd.org/x", ["oecd.org"]) == BASIS_SECONDARY


# ----------------------------------------------------------------------
# parallel sections
# ----------------------------------------------------------------------

class TestParallelSections:
    def test_locker_importance_update_is_safe_during_inserts(self, tmp_path):
        from deep_research_tool.evidence.locker import EvidenceLocker, EvidenceType
        locker = EvidenceLocker(research_id="t", output_dir=tmp_path)
        for i in range(200):
            locker.add_evidence(url=f"https://a.example/{i}", title=f"t{i}", content_excerpt=f"c{i}",
                                extracted_text=f"x{i}", evidence_type=EvidenceType.WEB_PAGE,
                                section_reference="1")
        errors, stop = [], threading.Event()

        def adder():
            i = 0
            while not stop.is_set():
                locker.add_evidence(url=f"https://b.example/{i}", title="x", content_excerpt=f"o{i}",
                                    extracted_text=f"y{i}", evidence_type=EvidenceType.WEB_PAGE,
                                    section_reference="2")
                i += 1

        def scorer():
            try:
                for _ in range(800):
                    locker.update_importance_by_url("https://a.example/199", 0.5, section="1")
                    locker.get_by_url("https://a.example/5")
            except Exception as e:
                errors.append(repr(e))
            finally:
                stop.set()
        t1, t2 = threading.Thread(target=adder), threading.Thread(target=scorer)
        t1.start(); t2.start(); t1.join(); t2.join()
        assert errors == []

    def test_selenium_client_is_serial(self):
        from deep_research_tool.search.selenium_browser import SeleniumBrowser
        assert SeleniumBrowser.requires_serial_access is True

    def test_finished_sections_are_completed_when_another_fails(self, tmp_path):
        from deep_research_tool.research.researcher import Researcher
        from deep_research_tool.tests.test_e2e_pipeline import FakeLLM, FakeSearchClient
        plan = make_plan([("1", "A", "a"), ("2", "B", "b")])
        session = make_session(plan, {})
        session.completed_sections = []
        r = Researcher(FakeLLM(), FakeSearchClient(), output_dir=tmp_path)
        r.session = session
        r.evidence_locker = make_locker(tmp_path, [])
        r._sections_running = []
        r._activity_lock = threading.Lock()
        r._checkpoint_lock = threading.Lock()
        done_sections = []
        r._complete_section = lambda s: done_sections.append(s.section)
        import time

        def run_section(section, queries, idx, total):
            if section.section == "2":
                raise RuntimeError("section 2 died")
            time.sleep(0.3)                       # finishes AFTER the failure
            session.section_contents["1"] = {"content": "done"}
        r._run_section = run_section
        r._check_cancel = lambda: None
        r._report_progress = lambda *a, **k: None
        sections = plan.table_of_contents.get_flat_sections()
        with pytest.raises(RuntimeError):
            r._run_sections_parallel(sections, {s.section: [] for s in sections}, 2, set())
        assert done_sections == ["1"]             # the finished one was checkpointed

    def test_write_phase_names_its_own_section(self, tmp_path):
        from deep_research_tool.research.researcher import Researcher
        from deep_research_tool.tests.test_e2e_pipeline import FakeLLM, FakeSearchClient
        plan = make_plan([("1", "A", "a"), ("2", "B", "b")])
        r = Researcher(FakeLLM(), FakeSearchClient(), output_dir=tmp_path)
        r.session = make_session(plan, {})
        r._sections_running = ["1", "2"]
        r._activity_lock = threading.Lock()
        s1, s2 = plan.table_of_contents.get_flat_sections()
        r._set_activity(s1, 0, 2)
        r._set_activity(s2, 1, 2)
        r._set_phase(s1, "write")
        assert r.current_activity["section"] == "1" and r.current_activity["phase"] == "write"
        assert r.current_activity["index"] == 1 and r.current_activity["running"] == ["1", "2"]


# ----------------------------------------------------------------------
# run modes / report loader
# ----------------------------------------------------------------------

class TestReportLoader:
    def test_leading_numbers_only_with_a_separator(self):
        from deep_research_tool.report.report_loader import _clean_title
        assert _clean_title("5G市場") == "5G市場"
        assert _clean_title("2030年の市場展望") == "2030年の市場展望"
        assert _clean_title("3Dプリンタ市場") == "3Dプリンタ市場"
        assert _clean_title("1. 市場規模") == "市場規模"
        assert _clean_title("1.2 概要") == "概要"
        assert _clean_title("1.市場規模") == "市場規模"
        assert _clean_title("第2章 動向") == "動向"
        assert _clean_title("IV. 結論") == "結論"
        assert _clean_title("(3) 価格") == "価格"

    def test_reference_lists_are_not_body_but_similar_titles_are(self):
        from deep_research_tool.report.report_loader import _NON_BODY
        for h in ("参考文献", "参考文献一覧", "出典一覧", "出典リスト", "付録A", "References", "Appendix B"):
            assert _NON_BODY.match(h), h
        for h in ("文献調査の結果", "出典の信頼性評価", "目次構成の考え方"):
            assert not _NON_BODY.match(h), h

    def test_cp932_text_report_is_decoded(self, tmp_path):
        from deep_research_tool.report.report_loader import report_text_from_file
        p = tmp_path / "report.txt"
        p.write_bytes("# 炭素繊維市場\n\n## 1. 市場規模\n本文。\n".encode("cp932"))
        title, text = report_text_from_file(p)
        assert "炭素繊維市場" in text and "\ufffd" not in text
        (tmp_path / "bad.txt").write_bytes(b"\xff\xfe\x00\x01garbage\x80\x81")
        with pytest.raises(ValueError):
            report_text_from_file(tmp_path / "bad.txt")


class TestVerifyOnlyServer:
    def test_label_is_display_only(self):
        """The display label 検証のみ: … must never reach verify_report
        (it would seed the evidence searches): the user's topic travels
        in the verify spec, the label only names the job."""
        from deep_research_tool.webui import server as srv
        src = Path(srv.__file__).read_text(encoding="utf-8")
        assert 'spec["query"] = user_query' in src
        assert 'query=spec.get("query", "")' in src


# ----------------------------------------------------------------------
# resume
# ----------------------------------------------------------------------

def _run_with_figures(tmp, resume_from=None):
    from unittest.mock import patch
    llm = make_llm()
    plan = make_plan([("1", "市場シェア", "A社のシェアを明らかにする")])
    session = make_session(plan, {"1": {"title": "市場シェア",
                                        "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                                        "evidence": SUPPORTED_EVIDENCE}})
    locker = make_locker(tmp, SUPPORTED_EVIDENCE)
    tool = build_tool(tmp, llm, FakeSearch(), auto_figures=True)
    fake = _ResumableResearcher(session, locker)
    with patch("deep_research_tool.main.Researcher", new=lambda **kw: fake):
        result = tool.run(query="市場調査", resume_from=resume_from)
    return tool, result


class TestResumeFixes:
    def test_figures_stage_without_artifact_resumes(self, tmp_path):
        tool, result = _run_with_figures(tmp_path)
        sid = result["session_id"]
        state = RunState.load(tmp_path / f"run_state_{sid}.json")
        assert state.done("figures") and state.artifact("figures") is None   # nothing was drawn
        tool2, res2 = _run_with_figures(tmp_path, resume_from=str(tmp_path / f"session_{sid}.json"))
        assert res2["run_status"] == "completed"
        assert "figures" in res2["resume_plan"]["stages_skipped"]

    def test_cancelled_verification_is_not_a_completed_stage(self, tmp_path):
        tool = build_tool(tmp_path, make_llm(), FakeSearch())
        plan = make_plan([("1", "市場シェア", "a")])
        session = make_session(plan, {"1": {"title": "市場シェア", "content": "x"}})
        tool.run_state = RunState(tmp_path / "run_state_itest.json")
        base = {"chapters": {"1": "x"}, "raw_chapters": {"1": "x"}, "ordered_evidence_ids": []}
        tool.finalization_outcome = dict(base, decision="cancelled")
        tool._mark_verification_done(session, None)
        assert not tool.run_state.done("verification")
        assert (tmp_path / "final_body_itest.json").is_file()      # the partial body is still saved
        tool.finalization_outcome = dict(base, decision="accept")
        tool._mark_verification_done(session, None)
        assert tool.run_state.done("verification")

    def test_runner_keeps_the_draft_and_checkpoints_on_interrupt(self, tmp_path):
        from deep_research_tool.verification.runtime import VerificationCancelled
        tool = build_tool(tmp_path, make_llm(), FakeSearch())
        plan = make_plan([("1", "市場シェア", "a")])
        session = make_session(plan, {"1": {"title": "市場シェア",
                                            "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                                            "evidence": SUPPORTED_EVIDENCE}})
        locker = make_locker(tmp_path, SUPPORTED_EVIDENCE)
        tool.verification_progress = None
        runner = tool._build_finalization_runner(session, locker, "q")
        calls = []
        runner.checkpoint_fn = lambda: calls.append("ckpt")
        runner.draft_path.write_text(json.dumps({"chapters": {"1": "draft"}}), encoding="utf-8")

        def cancel_run(current):
            raise VerificationCancelled()
        from deep_research_tool.report import finalization as fz
        orig = fz.FinalizationController.run
        fz.FinalizationController.run = lambda self, current: cancel_run(current)
        try:
            outcome = runner.run({"1": long_body("1", "市場シェア", CLAIM_SENTENCE)})
        finally:
            fz.FinalizationController.run = orig
        assert outcome["decision"] == "cancelled"
        assert runner.draft_path.is_file()                 # kept for the resume
        assert calls == ["ckpt"]
        # a persisted draft also checkpoints session + evidence
        runner._persist_draft({"1": "edited"})
        assert calls == ["ckpt", "ckpt"]

    def test_replacement_evidence_joins_the_session_store(self, tmp_path):
        from deep_research_tool.evidence.locker import EvidenceType
        from deep_research_tool.report.finalization import VerificationIssue
        tool = build_tool(tmp_path, make_llm(), FakeSearch())
        plan = make_plan([("1", "市場シェア", "a")])
        session = make_session(plan, {"1": {"title": "市場シェア",
                                            "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                                            "evidence": SUPPORTED_EVIDENCE}})
        locker = make_locker(tmp_path, SUPPORTED_EVIDENCE)
        new = locker.add_evidence(url="https://new.example.com/p", title="later",
                                  content_excerpt="x" * 300, extracted_text="y" * 600,
                                  evidence_type=EvidenceType.WEB_PAGE, section_reference="1")
        tool.verification_progress = None
        runner = tool._build_finalization_runner(session, locker, "q")
        runner._register_sections({"1": session.section_contents["1"]["content"]})
        issue = VerificationIssue(section_id="1", type="invalid_citation",
                                  supporting_source_ids=[new.id])
        hints = runner._register_replacements("1", [issue])
        assert hints
        urls = [e["url"] for e in session.section_contents["1"]["extracted_content"]]
        assert urls[-1] == "https://new.example.com/p"
        assert session.section_contents["1"]["sources"][-1] == "https://new.example.com/p"
        # rebuilding the registry from the store gives the same number
        runner2 = tool._build_finalization_runner(session, locker, "q")
        runner2._register_sections({"1": "x"})
        assert runner2.citation_mgr.append_evidence("1", new.id) == \
            int(hints[id(issue)].split("[SOURCE ")[1].rstrip("]"))

    def test_restored_body_reorders_the_locker(self, tmp_path):
        tool = build_tool(tmp_path, make_llm(), FakeSearch())
        plan = make_plan([("1", "市場シェア", "a")])
        session = make_session(plan, {"1": {"title": "市場シェア", "content": "x"}})
        locker = make_locker(tmp_path, [("https://e1", "a" * 300), ("https://e2", "b" * 300)])
        ids = [e.id for e in locker.get_all_evidence()]
        (tmp_path / "final_body_itest.json").write_text(json.dumps({
            "chapters": {"1": "本文 [1]"}, "raw_chapters": {"1": "本文 [SOURCE 2]"},
            "decision": "accept", "ordered_evidence_ids": [ids[1], ids[0]]}), encoding="utf-8")
        tool.run_state = RunState(tmp_path / "run_state_itest.json")
        tool.run_state.mark("verification")
        tool._resuming = True
        assert tool._skip_stage("verification") and tool._restore_finalization(session, None, locker)
        assert [e.id for e in locker.get_all_evidence()] == [ids[1], ids[0]]

    def test_deep_think_skip_needs_its_artifact(self, tmp_path):
        tool = build_tool(tmp_path, make_llm(), FakeSearch())
        tool.run_state = RunState(tmp_path / "run_state_x.json")
        tool.run_state.mark("deep_think")
        tool._resuming = True
        assert not tool._skip_stage("deep_think", require=str(tmp_path / "deep_think_missing.json"))
        assert tool.run_state.resumed_from == []


# ----------------------------------------------------------------------
# retries / config
# ----------------------------------------------------------------------

class TestRetryPolicy:
    def test_status_codes_decide_transience(self):
        from deep_research_tool.api.base import BaseLLMClient

        class APIStatusError(Exception):
            def __init__(self, status):
                self.status_code = status
        assert BaseLLMClient.is_transient_error(APIStatusError(429))
        assert BaseLLMClient.is_transient_error(APIStatusError(503))
        assert not BaseLLMClient.is_transient_error(APIStatusError(401))
        assert not BaseLLMClient.is_transient_error(APIStatusError(400))

        class RateLimitError(Exception):
            pass
        assert BaseLLMClient.is_transient_error(RateLimitError("slow down"))

    def test_sdk_clients_leave_retries_to_the_base_policy(self):
        from deep_research_tool.api import anthropic_client, openai_client
        for mod in (openai_client, anthropic_client):
            src = Path(mod.__file__).read_text(encoding="utf-8")
            assert '"max_retries": 0' in src and "_leaf_call(" in src

    def test_leaf_call_releases_the_permit_between_attempts(self):
        from contextlib import contextmanager
        from deep_research_tool.api.base import BaseLLMClient
        events = []

        class C:
            retry_attempts = 2
            retry_backoff_seconds = 0.0
            cancel_check = staticmethod(lambda: None)

            @contextmanager
            def _leaf_permit(self):
                events.append("acquire")
                try:
                    yield
                finally:
                    events.append("release")
        c = C()
        c._call_with_retry = lambda fn, label: BaseLLMClient._call_with_retry(c, fn, label=label)
        calls = []

        def fn():
            calls.append(1)
            if len(calls) < 2:
                raise ConnectionError("x")
            return "ok"
        assert BaseLLMClient._leaf_call(c, fn, label="t") == "ok"
        assert events == ["acquire", "release", "acquire", "release"]
