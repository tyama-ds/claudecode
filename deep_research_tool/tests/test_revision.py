"""Feedback re-research (revision.py + DeepResearchTool.revise_report +
POST /api/feedback): the plan from the LLM and the rule-based fallback,
the revised session (kept chapters reused, revised / added researched),
the evidence hand-over, reference research as documents, the result's
revision summary, the Web API. No network, no LLM."""
import json
import time
from pathlib import Path

import pytest

import deep_research_tool.main as main_mod
from deep_research_tool import revision as rev
from deep_research_tool.evidence.locker import EvidenceLocker
from deep_research_tool.main import DeepResearchTool
from deep_research_tool.research.query_generator import ResearchPlan, TableOfContents, TableOfContentsItem
from deep_research_tool.research.researcher import ResearchSession, ResearchState
from deep_research_tool.tests.test_concurrency_visibility import _post, web  # noqa: F401
from deep_research_tool.tests.test_e2e_pipeline import FakeLLM, FakeSearchClient
from deep_research_tool.tests.test_finalization_integration import build_tool
from deep_research_tool.tests.test_url_sources import LinkySearch
from deep_research_tool.utils.cancellation import RunCancelled

REVISION_JSON = json.dumps({
    "items": [{"section": "1", "action": "keep"},
              {"section": "2", "action": "revise",
               "instructions": "帝人と三菱ケミカルを加えて比較表にする",
               "queries": ["帝人 炭素繊維 戦略", "三菱ケミカル 炭素繊維 戦略"], "reason": "指摘"},
              {"section": "9", "action": "revise"}],
    "additions": [{"title": "リサイクル炭素繊維", "instructions": "リサイクル炭素繊維の動向を追加",
                   "queries": ["リサイクル炭素繊維 市場"]}],
    "requirements_addendum": "主要3社を比較する",
    "acknowledged": ["航空機向け需要の回復は妥当"], "notes": "n"}, ensure_ascii=False)
FEEDBACK = ("第2章は東レだけでなく帝人・三菱ケミカルの戦略も比較してほしい。"
            "航空機向け需要の回復の話は妥当。リサイクル炭素繊維の章を追加してほしい。")


class RevLLM(FakeLLM):
    """The e2e fake plus an answer for the revision-plan prompt."""
    plan_json = REVISION_JSON

    def _answer(self, prompt):
        if "修正計画" in prompt and "JSON のみを出力" in prompt:
            return self.plan_json
        return super()._answer(prompt)


class BrokenPlanLLM(RevLLM):
    plan_json = "no json here"


def _session(sections=(("1", "市場規模の推移"), ("2", "主要メーカーの動向")), query="炭素繊維の市場調査"):
    items = [TableOfContentsItem(section=s, title=t, description=f"{t}について") for s, t in sections]
    plan = ResearchPlan(title="炭素繊維レポート", summary="s",
                        table_of_contents=TableOfContents(title="t", items=items),
                        search_queries=[f"q{i}" for i in range(len(items) * 2)])
    session = ResearchSession(session_id="parent01", query=query, requirements="簡潔に",
                              state=ResearchState.COMPLETED, research_plan=plan)
    for s, t in sections:
        session.section_contents[s] = {"title": t, "content": f"{t}の本文。" * 10, "summary": f"{t}の要約",
                                       "confidence": "high", "sources": ["https://x/1"],
                                       "extracted_content": [{"url": "https://x/1", "title": "x", "content": "c"}]}
    session.section_contents["_executive_summary"] = {"executive_summary": "要旨"}
    session.completed_sections = [s for s, _ in sections]
    session.run_config = {"provider": "openai"}
    return session


class TestPlan:
    def test_parse_llm_plan(self):
        s = _session()
        rp = rev.parse_revision_response(REVISION_JSON, s, "add", k=3)
        assert [a.action for a in rp.actions] == ["keep", "revise", "add"]
        assert rp.actions[1].queries == ["帝人 炭素繊維 戦略", "三菱ケミカル 炭素繊維 戦略"]
        assert rp.actions[2].section == "3" and rp.actions[2].title == "リサイクル炭素繊維"
        assert rp.requirements_addendum == "主要3社を比較する" and rp.acknowledged == ["航空機向け需要の回復は妥当"]
        assert rp.source == "llm" and [a.section for a in rp.changes] == ["2", "3"]
        # revise mode: no additions; unknown section 9 ignored
        rp = rev.parse_revision_response(REVISION_JSON, s, "revise", k=3)
        assert [a.action for a in rp.actions] == ["keep", "revise"]
        with pytest.raises(rev.NoChangesRequested):
            rev.parse_revision_response('{"items": [{"section": "1", "action": "keep"}]}', s, "add")
        only_keep = RevLLM()
        only_keep.plan_json = '{"items": [{"section": "1", "action": "keep"}], "acknowledged": ["妥当"]}'
        rp = rev.plan_revision(only_keep, s, "妥当です", "add")
        assert rp.source == "llm" and not rp.changes and all(a.action == "keep" for a in rp.actions)
        with pytest.raises(ValueError):
            rev.parse_revision_response("not json", s, "add")
        # a revise item without queries gets one from the title + instructions
        rp = rev.parse_revision_response('{"items": [{"section": "1", "action": "update", "instructions": "数値を更新"}]}', s, "add")
        assert rp.actions[0].action == "revise" and rp.actions[0].queries == ["市場規模の推移 数値を更新"]

    def test_fallback_plan(self):
        s = _session()
        rp = rev.fallback_revision_plan(s, "第2章の数字が古い。2025年の資料で更新してほしい", "add")
        assert [a.action for a in rp.actions] == ["keep", "revise"] and rp.source == "fallback"
        assert rp.actions[1].queries and "主要メーカーの動向" in rp.actions[1].queries[0]
        rp = rev.fallback_revision_plan(s, "市場規模の推移の章を直して", "revise")
        assert [a.action for a in rp.actions] == ["revise", "keep"]
        rp = rev.fallback_revision_plan(s, "全体的に古い", "add")
        assert all(a.action == "revise" for a in rp.actions)

    def test_plan_revision_modes_and_fallback(self):
        s = _session()
        rp = rev.plan_revision(RevLLM(), s, FEEDBACK, "add", language="ja", k=2)
        assert rp.source == "llm" and len(rp.actions[1].queries) == 2
        rp = rev.plan_revision(BrokenPlanLLM(), s, "第1章を直して", "add")
        assert rp.source == "fallback" and rp.actions[0].action == "revise"
        rp = rev.plan_revision(RevLLM(), s, "全部やり直し", "full")
        assert rp.mode == "full" and not rp.actions and rp.requirements_addendum == "全部やり直し"
        with pytest.raises(ValueError, match="feedback"):
            rev.plan_revision(RevLLM(), s, "  ", "add")
        with pytest.raises(ValueError, match="mode"):
            rev.plan_revision(RevLLM(), s, "x", "nope")
        review = rev.review_plan_for(rev.parse_revision_response(REVISION_JSON, s, "add"), s)
        titles = [it.title for it in review.table_of_contents.items]
        assert titles == ["【そのまま】市場規模の推移", "【直す】主要メーカーの動向", "【追加】リサイクル炭素繊維"]
        assert review.search_queries[0] == "帝人 炭素繊維 戦略" and "記録のみ" in review.summary
        prompt = rev.build_revision_prompt(s, FEEDBACK, "revise", language="en", k=2)
        assert "never add chapters" in prompt and "主要メーカーの動向" in prompt


class TestRevisedSession:
    def test_build_revised_session(self):
        parent = _session()
        rp = rev.parse_revision_response(REVISION_JSON, parent, "add", k=3)
        new = rev.build_revised_session(parent, rp, k=3, session_id="child01", feedback=FEEDBACK)
        assert new.session_id == "child01" and new.state == ResearchState.RESEARCHING
        assert new.completed_sections == ["1"]
        assert new.section_contents["1"] == parent.section_contents["1"]
        assert new.section_contents["1"] is not parent.section_contents["1"]
        assert new.section_contents["2"]["content"] == "" and new.section_contents["2"]["previous_content"]
        assert new.section_contents["2"]["revision_instructions"].startswith("帝人")
        assert new.section_contents["3"]["title"] == "リサイクル炭素繊維" and new.section_contents["3"]["content"] == ""
        flat = new.research_plan.table_of_contents.get_flat_sections()
        assert [it.section for it in flat] == ["1", "2", "3"]
        assert "【修正指示】" in flat[1].description and flat[2].description.startswith("【追加の章】")
        assert len(new.research_plan.search_queries) == 3 * 3
        assert new.research_plan.search_queries[3:5] == ["帝人 炭素繊維 戦略", "三菱ケミカル 炭素繊維 戦略"]
        assert new.research_plan.search_queries[6] == "リサイクル炭素繊維 市場"
        assert "【追加要件（フィードバック）】主要3社を比較する" in new.requirements
        assert new.run_config["parent_session_id"] == "parent01" and new.run_config["revision_mode"] == "add"
        assert new.run_config["feedback"] == FEEDBACK
        assert new.is_section_done("1") and not new.is_section_done("2") and not new.is_section_done("3")
        # the parent is untouched
        assert parent.research_plan.table_of_contents.items[1].description == "主要メーカーの動向について"
        assert len(parent.research_plan.table_of_contents.items) == 2

    def test_seed_locker_and_reference_documents(self, tmp_path):
        parent_dir = tmp_path / "p"
        locker = EvidenceLocker(research_id="parent01", output_dir=parent_dir / "evidence")
        locker.add_evidence(url="https://a.example/x", title="A", content_excerpt="e", extracted_text="本文 " * 100,
                            section_reference="1")
        locker.add_evidence(url="local://doc.pdf#chunk0", title="L", content_excerpt="e", extracted_text="本文 " * 100)
        locker.add_evidence(url="https://a.example/short", title="S", content_excerpt="e", extracted_text="短い")
        ev_path = locker.export_to_json()
        parent = _session()
        parent.saved_artifacts = {"evidence_json": str(ev_path)}
        session_path = parent_dir / "session_parent01.json"
        parent.save(session_path)
        assert rev.evidence_path_for(session_path, parent) == Path(ev_path)
        seeded = rev.seed_locker(ev_path, "child01", tmp_path / "c")
        assert seeded.research_id == "child01" and len(seeded.get_all_evidence()) == 3
        out = seeded.export_to_json()
        assert out.name == "evidence_child01.json" and out.parent == tmp_path / "c" / "evidence"
        assert [e.url for e in seeded.get_section_evidence("1")] == ["https://a.example/x"]
        empty = rev.seed_locker(tmp_path / "missing.json", "child02", tmp_path / "d")
        assert empty.research_id == "child02" and not empty.get_all_evidence()
        docs, info = rev.reference_documents([str(session_path), str(tmp_path / "nope.json")])
        assert [d["path"] for d in docs] == ["https://a.example/x"]      # local:// and short pages skipped
        assert docs[0]["metadata"]["source"] == "reference" and docs[0]["metadata"]["reference_session_id"] == "parent01"
        assert info[0]["evidence"] == 3 and info[0]["used"] == 1 and info[1]["error"]
        notes = rev.reference_summaries([str(session_path)])
        assert notes and "炭素繊維の市場調査" in notes[0] and "市場規模の推移" in notes[0]


def _run_v1(tmp_path, llm=None, search=None):
    tool = build_tool(tmp_path, llm or RevLLM(), search or FakeSearchClient(),
                      enable_verification=False, research_iterations=1)
    tool.config.research.use_enhanced_synthesis = False
    tool.config.research.max_gap_fill_rounds = 0
    res = tool.run(query="炭素繊維の市場調査", requirements="目次は2章構成でよい")
    return tool, res


class TestToolRevision:
    def test_add_mode_reuses_kept_and_researches_the_rest(self, tmp_path):
        tool1, res1 = _run_v1(tmp_path)
        parent_path = res1["saved_artifacts"]["session"]
        parent = json.load(open(parent_path, encoding="utf-8"))
        search2 = FakeSearchClient()
        tool2 = build_tool(tmp_path, RevLLM(), search2, enable_verification=False, research_iterations=1)
        tool2.config.research.use_enhanced_synthesis = False
        tool2.config.research.max_gap_fill_rounds = 0
        msgs = []
        res2 = tool2.revise_report(parent_session_path=parent_path, feedback=FEEDBACK, mode="add",
                                   reference_session_paths=[parent_path], parent_job_id="job-1",
                                   progress_callback=lambda m, p: msgs.append(m))
        r = res2["revision"]
        assert res2["kind"] == "revision" and res2["run_status"] == "completed"
        assert r["mode"] == "add" and r["kept"] == ["1"] and r["revised"] == ["2"] and r["added"] == ["3"]
        assert r["parent_session_id"] == parent["session_id"] and r["parent_job_id"] == "job-1"
        assert r["acknowledged"] == ["航空機向け需要の回復は妥当"] and r["plan_source"] == "llm"
        rows = {x["section"]: x for x in r["sections"]}
        assert rows["1"]["action"] == "keep" and rows["1"]["new_evidence"] == 0
        assert rows["2"]["new_evidence"] > 0 and rows["3"]["chars_after"] > 0 and rows["3"]["chars_before"] == 0
        assert r["references"][0]["used"] > 0 and r["engine_rerun"] is False
        s2 = tool2.researcher.get_session()
        assert s2.session_id != parent["session_id"]
        assert s2.section_contents["1"]["content"] == parent["section_contents"]["1"]["content"]
        assert s2.run_config["parent_session_id"] == parent["session_id"]
        assert res2["resume_plan"]["reused"] == ["1"] and res2["resume_plan"]["to_process"] == ["2", "3"]
        assert search2.search_calls > 0
        assert any("修正計画" in m for m in msgs)
        # the parent file is untouched
        assert json.load(open(parent_path, encoding="utf-8")) == parent
        assert Path(res2["report_path"]).exists()

    def test_plan_review_cancel_and_revise(self, tmp_path):
        tool1, res1 = _run_v1(tmp_path)
        parent_path = res1["saved_artifacts"]["session"]
        tool2 = build_tool(tmp_path, RevLLM(), FakeSearchClient(), enable_verification=False)
        shown = []

        def cancel_cb(plan, revise_fn):
            shown.append(plan)
            raise RunCancelled("cancelled during plan review")

        with pytest.raises(RunCancelled):
            tool2.revise_report(parent_session_path=parent_path, feedback=FEEDBACK, mode="add",
                                plan_review_callback=cancel_cb)
        assert shown and shown[0].title.startswith("修正計画（追加調査）")
        assert [it.title for it in shown[0].table_of_contents.items][1] == "【直す】主要メーカーの動向"

        def revise_cb(plan, revise_fn):
            new_plan = revise_fn(plan, "第1章も直して")
            shown.append(new_plan)
            return new_plan

        tool3 = build_tool(tmp_path, RevLLM(), FakeSearchClient(), enable_verification=False, research_iterations=1)
        tool3.config.research.use_enhanced_synthesis = False
        tool3.config.research.max_gap_fill_rounds = 0
        res3 = tool3.revise_report(parent_session_path=parent_path, feedback=FEEDBACK, mode="revise",
                                   plan_review_callback=revise_cb)
        assert res3["revision"]["mode"] == "revise" and res3["revision"]["added"] == []
        assert res3["revision"]["revised"] == ["2"]

    def test_full_mode_and_errors(self, tmp_path):
        tool1, res1 = _run_v1(tmp_path)
        parent_path = res1["saved_artifacts"]["session"]
        tool2 = build_tool(tmp_path, RevLLM(), FakeSearchClient(), enable_verification=False, research_iterations=1)
        tool2.config.research.use_enhanced_synthesis = False
        tool2.config.research.max_gap_fill_rounds = 0
        res2 = tool2.revise_report(parent_session_path=parent_path, feedback="全面的にやり直し", mode="full")
        r = res2["revision"]
        assert r["mode"] == "full" and r["plan_source"] == "none" and r["engine_rerun"] is False
        assert tool2.researcher.get_session().session_id != json.load(open(parent_path))["session_id"]
        assert "【追加要件（フィードバック）】全面的にやり直し" in tool2.researcher.get_session().requirements
        with pytest.raises(ValueError, match="空です"):
            tool2.revise_report(parent_session_path=parent_path, feedback="  ")
        with pytest.raises(ValueError, match="見つかりません"):
            tool2.revise_report(parent_session_path=str(tmp_path / "nope.json"), feedback="x")
        with pytest.raises(ValueError, match="mode"):
            tool2.revise_report(parent_session_path=parent_path, feedback="x", mode="zzz")
        only_keep = RevLLM()
        only_keep.plan_json = '{"items": [{"section": "1", "action": "keep"}], "acknowledged": ["妥当"]}'
        tool4 = build_tool(tmp_path, only_keep, FakeSearchClient(), enable_verification=False)
        with pytest.raises(ValueError, match="修正すべき章が見つかりません"):
            tool4.revise_report(parent_session_path=parent_path, feedback="妥当です", mode="add")

    def test_external_engine_reruns_fully(self, tmp_path):
        tool1, res1 = _run_v1(tmp_path)
        parent_path = res1["saved_artifacts"]["session"]
        tool2 = build_tool(tmp_path / "e", RevLLM(), LinkySearch(), research_engine="mock-fast",
                           engine_options={"speed_factor": 0.01}, enable_verification=False)
        res2 = tool2.revise_report(parent_session_path=parent_path, feedback=FEEDBACK, mode="add")
        assert res2["revision"]["engine_rerun"] is True and res2["revision"]["mode"] == "full"
        assert res2["research_engine"] == "mock-fast"


class TestWebApi:
    BASE = {"provider": "openai", "openai_api_key": "sk-test", "plan_review": False,
            "run_mode": "research_only", "enable_verification": False}

    def _offline(self, monkeypatch):
        def offline_tool(config):
            tool = DeepResearchTool.__new__(DeepResearchTool)
            tool.config = config
            tool.config.research.use_enhanced_synthesis = False
            tool.config.research.max_gap_fill_rounds = 0
            tool.config.research.min_iterations = 1
            tool.config.research.max_iterations = 1
            tool.llm_client = RevLLM()
            tool.stage_llm_clients = {}
            tool.search_client = FakeSearchClient()
            tool.researcher = None
            tool.verifier = None
            tool.report_generator = None
            tool.deep_think_processor = None
            return tool
        monkeypatch.setattr(main_mod, "DeepResearchTool", offline_tool)

    def _wait(self, web, job_id):
        import urllib.request
        port = web.server_address[1]
        for _ in range(240):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status?job_id={job_id}", timeout=30) as res:
                rec = json.loads(res.read())
            if rec.get("state") in ("completed", "error", "cancelled"):
                return rec
            time.sleep(0.5)
        return rec

    def test_feedback_job_through_the_web(self, web, monkeypatch):
        self._offline(monkeypatch)
        status, d = _post(web, "/api/research", {**self.BASE, "query": "炭素繊維の市場調査",
                                                 "requirements": "目次は2章構成でよい"})
        assert status == 202, d
        parent = self._wait(web, d["job_id"])
        assert parent["state"] == "completed", parent.get("error")
        # validation
        status, d = _post(web, "/api/feedback", {**self.BASE, "job_id": "job-none", "feedback": "x"})
        assert status == 404
        status, d = _post(web, "/api/feedback", {**self.BASE, "job_id": parent["job_id"], "feedback": " "})
        assert status == 400 and d["field"] == "feedback_text"
        status, d = _post(web, "/api/feedback", {**self.BASE, "job_id": parent["job_id"], "feedback": "x", "mode": "zzz"})
        assert status == 400 and d["field"] == "feedback_mode"
        status, d = _post(web, "/api/feedback", {**self.BASE, "job_id": parent["job_id"], "feedback": "x",
                                                 "urls": ["ftp://nope"], "research_engine": "zzz-unknown"})
        assert status == 400 and d["field"] == "research_engine"
        # the real thing
        status, d = _post(web, "/api/feedback", {**self.BASE, "job_id": parent["job_id"], "feedback": FEEDBACK,
                                                 "mode": "add", "reference_jobs": [parent["job_id"], "job-none"],
                                                 "urls": "https://a.example/x", "preferred_domains": "meti.go.jp"})
        assert status == 202, d
        child = self._wait(web, d["job_id"])
        assert child["state"] == "completed", child.get("error")
        res = child["result"]
        assert res["kind"] == "revision" and res["parent_job_id"] == parent["job_id"]
        assert res["revision"]["kept"] == ["1"] and res["revision"]["added"] == ["3"]
        ps = child["params_summary"]
        assert ps["parent_job_id"] == parent["job_id"] and ps["source_mode"] == "urls_web"
        assert ps["source_urls"] == ["https://a.example/x"] and ps["preferred_domains"] == ["meti.go.jp"]
        assert ps["feedback"]["reference_session_paths"] == []   # the parent itself is not a reference
        assert ps["feedback"]["text"] == FEEDBACK and child["query"].startswith("再調査: ")
        assert res["url_sources"]["requested"] == 1
