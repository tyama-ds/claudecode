"""Length control end to end (v1.12.0).

The 「要約」 preset (fixed, ~4,000 chars) produced ~130,000-char reports:
the target never reached the drafting steps, the plan always had 15-20
sections, DeepThink appended uncapped text, References listed every
collected source, and the final check compressed without a number and
then accepted silently. Every test here uses fakes — no paid API.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from deep_research_tool.config import create_config
from deep_research_tool.report.finalization import (
    FinalizationController,
    LoopBudget,
    StructuredVerdict,
    VerificationIssue,
    VerificationMetrics,
    ISSUE_UNSUPPORTED,
    count_body_chars,
)


def _fixed_config(**kw):
    base = dict(length_mode="fixed", preferred_body_chars=4000)
    base.update(kw)
    return create_config(**base)


class TestWritingBudget(unittest.TestCase):
    def test_fixed_mode_target_is_the_writing_budget(self):
        self.assertEqual(_fixed_config().report.writing_budget_chars(), 4000)

    def test_adaptive_preference_is_not_a_budget(self):
        cfg = create_config(length_mode="adaptive", preferred_body_chars=20000)
        self.assertIsNone(cfg.report.writing_budget_chars())

    def test_hard_max_bounds_the_drafts_below_the_ceiling(self):
        cfg = create_config(hard_max_body_chars=10000)
        self.assertEqual(cfg.report.writing_budget_chars(), 9000)
        cfg = _fixed_config(hard_max_body_chars=3000)
        self.assertEqual(cfg.report.writing_budget_chars(), 2700)

    def test_web_ui_summary_preset_reaches_the_budget(self):
        from deep_research_tool.webui.server import build_config_kwargs
        kwargs = build_config_kwargs({"length_mode": "fixed",
                                      "preferred_body_chars": "4000"})
        cfg = create_config(**kwargs)
        self.assertIsNone(cfg.report.target_characters)
        self.assertEqual(cfg.report.writing_budget_chars(), 4000)

    def test_main_hands_the_budget_to_the_researcher(self):
        import inspect
        import deep_research_tool.main as main_mod
        text = inspect.getsource(main_mod.DeepResearchTool)
        self.assertIn(
            "target_characters=self.config.report.writing_budget_chars()", text)


class TestResearcherSizing(unittest.TestCase):
    def _researcher(self, **kw):
        from deep_research_tool.research.researcher import Researcher
        return Researcher(llm_client=MagicMock(), search_client=MagicMock(),
                          output_dir=Path(tempfile.mkdtemp()), language="ja",
                          **kw)

    def test_budget_reaches_planner_and_sections(self):
        r = self._researcher(target_characters=4000)
        self.assertEqual(r._body_budget(), 4000)
        self.assertEqual(r.query_generator.body_budget_chars, 4000)

    def test_no_budget_keeps_classic_behaviour(self):
        r = self._researcher()
        self.assertIsNone(r._body_budget())
        self.assertIsNone(r.query_generator.body_budget_chars)


class TestContentExtractorBudget(unittest.TestCase):
    def _extractor(self, per_section=None):
        from deep_research_tool.research.content_extractor import ContentExtractor
        llm = MagicMock()
        ex = ContentExtractor(llm_client=llm, language="ja",
                              target_chars_per_section=per_section)
        return ex, llm

    def test_small_budget_is_not_floored_at_500_per_point(self):
        ex, _ = self._extractor(800)
        # 3 points x 266 chars, not 3 x 500 (= 1,500 for an 800 section)
        self.assertEqual(ex._calc_target_per_point(3), 266)
        self.assertEqual(ex._outline_points_range(), (2, 3))

    def test_default_unchanged_without_budget(self):
        ex, _ = self._extractor(None)
        self.assertEqual(ex._calc_target_per_point(5), 600)
        self.assertEqual(ex._outline_points_range(), (4, 6))
        self.assertEqual(ex._length_rule(), "")

    def test_outline_prompt_and_cap_follow_the_budget(self):
        ex, llm = self._extractor(800)
        llm.generate.return_value = MagicMock(content=(
            '{"outline": [' + ",".join(
                f'{{"title": "論点{i}", "description": "d", "key_facts": []}}'
                for i in range(6)) + ']}'))
        outline = ex._generate_section_outline("市場", "説明", "src", "", "")
        prompt = llm.generate.call_args[0][0]
        self.assertIn("2〜3個の論点", prompt)
        self.assertEqual(len(outline), 3)      # never more points than asked

    def test_synthesis_and_integration_prompts_state_the_size(self):
        ex, llm = self._extractor(800)
        llm.generate.return_value = MagicMock(content="本文\n")
        ex._integrate_content("市場", "説明",
                              [{"title": "a", "content": "x"}], "")
        self.assertIn("約800字", llm.generate.call_args[0][0])
        from deep_research_tool.research.content_extractor import ExtractedContent
        ec = ExtractedContent(source_url="https://a.example", source_title="A",
                              raw_content="r", processed_content="p",
                              key_points=["k"], relevance_score=0.9)
        llm.generate.reset_mock()
        llm.generate.return_value = MagicMock(content="本文です。")
        ex.synthesize_section_content("市場", "説明", [ec], "")
        self.assertIn("約800字", llm.generate.call_args[0][0])

    def test_no_length_rule_in_prompts_without_budget(self):
        ex, llm = self._extractor(None)
        llm.generate.return_value = MagicMock(content="本文\n")
        ex._integrate_content("市場", "説明",
                              [{"title": "a", "content": "x"}], "")
        self.assertNotIn("本文の長さは約", llm.generate.call_args[0][0])


def _plan_json(mains=5, subs=3):
    import json
    toc = []
    for i in range(1, mains + 1):
        toc.append({"section": str(i), "title": f"炭素繊維 市場動向{i}",
                    "description": f"説明{i}",
                    "subsections": [{"section": f"{i}.{j}",
                                     "title": f"炭素繊維 需要詳細{i}-{j}",
                                     "description": "d"}
                                    for j in range(1, subs + 1)]})
    return json.dumps({"title": "市場", "summary": "s",
                       "table_of_contents": toc,
                       "search_queries": [f"市場 クエリ{i}" for i in range(30)],
                       "key_terms": [], "suggested_sources": []},
                      ensure_ascii=False)


class TestPlannerBudget(unittest.TestCase):
    def _qg(self, budget, response):
        from deep_research_tool.research.query_generator import QueryGenerator
        llm = MagicMock()
        llm.generate.return_value = MagicMock(content=response)
        qg = QueryGenerator(llm, language="ja")
        qg.body_budget_chars = budget
        return qg, llm

    def test_summary_budget_asks_for_a_small_plan(self):
        qg, llm = self._qg(4000, _plan_json(4, 0))
        qg.create_research_plan("炭素繊維")
        prompt = llm.generate.call_args[0][0]
        self.assertIn("メインセクションは3〜5個", prompt)
        self.assertIn("サブセクションは作らない", prompt)
        self.assertNotIn("最低5つのメインセクション", prompt)
        # a plan without subsections is valid for a small budget (no retry)
        self.assertEqual(llm.generate.call_count, 1)

    def test_oversized_plan_is_fitted_deterministically(self):
        qg, _ = self._qg(4000, _plan_json(7, 3))
        plan = qg.create_research_plan("炭素繊維")
        flat = plan.table_of_contents.get_flat_sections()
        self.assertEqual(len(flat), 5)          # 4000 // 800
        # folded subsection topics stay in scope via the description
        self.assertIn("炭素繊維 需要詳細1-1", plan.table_of_contents.items[0].description)

    def test_classic_plan_without_budget(self):
        qg, llm = self._qg(None, _plan_json(5, 2))
        plan = qg.create_research_plan("炭素繊維")
        self.assertIn("最低5つのメインセクション", llm.generate.call_args[0][0])
        self.assertEqual(len(plan.table_of_contents.get_flat_sections()), 15)

    def test_large_budget_keeps_classic_plan(self):
        qg, llm = self._qg(20000, _plan_json(5, 2))
        qg.create_research_plan("炭素繊維")
        self.assertIn("最低5つのメインセクション", llm.generate.call_args[0][0])


class TestDeepThinkCap(unittest.TestCase):
    def _tool(self, **cfg):
        import deep_research_tool.main as main_mod
        tool = main_mod.DeepResearchTool.__new__(main_mod.DeepResearchTool)
        tool.config = create_config(**cfg)
        return tool

    def test_conclusion_is_capped_by_whole_sentences(self):
        tool = self._tool(length_mode="fixed", preferred_body_chars=4000)
        conclusion = "結論の一文目である。" * 60       # 600 chars
        kept = tool._fit_deep_think_conclusion(conclusion, 5)
        # 4000 / 5 sections / 4 = 200 chars at most, whole sentences
        self.assertLessEqual(len(kept), 200)
        self.assertTrue(kept.endswith("。"))

    def test_uncapped_without_budget(self):
        tool = self._tool()
        conclusion = "結論である。" * 100
        self.assertEqual(tool._fit_deep_think_conclusion(conclusion, 5), conclusion)


def _verdict_for(chapters, issues=None, support=1.0):
    v = StructuredVerdict()
    v.metrics = VerificationMetrics(
        claim_support_score=support,
        critical_question_coverage=1.0,
        actual_body_chars=sum(count_body_chars(t) for t in chapters.values()),
        citations_valid=True)
    v.issues = list(issues or [])
    return v


class TestFixedLengthLoop(unittest.TestCase):
    def _halve(self, chapters, calls):
        def length_fn(sid, issues):
            calls.append(sid)
            return chapters[sid][: len(chapters[sid]) // 2]
        return length_fn

    def test_overshoot_is_compressed_even_without_revision_budget(self):
        chapters = {"1": "あ" * 8000, "2": "い" * 8000}
        calls = []
        ctrl = FinalizationController(
            verify_fn=lambda ch: _verdict_for(ch),
            length_fn=self._halve(chapters, calls),
            budget=LoopBudget(max_final_research_rounds=0,
                              max_final_revision_rounds=0),
            fixed_target_chars=4000, max_length_passes=2)
        out = ctrl.run(chapters)
        self.assertEqual(out["decision"], "accept")
        self.assertEqual(out["length_report"]["body_chars"], 4000)
        self.assertTrue(out["length_report"]["within"])
        self.assertEqual(len(calls), 4)              # 2 passes x 2 sections
        self.assertTrue(any(h["decision"] == "compress_to_target"
                            for h in out["history"]))

    def test_evidence_problems_do_not_skip_the_length_check(self):
        chapters = {"1": "あ" * 16000}
        issue = VerificationIssue(section_id="1", claim_id="C-1",
                                  type=ISSUE_UNSUPPORTED, severity="critical",
                                  claim="主張", reason="根拠なし",
                                  search_queries=["q"])
        calls = []
        ctrl = FinalizationController(
            verify_fn=lambda ch: _verdict_for(ch, [issue], support=0.5),
            length_fn=self._halve(chapters, calls),
            budget=LoopBudget(max_final_research_rounds=0,
                              max_final_revision_rounds=0),
            fixed_target_chars=4000, max_length_passes=2)
        out = ctrl.run(chapters)
        self.assertEqual(out["decision"], "finalize_with_limitations")
        self.assertEqual(len(calls), 2)
        # 16000 -> 8000 -> 4000 chars of body (+ the limitations note)
        self.assertLess(out["length_report"]["body_chars"], 4800)

    def test_what_does_not_fit_is_reported(self):
        chapters = {"1": "あ" * 20000}
        ctrl = FinalizationController(
            verify_fn=lambda ch: _verdict_for(ch),
            length_fn=lambda sid, issues: None,      # cannot shrink
            fixed_target_chars=4000)
        out = ctrl.run(chapters)
        lr = out["length_report"]
        self.assertTrue(lr["over"])
        self.assertGreaterEqual(lr["body_chars"], 20000)
        self.assertEqual(lr["upper"], 4800)

    def test_adaptive_mode_never_runs_length_passes(self):
        calls = []
        chapters = {"1": "あ" * 20000}
        ctrl = FinalizationController(
            verify_fn=lambda ch: _verdict_for(ch),
            length_fn=self._halve(chapters, calls))
        out = ctrl.run(chapters)
        self.assertEqual(calls, [])
        self.assertEqual(out["length_report"]["mode"], "adaptive")

    def test_limitations_skip_render_only_sections(self):
        issue = VerificationIssue(section_id="1", claim_id="C-1",
                                  type=ISSUE_UNSUPPORTED, severity="critical",
                                  claim="主張", reason="根拠なし",
                                  search_queries=["q"])
        chapters = {"1": "本文", "要旨": "要旨の本文"}
        ctrl = FinalizationController(
            verify_fn=lambda ch: _verdict_for(ch, [issue], support=0.5),
            budget=LoopBudget(max_final_research_rounds=0),
            limitations_skip_sections={"要旨"})
        out = ctrl.run(chapters)
        self.assertIn("調査上の限界", out["chapters"]["1"])
        self.assertNotIn("調査上の限界", out["chapters"]["要旨"])


class TestRunnerLengthEdits(unittest.TestCase):
    def _runner(self, tmp, **cfg):
        from deep_research_tool.evidence.locker import EvidenceLocker
        from deep_research_tool.report.finalization_runner import FinalizationRunner
        config = create_config(**cfg)
        llm = MagicMock()
        llm.generate.return_value = MagicMock(content="短くした本文")
        runner = FinalizationRunner(
            evidence_locker=EvidenceLocker(research_id="t", output_dir=Path(tmp)),
            session_contents={}, research_plan=None, query="調査",
            language="ja", llm_client=llm, search_client=None,
            report_config=config.report, research_config=config.research,
            output_dir=Path(tmp), session_id="len")
        return runner, llm

    def test_section_target_splits_quota_after_render_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, _ = self._runner(tmp, length_mode="fixed",
                                     preferred_body_chars=4000)
            runner.render_only_sections.add("要旨")
            current = {"1": "あ" * 6000, "2": "い" * 2000, "要旨": "う" * 800}
            # 4000 - 800 (要旨 cannot shrink) = 3200 split 3:1
            self.assertEqual(runner._section_length_target("1", current), (6000, 2400))
            self.assertEqual(runner._section_length_target("2", current), (2000, 800))
            self.assertIsNone(runner._section_length_target("要旨", current))

    def test_length_edit_prompt_states_the_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, llm = self._runner(tmp, length_mode="fixed",
                                       preferred_body_chars=4000)
            current = {"1": "あ" * 6000, "2": "い" * 2000}
            out = runner._edit_section("1", [], current, mode="length")
            self.assertEqual(out, "短くした本文")
            prompt = llm.generate.call_args[0][0]
            self.assertIn("約6,000字", prompt)
            self.assertIn("約3,000字", prompt)
            # a section already within its share is left alone
            llm.generate.reset_mock()
            self.assertIsNone(runner._edit_section(
                "2", [], {"1": "あ" * 2000, "2": "い" * 2000}, mode="length"))
            llm.generate.assert_not_called()

    def test_adaptive_compress_has_no_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner, llm = self._runner(tmp)
            self.assertIsNone(runner._section_length_target("1", {"1": "あ" * 100}))
            self.assertIsNone(runner._edit_section("1", [], {"1": "あ" * 100},
                                                   mode="length"))


class TestReferenceList(unittest.TestCase):
    def _locker(self, tmp):
        from deep_research_tool.evidence.locker import EvidenceLocker
        locker = EvidenceLocker(research_id="r", output_dir=Path(tmp))
        ids = []
        for i in range(5):
            ev = locker.add_evidence(
                url=f"https://example.com/{i}", title=f"資料{i}",
                content_excerpt=f"内容{i}", section_reference="1")
            ids.append(getattr(ev, "id", ev))
        return locker, ids

    def test_reference_view_lists_cited_sources_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            locker, ids = self._locker(tmp)
            self.assertEqual(len(locker.get_reference_evidence()), 5)
            locker.reorder([ids[3], ids[1]])
            locker.set_reference_ids([ids[3], ids[1], "missing"])
            self.assertEqual([e.id for e in locker.get_reference_evidence()],
                             [ids[3], ids[1]])
            # the locker itself keeps everything (evidence files, 根拠 view)
            self.assertEqual(len(locker.get_all_evidence()), 5)
            locker.set_reference_ids(None)
            self.assertEqual(len(locker.get_reference_evidence()), 5)

    def test_markdown_references_only_cited(self):
        from deep_research_tool.report.generator import ReportGenerator
        from deep_research_tool.report.generator import ReportFormat
        from deep_research_tool.research.researcher import ResearchSession
        with tempfile.TemporaryDirectory() as tmp:
            locker, ids = self._locker(tmp)
            locker.reorder([ids[2]])
            locker.set_reference_ids([ids[2]])
            session = ResearchSession(session_id="s1", query="市場")
            session.section_contents = {"1": {"title": "章", "content": "本文 [1]"}}
            gen = ReportGenerator(output_dir=Path(tmp), language="ja")
            path = gen.generate_report(session, locker, format=ReportFormat.MARKDOWN,
                                       filename="refs")
            text = Path(path).read_text(encoding="utf-8")
            refs = text.split("## References", 1)[1]
            self.assertIn("資料2", refs)
            for other in ("資料0", "資料1", "資料3", "資料4"):
                self.assertNotIn(other, refs)


class TestLengthSummary(unittest.TestCase):
    def _tool(self, **cfg):
        import deep_research_tool.main as main_mod
        tool = main_mod.DeepResearchTool.__new__(main_mod.DeepResearchTool)
        tool.config = create_config(**cfg)
        return tool

    def test_verified_body_count_is_reported(self):
        tool = self._tool(length_mode="fixed", preferred_body_chars=4000)
        tool.finalization_outcome = {"length_report": {"body_chars": 12000}}
        s = tool._length_summary(None)
        self.assertEqual(s["body_chars"], 12000)
        self.assertEqual(s["target"], 4000)
        self.assertTrue(s["over"])
        self.assertFalse(s["within"])

    def test_unverified_body_is_counted_from_the_session(self):
        tool = self._tool(length_mode="fixed", preferred_body_chars=4000)
        tool.finalization_outcome = None
        session = MagicMock()
        session.section_contents = {"1": {"content": "あ" * 3900},
                                    "_executive_summary": {"content": "x" * 999}}
        s = tool._length_summary(session)
        self.assertEqual(s["body_chars"], 3900)
        self.assertTrue(s["within"])

    def test_adaptive_summary(self):
        tool = self._tool()
        tool.finalization_outcome = {"length_report": {"body_chars": 9000}}
        s = tool._length_summary(None)
        self.assertEqual(s["mode"], "adaptive")
        self.assertNotIn("over", s)


if __name__ == "__main__":
    unittest.main()
