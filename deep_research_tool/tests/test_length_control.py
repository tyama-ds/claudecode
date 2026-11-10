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

    def test_main_hands_target_and_ceiling_to_the_researcher(self):
        import inspect
        import deep_research_tool.main as main_mod
        text = inspect.getsource(main_mod.DeepResearchTool)
        self.assertIn("target_characters=self.config.report.writing_target_chars(", text)
        self.assertIn("max_body_chars=self.config.report.writing_ceiling_chars()", text)
        self.assertIn("deep_think_share=self._deep_think_share()", text)

    def test_ceiling_never_overrides_a_smaller_page_target(self):
        # review C1: target_pages=2 with a 100,000 ceiling drafted to 90,000
        cfg = create_config(target_pages=2, hard_max_body_chars=100000)
        self.assertEqual(cfg.report.writing_target_chars("ja"), 3000)
        self.assertEqual(cfg.report.writing_target_chars("en"), 5000)
        self.assertEqual(cfg.report.writing_budget_chars("ja"), 3000)

    def test_a_ceiling_alone_is_a_bound_not_a_target(self):
        cfg = create_config(hard_max_body_chars=60000)
        self.assertIsNone(cfg.report.writing_target_chars())
        self.assertEqual(cfg.report.writing_ceiling_chars(), 54000)
        # steps that only cap still see it
        self.assertEqual(cfg.report.writing_budget_chars(), 54000)
        # and a target above the ceiling is clamped under it
        cfg = _fixed_config(hard_max_body_chars=3000)
        self.assertEqual(cfg.report.writing_target_chars(), 2700)


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

    def _run_sizing(self, r, sections=5):
        """Run _conduct_research_loop up to the section workers, which a
        sentinel stops: only the sizing step runs (no LLM, no search)."""
        from deep_research_tool.research.query_generator import QueryGenerator
        from deep_research_tool.research.researcher import ResearchSession
        llm = MagicMock()
        llm.generate.return_value = MagicMock(content=_plan_json(sections, 0))
        qg = QueryGenerator(llm, language="ja")
        plan = qg.create_research_plan("炭素繊維")
        self.assertEqual(len(plan.table_of_contents.get_flat_sections()), sections)
        r.session = ResearchSession(session_id="s", query="炭素繊維")
        r.session.research_plan = plan

        class _Stop(Exception):
            pass

        def _stop(*_a, **_k):
            raise _Stop()
        r._section_workers = _stop
        with self.assertRaises(_Stop):
            r._conduct_research_loop()
        return r.content_extractor

    def test_each_section_is_drafted_to_its_share(self):
        ex = self._run_sizing(self._researcher(target_characters=4000))
        self.assertEqual(ex.target_chars_per_section, 800)
        self.assertEqual(ex._outline_points_range(), (2, 3))

    def test_deep_think_keeps_its_part_of_the_share_free(self):
        # review C3: drafts at the full share + the appendix overshot by ~25%
        ex = self._run_sizing(self._researcher(target_characters=4000,
                                               deep_think_share=0.2))
        self.assertEqual(ex.target_chars_per_section, 640)

    def test_a_ceiling_caps_sections_without_asking_for_more(self):
        r = self._researcher(max_body_chars=10000)
        self.assertEqual(r._body_budget(), 10000)          # sizes the plan
        self.assertIsNone(r._body_target())
        ex = self._run_sizing(r)
        self.assertIsNone(ex.target_chars_per_section)
        self.assertEqual(ex.max_chars_per_section, 2000)
        rule = ex._length_rule()
        self.assertIn("2,000字以内", rule)
        self.assertNotIn("約", rule)

    def test_page_target_is_not_replaced_by_the_ceiling(self):
        r = self._researcher(target_pages=2, max_body_chars=90000)
        self.assertEqual(r._body_budget(), 3000)
        self.assertEqual(r.query_generator.body_budget_chars, 3000)


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

    def test_middle_tier_gets_three_to_four_points(self):
        ex, _ = self._extractor(1200)
        self.assertEqual(ex._outline_points_range(), (3, 4))

    def test_the_upper_band_never_passes_the_ceiling(self):
        ex, _ = self._extractor(1000)
        ex.max_chars_per_section = 1100
        rule = ex._length_rule()
        self.assertIn("約1,000字（800〜1,100字）", rule)

    def test_a_large_ceiling_leaves_the_default_size(self):
        ex, _ = self._extractor(None)
        ex.max_chars_per_section = 5000
        self.assertEqual(ex._calc_target_per_point(5), 600)
        self.assertEqual(ex._outline_points_range(), (4, 6))


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

    def test_topic_text_does_not_switch_off_the_cap(self):
        # review C2: 「・」「構成」 or 'stock' (contains 'toc') in a query
        # read as a user ToC and left 20 sections in a 4,000-char plan
        for query in ("炭素繊維 市場動向・主要プレイヤー", "炭素繊維 需要の構成",
                      "炭素繊維 stock outlook"):
            qg, _ = self._qg(4000, _plan_json(5, 3))
            plan = qg.create_research_plan(query)
            self.assertEqual(len(plan.table_of_contents.get_flat_sections()), 5, query)

    def test_a_section_list_the_user_wrote_is_never_cut(self):
        qg, _ = self._qg(4000, _plan_json(7, 1))
        reqs = "目次は以下の通り:\n" + "\n".join(f"{i}. 市場{i}" for i in range(1, 8))
        plan = qg.create_research_plan("炭素繊維", requirements=reqs)
        self.assertEqual(len(plan.table_of_contents.get_flat_sections()), 14)

    def test_explicit_toc_detection(self):
        from deep_research_tool.research.query_generator import QueryGenerator
        qg = QueryGenerator(MagicMock(), language="ja")
        for text in ("GPT-3.5以降のLLM市場", "経営層向け・簡潔に", "2023.の決算を重視",
                     "Focus on Asia, i.e. China", "市場規模は3.5兆円、成長率2.1%"):
            self.assertFalse(qg._has_explicit_toc(text, ""), text)
        for text in ("1. 市場規模 2. 競合 3. 規制", "・市場規模\n・競合\n・規制",
                     "第1章 市場 第2章 競合 第3章 規制", "①市場②競合③規制",
                     "章立て: 市場と競合"):
            self.assertTrue(qg._has_explicit_toc("炭素繊維", text), text)


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
        # 4000 / 5 sections x 0.2 (the part the drafts left free) = 160
        self.assertLessEqual(len(kept), 160)
        self.assertGreater(len(kept), 100)
        self.assertTrue(kept.endswith("。"))

    def test_a_decimal_point_is_not_a_sentence_end(self):
        # review C0: '12.75' was cut to '12.' and the wrong figure shipped
        import re
        tool = self._tool(length_mode="fixed", preferred_body_chars=4000)
        # cap 160: the first sentence (131) plus "...12." (9) fits, the
        # rest of the number's sentence does not — the old split kept "12."
        for first, second in (
                ("あ" * 130 + "。", "市場規模は12.75兆円に達し、" + "い" * 40 + "。"),
                ("x" * 129 + ". ", "Sales hit 12.75bn " + "y" * 40 + ". ")):
            kept = tool._fit_deep_think_conclusion(first + second, 5)
            self.assertEqual(kept, first.strip())
            self.assertIsNone(re.search(r"\d[.．](?!\d)", kept), kept)
        self.assertEqual(
            len(tool._split_sentences("前年比3.5%増となった[2]。需要は回復した。")), 2)

    def test_a_citation_after_an_english_period_stays_with_its_sentence(self):
        from deep_research_tool.main import DeepResearchTool
        parts = DeepResearchTool._split_sentences("Demand grew. [1] Supply fell. [2] ")
        self.assertEqual(parts[0].strip(), "Demand grew. [1]")
        self.assertTrue(parts[1].strip().startswith("Supply fell."))

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

    def test_a_length_pass_that_breaks_the_gates_is_not_an_accept(self):
        # review C5: compression dropped a claim's support, but the run
        # still finished as 'accept' (quality 'passed') without a note
        chapters = {"1": "あ" * 16000}
        issue = VerificationIssue(section_id="1", claim_id="C-1",
                                  type=ISSUE_UNSUPPORTED, severity="critical",
                                  claim="主張", reason="根拠が削られた")

        def verify(ch):
            if len(ch["1"]) < 16000:          # after the length pass
                v = _verdict_for(ch, [issue], support=0.4)
                v.metrics.unsupported_critical_claims = 1
                return v
            return _verdict_for(ch)
        hedged = []

        def hedge(sid, issues):
            hedged.append(sid)
            return None
        ctrl = FinalizationController(
            verify_fn=verify,
            length_fn=lambda sid, issues: chapters[sid][:4000],
            hedge_fn=hedge,
            budget=LoopBudget(max_final_research_rounds=0,
                              max_final_revision_rounds=0),
            fixed_target_chars=4000, max_length_passes=2)
        out = ctrl.run(chapters)
        self.assertEqual(out["decision"], "finalize_with_limitations")
        self.assertIn("調査上の限界", out["chapters"]["1"])
        self.assertEqual(hedged, ["1"])
        # the exit ran no further length passes (the body already fits)
        self.assertEqual(sum(1 for h in out["history"]
                             if h["decision"] == "compress_to_target"), 1)

    def test_an_accept_with_an_unchanged_body_is_kept(self):
        chapters = {"1": "あ" * 4000}
        ctrl = FinalizationController(
            verify_fn=lambda ch: _verdict_for(ch),
            length_fn=lambda sid, issues: None,
            fixed_target_chars=4000)
        self.assertEqual(ctrl.run(chapters)["decision"], "accept")

    def _issues(self, n):
        return [VerificationIssue(section_id="1", claim_id=f"C-{i}",
                                  type=ISSUE_UNSUPPORTED, severity="critical",
                                  claim=f"未確認の主張その{i}について述べた一文",
                                  reason="根拠が見つからない",
                                  search_queries=["q"]) for i in range(n)]

    def test_room_is_kept_for_the_limitations_note(self):
        # review C7: the body was fitted to the band, then the note pushed
        # it back over and the run warned about an overshoot
        chapters = {"1": "あ" * 18000}
        issues = self._issues(12)
        seen = []
        ctrl = None

        def length_fn(sid, _issues):
            # a writer that meets the per-section target exactly
            reserve = ctrl.length_reserve_chars
            seen.append(reserve)
            return chapters[sid][: 4000 - reserve]
        ctrl = FinalizationController(
            verify_fn=lambda ch: _verdict_for(ch, issues, support=0.5),
            length_fn=length_fn,
            budget=LoopBudget(max_final_research_rounds=0,
                              max_final_revision_rounds=0),
            fixed_target_chars=4000, max_length_passes=2)
        out = ctrl.run(chapters)
        lr = out["length_report"]
        self.assertGreater(seen[0], 0)
        self.assertTrue(lr["within"], lr)
        self.assertEqual(ctrl.length_reserve_chars, 0)       # reset after
        entry = [h for h in out["history"] if h["decision"] == "compress_to_target"][0]
        self.assertEqual(entry["reserved_for_note"], seen[0])

    def test_the_note_stays_small_under_a_fixed_target(self):
        chapters = {"1": "あ" * 3000}
        ctrl = FinalizationController(
            verify_fn=lambda ch: _verdict_for(ch, self._issues(60), support=0.5),
            budget=LoopBudget(max_final_research_rounds=0,
                              max_final_revision_rounds=0),
            fixed_target_chars=4000)
        out = ctrl.run(chapters)
        note = out["chapters"]["1"].split("### 調査上の限界", 1)[1]
        self.assertIn("ほか", note)
        self.assertLess(count_body_chars(note), 4000 * 0.15 + 120)
        # adaptive mode still lists every issue
        ctrl = FinalizationController(
            verify_fn=lambda ch: _verdict_for(ch, self._issues(60), support=0.5),
            budget=LoopBudget(max_final_research_rounds=0,
                              max_final_revision_rounds=0))
        out = ctrl.run({"1": "あ" * 3000})
        self.assertEqual(out["chapters"]["1"].count("- 未解決:"), 60)

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

    def test_a_section_under_its_share_is_never_told_to_shorten(self):
        # review C6: a redundancy compress of a 1,500-char section in a
        # 3,400-char body was told to 'shorten' to ~1,764 chars
        with tempfile.TemporaryDirectory() as tmp:
            runner, llm = self._runner(tmp, length_mode="fixed",
                                       preferred_body_chars=4000)
            current = {"1": "あ" * 1500, "2": "い" * 1900}
            runner._edit_section("1", [], current, mode="compress")
            prompt = llm.generate.call_args[0][0]
            self.assertNotIn("まで短くしてください", prompt)
            self.assertIn("重複・冗長", prompt)
            # over its share: the number is stated
            runner._edit_section("1", [], {"1": "あ" * 6000, "2": "い" * 2000},
                                 mode="compress")
            self.assertIn("約3,000字", llm.generate.call_args[0][0])

    def test_the_controller_reserve_shrinks_the_targets(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            runner, _ = self._runner(tmp, length_mode="fixed",
                                     preferred_body_chars=4000)
            current = {"1": "あ" * 6000, "2": "い" * 2000}
            runner._controller = SimpleNamespace(length_reserve_chars=800)
            # (4000 - 800) split 3:1
            self.assertEqual(runner._section_length_target("1", current), (6000, 2400))

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
            # review C11: the quality statistics say what they count
            self.assertIn("収集した 5 件の集計", text)
            self.assertIn("本文で引用した 1 件", text)
            # review C14: the generator keeps the session it rendered
            self.assertIs(gen.last_rendered_session, session)

    def test_live_report_lists_the_same_references(self):
        # review C9: the live panel / live .docx still listed every source
        from deep_research_tool.main import DeepResearchTool
        with tempfile.TemporaryDirectory() as tmp:
            locker, ids = self._locker(tmp)
            self.assertEqual(len(DeepResearchTool._reference_citations(locker)), 5)
            locker.reorder([ids[2], ids[4]])
            locker.set_reference_ids([ids[2], ids[4]])
            refs = DeepResearchTool._reference_citations(locker)
            self.assertEqual(len(refs), 2)
            self.assertIn("資料2", refs[0])
            self.assertIn("資料4", refs[1])


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
                                    "_meta": {"content": "x" * 999}}
        s = tool._length_summary(session)
        self.assertEqual(s["body_chars"], 3900)
        self.assertTrue(s["within"])

    def test_unverified_v1_counts_the_executive_summary(self):
        # review C13: V1 prints the summary as body text; the verified
        # path counts it as 要旨, the unverified count left it out
        tool = self._tool(length_mode="fixed", preferred_body_chars=4000)
        tool.finalization_outcome = None
        session = MagicMock()
        session.section_contents = {
            "1": {"content": "あ" * 3900},
            "_executive_summary": {"executive_summary": "い" * 999,
                                   "key_findings": [], "recommendations": []}}
        s = tool._length_summary(session)
        self.assertEqual(s["body_chars"], 4899)
        self.assertTrue(s["over"])
        self.assertFalse(s["within"])

    def test_unverified_v1_counts_what_was_rendered(self):
        # review C14: the legacy page adjustment renders a shrunk copy
        from types import SimpleNamespace
        tool = self._tool(target_pages=1)
        tool.finalization_outcome = None
        session = MagicMock()
        session.section_contents = {str(i): {"content": "あ" * 4000} for i in range(5)}
        shrunk = SimpleNamespace(section_contents={
            str(i): {"content": "あ" * 800} for i in range(5)})
        tool.report_generator = SimpleNamespace(last_rendered_session=shrunk)
        self.assertEqual(tool._length_summary(session)["body_chars"], 4000)

    def test_the_band_comes_from_the_finalization_report(self):
        # review C8: tolerance 0 was read as 0.20, so the UI said 'within'
        # while the warning said the body did not fit
        tool = self._tool(length_mode="fixed", preferred_body_chars=4000,
                          length_tolerance=0.0)
        tool.finalization_outcome = {"length_report": {
            "mode": "fixed", "target": 4000, "lower": 4000, "upper": 4000,
            "body_chars": 4300, "within": False, "over": True, "under": False}}
        s = tool._length_summary(None)
        self.assertEqual((s["lower"], s["upper"]), (4000, 4000))
        self.assertTrue(s["over"])
        tool.finalization_outcome = None
        session = MagicMock()
        session.section_contents = {"1": {"content": "あ" * 4300}}
        s = tool._length_summary(session)
        self.assertEqual((s["lower"], s["upper"]), (4000, 4000))
        self.assertTrue(s["over"])

    def test_adaptive_summary(self):
        tool = self._tool()
        tool.finalization_outcome = {"length_report": {"body_chars": 9000}}
        s = tool._length_summary(None)
        self.assertEqual(s["mode"], "adaptive")
        self.assertNotIn("over", s)


class TestEvidenceFillLength(unittest.TestCase):
    """review C4: the evidence-review fill for an empty section asked for
    'detailed, comprehensive' text with no size limit."""

    def _tool(self, **cfg):
        import deep_research_tool.main as main_mod
        tool = main_mod.DeepResearchTool.__new__(main_mod.DeepResearchTool)
        tool.config = create_config(**cfg)
        tool.llm_client = MagicMock()
        tool.llm_client.generate.return_value = MagicMock(content="本文" * 40)
        return tool

    def _prompt(self, tool, **kw):
        tool._generate_section_from_evidence(
            section_key="3", section_title="規制", evidence_overview="ev",
            re_extracted_data={}, all_sections_text="", query="炭素繊維", **kw)
        return tool.llm_client.generate.call_args[0][0]

    def test_the_fill_is_capped_at_the_share(self):
        prompt = self._prompt(self._tool(), max_chars=640)
        self.assertIn("640字以内", prompt)
        self.assertNotIn("網羅的に", prompt)

    def test_free_runs_keep_the_old_prompt(self):
        prompt = self._prompt(self._tool())
        self.assertIn("網羅的に", prompt)
        self.assertIn("詳細な内容を", prompt)
        self.assertNotIn("字以内", prompt)

    def test_the_enhancement_stays_inside_the_share(self):
        tool = self._tool(length_mode="fixed", preferred_body_chars=4000)
        tool._enhance_section_content(
            section_key="1", section_title="市場", current_content="あ" * 780,
            evidence_overview="ev", re_extracted_data={}, all_sections_text="",
            query="炭素繊維", section_share=800)
        self.assertIn("800字以内", tool.llm_client.generate.call_args[0][0])


class TestVerifyOnlyLength(unittest.TestCase):
    """review C16: verify-only runs fit a fixed body too, but their result
    had no 'length', so the UI showed nothing."""

    def test_verify_result_reports_the_length(self):
        from deep_research_tool.tests.test_finalization_integration import (
            FakeSearch, build_tool, make_llm)
        from deep_research_tool.tests.test_verify_only import _saved_research
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            session_path = _saved_research(tmp)
            tool = build_tool(tmp / "verify-job", make_llm(), FakeSearch())
            result = tool.verify_report(session_path=str(session_path))
            self.assertIn("length", result)
            self.assertGreater(result["length"]["body_chars"], 0)
            lr = (tool.finalization_outcome or {}).get("length_report") or {}
            self.assertEqual(result["length"]["body_chars"], lr.get("body_chars"))


if __name__ == "__main__":
    unittest.main()
