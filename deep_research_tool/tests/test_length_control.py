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


def _plan_json(mains=5, subs=3, titles=None, sub_titles=None):
    """A planner response; titles / sub_titles(i) name the sections the
    way a planner that follows the user's table of contents does."""
    import json
    toc = []
    for i in range(1, mains + 1):
        title = titles[i - 1] if titles else f"炭素繊維 市場動向{i}"
        subs_t = (sub_titles(i) if sub_titles else
                  [f"炭素繊維 需要詳細{i}-{j}" for j in range(1, subs + 1)])
        toc.append({"section": str(i), "title": title,
                    "description": f"説明{i}",
                    "subsections": [{"section": f"{i}.{j}", "title": t,
                                     "description": "d"}
                                    for j, t in enumerate(subs_t, 1)]})
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
        # the planner follows the user's list (titles carry its items)
        qg, _ = self._qg(4000, _plan_json(
            7, 1, titles=[f"炭素繊維の市場{i}" for i in range(1, 8)]))
        reqs = "目次は以下の通り:\n" + "\n".join(f"{i}. 市場{i}" for i in range(1, 8))
        plan = qg.create_research_plan("炭素繊維", requirements=reqs)
        toc = plan.table_of_contents
        # all 7 of the user's chapters stay; the subsection the planner
        # added to each is folded into the chapter's description
        self.assertEqual([i.title for i in toc.items],
                         [f"炭素繊維の市場{i}" for i in range(1, 8)])
        self.assertEqual(len(toc.get_flat_sections()), 7)
        self.assertIn("炭素繊維 需要詳細1-1", toc.items[0].description)

    def test_explicit_toc_detection(self):
        from deep_research_tool.research.query_generator import QueryGenerator
        qg = QueryGenerator(MagicMock(), language="ja")
        for text in ("GPT-3.5以降のLLM市場", "経営層向け・簡潔に", "2023.の決算を重視",
                     "Focus on Asia, i.e. China", "市場規模は3.5兆円、成長率2.1%",
                     "**重要**: a\n**必須**: b\n**任意**: c", "2024 年\n2025 年",
                     "国内製薬企業の第Ⅰ相・第Ⅱ相・第Ⅲ相試験", "Compare Phase I. Phase II. and Phase III. rates",
                     "1 位 TSMC\n2 位 Samsung\n3 位 Intel", "○○\n○○\n○○"):
            self.assertFalse(qg._has_explicit_toc(text, ""), text)
        for text in ("1. 市場規模 2. 競合 3. 規制", "・市場規模\n・競合\n・規制",
                     "第1章 市場 第2章 競合 第3章 規制", "①市場②競合③規制",
                     "章立て: 市場と競合", "下記の構成で: 市場、競合、規制",
                     "1 市場概況\n1.1 市場規模\n2 主要企業",
                     "I. Market II. Players III. Pricing",
                     "1.市場規模。2.競合。3.規制。", "1.市場規模/2.競合/3.規制",
                     "Ⅰ．市場\nⅡ．競合\nⅢ．規制"):
            self.assertTrue(qg._has_explicit_toc("炭素繊維", text), text)

    def test_text_that_only_looks_like_a_toc_never_loses_a_chapter(self):
        # fix review F1: the broad detector fired (the planner was told to
        # follow the user's structure) but no explicit list was found —
        # subsections are folded, but no main section is dropped
        chapters = ["市場", "競合", "価格", "規制", "技術", "課題", "提言"]
        qg, _ = self._qg(4000, _plan_json(
            7, 3, titles=[f"炭素繊維の{c}" for c in chapters]))
        plan = qg.create_research_plan(
            "炭素繊維", requirements="章の構成は市場、競合、価格、規制、技術、課題、提言で")
        toc = plan.table_of_contents
        self.assertEqual(len(toc.items), 7)
        self.assertTrue(all(not i.subsections for i in toc.items))

    def test_the_users_chapters_are_found_in_any_list_format(self):
        # post-merge review: whatever the format, chapters whose titles
        # carry the user's items are kept; subsections the planner
        # invented are folded, the user's own subsections are kept
        chapters = ["市場概況", "主要企業", "価格動向", "規制", "技術動向", "課題", "提言"]
        for reqs in ("下記の構成でまとめてください：" + "、".join(chapters),
                     "".join(f"{'①②③④⑤⑥⑦'[i]}{c}" for i, c in enumerate(chapters)),
                     "\n".join(f"{i}）{c}" for i, c in enumerate(chapters, 1))):
            qg, _ = self._qg(4000, _plan_json(7, 2, titles=chapters))
            plan = qg.create_research_plan("炭素繊維", requirements="目次: " + reqs)
            self.assertEqual([i.title for i in plan.table_of_contents.items],
                             chapters, reqs)
        # a ToC pasted from Word, with the user's subsections
        reqs = "目次\n1 市場概況\n1.1 市場規模\n1.2 成長率\n2 主要企業\n2.1 国内\n2.2 海外\n3 課題"
        qg, _ = self._qg(4000, _plan_json(
            3, 2, titles=["市場概況", "主要企業", "課題"],
            sub_titles=lambda i: {1: ["市場規模", "成長率"], 2: ["国内", "海外"],
                                  3: ["課題の詳細A", "課題の詳細B"]}[i]))
        plan = qg.create_research_plan("炭素繊維", requirements=reqs)
        flat = [s.title for s in plan.table_of_contents.get_flat_sections()]
        # the user's 3 chapters and 4 subsections stay; the 2 the planner
        # invented under 課題 are folded
        self.assertEqual(flat, ["市場概況", "市場規模", "成長率", "主要企業",
                                "国内", "海外", "課題"])

    def test_a_list_in_either_field_is_the_users(self):
        # #93 review H0/H4/H12: a ToC in the query was cut when the
        # requirements also tripped the broad detector, and vice versa
        chapters = ["市場規模", "主要企業", "技術動向", "規制", "課題", "今後の展望"]
        for query, reqs in (
                ("国内EV市場について、次の6章で調査: "
                 + " ".join(f"{i}. {c}" for i, c in enumerate(chapters, 1)),
                 "経営層向け・簡潔に"),
                ("国内EV市場の動向・見通し",
                 "".join(f"{'①②③④⑤⑥'[i]}{c}" for i, c in enumerate(chapters)))):
            qg, _ = self._qg(4000, _plan_json(6, 2, titles=chapters))
            plan = qg.create_research_plan(query, requirements=reqs)
            self.assertEqual([i.title for i in plan.table_of_contents.items],
                             chapters, query)

    def test_nested_and_long_items_are_still_the_users(self):
        # #93 review H1 (技術 / 技術課題) and H13 (long English chapters)
        nested = ["概要", "政策", "技術", "技術課題", "市場", "市場予測"]
        qg, _ = self._qg(4000, _plan_json(6, 0, titles=nested))
        plan = qg.create_research_plan(
            "全固体電池", requirements="目次: " + " ".join(
                f"{i}. {c}" for i, c in enumerate(nested, 1)))
        self.assertEqual([i.title for i in plan.table_of_contents.items], nested)
        long_titles = ["Market size and growth outlook through 2030",
                       "Competitive landscape and market share analysis",
                       "Regulatory environment in the United States and EU",
                       "Technology roadmap for solid-state batteries",
                       "Supply chain risks and raw material availability",
                       "Strategic recommendations for new market entrants"]
        qg, _ = self._qg(4000, _plan_json(6, 2, titles=long_titles))
        plan = qg.create_research_plan(
            "solid-state batteries", requirements="Use the following structure:\n"
            + "\n".join(f"{i}. {t}" for i, t in enumerate(long_titles, 1)))
        self.assertEqual([i.title for i in plan.table_of_contents.items], long_titles)

    def _run_plan(self, query, reqs, mains):
        import json as _json
        toc = [{"section": str(i), "title": t, "description": "d",
                "subsections": [{"section": f"{i}.{j}", "title": st, "description": "d"}
                                for j, st in enumerate(subs, 1)]}
               for i, (t, subs) in enumerate(mains, 1)]
        qg, _ = self._qg(4000, _json.dumps(
            {"title": "x", "summary": "s", "table_of_contents": toc,
             "search_queries": [f"q{i}" for i in range(30)], "key_terms": [],
             "suggested_sources": []}, ensure_ascii=False))
        plan = qg.create_research_plan(query, requirements=reqs)
        return [(i.title, [x.title for x in i.subsections])
                for i in plan.table_of_contents.items]

    def test_the_cap_grows_to_the_size_of_the_users_list(self):
        # #93 review round 4: repeated subsection names (I0), 「課題と提言」
        # and space-separated lists (I1), reworded titles (I14)
        user = [("国内市場", ["現状", "課題"]), ("海外市場", ["現状", "課題"]), ("提言", [])]
        self.assertEqual(self._run_plan(
            "全固体電池", "目次:\n1. 国内市場\n1.1 現状\n1.2 課題\n2. 海外市場\n"
            "2.1 現状\n2.2 課題\n3. 提言", user), user)
        seven = ["市場規模", "主要企業", "技術動向", "規制", "価格", "課題", "提言"]
        for reqs in ("目次: 市場規模、主要企業、技術動向、規制、価格、課題と提言",
                     "目次：市場規模　主要企業　技術動向　規制　価格　課題　提言"):
            got = self._run_plan("国内EV市場", reqs, [(t, []) for t in seven])
            self.assertEqual([t for t, _ in got], seven, reqs)
        reworded = ["国内EV市場の規模と成長", "主要プレイヤーの戦略", "技術開発の動向",
                    "政策と規制の枠組み", "価格の推移", "今後の課題"]
        got = self._run_plan("国内EV市場", "目次:\n1. 市場規模\n2. 主要企業\n3. 技術\n"
                             "4. 規制\n5. 価格\n6. 課題", [(t, ["a", "b"]) for t in reworded])
        self.assertEqual(got, [(t, []) for t in reworded])

    def test_nothing_but_the_users_list_raises_the_cap(self):
        # #93 review round 4: purposes (I6), a ・ inside a chapter name
        # (I7), the query's preamble (I8) never add sections
        for query, reqs, titles in (
                ("国内EV市場の動向", "目的: 投資判断\n読者: 経営層\n必須の観点: 価格、補助金、充電網",
                 ["投資判断の前提", "経営層向け概要", "価格", "補助金", "充電網"]),
                ("国内EV市場", "目次:\n1. 市場・競合\n2. 価格・コスト\n3. 規制\n4. 技術\n5. 展望",
                 ["市場・競合", "価格・コスト", "規制", "技術", "展望"]),
                ("EV市場: 1. 規模 2. 企業 3. 規制", "",
                 ["EV市場の概観", "規模", "企業", "規制", "展望"])):
            got = self._run_plan(query, reqs, [(t, ["市場", "競合", "x"]) for t in titles])
            self.assertEqual(sum(1 + len(s) for _, s in got), 5, (query, got))

    def test_topic_words_never_protect_subsections_or_late_chapters(self):
        # #93 review H2/H3/H5: topic words matched one chapter and one
        # subsection each; a 'お任せ' requirement with keyword 章構成 kept
        # a 20-section plan; late protected chapters pushed the plan over
        qg, _ = self._qg(4000, _plan_json(
            5, 3, titles=["炭素繊維の市場規模", "主要メーカーの動向", "用途別需要",
                          "価格とコスト", "CFRP市場の展望"],
            sub_titles=lambda i: [f"炭素繊維の生産量{i}", f"CFRP市場向け需要{i}",
                                  f"詳細{i}"]))
        plan = qg.create_research_plan("炭素繊維・CFRP市場")
        self.assertEqual(len(plan.table_of_contents.get_flat_sections()), 5)
        qg, _ = self._qg(4000, _plan_json(
            5, 3, titles=["市場規模", "競合", "価格", "需要", "展望"]))
        plan = qg.create_research_plan(
            "国内EV市場の動向", requirements="章構成はお任せします。市場規模・競合・価格を重視。")
        self.assertEqual(len(plan.table_of_contents.get_flat_sections()), 5)
        qg, _ = self._qg(4000, _plan_json(
            7, 0, titles=["概況", "需要", "供給", "価格", "市場規模", "競合", "規制"]))
        plan = qg.create_research_plan(
            "国内EV市場", requirements="必ず含める章: 市場規模、競合、規制")
        titles = [i.title for i in plan.table_of_contents.items]
        self.assertEqual(len(titles), 5)
        for c in ("市場規模", "競合", "規制"):
            self.assertIn(c, titles)

    def test_topic_words_and_purposes_never_protect_a_plan(self):
        # post-merge review G2/G3/G4: topic text that trips the broad
        # detector (・, 第Ⅲ相, stock) with a purposes list, or Roman numerals
        # in prose, must not switch the cap off
        for query, reqs in (
                ("国内EV市場の動向・見通し", "- 経営層向け\n- 日本市場に限定\n- 2030年までの見通し"),
                ("国内製薬企業の第Ⅰ相・第Ⅱ相・第Ⅲ相試験パイプライン", ""),
                ("Tesla stock outlook 2025", "- Audience: executives\n- Keep it concise\n- Compare vendors"),
                ("炭素繊維 市場動向・主要プレイヤー", "1 位 TSMC\n2 位 Samsung\n3 位 Intel")):
            qg, _ = self._qg(4000, _plan_json(5, 3))
            plan = qg.create_research_plan(query, requirements=reqs)
            self.assertEqual(len(plan.table_of_contents.get_flat_sections()), 5, query)

    def test_a_list_of_purposes_keeps_the_cap(self):
        # fix review F2/F10: '- ' bullets the planner never receives as a
        # ToC no longer switch the cap off
        for reqs in ("- 経営層向け\n- 日本市場に限定\n- 2030年までの見通し",
                     "1）経営層向け\n2）簡潔に\n3）主要3社の比較",
                     "Focus on: 1) executives, 2) brevity, 3) pricing"):
            qg, _ = self._qg(4000, _plan_json(5, 3))
            plan = qg.create_research_plan("炭素繊維", requirements=reqs)
            self.assertEqual(len(plan.table_of_contents.get_flat_sections()), 5, reqs)


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

    def test_abbreviations_and_terminator_runs_do_not_split(self):
        # fix review F0: 'U.S.', 'No. 1', 'approx. 32%', 'Corp. and' and
        # 'Dr. Smith' were split points; '...', '?!', '！？' lost characters
        import re
        from deep_research_tool.main import DeepResearchTool
        split = DeepResearchTool._split_sentences
        cases = {
            "The U.S. remains the largest market [1]. Next one.": 2,
            "It is No. 1 in share, holding approx. 32% of it. Done.": 2,
            "Key players, e.g. Acme Corp. and Beta Inc., grew. Then Dr. Smith said so.": 2,
            "Growth slowed... Then it recovered! Really?! Yes.": 4,
            "本当に持続するのか！？需要は回復した。前年比３．５％増。": 3,
            "See example.com for details. Sales hit 12.75bn.": 2,
        }
        for text, n in cases.items():
            parts = split(text)
            self.assertEqual(len(parts), n, parts)
            self.assertEqual(re.sub(r"\s+", "", "".join(parts)),
                             re.sub(r"\s+", "", text))          # nothing lost

    def test_quotes_urls_and_tight_citations(self):
        # post-merge review G0/G1/G7
        import re
        from deep_research_tool.main import DeepResearchTool
        split = DeepResearchTool._split_sentences
        cases = {
            "Demand grew.[1] Supply fell.[2] Prices rose.[3]": 3,
            "Demand grew.[SOURCE 1] Supply fell.[SOURCE 2] Prices rose.": 3,
            'He said "growth will continue." Then prices fell.': 2,
            "社長は「需要は回復した。今後も伸びる？」と述べた。次の文。": 2,
            "See https://example.com/a?q=1&b=2 for data. Next one.": 2,
        }
        for text, n in cases.items():
            parts = split(text)
            self.assertEqual(len(parts), n, parts)
            self.assertEqual(re.sub(r"\s+", "", "".join(parts)),
                             re.sub(r"\s+", "", text))
        self.assertTrue(split("Demand grew.[1] Supply fell.")[0]
                        .strip().endswith("[1]"))

    def test_abbreviation_quotes_lists_and_unclosed_brackets(self):
        # #93 review H6/H10 (Inc.'s, (etc.), Corp.)), H7/H11 (a ? inside
        # English quotes), H8 (an unclosed 「 swallowed the conclusion),
        # H9 (a list number became its own sentence)
        import re
        from deep_research_tool.main import DeepResearchTool
        split = DeepResearchTool._split_sentences
        cases = {
            "Acme Inc.'s revenue rose. Next.": 2,
            "Prices rose (etc.) last year. Next.": 2,
            "Toyota (Toyota Motor Corp.) grew. Next.": 2,
            'He asked "is it growing?" and then left. Next.': 2,
            "1. Demand rose.\n2. Supply fell.": 2,
        }
        for text, n in cases.items():
            parts = split(text)
            self.assertEqual(len(parts), n, parts)
            self.assertEqual(re.sub(r"\s+", "", "".join(parts)),
                             re.sub(r"\s+", "", text))
        tool = self._tool(length_mode="fixed", preferred_body_chars=4000)
        kept = tool._fit_deep_think_conclusion("「注目点。" + "需要は回復した。" * 60, 5)
        self.assertTrue(kept)                     # not swallowed whole
        self.assertLessEqual(len(kept), 160)

    def test_splitter_round_four(self):
        # #93 review round 4: no lowercase merge after a real sentence end
        # (I9), a stray opener with a later bracket pair (I10), a quote
        # spanning sentences (I11), full-width / indented list numbers (I12)
        import re
        from deep_research_tool.main import DeepResearchTool
        split = DeepResearchTool._split_sentences
        cases = {
            "需要は回復した。iPhoneの販売も伸びた。次の文。": 3,
            "a) first point. b) second point.": 2,
            "Is it growing? eBay says yes. Next.": 3,
            "（注：未確認の数値を含む。需要は回復した。価格は上昇した。（参考）次の文。": 4,
            '“Growth will continue. Prices will rise,” the CEO said. Next.': 2,
            '"Growth will continue. Prices will rise," the CEO said. Next.': 2,
            "１．需要は回復した。\n２．供給は減少した。": 2,
            "  4. Demand rose.\n  5. Supply fell.": 2,
        }
        for text, n in cases.items():
            parts = split(text)
            self.assertEqual(len(parts), n, parts)
            self.assertEqual(re.sub(r"\s+", "", "".join(parts)),
                             re.sub(r"\s+", "", text))
            for p in parts:
                self.assertIsNone(re.match(r"^\s*[0-9０-９]{1,2}[.．]\s*$", p), parts)

    def test_a_first_sentence_over_the_cap_adds_nothing(self):
        tool = self._tool(length_mode="fixed", preferred_body_chars=4000)
        text = ("The U.S. remains the largest market by far, " + "x" * 150
                + " [1]. Next sentence.")
        self.assertEqual(tool._fit_deep_think_conclusion(text, 5), "")

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

    def _accept_then(self, after, length_fn=None, passes=2, chapters=None):
        """ACCEPT on a 16,000-char body; the re-verified body after the
        length pass is judged by after(chapters)."""
        chapters = chapters or {"1": "あ" * 16000}

        def verify(ch):
            return after(ch) if len(ch["1"]) < 16000 else _verdict_for(ch)
        ctrl = FinalizationController(
            verify_fn=verify,
            length_fn=length_fn or (lambda sid, issues: chapters[sid][:4000]),
            hedge_fn=lambda sid, issues: None,
            budget=LoopBudget(max_final_research_rounds=0,
                              max_final_revision_rounds=0),
            fixed_target_chars=4000, max_length_passes=passes)
        return ctrl, ctrl.run(chapters)

    def test_an_important_gap_after_the_length_pass_is_not_an_accept(self):
        # fix review F3: only the hard gates were re-checked, so an
        # important unsupported claim (support 0.55) still shipped 'accept'
        issue = VerificationIssue(section_id="1", claim_id="C-1",
                                  type=ISSUE_UNSUPPORTED, severity="important",
                                  claim="主張", reason="根拠が削られた")
        _ctrl, out = self._accept_then(
            lambda ch: _verdict_for(ch, [issue], support=0.55))
        self.assertEqual(out["decision"], "finalize_with_limitations")

    def test_a_length_pass_that_undershoots_the_band_is_not_an_accept(self):
        chapters = {"1": "あ" * 16000}
        _ctrl, out = self._accept_then(
            lambda ch: _verdict_for(ch),
            length_fn=lambda sid, issues: chapters[sid][:1200], chapters=chapters)
        self.assertEqual(out["decision"], "finalize_with_limitations")
        self.assertTrue(out["length_report"]["under"])

    def test_the_fallback_keeps_one_pass_to_fit_the_note(self):
        # fix review F5: with one pass (fast profile) the ACCEPT pass used
        # it up, and the note then pushed body + note over the band
        chapters = {"1": "あ" * 16000}
        issues = self._issues(12)
        holder = {}

        def length_fn(sid, _issues):
            r = holder["ctrl"].length_reserve_chars
            # the writer lands high in the band unless told to keep room
            return chapters[sid][: (4600 if r == 0 else 4000 - r)]

        def after(ch):
            v = _verdict_for(ch, issues, support=0.5)
            v.metrics.unsupported_critical_claims = 1
            return v

        def verify(ch):
            return after(ch) if len(ch["1"]) < 16000 else _verdict_for(ch)
        ctrl = FinalizationController(
            verify_fn=verify, length_fn=length_fn,
            hedge_fn=lambda sid, issues: None,
            budget=LoopBudget(max_final_research_rounds=0,
                              max_final_revision_rounds=0),
            fixed_target_chars=4000, max_length_passes=1)
        holder["ctrl"] = ctrl
        out = ctrl.run(chapters)
        self.assertEqual(out["decision"], "finalize_with_limitations")
        self.assertTrue(out["length_report"]["within"], out["length_report"])
        self.assertEqual(sum(1 for h in out["history"]
                             if h["decision"] == "compress_to_target"), 2)

    def test_a_short_note_lists_the_critical_points_first(self):
        # fix review F4: the verifier appends critical coverage failures
        # after every claim issue; the cap summarised them as 'ほか N件'
        minor = [VerificationIssue(section_id="1", claim_id=f"C-{i}",
                                   type=ISSUE_UNSUPPORTED, severity="minor",
                                   claim=f"細かな主張その{i}についての比較的長い一文である",
                                   reason="根拠が見つからない") for i in range(14)]
        critical = VerificationIssue(section_id="", claim_id="Q-1",
                                     type=ISSUE_UNSUPPORTED, severity="critical",
                                     claim="主要3社の2025年の市場シェアは何%か",
                                     reason="回答なし")
        ctrl = FinalizationController(
            verify_fn=lambda ch: _verdict_for(ch, minor + [critical], support=0.5),
            budget=LoopBudget(max_final_research_rounds=0,
                              max_final_revision_rounds=0),
            fixed_target_chars=2000)
        out = ctrl.run({"1": "あ" * 2000})
        note = out["chapters"]["1"].split("### 調査上の限界", 1)[1]
        self.assertIn("ほか", note)
        self.assertIn("主要3社の2025年の市場シェア", note)

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
            # no extra line under the quality statistics: it would be an
            # unfrozen addition that fails the semantic artifact check
            self.assertNotIn("件の集計", text)
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

    def test_the_hard_bound_warning_uses_the_reported_count(self):
        # fix review F6/F9: the unverified hard-limit warning counted the
        # raw session while result['length'] counted the rendered body
        import inspect
        import deep_research_tool.main as main_mod
        src = inspect.getsource(main_mod.DeepResearchTool.run)
        block = src[src.index('"LengthBounds"') - 1500: src.index('"LengthBounds"')]
        self.assertIn('self._length_summary(session, result)["body_chars"]', block)

    def test_the_shortfall_is_measured_after_deep_think(self):
        # fix review F11: with DeepThink on the drafts leave 20% for the
        # appended conclusion, so measuring before DeepThink always warned
        import inspect
        import deep_research_tool.main as main_mod
        src = inspect.getsource(main_mod.DeepResearchTool.run)
        self.assertGreater(src.index("self._note_length_shortfall(session"),
                           src.index('self._mark_stage("deep_think"'))

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

    def test_uncited_sources_keep_the_artifact_check_passing(self):
        # fix review F7: a line added under the quality statistics when
        # References lists only the cited sources failed the two-way
        # artifact check, so every verified V1 run ended 'failed'
        from deep_research_tool.tests.test_finalization_integration import (
            CLAIM_SENTENCE, SUPPORTED_EVIDENCE, FakeSearch, build_tool,
            long_body, make_llm, make_locker, make_plan, make_session)
        from deep_research_tool.utils.helpers import ResearchWarnings
        ResearchWarnings.reset()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plan = make_plan([("1", "市場シェア", "A社のシェアを明らかにする")])
            session = make_session(plan, {"1": {
                "title": "市場シェア",
                "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                "evidence": SUPPORTED_EVIDENCE}})
            extra = [("https://unused.example.com/x", "未使用の資料。" * 40)]
            locker = make_locker(tmp, list(SUPPORTED_EVIDENCE) + extra)
            locker.export_to_json()
            path = tmp / f"session_{session.session_id}.json"
            session.saved_artifacts = {"session": str(path)}
            session.save(path)
            tool = build_tool(tmp / "verify-job", make_llm(), FakeSearch())
            res = tool.verify_report(session_path=str(path))
            self.assertEqual(res.get("semantic_artifact_check"), "pass")
            self.assertEqual(res.get("run_status"), "completed")
            report = Path(res["report_path"]).read_text(encoding="utf-8")
            self.assertNotIn("unused.example.com", report.split("References")[-1])



class TestManualExecutiveSummary(unittest.TestCase):
    """#93 review G5/H15 (older bug): a verified V1 manual run failed the
    artifact check whenever the researcher wrote an executive summary —
    the renderer printed it but the freeze never covered it."""

    def test_the_summary_is_frozen_and_the_check_passes(self):
        self._run("markdown", "本調査は市場の構造と主要企業の動向を整理し、需要の見通しを示したものである。"
                  "価格動向と規制の影響も併せて検討し、今後三年間の注目点をまとめた。")

    def test_an_ampersand_in_the_pdf_summary(self):
        # #93 review round 4 (I13): the PDF printed 'R&D' as 'R&D;'
        self._run("pdf", "本調査は主要企業のR&D投資とM&Aの動向を整理し、需要の見通しを示したものである。"
                  "価格動向と規制の影響も併せて検討し、今後三年間の注目点をまとめた。")

    def _run(self, fmt, summary):
        from unittest.mock import patch
        from deep_research_tool.main import run_manual_research
        from deep_research_tool.tests.test_finalization_integration import (
            CLAIM_SENTENCE, SUPPORTED_EVIDENCE, FakeResearcher, long_body,
            make_llm, make_locker, make_plan, make_session)
        from deep_research_tool.utils.helpers import ResearchWarnings
        ResearchWarnings.reset()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plan = make_plan([("1", "市場シェア", "A社のシェアを明らかにする")])
            session = make_session(plan, {"1": {
                "title": "市場シェア",
                "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                "evidence": SUPPORTED_EVIDENCE}})
            session.section_contents["_executive_summary"] = {
                "executive_summary": summary,
                "key_findings": ["主要企業の国内シェアは上位三社で過半を占めると整理される"],
                "recommendations": ["需要の変化を四半期ごとに確認し計画を見直すことが望ましい"]}
            locker = make_locker(tmp, SUPPORTED_EVIDENCE)
            fake = FakeResearcher(session, locker)
            fake.load_evidence_from_file = lambda **kw: None
            llm = make_llm()
            with patch("deep_research_tool.main.ManualResearcher", new=lambda **kw: fake), \
                    patch("deep_research_tool.main.get_client", new=lambda **kw: llm):
                result = run_manual_research(
                    evidence_file="dummy.csv", topic="市場調査", provider="openai",
                    api_key="sk-test", output_format=fmt, output_dir=str(tmp),
                    manual_toc_sections=None, auto_toc=True)
            self.assertEqual(result.get("semantic_artifact_check"), "pass")
            self.assertEqual(result.get("run_status"), "completed")
            if fmt == "markdown":
                report = Path(result["report_path"]).read_text(encoding="utf-8")
                self.assertEqual(report.count(summary[:30]), 1)   # rendered once

if __name__ == "__main__":
    unittest.main()
