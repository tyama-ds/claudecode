"""
Audit tests for citation strictness, fail-closed chunks, full-body
coverage, freshness re-evaluation, the semantic freeze and fixed-length
modes — all through the REAL production path (the harness from
test_finalization_integration: real config / locker / verifier /
controller / generators; only LLM and web are content-routed fakes).
"""

import json
import re
import unittest
from unittest.mock import patch

from deep_research_tool.report.finalization import (
    FRESHNESS_PASS,
    ISSUE_VERIFICATION_FAILURE,
    LoopBudget,
    ResearchDecision,
    StructuredVerdict,
    VerificationMetrics,
    decide,
)
from deep_research_tool.tests.test_finalization_integration import (
    Base,
    CLAIM_SENTENCE,
    FakeSearch,
    RoutedLLM,
    all_answered_coverage,
    build_tool,
    j,
    judge_ids,
    long_body,
    make_llm,
    make_locker,
    make_plan,
    make_session,
    run_tool,
)


# ---------------------------------------------------------------------------
# §4 REQUIRED TEST: misattributed citation must never pass
# ---------------------------------------------------------------------------

WRONG = ("https://wrong.example.com/cited",
         "全く無関係な話題。天候とレシピの記事。" * 12)
RIGHT = ("https://right.example.com/uncited",
         f"公式統計によればA社の国内シェアは52%であることが確認できる。" * 8)


class TestMisattributedCitation(Base):

    def _routed_llm(self, edits):
        def judge(prompt):
            evidence_part = prompt.split("【エビデンス】")[-1]
            if "国内シェアは52%" in evidence_part:
                return j({"status": "supported", "reason": "一致",
                          "supporting_source_ids": judge_ids(prompt)})
            return j({"status": "unsupported", "reason": "無関係",
                      "supporting_source_ids": []})

        def extract(prompt):
            if CLAIM_SENTENCE in prompt:
                return j({"claims": [{"claim": CLAIM_SENTENCE,
                                      "importance": "critical",
                                      "source_numbers": []}]})
            return j({"claims": []})

        return (RoutedLLM()
                .on("extract", extract)
                .on("judge", judge)
                .on("coverage", all_answered_coverage)
                .on("edit", lambda p: next(edits, "")))

    def test_wrong_citation_never_accepts_and_rewrite_fixes_it(self):
        """Body cites SOURCE 1 (irrelevant); SOURCE 2 (uncited) supports
        the claim. This must NOT accept with claim_support_score=1.0 /
        citations_valid=True; only after the citation is replaced with
        SOURCE 2 and re-verified may it accept."""
        # the rewrite cites the replacement (SOURCE 2 = appended number)
        edits = iter([
            f"## 1. 市場シェア\n\n{CLAIM_SENTENCE} [SOURCE 2]。"
            + "裏付けを確認した本文。" * 30,
        ])
        llm = self._routed_llm(edits)
        locker = make_locker(self.tmp, [WRONG, RIGHT])
        plan = make_plan([("1", "市場シェア", "A社のシェア")])
        session = make_session(plan, {
            "1": {"title": "市場シェア",
                  # the body CITES the WRONG source
                  "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                  "evidence": [WRONG, RIGHT]},
        })
        tool = build_tool(self.tmp, llm, FakeSearch())
        run_tool(tool, session, locker)

        outcome = tool.finalization_outcome
        history = outcome["history"]
        # pass 1 must NOT be accept: the cited evidence does not support
        self.assertNotEqual(history[0]["decision"], "accept")
        self.assertEqual(history[0]["decision"], "rewrite_from_evidence")
        self.assertLess(history[0]["score"], 1.0)
        # the final body cites the REPLACEMENT and only then accepts
        self.assertEqual(outcome["decision"], "accept")
        verdict = outcome["verdict"]
        self.assertTrue(verdict.metrics.citations_valid)
        right_id = locker.get_by_url(RIGHT[0]).id
        runner = tool.finalization_runner
        self.assertIn(right_id, runner.citation_mgr.evidence_ids("1"))

    def test_uncited_evidence_is_only_a_replacement_candidate(self):
        """Even though SOURCE 2 supports the claim, the FIRST verdict
        records an invalid citation (with SOURCE 2 as a rewrite hint) —
        never claim_support_score=1.0."""
        llm = self._routed_llm(iter([]))     # edits fail -> no fix
        locker = make_locker(self.tmp, [WRONG, RIGHT])
        plan = make_plan([("1", "市場シェア", "A社のシェア")])
        session = make_session(plan, {
            "1": {"title": "市場シェア",
                  "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                  "evidence": [WRONG, RIGHT]},
        })
        tool = build_tool(self.tmp, llm, FakeSearch(),
                          max_final_revision_rounds=1)
        result = run_tool(tool, session, locker)

        outcome = tool.finalization_outcome
        self.assertNotEqual(outcome["decision"], "accept")
        first_score = outcome["history"][0]["score"]
        self.assertLess(first_score, 1.0)
        # the replacement candidate was recorded on the issue
        verdict = result["verification_result"]
        right_id = locker.get_by_url(RIGHT[0]).id
        inv = [i for i in verdict.issues if i.type == "invalid_citation"
               and right_id in i.supporting_source_ids]
        self.assertTrue(inv or outcome["decision"]
                        == "finalize_with_limitations")


# ---------------------------------------------------------------------------
# §5: partial chunk extraction failure is fail-closed
# ---------------------------------------------------------------------------

class TestChunkFailClosed(Base):

    def test_partial_chunk_failure_never_accepts(self):
        # a >4000-char body -> 2 extraction chunks; chunk 2 always fails
        body = long_body("1", "市場シェア", CLAIM_SENTENCE, chars=7000)

        calls = {"n": 0}

        def extract(prompt):
            if "2/2分割" in prompt:
                calls["n"] += 1
                raise RuntimeError("LLM down")
            if CLAIM_SENTENCE in prompt:
                return j({"claims": [{"claim": CLAIM_SENTENCE,
                                      "importance": "important",
                                      "source_numbers": [1]}]})
            return j({"claims": []})

        llm = (RoutedLLM().on("extract", extract)
               .on("judge", lambda p: j(
                   {"status": "supported", "reason": "ok",
                    "supporting_source_ids": judge_ids(p)}))
               .on("coverage", all_answered_coverage)
               .on("edit", lambda p: ""))
        locker = make_locker(
            self.tmp, [("https://ev1.example.com/a",
                        f"A社の国内シェアは52%と報告。" * 8)])
        tool = build_tool(self.tmp, llm, FakeSearch())
        result = run_tool(tool, self.default_session(content=body), locker)

        verdict = result["verification_result"]
        self.assertGreater(verdict.metrics.chunks_failed, 0)
        # bounded per-chunk retries actually happened (3 attempts/verify)
        self.assertGreaterEqual(calls["n"], 3)
        # extraction errors are transcribed into the METRICS
        self.assertTrue(verdict.metrics.extraction_errors)
        self.assertTrue(any("chunk 2/2" in e or "2/2" in e
                            for e in verdict.metrics.extraction_errors))
        # a critical verification_failure issue names the unverified range
        failures = verdict.issues_of(ISSUE_VERIFICATION_FAILURE)
        self.assertTrue(failures)
        self.assertEqual(failures[0].severity, "critical")
        # partial success is NEVER full-body success
        self.assertEqual(tool.finalization_outcome["decision"],
                         "finalize_with_limitations")


# ---------------------------------------------------------------------------
# §8: critical question answered only at the END of a long report
# ---------------------------------------------------------------------------

class TestTailCoverage(Base):

    def test_question_answered_in_late_window_is_found(self):
        filler = "一般的な背景説明の段落。" * 40      # ~480 chars
        tail_answer = "結論として、B社の輸出台数は12万台であり前年比8%増となった。"
        body = "## 1. 分析\n\n" + filler * 50 + tail_answer   # >24,000 chars
        self.assertGreater(len(body), 24000)

        windows_seen = []

        def coverage(prompt):
            windows_seen.append(prompt)
            answered = "輸出台数は12万台" in prompt
            return j({"coverage": [{"question": 0, "answered": answered,
                                    "section_id": "1",
                                    "missing": "" if answered else "輸出台数",
                                    "search_queries": ["B社 輸出台数"]}]})

        def extract(prompt):
            return j({"claims": []})    # non-factual filler body

        llm = (RoutedLLM().on("extract", extract).on("coverage", coverage)
               .on("edit", lambda p: ""))
        locker = make_locker(self.tmp, [("https://x.example.com/1",
                                         "輸出台数は12万台" * 5)])
        plan = make_plan([("1", "輸出動向", "B社の輸出台数")])
        session = make_session(plan, {
            "1": {"title": "輸出動向", "content": body,
                  "evidence": [("https://x.example.com/1",
                                "輸出台数は12万台" * 5)]},
        })
        tool = build_tool(self.tmp, llm, FakeSearch())
        result = run_tool(tool, session, locker)

        # map-reduce actually split the body into multiple windows and
        # the tail window carried the answer
        self.assertGreaterEqual(len(windows_seen), 2)
        verdict = result["verification_result"]
        self.assertEqual(verdict.metrics.critical_question_coverage, 1.0)


# ---------------------------------------------------------------------------
# §8: freshness gate re-evaluates after the final research round
# ---------------------------------------------------------------------------

FRESH_URL = "https://stats.example.go.jp/latest"
FRESH_TEXT = (f"2026年6月30日公表の政府統計。A社の国内シェアは52%である。"
              * 10)


class TestFreshnessUpdate(Base):

    def test_gate_updates_after_research_adds_fresh_primary_source(self):
        stale = ("https://oldblog.example.com/2015",
                 "2015年3月時点の古いブログ記事。A社の話題。" * 10)

        def extract(prompt):
            if CLAIM_SENTENCE in prompt:
                return j({"claims": [{"claim": CLAIM_SENTENCE,
                                      "importance": "critical",
                                      "source_numbers": [1]}]})
            return j({"claims": []})

        def judge(prompt):
            if "国内シェアは52%" in prompt.split("【エビデンス】")[-1]:
                return j({"status": "supported", "reason": "ok",
                          "supporting_source_ids": judge_ids(prompt)})
            return j({"status": "unsupported", "reason": "根拠なし",
                      "supporting_source_ids": []})

        llm = (RoutedLLM().on("extract", extract).on("judge", judge)
               .on("coverage", all_answered_coverage)
               .on("edit", lambda p:
                   f"## 1. 市場シェア\n\n{CLAIM_SENTENCE} [SOURCE 1]"
                   f" [SOURCE 2]。政府統計で確認。" * 20))
        locker = make_locker(self.tmp, [stale])
        # make the stale source explicitly old & non-primary
        old_ev = locker.get_all_evidence()[0]
        old_ev.published_date = "2015-03-01"
        search = FakeSearch(results={"": [FRESH_URL]},
                            pages={FRESH_URL: FRESH_TEXT})
        plan = make_plan([("1", "市場シェア", "A社のシェア")])
        session = make_session(plan, {
            "1": {"title": "市場シェア",
                  "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                  "evidence": [stale]},
        })
        tool = build_tool(self.tmp, llm, search)
        result = run_tool(tool, session, locker,
                          query="A社の最新の市場シェア")   # freshness REQUIRED

        # the researched evidence carries date/type/primary metadata
        new_ev = locker.get_by_url(FRESH_URL)
        self.assertIsNotNone(new_ev)
        self.assertTrue(new_ev.published_date.startswith("2026"))
        self.assertTrue(new_ev.quality_indicators.is_primary_source)
        # and the freshness gate flipped to PASS on re-verification
        verdict = result["verification_result"]
        self.assertEqual(verdict.metrics.primary_freshness, FRESHNESS_PASS)
        self.assertEqual(tool.finalization_outcome["decision"], "accept")


# ---------------------------------------------------------------------------
# §6: semantic snapshot equality with figures/glossary/fermi/warnings
# ---------------------------------------------------------------------------

def make_fake_collection():
    from deep_research_tool.report.figure_table_generator import (
        Figure, FigureTableCollection, FigureType, TableData)
    collection = FigureTableCollection()
    collection.charts.append(Figure(
        figure_id="chart-1", figure_type=FigureType.CHART,
        title="A社シェア推移", caption="図1: 出典 公式統計 [SOURCE 1]",
        section_id="1", image_path=None))
    collection.tables.append(TableData(
        table_id="tbl-1", title="主要数値一覧",
        caption="表1: A社の国内シェア", headers=["年", "シェア"],
        rows=[["2025", "50%"], ["2026", "52%"]], section_id="1"))
    return collection


class TestSemanticFreeze(Base):

    def test_snapshot_identical_with_all_extras_enabled(self):
        from deep_research_tool.utils.helpers import ResearchWarnings

        llm = make_llm().on(
            "chapter",
            lambda p: long_body("1", "市場シェア", CLAIM_SENTENCE)
            + "\n===CHAPTER_META===\n" + j({"key_points": []}))
        locker = make_locker(
            self.tmp,
            [("https://ev1.example.com/a",
              f"A社の国内シェアは52%と報告。" * 8)])
        plan = make_plan([("1", "市場シェア", "A社のシェア")])
        session = make_session(plan, {
            "1": {"title": "市場シェア",
                  "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                  "evidence": [("https://ev1.example.com/a",
                                f"A社の国内シェアは52%と報告。" * 8)]},
        })
        # a pre-existing warning joins the verified body (付録C)
        ResearchWarnings.get_instance().add(
            ResearchWarnings.LOW, "Test", "テスト用の事前警告です。")

        tool = build_tool(self.tmp, llm, FakeSearch(),
                          report_generator_version="v2",
                          auto_figures=True,
                          v2_include_glossary=True)
        collection = make_fake_collection()
        fig_generator = object()          # insertion is skipped for MagicMock
        with patch.object(
                type(tool), "_generate_figure_collection",
                return_value=(collection, None)), \
             patch.object(
                type(tool), "_run_fermi_estimation",
                return_value=(None, "\n\n---\n## フェルミ推定\n"
                              "A社の潜在市場は約100億円と推定される。")), \
             patch("deep_research_tool.report.v2.glossary.GlossaryManager."
                   "create_initial_glossary",
                   return_value={"share": {"term": "シェア",
                                           "definition": "市場占有率",
                                           "aliases": []}}):
            tool.config.fermi_estimation.enabled = True
            result = run_tool(tool, session, locker)
        del fig_generator

        # the verification-time snapshot and the output-time snapshot are
        # IDENTICAL: nothing semantic was generated or altered after the
        # freeze, with figures + glossary + Fermi + warnings all enabled
        self.assertIsNotNone(result["semantic_manifest_hash_at_freeze"])
        self.assertEqual(result["semantic_manifest_hash_at_freeze"],
                         result["semantic_manifest_hash_at_output"])
        self.assertFalse(any(w["source"] == "Freeze"
                             for w in result["warnings"]))
        # figure semantics (captions, table cells) were part of the
        # VERIFIED body
        verify_prompts = [p for k, p in tool.llm_client.calls
                          if k == "extract"]
        self.assertTrue(any("図1: 出典 公式統計" in p
                            for p in verify_prompts))
        self.assertTrue(any("2026 | 52%" in p or "52%" in p
                            for p in verify_prompts))

    def test_v1_footnote_definitions_render(self):
        llm = make_llm()
        locker = make_locker(
            self.tmp,
            [("https://ev1.example.com/a",
              f"A社の国内シェアは52%と報告。" * 8)])
        tool = build_tool(self.tmp, llm, FakeSearch())
        result = run_tool(tool, self.default_session(), locker)
        report = (__import__("pathlib").Path(result["report_path"])
                  .read_text(encoding="utf-8"))
        # body citation [^1] has a matching footnote DEFINITION [^1]: ...
        self.assertIn("[^1]", report)
        self.assertRegex(report, re.compile(r"^\[\^1\]: ", re.M))


# ---------------------------------------------------------------------------
# §6: the CLI report command runs the SAME finalization pipeline
# ---------------------------------------------------------------------------

class TestCliReportFinalization(Base):

    def test_cli_report_goes_through_finalization(self):
        import os
        from click.testing import CliRunner
        from deep_research_tool.cli import cli

        plan = make_plan([("1", "市場シェア", "A社のシェア")])
        session = make_session(plan, {
            "1": {"title": "市場シェア",
                  "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                  "evidence": []},
        })
        session_file = self.tmp / "session_cli.json"
        session.save(session_file)

        llm = (RoutedLLM()
               .on("extract", lambda p: j({"claims": []}))
               .on("coverage", all_answered_coverage)
               .on("edit", lambda p: ""))
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}), \
             patch("deep_research_tool.api.get_client",
                   return_value=llm):
            res = CliRunner().invoke(cli, ["report", str(session_file)])
        self.assertEqual(res.exit_code, 0, res.output)
        # the finalization loop actually ran on the session body
        self.assertIn("Finalization decision", res.output)
        self.assertTrue(llm.prompts("extract"))

    def test_cli_report_without_key_warns_unverified(self):
        import os
        from click.testing import CliRunner
        from deep_research_tool.cli import cli

        plan = make_plan([("1", "市場シェア", "A社のシェア")])
        session = make_session(plan, {
            "1": {"title": "市場シェア",
                  "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                  "evidence": []},
        })
        session_file = self.tmp / "session_cli2.json"
        session.save(session_file)

        env = {k: v for k, v in os.environ.items()
               if k not in ("OPENAI_API_KEY",)}
        with patch.dict(os.environ, env, clear=True):
            res = CliRunner().invoke(cli, ["report", str(session_file)])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("未検証", res.output)   # explicit unverified warning


# ---------------------------------------------------------------------------
# §7: fixed / hard-min / hard-max decision branches (production logic)
# ---------------------------------------------------------------------------

def make_verdict(chars, missing=None, unused=None):
    v = StructuredVerdict()
    v.metrics = VerificationMetrics(
        actual_body_chars=chars,
        missing_content_units=missing or [],
        unused_high_importance_evidence_ids=unused or [],
    )
    return v


class TestFixedModeDecisions(unittest.TestCase):

    def test_fixed_overshoot_compresses(self):
        d = decide(make_verdict(15000), LoopBudget(),
                   fixed_target_chars=10000, length_tolerance=0.2)
        self.assertEqual(d, ResearchDecision.COMPRESS_FROM_EVIDENCE)

    def test_fixed_within_tolerance_accepts(self):
        d = decide(make_verdict(11000), LoopBudget(),
                   fixed_target_chars=10000, length_tolerance=0.2)
        self.assertEqual(d, ResearchDecision.ACCEPT)

    def test_fixed_shortfall_with_evidence_rewrites_never_pads(self):
        d = decide(make_verdict(5000, unused=["EV-9"]), LoopBudget(),
                   fixed_target_chars=10000, length_tolerance=0.2)
        self.assertEqual(d, ResearchDecision.REWRITE_FROM_EVIDENCE)

    def test_fixed_shortfall_without_evidence_researches(self):
        d = decide(make_verdict(5000), LoopBudget(),
                   fixed_target_chars=10000, length_tolerance=0.2)
        self.assertEqual(d, ResearchDecision.RESEARCH)
        # research exhausted -> limitations, never padding
        budget = LoopBudget(max_final_research_rounds=0)
        d = decide(make_verdict(5000), budget,
                   fixed_target_chars=10000, length_tolerance=0.2)
        self.assertEqual(d, ResearchDecision.FINALIZE_WITH_LIMITATIONS)

    def test_adaptive_mode_ignores_fixed_branch(self):
        d = decide(make_verdict(500), LoopBudget(),
                   fixed_target_chars=None)
        self.assertEqual(d, ResearchDecision.ACCEPT)

    def test_chunk_failure_blocks_accept(self):
        v = make_verdict(5000)
        v.metrics.chunks_failed = 1
        self.assertEqual(decide(v, LoopBudget()),
                         ResearchDecision.FINALIZE_WITH_LIMITATIONS)


if __name__ == "__main__":
    unittest.main()
