"""
Tests for the finalization pipeline (spec section 9, tests 1-19).

All tests are network-free and LLM-free: the controller takes injected
callables, and the claim verifier gets a scripted fake LLM.
"""

import json
import unittest
from unittest.mock import MagicMock

from deep_research_tool.report.citations import CitationManager
from deep_research_tool.report.finalization import (
    ISSUE_INSUFFICIENT_EXPLANATION,
    ISSUE_REDUNDANCY,
    ISSUE_UNSUPPORTED,
    FinalizationController,
    LoopBudget,
    ResearchDecision,
    StructuredVerdict,
    VerificationIssue,
    VerificationMetrics,
    count_body_chars,
    decide,
)
from deep_research_tool.report.length_planner import LengthPlanner, dedupe_units
from deep_research_tool.verification.claim_verifier import ClaimVerifier


def make_verdict(support=1.0, unsupported_critical=0, issues=None,
                 body_chars=5000, citations_valid=True,
                 missing_units=None, redundant=None, unused=None,
                 coverage=1.0):
    v = StructuredVerdict()
    v.metrics = VerificationMetrics(
        claim_support_score=support,
        unsupported_critical_claims=unsupported_critical,
        critical_question_coverage=coverage,
        actual_body_chars=body_chars,
        citations_valid=citations_valid,
        missing_content_units=missing_units or [],
        redundant_passages=redundant or [],
        unused_high_importance_evidence_ids=unused or [],
    )
    v.issues = issues or []
    return v


def unsupported_issue(section="1", severity="critical", query="q1"):
    return VerificationIssue(
        section_id=section, claim_id="C-1", type=ISSUE_UNSUPPORTED,
        severity=severity, claim="主張X", reason="根拠なし",
        needed_evidence="主張X", search_queries=[query])


class TestPipelineOrdering(unittest.TestCase):
    """Tests 1-2: verification covers the final body; nothing edits after."""

    def test_1_enhanced_body_is_what_gets_verified(self):
        """The text passed to verify_fn is the (already enhanced) final
        candidate, and edits made in the loop are re-verified."""
        seen_bodies = []

        def verify_fn(chapters):
            seen_bodies.append(dict(chapters))
            # first pass: demand a rewrite; afterwards accept
            if len(seen_bodies) == 1:
                return make_verdict(issues=[VerificationIssue(
                    section_id="1", type=ISSUE_INSUFFICIENT_EXPLANATION,
                    severity="important", reason="浅い")],
                    missing_units=["比較"])
            return make_verdict()

        ctrl = FinalizationController(
            verify_fn=verify_fn,
            rewrite_fn=lambda sid, issues: "増補済みの本文 [SOURCE 1] を含む説明。",
            validate_citations_fn=lambda s, t: True,
        )
        out = ctrl.run({"1": "初稿本文 [SOURCE 1]"})
        # the rewritten body was re-verified (2nd verify saw the new text)
        self.assertEqual(len(seen_bodies), 2)
        self.assertIn("増補済み", seen_bodies[1]["1"])
        # and the final chapters ARE the verified ones
        self.assertEqual(out["chapters"]["1"], seen_bodies[1]["1"])

    def test_2_no_body_llm_after_final_verification(self):
        """After the verify that led to ACCEPT, no editing callable runs."""
        calls = {"verify": 0, "edit": 0}

        def verify_fn(chapters):
            calls["verify"] += 1
            return make_verdict()

        def editor(sid, issues):
            calls["edit"] += 1
            return "edited"

        ctrl = FinalizationController(
            verify_fn=verify_fn, rewrite_fn=editor, compress_fn=editor,
            hedge_fn=editor)
        out = ctrl.run({"1": "本文"})
        self.assertEqual(out["decision"], "accept")
        self.assertEqual(calls["edit"], 0)
        self.assertEqual(calls["verify"], 1)   # nothing after acceptance


class TestLoopControl(unittest.TestCase):
    """Tests 3-5: bounded research, stagnation, limitations."""

    def test_3_one_research_round_resolves_to_accept(self):
        state = {"round": 0}

        def verify_fn(chapters):
            state["round"] += 1
            if state["round"] == 1:
                return make_verdict(support=0.5,
                                    unsupported_critical=1,
                                    issues=[unsupported_issue()])
            return make_verdict()   # resolved after research

        research = MagicMock(return_value={"new_sources": 2,
                                           "changed_sections": ["1"]})
        ctrl = FinalizationController(
            verify_fn=verify_fn, research_fn=research,
            rewrite_fn=lambda sid, issues: "新エビデンス反映済み本文")
        out = ctrl.run({"1": "本文"})
        self.assertEqual(out["decision"], "accept")
        research.assert_called_once()

    def test_4_always_failing_stops_at_max_rounds(self):
        verify = MagicMock(side_effect=lambda ch: make_verdict(
            support=0.4, unsupported_critical=1,
            issues=[unsupported_issue(query=f"q{verify.call_count}")]))
        research = MagicMock(return_value={"new_sources": 1,
                                           "changed_sections": ["1"]})
        budget = LoopBudget(max_final_research_rounds=2)
        ctrl = FinalizationController(
            verify_fn=verify, research_fn=research,
            rewrite_fn=lambda sid, issues: "変更後",
            budget=budget)
        out = ctrl.run({"1": "本文"})
        self.assertEqual(out["decision"], "finalize_with_limitations")
        self.assertLessEqual(research.call_count, 2)      # bounded
        self.assertTrue(out["limitations"])               # 限界を明記

    def test_5_no_improvement_finalizes_with_limitations(self):
        # research yields nothing new; score never improves
        verify = MagicMock(side_effect=lambda ch: make_verdict(
            support=0.5, unsupported_critical=1,
            issues=[unsupported_issue(query="同じクエリ")]))
        research = MagicMock(return_value={"new_sources": 0,
                                           "changed_sections": []})
        budget = LoopBudget(max_final_research_rounds=5,
                            max_no_improvement_rounds=1)
        ctrl = FinalizationController(
            verify_fn=verify, research_fn=research, budget=budget)
        out = ctrl.run({"1": "本文"})
        self.assertEqual(out["decision"], "finalize_with_limitations")
        # stagnation stopped it long before the 5-round budget
        self.assertLessEqual(research.call_count, 2)
        # unresolved issues are written into the body deterministically
        self.assertIn("調査上の限界", out["chapters"]["1"])


class TestDecisionStates(unittest.TestCase):
    """Tests 6-11: RESEARCH vs REWRITE vs COMPRESS vs ACCEPT separation."""

    def test_6_shallow_with_sufficient_evidence_rewrites_without_search(self):
        state = {"n": 0}

        def verify_fn(chapters):
            state["n"] += 1
            if state["n"] == 1:
                return make_verdict(
                    missing_units=["主要2社の比較"], unused=["EV-24"],
                    issues=[VerificationIssue(
                        section_id="1", type=ISSUE_INSUFFICIENT_EXPLANATION,
                        severity="important",
                        reason="Evidenceはあるが比較の説明がない")])
            return make_verdict()

        research = MagicMock()
        rewrite = MagicMock(return_value="比較を追加した本文 [SOURCE 1] です。")
        ctrl = FinalizationController(
            verify_fn=verify_fn, research_fn=research, rewrite_fn=rewrite)
        out = ctrl.run({"1": "浅い本文 [SOURCE 1]"})
        self.assertEqual(out["decision"], "accept")
        research.assert_not_called()          # 検索しない
        rewrite.assert_called_once()          # 増補する

    def test_7_short_but_complete_is_accepted(self):
        # 800 chars against a 2400-min recommendation, but nothing missing
        verdict = make_verdict(body_chars=800)
        verdict.metrics.recommended_min_chars = 2400
        decision = decide(verdict, LoopBudget())
        self.assertEqual(decision, ResearchDecision.ACCEPT)

    def test_8_short_and_evidence_poor_researches(self):
        verdict = make_verdict(
            support=0.6, body_chars=800,
            issues=[unsupported_issue()])
        decision = decide(verdict, LoopBudget())
        self.assertEqual(decision, ResearchDecision.RESEARCH)

    def test_9_redundancy_compresses(self):
        verdict = make_verdict(redundant=["同じ説明が3回"],
                               issues=[VerificationIssue(
                                   section_id="2", type=ISSUE_REDUNDANCY,
                                   severity="minor", reason="重複")])
        decision = decide(verdict, LoopBudget())
        self.assertEqual(decision, ResearchDecision.COMPRESS_FROM_EVIDENCE)

    def test_10_useful_overshoot_under_hard_max_is_kept(self):
        # over preferred (10k > 8k) but useful and under the hard max
        verdict = make_verdict(body_chars=10000)
        verdict.metrics.preferred_body_chars = 8000
        decision = decide(verdict, LoopBudget(), hard_max_body_chars=20000)
        self.assertEqual(decision, ResearchDecision.ACCEPT)

    def test_11_over_hard_max_compresses(self):
        verdict = make_verdict(body_chars=25000)
        decision = decide(verdict, LoopBudget(), hard_max_body_chars=20000)
        self.assertEqual(decision, ResearchDecision.COMPRESS_FROM_EVIDENCE)


class TestLengthPlanning(unittest.TestCase):
    """Tests 12-14: adaptive length from information units."""

    def test_12_null_length_settings_impose_no_quota(self):
        planner = LengthPlanner(preferred_body_chars=None,
                                hard_min_body_chars=None,
                                hard_max_body_chars=None)
        self.assertIsNone(planner.fixed_quota())
        rng = planner.document_range()
        self.assertIsNone(rng["preferred_body_chars"])
        self.assertIsNone(rng["hard_max_body_chars"])
        # and the decision logic never rejects on length alone
        verdict = make_verdict(body_chars=500)
        self.assertEqual(decide(verdict, LoopBudget()),
                         ResearchDecision.ACCEPT)

    def test_13_information_rich_sections_get_more_chars(self):
        planner = LengthPlanner()
        rich = [{"key_points": [f"独自の論点{i}" for i in range(8)],
                 "content": "売上は120億円、シェア30%、成長率8%"}]
        poor = [{"key_points": ["論点A"], "content": "少ない"}]
        units = {
            "1": planner.extract_units("1", rich),
            "2": planner.extract_units("2", poor),
        }
        planner.initial_allocation([{"section": "1"}, {"section": "2"}])
        plans = planner.recalc_after_research(units)
        self.assertGreater(plans["1"].recommended_chars,
                           plans["2"].recommended_chars)
        self.assertGreater(plans["1"].units, plans["2"].units)

    def test_14_duplicate_sources_count_once(self):
        planner = LengthPlanner()
        same = "国内市場規模は2,000億円に達したと報告されている"
        dup = [
            {"key_points": [same], "content": same},
            {"key_points": [same + "。"], "content": same},   # repost
            {"key_points": [same], "content": same},
        ]
        units = planner.extract_units("1", dup)
        self.assertEqual(len(units.key_points), 1)     # one unit, not three
        self.assertEqual(units.unique_sources, 1)      # one source, not three
        self.assertEqual(len(dedupe_units([same, same + "。"])), 1)


class ScriptedLLM:
    """Fake LLM returning queued JSON responses."""

    def __init__(self):
        self.queue = []
        self.prompts = []

    def push(self, obj):
        self.queue.append(json.dumps(obj, ensure_ascii=False))

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        r = MagicMock()
        r.content = self.queue.pop(0) if self.queue else "{}"
        return r


def fake_evidence(i, text, title=None):
    ev = MagicMock()
    ev.id = f"EV-{i}"
    ev.url = f"https://src{i}.example.com/page"
    ev.title = title or f"ソース{i}"
    ev.extracted_text = text
    ev.content_excerpt = text[:200]
    return ev


class TestFullTextVerification(unittest.TestCase):
    """Tests 15-17: no head truncation, deep evidence, separated counts."""

    def test_15_claims_beyond_6000_chars_are_extracted(self):
        llm = ScriptedLLM()
        # 9000-char section -> chunked into 3 x 4000-char pieces
        text = ("前段の説明。" * 1200) + "2030年の市場規模は9,999億円に達する。"
        self.assertGreater(len(text), 6000)
        # one extraction response per chunk
        n_chunks = (len(text) + 3999) // 4000
        for i in range(n_chunks):
            if i == n_chunks - 1:
                llm.push({"claims": [{"claim": "2030年の市場規模は9,999億円",
                                      "importance": "critical"}]})
            else:
                llm.push({"claims": []})
        verifier = ClaimVerifier(llm_client=llm, language="ja")
        claims = verifier.extract_claims("1", text)
        self.assertEqual(len(claims), 1)
        self.assertIn("9,999億円", claims[0].text)   # tail claim found

    def test_16_evidence_beyond_first_20_is_used(self):
        llm = ScriptedLLM()
        verifier = ClaimVerifier(llm_client=llm, language="ja")
        from deep_research_tool.verification.claim_verifier import Claim
        claim = Claim(claim_id="C-1", section_id="1",
                      text="リチウム価格は2023年に80%下落した")
        # 25 evidence items; ONLY #23 is relevant
        evidence = [fake_evidence(i, f"無関係な話題{i}です。天気や料理の記事。")
                    for i in range(1, 23)]
        evidence.append(fake_evidence(
            23, "リチウム価格は2023年に80%下落したと市場データが示す。"))
        evidence.append(fake_evidence(24, "別の無関係な記事"))
        evidence.append(fake_evidence(25, "さらに無関係"))
        selected = verifier.select_evidence(claim, evidence)
        self.assertTrue(any(ev.id == "EV-23" for ev in selected),
                        [ev.id for ev in selected])

    def test_17_unsupported_and_contradicted_counted_separately(self):
        llm = ScriptedLLM()
        # claim extraction for one section: 2 claims
        llm.push({"claims": [
            {"claim": "A社のシェアは50%", "importance": "important"},
            {"claim": "B社は撤退した", "importance": "important"},
        ]})
        # per-claim judgements
        llm.push({"status": "unsupported", "reason": "根拠なし",
                  "supporting_source_ids": []})
        llm.push({"status": "contradicted", "reason": "撤退していない",
                  "supporting_source_ids": ["EV-1"]})
        verifier = ClaimVerifier(llm_client=llm, language="ja")
        evidence = [fake_evidence(1, "A社とB社は事業を継続している。シェアは30%。")]
        verdict = verifier.verify_report({"1": "A社のシェアは50%。B社は撤退した。"},
                                         evidence)
        self.assertEqual(verdict.metrics.unsupported_count, 1)
        self.assertEqual(verdict.metrics.contradicted_count, 1)
        types = {i.type for i in verdict.issues}
        self.assertIn(ISSUE_UNSUPPORTED, types)
        self.assertIn("contradicted_claim", types)


class TestCitations(unittest.TestCase):
    """Test 18: citations stay valid across regeneration/compression."""

    def test_18_citation_ids_valid_after_edits(self):
        mgr = CitationManager()
        mgr.register_section("1", ["https://a", "https://b"])
        mgr.register_section("2", ["https://c"])

        # valid rewrite keeps [SOURCE N] within the registry
        self.assertTrue(mgr.validate("1", "本文の主張です [SOURCE 1]。追加の説明 [SOURCE 2]。"))
        # LLM inventing [SOURCE 5] fails machine validation
        self.assertFalse(mgr.validate("1", "捏造引用の本文 [SOURCE 5] です。"))
        # compression leaving a bare citation (claim deleted) fails
        self.assertFalse(mgr.validate("1", "[SOURCE 1]"))

        # newly researched evidence APPENDS: old numbers stay stable
        n = mgr.append_evidence("1", "https://new")
        self.assertEqual(n, 3)
        self.assertEqual(mgr.mapping("1")[1], "https://a")   # unchanged
        # other sections are untouched
        self.assertEqual(mgr.mapping("2"), {1: "https://c"})

        # display numbering happens only at render, globally consistent
        rendered, order = mgr.render_numbering({
            "1": "α [SOURCE 1] β [SOURCE 3]",
            "2": "γ [SOURCE 1]",
        })
        self.assertEqual(rendered["1"], "α [1] β [2]")
        self.assertEqual(rendered["2"], "γ [3]")
        self.assertEqual(order, ["https://a", "https://new", "https://c"])

        # controller rejects an edit whose citations fail validation
        verify = MagicMock(side_effect=[
            make_verdict(issues=[VerificationIssue(
                section_id="1", type=ISSUE_INSUFFICIENT_EXPLANATION,
                severity="important")], missing_units=["x"]),
            make_verdict(), make_verdict()])
        ctrl = FinalizationController(
            verify_fn=verify,
            rewrite_fn=lambda sid, issues: "偽の引用 [SOURCE 9] を含む本文",
            validate_citations_fn=mgr.validate)
        out = ctrl.run({"1": "元の本文 [SOURCE 1] です。"})
        self.assertEqual(out["chapters"]["1"], "元の本文 [SOURCE 1] です。")


class TestQueryDedup(unittest.TestCase):
    """Test 19: the loop never repeats the same query or URL."""

    def test_19_repeated_queries_are_dropped(self):
        budget = LoopBudget()
        first = budget.novel_queries(["EV市場 シェア", "EV市場 シェア",
                                      "電池 価格動向"])
        self.assertEqual(first, ["EV市場 シェア", "電池 価格動向"])
        # same and near-identical queries are rejected on later rounds
        second = budget.novel_queries(["EV市場 シェア", "ev市場　シェア",
                                       "EV市場 シェア 2024"])
        self.assertEqual(second, [])
        third = budget.novel_queries(["全固体電池 量産時期"])
        self.assertEqual(third, ["全固体電池 量産時期"])


class TestMainWiring(unittest.TestCase):
    """Integration: _run_finalization_loop wires verdicts back into the
    generation result and produces the verification HTML."""

    def test_finalization_loop_wiring(self):
        import tempfile
        from pathlib import Path
        from deep_research_tool.config import create_config
        from deep_research_tool.main import DeepResearchTool

        tmp = Path(tempfile.mkdtemp())
        config = create_config(
            provider="openai", openai_api_key="sk-test",
            output_dir=str(tmp), plan_review=False,
        )

        tool = DeepResearchTool.__new__(DeepResearchTool)
        tool.config = config
        tool.llm_client = ScriptedLLM()
        tool.stage_llm_clients = {}
        tool.search_client = MagicMock()

        # scripted LLM: claim extraction (1 claim) + claim judgement, twice
        # is not needed — single verify pass accepts
        llm = tool.llm_client
        llm.push({"claims": [{"claim": "市場規模は1000億円",
                              "importance": "important"}]})
        llm.push({"status": "supported", "reason": "一致",
                  "supporting_source_ids": ["EV-1"]})
        # coverage is FAIL-CLOSED: without an explicit LLM verdict the
        # question counts as unanswered, so the wiring test answers it
        llm.push({"coverage": [{"question": 0, "answered": True,
                                "section_id": "1", "missing": "",
                                "search_queries": []}]})

        # fake session / result / locker
        chapter = MagicMock()
        chapter.content = "## 1. 市場\n\n市場規模は1000億円です [SOURCE 1]。"
        result = MagicMock()
        result.chapters = {"1": chapter}

        session = MagicMock()
        session.session_id = "t1"
        session.section_contents = {"1": {
            "content": chapter.content,
            "sources": ["https://src1.example.com/page"],
            "extracted_content": [{
                "title": "ソース1", "url": "https://src1.example.com/page",
                "content": "市場規模は1000億円と報告", "raw_content": "市場規模は1000億円と報告",
                "key_points": ["市場規模1000億円"],
            }],
        }}
        item = MagicMock(); item.section = "1"; item.title = "市場"
        item.description = "市場規模"
        session.research_plan.table_of_contents.get_flat_sections.return_value = [item]
        session.research_plan.table_of_contents.items = [item]

        locker = MagicMock()
        ev = fake_evidence(1, "市場規模は1000億円と報告", title="ソース1")
        ev.url = "https://src1.example.com/page"
        locker.get_all_evidence.return_value = [ev]
        # resolve canonical ids like the real locker (a bare MagicMock
        # returns junk ids, silently breaking cited-first verification)
        locker.get_by_url = lambda url: ev if url == ev.url else None
        locker.get_evidence = lambda eid: ev if eid == "EV-1" else None

        verdict, html_path = tool._run_finalization_loop(
            result=result, session=session, evidence_locker=locker,
            query="市場調査")

        # final body verified, frozen back into the result, HTML written
        self.assertEqual(verdict.metrics.unsupported_count, 0)
        self.assertTrue(verdict.metrics.citations_valid)
        # render_numbering is CONNECTED to the real output: the frozen
        # chapter carries the display number, no editing tags remain
        self.assertIn("[1]", result.chapters["1"].content)
        self.assertNotIn("[SOURCE", result.chapters["1"].content)
        # references follow first-use order (locker reordered)
        locker.reorder.assert_called_once()
        self.assertTrue(html_path and Path(html_path).exists())
        html = Path(html_path).read_text(encoding="utf-8")
        self.assertIn("accept", html)


class TestBodyCharCount(unittest.TestCase):
    def test_markdown_toc_and_references_excluded(self):
        text = """# タイトル

## 目次

- 1. 序論

## 1. 本論

これが本文です [SOURCE 1]。**強調**も[リンク](https://x)も数えます。

| 列 | 値 |
|---|---|
| a | 100億円 |

## 参考文献

1. https://example.com とても長い参考文献リスト
"""
        n = count_body_chars(text)
        body_only = "これが本文です。強調もリンクも数えます。a100億円列値"
        # heading/TOC/references/markup/citation tags are excluded
        self.assertLess(abs(n - len(body_only.replace(" ", ""))), 8)
        self.assertNotIn("example", str(n))


if __name__ == "__main__":
    unittest.main()
