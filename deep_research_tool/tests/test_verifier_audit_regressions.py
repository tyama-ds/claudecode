"""Offline regressions for the September 2026 accuracy audit."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from deep_research_tool.evidence.locker import Evidence, EvidenceLocker
from deep_research_tool.report.citations import CitationManager
from deep_research_tool.report.finalization import (
    LoopBudget, ResearchDecision, decide,
)
from deep_research_tool.report.finalization_runner import FinalizationRunner
from deep_research_tool.verification.claim_verifier import ClaimVerifier, Claim
from deep_research_tool.verification.runtime import VerificationCache


class ScriptedLLM:
    model = "offline-audit"

    def __init__(self, route):
        self.route = route
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        return SimpleNamespace(content=json.dumps(self.route(prompt),
                                                  ensure_ascii=False))


class TestExtractionIntegrity(unittest.TestCase):
    def test_partial_bad_chunk_is_retried_failed_and_never_cached(self):
        malformed = [{}, [], {"claims": "wrong"},
                     {"claims": [{"claim": 42}]}, {"claims": []},
                     {"claims": [{"claim": "valid"}, {"claim": None}]}]
        for bad in malformed:
            with self.subTest(bad=bad):
                def route(prompt):
                    if "セクションbad" in prompt:
                        return bad
                    if "検証が必要な事実主張" in prompt:
                        return {"claims": [{"claim": "調査の参加人数は100人である",
                                            "source_numbers": [1]}]}
                    return {"status": "supported", "reason": "matches",
                            "supporting_source_ids": ["E1"]}
                llm = ScriptedLLM(route)
                verifier = ClaimVerifier(llm, cache=VerificationCache(True))
                evidence = Evidence(id="E1", extracted_text="調査の参加人数は100人である")
                mgr = CitationManager(lambda eid: eid == "E1")
                for sid in ("good", "bad"):
                    mgr.register_section(sid, ["E1"])
                body = {"good": "調査の参加人数は100人である [SOURCE 1]。",
                        "bad": "未確認の重大な結論として治療成功率は100%である [SOURCE 1]。"}
                first = verifier.verify_report(body, [evidence], mgr)
                self.assertEqual(first.metrics.chunks_failed, 1)
                self.assertNotEqual(decide(first, LoopBudget()), ResearchDecision.ACCEPT)
                count = len(llm.calls)
                verifier.verify_report(body, [evidence], mgr)
                # The healthy section and its judgement are cached;
                # the failed section alone receives all bounded retries.
                self.assertEqual(len(llm.calls) - count, verifier.EXTRACT_RETRIES + 1)

    def test_short_nonfactual_heading_can_have_no_claims(self):
        verifier = ClaimVerifier(ScriptedLLM(lambda _: {"claims": []}))
        self.assertEqual(verifier.extract_claims("intro", "## Introduction"), [])
        self.assertEqual(verifier.chunks_failed, 0)

    def test_validated_quote_positions_disambiguate_repeated_claims(self):
        claim = "The sample contains 100 participants"
        quote = claim + " [SOURCE 2]."
        text = claim + " [SOURCE 1].\n\n" + quote
        llm = ScriptedLLM(lambda _: {"claims": [
            {"claim": claim, "source_quote": quote, "source_numbers": [2]}]})
        extracted = ClaimVerifier(llm).extract_claims("1", text)[0]
        self.assertEqual(ClaimVerifier.locate_cited_numbers(
            claim, text, extracted.source_start, extracted.source_end), (True, [2]))


class TestCriticalUncertainty(unittest.TestCase):
    def test_critical_unknown_cannot_pass_on_average_or_close_ledger(self):
        claims = [{"claim": f"地域{chr(65+i)}の登録件数は{100+i}件である",
                   "importance": "important", "source_numbers": [1]}
                  for i in range(20)]
        claims.append({"claim": "本報告の核心となる効果は未確認である",
                       "importance": "critical", "source_numbers": [1]})

        def route(prompt):
            if "検証が必要な事実主張" in prompt:
                return {"claims": claims}
            if "【主張】（critical）" in prompt:
                return {"status": "uncertain", "reason": "insufficient evidence",
                        "supporting_source_ids": []}
            return {"status": "supported", "reason": "matches",
                    "supporting_source_ids": ["E1"]}

        body = "".join(c["claim"] + " [SOURCE 1]。" for c in claims)
        evidence = Evidence(id="E1", extracted_text=body)
        mgr = CitationManager(lambda _: True)
        mgr.register_section("1", ["E1"])
        verdict = ClaimVerifier(ScriptedLLM(route)).verify_report({"1": body}, [evidence], mgr)
        self.assertGreater(verdict.metrics.claim_support_score, 0.9)
        self.assertEqual(verdict.metrics.uncertain_critical_claims, 1)
        self.assertEqual(decide(verdict, LoopBudget()), ResearchDecision.RESEARCH)
        exhausted = LoopBudget(max_final_research_rounds=0)
        self.assertEqual(decide(verdict, exhausted), ResearchDecision.FINALIZE_WITH_LIMITATIONS)
        with tempfile.TemporaryDirectory() as tmp:
            runner = FinalizationRunner(evidence_locker=EvidenceLocker(output_dir=Path(tmp)),
                                        session_contents={}, output_dir=tmp)
            runner._build_requirements([], {"1": body})
            runner._update_ledger_from_verdict(verdict)
            self.assertEqual(runner.ledger.get("REQ-S1").status, "open")
            self.assertEqual(runner.ledger.coverage(), 0.0)


class TestEvidenceAndCitations(unittest.TestCase):
    def test_document_tail_remains_retrievable_without_expanding_top_k(self):
        tail = "Distinctive terminal result is 987654321 participants."
        evidence = Evidence(id="long", extracted_text="ordinary filler " * 4000 + tail)
        verifier = ClaimVerifier(None)
        index = verifier.build_chunk_index([evidence])
        selected = verifier.select_evidence(Claim("C1", "1", tail), [evidence], index)
        self.assertTrue(any(tail in chunk.text for chunk in selected))
        self.assertLessEqual(len(selected), verifier.evidence_per_claim)
        self.assertEqual(index[0].fetched_at, evidence.accessed_at)

    def test_decimal_abbreviation_and_post_sentence_citations(self):
        for claim, body in [
            ("The revenue grew by 3.5 percent in 2026.",
             "The revenue grew by 3.5 percent in 2026 [SOURCE 1]."),
            ("The study included 100 participants.",
             "The study included 100 participants. [SOURCE 1]"),
            ("Dr. Smith measured 3.5 percent growth.",
             "Dr. Smith measured 3.5 percent growth. [SOURCE 1]"),
            ("調査対象者は100人である", "調査対象者は100人である。[SOURCE 1]。"),
        ]:
            with self.subTest(body=body):
                self.assertEqual(ClaimVerifier.locate_cited_numbers(claim, body), (True, [1]))
        self.assertEqual(ClaimVerifier.locate_cited_numbers(
            "A社の収益は10億円である", "A社の収益は10億円である。B社の収益は20億円である [SOURCE 2]。"), (True, []))

    def test_future_or_invalid_publication_dates_do_not_establish_freshness(self):
        now = datetime.now()
        for date in (str(now.year + 1), (now + timedelta(days=1)).date().isoformat(),
                     f"{now.year}-02-31"):
            with self.subTest(date=date):
                self.assertFalse(ClaimVerifier._evidence_is_fresh(
                    Evidence(published_date=date), now.year, 3))
        self.assertTrue(ClaimVerifier._evidence_is_fresh(
            Evidence(published_date=now.date().isoformat()), now.year, 3))


if __name__ == "__main__":
    unittest.main()
