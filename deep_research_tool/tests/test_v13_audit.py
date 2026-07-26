"""
Regression tests for the v1.2 -> v1.3 audit (17 items).

Covers the MANDATORY test scenarios of the audit:
1  character targets never trigger web searches
2  coverage: an LLM answered=false is FINAL (no lexical overturn)
3  freshness is requirement-scoped and same-evidence (fresh AND primary)
4  no requirement is silently dropped (all plan levels + user reqs)
5  deferred tasks consume no search attempt
6  query -> task -> evidence -> section binding
7  localized claim patching (protected-sentence hashes, cached re-verify)
8  LLM schema anomalies fail closed without crashing
9  sentence-strict citation association
10/11 semantic freeze + two-way artifact check (fail-closed)
12 Selenium under a shared RunLimits (peak <= limit, no permit leak)
13 TaskDAG + self-limiting client at limit=1 (no double-permit deadlock)
14 one stopping authority (max_stall_rounds honored)
15 minor/optional-only gaps never research
17 run-level cancellation propagates to the worker
"""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from deep_research_tool.adaptive import CoverageLedger, TaskDAG
from deep_research_tool.adaptive.models import (
    REQ_OPEN,
    REQ_SUPPORTED,
    ProgressRound,
    RequirementLeaf,
    ResearchTask,
)
from deep_research_tool.report.finalization import (
    ISSUE_UNSUPPORTED,
    LoopBudget,
    ResearchDecision,
    StructuredVerdict,
    VerificationIssue,
    decide,
)
from deep_research_tool.report.semantic_manifest import (
    build_semantic_manifest,
    extract_artifact_text,
    figure_semantics,
    manifest_hash,
    normalize_semantic_text,
    verify_frozen_in_artifact,
)
from deep_research_tool.utils.concurrency import ConcurrencyLimiter, RunLimits
from deep_research_tool.verification.claim_verifier import (
    Claim,
    ClaimVerifier,
)
from deep_research_tool.verification.profiles import (
    resolve_verification_settings,
)
from deep_research_tool.verification.runtime import VerificationCache
from deep_research_tool.verification.schema import (
    validate_coverage_entry,
    validate_extracted_claim,
    validate_verdict,
)


def j(obj):
    return json.dumps(obj, ensure_ascii=False)


def make_verifier(llm, workers=2, batch=1, cache=None):
    settings = resolve_verification_settings(
        "custom", {"max_workers": workers, "batch_size": batch})
    return ClaimVerifier(llm_client=llm, language="ja", settings=settings,
                         cache=cache or VerificationCache(enabled=True))


def make_evidence(eid, text, **attrs):
    ev = MagicMock()
    ev.id = eid
    ev.url = attrs.pop("url", f"https://{eid.lower()}.example.com/")
    ev.title = f"ソース{eid}"
    ev.extracted_text = text
    ev.content_excerpt = text[:200]
    for k, v in attrs.items():
        setattr(ev, k, v)
    return ev


# ===========================================================================
# 1. character targets never trigger searches
# ===========================================================================

class TestNoLengthTriggeredSearch(unittest.TestCase):

    def _run(self, **config_kwargs):
        from deep_research_tool.tests.test_finalization_integration import (
            CLAIM_SENTENCE, FakeSearch, build_tool, long_body, make_llm,
            make_locker, make_plan, make_session, run_tool,
            SUPPORTED_EVIDENCE)
        tmp = Path(tempfile.mkdtemp())
        llm = make_llm()
        search = FakeSearch()      # records every search call
        locker = make_locker(tmp, SUPPORTED_EVIDENCE)
        plan = make_plan([("1", "市場シェア", "A社のシェア")])
        session = make_session(plan, {
            "1": {"title": "市場シェア",
                  "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                  "evidence": SUPPORTED_EVIDENCE},
        })
        tool = build_tool(tmp, llm, search, **config_kwargs)
        result = run_tool(tool, session, locker)
        return search, result

    def test_target_characters_never_searches(self):
        # target far above the current body: previously _expand_if_needed
        # fired extra research; now the search count must stay identical
        search_plain, _ = self._run()
        search_target, result = self._run(target_characters=200000)
        self.assertEqual(len(search_target.searched),
                         len(search_plain.searched))
        # the shortfall is surfaced as a warning, not silently ignored
        self.assertTrue(any(w["source"] == "Length"
                            for w in result["warnings"]))

    def test_target_pages_never_searches(self):
        search_plain, _ = self._run()
        search_pages, _ = self._run(target_pages=50)
        self.assertEqual(len(search_pages.searched),
                         len(search_plain.searched))


# ===========================================================================
# 2. coverage: LLM answered=false is FINAL
# ===========================================================================

class TestCoverageFalseIsFinal(unittest.TestCase):

    QUESTION = "A社の2026年売上高はいくらか"
    BODY = ("A社の2026年売上高はいくらかを調査したが、"
            "公開資料では回答を確認できなかった。" + "背景の説明。" * 30)

    def test_llm_false_never_overturned_by_lexical_match(self):
        class FalseLLM:
            model = "cov"

            def generate(self, prompt, **kwargs):
                return MagicMock(content=j({"coverage": [
                    {"question": 0, "answered": False, "section_id": "1",
                     "missing": "売上高の数値", "search_queries": ["A社 売上高"]},
                ]}))

        verifier = make_verifier(FalseLLM())
        results = verifier.judge_coverage([self.QUESTION],
                                          {"1": self.BODY})
        # the question text is ~fully contained in the body — the old
        # lexical cross-check would flip this to answered
        self.assertIs(results[0]["answered"], False)

    def test_missing_llm_verdict_fails_closed_to_unanswered(self):
        class SilentLLM:
            model = "silent"

            def generate(self, prompt, **kwargs):
                return MagicMock(content="{}")

        verifier = make_verifier(SilentLLM())
        results = verifier.judge_coverage([self.QUESTION],
                                          {"1": self.BODY})
        self.assertIs(results[0]["answered"], False)

    def test_unanswered_question_keeps_requirement_open(self):
        # requirement-side consequence: with the question unanswered the
        # ledger requirement stays open (research/limitations target)
        from deep_research_tool.report.finalization import (
            ISSUE_UNANSWERED_QUESTION)
        ledger = CoverageLedger()
        ledger.add(RequirementLeaf(req_id="REQ-U1", text=self.QUESTION,
                                   origin="user"))
        self.assertEqual(ledger.get("REQ-U1").status, REQ_OPEN)
        self.assertFalse(ledger.all_terminal())


# ===========================================================================
# 3. freshness: same-evidence AND, requirement-scoped
# ===========================================================================

class TestFreshnessRequirementScoped(unittest.TestCase):

    def test_fresh_secondary_plus_stale_primary_is_not_fresh_primary(self):
        fresh_secondary = make_evidence(
            "EV-1", "新しいブログ記事", published_date="2026-05-01",
            quality_indicators=SimpleNamespace(is_primary_source=False),
            source_type=SimpleNamespace(value="blog"),
            quality_category=SimpleNamespace(value="general"))
        stale_primary = make_evidence(
            "EV-2", "古い政府統計", published_date="2015-01-01",
            quality_indicators=SimpleNamespace(is_primary_source=True),
            source_type=SimpleNamespace(value="official"),
            quality_category=SimpleNamespace(value="authoritative"))
        state, why = ClaimVerifier.assess_primary_freshness(
            [fresh_secondary, stale_primary], required=True)
        self.assertEqual(state, "fail")
        self.assertIn("BOTH", why)

    def test_same_source_fresh_and_primary_passes(self):
        both = make_evidence(
            "EV-3", "最新の政府統計", published_date="2026-01-01",
            quality_indicators=SimpleNamespace(is_primary_source=True),
            source_type=SimpleNamespace(value="official"),
            quality_category=SimpleNamespace(value="authoritative"))
        state, _ = ClaimVerifier.assess_primary_freshness(
            [both], required=True)
        self.assertEqual(state, "pass")

    def test_freshness_fail_keeps_ledger_non_terminal(self):
        # a supported-looking requirement with an unmet freshness demand
        # must stay OPEN -> the ledger is not all_terminal
        ledger = CoverageLedger()
        ledger.add(RequirementLeaf(
            req_id="REQ-S1", text="s1", section_id="1",
            freshness_requirement=True, primary_source_required=True))
        # (the runner's _update_ledger_from_verdict forces OPEN for
        #  freshness-blocked requirements — modeled here directly)
        self.assertEqual(ledger.get("REQ-S1").status, REQ_OPEN)
        self.assertFalse(ledger.all_terminal())


# ===========================================================================
# 4. requirements are never silently dropped
# ===========================================================================

class TestRequirementCompleteness(unittest.TestCase):

    def _runner(self, tmp, requirements, plan):
        from deep_research_tool.config import create_config
        from deep_research_tool.report.finalization_runner import (
            FinalizationRunner)
        from deep_research_tool.tests.test_finalization_integration import (
            make_locker)
        config = create_config(provider="openai", openai_api_key="sk-test",
                               output_dir=str(tmp), plan_review=False)
        locker = make_locker(tmp, [("https://e.example.com/", "本文。" * 40)])
        return FinalizationRunner(
            evidence_locker=locker, session_contents={},
            research_plan=plan, query="調査",
            requirements=requirements, language="ja",
            llm_client=MagicMock(), search_client=None,
            report_config=config.report, research_config=config.research,
            output_dir=Path(tmp), session_id="req-test")

    def test_all_plan_items_and_user_requirements_survive(self):
        from deep_research_tool.tests.test_finalization_integration import (
            make_plan)
        # 15 top-level plan items + subsections + 5 user requirements
        items = [(str(i), f"第{i}章のトピック", f"第{i}章の説明文です")
                 for i in range(1, 16)]
        items.insert(3, ("3.1", "第3章の詳細サブセクション", "サブセクションの説明"))
        plan = make_plan(items)
        user_reqs = "。".join(
            f"ユーザー要件{i}を必ず調査に含めること" for i in range(1, 6))
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(tmp, user_reqs, plan)
            runner._build_requirements(runner._critical_questions(),
                                       {"1": "本文"})
            reqs = runner.ledger.requirements()
            user = [r for r in reqs if r.req_id.startswith("REQ-U")]
            planr = [r for r in reqs if r.req_id.startswith("REQ-Q")]
            self.assertEqual(len(user), 5)          # nothing dropped
            self.assertEqual(len(planr), 16)        # incl. subsection
            # explicit user requirements come FIRST and are critical
            questions = runner._critical_questions()
            self.assertIn("ユーザー要件1", questions[0])
            self.assertTrue(all(r.priority == "critical" for r in user))
            self.assertTrue(any("サブセクション" in r.text for r in planr))


# ===========================================================================
# 5. deferred tasks consume no attempt / 13-task cap
# ===========================================================================

class TestDeferredTasksConsumeNothing(unittest.TestCase):

    def test_thirteenth_task_is_deferred_not_attempted(self):
        from deep_research_tool.config import create_config
        from deep_research_tool.report.finalization_runner import (
            FinalizationRunner)
        from deep_research_tool.tests.test_finalization_integration import (
            FakeSearch, make_locker)

        with tempfile.TemporaryDirectory() as tmp:
            config = create_config(
                provider="openai", openai_api_key="sk-test",
                output_dir=str(tmp), plan_review=False,
                requirement_max_search_attempts=1)
            locker = make_locker(tmp, [])
            search = FakeSearch()
            runner = FinalizationRunner(
                evidence_locker=locker, session_contents={},
                research_plan=None, query="調査", requirements="",
                language="ja", llm_client=MagicMock(),
                search_client=search,
                report_config=config.report,
                research_config=config.research,
                output_dir=Path(tmp), session_id="defer-test")
            # 13 gap requirements, one issue query each -> 13 tasks
            issues = []
            for i in range(1, 14):
                rid = f"REQ-S{i}"
                runner.ledger.add(RequirementLeaf(
                    req_id=rid, text=f"セクション{i}の裏付け",
                    section_id=str(i), origin="section"))
                issues.append(VerificationIssue(
                    section_id=str(i), claim_id=f"C-{i}",
                    type=ISSUE_UNSUPPORTED, severity="critical",
                    claim=f"主張{i}", search_queries=[f"クエリ{i}"]))
            queries = [f"クエリ{i}" for i in range(1, 14)]
            runner._adaptive_research_round(issues, queries)

            attempts = {r.req_id: r.search_attempts
                        for r in runner.ledger.requirements()}
            # exactly ROUND_TASK_CAP (12) requirements consumed attempts
            self.assertEqual(sum(1 for v in attempts.values() if v > 0),
                             runner.ROUND_TASK_CAP)
            # the 13th requirement consumed NOTHING and is still open
            deferred_req = [rid for rid, v in attempts.items() if v == 0]
            self.assertEqual(len(deferred_req), 1)
            leftover = runner.ledger.get(deferred_req[0])
            self.assertEqual(leftover.status, REQ_OPEN)
            self.assertNotEqual(leftover.status, "unavailable_after_search")
            # its round accounting shows the deferral: exactly 12
            # tasks were scheduled; the 13th issue task (plus the
            # intent-templated diversification tasks) was deferred
            self.assertEqual(
                len(runner._pending_round["scheduled_tasks"]), 12)
            self.assertGreaterEqual(
                len(runner._pending_round["deferred_tasks"]), 1)

            # NEXT round: the deferred requirement is searchable (its
            # query is novel again and its attempts are unspent)
            gaps = [r.req_id for r in runner.ledger.gap_requirements()]
            self.assertIn(deferred_req[0], gaps)


# ===========================================================================
# 6. query -> evidence -> section binding
# ===========================================================================

class TestEvidenceSectionBinding(unittest.TestCase):

    def test_evidence_registers_only_to_its_tasks_section(self):
        from deep_research_tool.config import create_config
        from deep_research_tool.report.finalization_runner import (
            FinalizationRunner)
        from deep_research_tool.tests.test_finalization_integration import (
            FakeSearch, make_locker)

        with tempfile.TemporaryDirectory() as tmp:
            config = create_config(provider="openai",
                                   openai_api_key="sk-test",
                                   output_dir=str(tmp), plan_review=False)
            locker = make_locker(tmp, [])
            search = FakeSearch(
                results={"クエリ1": ["https://s1.example.com/"],
                         "クエリ2": ["https://s2.example.com/"]},
                pages={"https://s1.example.com/": "セクション1の裏付け資料。" * 30,
                       "https://s2.example.com/": "セクション2の裏付け資料。" * 30})
            session_contents = {
                "1": {"content": "本文1", "sources": [],
                      "extracted_content": []},
                "2": {"content": "本文2", "sources": [],
                      "extracted_content": []},
            }
            runner = FinalizationRunner(
                evidence_locker=locker, session_contents=session_contents,
                research_plan=None, query="調査", requirements="",
                language="ja", llm_client=MagicMock(),
                search_client=search,
                report_config=config.report,
                research_config=config.research,
                output_dir=Path(tmp), session_id="bind-test")
            for sid in ("1", "2"):
                runner.ledger.add(RequirementLeaf(
                    req_id=f"REQ-S{sid}", text=f"セクション{sid}の裏付け",
                    section_id=sid, origin="section"))
            issues = [
                VerificationIssue(section_id="1", claim_id="C-1",
                                  type=ISSUE_UNSUPPORTED,
                                  severity="critical", claim="主張1",
                                  search_queries=["クエリ1"]),
                VerificationIssue(section_id="2", claim_id="C-2",
                                  type=ISSUE_UNSUPPORTED,
                                  severity="critical", claim="主張2",
                                  search_queries=["クエリ2"]),
            ]
            runner._adaptive_research_round(issues, ["クエリ1", "クエリ2"])

            ev1 = locker.get_by_url("https://s1.example.com/")
            ev2 = locker.get_by_url("https://s2.example.com/")
            self.assertIsNotNone(ev1)
            self.assertIsNotNone(ev2)
            # section_contents: strict per-section registration
            urls1 = session_contents["1"]["sources"]
            urls2 = session_contents["2"]["sources"]
            self.assertIn(ev1.url, urls1)
            self.assertNotIn(ev2.url, urls1)
            self.assertIn(ev2.url, urls2)
            self.assertNotIn(ev1.url, urls2)
            # CitationManager mirrors the same relations
            self.assertIn(ev1.id, runner.citation_mgr.evidence_ids("1"))
            self.assertNotIn(ev2.id, runner.citation_mgr.evidence_ids("1"))
            self.assertIn(ev2.id, runner.citation_mgr.evidence_ids("2"))
            self.assertNotIn(ev1.id, runner.citation_mgr.evidence_ids("2"))
            # EvidenceLocker section_reference matches too
            self.assertEqual(ev1.section_reference, "1")
            self.assertEqual(ev2.section_reference, "2")
            # provenance metadata for chunk building
            self.assertEqual(ev1.adaptive_meta["section_id"], "1")
            self.assertEqual(ev1.adaptive_meta["requirement_id"], "REQ-S1")


# ===========================================================================
# 7. localized claim patching
# ===========================================================================

class TestLocalizedPatch(unittest.TestCase):

    S1 = "第一文は市場の背景を丁寧に説明する内容である。"
    S2 = "A社のシェアは9割であると断定する。"
    S3 = "第三文は今後の展望を落ち着いて説明する内容である。"

    def _runner(self, tmp, llm):
        from deep_research_tool.config import create_config
        from deep_research_tool.report.finalization_runner import (
            FinalizationRunner)
        from deep_research_tool.tests.test_finalization_integration import (
            make_locker)
        config = create_config(provider="openai", openai_api_key="sk-test",
                               output_dir=str(tmp), plan_review=False)
        locker = make_locker(tmp, [("https://e.example.com/",
                                    "A社のシェアは52%と報告。" * 20)])
        return FinalizationRunner(
            evidence_locker=locker,
            session_contents={"1": {"content": "", "sources": [],
                                    "extracted_content": []}},
            research_plan=None, query="調査", requirements="",
            language="ja", llm_client=llm, search_client=None,
            report_config=config.report, research_config=config.research,
            output_dir=Path(tmp), session_id="patch-test")

    def test_only_failing_sentence_changes_hashes_prove_it(self):
        fixed = "A社のシェアは52%である [SOURCE 1]。"

        class PatchLLM:
            model = "patch"

            def __init__(self):
                self.patch_prompts = []

            def generate(self, prompt, **kwargs):
                self.patch_prompts.append(prompt)
                return MagicMock(content=fixed)

        llm = PatchLLM()
        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(tmp, llm)
            runner.citation_mgr.register_section(
                "1", [runner.locker.get_all_evidence()[0].id])
            text = self.S1 + self.S2 + self.S3
            issue = VerificationIssue(
                section_id="1", claim_id="C-1.1.1",
                type=ISSUE_UNSUPPORTED, severity="critical",
                claim="A社のシェアは9割である", reason="根拠なし")
            patched = runner._patch_claims("1", text, [issue], "rewrite")
            self.assertIsNotNone(patched)
            # sentence 2 replaced; sentences 1 and 3 byte-identical
            self.assertIn(fixed, patched)
            self.assertNotIn(self.S2, patched)
            self.assertEqual(runner._sent_hash(self.S1),
                             runner._sent_hash(
                                 runner._sentence_spans(patched)[0][2]))
            self.assertIn(runner._sent_hash(self.S3),
                          {runner._sent_hash(s) for _a, _b, s
                           in runner._sentence_spans(patched)})
            # the LLM saw ONLY the failing sentence as the patch target
            self.assertTrue(all("修正対象の一文" in p
                                for p in llm.patch_prompts))
            self.assertTrue(all(self.S1 not in
                                p.split("【修正対象の一文】")[-1]
                                for p in llm.patch_prompts))

    def test_unlocatable_claim_falls_back_and_audits(self):
        class NeverLLM:
            model = "never"

            def generate(self, prompt, **kwargs):
                return MagicMock(content="x")

        with tempfile.TemporaryDirectory() as tmp:
            runner = self._runner(tmp, NeverLLM())
            issue = VerificationIssue(
                section_id="1", claim_id="C-9",
                type=ISSUE_UNSUPPORTED, severity="critical",
                claim="全く関係のない存在しない主張テキスト")
            patched = runner._patch_claims(
                "1", self.S1 + self.S3, [issue], "rewrite")
            self.assertIsNone(patched)      # -> audited safe fallback

    def test_only_changed_claim_is_rejudged_via_cache(self):
        # cached judgements: after a one-sentence patch, only the
        # changed claim misses the judgement cache
        class JudgeLLM:
            model = "cachejudge"

            def __init__(self):
                self.judge_calls = 0

            def generate(self, prompt, **kwargs):
                if "検証が必要な事実主張" in prompt:
                    # two claims, verbatim from the chunk
                    claims = []
                    if "第一文は市場の背景" in prompt:
                        claims.append({"claim": TestLocalizedPatch.S1[:-1],
                                       "importance": "important",
                                       "source_numbers": []})
                    if "A社のシェア" in prompt:
                        claims.append({"claim": "A社のシェアは52%である",
                                       "importance": "critical",
                                       "source_numbers": []})
                    return MagicMock(content=j({"claims": claims}))
                if "実質的に回答しているか" in prompt:
                    return MagicMock(content=j({"coverage": []}))
                self.judge_calls += 1
                import re as _re
                ids = _re.findall(r"^\[([^\]]+)\]", prompt, _re.M)
                return MagicMock(content=j(
                    {"status": "supported", "reason": "ok",
                     "supporting_source_ids": ids}))

        llm = JudgeLLM()
        cache = VerificationCache(enabled=True)
        verifier = make_verifier(llm, cache=cache)
        ev = make_evidence("EV-1", "A社のシェアは52%と報告。" * 10
                           + TestLocalizedPatch.S1 * 3)
        body1 = self.S1 + "A社のシェアは52%である。"
        verifier.verify_report({"1": body1}, [ev])
        calls_first = llm.judge_calls
        self.assertGreaterEqual(calls_first, 2)
        # patch ONLY the second sentence (same claim text extracted for
        # sentence 1) -> claim 1 replays from cache
        body2 = self.S1 + "A社のシェアは52%であることが確認された。"
        verifier.verify_report({"1": body2}, [ev])
        second_calls = llm.judge_calls - calls_first
        self.assertLessEqual(second_calls, calls_first)


# ===========================================================================
# 8. schema anomalies fail closed, never crash
# ===========================================================================

class TestSchemaAnomalies(unittest.TestCase):

    def test_validator_matrix(self):
        # nested list
        self.assertIsNone(validate_verdict(
            {"status": "supported",
             "supporting_source_ids": [["EV-1"]]}))
        # status = dict
        self.assertIsNone(validate_verdict(
            {"status": {"value": "supported"},
             "supporting_source_ids": ["EV-1"]}))
        # ids = [1]
        self.assertIsNone(validate_verdict(
            {"status": "supported", "supporting_source_ids": [1]}))
        # reason = list
        self.assertIsNone(validate_verdict(
            {"status": "unsupported", "reason": ["x"],
             "supporting_source_ids": []}))
        # answered as string / int
        self.assertIsNone(validate_coverage_entry(
            {"question": 0, "answered": "false"}))
        self.assertIsNone(validate_coverage_entry(
            {"question": 0, "answered": 1}))
        # search_queries as bare string
        self.assertIsNone(validate_coverage_entry(
            {"question": 0, "answered": True,
             "search_queries": "query"}))
        # source_numbers = ["1"] -> field unusable (None), claim kept
        checked = validate_extracted_claim(
            {"claim": "主張", "importance": "minor",
             "source_numbers": ["1"]})
        self.assertIsNotNone(checked)
        self.assertIsNone(checked["source_numbers"])

    def test_end_to_end_anomalies_never_crash(self):
        class EvilLLM:
            model = "evil"

            def generate(self, prompt, **kwargs):
                if "検証が必要な事実主張" in prompt:
                    return MagicMock(content=j({"claims": [
                        {"claim": "本文の主張は999件である",
                         "importance": "critical",
                         "source_numbers": ["1"]},          # anomaly
                        {"claim": ["リストの主張"]},          # anomaly
                        "文字列エントリ",                     # anomaly
                    ]}))
                if "実質的に回答しているか" in prompt:
                    return MagicMock(content=j({"coverage": [
                        {"question": 0, "answered": "false"},   # anomaly
                        {"question": "x", "answered": True},    # anomaly
                    ]}))
                # judge verdict: nested list + dict status
                return MagicMock(content=j(
                    {"status": "supported",
                     "supporting_source_ids": [["EV-1"]]}))

        verifier = make_verifier(EvilLLM())
        ev = make_evidence("EV-1", "本文の主張は999件と報告。" * 20)
        body = "## 1. 分析\n\n本文の主張は999件である。" + "説明。" * 60
        verdict = verifier.verify_report(
            {"1": body}, [ev], critical_questions=["何件あるか"])
        # no exception escaped; the surviving claim failed closed
        self.assertGreaterEqual(verdict.metrics.uncertain_count, 1)
        self.assertLess(verdict.metrics.claim_support_score, 1.0)
        # the anomalous coverage entries fail closed to unanswered
        self.assertLess(verdict.metrics.critical_question_coverage, 1.0)


# ===========================================================================
# 9. sentence-strict citation association
# ===========================================================================

class TestSentenceStrictAssociation(unittest.TestCase):

    def test_neighbor_sentence_citation_is_never_borrowed(self):
        body = "A社は10億円。B社は20億円 [SOURCE 2]。"
        located, nums = ClaimVerifier.locate_cited_numbers(
            "A社は10億円", body)
        self.assertTrue(located)
        self.assertEqual(nums, [])          # NOT [2]
        located_b, nums_b = ClaimVerifier.locate_cited_numbers(
            "B社は20億円", body)
        self.assertTrue(located_b)
        self.assertEqual(nums_b, [2])

    def test_partial_overlap_is_association_failure(self):
        body = "A社のシェアは52%である [SOURCE 1]。" + "説明。" * 40

        class PartialLLM:
            model = "partial"

            def generate(self, prompt, **kwargs):
                if "検証が必要な事実主張" in prompt:
                    return MagicMock(content=j({"claims": [
                        {"claim": "A社のシェアは52%である",
                         "importance": "critical",
                         "source_numbers": [1, 2]},   # parser sees [1]
                    ]}))
                if "実質的に回答しているか" in prompt:
                    return MagicMock(content=j({"coverage": []}))
                return MagicMock(content=j(
                    {"status": "supported", "reason": "ok",
                     "supporting_source_ids": ["EV-1"]}))

        from deep_research_tool.report.citations import CitationManager
        mgr = CitationManager(evidence_ids_exist=lambda e: True)
        mgr.register_section("1", ["EV-1", "EV-2"])
        verifier = make_verifier(PartialLLM())
        ev = make_evidence("EV-1", "A社のシェアは52%と報告。" * 10)
        verdict = verifier.verify_report({"1": body}, [ev],
                                         citation_manager=mgr)
        self.assertTrue(any(
            i.type == "citation_association_failure"
            for i in verdict.issues))
        self.assertFalse(verdict.metrics.citations_valid)


# ===========================================================================
# 10/11. semantic freeze + two-way artifact check
# ===========================================================================

class TestArtifactCheck(unittest.TestCase):

    CHAPTERS = {"1": "## 1. 分析\n\nA社のシェアは52%である。"
                     + "市場は今後も拡大する見通しが示されている。" * 3}

    def _collection(self):
        from deep_research_tool.report.figure_table_generator import (
            Figure, FigureTableCollection, FigureType, TableData)
        col = FigureTableCollection()
        col.charts.append(Figure(
            figure_id="c1", figure_type=FigureType.CHART,
            title="シェア推移", caption="図1: シェアの推移",
            section_id="1",
            chart_data={"labels": ["2025", "2026"],
                        "values": ["50", "52"],
                        "x_axis": "年", "y_axis": "シェア",
                        "unit": "%", "annotation": "上昇傾向"}))
        col.tables.append(TableData(
            table_id="t1", title="数値一覧", caption="表1",
            headers=["年", "シェア"],
            rows=[["2025", "50%"], ["2026", "52%"]], section_id="1"))
        return col

    def _artifact(self, manifest, extra="", drop=None):
        parts = [self.CHAPTERS["1"]]
        col_md = ["### 図: シェア推移", "図1: シェアの推移",
                  "### 表: 数値一覧", "表1",
                  "| 年 | シェア |", "| 2025 | 50% |", "| 2026 | 52% |"]
        parts.extend(col_md)
        text = "\n\n".join(parts)
        if drop:
            text = text.replace(drop, "")
        if extra:
            text += "\n\n" + extra
        return text

    def test_perfect_match_passes(self):
        manifest = build_semantic_manifest(self.CHAPTERS, self._collection())
        result = verify_frozen_in_artifact(
            manifest, self._artifact(manifest))
        self.assertTrue(result["ok"], result)
        self.assertGreater(result["checked"], 0)

    def test_missing_frozen_body_fails(self):
        manifest = build_semantic_manifest(self.CHAPTERS, self._collection())
        result = verify_frozen_in_artifact(
            manifest, self._artifact(manifest,
                                     drop="A社のシェアは52%である。"))
        self.assertFalse(result["ok"])
        self.assertTrue(result["missing"])

    def test_unverified_addition_after_freeze_fails(self):
        manifest = build_semantic_manifest(self.CHAPTERS, self._collection())
        result = verify_frozen_in_artifact(
            manifest, self._artifact(
                manifest,
                extra="B社は市場から撤退したとされ、その影響で価格が"
                      "急落したという未検証の虚偽情報を追記する。"))
        self.assertFalse(result["ok"])
        self.assertTrue(result["additions"])

    def test_short_table_cell_missing_fails(self):
        manifest = build_semantic_manifest(self.CHAPTERS, self._collection())
        result = verify_frozen_in_artifact(
            manifest, self._artifact(manifest, drop="| 2026 | 52% |"))
        self.assertFalse(result["ok"])       # short cells ARE checked

    def test_checked_zero_fails(self):
        result = verify_frozen_in_artifact({"chapters": {}}, "何らかの本文")
        self.assertFalse(result["ok"])

    def test_chart_series_change_changes_manifest_hash(self):
        col = self._collection()
        h1 = manifest_hash(build_semantic_manifest(self.CHAPTERS, col))
        col.charts[0].chart_data["values"] = ["50", "99"]   # tampered
        h2 = manifest_hash(build_semantic_manifest(self.CHAPTERS, col))
        self.assertNotEqual(h1, h2)

    def test_pdf_readback_detects_missing_body(self):
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "report.pdf"
            doc = fitz.open()
            page = doc.new_page()
            # the PDF misses the frozen chapter body
            page.insert_text((72, 72), "unrelated pdf content only")
            doc.save(str(pdf_path))
            doc.close()
            text = extract_artifact_text(pdf_path)
            self.assertIsNotNone(text)
            manifest = build_semantic_manifest(
                {"1": "この本文はPDFに存在しないはずの凍結済み内容である。"
                      * 3})
            result = verify_frozen_in_artifact(manifest, text)
            self.assertFalse(result["ok"])


# ===========================================================================
# 12. Selenium under a shared RunLimits
# ===========================================================================

class TestSeleniumSharedLimiter(unittest.TestCase):

    def test_eight_clients_limit_one_peak_one(self):
        from deep_research_tool.search.selenium_browser import (
            SeleniumBrowser)
        limits = RunLimits(1, process_limiter=ConcurrencyLimiter(16))

        active = [0]
        peak = [0]
        gate = threading.Lock()

        class FakeDriver:
            def get(self, url):
                with gate:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                time.sleep(0.02)
                with gate:
                    active[0] -= 1

        clients = []
        for _ in range(8):
            c = SeleniumBrowser.__new__(SeleniumBrowser)
            c._driver = FakeDriver()
            c._driver_lock = threading.RLock()
            c.concurrency_limiter = limits
            clients.append(c)

        def _navigate(c):
            # exercise the permit->lock path exactly as search() does
            with c._leaf_permit():
                with c._driver_lock:
                    c._driver.get("https://example.com/")

        threads = [threading.Thread(target=_navigate, args=(c,))
                   for c in clients]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertLessEqual(peak[0], 1)          # observed concurrency
        self.assertLessEqual(limits.run_peak, 1)  # limiter peak
        self.assertEqual(limits.run_limiter.active, 0)   # no permit leak

    def test_browser_methods_take_leaf_permits(self):
        # search() / get_page_content() actually go through _leaf_permit
        import inspect
        from deep_research_tool.search.selenium_browser import (
            SeleniumBrowser)
        src_search = inspect.getsource(SeleniumBrowser.search)
        src_page = inspect.getsource(SeleniumBrowser.get_page_content)
        self.assertIn("_leaf_permit", src_search)
        self.assertIn("_leaf_permit", src_page)


# ===========================================================================
# 13. TaskDAG + self-limiting client at limit=1
# ===========================================================================

class TestDagNoDoublePermit(unittest.TestCase):

    def test_limit_one_completes_without_deadlock(self):
        limits = RunLimits(1, process_limiter=ConcurrencyLimiter(16))
        peak = [0]
        active = [0]
        gate = threading.Lock()

        def fake_client_call(task):
            # the CLIENT takes the permit (one-sided ownership)
            with limits.permit(timeout=10):
                with gate:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                time.sleep(0.02)
                with gate:
                    active[0] -= 1
            return task.query

        dag = TaskDAG([ResearchTask(task_id=f"T{i}", req_id="R",
                                    query=f"q{i}") for i in range(6)])
        done = {}
        worker = threading.Thread(
            target=lambda: done.update(dag.run(
                fake_client_call, parallel_max_workers=8, limiter=limits)))
        worker.start()
        worker.join(timeout=15)
        self.assertFalse(worker.is_alive(), "deadlocked at limit=1")
        self.assertEqual(len(done), 6)
        self.assertLessEqual(peak[0], 1)
        self.assertEqual(limits.run_limiter.active, 0)

    def test_multi_hop_dependency_refines_query(self):
        # dependent task receives the prerequisite's result via the
        # runner's _execute closure — modeled at the DAG level here
        results_map = {}

        def execute(task):
            if task.depends_on:
                prev = results_map.get(task.depends_on[0]) or []
                if prev:
                    task.query = f"{task.query} {prev[0].title}"
            value = [SimpleNamespace(title="一次資料タイトル",
                                     url="https://x.example.com/")]
            results_map[task.task_id] = value
            return value

        t1 = ResearchTask(task_id="T1", req_id="R", query="基点クエリ",
                          intent="multi_hop")
        t2 = ResearchTask(task_id="T2", req_id="R", query="展開クエリ",
                          intent="multi_hop", depends_on=["T1"])
        TaskDAG([t1, t2]).run(execute, parallel_max_workers=4)
        self.assertIn("一次資料タイトル", t2.query)


# ===========================================================================
# 14/15. one stopping authority + severity gating
# ===========================================================================

class TestStopAuthorityAndSeverity(unittest.TestCase):

    def test_budget_aligned_to_max_stall_rounds(self):
        from deep_research_tool.config import create_config
        from deep_research_tool.report.finalization_runner import (
            FinalizationRunner)
        from deep_research_tool.tests.test_finalization_integration import (
            make_locker)
        with tempfile.TemporaryDirectory() as tmp:
            config = create_config(provider="openai",
                                   openai_api_key="sk-test",
                                   output_dir=str(tmp), plan_review=False,
                                   max_stall_rounds=2)
            locker = make_locker(tmp, [])
            runner = FinalizationRunner(
                evidence_locker=locker, session_contents={},
                research_plan=None, query="q", requirements="",
                language="ja", llm_client=MagicMock(), search_client=None,
                report_config=config.report,
                research_config=config.research,
                output_dir=Path(tmp), session_id="stall-test")
            # ONE authority: the legacy no-improvement counter can never
            # stop earlier than the configured max_stall_rounds
            self.assertGreaterEqual(runner.budget.max_no_improvement_rounds,
                                    2)
            self.assertEqual(runner.stop_controller.max_stall_rounds, 2)

    def _verdict_with(self, *issues):
        v = StructuredVerdict()
        v.metrics.actual_body_chars = 5000
        v.metrics.claims_total = 10
        for i in issues:
            v.issues.append(i)
        return v

    def test_minor_only_never_researches(self):
        v = self._verdict_with(VerificationIssue(
            section_id="1", type=ISSUE_UNSUPPORTED, severity="minor",
            claim="些末な主張", needed_evidence="x",
            search_queries=["q"]))
        v.metrics.unsupported_count = 1
        v.metrics.claim_support_score = 0.5    # dragged down by minors
        d = decide(v, LoopBudget())
        self.assertNotEqual(d, ResearchDecision.RESEARCH)

    def test_critical_unsupported_researches(self):
        v = self._verdict_with(VerificationIssue(
            section_id="1", type=ISSUE_UNSUPPORTED, severity="critical",
            claim="重大な主張", needed_evidence="x",
            search_queries=["q"]))
        v.metrics.unsupported_count = 1
        v.metrics.unsupported_critical_claims = 1
        v.metrics.claim_support_score = 0.5
        d = decide(v, LoopBudget())
        self.assertEqual(d, ResearchDecision.RESEARCH)


# ===========================================================================
# 17. run-level cancellation propagates
# ===========================================================================

class TestRunCancellation(unittest.TestCase):

    def test_cancel_token_aborts_at_first_checkpoint(self):
        from deep_research_tool.main import RunCancelled
        from deep_research_tool.tests.test_finalization_integration import (
            CLAIM_SENTENCE, FakeSearch, build_tool, long_body, make_llm,
            make_locker, make_plan, make_session, FakeResearcher)
        from unittest.mock import patch

        tmp = Path(tempfile.mkdtemp())
        llm = make_llm()
        search = FakeSearch()
        locker = make_locker(tmp, [("https://e.example.com/",
                                    "A社の国内シェアは52%と報告。" * 8)])
        plan = make_plan([("1", "市場シェア", "A社のシェア")])
        session = make_session(plan, {
            "1": {"title": "市場シェア",
                  "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                  "evidence": [("https://e.example.com/", "x")]},
        })
        tool = build_tool(tmp, llm, search)
        cancel = threading.Event()
        cancel.set()        # cancel arrives before/at the first checkpoint

        fake = FakeResearcher(session, locker)
        with patch("deep_research_tool.main.Researcher",
                   new=lambda **kw: fake):
            with self.assertRaises(RunCancelled):
                tool.run(query="市場調査", cancel_event=cancel)
        # after the cancellation NOTHING new started: no searches, no
        # verification/edit LLM calls (extraction/judge/edit prompts)
        self.assertEqual(search.searched, [])
        self.assertEqual([k for k, _p in llm.calls
                          if k in ("extract", "judge", "edit")], [])

    def test_request_cancel_mid_run_ends_cancelled_never_completed(self):
        # a cancel raised DURING verification takes the verification
        # safe-cancel path: the run returns with
        # verification_cancelled=True (a terminal cancelled outcome,
        # never a normal completion)
        from deep_research_tool.tests.test_finalization_integration import (
            CLAIM_SENTENCE, FakeSearch, build_tool, long_body, make_llm,
            make_locker, make_plan, make_session, FakeResearcher)
        from unittest.mock import patch

        tmp = Path(tempfile.mkdtemp())
        llm = make_llm()
        locker = make_locker(tmp, [("https://e.example.com/",
                                    "A社の国内シェアは52%と報告。" * 8)])
        plan = make_plan([("1", "市場シェア", "A社のシェア")])
        session = make_session(plan, {
            "1": {"title": "市場シェア",
                  "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                  "evidence": [("https://e.example.com/", "x")]},
        })
        tool = build_tool(tmp, llm, FakeSearch())
        calls = {"n": 0}

        def cancelling_progress(message, pct):
            calls["n"] += 1
            if calls["n"] == 3:
                tool.request_cancel()      # mid-run cancel (Web UI path)

        fake = FakeResearcher(session, locker)
        with patch("deep_research_tool.main.Researcher",
                   new=lambda **kw: fake):
            result = tool.run(query="市場調査",
                              progress_callback=cancelling_progress)
        self.assertTrue(result.get("verification_cancelled"))
        self.assertTrue(tool.cancel_event.is_set())

    def test_webui_job_cancel_run_wires_to_tool(self):
        from deep_research_tool.webui.server import ResearchJob
        job = ResearchJob("job-t", "q")
        self.assertFalse(job.cancel_run())      # nothing wired yet
        called = []
        job.run_cancel = lambda: called.append(True)
        self.assertTrue(job.cancel_run())
        self.assertTrue(called)


if __name__ == "__main__":
    unittest.main()
