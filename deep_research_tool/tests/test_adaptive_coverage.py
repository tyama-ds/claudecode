"""
Tests for adaptive Deep Research (v1.2):

- audit regressions: A citation association (no parser/LLM union,
  fail-closed association failure), B schema fail-closed (no auto-fill,
  strict booleans, batch retry), C content-addressed cache keys
  (same-length evidence flip -> cache MISS);
- Coverage Ledger state machine + StopController stall detection;
- TaskDAG scheduling (worker cap, dependencies, cycles, failures);
- query intent classification + intent queries + internal RRF;
- audit log masking;
- FinalizationRunner adaptive integration (gap-only research,
  requirement transitions, ledger in the outcome, audit JSONL).
"""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from deep_research_tool.adaptive import (
    REQ_BUDGET_EXHAUSTED,
    REQ_CONFLICTED,
    REQ_NOT_APPLICABLE,
    REQ_OPEN,
    REQ_SUPPORTED,
    REQ_UNAVAILABLE,
    CoverageLedger,
    StopController,
    TaskDAG,
    build_intent_queries,
    classify_intent,
    rrf_merge,
)
from deep_research_tool.adaptive.models import (
    ProgressRound,
    RequirementLeaf,
    ResearchTask,
)
from deep_research_tool.report.citations import CitationManager
from deep_research_tool.report.finalization import (
    ISSUE_CITATION_ASSOCIATION_FAILURE,
)
from deep_research_tool.utils.audit_log import AuditLog
from deep_research_tool.verification.claim_verifier import Claim, ClaimVerifier
from deep_research_tool.verification.profiles import (
    resolve_verification_settings,
)
from deep_research_tool.verification.runtime import VerificationCache


def j(obj):
    return json.dumps(obj, ensure_ascii=False)


def make_verifier(llm, workers=2, batch=1, rate=1.0, cache=None):
    settings = resolve_verification_settings(
        "custom", {"max_workers": workers, "batch_size": batch,
                   "minor_claim_sample_rate": rate})
    return ClaimVerifier(llm_client=llm, language="ja", settings=settings,
                         cache=cache or VerificationCache(enabled=True))


def make_evidence(eid, text, url=None):
    ev = MagicMock()
    ev.id = eid
    ev.url = url or f"https://{eid.lower()}.example.com/"
    ev.title = f"ソース{eid}"
    ev.extracted_text = text
    ev.content_excerpt = text[:200]
    return ev


# ===========================================================================
# Coverage Ledger
# ===========================================================================

class TestCoverageLedger(unittest.TestCase):

    def _req(self, rid="R1", **kw):
        defaults = dict(req_id=rid, text=f"要件{rid}")
        defaults.update(kw)
        return RequirementLeaf(**defaults)

    def test_valid_transitions_and_history(self):
        ledger = CoverageLedger()
        ledger.add(self._req())
        ledger.transition("R1", REQ_SUPPORTED)
        ledger.transition("R1", REQ_CONFLICTED)   # new contradiction reopens
        ledger.transition("R1", REQ_SUPPORTED)
        req = ledger.get("R1")
        self.assertEqual(req.status, REQ_SUPPORTED)
        self.assertEqual(req.history,
                         [REQ_OPEN, REQ_SUPPORTED, REQ_CONFLICTED,
                          REQ_SUPPORTED])

    def test_illegal_transition_raises(self):
        ledger = CoverageLedger()
        ledger.add(self._req())
        ledger.transition("R1", REQ_NOT_APPLICABLE)
        with self.assertRaises(ValueError):
            ledger.transition("R1", REQ_SUPPORTED)   # NA is terminal
        # explicit reopen is the ONLY way back
        ledger.reopen("R1", "ユーザー指示")
        self.assertEqual(ledger.get("R1").status, REQ_OPEN)

    def test_budget_exhausted_is_terminal(self):
        ledger = CoverageLedger()
        ledger.add(self._req())
        ledger.transition("R1", REQ_BUDGET_EXHAUSTED)
        with self.assertRaises(ValueError):
            ledger.transition("R1", REQ_OPEN)

    def test_unknown_state_and_requirement(self):
        ledger = CoverageLedger()
        ledger.add(self._req())
        with self.assertRaises(ValueError):
            ledger.transition("R1", "sorta_done")
        with self.assertRaises(KeyError):
            ledger.transition("R9", REQ_SUPPORTED)

    def test_close_exhausted_after_attempts(self):
        ledger = CoverageLedger(max_search_attempts=2)
        ledger.add(self._req("R1"))
        ledger.add(self._req("R2"))
        ledger.record_search_attempt("R1")
        ledger.record_search_attempt("R1")
        moved = ledger.close_exhausted()
        self.assertEqual(moved, ["R1"])
        self.assertEqual(ledger.get("R1").status, REQ_UNAVAILABLE)
        self.assertEqual(ledger.get("R2").status, REQ_OPEN)
        # exhausted requirements no longer generate gap searches
        self.assertEqual([r.req_id for r in ledger.gap_requirements()],
                         ["R2"])

    def test_close_budget_exhausted_leaves_nothing_open(self):
        ledger = CoverageLedger()
        ledger.add(self._req("R1"))
        ledger.add(self._req("R2", status=REQ_SUPPORTED))
        ledger.close_budget_exhausted("終了")
        self.assertEqual(ledger.get("R1").status, REQ_BUDGET_EXHAUSTED)
        self.assertEqual(ledger.get("R2").status, REQ_SUPPORTED)
        self.assertTrue(ledger.all_terminal())

    def test_coverage_and_gap_priority_order(self):
        ledger = CoverageLedger()
        ledger.add(self._req("R1", priority="minor"))
        ledger.add(self._req("R2", priority="critical"))
        ledger.add(self._req("R3", status=REQ_SUPPORTED))
        ledger.add(self._req("R4", status=REQ_NOT_APPLICABLE))
        self.assertAlmostEqual(ledger.coverage(), 0.5)
        gaps = [r.req_id for r in ledger.gap_requirements()]
        self.assertEqual(gaps, ["R2", "R1"])     # critical first


class TestStopController(unittest.TestCase):

    def test_stops_when_all_terminal(self):
        ledger = CoverageLedger()
        ledger.add(RequirementLeaf(req_id="R1", text="t",
                                   status=REQ_SUPPORTED))
        stop, reason = StopController().should_stop(ledger)
        self.assertTrue(stop)
        self.assertEqual(reason, "all_requirements_terminal")

    def test_stops_on_round_budget(self):
        ledger = CoverageLedger()
        ledger.add(RequirementLeaf(req_id="R1", text="t"))
        ledger.record_round(ProgressRound(1, new_unique_evidence=3,
                                          coverage_delta=0.2))
        stop, reason = StopController(max_rounds=1).should_stop(ledger)
        self.assertTrue(stop)
        self.assertEqual(reason, "round_budget_exhausted")

    def test_stall_detection_from_measured_deltas(self):
        ledger = CoverageLedger()
        ledger.add(RequirementLeaf(req_id="R1", text="t"))
        ctl = StopController(max_rounds=10, max_stall_rounds=2)
        # productive round -> keep going
        ledger.record_round(ProgressRound(1, new_unique_evidence=2))
        self.assertFalse(ctl.should_stop(ledger)[0])
        # one stalled round -> still going (threshold is 2)
        ledger.record_round(ProgressRound(2))
        self.assertFalse(ctl.should_stop(ledger)[0])
        # second consecutive stalled round -> stop
        ledger.record_round(ProgressRound(3))
        stop, reason = ctl.should_stop(ledger)
        self.assertTrue(stop)
        self.assertIn("stalled", reason)

    def test_resolved_conflicts_count_as_progress(self):
        p = ProgressRound(1, resolved_conflicts=1)
        self.assertFalse(p.is_stalled())
        p2 = ProgressRound(2, coverage_delta=0.001)
        self.assertTrue(p2.is_stalled())


# ===========================================================================
# TaskDAG scheduler
# ===========================================================================

class TestTaskDAG(unittest.TestCase):

    def test_dependencies_and_results(self):
        order = []
        lock = threading.Lock()
        dag = TaskDAG([
            ResearchTask(task_id="A", req_id="R", query="qa"),
            ResearchTask(task_id="B", req_id="R", query="qb",
                         depends_on=["A"]),
            ResearchTask(task_id="C", req_id="R", query="qc",
                         depends_on=["A", "B"]),
        ])

        def run(task):
            with lock:
                order.append(task.task_id)
            return task.query.upper()

        results = dag.run(run, parallel_max_workers=4)
        self.assertEqual(results, {"A": "QA", "B": "QB", "C": "QC"})
        self.assertLess(order.index("A"), order.index("B"))
        self.assertLess(order.index("B"), order.index("C"))

    def test_worker_cap_respected(self):
        active = [0]
        peak = [0]
        lock = threading.Lock()

        def run(task):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.03)
            with lock:
                active[0] -= 1
            return True

        dag = TaskDAG([ResearchTask(task_id=f"T{i}", req_id="R", query="q")
                       for i in range(8)])
        dag.run(run, parallel_max_workers=2)
        self.assertLessEqual(peak[0], 2)

    def test_cycle_and_unknown_dependency_rejected(self):
        dag = TaskDAG([
            ResearchTask(task_id="A", req_id="R", query="q",
                         depends_on=["B"]),
            ResearchTask(task_id="B", req_id="R", query="q",
                         depends_on=["A"]),
        ])
        with self.assertRaises(ValueError):
            dag.run(lambda t: True)
        dag2 = TaskDAG([ResearchTask(task_id="A", req_id="R", query="q",
                                     depends_on=["missing"])])
        with self.assertRaises(ValueError):
            dag2.run(lambda t: True)

    def test_failed_task_skips_dependents_without_raising(self):
        dag = TaskDAG([
            ResearchTask(task_id="A", req_id="R", query="boom"),
            ResearchTask(task_id="B", req_id="R", query="q",
                         depends_on=["A"]),
            ResearchTask(task_id="C", req_id="R", query="q"),
        ])

        def run(task):
            if task.query == "boom":
                raise RuntimeError("search failed")
            return "ok"

        results = dag.run(run, parallel_max_workers=2)
        self.assertIsNone(results["A"])
        self.assertIsNone(results["B"])
        self.assertEqual(results["C"], "ok")
        tasks = {t.task_id: t for t in dag.tasks()}
        self.assertEqual(tasks["A"].status, "failed")
        self.assertIn("dependency failed", tasks["B"].error)


# ===========================================================================
# Query intent + RRF (internal only)
# ===========================================================================

class TestIntent(unittest.TestCase):

    def test_classification(self):
        cases = {
            "国内市場規模の推移と統計": "quantitative",
            "2026年の最新動向を確認する": "recent_update",
            "政府の白書など一次資料で裏付ける": "primary_source",
            "A社とB社の比較": "comparison",
            "この技術のリスクと課題": "counterevidence",
            "カーボンニュートラルとは何か": "definition",
            "円安が輸出産業へ与える影響": "multi_hop",
            "業界の歴史": "background",
        }
        for text, expected in cases.items():
            self.assertEqual(classify_intent(text), expected, text)

    def test_intent_queries_deterministic(self):
        q1 = build_intent_queries("水素市場の規模 統計", language="ja")
        q2 = build_intent_queries("水素市場の規模 統計", language="ja")
        self.assertEqual(q1, q2)
        self.assertTrue(all("水素市場の規模" in q for q in q1))
        self.assertGreaterEqual(len(q1), 2)

    def test_rrf_merge_fuses_and_dedups(self):
        a = [SimpleNamespace(url="u1"), SimpleNamespace(url="u2"),
             SimpleNamespace(url="u3")]
        b = [SimpleNamespace(url="u3"), SimpleNamespace(url="u2"),
             SimpleNamespace(url="u4")]
        merged = rrf_merge([a, b])
        urls = [r.url for r in merged]
        self.assertEqual(sorted(urls), ["u1", "u2", "u3", "u4"])
        # u2 (ranks 2+2) and u3 (ranks 3+1) beat u1 (rank 1 only)
        self.assertLess(urls.index("u2"), urls.index("u1"))
        self.assertLess(urls.index("u3"), urls.index("u1"))
        self.assertEqual(len(rrf_merge([a, b], limit=2)), 2)


# ===========================================================================
# Audit log
# ===========================================================================

class TestAuditLog(unittest.TestCase):

    def test_masks_secrets_and_truncates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            log = AuditLog(path=path, enabled=True, session_id="s1")
            log.event("start", api_key="sk-super-secret",
                      authorization="Bearer xyz",
                      nested={"token": "abc", "ok": 1},
                      long_text="あ" * 1000, count=3)
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["api_key"], "***")
            self.assertEqual(record["authorization"], "***")
            self.assertEqual(record["nested"]["token"], "***")
            self.assertEqual(record["nested"]["ok"], 1)
            self.assertNotIn("sk-super-secret", lines[0])
            self.assertLess(len(record["long_text"]), 400)
            self.assertEqual(record["count"], 3)
            self.assertEqual(record["event"], "start")

    def test_disabled_is_noop(self):
        log = AuditLog(path=None, enabled=True)
        log.event("x", a=1)          # must not raise
        self.assertEqual(log.count, 0)


# ===========================================================================
# Audit regression A: citation association (no union, fail-closed)
# ===========================================================================

BODY = ("## 1. 分析\n\nA社のシェアは52%である [SOURCE 1]。"
        + "背景説明が続く。" * 40)


class AssocLLM:
    """Extraction reports a claim ABSENT from the body with LLM-only
    source_numbers; judge supports everything it is shown."""

    model = "assoc"

    def __init__(self):
        self.judge_prompts = []

    def generate(self, prompt, **kwargs):
        if "検証が必要な事実主張" in prompt:
            content = j({"claims": [
                {"claim": "A社のシェアは52%である", "importance": "critical",
                 "source_numbers": [1]},
                {"claim": "架空の統計値は999件にのぼる",
                 "importance": "critical", "source_numbers": [1]},
            ]})
        elif "実質的に回答しているか" in prompt:
            content = j({"coverage": []})
        else:
            self.judge_prompts.append(prompt)
            import re as _re
            ids = _re.findall(r"^\[([^\]]+)\]", prompt, _re.M)
            content = j({"status": "supported", "reason": "ok",
                         "supporting_source_ids": ids} if
                        "次の主張" in prompt else
                        {"verdicts": [{"id": _re.search(
                            r"主張 (C-[\w.]+)", prompt).group(1),
                            "status": "supported", "reason": "ok",
                            "supporting_source_ids": ids}]})
        return MagicMock(content=content)


class TestCitationAssociation(unittest.TestCase):

    def _run(self):
        llm = AssocLLM()
        ev = make_evidence("EV-1", "A社のシェアは52%であると報告。" * 10)
        mgr = CitationManager(evidence_ids_exist=lambda e: True)
        mgr.register_section("1", ["EV-1"])
        verifier = make_verifier(llm)
        return verifier.verify_report({"1": BODY}, [ev],
                                      citation_manager=mgr)

    def test_unlocatable_claim_with_llm_numbers_fails_closed(self):
        verdict = self._run()
        assoc = verdict.issues_of(ISSUE_CITATION_ASSOCIATION_FAILURE)
        self.assertEqual(len(assoc), 1)
        self.assertIn("架空の統計値", assoc[0].claim)
        # fail-closed: counted as uncertain, never supported
        self.assertGreaterEqual(verdict.metrics.uncertain_count, 1)
        # and the citation gate is invalidated
        self.assertFalse(verdict.metrics.citations_valid)

    def test_parser_is_sole_authority_no_union(self):
        located, nums = ClaimVerifier.locate_cited_numbers(
            "A社のシェアは52%である", BODY)
        self.assertTrue(located)
        self.assertEqual(nums, [1])
        located2, nums2 = ClaimVerifier.locate_cited_numbers(
            "架空の統計値は999件にのぼる", BODY)
        self.assertFalse(located2)
        self.assertEqual(nums2, [])
        # verify_report kept the LLM numbers as DIAGNOSTICS only: the
        # locatable claim carries the parser's numbers, and the
        # unlocatable one carries NO verified citations (no union)
        verdict = self._run()
        by_claim = {i.claim: i for i in verdict.issues}
        self.assertIn("架空の統計値は999件にのぼる", by_claim)


# ===========================================================================
# Audit regression B: schema fail-closed
# ===========================================================================

class TestSchemaFailClosed(unittest.TestCase):

    def test_supported_without_ids_is_never_autofilled(self):
        class NoIdsLLM:
            model = "noids"

            def __init__(self):
                self.n = 0

            def generate(self, prompt, **kwargs):
                self.n += 1
                return MagicMock(content=j(
                    {"status": "supported", "reason": "ok"}))   # no ids!

        llm = NoIdsLLM()
        verifier = make_verifier(llm)
        ev = make_evidence("EV-1", "売上高は増加したと記載。" * 20)
        claims = [Claim(claim_id="C-1", section_id="1",
                        text="売上高は増加した")]
        verifier.verify_claims(claims, [ev])
        self.assertEqual(claims[0].status, "uncertain")
        self.assertEqual(claims[0].supporting_source_ids, [])

    def test_batch_supported_without_ids_retried_then_fail_closed(self):
        class BatchLLM:
            model = "batch"

            def __init__(self):
                self.batch_calls = 0
                self.single_calls = 0

            def generate(self, prompt, **kwargs):
                if "各主張を、それぞれに提示されたエビデンス" in prompt:
                    self.batch_calls += 1
                    import re as _re
                    cids = _re.findall(r"◆ 主張 (C-[\w.\-]+)", prompt)
                    return MagicMock(content=j({"verdicts": [
                        # first claim: VALID supported (with ids)
                        {"id": cids[0], "status": "supported",
                         "reason": "ok",
                         "supporting_source_ids": ["EV-1"]},
                        # second claim: schema anomaly (supported, no ids)
                        {"id": cids[1], "status": "supported",
                         "reason": "ok", "supporting_source_ids": []},
                    ]}))
                self.single_calls += 1     # individual retry
                return MagicMock(content=j(
                    {"status": "supported", "reason": "ok",
                     "supporting_source_ids": []}))

        llm = BatchLLM()
        verifier = make_verifier(llm, batch=2)
        ev = make_evidence("EV-1", "事実1と事実2が記載されている。" * 20)
        claims = [
            Claim(claim_id="C-1", section_id="1", text="事実1がある"),
            Claim(claim_id="C-2", section_id="1", text="事実2がある"),
        ]
        verifier.verify_claims(claims, [ev])
        self.assertEqual(llm.batch_calls, 1)
        self.assertEqual(llm.single_calls, 1)       # only the anomaly
        self.assertEqual(claims[0].status, "supported")
        self.assertEqual(claims[1].status, "uncertain")   # fail-closed

    def test_coverage_answered_requires_json_true(self):
        class StringBoolLLM:
            model = "strbool"

            def generate(self, prompt, **kwargs):
                return MagicMock(content=j({"coverage": [
                    {"question": 0, "answered": "true",
                     "section_id": "1", "missing": "",
                     "search_queries": ["q"]},
                ]}))

        verifier = make_verifier(StringBoolLLM())
        results = verifier.judge_coverage(
            ["全く関係のない架空の問いに答えているか"],
            {"1": "この本文は別の話題だけを扱う。" * 30})
        self.assertIs(results[0]["answered"], False)


# ===========================================================================
# Audit regression C: content-addressed cache keys
# ===========================================================================

class TestCacheContentAddressing(unittest.TestCase):

    def test_same_length_flip_support_to_contradiction_misses_cache(self):
        class FlipLLM:
            model = "flip"

            def __init__(self):
                self.judge_calls = 0

            def generate(self, prompt, **kwargs):
                self.judge_calls += 1
                import re as _re
                ids = _re.findall(r"^\[([^\]]+)\]", prompt, _re.M)
                evidence_part = prompt.split("【エビデンス】")[-1]
                if "増加した" in evidence_part:
                    return MagicMock(content=j(
                        {"status": "supported", "reason": "一致",
                         "supporting_source_ids": ids}))
                return MagicMock(content=j(
                    {"status": "contradicted", "reason": "矛盾",
                     "supporting_source_ids": ids}))

        llm = FlipLLM()
        cache = VerificationCache(enabled=True)
        verifier = make_verifier(llm, cache=cache)

        text_a = "第3四半期の売上高は前年比で増加したことが確認された。" * 8
        text_b = "第3四半期の売上高は前年比で減少したことが確認された。" * 8
        self.assertEqual(len(text_a), len(text_b))   # SAME length

        ev = make_evidence("EV-1", text_a)
        claims = [Claim(claim_id="C-1", section_id="1",
                        text="売上高は前年から変動した")]
        verifier.verify_claims(claims, [ev])
        self.assertEqual(claims[0].status, "supported")
        calls_first = llm.judge_calls

        # identical content replays from cache (no new judge call)
        claims2 = [Claim(claim_id="C-1", section_id="1",
                         text="売上高は前年から変動した")]
        verifier.verify_claims(claims2, [ev])
        self.assertEqual(llm.judge_calls, calls_first)
        self.assertEqual(claims2[0].status, "supported")

        # same-length CONTENT change -> cache MISS -> re-judged
        ev_flipped = make_evidence("EV-1", text_b)
        claims3 = [Claim(claim_id="C-1", section_id="1",
                         text="売上高は前年から変動した")]
        verifier.verify_claims(claims3, [ev_flipped])
        self.assertGreater(llm.judge_calls, calls_first)
        self.assertEqual(claims3[0].status, "contradicted")


# ===========================================================================
# FinalizationRunner adaptive integration
# ===========================================================================

from deep_research_tool.tests.test_finalization_integration import (  # noqa: E402
    FakeSearch,
    RoutedLLM,
    judge_ids,
    make_locker,
)


class TestRunnerAdaptiveIntegration(unittest.TestCase):

    CLAIM = "A社のシェアは52%である"

    def _build(self, tmp, llm, search):
        from deep_research_tool.config import create_config
        from deep_research_tool.report.finalization_runner import (
            FinalizationRunner)
        config = create_config(provider="openai", openai_api_key="sk-test",
                               output_dir=str(tmp), plan_review=False)
        locker = make_locker(
            tmp, [("https://old.example.com/1",
                   "全く別の話題について書かれた古い記事。" * 20)])
        session_contents = {"1": {
            "title": "市場シェア",
            "content": "",
            "sources": ["https://old.example.com/1"],
            "extracted_content": [{
                "title": "旧ソース", "url": "https://old.example.com/1",
                "content": "別の話題", "raw_content": "別の話題",
                "key_points": [], "relevance_score": 0.4}],
        }}
        runner = FinalizationRunner(
            evidence_locker=locker,
            session_contents=session_contents,
            research_plan=None,
            query="A社の市場シェア調査",
            requirements="A社の最新の市場シェアを統計で裏付けること",
            language="ja",
            llm_client=llm,
            search_client=search,
            report_config=config.report,
            research_config=config.research,
            output_dir=Path(tmp),
            session_id="adapt1",
        )
        return runner, locker

    def _wire_llm(self, llm, support_after_research):
        state = {"researched": False}

        def extract(prompt):
            return j({"claims": [{"claim": self.CLAIM,
                                  "importance": "critical",
                                  "source_numbers": [1]}]})

        def judge(prompt):
            ids = judge_ids(prompt)
            if "新ソース" in prompt or any("EV-2" in i for i in ids):
                return j({"status": "supported", "reason": "新資料が支持",
                          "supporting_source_ids": ids})
            return j({"status": "unsupported", "reason": "無関係",
                      "supporting_source_ids": []})

        def batch(prompt):
            import re as _re
            cids = _re.findall(r"◆ 主張 (C-[\w.\-]+)", prompt)
            ids = judge_ids(prompt)
            supported = any("EV-2" in i for i in ids)
            return j({"verdicts": [
                {"id": cid,
                 "status": "supported" if supported else "unsupported",
                 "reason": "新資料が支持" if supported else "無関係",
                 "supporting_source_ids": ids if supported else []}
                for cid in cids]})

        def edit(prompt):
            state["researched"] = True
            return (f"## 1. 市場シェア\n\n{self.CLAIM} [SOURCE 2]。"
                    + "補足の説明が続く。" * 30)

        llm.on("extract", extract)
        llm.on("judge", judge)
        llm.on("edit", edit)
        llm.on("coverage", lambda p: j({"coverage": [
            {"question": 0, "answered": True, "section_id": "1",
             "missing": "", "search_queries": []}]}))
        llm.MARKERS = dict(llm.MARKERS)
        llm.MARKERS["batch"] = ("各主張を、それぞれに提示されたエビデンス",)
        llm.on("batch", batch)
        return llm

    def test_gap_only_research_and_ledger_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = self._wire_llm(RoutedLLM(), True)
            search = FakeSearch(
                results={"A社": ["https://new.example.com/stats"]},
                pages={"https://new.example.com/stats":
                       f"新ソース: {self.CLAIM}と統計庁の資料が示す。"
                       * 20})
            runner, locker = self._build(tmp, llm, search)
            chapters = {"1": f"## 1. 市場シェア\n\n{self.CLAIM} "
                             f"[SOURCE 1]。" + "背景。" * 40}
            outcome = runner.run(chapters)

            # ledger present in the outcome, with requirement leaves for
            # the question AND the section
            ledger = outcome["coverage_ledger"]
            ids = {r["req_id"] for r in ledger["requirements"]}
            self.assertIn("REQ-Q1", ids)
            self.assertIn("REQ-S1", ids)

            # the section requirement went open -> supported after the
            # gap research (transition history proves the reopen loop)
            sec = next(r for r in ledger["requirements"]
                       if r["req_id"] == "REQ-S1")
            self.assertIn("open", sec["history"])
            self.assertEqual(sec["status"], "supported")
            self.assertGreaterEqual(sec["search_attempts"], 1)

            # question requirement was ALWAYS supported (coverage said
            # answered) -> it was never searched: no query derived from
            # its text was sent to the search client
            self.assertTrue(search.searched)
            self.assertFalse(any("裏付けること" in q
                                 for q in search.searched))

            # measured progress round exists with real deltas
            self.assertGreaterEqual(len(ledger["rounds"]), 1)
            self.assertGreaterEqual(
                ledger["rounds"][0]["new_unique_evidence"], 1)

            # audit JSONL exists, records the round, and leaks no key
            audit = Path(tmp) / "audit_adapt1.jsonl"
            self.assertTrue(audit.exists())
            text = audit.read_text(encoding="utf-8")
            self.assertIn("research_round", text)
            self.assertIn("requirement_transition", text)
            self.assertNotIn("sk-test", text)
            for line in text.strip().split("\n"):
                json.loads(line)                    # well-formed JSONL

    def test_no_gap_means_no_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = RoutedLLM()
            llm.on("extract", j({"claims": [
                {"claim": self.CLAIM, "importance": "critical",
                 "source_numbers": [1]}]}))
            llm.on("judge", lambda p: j(
                {"status": "supported", "reason": "一致",
                 "supporting_source_ids": judge_ids(p)}))
            llm.MARKERS = dict(llm.MARKERS)
            llm.MARKERS["batch"] = ("各主張を、それぞれに提示されたエビデンス",)

            def batch(prompt):
                import re as _re
                cids = _re.findall(r"◆ 主張 (C-[\w.\-]+)", prompt)
                ids = judge_ids(prompt)
                return j({"verdicts": [
                    {"id": cid, "status": "supported", "reason": "一致",
                     "supporting_source_ids": ids} for cid in cids]})
            llm.on("batch", batch)
            llm.on("coverage", lambda p: j({"coverage": [
                {"question": 0, "answered": True, "section_id": "1",
                 "missing": "", "search_queries": []}]}))

            search = FakeSearch()
            runner, locker = self._build(tmp, llm, search)
            # evidence actually supports the claim this time
            locker.add_evidence(
                url="https://good.example.com/", title="良いソース",
                content_excerpt="x", extracted_text=f"{self.CLAIM}。" * 20,
                evidence_type=__import__(
                    "deep_research_tool.evidence.locker",
                    fromlist=["EvidenceType"]).EvidenceType.WEB_PAGE)
            chapters = {"1": f"## 1. 市場シェア\n\n{self.CLAIM} "
                             f"[SOURCE 1]。" + "背景。" * 40}
            outcome = runner.run(chapters)
            self.assertEqual(search.searched, [])   # nothing to research
            ledger = outcome["coverage_ledger"]
            self.assertEqual(ledger["counts"]["open"], 0)


if __name__ == "__main__":
    unittest.main()
