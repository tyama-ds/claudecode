"""
Integration tests for the finalization pipeline through the REAL
production path:

    DeepResearchTool.run() / run_manual_research()
      -> EvidenceLocker (real)
      -> ClaimVerifier (real, fake LLM)
      -> FinalizationController / FinalizationRunner (real)
      -> ReportGenerator V1/V2/V3 (real)
      -> final report file on disk

Only the LLM and the web are faked: the LLM is a content-routed fake
(responses chosen by prompt content, not by call order) and the search
client serves canned pages. Everything else — config, locker, citation
registry, planner, controller, generators, file output — is the real
code.

Tests 1-22 correspond to the audit's required integration scenarios.
"""

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from deep_research_tool.config import create_config
from deep_research_tool.evidence.locker import (
    EvidenceLocker,
    EvidenceType,
    QualityCategory,
    QualityIndicators,
    SourceType,
)
from deep_research_tool.main import DeepResearchTool, run_manual_research
from deep_research_tool.report.finalization import (
    FRESHNESS_FAIL,
    FRESHNESS_NOT_REQUIRED,
    FRESHNESS_PASS,
    ISSUE_UNANSWERED_QUESTION,
    LoopBudget,
    StructuredVerdict,
    decide,
    passes_hard_gates,
    ResearchDecision,
)
from deep_research_tool.research.query_generator import (
    ResearchPlan,
    TableOfContents,
    TableOfContentsItem,
)
from deep_research_tool.research.researcher import ResearchSession, ResearchState
from deep_research_tool.utils.helpers import ResearchWarnings
from deep_research_tool.verification.claim_verifier import ClaimVerifier


# ---------------------------------------------------------------------------
# Harness: content-routed fake LLM, fake search, fake researcher
# ---------------------------------------------------------------------------

class RoutedLLM:
    """Fake LLM that routes responses by prompt content (order-free)."""

    MARKERS = {
        "extract": ("検証が必要な事実主張", "Extract factual claims"),
        "judge": ("次の主張を、提示されたエビデンス", "Verify the claim using ONLY"),
        "coverage": ("実質的に回答しているか", "SUBSTANTIVELY ANSWERS"),
        "edit": ("あなたはレポート編集者です", "You are a report editor."),
        "chapter": ("現在執筆中のセクション", "[CURRENT SECTION]"),
    }

    def __init__(self):
        self.handlers = {}
        self.calls = []          # (kind, prompt)

    def on(self, kind, handler):
        """handler: str, or callable(prompt) -> str"""
        self.handlers[kind] = handler
        return self

    def prompts(self, kind):
        return [p for k, p in self.calls if k == kind]

    def _kind(self, prompt):
        for kind, markers in self.MARKERS.items():
            if any(m in prompt for m in markers):
                return kind
        return "other"

    def generate(self, prompt, **kwargs):
        kind = self._kind(prompt)
        self.calls.append((kind, prompt))
        handler = self.handlers.get(kind)
        if handler is None:
            content = "{}"
        elif callable(handler):
            content = handler(prompt)
        else:
            content = handler
        r = MagicMock()
        r.content = content if content is not None else ""
        return r


def judge_ids(prompt):
    """Evidence ids offered to a judge prompt (for supported responses)."""
    tail = prompt.split("【エビデンス】")[-1]
    return re.findall(r"^\[([^\]]+)\]", tail, re.M)


def j(obj):
    return json.dumps(obj, ensure_ascii=False)


class FakeSearch:
    """Canned search results / page bodies; records every fetch."""

    def __init__(self, results=None, pages=None):
        self.results = results or {}    # query prefix -> [urls]
        self.pages = pages or {}        # url -> text
        self.searched = []
        self.fetched = []

    def search(self, query, **kwargs):
        self.searched.append(query)
        for prefix, urls in self.results.items():
            if query.startswith(prefix) or prefix in query:
                return [SimpleNamespace(url=u, title=f"t:{u}", snippet="")
                        for u in urls]
        return []

    def get_page_content(self, url, **kwargs):
        self.fetched.append(url)
        text = self.pages.get(url, "")
        return SimpleNamespace(url=url, title=f"page:{url}",
                               text_content=text, images=[])

    def close(self):
        pass


class FakeResearcher:
    def __init__(self, session, locker):
        self.session = session
        self.locker = locker

    def conduct_research(self, **kwargs):
        return self.session

    def get_evidence_locker(self):
        return self.locker

    def expand_section_content(self, **kwargs):
        return {"characters_added": 0, "new_sources": 0}

    def get_session(self):
        return self.session


def make_plan(items):
    """items: [(section, title, description)] — top-level unless '.' in id."""
    top = []
    for sec, title, desc in items:
        node = TableOfContentsItem(section=sec, title=title, description=desc)
        if "." in sec and top:
            top[-1].subsections.append(node)
        else:
            top.append(node)
    toc = TableOfContents(title="統合テストレポート", items=top)
    return ResearchPlan(title="統合テストレポート", summary="s",
                        table_of_contents=toc, search_queries=[])


def make_session(plan, sections):
    """sections: {sid: {"title", "content", "evidence": [(url, text)]}}"""
    session = ResearchSession(session_id="itest", query="市場調査",
                              state=ResearchState.COMPLETED,
                              research_plan=plan)
    for sid, spec in sections.items():
        extracted = [{"title": f"t:{url}", "url": url,
                      "content": text[:2000], "raw_content": text,
                      "key_points": [text[:40]], "relevance_score": 0.8}
                     for url, text in spec.get("evidence", [])]
        session.section_contents[sid] = {
            "title": spec.get("title", sid),
            "content": spec.get("content", ""),
            "sources": [url for url, _ in spec.get("evidence", [])],
            "extracted_content": extracted,
            "confidence": "high",
        }
    return session


def make_locker(tmp, evidence):
    """evidence: [(url, text)] -> real EvidenceLocker."""
    locker = EvidenceLocker(research_id="itest",
                            output_dir=Path(tmp) / "evidence")
    for url, text in evidence:
        locker.add_evidence(url=url, title=f"t:{url}",
                            content_excerpt=text[:400], extracted_text=text,
                            evidence_type=EvidenceType.WEB_PAGE)
    return locker


def build_tool(tmp, llm, search, **config_kwargs):
    defaults = dict(
        provider="openai", openai_api_key="sk-test",
        output_dir=str(tmp), plan_review=False,
        auto_figures=False,
        v2_enable_consistency_check=False,
        v2_enable_two_phase=False,
        v2_enable_polish=False,
        v2_include_glossary=False,
    )
    defaults.update(config_kwargs)
    config = create_config(**defaults)
    tool = DeepResearchTool.__new__(DeepResearchTool)
    tool.config = config
    tool.llm_client = llm
    tool.stage_llm_clients = {}
    tool.search_client = search
    tool.researcher = None
    tool.verifier = None
    tool.report_generator = None
    tool.deep_think_processor = None
    return tool


def run_tool(tool, session, locker, query="市場調査", requirements=""):
    fake = FakeResearcher(session, locker)
    with patch("deep_research_tool.main.Researcher",
               new=lambda **kw: fake):
        return tool.run(query=query, requirements=requirements)


# Default LLM behaviors -----------------------------------------------------

CLAIM_SENTENCE = "A社の国内シェアは52%である"
SUPPORT_PHRASE = "国内シェアは52%"


def all_answered_coverage(prompt):
    return j({"coverage": [{"question": i, "answered": True,
                            "section_id": "1"} for i in range(15)]})


def default_extract(prompt):
    if CLAIM_SENTENCE in prompt:
        return j({"claims": [{"claim": CLAIM_SENTENCE,
                              "importance": "critical",
                              "source_numbers": [1]}]})
    return j({"claims": []})


def support_if_evidence(prompt):
    """Judge: supported iff the offered evidence contains the phrase."""
    if SUPPORT_PHRASE in prompt.split("【エビデンス】")[-1]:
        return j({"status": "supported", "reason": "一致",
                  "supporting_source_ids": judge_ids(prompt)})
    return j({"status": "unsupported", "reason": "根拠なし",
              "supporting_source_ids": []})


def make_llm():
    return (RoutedLLM()
            .on("extract", default_extract)
            .on("judge", support_if_evidence)
            .on("coverage", all_answered_coverage))


def long_body(sid, title, sentence, chars=800):
    filler = ("本章では市場の構造、需要動向、主要企業の動きについて"
              "多面的に検討し、根拠に基づく評価を述べる。")
    body = f"## {sid}. {title}\n\n{sentence} [SOURCE 1]。"
    while len(body) < chars:
        body += filler + str(len(body) % 7) + "。"
    return body


SUPPORTED_EVIDENCE = [("https://ev1.example.com/a",
                       f"調査会社の統計によればA社の{SUPPORT_PHRASE}と報告されている。" * 5)]


class Base(unittest.TestCase):
    def setUp(self):
        ResearchWarnings.reset()
        self.tmp = Path(tempfile.mkdtemp())

    def default_session(self, content=None, plan_items=None):
        plan = make_plan(plan_items or
                         [("1", "市場シェア", "A社のシェアを明らかにする")])
        return make_session(plan, {
            "1": {"title": "市場シェア",
                  "content": content or long_body(
                      "1", "市場シェア", CLAIM_SENTENCE),
                  "evidence": SUPPORTED_EVIDENCE},
        })


# ---------------------------------------------------------------------------
# 1-5: every report version through the ONE finalization path
# ---------------------------------------------------------------------------

class TestCommonPath(Base):

    def test_01_default_v1_path_runs_finalization(self):
        """Default config (V1) verifies the final body and renders it."""
        llm = make_llm()
        locker = make_locker(self.tmp, SUPPORTED_EVIDENCE)
        tool = build_tool(self.tmp, llm, FakeSearch())
        result = run_tool(tool, self.default_session(), locker)

        self.assertIsInstance(result["verification_result"], StructuredVerdict)
        self.assertEqual(tool.finalization_outcome["decision"], "accept")
        # the verifier extracted claims from the FINAL body
        self.assertTrue(llm.prompts("extract"))
        self.assertIn(CLAIM_SENTENCE, llm.prompts("extract")[0])
        report = Path(result["report_path"]).read_text(encoding="utf-8")
        self.assertIn(CLAIM_SENTENCE, report)
        self.assertTrue(Path(result["verification_html"]).exists())

    def test_02_v2_path_rendered_citations_and_references(self):
        llm = make_llm().on(
            "chapter",
            lambda p: long_body("1", "市場シェア", CLAIM_SENTENCE)
            + "\n===CHAPTER_META===\n"
            + j({"key_points": ["シェア52%"], "terms_used": [],
                 "facts_stated": []}))
        locker = make_locker(self.tmp, SUPPORTED_EVIDENCE)
        tool = build_tool(self.tmp, llm, FakeSearch(),
                          report_generator_version="v2")
        result = run_tool(tool, self.default_session(), locker)

        report = Path(result["report_path"]).read_text(encoding="utf-8")
        body = report.split("参考文献")[0]
        self.assertIn("[1]", body)                 # display number rendered
        self.assertNotIn("[SOURCE", body)          # no editing tags remain
        self.assertIn("参考文献", report)
        self.assertEqual(tool.finalization_outcome["decision"], "accept")

    def test_03_v3_path_through_finalization(self):
        llm = make_llm().on(
            "chapter",
            lambda p: long_body("1", "市場シェア", CLAIM_SENTENCE)
            + "\n===CHAPTER_META===\n" + j({"key_points": []}))
        locker = make_locker(self.tmp, SUPPORTED_EVIDENCE)
        tool = build_tool(self.tmp, llm, FakeSearch(),
                          report_generator_version="v3",
                          output_format="docx")
        result = run_tool(tool, self.default_session(), locker)

        self.assertTrue(Path(result["report_path"]).exists())
        self.assertTrue(str(result["report_path"]).endswith(".docx"))
        self.assertEqual(tool.finalization_outcome["decision"], "accept")
        chapters = tool.finalization_outcome["chapters"]
        self.assertIn("[1]", chapters["1"])
        self.assertNotIn("[SOURCE", chapters["1"])

    def test_04_manual_mode_through_finalization(self):
        llm = make_llm()
        session = self.default_session()
        locker = make_locker(self.tmp, SUPPORTED_EVIDENCE)
        fake = FakeResearcher(session, locker)
        fake.load_evidence_from_file = lambda **kw: None
        with patch("deep_research_tool.main.ManualResearcher",
                   new=lambda **kw: fake), \
             patch("deep_research_tool.main.get_client",
                   new=lambda **kw: llm):
            result = run_manual_research(
                evidence_file="dummy.csv", topic="市場調査",
                provider="openai", api_key="sk-test",
                output_format="markdown", output_dir=str(self.tmp),
                manual_toc_sections=None, auto_toc=True,
            )
        self.assertIsInstance(result["verification_result"], StructuredVerdict)
        report = Path(result["report_path"]).read_text(encoding="utf-8")
        self.assertIn(CLAIM_SENTENCE, report)
        self.assertTrue(Path(result["evidence_json"]).exists())

    def test_05_v1_no_length_adjustment_after_verification(self):
        """A tiny legacy target_characters must NOT re-cut the verified
        body: the frozen text ships unchanged."""
        llm = make_llm()
        content = long_body("1", "市場シェア", CLAIM_SENTENCE, chars=2500)
        locker = make_locker(self.tmp, SUPPORTED_EVIDENCE)
        tool = build_tool(self.tmp, llm, FakeSearch(),
                          target_characters=200)   # legacy fixed target
        session = self.default_session(content=content)
        result = run_tool(tool, session, locker)

        frozen = tool.finalization_outcome["chapters"]["1"]
        report = Path(result["report_path"]).read_text(encoding="utf-8")
        # the exact frozen body tail is in the report — not a shortened
        # variant (the V1 renderer only reformats [1] as footnote [^1])
        self.assertIn(frozen[-400:], report)
        self.assertGreater(len(frozen), 2000)


# ---------------------------------------------------------------------------
# 6-8, 20-21: live locker, propagation, dedup, loop control, exports
# ---------------------------------------------------------------------------

NEW_URL = "https://new-source.example.com/data"
NEW_TEXT = (f"業界レポートによると、A社の{SUPPORT_PHRASE}であると"
            f"確認された。生産能力と出荷実績の両面から裏付けられる。" * 6)
OLD_TEXT = "無関係な一般論。景気動向と雇用の話題。" * 30


class TestResearchRounds(Base):

    def _research_setup(self, pages=None, results=None):
        """Section content whose claim is NOT supported by initial
        evidence; supporting content is only reachable via research."""
        llm = (RoutedLLM()
               .on("extract", default_extract)
               .on("judge", support_if_evidence)
               .on("coverage", all_answered_coverage)
               .on("edit", lambda p:
                   f"## 1. 市場シェア\n\n{CLAIM_SENTENCE} [SOURCE 1]"
                   f" [SOURCE 2]。追加調査の結果を反映した本文。" * 10))
        locker = make_locker(
            self.tmp, [("https://old.example.com/x", OLD_TEXT)])
        search = FakeSearch(
            results=results or {"": [NEW_URL]},
            pages=pages or {NEW_URL: NEW_TEXT})
        plan = make_plan([("1", "市場シェア", "A社のシェア")])
        session = make_session(plan, {
            "1": {"title": "市場シェア",
                  "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                  "evidence": [("https://old.example.com/x", OLD_TEXT)]},
        })
        return llm, locker, search, session

    def test_06_live_locker_second_pass_sees_new_evidence(self):
        llm, locker, search, session = self._research_setup()
        tool = build_tool(self.tmp, llm, search)
        run_tool(tool, session, locker)

        new_ev = locker.get_by_url(NEW_URL)
        self.assertIsNotNone(new_ev)     # research round updated the locker
        # a judge pass AFTER the research used the LIVE locker: the new
        # evidence id appears in a judge prompt
        self.assertTrue(any(new_ev.id in p for p in llm.prompts("judge")))
        self.assertEqual(tool.finalization_outcome["decision"], "accept")

    def test_07_new_evidence_propagates_everywhere(self):
        llm, locker, search, session = self._research_setup()
        tool = build_tool(self.tmp, llm, search)
        run_tool(tool, session, locker)

        new_ev = locker.get_by_url(NEW_URL)
        runner = tool.finalization_runner
        # citation registry: appended under a stable new number
        self.assertIn(new_ev.id, runner.citation_mgr.evidence_ids("1"))
        # numbers of pre-existing evidence did not shift
        self.assertEqual(runner.citation_mgr.mapping("1")[1],
                         locker.get_by_url("https://old.example.com/x").id)
        # section-evidence relations updated
        urls = [ec.get("url") for ec in
                session.section_contents["1"]["extracted_content"]]
        self.assertIn(NEW_URL, urls)
        # planner units recalculated with the new evidence
        self.assertGreater(runner.planner.plans["1"].units, 0)

    def test_08_duplicate_repost_not_counted_as_new_source(self):
        dup_url = "https://mirror.example.com/copy"
        llm, locker, search, session = self._research_setup(
            results={"": [dup_url]},
            pages={dup_url: OLD_TEXT})   # same content, different URL
        tool = build_tool(self.tmp, llm, search)
        run_tool(tool, session, locker)

        self.assertIsNone(locker.get_by_url(dup_url))   # never added
        self.assertEqual(tool.finalization_outcome["decision"],
                         "finalize_with_limitations")

    def test_20_seen_urls_never_refetched_and_stagnation_stops(self):
        known = "https://old.example.com/x"
        llm, locker, search, session = self._research_setup(
            results={"": [known]}, pages={known: OLD_TEXT})
        tool = build_tool(self.tmp, llm, search,
                          max_final_research_rounds=5,
                          max_no_improvement_rounds=1)
        run_tool(tool, session, locker)

        # the already-collected URL was never fetched again
        self.assertNotIn(known, search.fetched)
        # stagnation stopped the loop long before 5 rounds
        self.assertEqual(tool.finalization_outcome["decision"],
                         "finalize_with_limitations")
        self.assertLessEqual(tool.finalization_runner.budget.research_rounds, 2)

    def test_21_exports_after_finalization_include_new_evidence(self):
        llm, locker, search, session = self._research_setup()
        tool = build_tool(self.tmp, llm, search)
        result = run_tool(tool, session, locker)

        exported = Path(result["evidence_json"]).read_text(encoding="utf-8")
        self.assertIn(NEW_URL, exported)    # researched evidence exported
        csv_text = Path(result["evidence_csv"]).read_text(encoding="utf-8") \
            if result.get("evidence_csv") and result["evidence_csv"] != "None" \
            else ""
        del csv_text  # json export is the default format


# ---------------------------------------------------------------------------
# 9-12: canonical ids, rendering, citation integrity, cited-first
# ---------------------------------------------------------------------------

class TestCitations(Base):

    def test_09_canonical_evidence_ids_and_colon_variant(self):
        llm = make_llm()
        locker = make_locker(self.tmp, SUPPORTED_EVIDENCE)
        content = long_body("1", "市場シェア", CLAIM_SENTENCE)
        content = content.replace("[SOURCE 1]", "[SOURCE: 1]")  # legacy form
        tool = build_tool(self.tmp, llm, FakeSearch())
        result = run_tool(tool, self.default_session(content=content), locker)

        runner = tool.finalization_runner
        eid = locker.get_by_url("https://ev1.example.com/a").id
        # registry maps [SOURCE 1] -> canonical Evidence.id, not the URL
        self.assertEqual(runner.citation_mgr.mapping("1")[1], eid)
        # [SOURCE: 1] followed the same rules and rendered to [1]
        self.assertTrue(runner.citation_mgr.validate("1", content))
        report = Path(result["report_path"]).read_text(encoding="utf-8")
        # display number rendered (V1 markdown uses footnote form [^1])
        self.assertTrue("[1]" in report or "[^1]" in report, report[:500])
        self.assertNotIn("[SOURCE:", report)

    def test_10_references_follow_first_use_order(self):
        e1 = ("https://cite-a.example.com/1",
              f"A社の{SUPPORT_PHRASE}という統計。" * 8)
        e2 = ("https://cite-b.example.com/2",
              "B社の輸出台数は12万台と報告。" * 8)
        # locker insertion order is e2 FIRST — rendering must reorder
        locker = make_locker(self.tmp, [e2, e1])

        def extract(prompt):
            if CLAIM_SENTENCE in prompt:
                return j({"claims": [{"claim": CLAIM_SENTENCE,
                                      "importance": "important",
                                      "source_numbers": [1]}]})
            if "輸出台数は12万台" in prompt:
                return j({"claims": [{"claim": "B社の輸出台数は12万台",
                                      "importance": "important",
                                      "source_numbers": [1]}]})
            return j({"claims": []})

        def judge(prompt):
            return j({"status": "supported", "reason": "ok",
                      "supporting_source_ids": judge_ids(prompt)})

        llm = (RoutedLLM().on("extract", extract).on("judge", judge)
               .on("coverage", all_answered_coverage))
        plan = make_plan([("1", "シェア", ""), ("2", "輸出", "")])
        session = make_session(plan, {
            "1": {"title": "シェア",
                  "content": long_body("1", "シェア", CLAIM_SENTENCE),
                  "evidence": [e1]},
            "2": {"title": "輸出",
                  "content": long_body("2", "輸出", "B社の輸出台数は12万台"),
                  "evidence": [e2]},
        })
        tool = build_tool(self.tmp, llm, FakeSearch())
        result = run_tool(tool, session, locker)

        # locker was reordered to first-use order: e1 (cited in section 1)
        # comes before e2
        ordered = [e.url for e in locker.get_all_evidence()]
        self.assertEqual(ordered[0], e1[0])
        self.assertEqual(ordered[1], e2[0])
        report = Path(result["report_path"]).read_text(encoding="utf-8")
        # section 1 cites reference 1 (e1), section 2 cites reference 2
        # (V1 markdown renders display numbers as footnotes [^n])
        self.assertTrue("[1]" in report or "[^1]" in report)
        self.assertTrue("[2]" in report or "[^2]" in report)
        refs = report.split("References")[-1]
        self.assertLess(refs.index(e1[0]), refs.index(e2[0]))

    def test_11_edits_with_broken_citations_are_rejected(self):
        """An edit that invents [SOURCE 9] or deletes every citation is
        rejected by the machine check; the previous text survives."""
        state = {"n": 0}

        def extract(prompt):
            state["n"] += 1
            if CLAIM_SENTENCE in prompt and state["n"] == 1:
                # first pass: claim looks uncited -> rewrite path
                return j({"claims": [{"claim": CLAIM_SENTENCE,
                                      "importance": "important",
                                      "source_numbers": []}]})
            if CLAIM_SENTENCE in prompt:
                return j({"claims": [{"claim": CLAIM_SENTENCE,
                                      "importance": "important",
                                      "source_numbers": [1]}]})
            return j({"claims": []})

        edits = iter([
            "捏造引用の本文 [SOURCE 9] です。" * 20,        # invalid number
            "引用を全て削除した本文です。" * 20,            # deletes citations
        ])
        llm = (RoutedLLM()
               .on("extract", extract)
               .on("judge", support_if_evidence)
               .on("coverage", all_answered_coverage)
               .on("edit", lambda p: next(edits, "")))
        locker = make_locker(self.tmp, SUPPORTED_EVIDENCE)
        original = long_body("1", "市場シェア", CLAIM_SENTENCE)
        tool = build_tool(self.tmp, llm, FakeSearch())
        run_tool(tool, self.default_session(content=original), locker)

        self.assertEqual(tool.finalization_outcome["raw_chapters"]["1"],
                         original)     # both bad edits rejected

    def test_12_cited_evidence_is_checked_first(self):
        cited = ("https://cited.example.com/main",
                 "全く語彙の重ならない資料。財務諸表の注記等。" * 10)
        decoy = ("https://decoy.example.com/similar",
                 f"A社の国内シェアは52%であるがこれは誤植の多い転載記事。" * 10)
        locker = make_locker(self.tmp, [cited, decoy])
        # section evidence order: cited is [SOURCE 1], decoy is [SOURCE 2]
        plan = make_plan([("1", "シェア", "")])
        session = make_session(plan, {
            "1": {"title": "シェア",
                  "content": long_body("1", "シェア", CLAIM_SENTENCE),
                  "evidence": [cited, decoy]},
        })
        llm = make_llm()
        tool = build_tool(self.tmp, llm, FakeSearch())
        run_tool(tool, session, locker)

        cited_id = locker.get_by_url(cited[0]).id
        judge_prompts = llm.prompts("judge")
        self.assertTrue(judge_prompts)
        # despite zero lexical overlap, the CITED evidence ([SOURCE 1])
        # is present in the judgement evidence
        self.assertTrue(any(cited_id in p for p in judge_prompts),
                        "cited evidence must be judged first")


# ---------------------------------------------------------------------------
# 13-15: fail-closed verifier, status handling, deep chunks
# ---------------------------------------------------------------------------

class TestFailClosed(Base):

    def test_13_zero_claims_on_factual_body_is_verification_failure(self):
        llm = (RoutedLLM()
               .on("extract", j({"claims": []}))
               .on("coverage", all_answered_coverage))
        locker = make_locker(self.tmp, SUPPORTED_EVIDENCE)
        tool = build_tool(self.tmp, llm, FakeSearch())
        result = run_tool(tool, self.default_session(), locker)

        verdict = result["verification_result"]
        self.assertTrue(verdict.metrics.verification_failed)
        self.assertEqual(verdict.metrics.claim_support_score, 0.0)
        # NEVER accept an unverifiable body
        self.assertEqual(tool.finalization_outcome["decision"],
                         "finalize_with_limitations")
        self.assertTrue(any("検証" in w["message"]
                            for w in result["warnings"]))

    def test_14_unknown_status_and_zero_source_supported_downgraded(self):
        def extract(prompt):
            if CLAIM_SENTENCE in prompt:
                return j({"claims": [
                    {"claim": CLAIM_SENTENCE, "importance": "minor",
                     "source_numbers": [1]},
                    {"claim": "B社は黒字である", "importance": "minor",
                     "source_numbers": [1]},
                ]})
            return j({"claims": []})

        def judge(prompt):
            # content-routed so EVERY verification pass (including the
            # post-limitations re-verify) sees the same judgements
            claim_part = prompt.split("【エビデンス】")[0]
            if "B社は黒字" in claim_part:
                return j({"status": "supported", "reason": "ok",
                          "supporting_source_ids": []})   # no valid sources
            return j({"status": "totally_fine", "reason": "?"})  # unknown

        llm = (RoutedLLM()
               .on("extract", extract)
               .on("judge", judge)
               .on("coverage", all_answered_coverage)
               .on("edit", lambda p: ""))
        evidence = SUPPORTED_EVIDENCE + [
            ("https://ev2.example.com/b", "B社は黒字であると決算短信が示す。" * 6)]
        locker = make_locker(self.tmp, evidence)
        content = long_body("1", "市場シェア",
                            CLAIM_SENTENCE + "。B社は黒字である")
        plan = make_plan([("1", "市場シェア", "A社のシェア")])
        session = make_session(plan, {
            "1": {"title": "市場シェア", "content": content,
                  "evidence": evidence},
        })
        tool = build_tool(self.tmp, llm, FakeSearch(),
                          max_final_research_rounds=0,
                          max_final_revision_rounds=0)
        result = run_tool(tool, session, locker)

        verdict = result["verification_result"]
        # both claims became "uncertain": unknown status + supported w/o
        # any valid source are never silently fine
        self.assertGreaterEqual(verdict.metrics.uncertain_count, 2)
        self.assertLess(verdict.metrics.claim_support_score, 1.0)

    def test_15_fact_deep_inside_long_evidence_is_found(self):
        deep_fact = "特殊合金の融点は1892度と測定された"
        long_text = ("前置きの説明が続く。" * 160) + deep_fact + ("。後続の記述。" * 40)
        self.assertGreater(long_text.index(deep_fact), 1500)

        verifier = ClaimVerifier(
            llm_client=(RoutedLLM().on("judge", lambda p: j(
                {"status": "supported", "reason": "ok",
                 "supporting_source_ids": judge_ids(p)}))
                .on("extract", j({"claims": [
                    {"claim": deep_fact, "importance": "critical",
                     "source_numbers": []}]}))),
            language="ja")
        locker = make_locker(self.tmp,
                             [("https://deep.example.com/long", long_text)])
        verdict = verifier.verify_report(
            {"1": f"## 1\n\n{deep_fact} [SOURCE 1]。" + "説明。" * 100},
            locker.get_all_evidence())
        # the chunk PAST the old 800/1500-char caps was offered to the judge
        judge_prompts = verifier.llm.prompts("judge")
        self.assertTrue(any("1892度" in p for p in judge_prompts),
                        "deep chunk content must reach the judge")
        self.assertEqual(verdict.metrics.unsupported_count, 0)


# ---------------------------------------------------------------------------
# 16-17: coverage (answered vs mentioned), primary/freshness 3-state
# ---------------------------------------------------------------------------

class TestCoverageAndFreshness(Base):

    def test_16_unanswered_question_generates_issue_and_queries(self):
        def coverage(prompt):
            return j({"coverage": [
                {"question": 0, "answered": True, "section_id": "1"},
                {"question": 1, "answered": False, "section_id": "1",
                 "missing": "価格動向の分析がない",
                 "search_queries": ["A社 価格動向 2026"]},
            ]})

        llm = (RoutedLLM()
               .on("extract", default_extract)
               .on("judge", support_if_evidence)
               .on("coverage", coverage)
               .on("edit", lambda p: ""))
        locker = make_locker(self.tmp, SUPPORTED_EVIDENCE)
        search = FakeSearch()      # returns no results
        plan = make_plan([("1", "市場シェア", "A社のシェア"),
                          ("2", "価格動向", "価格の推移")])
        session = make_session(plan, {
            "1": {"title": "市場シェア",
                  "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                  "evidence": SUPPORTED_EVIDENCE},
        })
        tool = build_tool(self.tmp, llm, search)
        result = run_tool(tool, session, locker)

        verdict = result["verification_result"]
        unanswered = verdict.issues_of(ISSUE_UNANSWERED_QUESTION)
        self.assertTrue(unanswered)
        self.assertTrue(all(i.search_queries for i in unanswered))
        self.assertLess(verdict.metrics.critical_question_coverage, 1.0)
        # the generated query actually drove the research round
        self.assertTrue(any("価格動向" in q for q in search.searched))
        self.assertEqual(tool.finalization_outcome["decision"],
                         "finalize_with_limitations")

    def test_17_primary_freshness_three_states(self):
        # (a) not required by default
        state, _ = ClaimVerifier.assess_primary_freshness([], required=False)
        self.assertEqual(state, FRESHNESS_NOT_REQUIRED)

        # (b) required + stale, non-primary evidence -> FAIL, gates fail
        stale = SimpleNamespace(
            published_date="2015-04-01",
            quality_indicators=QualityIndicators(is_primary_source=False),
            source_type=SourceType.BLOG,
            quality_category=QualityCategory.LOW)
        state, why = ClaimVerifier.assess_primary_freshness(
            [stale], required=True)
        self.assertEqual(state, FRESHNESS_FAIL)
        self.assertTrue(why)
        verdict = StructuredVerdict()
        verdict.metrics.primary_freshness = FRESHNESS_FAIL
        self.assertFalse(passes_hard_gates(verdict, LoopBudget()))
        budget = LoopBudget(max_final_research_rounds=0)
        self.assertEqual(decide(verdict, budget),
                         ResearchDecision.FINALIZE_WITH_LIMITATIONS)

        # (c) fresh + primary -> PASS
        fresh = SimpleNamespace(
            published_date="2026-05-01",
            quality_indicators=QualityIndicators(is_primary_source=True),
            source_type=SourceType.OFFICIAL,
            quality_category=QualityCategory.AUTHORITATIVE)
        state, _ = ClaimVerifier.assess_primary_freshness(
            [fresh], required=True)
        self.assertEqual(state, FRESHNESS_PASS)

        # (d) the requirement is detected from the query wording
        llm = make_llm()
        locker = make_locker(self.tmp, SUPPORTED_EVIDENCE)
        tool = build_tool(self.tmp, llm, FakeSearch(),
                          max_final_research_rounds=0)
        result = run_tool(tool, self.default_session(), locker,
                          query="A社の最新の市場シェア")
        self.assertEqual(result["verification_result"]
                         .metrics.primary_freshness, FRESHNESS_FAIL)


# ---------------------------------------------------------------------------
# 18-19: adaptive length wiring, fixed / hard_min / hard_max
# ---------------------------------------------------------------------------

class TestLengthModes(Base):

    def test_18_stage1_allocation_feeds_v2_draft_prompts(self):
        def chapter(prompt):
            m = re.search(r"セクション番号: (\S+)", prompt)
            sid = m.group(1) if m else "1"
            title = "市場シェア" if sid == "1" else "詳細"
            return (long_body(sid, title, CLAIM_SENTENCE, chars=3800)
                    + "\n===CHAPTER_META===\n" + j({"key_points": []}))

        llm = make_llm().on("chapter", chapter)
        locker = make_locker(self.tmp, SUPPORTED_EVIDENCE)
        plan = make_plan([("1", "市場シェア", "本論"),
                          ("1.1", "詳細", "補足")])
        session = make_session(plan, {
            "1": {"title": "市場シェア",
                  "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                  "evidence": SUPPORTED_EVIDENCE},
            "1.1": {"title": "詳細",
                    "content": long_body("1.1", "詳細", CLAIM_SENTENCE),
                    "evidence": SUPPORTED_EVIDENCE},
        })
        tool = build_tool(self.tmp, llm, FakeSearch(),
                          report_generator_version="v2",
                          preferred_body_chars=9000)
        run_tool(tool, session, locker)

        chapter_prompts = llm.prompts("chapter")
        p1 = next(p for p in chapter_prompts if "セクション番号: 1\n" in p)
        p11 = next(p for p in chapter_prompts if "セクション番号: 1.1" in p)
        # importance-weighted stage-1 allocation: 9000 * 1.5/2.5 = 5400
        # for the top-level chapter, 9000 * 1.0/2.5 = 3600 for the sub
        self.assertIn("約5400文字", p1)
        self.assertIn("約3600文字", p11)

    def test_19_hard_min_never_pads_and_hard_max_never_completes_normally(self):
        # (a) hard_min: short, evidence-poor body -> limitations, NO padding
        llm = make_llm().on("edit", lambda p: "")
        locker = make_locker(self.tmp, SUPPORTED_EVIDENCE)
        short = long_body("1", "市場シェア", CLAIM_SENTENCE, chars=600)
        tool = build_tool(self.tmp, llm, FakeSearch(),
                          hard_min_body_chars=50000)
        run_tool(tool, self.default_session(content=short), locker)
        outcome = tool.finalization_outcome
        self.assertEqual(outcome["decision"], "finalize_with_limitations")
        # body was not inflated toward the floor
        self.assertLess(
            outcome["verdict"].metrics.actual_body_chars, 5000)

        # (b) hard_max exceeded and compression fails -> NOT a normal
        # completion (over_hard_max flag + CRITICAL warning)
        ResearchWarnings.reset()
        llm2 = make_llm().on("edit", lambda p: "")
        locker2 = make_locker(self.tmp, SUPPORTED_EVIDENCE)
        big = long_body("1", "市場シェア", CLAIM_SENTENCE, chars=4000)
        tool2 = build_tool(self.tmp, llm2, FakeSearch(),
                           hard_max_body_chars=300)
        result2 = run_tool(tool2, self.default_session(content=big), locker2)
        outcome2 = tool2.finalization_outcome
        self.assertEqual(outcome2["decision"], "finalize_with_limitations")
        self.assertTrue(outcome2["over_hard_max"])
        self.assertTrue(any("上限文字数" in w["message"]
                            for w in result2["warnings"]))

        # (c) adaptive mode + legacy target alone never triggers research
        llm3 = make_llm()
        locker3 = make_locker(self.tmp, SUPPORTED_EVIDENCE)
        search3 = FakeSearch()
        tool3 = build_tool(self.tmp, llm3, search3,
                           target_characters=99999)   # far above actual
        run_tool(tool3, self.default_session(), locker3)
        self.assertEqual(tool3.finalization_outcome["decision"], "accept")
        self.assertEqual(search3.searched, [])


# ---------------------------------------------------------------------------
# 22: CLI — the report command executes (TypeError fixed), options work
# ---------------------------------------------------------------------------

class TestCLI(Base):

    def _make_session_file(self):
        plan = make_plan([("1", "市場シェア", "A社のシェア")])
        session = make_session(plan, {
            "1": {"title": "市場シェア",
                  "content": long_body("1", "市場シェア", CLAIM_SENTENCE),
                  "evidence": SUPPORTED_EVIDENCE},
        })
        path = self.tmp / "session_itest.json"
        session.save(path)
        return path

    def test_22_cli_report_and_research_options(self):
        from click.testing import CliRunner
        from deep_research_tool.cli import cli

        runner = CliRunner()
        session_file = self._make_session_file()

        # (a) the previous TypeError: report() got an unexpected keyword
        #     argument 'length_mode' — the callback must now execute
        res = runner.invoke(cli, ["report", str(session_file),
                                  "--length-mode", "adaptive"])
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Report generated", res.output)

        # (b) fixed mode with an explicit preferred length
        res = runner.invoke(cli, ["report", str(session_file),
                                  "--length-mode", "fixed",
                                  "--preferred-body-chars", "500"])
        self.assertEqual(res.exit_code, 0, res.output)

        # (c) invalid hard bounds are rejected with a clear error
        res = runner.invoke(cli, ["report", str(session_file),
                                  "--hard-min-body-chars", "100",
                                  "--hard-max-body-chars", "50"])
        self.assertEqual(res.exit_code, 1)
        self.assertIn("hard-min", res.output)

        # (d) fixed mode without a target is rejected
        res = runner.invoke(cli, ["report", str(session_file),
                                  "--length-mode", "fixed"])
        self.assertEqual(res.exit_code, 1)

        # (e) research command accepts the new options end-to-end
        #     (aborted at the confirmation prompt after config validation)
        res = runner.invoke(cli, [
            "research", "テスト調査",
            "--openai-key", "sk-test",
            "--report-version", "v2",
            "--length-mode", "fixed",
            "--preferred-body-chars", "12000",
            "--hard-max-body-chars", "30000",
            "--length-tolerance", "0.3",
            "--max-final-research-rounds", "3",
            "--max-final-revision-rounds", "1",
            "--max-no-improvement-rounds", "2",
            "--min-score-improvement", "0.05",
            "--min-new-independent-sources", "2",
            "--min-claim-support-score", "0.9",
            "--required-critical-coverage", "0.8",
        ], input="n\n")
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Cancelled", res.output)

        # (f) invalid threshold fails config validation before running
        res = runner.invoke(cli, [
            "research", "テスト調査",
            "--openai-key", "sk-test",
            "--min-claim-support-score", "1.5",
        ], input="n\n")
        self.assertEqual(res.exit_code, 1)
        self.assertIn("min_claim_support_score", res.output)


# ---------------------------------------------------------------------------
# WebUI parameter conversion / rejection (item 15)
# ---------------------------------------------------------------------------

class TestWebUIParams(unittest.TestCase):

    def test_all_twelve_settings_are_mapped(self):
        from deep_research_tool.webui.server import _CONFIG_PARAM_MAP
        for key in ("length_mode", "preferred_body_chars",
                    "hard_min_body_chars", "hard_max_body_chars",
                    "length_tolerance", "max_final_research_rounds",
                    "max_final_revision_rounds", "max_no_improvement_rounds",
                    "min_score_improvement", "min_new_independent_sources",
                    "min_claim_support_score", "required_critical_coverage"):
            self.assertIn(key, _CONFIG_PARAM_MAP)

    def test_type_conversion_and_empty_means_none(self):
        from deep_research_tool.webui.server import build_config_kwargs
        kwargs = build_config_kwargs({
            "query": "q",
            "preferred_body_chars": "20000",      # string -> int
            "length_tolerance": "0.5",            # string -> float
            "hard_max_body_chars": "",            # empty -> dropped (None)
            "max_final_research_rounds": 3,
            "min_claim_support_score": "0.9",
            "length_mode": "Fixed",               # normalized
        })
        self.assertEqual(kwargs["preferred_body_chars"], 20000)
        self.assertEqual(kwargs["length_tolerance"], 0.5)
        self.assertNotIn("hard_max_body_chars", kwargs)
        self.assertEqual(kwargs["max_final_research_rounds"], 3)
        self.assertEqual(kwargs["min_claim_support_score"], 0.9)
        self.assertEqual(kwargs["length_mode"], "fixed")

    def test_invalid_values_rejected(self):
        from deep_research_tool.webui.server import build_config_kwargs
        for bad in (
            {"preferred_body_chars": "abc"},
            {"preferred_body_chars": "-5"},
            {"length_tolerance": "1.0"},          # must be < 1
            {"min_claim_support_score": "1.5"},
            {"max_final_research_rounds": "-1"},
            {"length_mode": "bogus"},
            {"hard_min_body_chars": "200", "hard_max_body_chars": "100"},
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                build_config_kwargs(bad)


# ---------------------------------------------------------------------------
# Config validation (item 16)
# ---------------------------------------------------------------------------

class TestConfigValidation(unittest.TestCase):

    def _errors(self, **kw):
        config = create_config(provider="openai", openai_api_key="sk-test",
                               **kw)
        return config.validate()

    def test_valid_defaults_pass(self):
        self.assertEqual(self._errors(), [])

    def test_length_mode_enum(self):
        config = create_config(provider="openai", openai_api_key="sk-test")
        config.report.length_mode = "bogus"
        self.assertTrue(any("length_mode" in e for e in config.validate()))

    def test_negative_chars_rejected(self):
        self.assertTrue(any("preferred_body_chars" in e for e in
                            self._errors(preferred_body_chars=-1)))

    def test_hard_min_le_hard_max(self):
        errors = self._errors(hard_min_body_chars=200,
                              hard_max_body_chars=100)
        self.assertTrue(any("hard_min_body_chars" in e for e in errors))

    def test_tolerance_range(self):
        self.assertTrue(any("length_tolerance" in e for e in
                            self._errors(length_tolerance=1.0)))
        self.assertEqual([e for e in self._errors(length_tolerance=0.0)
                          if "length_tolerance" in e], [])

    def test_fixed_mode_requires_target(self):
        self.assertTrue(any("fixed" in e for e in
                            self._errors(length_mode="fixed")))
        self.assertEqual(
            [e for e in self._errors(length_mode="fixed",
                                     preferred_body_chars=1000)
             if "fixed" in e], [])

    def test_rounds_and_thresholds(self):
        self.assertTrue(any("max_final_research_rounds" in e for e in
                            self._errors(max_final_research_rounds=-1)))
        self.assertTrue(any("min_claim_support_score" in e for e in
                            self._errors(min_claim_support_score=1.5)))
        self.assertTrue(any("required_critical_coverage" in e for e in
                            self._errors(required_critical_coverage=-0.1)))
        self.assertTrue(any("min_score_improvement" in e for e in
                            self._errors(min_score_improvement=-0.5)))


if __name__ == "__main__":
    unittest.main()
