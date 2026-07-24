"""
Tests for the verification profiles and the fast verification engine
(parallel / batch / cache / differential verification / cancel).
"""

import json
import threading
import time
import unittest
from unittest.mock import MagicMock

from deep_research_tool.config import create_config
from deep_research_tool.report.citations import CitationManager
from deep_research_tool.verification.claim_verifier import ClaimVerifier
from deep_research_tool.verification.profiles import (
    CUSTOM_WARNING,
    resolve_verification_settings,
    settings_from_research_config,
)
from deep_research_tool.verification.runtime import (
    PROMPT_VERSION,
    VerificationCache,
    VerificationCancelled,
    VerificationProgress,
    stable_hash,
)


def j(obj):
    return json.dumps(obj, ensure_ascii=False)


class CountingLLM:
    """Content-routed fake LLM with thread-safe call accounting."""

    def __init__(self, latency=0.0, model="fake-model"):
        self.model = model
        self.latency = latency
        self.lock = threading.Lock()
        self.calls = []            # prompts
        self.active = 0
        self.peak = 0
        self.cancel_after = None   # cancel this progress after N calls
        self.progress = None

    def generate(self, prompt, **kwargs):
        with self.lock:
            self.calls.append(prompt)
            n = len(self.calls)
            self.active += 1
            self.peak = max(self.peak, self.active)
            if self.cancel_after is not None and n >= self.cancel_after \
                    and self.progress is not None:
                self.progress.cancel()
        try:
            if self.latency:
                time.sleep(self.latency)
            return MagicMock(content=self.route(prompt))
        finally:
            with self.lock:
                self.active -= 1

    # override in tests
    def route(self, prompt):
        return "{}"


class ExtractJudgeLLM(CountingLLM):
    """Standard responses: N claims per section, supported judgements."""

    def __init__(self, claims_per_section=3, **kw):
        super().__init__(**kw)
        self.claims_per_section = claims_per_section

    def route(self, prompt):
        if "検証が必要な事実主張" in prompt:
            return j({"claims": [
                {"claim": f"主張{i}は事実である{i * 111}件",
                 "importance": "important", "source_numbers": [1]}
                for i in range(1, self.claims_per_section + 1)]})
        if "各主張を、それぞれに提示されたエビデンス" in prompt:
            # batch prompt: answer every claim id found in it
            import re
            ids = re.findall(r"主張 (C-[^\s（]+)", prompt)
            return j({"verdicts": [
                {"id": cid, "status": "supported", "reason": "ok",
                 "supporting_source_ids": self._ids(prompt)}
                for cid in ids]})
        if "次の主張を、提示されたエビデンス" in prompt:
            return j({"status": "supported", "reason": "ok",
                      "supporting_source_ids": self._ids(prompt)})
        if "実質的に回答しているか" in prompt:
            return j({"coverage": [{"question": i, "answered": True,
                                    "section_id": "1"}
                                   for i in range(10)]})
        return "{}"

    @staticmethod
    def _ids(prompt):
        import re
        tail = prompt.split("【エビデンス】")[-1]
        return re.findall(r"^\[([^\]]+)\]", tail, re.M)


def make_locker_evidence(n=3):
    evs = []
    for i in range(1, n + 1):
        ev = MagicMock()
        ev.id = f"EV-{i}"
        ev.url = f"https://e{i}.example.com/"
        ev.title = f"ソース{i}"
        ev.extracted_text = f"主張1は事実である111件。主張2は事実である222件。主張3は事実である333件。統計より。" * 4
        ev.content_excerpt = ev.extracted_text[:200]
        return_none = None
        evs.append(ev)
    return evs


def make_registry(evidence, sections=("1",)):
    mgr = CitationManager(evidence_ids_exist=lambda eid: True)
    for sid in sections:
        mgr.register_section(sid, [e.id for e in evidence])
    return mgr


SECTION_TEXT = ("## 1. 分析\n\n主張1は事実である111件 [SOURCE 1]。"
                "主張2は事実である222件 [SOURCE 1]。"
                "主張3は事実である333件 [SOURCE 1]。" + "背景説明。" * 60)


class TestProfileResolution(unittest.TestCase):

    def test_presets(self):
        fast = resolve_verification_settings("fast")
        self.assertEqual((fast.max_final_research_rounds,
                          fast.max_final_revision_rounds,
                          fast.max_no_improvement_rounds), (0, 0, 1))
        self.assertLess(fast.minor_claim_sample_rate, 1.0)
        self.assertTrue(fast.cache_enabled)

        balanced = resolve_verification_settings("balanced")
        self.assertEqual((balanced.max_final_research_rounds,
                          balanced.max_final_revision_rounds), (1, 1))
        self.assertEqual(balanced.minor_claim_sample_rate, 1.0)

        strict = resolve_verification_settings("strict")
        self.assertEqual((strict.max_final_research_rounds,
                          strict.max_final_revision_rounds), (2, 2))
        self.assertEqual(strict.min_claim_support_score, 0.85)
        self.assertEqual(strict.required_critical_coverage, 1.0)
        self.assertEqual(strict.minor_claim_sample_rate, 1.0)

    def test_default_profile_is_balanced(self):
        config = create_config(provider="openai", openai_api_key="sk-t")
        self.assertEqual(config.research.verification_profile, "balanced")
        settings = settings_from_research_config(config.research)
        self.assertEqual(settings.profile, "balanced")
        self.assertEqual(settings.max_final_research_rounds, 1)

    def test_legacy_explicit_rounds_still_win(self):
        config = create_config(provider="openai", openai_api_key="sk-t",
                               max_final_research_rounds=5)
        settings = settings_from_research_config(config.research)
        self.assertEqual(settings.max_final_research_rounds, 5)

    def test_strict_never_weaker_than_legacy(self):
        # attempts to weaken strict acceptance criteria are clamped UP
        settings = resolve_verification_settings(
            "strict", {"min_claim_support_score": 0.3,
                       "required_critical_coverage": 0.5,
                       "minor_claim_sample_rate": 0.1})
        self.assertGreaterEqual(settings.min_claim_support_score, 0.85)
        self.assertGreaterEqual(settings.required_critical_coverage, 1.0)
        self.assertEqual(settings.minor_claim_sample_rate, 1.0)

    def test_custom_validation(self):
        ok = resolve_verification_settings(
            "custom", {"max_final_research_rounds": 3,
                       "max_workers": 16, "batch_size": 32,
                       "minor_claim_sample_rate": 0.5,
                       "timeout_seconds": 600})
        self.assertEqual(ok.max_final_research_rounds, 3)
        for bad in (
            {"max_final_research_rounds": -1},
            {"max_final_research_rounds": 6},
            {"max_workers": 0},
            {"max_workers": 17},
            {"batch_size": 0},
            {"batch_size": 33},
            {"minor_claim_sample_rate": 1.5},
            {"timeout_seconds": -5},
            {"max_workers": True},
            {"max_workers": 2.5},
            {"unknown_setting": 1},
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                resolve_verification_settings("custom", bad)
        with self.assertRaises(ValueError):
            resolve_verification_settings("turbo")

    def test_invalid_custom_blocks_config(self):
        config = create_config(provider="openai", openai_api_key="sk-t",
                               verification_profile="custom",
                               verification_batch_size=99)
        errors = config.validate()
        self.assertTrue(any("batch_size" in e for e in errors))

    def test_webui_converters(self):
        from deep_research_tool.webui.server import build_config_kwargs
        kwargs = build_config_kwargs({"verification_profile": "Fast",
                                      "verification_batch_size": "12"})
        self.assertEqual(kwargs["verification_profile"], "fast")
        self.assertEqual(kwargs["verification_batch_size"], 12)
        for bad in ({"verification_profile": "turbo"},
                    {"verification_batch_size": "0"},
                    {"verification_max_workers": "17"},
                    {"verification_minor_claim_sample_rate": "1.5"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                build_config_kwargs(bad)

    def test_cli_profile_options(self):
        from click.testing import CliRunner
        from deep_research_tool.cli import cli
        res = CliRunner().invoke(cli, [
            "research", "テーマ", "--openai-key", "sk-t",
            "--verification-profile", "custom",
            "--max-final-research-rounds", "1",
            "--max-final-revision-rounds", "2",
            "--verification-max-workers", "8",
            "--verification-batch-size", "12"], input="n\n")
        self.assertEqual(res.exit_code, 0, res.output)
        self.assertIn("Cancelled", res.output)

    def test_custom_warning_text_exists_in_ui(self):
        html = (__import__("pathlib").Path(__file__).parent.parent
                / "webui" / "static" / "index.html").read_text(
                    encoding="utf-8")
        self.assertIn("非常に長い時間がかかる可能性", html)
        self.assertIn("verification_profile", html)
        self.assertIn("推奨", html)
        # backend shares the same warning text
        self.assertIn("非常に長い時間がかかる可能性", CUSTOM_WARNING)


def make_verifier(llm, workers=4, batch=8, cache=None, rate=1.0,
                  progress=None):
    settings = resolve_verification_settings(
        "custom", {"max_workers": workers, "batch_size": batch,
                   "minor_claim_sample_rate": rate})
    return ClaimVerifier(llm_client=llm, language="ja", settings=settings,
                         cache=cache or VerificationCache(enabled=True),
                         progress=progress)


class TestCacheAndDiff(unittest.TestCase):

    def _verify(self, verifier, chapters, evidence, mgr):
        return verifier.verify_report(chapters, evidence,
                                      citation_manager=mgr)

    def test_unchanged_section_not_reverified(self):
        llm = ExtractJudgeLLM()
        evidence = make_locker_evidence()
        mgr = make_registry(evidence)
        verifier = make_verifier(llm)
        chapters = {"1": SECTION_TEXT}

        self._verify(verifier, chapters, evidence, mgr)
        calls_first = len(llm.calls)
        self.assertGreater(calls_first, 0)

        # SECOND pass over identical content: extraction and judgements
        # replay from the cache — zero new extract/judge calls
        self._verify(verifier, chapters, evidence, mgr)
        new_calls = [p for p in llm.calls[calls_first:]
                     if "検証が必要な事実主張" in p
                     or "提示されたエビデンス" in p]
        self.assertEqual(new_calls, [])

    def test_partial_edit_reverifies_only_changed_section(self):
        llm = ExtractJudgeLLM()
        evidence = make_locker_evidence()
        mgr = make_registry(evidence, sections=("1", "2"))
        verifier = make_verifier(llm)
        chapters = {"1": SECTION_TEXT,
                    "2": SECTION_TEXT.replace("## 1. 分析", "## 2. 動向")}
        self._verify(verifier, chapters, evidence, mgr)
        n1 = len(llm.calls)

        # edit ONLY section 2
        chapters["2"] += "\n追記された新しい段落。"
        self._verify(verifier, chapters, evidence, mgr)
        extracts = [p for p in llm.calls[n1:] if "検証が必要な事実主張" in p]
        # only the edited section was re-extracted
        self.assertTrue(all("セクション2" in p for p in extracts))
        self.assertGreaterEqual(len(extracts), 1)

    def test_cache_key_includes_model_and_prompt_version(self):
        base = stable_hash("extract", "1", "text", "model-a", PROMPT_VERSION)
        other_model = stable_hash("extract", "1", "text", "model-b",
                                  PROMPT_VERSION)
        other_pv = stable_hash("extract", "1", "text", "model-a", "v999")
        self.assertNotEqual(base, other_model)
        self.assertNotEqual(base, other_pv)

        # end-to-end: switching the model misses the cache
        evidence = make_locker_evidence()
        mgr = make_registry(evidence)
        cache = VerificationCache(enabled=True)
        llm_a = ExtractJudgeLLM(model="model-a")
        make_verifier(llm_a, cache=cache).verify_report(
            {"1": SECTION_TEXT}, evidence, citation_manager=mgr)
        llm_b = ExtractJudgeLLM(model="model-b")
        make_verifier(llm_b, cache=cache).verify_report(
            {"1": SECTION_TEXT}, evidence, citation_manager=mgr)
        self.assertTrue(any("検証が必要な事実主張" in p for p in llm_b.calls))


class TestParallelSafety(unittest.TestCase):

    def test_claim_ids_stable_across_parallelism(self):
        evidence = make_locker_evidence()
        mgr = make_registry(evidence)
        chapters = {"1": SECTION_TEXT * 3}   # multiple extraction chunks

        def run(workers):
            llm = ExtractJudgeLLM(latency=0.01)
            verifier = make_verifier(llm, workers=workers,
                                     cache=VerificationCache(enabled=False))
            verdict = verifier.verify_report(chapters, evidence,
                                             citation_manager=mgr)
            return verdict

        seq = run(1)
        par = run(8)
        # deterministic ids and metrics regardless of completion order
        self.assertEqual(seq.metrics.claims_total, par.metrics.claims_total)
        self.assertEqual(
            sorted(i.claim_id for i in seq.issues),
            sorted(i.claim_id for i in par.issues))

    def test_llm_concurrency_never_exceeds_worker_limit(self):
        llm = ExtractJudgeLLM(latency=0.03)
        evidence = make_locker_evidence()
        mgr = make_registry(evidence, sections=tuple(str(i) for i in range(1, 7)))
        chapters = {str(i): SECTION_TEXT.replace("## 1", f"## {i}")
                    for i in range(1, 7)}
        verifier = make_verifier(llm, workers=3, batch=1,
                                 cache=VerificationCache(enabled=False))
        verifier.verify_report(chapters, evidence, citation_manager=mgr)
        self.assertLessEqual(llm.peak, 3)


class TestBatching(unittest.TestCase):

    def test_partial_batch_response_retries_only_missing(self):
        class PartialBatchLLM(ExtractJudgeLLM):
            def route(self, prompt):
                if "各主張を、それぞれに提示されたエビデンス" in prompt:
                    import re
                    ids = re.findall(r"主張 (C-[^\s（]+)", prompt)
                    # DROP the last claim's verdict from the batch
                    return j({"verdicts": [
                        {"id": cid, "status": "supported", "reason": "ok",
                         "supporting_source_ids": self._ids(prompt)}
                        for cid in ids[:-1]]})
                return super().route(prompt)

        llm = PartialBatchLLM()
        evidence = make_locker_evidence()
        mgr = make_registry(evidence)
        verifier = make_verifier(llm, workers=1, batch=8)
        verdict = verifier.verify_report({"1": SECTION_TEXT}, evidence,
                                         citation_manager=mgr)
        batch_calls = [p for p in llm.calls
                       if "各主張を、それぞれに提示されたエビデンス" in p]
        single_calls = [p for p in llm.calls
                        if "次の主張を、提示されたエビデンス" in p]
        self.assertEqual(len(batch_calls), 1)
        self.assertEqual(len(single_calls), 1)     # ONLY the missing one
        self.assertEqual(verdict.metrics.unsupported_count, 0)

    def test_unparseable_verdict_is_never_supported(self):
        class GarbageLLM(ExtractJudgeLLM):
            def route(self, prompt):
                if "提示されたエビデンス" in prompt:
                    return "not json at all"
                return super().route(prompt)

        llm = GarbageLLM()
        evidence = make_locker_evidence()
        mgr = make_registry(evidence)
        verifier = make_verifier(llm, workers=1, batch=4)
        verdict = verifier.verify_report({"1": SECTION_TEXT}, evidence,
                                         citation_manager=mgr)
        # fail-safe: nothing silently passes
        self.assertLess(verdict.metrics.claim_support_score, 1.0)
        self.assertGreater(verdict.metrics.uncertain_count
                           + verdict.metrics.unsupported_count, 0)


class TestCancelAndSampling(unittest.TestCase):

    def test_no_new_llm_calls_after_cancel(self):
        llm = ExtractJudgeLLM(latency=0.01)
        progress = VerificationProgress()
        progress.start("balanced", max_rounds=4)
        llm.progress = progress
        llm.cancel_after = 3          # cancel is requested mid-run

        evidence = make_locker_evidence()
        mgr = make_registry(evidence, sections=tuple(str(i) for i in range(1, 9)))
        chapters = {str(i): SECTION_TEXT.replace("## 1", f"## {i}")
                    for i in range(1, 9)}
        verifier = make_verifier(llm, workers=1, batch=1, progress=progress)
        with self.assertRaises(VerificationCancelled):
            verifier.verify_report(chapters, evidence, citation_manager=mgr)
        calls_at_cancel = len(llm.calls)
        time.sleep(0.05)
        # not a single NEW request after the cancellation boundary
        self.assertEqual(len(llm.calls), calls_at_cancel)
        self.assertLessEqual(calls_at_cancel, llm.cancel_after + 1)

    def test_fast_mode_never_skips_critical_or_important(self):
        class MixedLLM(ExtractJudgeLLM):
            def route(self, prompt):
                if "検証が必要な事実主張" in prompt:
                    return j({"claims": [
                        {"claim": "重大な主張である", "importance": "critical",
                         "source_numbers": [1]},
                        {"claim": "重要な主張である", "importance": "important",
                         "source_numbers": [1]},
                    ] + [{"claim": f"軽微な話{i}である",
                          "importance": "minor", "source_numbers": [1]}
                         for i in range(12)]})
                return super().route(prompt)

        llm = MixedLLM()
        evidence = make_locker_evidence()
        mgr = make_registry(evidence)
        verifier = make_verifier(llm, rate=0.3)     # fast-style sampling
        # every extracted claim exists VERBATIM in the body so the
        # deterministic citation parser can associate it ([SOURCE 1])
        body = (SECTION_TEXT
                + "重大な主張である [SOURCE 1]。重要な主張である [SOURCE 1]。"
                + "".join(f"軽微な話{i}である [SOURCE 1]。"
                          for i in range(12)))
        verdict = verifier.verify_report({"1": body}, evidence,
                                         citation_manager=mgr)
        self.assertGreater(verdict.metrics.skipped_minor_claims, 0)
        # every skipped claim is MINOR; critical/important all verified
        self.assertLessEqual(verdict.metrics.skipped_minor_claims, 12)
        self.assertEqual(verdict.metrics.unsupported_critical_claims, 0)

    def test_progress_snapshot_contains_no_secrets(self):
        progress = VerificationProgress()
        progress.start("balanced", max_rounds=3)
        progress.set_label("セクション1のバッチ2")
        snap = progress.snapshot()
        text = json.dumps(snap, ensure_ascii=False)
        # metrics only: no api keys, no prompt BODIES
        self.assertNotIn("sk-", text)
        self.assertNotIn("api_key", text)
        self.assertNotIn("Bearer", text)
        self.assertNotIn("検証してください", text)   # prompt wording
        # every value is a number/bool/short label — nothing resembling
        # a prompt or key can fit
        for value in snap.values():
            if isinstance(value, str):
                self.assertLess(len(value), 120)
        for key in ("phase", "claims_done", "llm_calls", "cache_hit_rate",
                    "elapsed_seconds", "eta_seconds"):
            self.assertIn(key, snap)


if __name__ == "__main__":
    unittest.main()
