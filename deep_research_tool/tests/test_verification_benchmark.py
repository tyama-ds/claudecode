"""
FakeLLM benchmark for the fast verification engine.

Dataset (per the specification):
- ~40,000 chars over 10 sections
- ~90 claims (9 per section)
- 50 evidence items
- 200 ms simulated LLM latency
- 20% citation mismatches (cited evidence does not support the claim;
  an uncited source does -> replacement candidate)
- one section edited, then re-verified (differential verification)

Asserts the acceptance criterion: wall time is at least 70% below the
serial baseline (analytically: the SAME number of naive LLM calls run
strictly one-by-one at the same latency), and that the differential
second pass costs a small fraction of the first.
"""

import json
import re
import threading
import time
import unittest
from unittest.mock import MagicMock

from deep_research_tool.report.citations import CitationManager
from deep_research_tool.verification.claim_verifier import ClaimVerifier
from deep_research_tool.verification.profiles import (
    resolve_verification_settings,
)
from deep_research_tool.verification.runtime import VerificationCache

LATENCY = 0.2
N_SECTIONS = 10
CLAIMS_PER_SECTION = 9              # 90 claims
N_EVIDENCE = 50
MISMATCH_EVERY = 5                  # 20% of claims


def j(obj):
    return json.dumps(obj, ensure_ascii=False)


class BenchLLM:
    """200ms-latency fake LLM with deterministic, content-routed output."""

    model = "bench-model"

    def __init__(self):
        self.lock = threading.Lock()
        self.n_calls = 0

    def generate(self, prompt, **kwargs):
        with self.lock:
            self.n_calls += 1
        time.sleep(LATENCY)
        return MagicMock(content=self.route(prompt))

    @staticmethod
    def _claim_index(text):
        m = re.search(r"ベンチ主張(\d+)-(\d+)", text)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    def _verdict_for(self, claim_text, offered_ids):
        sec, idx = self._claim_index(claim_text)
        global_idx = (sec - 1) * CLAIMS_PER_SECTION + idx
        mismatch = global_idx % MISMATCH_EVERY == 0
        cited_pass = f"EV-{sec}" in offered_ids
        if mismatch and cited_pass:
            # the CITED evidence does not support this claim
            return {"status": "unsupported", "reason": "無関係",
                    "supporting_source_ids": []}
        return {"status": "supported", "reason": "ok",
                "supporting_source_ids": offered_ids}

    def route(self, prompt):
        if "検証が必要な事実主張" in prompt:
            m = re.search(r"セクション(\d+)", prompt)
            sec = int(m.group(1)) if m else 1
            # claims live in the FIRST chunk of each section
            part = re.search(r"（(\d+)/\d+分割）", prompt)
            if part and int(part.group(1)) > 1:
                return j({"claims": []})
            return j({"claims": [
                {"claim": f"ベンチ主張{sec}-{i}は{sec * 100 + i}億円である",
                 "importance": ["critical", "important", "minor"][i % 3],
                 "source_numbers": [1]}
                for i in range(1, CLAIMS_PER_SECTION + 1)]})
        if "各主張を、それぞれに提示されたエビデンス" in prompt:
            verdicts = []
            for block in prompt.split("◆ 主張 ")[1:]:
                cid = block.split("（")[0].strip()
                ids = re.findall(r"^\[([^\]]+)\]", block, re.M)
                v = self._verdict_for(block, ids)
                v["id"] = cid
                verdicts.append(v)
            return j({"verdicts": verdicts})
        if "次の主張を、提示されたエビデンス" in prompt:
            ids = re.findall(r"^\[([^\]]+)\]",
                             prompt.split("【エビデンス】")[-1], re.M)
            return j(self._verdict_for(prompt, ids))
        if "実質的に回答しているか" in prompt:
            return j({"coverage": [{"question": i, "answered": True,
                                    "section_id": "1"}
                                   for i in range(10)]})
        return "{}"


def build_dataset():
    evidence = []
    for i in range(1, N_EVIDENCE + 1):
        ev = MagicMock()
        ev.id = f"EV-{i}"
        ev.url = f"https://bench{i}.example.com/"
        ev.title = f"ベンチソース{i}"
        sec = ((i - 1) % N_SECTIONS) + 1
        ev.extracted_text = "。".join(
            f"ベンチ主張{sec}-{k}は{sec * 100 + k}億円である"
            for k in range(1, CLAIMS_PER_SECTION + 1)) + ("。統計解説。" * 30)
        ev.content_excerpt = ev.extracted_text[:200]
        evidence.append(ev)

    chapters = {}
    for sec in range(1, N_SECTIONS + 1):
        body = f"## {sec}. セクション{sec}\n\n"
        for k in range(1, CLAIMS_PER_SECTION + 1):
            body += (f"ベンチ主張{sec}-{k}は{sec * 100 + k}億円である "
                     f"[SOURCE 1]。" + "背景の説明が続く。" * 46)
        chapters[str(sec)] = body            # ~4,000 chars/section

    mgr = CitationManager(evidence_ids_exist=lambda eid: True)
    for sec in range(1, N_SECTIONS + 1):
        # [SOURCE 1] of section N -> EV-N; the rest are uncited pool
        ordered = [f"EV-{sec}"] + [e.id for e in evidence
                                   if e.id != f"EV-{sec}"][:4]
        mgr.register_section(str(sec), ordered)
    return chapters, evidence, mgr


class TestVerificationBenchmark(unittest.TestCase):

    def test_benchmark_70_percent_reduction_and_differential_pass(self):
        chapters, evidence, mgr = build_dataset()
        total_chars = sum(len(t) for t in chapters.values())
        self.assertGreater(total_chars, 35000)

        llm = BenchLLM()
        settings = resolve_verification_settings(
            "custom", {"max_workers": 8, "batch_size": 12})
        cache = VerificationCache(enabled=True)
        verifier = ClaimVerifier(llm_client=llm, language="ja",
                                 settings=settings, cache=cache)

        # ---- pass 1: full verification -------------------------------
        t0 = time.time()
        verdict = verifier.verify_report(chapters, evidence,
                                         citation_manager=mgr)
        wall_1 = time.time() - t0
        calls_1 = llm.n_calls

        self.assertGreaterEqual(verdict.metrics.claims_total, 80)
        self.assertLessEqual(verdict.metrics.claims_total, 100)
        # 20% mismatches were detected (not silently passed)
        mismatched = [i for i in verdict.issues
                      if i.type in ("invalid_citation", "unsupported_claim")]
        self.assertGreaterEqual(len(mismatched), 10)

        # serial baseline: the naive engine runs ONE LLM call at a time —
        # same work, latency-bound. (measured-call analytic baseline)
        serial_baseline = calls_1 * LATENCY
        reduction = 1 - (wall_1 / serial_baseline)
        print(f"\n[BENCH] pass1: wall={wall_1:.2f}s calls={calls_1} "
              f"serial-baseline={serial_baseline:.2f}s "
              f"reduction={reduction:.0%}")
        self.assertGreaterEqual(
            reduction, 0.70,
            f"acceptance requires >=70% reduction, got {reduction:.0%}")

        # ---- pass 2: ONE section edited (differential verification) ---
        chapters["7"] += "\n新しく追記された分析の段落。"
        t0 = time.time()
        verifier.verify_report(chapters, evidence, citation_manager=mgr)
        wall_2 = time.time() - t0
        calls_2 = llm.n_calls - calls_1
        print(f"[BENCH] pass2 (1 section edited): wall={wall_2:.2f}s "
              f"calls={calls_2} cache_hit_rate={cache.hit_rate:.0%}")

        # the second pass re-verifies ONLY the changed section's claims
        self.assertLess(calls_2, calls_1 * 0.35)
        self.assertLess(wall_2, wall_1 * 0.6)
        self.assertGreater(cache.hit_rate, 0.3)


if __name__ == "__main__":
    unittest.main()
