"""
One overall progress scale for a run.

Every stage (plan, research, DeepThink, figures, writing, verification,
rendering …) reports its OWN 0–100; the plan maps that onto a slice of
the overall bar whose width depends on which stages are enabled, so the
bar reaches 100 only when the run is complete. Before this, the research
loop reported "Research completed!" at 100 and the multilingual search
reported its own 100 — the bar was full hours before the report existed.

The Web UI job keeps the overall value monotonic; the plan itself also
never reports a lower value than the last one.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ProgressCallback = Optional[Callable[[str, float], None]]

STAGE_LABELS = {
    "plan": "計画",
    "research": "調査・収集",
    "deep_think": "DeepThink",
    "evidence_review": "根拠の見直し",
    "fermi": "フェルミ推定",
    "figures": "図表生成",
    "writing": "レポート生成",
    "verification": "検証",
    "rendering": "出力",
    "done": "完了",
}

# which of the Web UI's four pipeline steps a stage belongs to
# (計画 / 調査・収集 / 統合・評価 / レポート生成)
STAGE_STEP = {
    "plan": 0, "research": 1,
    "deep_think": 2, "evidence_review": 2, "fermi": 2, "figures": 2,
    "writing": 3, "verification": 3, "rendering": 3, "done": 3,
}


class ProgressPlan:
    """Stage slices of one 0–100 bar, decided when the run starts."""

    def __init__(self, stages: Sequence[Tuple[str, float]]):
        if not stages:
            raise ValueError("a progress plan needs at least one stage")
        total = float(sum(max(0.0, float(w)) for _, w in stages)) or 1.0
        self.order: List[str] = []
        self.ranges: Dict[str, Tuple[float, float]] = {}
        pos = 0.0
        for key, weight in stages:
            width = max(0.0, float(weight)) / total * 100.0
            self.order.append(key)
            self.ranges[key] = (pos, pos + width)
            pos += width
        # rounding: the last stage ends exactly at 100
        last = self.order[-1]
        self.ranges[last] = (self.ranges[last][0], 100.0)
        self._lock = threading.Lock()
        self.current: str = self.order[0]
        self.last: float = 0.0          # highest overall value reported

    # ------------------------------------------------------------------
    @classmethod
    def for_config(cls, config: Any) -> "ProgressPlan":
        """The stage list in the order DeepResearchTool.run executes them
        (V1 verifies the session sections BEFORE it writes the report;
        V2/V3 write the chapters, then verify, then render)."""
        rep = getattr(config, "report", None)
        version = getattr(getattr(rep, "generator_version", None), "value", "v1")
        v1 = str(version).lower() == "v1"
        stages: List[Tuple[str, float]] = [("plan", 4.0), ("research", 50.0)]
        if getattr(getattr(config, "deep_think", None), "enabled", False):
            stages.append(("deep_think", 6.0))
        if v1:
            stages.append(("evidence_review", 3.0))
        if getattr(getattr(config, "fermi_estimation", None), "enabled", False):
            stages.append(("fermi", 3.0))
        if getattr(rep, "auto_figures", False):
            stages.append(("figures", 4.0))
        verify = bool(getattr(config, "enable_verification", False))
        if v1:
            if verify:
                stages.append(("verification", 20.0))
            stages.append(("writing", 6.0))
        else:
            stages.append(("writing", 10.0))
            if verify:
                stages.append(("verification", 20.0))
        stages.append(("rendering", 3.0))
        return cls(stages)

    # ------------------------------------------------------------------
    def enter(self, key: str) -> None:
        """Switch to a stage (unknown stages — a feature that is off —
        are ignored); the bar never goes back below the stage start."""
        if key not in self.ranges:
            return
        with self._lock:
            self.current = key
            self.last = max(self.last, self.ranges[key][0])

    def overall(self, pct: float, stage: Optional[str] = None) -> float:
        """Stage-relative 0–100 → overall 0–100 (never below ``last``)."""
        key = stage if stage in self.ranges else self.current
        start, end = self.ranges[key]
        frac = max(0.0, min(100.0, float(pct))) / 100.0
        # 100 itself is reserved for done(): the run is not complete
        # just because its last stage reported 100
        value = min(start + frac * (end - start), 99.9)
        with self._lock:
            self.last = max(self.last, value)
            return self.last

    def callback(self, cb: ProgressCallback,
                 stage: Optional[str] = None) -> Callable[[str, Any], None]:
        """A (message, stage_pct) callback that reports overall values.
        Negative percentages (error markers) pass through unchanged;
        ``None`` keeps the current value (a note)."""
        def _report(message: str, pct: Any) -> None:
            if pct is None:
                value = self.last
            elif float(pct) < 0:
                value = float(pct)
            else:
                value = self.overall(pct, stage)
            if cb:
                cb(message, value)
        return _report

    def research_callback(self, cb: ProgressCallback) -> Callable[[str, Any], None]:
        """The researcher reports 0–10 for planning and 10–100 for the
        research loop; split those onto the plan / research slices. The
        first section starts at exactly 10, so a 10 switches to research
        only when it announces a section."""
        def _report(message: str, pct: Any) -> None:
            if pct is None:
                value = self.last
            elif float(pct) < 0:
                value = float(pct)
            else:
                p = float(pct)
                section = str(message).startswith(("Researching", "Reusing completed section"))
                # once research started it never flips back to planning (a
                # crawler's first report of the first section is exactly 10)
                if "plan" in self.ranges and self.current == "plan" \
                        and (p < 10.0 or (p == 10.0 and not section)):
                    value = self.overall(p * 10.0, "plan")
                else:
                    self.enter("research")
                    value = self.overall(max(0.0, p - 10.0) / 90.0 * 100.0, "research")
            if cb:
                cb(message, value)
        return _report

    def note(self, cb: ProgressCallback, message: str) -> None:
        if cb:
            cb(message, self.last)

    def done(self, cb: ProgressCallback, message: str = "Completed") -> None:
        with self._lock:
            self.current = "done"
            self.last = 100.0
        if cb:
            cb(message, 100.0)

    # ------------------------------------------------------------------
    @staticmethod
    def verification_fraction(snapshot: Optional[Dict[str, Any]]) -> float:
        """0–1 progress of the verification stage from the live
        VerificationProgress snapshot (rounds × phases × counters)."""
        if not snapshot:
            return 0.0
        phase = snapshot.get("phase") or "idle"
        if phase == "idle":
            return 0.0
        if phase in ("done", "cancelled", "error"):
            return 1.0
        rounds = max(1, int(snapshot.get("max_rounds") or 1))
        rnd = max(1, int(snapshot.get("round") or 1))

        def _ratio(done_key, total_key):
            total = snapshot.get(total_key) or 0
            return (min(snapshot.get(done_key) or 0, total) / total) if total else 0.0

        if phase == "extracting":
            within = 0.15 * _ratio("chunks_done", "chunks_total")
        elif phase == "judging":
            within = 0.15 + 0.75 * _ratio("claims_done", "claims_total")
        elif phase == "coverage":
            within = 0.95
        else:
            within = 0.5
        return max(0.0, min(0.99, ((rnd - 1) + within) / rounds))

    def live(self, verification_snapshot: Optional[Dict[str, Any]] = None) -> float:
        """Current overall value, moved by the live verification counters
        while the verification stage runs (it reports no messages of its
        own while judging hundreds of claims)."""
        with self._lock:
            current, last = self.current, self.last
        if current == "verification" and verification_snapshot:
            start, end = self.ranges["verification"]
            value = start + self.verification_fraction(verification_snapshot) * (end - start)
            value = max(last, min(value, end, 99.9))
            # a new verification round resets the counters: the bar keeps
            # the highest value it reached
            with self._lock:
                self.last = max(self.last, value)
                return self.last
        return last

    def snapshot(self, verification_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            current = self.current
        start, end = self.ranges.get(current, (100.0, 100.0))
        return {
            "overall": round(self.live(verification_snapshot), 1),
            "stage": {
                "key": current,
                "label": STAGE_LABELS.get(current, current),
                "step": STAGE_STEP.get(current, 3),
                "start": round(start, 1),
                "end": round(end, 1),
            },
            "stages": [
                {"key": k, "label": STAGE_LABELS.get(k, k),
                 "start": round(self.ranges[k][0], 1),
                 "end": round(self.ranges[k][1], 1)}
                for k in self.order
            ],
        }
