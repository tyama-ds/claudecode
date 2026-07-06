"""
Plan review - let the user inspect / revise the research plan before the
research loop starts.

The Researcher accepts a ``plan_review_callback(plan, revise_fn)``:

- ``plan``: the generated ResearchPlan
- ``revise_fn(current_plan, instructions) -> ResearchPlan``: revises the
  plan with an LLM according to natural-language instructions

The callback returns a (possibly revised) plan to use, or ``None`` to keep
the original. This module provides the console implementation with a
timeout: when the user does not respond within ``timeout`` seconds, the
research starts with the plan as-is.
"""

import queue
import sys
import threading
from typing import Optional


def _is_interactive() -> bool:
    """Whether we can realistically expect keyboard input."""
    if "ipykernel" in sys.modules:  # Jupyter: input() works via the kernel
        return True
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except Exception:
        return False


def _input_with_timeout(prompt: str, timeout: float) -> Optional[str]:
    """input() with a timeout. Returns None on timeout or closed stdin."""
    q: "queue.Queue[Optional[str]]" = queue.Queue()

    def _reader():
        try:
            q.put(input(prompt))
        except (EOFError, OSError):
            q.put(None)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


def format_plan_summary(plan, max_queries: int = 15, language: str = "ja") -> str:
    """Human-readable summary of a ResearchPlan for review."""
    lines = []
    if language == "ja":
        lines.append("=" * 60)
        lines.append(f"【調査計画】{plan.title}")
        if plan.summary:
            lines.append(f"概要: {plan.summary}")
        lines.append("-" * 60)
        lines.append("目次:")
    else:
        lines.append("=" * 60)
        lines.append(f"[RESEARCH PLAN] {plan.title}")
        if plan.summary:
            lines.append(f"Summary: {plan.summary}")
        lines.append("-" * 60)
        lines.append("Table of contents:")

    for item in plan.table_of_contents.items:
        lines.append(f"  {item.section}. {item.title}")
        if item.description:
            lines.append(f"      {item.description}")
        for sub in item.subsections:
            lines.append(f"      {sub.section} {sub.title}")

    lines.append("-" * 60)
    n = len(plan.search_queries)
    if language == "ja":
        lines.append(f"検索クエリ（全{n}件、先頭{min(n, max_queries)}件を表示）:")
    else:
        lines.append(f"Search queries ({n} total, showing {min(n, max_queries)}):")
    for q_ in plan.search_queries[:max_queries]:
        lines.append(f"  - {q_}")
    if n > max_queries:
        lines.append(f"  ... (+{n - max_queries})")
    lines.append("=" * 60)
    return "\n".join(lines)


def make_console_plan_review_callback(
    timeout: int = 60,
    language: str = "ja",
    max_revision_rounds: int = 5,
):
    """
    Build a console plan-review callback.

    Shows the generated plan and waits up to ``timeout`` seconds for
    revision instructions. Empty input or timeout starts the research
    with the current plan. Typed instructions are applied by the LLM and
    the revised plan is shown again (up to ``max_revision_rounds`` times).

    In non-interactive sessions (no TTY, not a notebook) the review is
    skipped so batch runs never stall.
    """

    def callback(plan, revise_fn):
        if not _is_interactive():
            print("[PlanReview] Non-interactive session; starting research "
                  "with the generated plan")
            return None

        current = plan
        changed = False
        for _ in range(max_revision_rounds):
            print(format_plan_summary(current, language=language))
            if language == "ja":
                print(f"[PlanReview] この計画を修正する場合は指示を入力してください"
                      f"（例: 「3章を価格動向の章に差し替えて」）。")
                print(f"[PlanReview] {timeout}秒以内に入力がなければ、"
                      f"このまま調査を開始します（Enterのみでも開始）。")
            else:
                print(f"[PlanReview] Type revision instructions to change this plan.")
                print(f"[PlanReview] Research starts automatically in {timeout}s "
                      f"without input (or press Enter to start now).")

            text = _input_with_timeout("> ", timeout)

            if text is None:
                print("\n[PlanReview] " + (
                    "応答がないため、この計画で調査を開始します"
                    if language == "ja" else
                    "No response; starting research with this plan"))
                return current if changed else None
            if not text.strip():
                print("[PlanReview] " + (
                    "この計画で調査を開始します" if language == "ja"
                    else "Starting research with this plan"))
                return current if changed else None

            try:
                print("[PlanReview] " + (
                    "計画を修正しています..." if language == "ja"
                    else "Revising the plan..."))
                current = revise_fn(current, text.strip())
                changed = True
            except Exception as e:
                print(f"[PlanReview] " + (
                    f"修正に失敗しました（{e}）。現在の計画で調査を開始します"
                    if language == "ja" else
                    f"Revision failed ({e}); starting with the current plan"))
                return current if changed else None

        print("[PlanReview] " + (
            "修正回数の上限に達したため、この計画で調査を開始します"
            if language == "ja" else
            "Revision round limit reached; starting research"))
        return current if changed else None

    return callback
