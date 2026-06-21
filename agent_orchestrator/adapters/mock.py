"""A deterministic, offline adapter.

The mock adapter lets the whole system run — UI, streaming, every collaboration
strategy — with no API keys, no network, and no external CLIs installed. It is
also what the test-suite runs against. Responses are role-aware so a transcript
reads like a plausible collaboration.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

from .base import AgentAdapter, Message


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


class MockAdapter(AgentAdapter):
    """Generates canned but context-aware text based on the agent's persona."""

    kind = "mock"

    def __init__(self, name: str = "mock", display_name: Optional[str] = None,
                 persona: str = "assistant"):
        super().__init__(name, display_name)
        self.persona = persona

    def _generate(self, prompt: str, system: Optional[str], history: List[Message]) -> str:
        persona = (system or self.persona or "").lower()
        tag = _short_hash(prompt)
        # Pull the task line out of the prompt for a touch of realism.
        lines = [ln.strip() for ln in prompt.splitlines() if ln.strip()]
        snippet = next((ln for ln in lines if not ln.lower().startswith("task")), lines[0] if lines else "the task")
        snippet = snippet[:80]

        # Detect the role from the distinctive role noun. Order matters: some
        # prompts mention another role in passing (a reviewer reviews the
        # *implementer's* work, an executor runs the *planner's* plan), so the
        # more-specific noun is checked first.
        if "synthesizer" in persona:
            return (
                f"Synthesis ({tag}): Weighing both arguments, the pragmatic option is "
                f"best supported here. It satisfies the requirement with less complexity; "
                f"the alternative's flexibility isn't yet justified. Decisive trade-off: "
                f"maintainability now over speculative generality."
            )
        if "reviewer" in persona:
            return (
                f"Review ({tag}): The approach is sound. Findings:\n"
                f"1. Add input validation at the boundary.\n"
                f"2. Cover the empty-input edge case with a test.\n"
                f"3. Naming is clear; no blocking issues.\n"
                f"Verdict: APPROVE once the two nits above are addressed."
            )
        if "executor" in persona:
            return (
                f"Execution ({tag}): completed the planned steps.\n"
                f"```python\n"
                f"def solve(data):\n"
                f"    # implements the planned behaviour\n"
                f"    return sorted(data)\n"
                f"```\n"
                f"All planned steps done; ready for review."
            )
        if "planner" in persona:
            return (
                f"Plan ({tag}) for: {snippet}\n"
                f"Step 1. Clarify inputs/outputs and constraints.\n"
                f"Step 2. Implement the core function.\n"
                f"Step 3. Add tests for the happy path and one edge case.\n"
                f"Step 4. Wire it up and document usage."
            )
        if "debater" in persona:
            return (
                f"Position ({tag}): For \"{snippet}\", I argue the pragmatic option. "
                f"It is simpler to maintain and meets the stated requirement without "
                f"speculative complexity. I acknowledge the alternative's flexibility "
                f"but contend it is premature here."
            )
        # Default: an "implementer" style answer.
        return (
            f"Implementation ({tag}) for: {snippet}\n"
            f"```python\n"
            f"def solve(data):\n"
            f"    \"\"\"A first cut at the requested behaviour.\"\"\"\n"
            f"    return sorted(data)\n"
            f"```\n"
            f"This handles the main case; happy to refine after review.\n"
            f"NOTE: ({tag}) assume input is a list of comparable items."
        )
