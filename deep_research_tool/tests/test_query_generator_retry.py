"""
Regression tests for QueryGenerator.create_research_plan retry loop.

Guards against UnboundLocalError on `issues` when the first plan
generation attempt returns None and the loop retries (see commit 3f26429).
"""

import pytest
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deep_research_tool.research.query_generator import (
    QueryGenerator,
    ResearchPlan,
    TableOfContents,
    TableOfContentsItem,
)


def _make_plan(title: str = "Test Plan") -> ResearchPlan:
    """Build a minimal valid ResearchPlan for mocking."""
    toc = TableOfContents(
        title=title,
        items=[
            TableOfContentsItem("1", "Section One", "First section"),
            TableOfContentsItem("2", "Section Two", "Second section"),
        ],
    )
    return ResearchPlan(
        title=title,
        summary="Test summary",
        table_of_contents=toc,
        search_queries=["query one", "query two"],
    )


@pytest.fixture
def generator():
    """QueryGenerator with a dummy LLM client (never called directly)."""
    return QueryGenerator(llm_client=object(), language="en")


class TestCreateResearchPlanRetry:
    """Retry-loop behavior of create_research_plan."""

    def test_first_attempt_none_then_success_no_unbound_error(self, generator):
        """First attempt fails (None), retry succeeds.

        Before the fix, referencing `issues` on attempt 1 raised
        UnboundLocalError because it was only assigned after the
        `if plan is None: continue` guard.
        """
        plan = _make_plan()

        with patch.object(
            generator,
            "_generate_research_plan_attempt",
            side_effect=[None, plan],
        ), patch.object(
            generator,
            "_validate_toc_quality",
            return_value=(True, []),
        ):
            result = generator.create_research_plan("test topic")

        assert result is plan

    def test_all_attempts_none_returns_fallback_plan(self, generator):
        """All attempts fail -> fallback plan, not an exception."""
        with patch.object(
            generator,
            "_generate_research_plan_attempt",
            return_value=None,
        ):
            result = generator.create_research_plan("test topic", max_retries=2)

        assert isinstance(result, ResearchPlan)
        assert "test topic" in result.title

    def test_validation_failure_passes_issues_to_retry(self, generator):
        """Issues from a failed validation are forwarded to the retry attempt."""
        plan = _make_plan()
        seen_issues = []

        def fake_attempt(*args, **kwargs):
            seen_issues.append(list(kwargs.get("previous_issues") or []))
            return plan

        with patch.object(
            generator,
            "_generate_research_plan_attempt",
            side_effect=fake_attempt,
        ), patch.object(
            generator,
            "_validate_toc_quality",
            side_effect=[(False, ["too generic"]), (True, [])],
        ):
            result = generator.create_research_plan("test topic")

        assert result is plan
        assert seen_issues[0] == []
        assert seen_issues[1] == ["too generic"]
