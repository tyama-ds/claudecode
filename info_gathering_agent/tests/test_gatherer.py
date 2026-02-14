"""
Tests for the Gatherer module.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime


class TestGatheringState:
    """Test GatheringState enum."""

    def test_state_values(self):
        from info_gathering_agent.research.gatherer import GatheringState

        assert GatheringState.INITIALIZED.value == "initialized"
        assert GatheringState.PLANNING.value == "planning"
        assert GatheringState.GATHERING.value == "gathering"
        assert GatheringState.SYNTHESIZING.value == "synthesizing"
        assert GatheringState.COMPLETED.value == "completed"
        assert GatheringState.ERROR.value == "error"

    def test_no_verifying_state(self):
        """Verify that VERIFYING state does NOT exist."""
        from info_gathering_agent.research.gatherer import GatheringState

        state_values = [s.value for s in GatheringState]
        assert "verifying" not in state_values

    def test_state_count(self):
        from info_gathering_agent.research.gatherer import GatheringState

        assert len(GatheringState) == 6


class TestGatheringIteration:
    """Test GatheringIteration dataclass."""

    def test_creation(self):
        from info_gathering_agent.research.gatherer import GatheringIteration

        iteration = GatheringIteration(
            iteration_number=1,
            section="1",
        )
        assert iteration.iteration_number == 1
        assert iteration.section == "1"
        assert iteration.queries_executed == []
        assert iteration.sources_found == 0
        assert iteration.content_extracted == 0

    def test_to_dict(self):
        from info_gathering_agent.research.gatherer import GatheringIteration

        iteration = GatheringIteration(
            iteration_number=2,
            section="1.1",
            queries_executed=["query1", "query2"],
            sources_found=5,
            content_extracted=3,
        )
        d = iteration.to_dict()
        assert d["iteration_number"] == 2
        assert d["section"] == "1.1"
        assert len(d["queries_executed"]) == 2
        assert d["sources_found"] == 5
        assert d["content_extracted"] == 3


class TestGatheringSession:
    """Test GatheringSession dataclass."""

    def test_creation(self):
        from info_gathering_agent.research.gatherer import GatheringSession, GatheringState

        session = GatheringSession(query="test query")
        assert session.query == "test query"
        assert session.state == GatheringState.INITIALIZED
        assert session.session_id is not None
        assert len(session.session_id) == 8

    def test_to_dict(self):
        from info_gathering_agent.research.gatherer import GatheringSession, GatheringState

        session = GatheringSession(
            query="test",
            requirements="req",
            state=GatheringState.COMPLETED,
        )
        d = session.to_dict()
        assert d["query"] == "test"
        assert d["requirements"] == "req"
        assert d["state"] == "completed"
        assert d["session_id"] is not None

    def test_save_and_load(self, tmp_path):
        from info_gathering_agent.research.gatherer import (
            GatheringSession,
            GatheringState,
            GatheringIteration,
        )

        session = GatheringSession(
            query="save test",
            requirements="some requirements",
            state=GatheringState.COMPLETED,
        )
        session.iterations.append(
            GatheringIteration(
                iteration_number=1,
                section="1",
                queries_executed=["q1"],
                sources_found=3,
                content_extracted=2,
            )
        )
        session.section_contents = {
            "1": {"title": "Test Section", "content": "Some content"},
        }

        filepath = tmp_path / "test_session.json"
        session.save(filepath)

        # Verify file exists and is valid JSON
        assert filepath.exists()
        with open(filepath) as f:
            data = json.load(f)
        assert data["query"] == "save test"

        # Load and verify
        loaded = GatheringSession.load(filepath)
        assert loaded.query == "save test"
        assert loaded.state == GatheringState.COMPLETED
        assert len(loaded.iterations) == 1
        assert loaded.iterations[0].section == "1"
        assert "1" in loaded.section_contents


class TestGatherer:
    """Test Gatherer class initialization."""

    def _make_mock_clients(self):
        """Create mock LLM and search clients."""
        llm_client = MagicMock()
        search_client = MagicMock()
        return llm_client, search_client

    def test_init_defaults(self, tmp_path):
        from info_gathering_agent.research.gatherer import Gatherer

        llm, search = self._make_mock_clients()
        gatherer = Gatherer(
            llm_client=llm,
            search_client=search,
            output_dir=tmp_path,
        )
        assert gatherer.min_iterations == 3
        assert gatherer.max_iterations == 10
        assert gatherer.language == "ja"
        assert gatherer.session is None
        assert gatherer.evidence_locker is None

    def test_init_custom_params(self, tmp_path):
        from info_gathering_agent.research.gatherer import Gatherer

        llm, search = self._make_mock_clients()
        gatherer = Gatherer(
            llm_client=llm,
            search_client=search,
            min_iterations=5,
            max_iterations=15,
            max_queries_per_iteration=5,
            max_pages_per_query=5,
            language="en",
            output_dir=tmp_path,
        )
        assert gatherer.min_iterations == 5
        assert gatherer.max_iterations == 15
        assert gatherer.max_queries_per_iteration == 5
        assert gatherer.language == "en"

    def test_init_with_extended_mode(self, tmp_path):
        from info_gathering_agent.research.gatherer import Gatherer

        llm, search = self._make_mock_clients()
        gatherer = Gatherer(
            llm_client=llm,
            search_client=search,
            extended_mode=True,
            output_dir=tmp_path,
        )
        assert gatherer.extended_mode is True
        assert gatherer.site_crawler is not None

    def test_init_with_content_filter_none(self, tmp_path):
        from info_gathering_agent.research.gatherer import Gatherer

        llm, search = self._make_mock_clients()
        gatherer = Gatherer(
            llm_client=llm,
            search_client=search,
            filter_mode="none",
            output_dir=tmp_path,
        )
        assert gatherer.content_filter is None

    def test_init_with_content_filter_moderate(self, tmp_path):
        from info_gathering_agent.research.gatherer import Gatherer

        llm, search = self._make_mock_clients()
        gatherer = Gatherer(
            llm_client=llm,
            search_client=search,
            filter_mode="moderate",
            output_dir=tmp_path,
        )
        assert gatherer.content_filter is not None

    def test_get_session_before_research(self, tmp_path):
        from info_gathering_agent.research.gatherer import Gatherer

        llm, search = self._make_mock_clients()
        gatherer = Gatherer(llm_client=llm, search_client=search, output_dir=tmp_path)
        assert gatherer.get_session() is None

    def test_get_evidence_locker_before_research(self, tmp_path):
        from info_gathering_agent.research.gatherer import Gatherer

        llm, search = self._make_mock_clients()
        gatherer = Gatherer(llm_client=llm, search_client=search, output_dir=tmp_path)
        assert gatherer.get_evidence_locker() is None


class TestGatheringResult:
    """Test GatheringResult dataclass."""

    def test_creation(self):
        from info_gathering_agent.main import GatheringResult

        result = GatheringResult(
            session_id="abc123",
            query="test query",
        )
        assert result.session_id == "abc123"
        assert result.query == "test query"
        assert result.section_summaries == {}
        assert result.executive_summary == {}

    def test_to_dict(self):
        from info_gathering_agent.main import GatheringResult

        result = GatheringResult(
            session_id="abc123",
            query="test query",
            section_summaries={"1": {"title": "Test", "content": "Content"}},
            executive_summary={"key_findings": ["finding1"]},
        )
        d = result.to_dict()
        assert d["session_id"] == "abc123"
        assert d["query"] == "test query"
        assert "1" in d["section_summaries"]
        assert "key_findings" in d["executive_summary"]
