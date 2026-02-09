"""Tests for token usage tracking."""

import pytest
from datetime import datetime

from deep_research_tool.api.base import (
    TokenUsage,
    TokenUsageStats,
    get_token_stats,
    reset_token_stats,
)


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_default_values(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.model == ""
        assert usage.timestamp is not None

    def test_custom_values(self):
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-5-mini"
        )
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.model == "gpt-5-mini"

    def test_to_dict(self):
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-5-mini"
        )
        data = usage.to_dict()
        assert data["prompt_tokens"] == 100
        assert data["completion_tokens"] == 50
        assert data["total_tokens"] == 150
        assert data["model"] == "gpt-5-mini"
        assert "timestamp" in data


class TestTokenUsageStats:
    """Tests for TokenUsageStats dataclass."""

    def test_default_values(self):
        stats = TokenUsageStats()
        assert stats.total_prompt_tokens == 0
        assert stats.total_completion_tokens == 0
        assert stats.total_tokens == 0
        assert stats.total_calls == 0
        assert stats.calls_by_model == {}
        assert stats.tokens_by_model == {}
        assert stats.history == []

    def test_add_usage(self):
        stats = TokenUsageStats()
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-5-mini"
        )
        stats.add_usage(usage)

        assert stats.total_prompt_tokens == 100
        assert stats.total_completion_tokens == 50
        assert stats.total_tokens == 150
        assert stats.total_calls == 1
        assert stats.calls_by_model["gpt-5-mini"] == 1
        assert stats.tokens_by_model["gpt-5-mini"] == 150
        assert len(stats.history) == 1

    def test_add_multiple_usages(self):
        stats = TokenUsageStats()

        # Add first usage
        stats.add_usage(TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-5-mini"
        ))

        # Add second usage (same model)
        stats.add_usage(TokenUsage(
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            model="gpt-5-mini"
        ))

        # Add third usage (different model)
        stats.add_usage(TokenUsage(
            prompt_tokens=500,
            completion_tokens=250,
            total_tokens=750,
            model="gpt-5"
        ))

        assert stats.total_prompt_tokens == 800
        assert stats.total_completion_tokens == 400
        assert stats.total_tokens == 1200
        assert stats.total_calls == 3
        assert stats.calls_by_model["gpt-5-mini"] == 2
        assert stats.calls_by_model["gpt-5"] == 1
        assert stats.tokens_by_model["gpt-5-mini"] == 450
        assert stats.tokens_by_model["gpt-5"] == 750
        assert len(stats.history) == 3

    def test_to_dict(self):
        stats = TokenUsageStats()
        stats.add_usage(TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-5-mini"
        ))

        data = stats.to_dict()
        assert data["total_prompt_tokens"] == 100
        assert data["total_completion_tokens"] == 50
        assert data["total_tokens"] == 150
        assert data["total_calls"] == 1
        assert "calls_by_model" in data
        assert "tokens_by_model" in data

    def test_get_summary_english(self):
        stats = TokenUsageStats()
        stats.add_usage(TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-5-mini"
        ))

        summary = stats.get_summary("en")
        assert "Token Usage" in summary
        assert "150" in summary
        assert "100" in summary
        assert "50" in summary
        assert "gpt-5-mini" in summary

    def test_get_summary_japanese(self):
        stats = TokenUsageStats()
        stats.add_usage(TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-5-mini"
        ))

        summary = stats.get_summary("ja")
        assert "トークン使用量" in summary
        assert "150" in summary
        assert "100" in summary
        assert "50" in summary
        assert "gpt-5-mini" in summary

    def test_reset(self):
        stats = TokenUsageStats()
        stats.add_usage(TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-5-mini"
        ))

        stats.reset()

        assert stats.total_prompt_tokens == 0
        assert stats.total_completion_tokens == 0
        assert stats.total_tokens == 0
        assert stats.total_calls == 0
        assert stats.calls_by_model == {}
        assert stats.tokens_by_model == {}
        assert stats.history == []


class TestGlobalTokenStats:
    """Tests for global token stats functions."""

    def test_get_token_stats(self):
        stats = get_token_stats()
        assert isinstance(stats, TokenUsageStats)

    def test_reset_token_stats(self):
        # Add some usage
        stats = get_token_stats()
        stats.add_usage(TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-5-mini"
        ))

        # Reset
        reset_token_stats()

        # Verify reset
        stats = get_token_stats()
        assert stats.total_tokens == 0
        assert stats.total_calls == 0

    def test_global_stats_persistence(self):
        reset_token_stats()

        # Add usage through global stats
        stats1 = get_token_stats()
        stats1.add_usage(TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            model="gpt-5-mini"
        ))

        # Get stats again and verify
        stats2 = get_token_stats()
        assert stats2.total_tokens == 150
        assert stats2.total_calls == 1

        # Clean up
        reset_token_stats()
