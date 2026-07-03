"""
Tests for FermiEstimator (Fermi estimation with mock LLM).
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deep_research_tool.thinking.fermi_estimator import (
    FermiEstimator,
    FermiEstimate,
    FermiFactor,
    format_number,
)


def _mock_llm(payload: dict):
    """LLM client mock whose generate() returns the payload as JSON."""
    llm = MagicMock()
    response = MagicMock()
    response.content = f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
    llm.generate.return_value = response
    return llm


PIANO_TUNER_PAYLOAD = {
    "unit": "人",
    "formula": "世帯数 × ピアノ保有率 ÷ 調律師1人あたり担当ピアノ数",
    "factors": [
        {
            "name": "世帯数",
            "description": "日本の総世帯数",
            "low": 45_000_000, "mid": 50_000_000, "high": 55_000_000,
            "unit": "世帯", "operation": "multiply", "basis": "known_value",
        },
        {
            "name": "ピアノ保有率",
            "description": "ピアノを保有する世帯の割合",
            "low": 0.02, "mid": 0.05, "high": 0.10,
            "unit": "", "operation": "multiply", "basis": "assumption",
        },
        {
            "name": "調律師1人あたり担当ピアノ数",
            "description": "年間に1人が調律できるピアノ数",
            "low": 500, "mid": 1000, "high": 2000,
            "unit": "台/人", "operation": "divide", "basis": "llm_knowledge",
        },
    ],
    "assumptions": ["ピアノは年1回調律される"],
    "reasoning": "世帯数からピアノ台数を推定し、調律師の処理能力で割る",
}


class TestFermiEstimator:
    """Fermi estimation pipeline with mocked LLM."""

    def test_estimate_combines_factors(self):
        estimator = FermiEstimator(llm_client=_mock_llm(PIANO_TUNER_PAYLOAD))
        result = estimator.estimate("日本のピアノ調律師の人数は？")

        assert isinstance(result, FermiEstimate)
        assert len(result.factors) == 3
        assert result.unit == "人"
        # mid: 50M * 0.05 / 1000 = 2500
        assert result.value == pytest.approx(2500)
        # low: 45M * 0.02 / 2000 = 450 (divisor uses its high for low bound)
        assert result.low == pytest.approx(450)
        # high: 55M * 0.10 / 500 = 11000
        assert result.high == pytest.approx(11000)
        assert result.low < result.value < result.high

    def test_confidence_in_valid_range(self):
        estimator = FermiEstimator(llm_client=_mock_llm(PIANO_TUNER_PAYLOAD))
        result = estimator.estimate("test")
        assert 0.0 <= result.confidence <= 1.0

    def test_known_values_included_in_prompt(self):
        llm = _mock_llm(PIANO_TUNER_PAYLOAD)
        estimator = FermiEstimator(llm_client=llm)
        estimator.estimate("test", known_values={"日本の人口": 125_000_000})

        prompt = llm.generate.call_args[0][0]
        assert "日本の人口" in prompt
        assert "125000000" in prompt

    def test_no_factors_raises_value_error(self):
        estimator = FermiEstimator(llm_client=_mock_llm({"factors": []}))
        with pytest.raises(ValueError):
            estimator.estimate("test")

    def test_unparsable_response_raises_value_error(self):
        llm = MagicMock()
        response = MagicMock()
        response.content = "すみません、推定できませんでした。"
        llm.generate.return_value = response

        estimator = FermiEstimator(llm_client=llm)
        with pytest.raises(ValueError):
            estimator.estimate("test")

    def test_to_markdown_contains_estimate(self):
        estimator = FermiEstimator(llm_client=_mock_llm(PIANO_TUNER_PAYLOAD))
        result = estimator.estimate("日本のピアノ調律師の人数は？")
        md = result.to_markdown()
        assert "フェルミ推定" in md
        assert "世帯数" in md
        assert "前提条件" in md


class TestFormatNumber:
    """Number formatting for display."""

    def test_zero(self):
        assert format_number(0) == "0"

    def test_large_uses_scientific(self):
        assert "10^" in format_number(1.25e8)

    def test_small_uses_scientific(self):
        assert "10^" in format_number(0.0001)

    def test_moderate_plain(self):
        assert format_number(2500) == "2,500"
