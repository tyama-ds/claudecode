"""Offline regressions for complete, date-grounded research planning."""

import copy
import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from deep_research_tool.research.query_generator import QueryGenerator


def plan_data():
    # A user-defined structure may legitimately be smaller than the default
    # five-chapter style guide. Essential validation must still accept it.
    return {
        "title": "Carbon fiber research",
        "table_of_contents": [{"section": "1", "title": "Manufacturing evidence"}],
        "search_queries": ["carbon fiber manufacturing"],
    }


def response(content, finish_reason="stop"):
    return SimpleNamespace(
        content=content, finish_reason=finish_reason, model="offline-model", usage={}
    )


def generator_for(content, finish_reason="stop", language="en"):
    llm = Mock()
    llm.generate.return_value = response(content, finish_reason)
    return QueryGenerator(llm, language=language)


@pytest.mark.parametrize("language", ["ja", "en"])
def test_initial_plan_uses_today_once_and_preserves_explicit_historical_period(language):
    data = plan_data()
    data["search_queries"] = ["carbon fiber 2021-2023"]
    generator = generator_for(json.dumps(data), language=language)
    query = "Carbon fiber during 2021-2023"
    with patch("deep_research_tool.research.query_generator.date") as clock:
        clock.today.return_value = date(2026, 9, 15)
        result = generator.create_research_plan(query, requirements="1. Manufacturing")
        clock.today.assert_called_once_with()

    prompt = generator.llm.generate.call_args.args[0]
    assert "2026-09-15" in prompt
    assert "2023-09-15" in prompt
    assert query in prompt
    assert "2024" not in prompt
    assert "2026" in prompt
    assert result.search_queries == data["search_queries"]
    if language == "ja":
        assert "過去の年・期間や暦年の指定は、その指定を維持" in prompt
        assert "確認済みの事実として記載しない" in prompt
    else:
        assert "Preserve the user's explicitly requested historical dates" in prompt
        assert "present them as verified facts" in prompt


@pytest.mark.parametrize("language", ["ja", "en"])
def test_revision_receives_same_date_grounding(language):
    data = plan_data()
    generator = generator_for(json.dumps(data), language=language)
    original = generator._build_plan_from_data(data, "carbon fiber")
    with patch("deep_research_tool.research.query_generator.date") as clock:
        clock.today.return_value = date(2026, 9, 15)
        generator.revise_research_plan(original, "Cover the past 3 years", "carbon fiber")
        clock.today.assert_called_once_with()
    prompt = generator.llm.generate.call_args.args[0]
    assert "2026-09-15" in prompt
    assert "2023-09-15" in prompt
    assert "Cover the past 3 years" in prompt


def test_date_context_handles_leap_day():
    context = QueryGenerator(object(), language="en")._planning_context(date(2028, 2, 29))
    assert "2028-02-29" in context
    assert "2025-02-28" in context


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens", "max_output_tokens", "max_token_limit"])
def test_token_limit_rejects_even_a_syntactically_complete_plan(finish_reason):
    generator = generator_for(json.dumps(plan_data()), finish_reason)
    assert generator._generate_research_plan_attempt("carbon fiber", "", "") is None


@pytest.mark.parametrize("content", [
    '{"title":"Carbon fiber","table_of_contents":[{"section":"1","title":"Manufacturing"}],"search_queries":["query",',
    json.dumps(plan_data()) + '\n{"search_queries":["unfinished",',
    "```json\n" + json.dumps(plan_data()),
    json.dumps([plan_data()]),
])
def test_normal_stop_does_not_salvage_truncated_or_wrong_root_json(content):
    generator = generator_for(content, finish_reason="stop")
    assert generator._generate_research_plan_attempt("carbon fiber", "", "") is None


@pytest.mark.parametrize("wrapper", ["{}", "```json\n{}\n```", "```\n{}\n```"])
def test_complete_plan_accepts_plain_json_or_enclosing_fence(wrapper):
    generator = generator_for(wrapper.format(json.dumps(plan_data())))
    result = generator.create_research_plan("carbon fiber", requirements="1. Manufacturing")
    assert result.title == "Carbon fiber research"
    assert len(result.table_of_contents.items) == 1
    assert result.search_queries == ["carbon fiber manufacturing"]


@pytest.mark.parametrize("change", [
    {"title": " "},
    {"table_of_contents": []},
    {"table_of_contents": {}},
    {"table_of_contents": ["Manufacturing"]},
    {"table_of_contents": [{"section": "1", "title": 12}]},
    {"table_of_contents": [{"title": "Manufacturing", "section": []}]},
    {"table_of_contents": [{"section": "1", "title": "Manufacturing", "subsections": {}}]},
    {"table_of_contents": [{"section": "1", "title": "Manufacturing", "subsections": [None]}]},
    {"table_of_contents": [{"section": "1", "title": "Manufacturing", "subsections": [{"section": "1.1", "title": " "}]}]},
    {"table_of_contents": [{"section": "1", "title": "Manufacturing", "subsections": [{"section": "1.1", "title": "Subsection", "description": []}]}]},
    {"search_queries": []},
    {"search_queries": "query"},
    {"search_queries": [" "]},
    {"search_queries": [123]},
    {"key_terms": "term"},
    {"suggested_sources": [None]},
    {"summary": []},
])
def test_schema_validation_is_required_even_with_user_toc_preferences(change):
    data = plan_data()
    data.update(change)
    generator = generator_for(json.dumps(data))
    with patch.object(generator, "_validate_toc_quality") as style_check:
        result = generator.create_research_plan(
            "carbon fiber", requirements="1. Manufacturing", max_retries=0
        )
    style_check.assert_not_called()
    assert result.title != data["title"]
    assert result.summary.startswith("Fallback plan:")
    assert result.methodology_notes.startswith("Fallback plan created due to:")


def test_missing_required_fields_retry_then_return_a_visibly_labeled_fallback():
    generator = generator_for('{"title":"Unsupported generated claims"}', language="ja")
    result = generator.create_research_plan("carbon fiber", requirements="1. Manufacturing")
    assert generator.llm.generate.call_count == 3
    assert "暫定計画" in result.summary
    assert "Unsupported generated claims" not in result.title
    assert result.table_of_contents.items
    assert result.search_queries


def test_truncated_normal_stop_retries_with_a_complete_new_response():
    generator = generator_for("")
    generator.llm.generate.side_effect = [
        response('{"title":"Unfinished","table_of_contents":['),
        response(json.dumps(plan_data())),
    ]
    result = generator.create_research_plan("carbon fiber", requirements="1. Manufacturing")
    assert result.title == "Carbon fiber research"
    assert generator.llm.generate.call_count == 2


@pytest.mark.parametrize("content,finish_reason", [
    ('{"title":"Unfinished","table_of_contents":[', "stop"),
    (json.dumps({**plan_data(), "search_queries": []}), "stop"),
    (json.dumps(plan_data()), "length"),
])
def test_invalid_revision_raises_without_mutating_the_original_plan(content, finish_reason):
    generator = generator_for(content, finish_reason)
    original = generator._build_plan_from_data(plan_data(), "carbon fiber")
    before = copy.deepcopy(original.to_dict())
    with pytest.raises(ValueError):
        generator.revise_research_plan(original, "Update the plan", "carbon fiber")
    assert original.to_dict() == before


def test_direct_builder_cannot_bypass_required_plan_schema():
    with pytest.raises(ValueError, match="table_of_contents"):
        QueryGenerator._build_plan_from_data({"title": "Incomplete"}, "carbon fiber")


@pytest.mark.parametrize("toc", [
    [{"title": "A"}, {"title": "B"}],
    [{"section": "", "title": "A"}],
    [{"section": "  ", "title": "A"}],
    [{"section": "1", "title": "A"}, {"section": "1", "title": "B"}],
    [{"section": "1", "title": "A"}, {"section": " 1 ", "title": "B"}],
    [{"section": "1", "title": "A", "subsections": [{"title": "Subsection"}]}],
    [{"section": "1", "title": "A", "subsections": [{"section": "1", "title": "Subsection"}]}],
    [{"section": "1", "title": "A", "subsections": [
        {"section": "1.1", "title": "Subsection A"},
        {"section": "1.1", "title": "Subsection B"},
    ]}],
    [
        {"section": "1", "title": "A", "subsections": [{"section": "2", "title": "Subsection"}]},
        {"section": "2", "title": "B"},
    ],
])
def test_missing_or_duplicate_section_ids_cannot_overwrite_research_chapters(toc):
    data = {**plan_data(), "table_of_contents": toc}
    generator = generator_for(json.dumps(data))
    result = generator.create_research_plan(
        "carbon fiber", requirements="1. Manufacturing", max_retries=0
    )
    assert result.summary.startswith("Fallback plan:")
    with pytest.raises(ValueError, match="section"):
        generator._build_plan_from_data(data, "carbon fiber")


def test_distinct_section_ids_preserve_every_chapter_and_subsection():
    data = {**plan_data(), "table_of_contents": [
        {"section": "1", "title": "A", "subsections": [{"section": "1.1", "title": "Subsection A"}]},
        {"section": "2", "title": "B", "subsections": [{"section": "2.1", "title": "Subsection B"}]},
    ]}
    generator = generator_for(json.dumps(data))
    result = generator.create_research_plan("carbon fiber", requirements="1. Manufacturing")
    sections = result.table_of_contents.get_flat_sections()
    assert [section.section for section in sections] == ["1", "1.1", "2", "2.1"]
    assert len({section.section: section.title for section in sections}) == 4
