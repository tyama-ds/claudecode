"""
Tests for naturalness-focused prompt/parsing behavior in ContentExtractor
and the shared prose/meta helpers.
"""

import json

import pytest
from unittest.mock import Mock

from deep_research_tool.research.content_extractor import (
    ContentExtractor,
    ExtractedContent,
    JA_STYLE_GUIDE,
    SECTION_META_DELIMITER,
)
from deep_research_tool.utils.helpers import (
    extract_json_array_from_response,
    split_prose_and_meta,
)


def make_llm(content):
    llm = Mock()
    llm.generate = Mock(return_value=Mock(content=content))
    return llm


def make_extracted():
    return [
        ExtractedContent(
            source_url="https://example.com",
            source_title="Test",
            raw_content="Raw",
            processed_content="Processed",
        )
    ]


class TestSplitProseAndMeta:
    """Tests for the prose + delimiter + metadata parsing helper."""

    def test_delimiter_format(self):
        text = (
            "本文である。[SOURCE 1]\n\n続く段落である。\n\n"
            "===SECTION_META===\n"
            '{"summary": "要約", "confidence_level": "high"}'
        )
        body, meta = split_prose_and_meta(text, "===SECTION_META===")
        assert body == "本文である。[SOURCE 1]\n\n続く段落である。"
        assert meta["summary"] == "要約"
        assert meta["confidence_level"] == "high"

    def test_legacy_json_format(self):
        text = json.dumps({"content": "レガシー本文", "summary": "要約"}, ensure_ascii=False)
        body, meta = split_prose_and_meta(text, "===SECTION_META===")
        assert body == "レガシー本文"
        assert meta["summary"] == "要約"

    def test_plain_text_fallback(self):
        body, meta = split_prose_and_meta("ただの本文。", "===SECTION_META===")
        assert body == "ただの本文。"
        assert meta == {}

    def test_delimiter_with_broken_meta_keeps_body(self):
        text = "本文。\n===SECTION_META===\nnot json here"
        body, meta = split_prose_and_meta(text, "===SECTION_META===")
        assert body == "本文。"
        assert meta == {}

    def test_code_fenced_body_is_unwrapped(self):
        text = "```markdown\n本文。\n```\n===SECTION_META===\n{\"summary\": \"s\"}"
        body, meta = split_prose_and_meta(text, "===SECTION_META===")
        assert body == "本文。"
        assert meta["summary"] == "s"


class TestExtractJsonArray:
    def test_plain_array(self):
        assert extract_json_array_from_response('[{"a": 1}]') == [{"a": 1}]

    def test_fenced_array(self):
        assert extract_json_array_from_response('```json\n[1, 2]\n```') == [1, 2]

    def test_array_with_surrounding_text(self):
        assert extract_json_array_from_response('Here: [1, 2] done') == [1, 2]

    def test_no_array_raises(self):
        with pytest.raises(ValueError):
            extract_json_array_from_response("no array here")


class TestJapanesePrompts:
    """The ja path must use fully Japanese prompts with style guidance."""

    def test_synthesis_prompt_is_japanese_with_style_guide(self):
        llm = make_llm("本文。[SOURCE 1]\n" + SECTION_META_DELIMITER + '\n{"summary": "s"}')
        extractor = ContentExtractor(llm, language="ja")
        extractor.synthesize_section_content(
            section_title="市場動向",
            section_description="説明",
            extracted_contents=make_extracted(),
        )
        prompt = llm.generate.call_args.args[0]
        assert "文体ガイドライン" in prompt
        assert "[SOURCE" in prompt
        assert "Write in Japanese." not in prompt
        assert SECTION_META_DELIMITER in prompt

    def test_synthesis_parses_delimiter_response(self):
        llm = make_llm(
            "調査の結果、市場は拡大している。[SOURCE 1]\n\n"
            + SECTION_META_DELIMITER
            + '\n{"summary": "市場拡大の要約", "information_gaps": ["価格動向"], "confidence_level": "high"}'
        )
        extractor = ContentExtractor(llm, language="ja")
        result = extractor.synthesize_section_content(
            section_title="市場動向",
            section_description="説明",
            extracted_contents=make_extracted(),
        )
        assert result["content"] == "調査の結果、市場は拡大している。[SOURCE 1]"
        assert result["summary"] == "市場拡大の要約"
        assert result["information_gaps"] == ["価格動向"]
        assert result["confidence_level"] == "high"
        assert result["source_references"] == [1]

    def test_synthesis_plain_prose_without_delimiter(self):
        """Delimiter missing + non-JSON response: whole text becomes content."""
        llm = make_llm("区切り記号のない本文である。[SOURCE 1]")
        extractor = ContentExtractor(llm, language="ja")
        result = extractor.synthesize_section_content(
            section_title="t", section_description="d",
            extracted_contents=make_extracted(),
        )
        assert result["content"] == "区切り記号のない本文である。[SOURCE 1]"
        assert result["confidence_level"] == "medium"

    def test_english_path_keeps_english_prompt(self):
        llm = make_llm("Body text. [SOURCE 1]\n" + SECTION_META_DELIMITER + '\n{"summary": "s"}')
        extractor = ContentExtractor(llm, language="en")
        extractor.synthesize_section_content(
            section_title="Trends", section_description="desc",
            extracted_contents=make_extracted(),
        )
        prompt = llm.generate.call_args.args[0]
        assert "Write in en." in prompt
        assert "文体ガイドライン" not in prompt

    def test_extract_relevant_content_ja_prompt(self):
        llm = make_llm(json.dumps({
            "processed_content": "自然な要約文である。",
            "key_points": ["p1"],
            "relevance_score": 0.7,
        }, ensure_ascii=False))
        extractor = ContentExtractor(llm, language="ja")
        result = extractor.extract_relevant_content(
            raw_content="生テキスト",
            source_url="https://example.com",
            source_title="タイトル",
            section_context="セクション",
            research_query="クエリ",
        )
        prompt = llm.generate.call_args.args[0]
        # Chunked-extraction architecture: unified English prompt with a
        # language instruction, carrying the query/section context
        assert "Respond in Japanese" in prompt
        assert "クエリ" in prompt and "セクション" in prompt
        assert result.processed_content == "自然な要約文である。"
        assert result.relevance_score == 0.7

    def test_extract_relevant_content_parses_fenced_json(self):
        llm = make_llm('```json\n{"processed_content": "フェンス内の要約である。", "relevance_score": 0.6}\n```')
        extractor = ContentExtractor(llm, language="ja")
        result = extractor.extract_relevant_content(
            raw_content="生テキスト",
            source_url="https://example.com",
            source_title="タイトル",
            section_context="セクション",
            research_query="クエリ",
        )
        assert result.processed_content == "フェンス内の要約である。"


class TestOutlineParsing:
    """Outline generation accepts object and legacy bare-array formats."""

    def test_outline_object_format(self):
        llm = make_llm(json.dumps({
            "outline": [{"title": "論点1", "description": "説明", "key_facts": []}]
        }, ensure_ascii=False))
        extractor = ContentExtractor(llm, language="ja")
        outline = extractor._generate_section_outline("t", "d", "sources", "", "")
        assert outline[0]["title"] == "論点1"

    def test_outline_legacy_array_format(self):
        llm = make_llm('[{"title": "Point 1", "description": "d", "key_facts": []}]')
        extractor = ContentExtractor(llm, language="ja")
        outline = extractor._generate_section_outline("t", "d", "sources", "", "")
        assert outline[0]["title"] == "Point 1"

    def test_outline_fenced_object(self):
        llm = make_llm('```json\n{"outline": [{"title": "A", "description": "d", "key_facts": []}]}\n```')
        extractor = ContentExtractor(llm, language="ja")
        outline = extractor._generate_section_outline("t", "d", "sources", "", "")
        assert outline[0]["title"] == "A"

    def test_outline_failure_returns_fallback(self):
        llm = make_llm("not json")
        extractor = ContentExtractor(llm, language="ja")
        outline = extractor._generate_section_outline("MyTitle", "d", "s", "", "")
        assert len(outline) == 3  # fallback outline

    def test_integrate_content_ja_prompt(self):
        llm = make_llm("統合された本文である。[SOURCE 1]")
        extractor = ContentExtractor(llm, language="ja")
        result = extractor._integrate_content(
            "セクション", "説明",
            [{"title": "論点1", "content": "断片1 [SOURCE 1]"}],
            "",
        )
        prompt = llm.generate.call_args.args[0]
        assert "一人の書き手" in prompt
        assert "文体ガイドライン" in prompt
        assert result == "統合された本文である。[SOURCE 1]"


class TestStyleGuideContent:
    def test_style_guide_mentions_translationese(self):
        assert "翻訳調" in JA_STYLE_GUIDE
        assert "である" in JA_STYLE_GUIDE
