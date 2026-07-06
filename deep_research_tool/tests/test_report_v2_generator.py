"""
Tests for ReportGeneratorV2 chapter generation, material mapping,
and the naturalness polish pass.
"""

import json

import pytest
from unittest.mock import Mock

from deep_research_tool.report.v2.generator import (
    CHAPTER_META_DELIMITER,
    ChapterContent,
    ReportGeneratorV2,
)
from deep_research_tool.report.v2.context import ReportContext


def make_llm(contents):
    """Mock LLM returning the given contents in order (last repeats)."""
    llm = Mock()
    mocks = [Mock(content=c) for c in contents]

    def generate(prompt, **kwargs):
        if len(mocks) > 1:
            return mocks.pop(0)
        return mocks[0]

    llm.generate = Mock(side_effect=generate)
    return llm


def make_generator(llm, **kwargs):
    return ReportGeneratorV2(llm_client=llm, **kwargs)


def make_context():
    return ReportContext(research_topic="炭素繊維の市場調査")


class TestGenerateChapter:
    """Tests for _generate_chapter parsing and material mapping."""

    def test_parses_delimiter_output_and_computes_word_count(self):
        body = "## 1. 市場動向\n\n市場は拡大している。[SOURCE 1] 今後も成長が見込まれる。"
        llm = make_llm([
            body + "\n\n" + CHAPTER_META_DELIMITER + "\n"
            + json.dumps({
                "key_points": ["市場拡大"],
                "terms_used": ["炭素繊維"],
                "facts_stated": ["市場は拡大している"],
            }, ensure_ascii=False)
        ])
        generator = make_generator(llm)
        chapter = generator._generate_chapter(
            section_number="1",
            section_title="市場動向",
            section_description="説明",
            content_data={"content": "素材テキスト", "summary": "要約", "sources": []},
            context=make_context(),
        )
        assert chapter.content == body
        assert chapter.word_count == len(body)  # computed in Python
        assert chapter.key_points == ["市場拡大"]
        assert chapter.terms_used == ["炭素繊維"]

    def test_legacy_json_response_still_parsed(self):
        llm = make_llm([json.dumps({
            "content": "## 1. 章\n\nレガシー形式の本文。",
            "key_points": ["p"],
            "terms_used": [],
            "facts_stated": [],
            "word_count": 999,
        }, ensure_ascii=False)])
        generator = make_generator(llm)
        chapter = generator._generate_chapter(
            section_number="1",
            section_title="章",
            section_description="",
            content_data={"content": "素材"},
            context=make_context(),
        )
        assert chapter.content == "## 1. 章\n\nレガシー形式の本文。"
        assert chapter.key_points == ["p"]
        # word_count recomputed from the actual body, not trusted from the LLM
        assert chapter.word_count == len(chapter.content)

    def test_prompt_contains_researcher_stored_content(self):
        """Material bug fix: content stored under 'content' key reaches the prompt."""
        llm = make_llm(["## 1. 章\n\n本文。\n" + CHAPTER_META_DELIMITER + "\n{}"])
        generator = make_generator(llm)
        generator._generate_chapter(
            section_number="1",
            section_title="章",
            section_description="",
            content_data={
                "content": "リサーチで収集済みの本文素材テキスト",
                "summary": "セクション要約",
                "sources": ["https://example.com/src"],
            },
            context=make_context(),
        )
        prompt = llm.generate.call_args.args[0]
        assert "リサーチで収集済みの本文素材テキスト" in prompt
        assert "セクション要約" in prompt
        assert "https://example.com/src" in prompt
        assert "情報なし" not in prompt

    def test_prompt_prefers_extracted_content_when_present(self):
        llm = make_llm(["## 1. 章\n\n本文。\n" + CHAPTER_META_DELIMITER + "\n{}"])
        generator = make_generator(llm)
        generator._generate_chapter(
            section_number="1",
            section_title="章",
            section_description="",
            content_data={
                "content": "fallback本文",
                "extracted_content": [
                    {"title": "ソースA", "content": "抽出済みコンテンツA"},
                ],
            },
            context=make_context(),
        )
        prompt = llm.generate.call_args.args[0]
        assert "抽出済みコンテンツA" in prompt

    def test_empty_content_data_yields_no_info(self):
        llm = make_llm(["## 1. 章\n\n本文。\n" + CHAPTER_META_DELIMITER + "\n{}"])
        generator = make_generator(llm)
        generator._generate_chapter(
            section_number="1",
            section_title="章",
            section_description="",
            content_data={},
            context=make_context(),
        )
        prompt = llm.generate.call_args.args[0]
        assert "情報なし" in prompt


class TestPolishChapters:
    """Tests for the naturalness polish pass."""

    def make_chapters(self):
        return {
            "1": ChapterContent(
                section_number="1", section_title="第一章",
                content="## 1. 第一章\n\n" + "これは第一章の本文である。" * 5,
            ),
            "2": ChapterContent(
                section_number="2", section_title="第二章",
                content="## 2. 第二章\n\n" + "これは第二章の本文である。" * 5,
            ),
        }

    def test_polish_called_once_per_chapter(self):
        chapters = self.make_chapters()
        # Return a same-length polished text so the length guard accepts it
        polished_1 = chapters["1"].content.replace("である。", "だ。よい。")
        polished_2 = chapters["2"].content.replace("である。", "だ。よい。")
        llm = make_llm([polished_1, polished_2])
        generator = make_generator(llm)

        result = generator._polish_chapters(chapters, make_context())

        assert llm.generate.call_count == 2
        assert result["1"].content == polished_1
        assert result["1"].is_draft is False
        assert result["1"].word_count == len(polished_1)

    def test_polish_prompt_carries_previous_chapter_tail(self):
        chapters = self.make_chapters()
        llm = make_llm([chapters["1"].content, chapters["2"].content])
        generator = make_generator(llm)

        generator._polish_chapters(chapters, make_context())

        second_prompt = llm.generate.call_args_list[1].args[0]
        assert "第一章の本文" in second_prompt  # tail of chapter 1 as context

    def test_length_guard_rejects_short_rewrite(self):
        chapters = self.make_chapters()
        original = chapters["1"].content
        llm = make_llm(["短すぎ。"])
        generator = make_generator(llm)

        result = generator._polish_chapters({"1": chapters["1"]}, make_context())

        assert result["1"].content == original  # rejected, original kept
        assert result["1"].is_draft is True

    def test_polish_survives_llm_error(self):
        chapters = self.make_chapters()
        original = chapters["1"].content
        llm = Mock()
        llm.generate = Mock(side_effect=RuntimeError("api down"))
        generator = make_generator(llm)

        result = generator._polish_chapters({"1": chapters["1"]}, make_context())

        assert result["1"].content == original

    def test_enable_polish_flag_default_true(self):
        generator = make_generator(make_llm(["x"]))
        assert generator.enable_polish is True
        generator_off = make_generator(make_llm(["x"]), enable_polish=False)
        assert generator_off.enable_polish is False


class TestStyleInstructions:
    def test_ja_style_instructions_include_naturalness_rules(self):
        context = make_context()
        instructions = context.get_style_instructions()
        assert "翻訳調" in instructions
        assert "自然な日本語のための共通ルール" in instructions

    def test_en_style_instructions_unchanged(self):
        context = ReportContext(research_topic="topic", language="en")
        instructions = context.get_style_instructions()
        assert "Writing Style Instructions" in instructions
        assert "翻訳調" not in instructions
