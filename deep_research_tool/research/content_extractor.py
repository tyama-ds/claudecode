"""
Content Extractor - Extract and process information from search results.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..utils.helpers import (
    extract_json_array_from_response,
    extract_json_from_response,
    split_prose_and_meta,
)

# Delimiter between prose body and metadata JSON in synthesis responses
SECTION_META_DELIMITER = "===SECTION_META==="

# Style guide embedded in Japanese prose-generation prompts
JA_STYLE_GUIDE = """【文体ガイドライン】
- 常体（「だ・である」調）で統一し、調査報告書として自然で引き締まった日本語で書くこと。
- 翻訳調を避けること。例：「〜することができる」→「〜できる」、「〜が存在する」→「〜がある」、「〜という事実」→「〜こと」、「それは〜を意味する」→「つまり〜だ」。不要な受動態や「それ」「これら」の多用も避ける。
- 一文はおおむね60字以内を目安とし、長くなった文は分割する。文末表現（「〜である」「〜だ」「〜といえる」「〜がうかがえる」など）が同じ形で連続しないよう変化をつける。
- 「まず」「一方で」「さらに」「このため」「その結果」「もっとも」などの接続表現で論理の流れを示す。ただし機械的に多用しない。
- 箇条書きは、並列的な列挙が明らかに読みやすい場合に限って使い、基本は段落の文章として論じる。段落は3〜5文でまとめ、段落間のつながりを意識する。
- 主語と述語のねじれ、同じ語の近接した繰り返し、冗長な修飾（「非常に大きな重要性を持つ」→「きわめて重要だ」）に注意する。
- 数値・固有名詞・年号は出典の表記を正確に保持する。"""


@dataclass
class ExtractedContent:
    """Extracted and processed content from a source."""
    source_url: str
    source_title: str
    raw_content: str
    processed_content: str = ""
    key_points: List[str] = field(default_factory=list)
    quotes: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    relevance_score: float = 0.0
    importance_score: float = 0.0  # importance to the overall research purpose
    extraction_notes: str = ""
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_url": self.source_url,
            "source_title": self.source_title,
            "raw_content": self.raw_content,
            "processed_content": self.processed_content,
            "key_points": self.key_points,
            "quotes": self.quotes,
            "images": self.images,
            "relevance_score": self.relevance_score,
            "importance_score": self.importance_score,
            "extraction_notes": self.extraction_notes,
            "extracted_at": self.extracted_at,
        }


class ContentExtractor:
    """Extract and process content from search results using LLM."""

    def __init__(self, llm_client, language: str = "ja"):
        """
        Initialize ContentExtractor.

        Args:
            llm_client: LLM API client instance
            language: Target language for processing
        """
        self.llm = llm_client
        self.language = language

    def extract_relevant_content(
        self,
        raw_content: str,
        source_url: str,
        source_title: str,
        section_context: str,
        research_query: str,
    ) -> ExtractedContent:
        """
        Extract relevant content from raw source material.

        Args:
            raw_content: Raw text content from the source
            source_url: URL of the source
            source_title: Title of the source
            section_context: The section this content is for
            research_query: The original research query

        Returns:
            ExtractedContent with processed information
        """
        # Truncate content if too long
        max_content_length = 8000
        truncated_content = raw_content[:max_content_length]
        if len(raw_content) > max_content_length:
            truncated_content += "\n... [content truncated]"

        if self.language == "ja":
            prompt = f"""情報源URL: {source_url}
情報源タイトル: {source_title}

【調査の文脈】
- 調査テーマ: {research_query}
- 対象セクション: {section_context}

【情報源の内容】
{truncated_content}

この内容を分析し、調査テーマに関連する情報を抽出してください。JSONで回答:
{{
    "processed_content": "関連する内容の要約（500〜1000字）。翻訳調を避け、自然な日本語の文章として書く",
    "key_points": ["要点1", "要点2", ...],
    "quotes": [
        {{"text": "情報源からの正確な引用", "context": "この引用が重要な理由"}}
    ],
    "relevance_score": 0.0から1.0の数値,
    "extraction_notes": "この情報源の品質・限界に関するメモ"
}}

調査テーマとセクションに直接関連する情報に絞ること。
レポートで引用できる正確な引用文を含めること。
relevance_score は 0（無関係）〜1（きわめて関連が高い）で評価すること。"""
        else:
            prompt = f"""Source URL: {source_url}
Source Title: {source_title}

Research Context:
- Query: {research_query}
- Section: {section_context}

Source Content:
{truncated_content}

Respond in {self.language}.

Analyze this content and extract relevant information. Return as JSON:
{{
    "processed_content": "Summarized relevant content (500-1000 words)",
    "key_points": ["key point 1", "key point 2", ...],
    "quotes": [
        {{"text": "exact quote from source", "context": "why this quote is relevant"}}
    ],
    "relevance_score": 0.0-1.0,
    "extraction_notes": "Notes about the quality/limitations of this source"
}}

Focus on information directly relevant to the research query and section.
Include exact quotes that could be cited in the report.
Rate relevance from 0 (not relevant) to 1 (highly relevant)."""

        response = self.llm.generate(prompt)

        try:
            data = extract_json_from_response(response.content)

            # Use processed_content from LLM, fall back to truncated raw content if empty
            processed = data.get("processed_content", "")
            if not processed or len(processed.strip()) < 10:
                # LLM returned empty or very short content, use raw content
                processed = truncated_content[:2000]

            return ExtractedContent(
                source_url=source_url,
                source_title=source_title,
                raw_content=raw_content,
                processed_content=processed,
                key_points=data.get("key_points", []),
                quotes=data.get("quotes", []),
                relevance_score=float(data.get("relevance_score", 0.5)),
                extraction_notes=data.get("extraction_notes", ""),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback: return basic extraction
        return ExtractedContent(
            source_url=source_url,
            source_title=source_title,
            raw_content=raw_content,
            processed_content=truncated_content[:2000],
            relevance_score=0.5,
            extraction_notes="LLM extraction failed, returning raw content",
        )

    def synthesize_section_content(
        self,
        section_title: str,
        section_description: str,
        extracted_contents: List[ExtractedContent],
        requirements: str = "",
    ) -> Dict[str, Any]:
        """
        Synthesize multiple extracted contents into section content.

        Args:
            section_title: Title of the section
            section_description: Description of what the section should cover
            extracted_contents: List of extracted content from sources
            requirements: Original research requirements

        Returns:
            Dictionary with synthesized content and metadata
        """
        # Prepare source summaries (importance = relevance to the overall
        # research purpose, scored against the research goal upstream)
        source_summaries = []
        for i, ec in enumerate(extracted_contents, 1):
            importance = ec.importance_score or ec.relevance_score
            summary = f"""
[SOURCE {i}] {ec.source_title}
URL: {ec.source_url}
Relevance: {ec.relevance_score:.2f}
Importance: {importance:.2f}

Content:
{ec.processed_content[:2000]}

Key Points:
{chr(10).join('- ' + kp for kp in ec.key_points[:5])}
"""
            source_summaries.append(summary)

        sources_text = "\n---\n".join(source_summaries)

        if self.language == "ja":
            # Japanese: write prose directly (no JSON wrapping), metadata after a delimiter
            prompt = f"""あなたは調査レポートの執筆者です。以下の情報源をもとに、レポートの1セクションの本文を執筆してください。

【セクション】{section_title}
【セクションの説明】{section_description}
【調査要件】{requirements if requirements else "包括的な分析"}

【情報源】
{sources_text}

{JA_STYLE_GUIDE}

【執筆ルール】
1. 情報源に基づく事実には、文末に [SOURCE N] の形式で出典番号を付けること。この表記は一字一句変えない（翻訳・省略・変形しない）。
2. あなた自身の分析・解釈は、「〜と考えられる」「〜と推察される」「〜可能性がある」などの表現で、事実の記述と自然に区別すること。
3. 情報源にない主張はしない。情報源の間で内容が食い違う場合は、その旨を本文中で明記する。
4. 各情報源の Importance（調査目的への重要度）を考慮し、重要度の高い情報源を中心に本文を構成すること。重要度の低い情報源は補足として簡潔に扱うか、内容が薄ければ使わなくてよい。
5. 見出しや箇条書きの多用は避け、本文（マークダウンの段落）として書く。セクションタイトルは出力しない。
6. JSONやコードブロックで囲まず、本文をそのまま書き始めること。

本文を書き終えたら、最後に次の区切り記号とメタ情報を必ず出力すること:

{SECTION_META_DELIMITER}
{{"summary": "このセクションの要約（2〜3文）", "analysis_points": ["主要な分析ポイント"], "information_gaps": ["さらに調査が必要な点"], "confidence_level": "high/medium/low"}}"""
        else:
            prompt = f"""Section: {section_title}
Description: {section_description}

Research Requirements: {requirements if requirements else "Comprehensive analysis"}

Available Sources:
{sources_text}

Write in {self.language}.

Synthesize the information from these sources into well-written section content.

IMPORTANT:
1. Clearly distinguish between:
   - Factual information from sources (cite as [SOURCE N])
   - Your own analysis or interpretation (phrase naturally as your assessment)
2. Do not make claims that aren't supported by the sources
3. Note any conflicting information between sources
4. Build the body around high-Importance sources (importance to the research
   purpose); treat low-importance sources as brief supplements or omit them
5. Write flowing paragraphs (markdown), not JSON. Do not output the section title.

After the body, output this delimiter followed by metadata:

{SECTION_META_DELIMITER}
{{"summary": "Brief summary of key findings", "analysis_points": ["Your analytical insights"], "information_gaps": ["Areas needing more research"], "confidence_level": "high/medium/low"}}"""

        response = self.llm.generate(prompt)

        # Prepare fallback content in case parsing fails or content is empty
        fallback_content = "\n\n".join(
            ec.processed_content for ec in extracted_contents if ec.processed_content
        )

        try:
            body, meta = split_prose_and_meta(response.content, SECTION_META_DELIMITER)

            if not body or len(body.strip()) < 10:
                body = fallback_content
                if not meta.get("summary"):
                    meta["summary"] = "Content compiled from multiple sources"

            return {
                "content": body,
                "summary": meta.get("summary", ""),
                "source_references": meta.get(
                    "source_references",
                    list(range(1, len(extracted_contents) + 1)),
                ),
                "analysis_points": meta.get("analysis_points", []),
                "information_gaps": meta.get("information_gaps", []),
                "confidence_level": meta.get("confidence_level", "medium"),
            }
        except Exception:
            pass

        # Fallback
        return {
            "content": fallback_content,
            "summary": "Content synthesized from multiple sources",
            "source_references": list(range(1, len(extracted_contents) + 1)),
            "analysis_points": [],
            "information_gaps": ["Synthesis failed, review needed"],
            "confidence_level": "low",
        }

    def synthesize_section_content_enhanced(
        self,
        section_title: str,
        section_description: str,
        extracted_contents: List[ExtractedContent],
        requirements: str = "",
    ) -> Dict[str, Any]:
        """
        Enhanced multi-pass content synthesis for better report quality.

        This method generates content in three phases:
        1. Generate detailed outline with key points
        2. Generate detailed content for each point
        3. Integrate all content into cohesive section

        Args:
            section_title: Title of the section
            section_description: Description of what the section should cover
            extracted_contents: List of extracted content from sources
            requirements: Original research requirements

        Returns:
            Dictionary with synthesized content and metadata
        """
        lang_instruction = (
            "Write in Japanese." if self.language == "ja"
            else f"Write in {self.language}."
        )

        # Prepare source summaries
        source_summaries = []
        for i, ec in enumerate(extracted_contents, 1):
            importance = ec.importance_score or ec.relevance_score
            summary = f"""[SOURCE {i}] {ec.source_title}
URL: {ec.source_url}
Importance: {importance:.2f}
Content: {ec.processed_content[:1500]}
Key Points: {', '.join(ec.key_points[:3]) if ec.key_points else 'N/A'}
"""
            source_summaries.append(summary)

        sources_text = "\n---\n".join(source_summaries)

        # Phase 1: Generate detailed outline
        print(f"[DEBUG] Phase 1: Generating outline for {section_title}")
        outline = self._generate_section_outline(
            section_title, section_description, sources_text, requirements, lang_instruction
        )

        if not outline:
            print(f"[WARNING] Outline generation failed, falling back to basic synthesis")
            return self.synthesize_section_content(
                section_title, section_description, extracted_contents, requirements
            )

        # Phase 2: Generate detailed content for each outline point
        print(f"[DEBUG] Phase 2: Generating content for {len(outline)} outline points")
        detailed_sections = []
        for i, point in enumerate(outline):
            print(f"[DEBUG] Generating content for point {i+1}: {point.get('title', '')[:30]}...")
            point_content = self._generate_point_content(
                section_title, point, sources_text, lang_instruction
            )
            detailed_sections.append({
                "title": point.get("title", f"Point {i+1}"),
                "content": point_content
            })

        # Phase 3: Integrate all content
        print(f"[DEBUG] Phase 3: Integrating {len(detailed_sections)} sections")
        final_content = self._integrate_content(
            section_title, section_description, detailed_sections, lang_instruction
        )

        # Collect source references
        source_refs = list(range(1, len(extracted_contents) + 1))

        return {
            "content": final_content,
            "summary": self._generate_section_summary(section_title, final_content, lang_instruction),
            "source_references": source_refs,
            "analysis_points": [s["title"] for s in detailed_sections],
            "information_gaps": [],
            "confidence_level": "medium" if len(extracted_contents) >= 2 else "low",
            "outline": outline,
        }

    def _generate_section_outline(
        self,
        section_title: str,
        section_description: str,
        sources_text: str,
        requirements: str,
        lang_instruction: str,
    ) -> List[Dict[str, str]]:
        """Generate a detailed outline for the section."""
        if self.language == "ja":
            prompt = f"""【セクション】{section_title}
【セクションの説明】{section_description}
【調査要件】{requirements if requirements else "包括的な分析"}

【情報源】
{sources_text[:6000]}

情報源に基づいて、このセクションの詳細なアウトラインを作成してください。
トピックを網羅する4〜6個の論点で構成すること。論点の並びは、読み手が自然に理解できる論理的な流れ（概要→詳細→分析など）にすること。

JSONで回答:
{{
    "outline": [
        {{"title": "論点のタイトル", "description": "この論点で扱う内容", "key_facts": ["含めるべき事実1", "事実2"]}}
    ]
}}

JSONのみを出力:"""
        else:
            prompt = f"""Section: {section_title}
Description: {section_description}
Requirements: {requirements if requirements else "Comprehensive analysis"}

Available Sources:
{sources_text[:6000]}

{lang_instruction}

Based on the available sources, create a detailed outline for this section.
The outline should have 4-6 key points that comprehensively cover the topic.

Return as JSON:
{{
    "outline": [
        {{"title": "Point title", "description": "What this point should cover", "key_facts": ["fact1", "fact2"]}}
    ]
}}

Output JSON only:"""

        try:
            response = self.llm.generate(prompt)
            content = response.content
            outline = []
            try:
                data = extract_json_from_response(content)
                outline = data.get("outline", [])
            except (json.JSONDecodeError, ValueError):
                pass
            if not outline:
                # Tolerate legacy bare-array responses
                try:
                    outline = extract_json_array_from_response(content)
                except (json.JSONDecodeError, ValueError):
                    pass
            if outline and len(outline) > 0:
                return outline
            print("[WARNING] Outline parsing produced no outline entries")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARNING] Outline parsing failed: {e}")

        # Fallback: create basic outline
        return [
            {"title": "Overview", "description": f"Overview of {section_title}", "key_facts": []},
            {"title": "Key Findings", "description": "Main findings from research", "key_facts": []},
            {"title": "Analysis", "description": "Analysis and interpretation", "key_facts": []},
        ]

    def _generate_point_content(
        self,
        section_title: str,
        point: Dict[str, Any],
        sources_text: str,
        lang_instruction: str,
    ) -> str:
        """Generate detailed content for a single outline point."""
        if self.language == "ja":
            prompt = f"""【セクション】{section_title}
【執筆する論点】{point.get('title', '')}
【論点の説明】{point.get('description', '')}
【含めるべき事実】{', '.join(point.get('key_facts', []))}

【情報源】
{sources_text[:5000]}

この論点について、400〜600字程度の本文を執筆してください。

【執筆ルール】
1. 常体（である調）で、翻訳調を避けた自然な日本語の段落として書く。文末表現が単調に繰り返されないよう変化をつける。
2. 情報源に基づく事実には [SOURCE N] の形式で出典番号を付ける（この表記は一字一句変えない）。
3. 具体的で情報量のある記述にする。情報源にない主張はしない。
4. 見出し・タイトル・箇条書きは付けず、本文の段落のみを書く。

本文のみを出力（JSON不要）:"""
        else:
            prompt = f"""Section: {section_title}
Point to elaborate: {point.get('title', '')}
Point description: {point.get('description', '')}
Key facts to include: {', '.join(point.get('key_facts', []))}

Available Sources:
{sources_text[:5000]}

{lang_instruction}

Write detailed content (300-500 characters) for this specific point.

IMPORTANT:
1. Use factual information from the sources
2. Cite sources as [SOURCE N] where appropriate
3. Be specific and informative
4. Do not include a title or heading, just the content paragraph(s)

Write the content directly (no JSON):"""

        try:
            response = self.llm.generate(prompt)
            content = response.content.strip()
            # Remove any JSON wrapping if present
            if content.startswith("{") or content.startswith("["):
                try:
                    data = json.loads(content)
                    if isinstance(data, dict) and "content" in data:
                        content = data["content"]
                except:
                    pass
            return content if content else f"{point.get('title', '')}: Information could not be generated."
        except Exception as e:
            print(f"[WARNING] Point content generation failed: {e}")
            return f"{point.get('title', '')}: Information could not be generated due to an error."

    def _integrate_content(
        self,
        section_title: str,
        section_description: str,
        detailed_sections: List[Dict[str, str]],
        lang_instruction: str,
    ) -> str:
        """Integrate multiple content sections into cohesive text."""
        sections_text = "\n\n".join(
            f"### {s['title']}\n{s['content']}"
            for s in detailed_sections
        )

        if self.language == "ja":
            prompt = f"""以下は同じセクション「{section_title}」のために書かれた下書きの断片です。これらを、一人の書き手が最初から書き下ろしたような、一貫した流れを持つ本文に統合してください。

【セクションの説明】{section_description}

【下書き断片】
{sections_text}

{JA_STYLE_GUIDE}

【統合ルール】
1. 事実と [SOURCE N] の出典表記はすべて正確に維持する（[SOURCE N] は一字一句変えない）。
2. 重複する記述は一度だけにまとめ、断片の順序も論理的な流れを優先して入れ替えてよい。
3. 「### 見出し」のような断片の区切りは残さず、段落のつながり（接続表現・指示の受け直し）で文章として流れを作る。
4. 断片にない情報を加えない。
5. 冒頭はセクションの主題を1〜2文で導入し、末尾は次の論点につながる形か、小括で締める。

統合後の本文のみを出力（JSON・見出し不要）:"""
        else:
            prompt = f"""Section: {section_title}
Description: {section_description}

Content parts to integrate:
{sections_text}

{lang_instruction}

Rewrite these draft fragments into one cohesive section that reads as if
written by a single author from scratch.

Requirements:
1. Maintain all factual information and citations [SOURCE N] exactly
2. Merge duplicated statements; you may reorder fragments for logical flow
3. Remove the "### heading" fragment markers; connect paragraphs with transitions
4. Do not add information not present in the original parts
5. Open with 1-2 sentences introducing the section's theme, close with a brief wrap-up

Write the integrated content directly (no JSON, no section title):"""

        try:
            response = self.llm.generate(prompt)
            content = response.content.strip()
            if content:
                return content
        except Exception as e:
            print(f"[WARNING] Content integration failed: {e}")

        # Fallback: just join the sections
        return "\n\n".join(s['content'] for s in detailed_sections)

    def _generate_section_summary(
        self,
        section_title: str,
        content: str,
        lang_instruction: str,
    ) -> str:
        """Generate a brief summary of the section content."""
        if self.language == "ja":
            prompt = f"""【セクション】{section_title}

【本文】
{content[:2000]}

このセクションの要点を、常体（である調）の自然な日本語で2〜3文に要約してください。
要約のみを出力（JSON不要）:"""
        else:
            prompt = f"""Section: {section_title}

Content:
{content[:2000]}

{lang_instruction}

Write a brief summary (2-3 sentences) of this section's key points.
Write the summary directly (no JSON):"""

        try:
            response = self.llm.generate(prompt)
            return response.content.strip()
        except Exception:
            return f"Summary of {section_title}"

    def evaluate_source_quality(
        self,
        source_url: str,
        source_content: str,
    ) -> Dict[str, Any]:
        """
        Evaluate the quality and reliability of a source.

        Args:
            source_url: URL of the source
            source_content: Content from the source

        Returns:
            Quality evaluation dictionary
        """
        prompt = f"""Evaluate the quality and reliability of this source:

URL: {source_url}

Content Sample:
{source_content[:3000]}

Analyze and return as JSON:
{{
    "source_type": "official/academic/news/blog/social/commercial/unknown",
    "reliability_indicators": {{
        "has_author": true/false,
        "has_date": true/false,
        "has_citations": true/false,
        "professional_tone": true/false
    }},
    "reliability_score": 0.0-1.0,
    "potential_biases": ["bias 1", "bias 2"],
    "recommended_use": "primary source/supporting source/verify with other sources/avoid",
    "evaluation_notes": "Brief notes on source quality"
}}"""

        response = self.llm.generate(prompt)

        try:
            return extract_json_from_response(response.content)
        except (json.JSONDecodeError, ValueError):
            pass

        return {
            "source_type": "unknown",
            "reliability_score": 0.5,
            "recommended_use": "verify with other sources",
            "evaluation_notes": "Automatic evaluation failed",
        }

    def extract_images_with_context(
        self,
        images: List[Dict[str, str]],
        page_content: str,
        section_context: str,
    ) -> List[Dict[str, Any]]:
        """
        Analyze and contextualize images from a page.

        Args:
            images: List of image info (src, alt, title)
            page_content: Content of the page
            section_context: Research section context

        Returns:
            List of images with relevance analysis
        """
        if not images:
            return []

        # Filter to potentially relevant images
        relevant_images = []
        for img in images[:10]:  # Limit to first 10 images
            src = img.get("src", "")
            alt = img.get("alt", "")

            # Skip small icons and UI elements
            if any(skip in src.lower() for skip in ["icon", "logo", "button", "avatar"]):
                continue

            relevant_images.append({
                "src": src,
                "alt": alt,
                "title": img.get("title", ""),
                "relevance": "unknown",
            })

        if not relevant_images:
            return []

        # Use LLM to evaluate image relevance
        images_text = "\n".join(
            f"{i+1}. ALT: {img['alt']}, Title: {img['title']}"
            for i, img in enumerate(relevant_images)
        )

        prompt = f"""Research Context: {section_context}

Page Content Summary:
{page_content[:1500]}

Images found:
{images_text}

Which images might be useful for illustrating this research topic?
Return as JSON array with relevance (high/medium/low/none):
[{{"index": 1, "relevance": "high/medium/low/none", "suggested_caption": "..."}}]"""

        response = self.llm.generate(prompt)

        try:
            evaluations = extract_json_array_from_response(response.content)

            for eval_item in evaluations:
                idx = eval_item.get("index", 0) - 1
                if 0 <= idx < len(relevant_images):
                    relevant_images[idx]["relevance"] = eval_item.get("relevance", "low")
                    relevant_images[idx]["suggested_caption"] = eval_item.get("suggested_caption", "")
        except (json.JSONDecodeError, ValueError):
            pass

        # Return only high/medium relevance images
        return [
            img for img in relevant_images
            if img.get("relevance") in ["high", "medium"]
        ]
