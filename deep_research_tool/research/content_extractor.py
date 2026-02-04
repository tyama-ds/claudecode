"""
Content Extractor - Extract and process information from search results.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


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
        lang_instruction = (
            "Respond in Japanese." if self.language == "ja"
            else f"Respond in {self.language}."
        )

        # Truncate content if too long
        max_content_length = 8000
        truncated_content = raw_content[:max_content_length]
        if len(raw_content) > max_content_length:
            truncated_content += "\n... [content truncated]"

        prompt = f"""Source URL: {source_url}
Source Title: {source_title}

Research Context:
- Query: {research_query}
- Section: {section_context}

Source Content:
{truncated_content}

{lang_instruction}

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
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])

                return ExtractedContent(
                    source_url=source_url,
                    source_title=source_title,
                    raw_content=raw_content,
                    processed_content=data.get("processed_content", ""),
                    key_points=data.get("key_points", []),
                    quotes=data.get("quotes", []),
                    relevance_score=float(data.get("relevance_score", 0.5)),
                    extraction_notes=data.get("extraction_notes", ""),
                )
        except (json.JSONDecodeError, ValueError):
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
        lang_instruction = (
            "Write in Japanese." if self.language == "ja"
            else f"Write in {self.language}."
        )

        # Prepare source summaries
        source_summaries = []
        for i, ec in enumerate(extracted_contents, 1):
            summary = f"""
[SOURCE {i}] {ec.source_title}
URL: {ec.source_url}
Relevance: {ec.relevance_score:.2f}

Content:
{ec.processed_content[:2000]}

Key Points:
{chr(10).join('- ' + kp for kp in ec.key_points[:5])}
"""
            source_summaries.append(summary)

        sources_text = "\n---\n".join(source_summaries)

        prompt = f"""Section: {section_title}
Description: {section_description}

Research Requirements: {requirements if requirements else "Comprehensive analysis"}

Available Sources:
{sources_text}

{lang_instruction}

Synthesize the information from these sources into well-written section content.

IMPORTANT:
1. Clearly distinguish between:
   - Factual information from sources (cite as [SOURCE N])
   - Your own analysis or interpretation (mark as [ANALYSIS])
2. Do not make claims that aren't supported by the sources
3. Note any conflicting information between sources
4. Identify gaps where more information is needed

Return as JSON:
{{
    "content": "Full section content with citations",
    "summary": "Brief summary of key findings",
    "source_references": [1, 2, 3],
    "analysis_points": ["Your analytical insights"],
    "information_gaps": ["Areas needing more research"],
    "confidence_level": "high/medium/low"
}}"""

        response = self.llm.generate(prompt)

        try:
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback
        return {
            "content": "\n\n".join(ec.processed_content for ec in extracted_contents),
            "summary": "Content synthesized from multiple sources",
            "source_references": list(range(1, len(extracted_contents) + 1)),
            "analysis_points": [],
            "information_gaps": ["Synthesis failed, review needed"],
            "confidence_level": "low",
        }

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
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
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
            content = response.content
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                evaluations = json.loads(content[start:end])

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
