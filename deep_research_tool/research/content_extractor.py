"""
Content Extractor - Extract and process information from search results.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..utils.helpers import ResearchWarnings


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
        except (json.JSONDecodeError, ValueError) as _parse_err:
            ResearchWarnings.get_instance().add(
                ResearchWarnings.CRITICAL,
                "ContentExtractor",
                f"JSON parse failed for '{source_title[:60]}' ({source_url[:80]}). "
                f"Key points / quotes / relevance scoring lost; using raw text. "
                f"Error: {_parse_err}",
            )

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

        # Prepare fallback content in case JSON parsing fails or content is empty
        fallback_content = "\n\n".join(
            ec.processed_content for ec in extracted_contents if ec.processed_content
        )

        try:
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                result = json.loads(content[start:end])

                # Check if content is empty or too short, use fallback
                result_content = result.get("content", "")
                if not result_content or len(result_content.strip()) < 10:
                    result["content"] = fallback_content
                    if not result.get("summary"):
                        result["summary"] = "Content compiled from multiple sources"

                return result
        except (json.JSONDecodeError, ValueError) as _parse_err:
            ResearchWarnings.get_instance().add(
                ResearchWarnings.CRITICAL,
                "ContentExtractor",
                f"Section synthesis JSON parse failed for '{section_title}'. "
                f"Using concatenated raw content (confidence=low). "
                f"Analysis points and structured source references lost. "
                f"Error: {_parse_err}",
            )

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
            summary = f"""[SOURCE {i}] {ec.source_title}
URL: {ec.source_url}
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
        prompt = f"""Section: {section_title}
Description: {section_description}
Requirements: {requirements if requirements else "Comprehensive analysis"}

Available Sources:
{sources_text[:6000]}

{lang_instruction}

Based on the available sources, create a detailed outline for this section.
The outline should have 4-6 key points that comprehensively cover the topic.

Return as JSON array:
[
    {{"title": "Point title", "description": "What this point should cover", "key_facts": ["fact1", "fact2"]}},
    ...
]"""

        try:
            response = self.llm.generate(prompt)
            content = response.content
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                outline = json.loads(content[start:end])
                if outline and len(outline) > 0:
                    return outline
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

        prompt = f"""Section: {section_title}
Description: {section_description}

Content parts to integrate:
{sections_text}

{lang_instruction}

Integrate these content parts into a cohesive, well-structured section.

Requirements:
1. Maintain all factual information and citations [SOURCE N]
2. Ensure smooth transitions between topics
3. Remove any redundancy
4. Keep the section well-organized with clear flow
5. Do not add information not present in the original parts

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
