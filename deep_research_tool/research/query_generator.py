"""
Query Generator - Create and manage research queries and plans.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class TableOfContentsItem:
    """A single item in the table of contents."""
    section: str
    title: str
    description: str = ""
    subsections: List["TableOfContentsItem"] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed
    content: str = ""
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "section": self.section,
            "title": self.title,
            "description": self.description,
            "subsections": [s.to_dict() for s in self.subsections],
            "status": self.status,
            "content": self.content,
            "sources": self.sources,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableOfContentsItem":
        """Create from dictionary."""
        data = data.copy()
        if "subsections" in data:
            data["subsections"] = [
                cls.from_dict(s) for s in data["subsections"]
            ]
        return cls(**data)


@dataclass
class TableOfContents:
    """Table of contents for the research report."""
    title: str
    items: List[TableOfContentsItem] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "items": [item.to_dict() for item in self.items],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TableOfContents":
        """Create from dictionary."""
        return cls(
            title=data.get("title", ""),
            items=[TableOfContentsItem.from_dict(i) for i in data.get("items", [])],
            created_at=data.get("created_at", datetime.now().isoformat()),
        )

    def get_flat_sections(self) -> List[TableOfContentsItem]:
        """Get flattened list of all sections including subsections."""
        sections = []
        for item in self.items:
            sections.append(item)
            for sub in item.subsections:
                sections.append(sub)
        return sections

    def get_section(self, section_number: str) -> Optional[TableOfContentsItem]:
        """Get section by number."""
        for item in self.items:
            if item.section == section_number:
                return item
            for sub in item.subsections:
                if sub.section == section_number:
                    return sub
        return None

    def update_section_status(self, section_number: str, status: str) -> None:
        """Update status of a section."""
        section = self.get_section(section_number)
        if section:
            section.status = status

    def get_pending_sections(self) -> List[TableOfContentsItem]:
        """Get sections that are still pending."""
        return [s for s in self.get_flat_sections() if s.status == "pending"]

    def to_markdown(self) -> str:
        """Generate markdown representation of TOC."""
        lines = [f"# {self.title}\n"]
        for item in self.items:
            lines.append(f"## {item.section}. {item.title}")
            if item.description:
                lines.append(f"   {item.description}")
            for sub in item.subsections:
                lines.append(f"   ### {sub.section}. {sub.title}")
                if sub.description:
                    lines.append(f"      {sub.description}")
        return "\n".join(lines)


@dataclass
class ResearchPlan:
    """Complete research plan including TOC and queries."""
    title: str
    summary: str
    table_of_contents: TableOfContents
    search_queries: List[str] = field(default_factory=list)
    key_terms: List[str] = field(default_factory=list)
    suggested_sources: List[str] = field(default_factory=list)
    methodology_notes: str = ""
    estimated_complexity: str = "medium"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "summary": self.summary,
            "table_of_contents": self.table_of_contents.to_dict(),
            "search_queries": self.search_queries,
            "key_terms": self.key_terms,
            "suggested_sources": self.suggested_sources,
            "methodology_notes": self.methodology_notes,
            "estimated_complexity": self.estimated_complexity,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchPlan":
        """Create from dictionary."""
        return cls(
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            table_of_contents=TableOfContents.from_dict(
                data.get("table_of_contents", {"title": "", "items": []})
            ),
            search_queries=data.get("search_queries", []),
            key_terms=data.get("key_terms", []),
            suggested_sources=data.get("suggested_sources", []),
            methodology_notes=data.get("methodology_notes", ""),
            estimated_complexity=data.get("estimated_complexity", "medium"),
            created_at=data.get("created_at", datetime.now().isoformat()),
        )


class QueryGenerator:
    """Generate research queries and plans using LLM."""

    def __init__(self, llm_client, language: str = "ja"):
        """
        Initialize QueryGenerator.

        Args:
            llm_client: LLM API client instance
            language: Target language for queries and output
        """
        self.llm = llm_client
        self.language = language

    def create_research_plan(
        self,
        query: str,
        requirements: str = "",
        additional_context: str = "",
    ) -> ResearchPlan:
        """
        Create a complete research plan from query.

        Args:
            query: The main research query/topic
            requirements: Specific requirements for the research
            additional_context: Additional context or information

        Returns:
            Complete ResearchPlan object
        """
        lang_instruction = (
            "Respond entirely in Japanese." if self.language == "ja"
            else f"Respond in {self.language}."
        )

        system_prompt = f"""You are an expert research analyst. Your task is to create comprehensive research plans.
{lang_instruction}

When creating a research plan:
1. Break down the topic into logical sections and subsections
2. Create a detailed table of contents
3. Generate specific, targeted search queries
4. Identify key terms and concepts
5. Suggest types of sources to explore"""

        prompt = f"""Research Topic: {query}

Requirements: {requirements if requirements else "General comprehensive research"}

Additional Context: {additional_context if additional_context else "None"}

Create a detailed research plan. Return your response as a JSON object with this exact structure:
{{
    "title": "Report title",
    "summary": "Brief summary of research scope and objectives",
    "table_of_contents": [
        {{
            "section": "1",
            "title": "Section title",
            "description": "What this section covers",
            "subsections": [
                {{"section": "1.1", "title": "Subsection title", "description": "..."}}
            ]
        }}
    ],
    "search_queries": ["query1", "query2", ...],
    "key_terms": ["term1", "term2", ...],
    "suggested_sources": ["source type 1", "source type 2", ...],
    "methodology_notes": "Any methodological considerations",
    "estimated_complexity": "low/medium/high"
}}

Generate at least 5 main sections and 15 search queries.
Make search queries specific and actionable."""

        response = self.llm.generate(prompt, system_prompt=system_prompt)

        # Parse response
        try:
            # Try to extract JSON from response
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
            else:
                raise ValueError("No JSON found in response")

            # Build TOC
            toc_items = []
            for item_data in data.get("table_of_contents", []):
                subsections = [
                    TableOfContentsItem(
                        section=sub.get("section", ""),
                        title=sub.get("title", ""),
                        description=sub.get("description", ""),
                    )
                    for sub in item_data.get("subsections", [])
                ]
                toc_items.append(TableOfContentsItem(
                    section=item_data.get("section", ""),
                    title=item_data.get("title", ""),
                    description=item_data.get("description", ""),
                    subsections=subsections,
                ))

            toc = TableOfContents(
                title=data.get("title", query),
                items=toc_items,
            )

            return ResearchPlan(
                title=data.get("title", query),
                summary=data.get("summary", ""),
                table_of_contents=toc,
                search_queries=data.get("search_queries", []),
                key_terms=data.get("key_terms", []),
                suggested_sources=data.get("suggested_sources", []),
                methodology_notes=data.get("methodology_notes", ""),
                estimated_complexity=data.get("estimated_complexity", "medium"),
            )

        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: create basic plan from query
            return self._create_fallback_plan(query, str(e))

    def generate_follow_up_queries(
        self,
        section: TableOfContentsItem,
        gathered_info: str,
        gaps: List[str] = None,
    ) -> List[str]:
        """
        Generate follow-up queries for a section.

        Args:
            section: The section to generate queries for
            gathered_info: Summary of information already gathered
            gaps: Identified information gaps

        Returns:
            List of follow-up search queries
        """
        lang_instruction = (
            "Generate queries in Japanese." if self.language == "ja"
            else f"Generate queries in {self.language}."
        )

        gaps_text = "\n".join(f"- {gap}" for gap in gaps) if gaps else "Not yet identified"

        prompt = f"""Section: {section.section}. {section.title}
Description: {section.description}

Information Already Gathered:
{gathered_info[:2000] if gathered_info else "Initial research phase"}

Information Gaps:
{gaps_text}

{lang_instruction}

Generate 5-8 specific search queries that will:
1. Fill the identified gaps
2. Find supporting evidence
3. Explore different perspectives
4. Deepen understanding of key concepts

Return as a JSON array of strings only:
["query1", "query2", ...]"""

        response = self.llm.generate(prompt)

        try:
            content = response.content
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract any quoted strings
        import re
        queries = re.findall(r'"([^"]+)"', response.content)
        return queries[:8] if queries else [f"{section.title} detailed information"]

    def identify_gaps(
        self,
        section: TableOfContentsItem,
        content: str,
        requirements: str = "",
    ) -> List[str]:
        """
        Identify information gaps in section content.

        Args:
            section: The section being analyzed
            content: Current content of the section
            requirements: Original research requirements

        Returns:
            List of identified gaps
        """
        lang_instruction = (
            "Respond in Japanese." if self.language == "ja"
            else f"Respond in {self.language}."
        )

        prompt = f"""Section: {section.section}. {section.title}
Section Description: {section.description}

Current Content:
{content[:3000] if content else "No content yet"}

Research Requirements:
{requirements if requirements else "Comprehensive research"}

{lang_instruction}

Analyze the current content and identify what is missing or needs more detail.
Consider:
1. Key points that should be covered based on the section title
2. Evidence or data that would strengthen the content
3. Different perspectives that should be represented
4. Connections to other aspects of the research topic

Return as a JSON array of specific gaps:
["gap1", "gap2", ...]"""

        response = self.llm.generate(prompt)

        try:
            content = response.content
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

        return ["More detailed information needed", "Additional sources required"]

    def _create_fallback_plan(self, query: str, error: str) -> ResearchPlan:
        """Create a basic fallback plan when LLM parsing fails."""
        # Create basic TOC structure
        basic_sections = [
            TableOfContentsItem("1", "Introduction", "Background and context"),
            TableOfContentsItem("2", "Main Analysis", "Core analysis of the topic"),
            TableOfContentsItem("3", "Key Findings", "Important discoveries"),
            TableOfContentsItem("4", "Discussion", "Analysis and interpretation"),
            TableOfContentsItem("5", "Conclusion", "Summary and recommendations"),
        ]

        toc = TableOfContents(
            title=f"Research Report: {query}",
            items=basic_sections,
        )

        # Generate basic queries
        queries = [
            query,
            f"{query} overview",
            f"{query} analysis",
            f"{query} latest developments",
            f"{query} case studies",
        ]

        return ResearchPlan(
            title=f"Research Report: {query}",
            summary=f"Research analysis of: {query}",
            table_of_contents=toc,
            search_queries=queries,
            key_terms=[query],
            suggested_sources=["Web search", "News articles", "Academic sources"],
            methodology_notes=f"Fallback plan created due to: {error}",
            estimated_complexity="medium",
        )
