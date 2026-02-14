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

    # Generic section titles that indicate a too-plain ToC
    GENERIC_SECTION_TITLES_JA = [
        "はじめに", "緒言", "序論", "導入", "イントロダクション",
        "背景", "概要", "概説",
        "歴史", "経緯", "沿革",
        "考察", "ディスカッション", "議論",
        "結論", "まとめ", "おわりに", "結び", "総括",
        "今後の展望", "将来展望", "展望",
        "参考文献", "引用文献", "文献",
        "付録", "補遺", "appendix",
    ]

    GENERIC_SECTION_TITLES_EN = [
        "introduction", "background", "overview", "preface",
        "history", "historical background",
        "discussion", "analysis",
        "conclusion", "summary", "closing remarks",
        "future work", "future directions", "outlook",
        "references", "bibliography",
        "appendix", "appendices",
    ]

    def __init__(self, llm_client, language: str = "ja"):
        """
        Initialize QueryGenerator.

        Args:
            llm_client: LLM API client instance
            language: Target language for queries and output
        """
        self.llm = llm_client
        self.language = language

    def _is_generic_title(self, title: str) -> bool:
        """Check if a section title is too generic."""
        title_lower = title.lower().strip()

        # Check Japanese generic titles
        for generic in self.GENERIC_SECTION_TITLES_JA:
            if generic in title_lower or title_lower == generic:
                return True

        # Check English generic titles
        for generic in self.GENERIC_SECTION_TITLES_EN:
            if generic in title_lower or title_lower == generic:
                return True

        return False

    def _validate_toc_quality(self, toc: TableOfContents, query: str) -> tuple[bool, list[str]]:
        """
        Validate that the ToC is specific enough and not too generic.

        Args:
            toc: The table of contents to validate
            query: Original research query for context

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if not toc.items:
            issues.append("No sections generated")
            return False, issues

        # Check minimum number of sections
        if len(toc.items) < 3:
            issues.append(f"Too few main sections: {len(toc.items)} (minimum 3)")

        # Count generic titles
        generic_count = 0
        total_sections = 0
        sections_with_subsections = 0

        for item in toc.items:
            total_sections += 1

            if self._is_generic_title(item.title):
                generic_count += 1

            # Check for subsections
            if item.subsections and len(item.subsections) > 0:
                sections_with_subsections += 1
                for sub in item.subsections:
                    total_sections += 1
                    if self._is_generic_title(sub.title):
                        generic_count += 1

        # Calculate generic ratio
        generic_ratio = generic_count / total_sections if total_sections > 0 else 1.0

        # Too many generic titles (more than 50% is too generic)
        if generic_ratio > 0.5:
            issues.append(f"Too many generic section titles: {generic_count}/{total_sections} ({generic_ratio:.0%})")

        # Check that at least some sections have subsections
        if len(toc.items) >= 3 and sections_with_subsections < 2:
            issues.append(f"Not enough sections with subsections: {sections_with_subsections} (minimum 2)")

        # Check if titles are topic-specific (contain keywords from query)
        query_words = set(query.lower().split())
        # Remove common stop words
        stop_words = {"の", "を", "に", "は", "が", "と", "で", "a", "the", "of", "in", "to", "and", "for"}
        query_words = query_words - stop_words

        topic_relevant_sections = 0
        for item in toc.items:
            item_words = set(item.title.lower().split())
            if query_words & item_words:  # intersection
                topic_relevant_sections += 1

        if len(toc.items) >= 3 and topic_relevant_sections < 1:
            issues.append("Section titles don't reflect the research topic")

        is_valid = len(issues) == 0
        return is_valid, issues

    def create_research_plan(
        self,
        query: str,
        requirements: str = "",
        additional_context: str = "",
        max_retries: int = 2,
    ) -> ResearchPlan:
        """
        Create a complete research plan from query.

        Args:
            query: The main research query/topic
            requirements: Specific requirements for the research
            additional_context: Additional context or information
            max_retries: Maximum number of retries for ToC regeneration

        Returns:
            Complete ResearchPlan object
        """
        for attempt in range(max_retries + 1):
            plan = self._generate_research_plan_attempt(
                query=query,
                requirements=requirements,
                additional_context=additional_context,
                is_retry=attempt > 0,
                previous_issues=[] if attempt == 0 else issues,
            )

            if plan is None:
                continue

            # Validate ToC quality
            is_valid, issues = self._validate_toc_quality(plan.table_of_contents, query)

            if is_valid:
                return plan

            print(f"[QueryGenerator] ToC validation failed (attempt {attempt + 1}/{max_retries + 1}): {issues}")

        # If all retries failed, return the last generated plan with a warning
        print(f"[QueryGenerator] Warning: Could not generate ideal ToC after {max_retries + 1} attempts. Using best effort.")
        return plan if plan else self._create_fallback_plan(query, "All generation attempts failed")

    def _generate_research_plan_attempt(
        self,
        query: str,
        requirements: str,
        additional_context: str,
        is_retry: bool = False,
        previous_issues: List[str] = None,
    ) -> Optional[ResearchPlan]:
        """
        Single attempt at generating a research plan.

        Args:
            query: The main research query/topic
            requirements: Specific requirements for the research
            additional_context: Additional context or information
            is_retry: Whether this is a retry attempt
            previous_issues: Issues from previous attempt (for retry)

        Returns:
            ResearchPlan object or None if parsing failed
        """
        lang_instruction = (
            "Respond entirely in Japanese." if self.language == "ja"
            else f"Respond in {self.language}."
        )

        # Build retry-specific instructions
        retry_instruction = ""
        if is_retry and previous_issues:
            if self.language == "ja":
                issues_text = "\n".join(f"- {issue}" for issue in previous_issues)
                retry_instruction = f"""
【重要：前回の目次は以下の理由でリジェクトされました。これらの問題を解決してください】
{issues_text}

特に以下に注意してください：
- 「はじめに」「背景」「考察」「結論」などの一般的なセクション名は避け、調査テーマに具体的なセクション名にする
- 各主要セクションには必ず2-3個のサブセクションを含める
- セクション名には調査テーマのキーワードを含める
"""
            else:
                issues_text = "\n".join(f"- {issue}" for issue in previous_issues)
                retry_instruction = f"""
IMPORTANT: The previous table of contents was rejected for these reasons. Please fix:
{issues_text}

Pay special attention to:
- Avoid generic section names like "Introduction", "Background", "Discussion", "Conclusion"
- Use topic-specific section titles that reflect the research subject
- Include 2-3 subsections for each main section
- Section titles should contain keywords from the research topic
"""

        if self.language == "ja":
            system_prompt = f"""あなたは専門的なリサーチアナリストです。詳細で実践的な調査計画を作成してください。
{lang_instruction}

【重要な注意事項】
- 「はじめに」「緒言」「背景」「考察」「結論」「まとめ」などの一般的・学術論文的なセクション名は避けてください
- 代わりに、調査テーマに直接関連する具体的なセクション名を使用してください
- 例：「炭素繊維市場調査」の場合
  ✗ 悪い例：1.はじめに、2.炭素繊維の歴史、3.考察、4.結論
  ○ 良い例：1.炭素繊維の種類と製造プロセス、2.主要メーカーと市場シェア、3.用途別市場規模、4.価格動向と原材料、5.需要予測と成長ドライバー

調査計画作成のポイント：
1. テーマを論理的なセクションとサブセクションに分解する
2. 各メインセクションには必ず2-3個のサブセクションを含める
3. 具体的で実行可能な検索クエリを生成する
4. 重要な用語と概念を特定する"""
        else:
            system_prompt = f"""You are an expert research analyst. Create detailed and practical research plans.
{lang_instruction}

IMPORTANT GUIDELINES:
- AVOID generic/academic section names like "Introduction", "Background", "Discussion", "Conclusion", "Summary"
- Instead, use specific section names directly related to the research topic
- Example for "Carbon Fiber Market Research":
  ✗ Bad: 1. Introduction, 2. History of Carbon Fiber, 3. Discussion, 4. Conclusion
  ○ Good: 1. Types and Manufacturing Processes, 2. Key Manufacturers and Market Share, 3. Market Size by Application, 4. Pricing Trends and Raw Materials, 5. Demand Forecast and Growth Drivers

When creating a research plan:
1. Break down the topic into logical, topic-specific sections and subsections
2. Each main section MUST have 2-3 subsections
3. Generate specific, targeted search queries
4. Identify key terms and concepts"""

        if self.language == "ja":
            prompt = f"""調査テーマ: {query}

要件: {requirements if requirements else "包括的な調査"}

追加コンテキスト: {additional_context if additional_context else "なし"}
{retry_instruction}
詳細な調査計画を作成してください。以下のJSON形式で回答してください：
{{
    "title": "レポートタイトル（調査テーマを反映）",
    "summary": "調査の範囲と目的の概要",
    "table_of_contents": [
        {{
            "section": "1",
            "title": "具体的なセクションタイトル（「はじめに」等の一般名は不可）",
            "description": "このセクションでカバーする内容",
            "subsections": [
                {{"section": "1.1", "title": "サブセクションタイトル", "description": "..."}},
                {{"section": "1.2", "title": "サブセクションタイトル", "description": "..."}}
            ]
        }}
    ],
    "search_queries": ["クエリ1", "クエリ2", ...],
    "key_terms": ["用語1", "用語2", ...],
    "suggested_sources": ["情報源タイプ1", "情報源タイプ2", ...],
    "methodology_notes": "方法論に関する注意点",
    "estimated_complexity": "low/medium/high"
}}

必須条件：
- 最低5つのメインセクション
- 各メインセクションに2-3個のサブセクション
- 15個以上の検索クエリ
- 一般的なセクション名（はじめに、背景、考察、結論等）は使用しない"""
        else:
            prompt = f"""Research Topic: {query}

Requirements: {requirements if requirements else "General comprehensive research"}

Additional Context: {additional_context if additional_context else "None"}
{retry_instruction}
Create a detailed research plan. Return your response as a JSON object with this exact structure:
{{
    "title": "Report title (reflecting the research topic)",
    "summary": "Brief summary of research scope and objectives",
    "table_of_contents": [
        {{
            "section": "1",
            "title": "Specific section title (NO generic names like Introduction)",
            "description": "What this section covers",
            "subsections": [
                {{"section": "1.1", "title": "Subsection title", "description": "..."}},
                {{"section": "1.2", "title": "Subsection title", "description": "..."}}
            ]
        }}
    ],
    "search_queries": ["query1", "query2", ...],
    "key_terms": ["term1", "term2", ...],
    "suggested_sources": ["source type 1", "source type 2", ...],
    "methodology_notes": "Any methodological considerations",
    "estimated_complexity": "low/medium/high"
}}

Requirements:
- Generate at least 5 main sections
- Each main section MUST have 2-3 subsections
- Generate at least 15 search queries
- DO NOT use generic section names (Introduction, Background, Discussion, Conclusion)"""

        response = self.llm.generate(prompt, system_prompt=system_prompt)

        # Parse response
        try:
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
            print(f"[QueryGenerator] JSON parsing failed: {e}")
            return None

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
