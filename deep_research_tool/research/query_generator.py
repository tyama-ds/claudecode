"""
Query Generator - Create and manage research queries and plans.
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

from deep_research_tool.utils.helpers import extract_json_from_response
from deep_research_tool.utils.japanese_text import (
    extract_content_words_set,
    title_contains_topic_keywords,
    is_generic_title_morphological,
)


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
        """Check if a section title is too generic.

        Uses morphological analysis (janome) for accurate base-form comparison
        of Japanese titles, with regex fallback.
        """
        # Direct string match first (fast path)
        title_lower = title.lower().strip()

        for generic in self.GENERIC_SECTION_TITLES_JA:
            if generic in title_lower or title_lower == generic:
                return True

        for generic in self.GENERIC_SECTION_TITLES_EN:
            if generic in title_lower or title_lower == generic:
                return True

        # Morphological base-form check (catches inflected forms)
        all_generics = self.GENERIC_SECTION_TITLES_JA + self.GENERIC_SECTION_TITLES_EN
        if is_generic_title_morphological(title, all_generics):
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
        # Use morphological analysis for proper Japanese keyword extraction
        query_words = extract_content_words_set(query)

        topic_relevant_sections = 0
        for item in toc.items:
            if title_contains_topic_keywords(item.title, query):
                topic_relevant_sections += 1

        if len(toc.items) >= 3 and topic_relevant_sections < 1:
            issues.append("Section titles don't reflect the research topic")

        is_valid = len(issues) == 0
        return is_valid, issues

    def _detect_user_toc_preference(self, query: str, requirements: str) -> tuple[bool, str]:
        """
        Detect if the user has specified ToC preferences or structure hints.

        Args:
            query: The research query
            requirements: The requirements string

        Returns:
            Tuple of (has_preference, preference_text)
        """
        combined_text = f"{query} {requirements}".lower()

        # Japanese indicators of ToC preference
        toc_indicators_ja = [
            "目次", "構成", "章立て", "セクション", "項目",
            "以下の構成", "以下の目次", "以下の章",
            "1.", "2.", "3.",  # Numbered items
            "第1章", "第2章", "第一章", "第二章",
            "・", "－",  # Bullet points
        ]

        # English indicators of ToC preference
        toc_indicators_en = [
            "table of contents", "toc", "structure", "outline",
            "sections", "chapters", "following structure",
            "include sections", "cover the following",
            "i.", "ii.", "iii.",  # Roman numerals
        ]

        # Check for indicators
        has_preference = False
        for indicator in toc_indicators_ja + toc_indicators_en:
            if indicator in combined_text:
                has_preference = True
                break

        # Also check if the text contains multiple numbered items (likely a user-defined structure)
        import re
        numbered_pattern = r'[1-9]\.\s*\S+'
        numbered_matches = re.findall(numbered_pattern, combined_text)
        if len(numbered_matches) >= 3:
            has_preference = True

        # Extract the preference text for passing to the prompt
        preference_text = ""
        if has_preference:
            # If requirements contains the preference, use it
            if any(ind in requirements.lower() for ind in toc_indicators_ja + toc_indicators_en):
                preference_text = requirements
            elif any(ind in query.lower() for ind in toc_indicators_ja + toc_indicators_en):
                preference_text = query

        return has_preference, preference_text

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
        # Check if user has specified ToC preferences
        has_user_toc_preference, user_toc_text = self._detect_user_toc_preference(query, requirements)

        if has_user_toc_preference:
            print(f"[QueryGenerator] User ToC preference detected, skipping validation")

        # Initialize issues before loop to avoid UnboundLocalError when plan is None on first attempt
        issues = []

        for attempt in range(max_retries + 1):
            plan = self._generate_research_plan_attempt(
                query=query,
                requirements=requirements,
                additional_context=additional_context,
                is_retry=attempt > 0,
                previous_issues=issues,
                user_toc_preference=user_toc_text if has_user_toc_preference else "",
            )

            if plan is None:
                continue

            # Split long compound queries into focused sub-queries
            original_count = len(plan.search_queries)
            plan.search_queries = self.split_complex_queries(plan.search_queries)
            if len(plan.search_queries) != original_count:
                print(f"[QueryGenerator] Split queries: {original_count} → {len(plan.search_queries)}")

            # Skip validation if user specified their own ToC preference
            if has_user_toc_preference:
                print(f"[QueryGenerator] Respecting user ToC preference, returning plan without strict validation")
                return plan

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
        user_toc_preference: str = "",
    ) -> Optional[ResearchPlan]:
        """
        Single attempt at generating a research plan.

        Args:
            query: The main research query/topic
            requirements: Specific requirements for the research
            additional_context: Additional context or information
            is_retry: Whether this is a retry attempt
            previous_issues: Issues from previous attempt (for retry)
            user_toc_preference: User-specified ToC preference text

        Returns:
            ResearchPlan object or None if parsing failed
        """
        lang_instruction = (
            "Respond entirely in Japanese." if self.language == "ja"
            else f"Respond in {self.language}."
        )

        # Build user ToC preference instruction
        user_toc_instruction = ""
        if user_toc_preference:
            if self.language == "ja":
                user_toc_instruction = f"""
【ユーザー指定の目次構成】
ユーザーが以下の目次構成または希望を指定しています。この構成を最優先で尊重してください：
{user_toc_preference}

上記のユーザー指定に従い、目次を作成してください。ユーザーの指定がある場合は、一般的なセクション名の制約よりもユーザーの希望を優先します。
"""
            else:
                user_toc_instruction = f"""
[USER-SPECIFIED TABLE OF CONTENTS]
The user has specified the following ToC structure or preferences. Respect this structure as the top priority:
{user_toc_preference}

Follow the user's specification above. When user preferences are specified, prioritize them over the generic section name constraints.
"""

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
{user_toc_instruction}{retry_instruction}
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
- 25個以上の検索クエリ
- 一般的なセクション名（はじめに、背景、考察、結論等）は使用しない

【検索クエリに関する重要ルール】
- 各クエリは1つの具体的な概念・指標・トピックに絞ること（1クエリ1トピック原則）
- 1クエリは30文字以内を目安とし、最大でも40文字以内にすること
- 複数の項目を「・」「、」「/」で列挙した長いクエリは禁止
- 例：
  ✗ 悪い例: "IR KPI 必須項目 年度別売上・CF部門比率・生産能力・CAPEX・主要顧客・用途別比率 テンプレート"
  ○ 良い例: "IR KPI 年度別売上 テンプレート", "IR CF部門比率 開示事例", "IR CAPEX 開示項目", "IR 主要顧客 開示例", "IR 用途別比率 チェックリスト"
  ✗ 悪い例: "炭素繊維 市場規模・メーカーシェア・用途・価格動向 分析"
  ○ 良い例: "炭素繊維 市場規模 2024", "炭素繊維 メーカー シェア", "炭素繊維 用途別 需要", "炭素繊維 価格動向"
- 焦点を絞ったクエリを数多く生成することで、検索精度が向上する"""
        else:
            prompt = f"""Research Topic: {query}

Requirements: {requirements if requirements else "General comprehensive research"}

Additional Context: {additional_context if additional_context else "None"}
{user_toc_instruction}{retry_instruction}
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
- Generate at least 25 search queries
- DO NOT use generic section names (Introduction, Background, Discussion, Conclusion)

CRITICAL RULES FOR SEARCH QUERIES:
- Each query MUST focus on ONE specific concept, metric, or topic (one-query-one-topic rule)
- Keep each query under 40 characters (aim for ~30 characters)
- NEVER combine multiple items with separators like ・ 、 / , in a single query
- Examples:
  ✗ Bad: "carbon fiber market size, manufacturer share, applications, price trends analysis"
  ○ Good: "carbon fiber market size 2024", "carbon fiber manufacturer share", "carbon fiber applications demand", "carbon fiber price trends"
- More focused queries yield better search results"""

        response = self.llm.generate(prompt, system_prompt=system_prompt)

        # Parse response
        try:
            content = response.content
            if not content or not content.strip():
                finish_reason = getattr(response, "finish_reason", "unknown")
                model = getattr(response, "model", "unknown")
                usage = getattr(response, "usage", {})
                print(f"[QueryGenerator] Empty response from LLM. "
                      f"finish_reason={finish_reason}, model={model}, usage={usage}")
                return None

            data = extract_json_from_response(content)
            return self._build_plan_from_data(data, query)

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[QueryGenerator] JSON parsing failed: {e}")
            return None

    @staticmethod
    def _build_plan_from_data(data: Dict[str, Any], query: str) -> ResearchPlan:
        """Build a ResearchPlan from the plan-JSON structure the LLM returns."""
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

    def revise_research_plan(
        self,
        plan: ResearchPlan,
        instructions: str,
        query: str,
        requirements: str = "",
    ) -> ResearchPlan:
        """
        Revise an existing research plan according to user instructions.

        Used by the plan-review step: the user can request changes to the
        title, table of contents, or search queries in natural language
        before the research loop starts.

        Args:
            plan: The current research plan
            instructions: User's revision instructions (natural language)
            query: The original research query/topic
            requirements: The original research requirements

        Returns:
            Revised ResearchPlan (raises ValueError if the LLM response
            cannot be parsed)
        """
        plan_json = json.dumps(
            {
                "title": plan.title,
                "summary": plan.summary,
                "table_of_contents": [i.to_dict() for i in plan.table_of_contents.items],
                "search_queries": plan.search_queries,
                "key_terms": plan.key_terms,
                "suggested_sources": plan.suggested_sources,
                "methodology_notes": plan.methodology_notes,
                "estimated_complexity": plan.estimated_complexity,
            },
            ensure_ascii=False,
            indent=1,
        )

        if self.language == "ja":
            prompt = f"""以下は調査計画です。ユーザーの修正指示に従って計画を修正してください。

調査テーマ: {query}
要件: {requirements if requirements else "包括的な調査"}

【現在の調査計画】
{plan_json}

【ユーザーの修正指示】
{instructions}

修正指示に該当しない部分は変更せず維持してください。目次（章立て）を変更した場合は、
それに合わせて検索クエリも整合するよう調整してください。
検索クエリは「1クエリ1トピック・40文字以内」のルールを守ってください。

修正後の計画全体を、現在の計画と同じJSON形式で出力してください（JSON以外の文章は不要）。"""
        else:
            prompt = f"""Below is a research plan. Revise it according to the user's instructions.

Research Topic: {query}
Requirements: {requirements if requirements else "General comprehensive research"}

[CURRENT PLAN]
{plan_json}

[USER'S REVISION INSTRUCTIONS]
{instructions}

Keep everything not covered by the instructions unchanged. If you change the
table of contents, adjust the search queries so they stay consistent with it.
Search queries must follow the one-query-one-topic rule and stay under 40 characters.

Output the complete revised plan in the same JSON format as the current plan
(JSON only, no other text)."""

        response = self.llm.generate(prompt)
        if not response or not response.content:
            raise ValueError("LLM returned empty response for plan revision")

        data = extract_json_from_response(response.content)
        revised = self._build_plan_from_data(data, query)

        # Guard against a degenerate revision (LLM dropped the ToC or queries)
        if not revised.table_of_contents.items:
            raise ValueError("Revised plan has no table of contents")
        if not revised.search_queries:
            revised.search_queries = list(plan.search_queries)
        else:
            revised.search_queries = self.split_complex_queries(revised.search_queries)
        return revised

    def generate_follow_up_queries(
        self,
        section: TableOfContentsItem,
        gathered_info: str,
        gaps: List[str] = None,
        research_topic: str = "",
    ) -> List[str]:
        """
        Generate follow-up queries for a section.

        Args:
            section: The section to generate queries for
            gathered_info: Summary of information already gathered
            gaps: Identified information gaps
            research_topic: Original research topic/query to keep queries on-topic

        Returns:
            List of follow-up search queries
        """
        lang_instruction = (
            "Generate queries in Japanese." if self.language == "ja"
            else f"Generate queries in {self.language}."
        )

        gaps_text = "\n".join(f"- {gap}" for gap in gaps) if gaps else "Not yet identified"

        # Build topic anchoring instruction
        topic_anchor = ""
        if research_topic:
            if self.language == "ja":
                topic_anchor = f"""
【重要：調査テーマの文脈を維持すること】
元の調査テーマ: {research_topic}
検索クエリは必ずこの調査テーマの文脈内で生成してください。
セクションタイトルだけを見て一般論的なクエリを生成してはいけません。
例：調査テーマが「炭素繊維の市場・技術調査」でセクションが「インタビュー・現地調査」の場合
  ✗ 悪い例: "リモートインタビューと対面調査の比較"（一般論）
  ○ 良い例: "炭素繊維メーカー 工場視察 調査手法"（テーマに紐づく）
"""
            else:
                topic_anchor = f"""
CRITICAL: Keep queries anchored to the original research topic.
Original Research Topic: {research_topic}
All queries MUST be generated within the context of this topic.
Do NOT generate generic queries based only on the section title.
Example: If the topic is "Carbon fiber market research" and the section is "Interview methodology":
  ✗ Bad: "remote interview vs in-person survey comparison" (too generic)
  ○ Good: "carbon fiber manufacturer site visit methodology" (topic-anchored)
"""

        prompt = f"""Section: {section.section}. {section.title}
Description: {section.description}
{topic_anchor}
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

IMPORTANT query rules:
- Each query must focus on ONE specific topic (one-query-one-topic)
- Keep each query under 40 characters
- Do NOT combine multiple items with ・ 、 / in a single query
- Every query MUST relate to the original research topic

Return as a JSON array of strings only:
["query1", "query2", ...]"""

        response = self.llm.generate(prompt)

        try:
            content = response.content
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                queries = json.loads(content[start:end])
                queries = self.split_complex_queries(queries)
                return self._anchor_queries_to_topic(queries, research_topic)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract any quoted strings
        import re
        queries = re.findall(r'"([^"]+)"', response.content)
        queries = queries[:8] if queries else [f"{section.title} detailed information"]
        queries = self.split_complex_queries(queries)
        return self._anchor_queries_to_topic(queries, research_topic)

    def _anchor_queries_to_topic(
        self,
        queries: List[str],
        research_topic: str,
    ) -> List[str]:
        """
        Ensure generated queries contain at least one keyword from the research topic.

        If a query has no overlap with the topic keywords, prepend the primary
        topic keyword to anchor it.

        Args:
            queries: Generated search queries
            research_topic: Original research topic

        Returns:
            Queries with topic-anchoring applied where needed
        """
        if not research_topic:
            return queries

        topic_words = extract_content_words_set(research_topic)
        if not topic_words:
            return queries

        # Pick the most representative keyword (longest content word)
        primary_keyword = max(topic_words, key=len)

        anchored = []
        for q in queries:
            q_words = extract_content_words_set(q)
            if q_words & topic_words:
                # Query already contains a topic keyword
                anchored.append(q)
            else:
                # Prepend primary keyword to anchor the query
                new_q = f"{primary_keyword} {q}"
                # Trim if too long
                if len(new_q) > 50:
                    new_q = new_q[:50]
                anchored.append(new_q)
                print(f"[QueryGenerator] Anchored drifting query: '{q}' → '{new_q}'")

        return anchored

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

    @staticmethod
    def split_complex_queries(queries: List[str]) -> List[str]:
        """
        Split long compound queries into focused sub-queries.

        Detects queries that combine multiple distinct topics (e.g. items
        separated by ・、/ or parenthesised lists) and splits them so that
        each resulting query targets a single concept.

        Rules:
        1. Queries <= 35 chars → keep as-is (already focused).
        2. Queries containing enumerated items (・/ 、separated, or
           parenthesised lists with ・) → extract common prefix/suffix and
           create one query per item.
        3. Remaining long queries (> 60 chars) with Japanese commas/
           connectors → attempt a simpler split.

        Args:
            queries: Original list of search queries.

        Returns:
            Expanded list with long compound queries split into shorter ones.
        """
        import re

        result: List[str] = []

        for query in queries:
            q = query.strip()

            # Rule 1: short enough → keep
            if len(q) <= 35:
                result.append(q)
                continue

            # Rule 2: detect parenthesised list  e.g. "...（A・B・C等）..."
            paren_match = re.search(r'[（(]([^）)]+)[）)]', q)
            if paren_match:
                inner = paren_match.group(1)
                # Remove trailing 等/など
                inner_clean = re.sub(r'[等など]$', '', inner).strip()
                items = re.split(r'[・/、，,]', inner_clean)
                items = [it.strip() for it in items if len(it.strip()) >= 2]

                if len(items) >= 3:
                    prefix = q[:paren_match.start()].strip()
                    suffix = q[paren_match.end():].strip()
                    # Remove trailing punctuation from prefix
                    prefix = re.sub(r'[：:、，,\s]+$', '', prefix)
                    for item in items:
                        sub = f"{prefix} {item}" if prefix else item
                        if suffix:
                            sub = f"{sub} {suffix}"
                        # Trim to reasonable length
                        if len(sub) <= 60:
                            result.append(sub)
                        else:
                            result.append(sub[:60])
                    continue

            # Rule 3: mid-dot separated list without parentheses
            #   e.g. "IR収集 年度別売上・CF比率・CAPEX チェックリスト"
            dot_items = re.split(r'[・]', q)
            if len(dot_items) >= 3:
                # Heuristic: first segment likely contains the prefix context,
                # last segment likely contains the suffix context.
                # Items in between are the enumerated concepts.
                first = dot_items[0].strip()
                last = dot_items[-1].strip()
                mid_items = [it.strip() for it in dot_items[1:-1] if len(it.strip()) >= 2]

                # Split first segment into prefix words and the first item
                # e.g. "IR収集テンプレート KPI 年度別売上" → prefix="IR収集テンプレート KPI", first_item="年度別売上"
                # We take everything up to the last space-separated token as prefix
                first_parts = re.split(r'\s+', first)
                if len(first_parts) > 1:
                    prefix = ' '.join(first_parts[:-1])
                    first_item = first_parts[-1]
                else:
                    prefix = ''
                    first_item = first

                all_items = [first_item] + mid_items + [last]

                for item in all_items:
                    sub = f"{prefix} {item}" if prefix else item
                    if len(sub) <= 60:
                        result.append(sub)
                    else:
                        result.append(sub[:60])
                continue

            # Rule 4: comma / 、 separated long query
            comma_items = re.split(r'[、，,]', q)
            if len(comma_items) >= 3 and len(q) > 50:
                for item in comma_items:
                    item = item.strip()
                    if len(item) >= 4:
                        result.append(item)
                continue

            # Default: keep original
            result.append(q)

        # Deduplicate while preserving order
        seen = set()
        deduped: List[str] = []
        for q in result:
            if q not in seen:
                seen.add(q)
                deduped.append(q)

        return deduped

    def _create_fallback_plan(self, query: str, error: str) -> ResearchPlan:
        """Create a basic fallback plan when LLM parsing fails."""
        # Use query-derived section names instead of generic ones
        short_query = query[:60] if len(query) > 60 else query
        basic_sections = [
            TableOfContentsItem("1", f"{short_query} - Current State and Context", f"Current state and context of {short_query}"),
            TableOfContentsItem("2", f"{short_query} - Technical Details and Mechanisms", f"Detailed technical analysis of {short_query}"),
            TableOfContentsItem("3", f"{short_query} - Real-World Examples and Evidence", f"Case studies and evidence related to {short_query}"),
            TableOfContentsItem("4", f"{short_query} - Challenges and Trade-offs", f"Challenges and trade-offs in {short_query}"),
            TableOfContentsItem("5", f"{short_query} - Future Directions and Implications", f"Future directions and implications for {short_query}"),
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
