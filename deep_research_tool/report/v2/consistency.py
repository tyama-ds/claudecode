"""
Consistency Checker - Analyze and fix inconsistencies across report chapters.

Version 2.0 feature for ensuring report-wide consistency.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum


class IssueType(str, Enum):
    """Types of consistency issues."""
    TERMINOLOGY = "terminology"         # 用語の不統一
    CONTRADICTION = "contradiction"     # 矛盾した記述
    STYLE = "style"                     # 文体の変動
    REFERENCE = "reference"             # 参照の不整合
    DUPLICATION = "duplication"         # 重複した説明
    MISSING_CONTEXT = "missing_context" # コンテキスト不足


class IssueSeverity(str, Enum):
    """Severity levels for issues."""
    ERROR = "error"         # 修正必須
    WARNING = "warning"     # 修正推奨
    INFO = "info"           # 参考情報


@dataclass
class ConsistencyIssue:
    """A single consistency issue found in the report."""
    issue_type: IssueType
    severity: IssueSeverity
    section: str                    # 問題のあるセクション
    description: str                # 問題の説明
    original_text: str = ""         # 問題のあるテキスト
    suggested_fix: str = ""         # 修正案
    related_sections: List[str] = field(default_factory=list)  # 関連セクション


@dataclass
class ConsistencyReport:
    """Report of all consistency issues found."""
    total_issues: int = 0
    errors: int = 0
    warnings: int = 0
    infos: int = 0
    issues: List[ConsistencyIssue] = field(default_factory=list)
    is_consistent: bool = True

    def add_issue(self, issue: ConsistencyIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)
        self.total_issues += 1
        if issue.severity == IssueSeverity.ERROR:
            self.errors += 1
            self.is_consistent = False
        elif issue.severity == IssueSeverity.WARNING:
            self.warnings += 1
        else:
            self.infos += 1

    def get_issues_by_type(self, issue_type: IssueType) -> List[ConsistencyIssue]:
        """Get all issues of a specific type."""
        return [i for i in self.issues if i.issue_type == issue_type]

    def get_issues_by_section(self, section: str) -> List[ConsistencyIssue]:
        """Get all issues in a specific section."""
        return [i for i in self.issues if i.section == section]


class ConsistencyChecker:
    """
    Check and fix consistency issues across report chapters.

    Analyzes:
    - Terminology consistency (same terms used consistently)
    - Factual contradictions (conflicting statements)
    - Style consistency (tone, formality)
    - Cross-reference validity
    - Duplicate content
    """

    def __init__(self, llm_client=None, language: str = "ja"):
        """
        Initialize ConsistencyChecker.

        Args:
            llm_client: Optional LLM client for advanced analysis
            language: Report language
        """
        self.llm = llm_client
        self.language = language

        # Common terminology variations to check
        self.terminology_patterns = {
            "ja": [
                # (パターン, 正規化形式, 説明)
                (r"CFRP|炭素繊維強化プラスチック|カーボンファイバー強化プラスチック", "CFRP（炭素繊維強化プラスチック）", "CFRP表記"),
                (r"AI|人工知能|エーアイ", "AI（人工知能）", "AI表記"),
                (r"EV|電気自動車|電動車", "EV（電気自動車）", "EV表記"),
                (r"IoT|アイオーティー|モノのインターネット", "IoT", "IoT表記"),
                (r"%|％|パーセント", "%", "パーセント表記"),
                (r"億円|億|おく円", "億円", "金額単位"),
            ],
            "en": [
                (r"percent|%|per cent", "%", "percent notation"),
                (r"billion|bn|B", "billion", "billion notation"),
                (r"million|mn|M", "million", "million notation"),
            ],
        }

        # Style indicators
        self.style_patterns = {
            "ja": {
                "formal": [r"である\。", r"であった\。", r"と考えられる\。"],
                "polite": [r"です\。", r"ます\。", r"でした\。", r"ました\。"],
                "casual": [r"だ\。", r"だった\。", r"んだ\。"],
            },
            "en": {
                "formal": [r"\bshall\b", r"\bwhilst\b", r"\bhence\b"],
                "informal": [r"\bcan't\b", r"\bwon't\b", r"\bdon't\b"],
            },
        }

    def check_all(
        self,
        chapters: Dict[str, str],
        context: Optional[Any] = None,
    ) -> ConsistencyReport:
        """
        Run all consistency checks on the chapters.

        Args:
            chapters: Dict of section_number -> chapter_content
            context: Optional ReportContext for additional checks

        Returns:
            ConsistencyReport with all found issues
        """
        report = ConsistencyReport()

        # Run individual checks
        report = self._merge_reports(report, self.check_terminology(chapters, context))
        report = self._merge_reports(report, self.check_style(chapters))
        report = self._merge_reports(report, self.check_duplication(chapters))

        if self.llm:
            report = self._merge_reports(report, self.check_contradictions_with_llm(chapters, context))

        return report

    def check_terminology(
        self,
        chapters: Dict[str, str],
        context: Optional[Any] = None,
    ) -> ConsistencyReport:
        """
        Check for terminology inconsistencies.

        Args:
            chapters: Dict of section_number -> chapter_content
            context: Optional ReportContext with glossary

        Returns:
            ConsistencyReport with terminology issues
        """
        report = ConsistencyReport()
        patterns = self.terminology_patterns.get(self.language, [])

        # Track term usage across chapters
        term_usage: Dict[str, Dict[str, List[str]]] = {}  # pattern -> {variant -> [sections]}

        for section, content in chapters.items():
            for pattern, preferred, description in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    if pattern not in term_usage:
                        term_usage[pattern] = {}
                    for match in matches:
                        if match not in term_usage[pattern]:
                            term_usage[pattern][match] = []
                        if section not in term_usage[pattern][match]:
                            term_usage[pattern][match].append(section)

        # Check for inconsistencies
        for pattern, variants in term_usage.items():
            if len(variants) > 1:
                # Multiple variants used
                _, preferred, description = next(
                    (p, pref, desc) for p, pref, desc in patterns if p == pattern
                )

                variant_info = ", ".join(
                    f"「{v}」({', '.join(secs)})" for v, secs in variants.items()
                )

                report.add_issue(ConsistencyIssue(
                    issue_type=IssueType.TERMINOLOGY,
                    severity=IssueSeverity.WARNING,
                    section="全体",
                    description=f"{description}が統一されていません: {variant_info}",
                    original_text=str(list(variants.keys())),
                    suggested_fix=f"「{preferred}」に統一することを推奨",
                    related_sections=list(set(
                        sec for secs in variants.values() for sec in secs
                    )),
                ))

        # Check against glossary if provided
        if context and hasattr(context, 'glossary'):
            for term_key, entry in context.glossary.items():
                preferred = entry.preferred_form
                aliases = entry.aliases

                for section, content in chapters.items():
                    # Check if aliases are used instead of preferred form
                    for alias in aliases:
                        if alias.lower() != preferred.lower():
                            if re.search(r'\b' + re.escape(alias) + r'\b', content, re.IGNORECASE):
                                if not re.search(r'\b' + re.escape(preferred) + r'\b', content, re.IGNORECASE):
                                    report.add_issue(ConsistencyIssue(
                                        issue_type=IssueType.TERMINOLOGY,
                                        severity=IssueSeverity.INFO,
                                        section=section,
                                        description=f"用語「{alias}」は「{preferred}」に統一することを推奨",
                                        original_text=alias,
                                        suggested_fix=preferred,
                                    ))

        return report

    def check_style(self, chapters: Dict[str, str]) -> ConsistencyReport:
        """
        Check for style inconsistencies.

        Args:
            chapters: Dict of section_number -> chapter_content

        Returns:
            ConsistencyReport with style issues
        """
        report = ConsistencyReport()
        style_patterns = self.style_patterns.get(self.language, {})

        if not style_patterns:
            return report

        # Count style indicators per chapter
        chapter_styles: Dict[str, Dict[str, int]] = {}

        for section, content in chapters.items():
            chapter_styles[section] = {}
            for style_name, patterns in style_patterns.items():
                count = sum(
                    len(re.findall(pattern, content))
                    for pattern in patterns
                )
                if count > 0:
                    chapter_styles[section][style_name] = count

        # Determine dominant style
        total_style_counts: Dict[str, int] = {}
        for section_styles in chapter_styles.values():
            for style, count in section_styles.items():
                total_style_counts[style] = total_style_counts.get(style, 0) + count

        if not total_style_counts:
            return report

        dominant_style = max(total_style_counts, key=total_style_counts.get)

        # Check for chapters that deviate from dominant style
        for section, section_styles in chapter_styles.items():
            if section_styles:
                section_dominant = max(section_styles, key=section_styles.get)
                if section_dominant != dominant_style:
                    report.add_issue(ConsistencyIssue(
                        issue_type=IssueType.STYLE,
                        severity=IssueSeverity.WARNING,
                        section=section,
                        description=f"文体が他の章と異なります。全体: {dominant_style}, この章: {section_dominant}",
                        suggested_fix=f"文体を「{dominant_style}」に統一することを推奨",
                    ))

        return report

    def check_duplication(self, chapters: Dict[str, str]) -> ConsistencyReport:
        """
        Check for duplicate content across chapters.

        Args:
            chapters: Dict of section_number -> chapter_content

        Returns:
            ConsistencyReport with duplication issues
        """
        report = ConsistencyReport()

        # Extract sentences from each chapter using morphological analysis
        from deep_research_tool.utils.japanese_text import split_sentences
        chapter_sentences: Dict[str, List[str]] = {}
        for section, content in chapters.items():
            sentences = split_sentences(content, min_length=30)
            # Normalize for comparison
            chapter_sentences[section] = [
                s.strip().lower() for s in sentences
            ]

        # Check for duplicates
        sections = list(chapter_sentences.keys())
        for i, section1 in enumerate(sections):
            for section2 in sections[i + 1:]:
                for sent1 in chapter_sentences[section1]:
                    for sent2 in chapter_sentences[section2]:
                        # Check for high similarity (exact or near-exact)
                        if sent1 == sent2:
                            report.add_issue(ConsistencyIssue(
                                issue_type=IssueType.DUPLICATION,
                                severity=IssueSeverity.INFO,
                                section=section1,
                                description=f"セクション{section2}と重複した内容があります",
                                original_text=sent1[:100] + "...",
                                suggested_fix="一方を削除するか、参照に変更することを検討",
                                related_sections=[section2],
                            ))

        return report

    def check_contradictions_with_llm(
        self,
        chapters: Dict[str, str],
        context: Optional[Any] = None,
    ) -> ConsistencyReport:
        """
        Use LLM to check for factual contradictions.

        Args:
            chapters: Dict of section_number -> chapter_content
            context: Optional ReportContext with established facts

        Returns:
            ConsistencyReport with contradiction issues
        """
        report = ConsistencyReport()

        if not self.llm:
            return report

        # Prepare chapter summaries for analysis
        chapter_summaries = []
        for section, content in chapters.items():
            # Take first 1000 chars as summary
            summary = content[:1000] if len(content) > 1000 else content
            chapter_summaries.append(f"【{section}】\n{summary}")

        combined = "\n\n".join(chapter_summaries)

        if self.language == "ja":
            prompt = f"""以下の報告書の各章を分析し、矛盾や不整合を検出してください。

{combined}

以下の観点でチェックしてください：
1. 数値データの矛盾（異なる章で異なる数値が記載されている）
2. 事実の矛盾（ある章の主張が別の章と矛盾）
3. 時系列の矛盾（イベントの順序が不整合）

矛盾が見つかった場合は、以下の形式でJSON配列として出力してください：
[
  {{"section1": "セクション番号", "section2": "セクション番号", "description": "矛盾の説明", "severity": "error/warning"}}
]

矛盾がない場合は空の配列 [] を出力してください。
JSONのみを出力:"""
        else:
            prompt = f"""Analyze the following report chapters and detect any contradictions or inconsistencies.

{combined}

Check for:
1. Numerical contradictions (different numbers in different chapters)
2. Factual contradictions (claims that conflict between chapters)
3. Timeline contradictions (inconsistent event ordering)

If contradictions are found, output as JSON array:
[
  {{"section1": "section number", "section2": "section number", "description": "description", "severity": "error/warning"}}
]

Output empty array [] if no contradictions found.
Output only JSON:"""

        try:
            response = self.llm.generate(prompt)
            import json
            content = response.content.strip()
            if "```" in content:
                content = content.split("```")[1].split("```")[0]
                if content.startswith("json"):
                    content = content[4:]

            contradictions = json.loads(content)

            for c in contradictions:
                severity = IssueSeverity.ERROR if c.get("severity") == "error" else IssueSeverity.WARNING
                report.add_issue(ConsistencyIssue(
                    issue_type=IssueType.CONTRADICTION,
                    severity=severity,
                    section=c.get("section1", ""),
                    description=c.get("description", ""),
                    related_sections=[c.get("section2", "")],
                ))

        except Exception as e:
            print(f"[ConsistencyChecker] LLM contradiction check failed: {e}")

        return report

    def suggest_fixes(
        self,
        report: ConsistencyReport,
        chapters: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Generate suggested fixes for all issues.

        Args:
            report: ConsistencyReport with issues
            chapters: Original chapter content

        Returns:
            Dict of section_number -> suggested_fixed_content
        """
        fixed_chapters = dict(chapters)

        # Apply terminology fixes
        for issue in report.get_issues_by_type(IssueType.TERMINOLOGY):
            if issue.original_text and issue.suggested_fix:
                for section in [issue.section] + issue.related_sections:
                    if section in fixed_chapters and section != "全体":
                        # Simple replacement (could be more sophisticated)
                        fixed_chapters[section] = fixed_chapters[section].replace(
                            issue.original_text,
                            issue.suggested_fix,
                        )

        return fixed_chapters

    def _merge_reports(
        self,
        base: ConsistencyReport,
        new: ConsistencyReport,
    ) -> ConsistencyReport:
        """Merge two consistency reports."""
        for issue in new.issues:
            base.add_issue(issue)
        return base

    def generate_summary(self, report: ConsistencyReport) -> str:
        """Generate a human-readable summary of the consistency report."""
        if self.language == "ja":
            lines = [
                f"## 一貫性チェック結果",
                f"",
                f"- 総問題数: {report.total_issues}",
                f"- エラー: {report.errors}",
                f"- 警告: {report.warnings}",
                f"- 情報: {report.infos}",
                f"- 一貫性: {'OK' if report.is_consistent else 'NG'}",
                f"",
            ]

            if report.issues:
                lines.append("### 検出された問題")
                lines.append("")
                for i, issue in enumerate(report.issues, 1):
                    severity_mark = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}
                    mark = severity_mark.get(issue.severity.value, "")
                    lines.append(f"{i}. {mark} [{issue.issue_type.value}] {issue.description}")
                    if issue.suggested_fix:
                        lines.append(f"   → 修正案: {issue.suggested_fix}")
        else:
            lines = [
                f"## Consistency Check Results",
                f"",
                f"- Total Issues: {report.total_issues}",
                f"- Errors: {report.errors}",
                f"- Warnings: {report.warnings}",
                f"- Info: {report.infos}",
                f"- Consistent: {'Yes' if report.is_consistent else 'No'}",
                f"",
            ]

            if report.issues:
                lines.append("### Detected Issues")
                lines.append("")
                for i, issue in enumerate(report.issues, 1):
                    lines.append(f"{i}. [{issue.severity.value}] [{issue.issue_type.value}] {issue.description}")
                    if issue.suggested_fix:
                        lines.append(f"   → Suggested fix: {issue.suggested_fix}")

        return "\n".join(lines)
