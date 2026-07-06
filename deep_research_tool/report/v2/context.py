"""
Report Context - Global context maintained across all chapters for consistency.

Version 2.0 feature for ensuring report-wide consistency.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


_JA_NATURALNESS_RULES = """【自然な日本語のための共通ルール】
- 翻訳調を避ける：「〜することが可能である」→「〜できる」、「〜という事実」→「〜こと」。「〜が行われた」の乱用を避け、能動態を基本にする。
- 文末表現の単調な繰り返しを避け、文の長短に変化をつける（一文はおおむね60字以内を目安）。
- 「一方」「さらに」「このため」「もっとも」などの接続表現で段落内の論理をつなぐ。ただし各段落で多用しない。
- 箇条書きに頼らず段落の文章として論じる。箇条書きは並列的な列挙が明確に読みやすい場合のみ。
- 見出しの直後にいきなり箇条書きを置かず、まず1〜2文で導入する。
- 同じ語・同じ言い回しを近い場所で繰り返さない。指示語（「それ」「これ」）が何を指すか曖昧にしない。
- 数値・固有名詞・単位は情報源の表記を正確に保つ。"""


class WritingStyle(str, Enum):
    """Writing style for the report."""
    FORMAL = "formal"           # 学術的・フォーマル
    BUSINESS = "business"       # ビジネスレポート
    TECHNICAL = "technical"     # 技術文書
    EXECUTIVE = "executive"     # エグゼクティブサマリー向け
    CASUAL = "casual"           # カジュアル・ブログ調


class TargetAudience(str, Enum):
    """Target audience for the report."""
    EXPERT = "expert"           # 専門家・研究者
    BUSINESS = "business"       # 経営層・ビジネスパーソン
    ENGINEER = "engineer"       # エンジニア・技術者
    GENERAL = "general"         # 一般読者
    STUDENT = "student"         # 学生・学習者


@dataclass
class GlossaryEntry:
    """A single glossary entry."""
    term: str                           # 用語
    definition: str                     # 定義
    aliases: List[str] = field(default_factory=list)  # 別名・略称
    preferred_form: str = ""            # 優先表記
    first_appearance_section: str = ""  # 最初に登場したセクション
    usage_count: int = 0                # 使用回数

    def __post_init__(self):
        if not self.preferred_form:
            self.preferred_form = self.term


@dataclass
class EstablishedFact:
    """A fact established in the report that should remain consistent."""
    fact: str                           # 事実の内容
    source_section: str                 # 出典セクション
    source_url: str = ""                # 情報源URL
    confidence: float = 1.0             # 信頼度 (0.0-1.0)
    related_terms: List[str] = field(default_factory=list)


@dataclass
class ChapterSummary:
    """Summary of a completed chapter."""
    section_number: str
    section_title: str
    summary: str                        # 要約（200-500文字）
    key_points: List[str]               # 主要ポイント
    terms_introduced: List[str]         # 導入された用語
    facts_established: List[str]        # 確立された事実
    word_count: int = 0
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CrossReference:
    """Cross-reference within the report."""
    reference_id: str                   # "図1", "表2", etc.
    description: str                    # "市場規模推移グラフ"
    section: str                        # 所属セクション
    reference_type: str = "figure"      # figure, table, section, citation


@dataclass
class ReportContext:
    """
    Global context maintained across all chapters for consistency.

    This is the central data structure for V2 report generation that ensures
    consistency in terminology, facts, style, and cross-references.
    """

    # Basic information
    research_topic: str
    report_title: str = ""
    language: str = "ja"

    # Style settings
    writing_style: WritingStyle = WritingStyle.BUSINESS
    target_audience: TargetAudience = TargetAudience.BUSINESS
    technical_level: int = 3            # 1-5 (1=基礎, 5=高度専門)

    # Terminology management
    glossary: Dict[str, GlossaryEntry] = field(default_factory=dict)

    # Fact tracking (to prevent contradictions)
    established_facts: List[EstablishedFact] = field(default_factory=list)

    # Chapter continuity
    chapter_summaries: List[ChapterSummary] = field(default_factory=list)

    # Cross-references
    cross_references: Dict[str, CrossReference] = field(default_factory=dict)

    # Statistics
    total_word_count: int = 0
    total_citations: int = 0

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_glossary_term(
        self,
        term: str,
        definition: str,
        aliases: List[str] = None,
        preferred_form: str = "",
        section: str = "",
    ) -> None:
        """Add or update a glossary term."""
        key = term.lower()
        if key in self.glossary:
            # Update existing entry
            entry = self.glossary[key]
            entry.usage_count += 1
            if aliases:
                entry.aliases = list(set(entry.aliases + aliases))
        else:
            # Create new entry
            self.glossary[key] = GlossaryEntry(
                term=term,
                definition=definition,
                aliases=aliases or [],
                preferred_form=preferred_form or term,
                first_appearance_section=section,
                usage_count=1,
            )
        self._update_timestamp()

    def get_preferred_term(self, term: str) -> str:
        """Get the preferred form of a term."""
        key = term.lower()
        if key in self.glossary:
            return self.glossary[key].preferred_form
        # Check aliases
        for entry in self.glossary.values():
            if term.lower() in [a.lower() for a in entry.aliases]:
                return entry.preferred_form
        return term

    def add_established_fact(
        self,
        fact: str,
        source_section: str,
        source_url: str = "",
        confidence: float = 1.0,
        related_terms: List[str] = None,
    ) -> None:
        """Add an established fact."""
        self.established_facts.append(EstablishedFact(
            fact=fact,
            source_section=source_section,
            source_url=source_url,
            confidence=confidence,
            related_terms=related_terms or [],
        ))
        self._update_timestamp()

    def add_chapter_summary(
        self,
        section_number: str,
        section_title: str,
        summary: str,
        key_points: List[str],
        terms_introduced: List[str] = None,
        facts_established: List[str] = None,
        word_count: int = 0,
    ) -> None:
        """Add a summary for a completed chapter."""
        self.chapter_summaries.append(ChapterSummary(
            section_number=section_number,
            section_title=section_title,
            summary=summary,
            key_points=key_points,
            terms_introduced=terms_introduced or [],
            facts_established=facts_established or [],
            word_count=word_count,
        ))
        self.total_word_count += word_count
        self._update_timestamp()

    def get_previous_summaries(self, max_count: int = 3) -> List[ChapterSummary]:
        """Get the most recent chapter summaries."""
        return self.chapter_summaries[-max_count:] if self.chapter_summaries else []

    def add_cross_reference(
        self,
        reference_id: str,
        description: str,
        section: str,
        reference_type: str = "figure",
    ) -> None:
        """Add a cross-reference."""
        self.cross_references[reference_id] = CrossReference(
            reference_id=reference_id,
            description=description,
            section=section,
            reference_type=reference_type,
        )
        self._update_timestamp()

    def get_style_instructions(self) -> str:
        """Generate style instructions for chapter generation prompts."""
        prose_rule = "本文は散文パラグラフで記述し、箇条書きは比較・手順など列挙が自然な場合に限定する。"
        style_map = {
            WritingStyle.FORMAL: (
                "学術的な報告書の文体。「である」調で統一。客観的・分析的に論じ、"
                "感情的・主観的な形容（「素晴らしい」「驚くべき」等）は使わない。"
                "断定できない事項は「〜と考えられる」「〜が示唆される」と書く。"
                f"{prose_rule}"
            ),
            WritingStyle.BUSINESS: (
                "ビジネス報告書の文体。「です・ます」調で統一。各段落は結論を先に述べ、"
                "根拠を後に続ける。「〜について見ていきましょう」のような話し言葉・講義調は使わない。"
                f"{prose_rule}"
            ),
            WritingStyle.TECHNICAL: (
                "技術文書の文体。「である」調。条件・数値・手順を正確に記述し、"
                "「かなり」「非常に」などの曖昧な程度表現を避ける。専門用語は初出時に簡潔に定義する。"
                f"{prose_rule}"
            ),
            WritingStyle.EXECUTIVE: (
                "経営層向け。「です・ます」調。冒頭に要点、続けて根拠と示唆。"
                "1段落は短く（2〜4文）、意思決定に関わる含意を明示する。"
                f"{prose_rule}"
            ),
            WritingStyle.CASUAL: (
                "平易で読みやすい書き言葉。「です・ます」調。専門用語は日常語に言い換えるか、"
                "必ず説明を添える。ただし砕けすぎた表現・絵文字は使わない。"
                f"{prose_rule}"
            ),
        }

        audience_map = {
            TargetAudience.EXPERT: "専門家向け。高度な専門用語の使用可。",
            TargetAudience.BUSINESS: "ビジネスパーソン向け。専門用語は必要に応じて説明。",
            TargetAudience.ENGINEER: "技術者向け。技術的詳細を含める。",
            TargetAudience.GENERAL: "一般読者向け。専門用語は避けるか、必ず説明を付ける。",
            TargetAudience.STUDENT: "学習者向け。基礎から丁寧に説明。",
        }

        level_desc = ["基礎的", "入門的", "標準的", "専門的", "高度専門的"][self.technical_level - 1]

        if self.language == "ja":
            return f"""【文体・スタイル指示】
- 文体: {style_map.get(self.writing_style, "")}
- 対象読者: {audience_map.get(self.target_audience, "")}
- 技術レベル: {level_desc}（レベル{self.technical_level}/5）

{_JA_NATURALNESS_RULES}"""
        else:
            return f"""[Writing Style Instructions]
- Style: {self.writing_style.value}
- Target Audience: {self.target_audience.value}
- Technical Level: {self.technical_level}/5"""

    def get_glossary_instructions(self) -> str:
        """Generate glossary instructions for chapter generation prompts."""
        if not self.glossary:
            return ""

        terms = []
        for entry in list(self.glossary.values())[:20]:  # Limit to 20 terms
            if entry.aliases:
                terms.append(f"- {entry.preferred_form}（{', '.join(entry.aliases)}）: {entry.definition}")
            else:
                terms.append(f"- {entry.preferred_form}: {entry.definition}")

        if self.language == "ja":
            return f"""【用語統一ルール】
以下の用語は統一された表記を使用してください：
{chr(10).join(terms)}"""
        else:
            return f"""[Terminology Consistency Rules]
Use the following standardized terms:
{chr(10).join(terms)}"""

    def get_previous_context(self) -> str:
        """Generate context from previous chapters for continuity."""
        if not self.chapter_summaries:
            return ""

        summaries = self.get_previous_summaries(3)
        summary_texts = []
        for s in summaries:
            points = ", ".join(s.key_points[:3]) if s.key_points else ""
            summary_texts.append(f"- {s.section_number}. {s.section_title}: {s.summary[:200]}... (要点: {points})")

        if self.language == "ja":
            return f"""【前章までの内容】
{chr(10).join(summary_texts)}

上記の内容との一貫性を保ち、矛盾しないようにしてください。"""
        else:
            return f"""[Previous Chapters Summary]
{chr(10).join(summary_texts)}

Maintain consistency with the above content and avoid contradictions."""

    def get_established_facts_context(self) -> str:
        """Generate context from established facts."""
        if not self.established_facts:
            return ""

        facts = [f"- {f.fact}" for f in self.established_facts[-10:]]  # Last 10 facts

        if self.language == "ja":
            return f"""【確立された事実（矛盾しないこと）】
{chr(10).join(facts)}"""
        else:
            return f"""[Established Facts (Do Not Contradict)]
{chr(10).join(facts)}"""

    def get_full_context_prompt(self) -> str:
        """Generate the full context prompt for chapter generation."""
        parts = [
            f"調査テーマ: {self.research_topic}" if self.language == "ja" else f"Research Topic: {self.research_topic}",
            self.get_style_instructions(),
            self.get_glossary_instructions(),
            self.get_previous_context(),
            self.get_established_facts_context(),
        ]
        return "\n\n".join(p for p in parts if p)

    def _update_timestamp(self) -> None:
        """Update the last_updated timestamp."""
        self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "research_topic": self.research_topic,
            "report_title": self.report_title,
            "language": self.language,
            "writing_style": self.writing_style.value,
            "target_audience": self.target_audience.value,
            "technical_level": self.technical_level,
            "glossary": {k: {
                "term": v.term,
                "definition": v.definition,
                "aliases": v.aliases,
                "preferred_form": v.preferred_form,
                "first_appearance_section": v.first_appearance_section,
                "usage_count": v.usage_count,
            } for k, v in self.glossary.items()},
            "established_facts": [{
                "fact": f.fact,
                "source_section": f.source_section,
                "source_url": f.source_url,
                "confidence": f.confidence,
                "related_terms": f.related_terms,
            } for f in self.established_facts],
            "chapter_summaries": [{
                "section_number": s.section_number,
                "section_title": s.section_title,
                "summary": s.summary,
                "key_points": s.key_points,
                "terms_introduced": s.terms_introduced,
                "facts_established": s.facts_established,
                "word_count": s.word_count,
                "generated_at": s.generated_at,
            } for s in self.chapter_summaries],
            "cross_references": {k: {
                "reference_id": v.reference_id,
                "description": v.description,
                "section": v.section,
                "reference_type": v.reference_type,
            } for k, v in self.cross_references.items()},
            "total_word_count": self.total_word_count,
            "total_citations": self.total_citations,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportContext":
        """Create from dictionary."""
        ctx = cls(
            research_topic=data.get("research_topic", ""),
            report_title=data.get("report_title", ""),
            language=data.get("language", "ja"),
            writing_style=WritingStyle(data.get("writing_style", "business")),
            target_audience=TargetAudience(data.get("target_audience", "business")),
            technical_level=data.get("technical_level", 3),
            total_word_count=data.get("total_word_count", 0),
            total_citations=data.get("total_citations", 0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_updated=data.get("last_updated", datetime.now().isoformat()),
        )

        # Restore glossary
        for k, v in data.get("glossary", {}).items():
            ctx.glossary[k] = GlossaryEntry(**v)

        # Restore established facts
        for f in data.get("established_facts", []):
            ctx.established_facts.append(EstablishedFact(**f))

        # Restore chapter summaries
        for s in data.get("chapter_summaries", []):
            ctx.chapter_summaries.append(ChapterSummary(**s))

        # Restore cross references
        for k, v in data.get("cross_references", {}).items():
            ctx.cross_references[k] = CrossReference(**v)

        return ctx
