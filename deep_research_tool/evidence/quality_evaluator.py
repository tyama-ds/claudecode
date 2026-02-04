"""
Quality Evaluator - Evaluate and categorize information quality from web sources.
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse

from .locker import (
    Evidence,
    QualityCategory,
    SourceType,
    QualityIndicators,
    EvidenceLocker,
)


# Domain authority patterns for automatic categorization
AUTHORITATIVE_DOMAINS = [
    # Government domains
    r"\.gov$", r"\.gov\.[a-z]{2}$", r"\.go\.[a-z]{2}$",
    # Academic domains
    r"\.edu$", r"\.edu\.[a-z]{2}$", r"\.ac\.[a-z]{2}$",
    # International organizations
    r"\.int$", r"who\.int", r"un\.org", r"worldbank\.org",
]

HIGH_QUALITY_DOMAINS = [
    # Major news outlets
    r"reuters\.com", r"apnews\.com", r"bbc\.(com|co\.uk)",
    r"nytimes\.com", r"washingtonpost\.com", r"theguardian\.com",
    r"economist\.com", r"ft\.com", r"wsj\.com",
    r"nhk\.or\.jp", r"asahi\.com", r"nikkei\.com",
    # Research/Academic publishers
    r"nature\.com", r"science\.org", r"sciencedirect\.com",
    r"springer\.com", r"wiley\.com", r"arxiv\.org",
    r"pubmed\.ncbi\.nlm\.nih\.gov", r"ncbi\.nlm\.nih\.gov",
    # Tech/Industry authorities
    r"microsoft\.com", r"google\.com", r"apple\.com",
    r"github\.com", r"stackoverflow\.com",
]

MEDIUM_QUALITY_DOMAINS = [
    r"medium\.com", r"substack\.com",
    r"techcrunch\.com", r"wired\.com", r"arstechnica\.com",
    r"dev\.to", r"hashnode\.com",
]

LOW_QUALITY_PATTERNS = [
    r"forum", r"reddit\.com", r"quora\.com",
    r"twitter\.com", r"x\.com", r"facebook\.com",
    r"instagram\.com", r"tiktok\.com",
    r"blogspot\.com", r"wordpress\.com",
]


@dataclass
class QualityEvaluation:
    """Result of quality evaluation."""
    quality_category: QualityCategory
    source_type: SourceType
    quality_indicators: QualityIndicators
    confidence: float  # 0-1, how confident we are in this evaluation
    quality_notes: str
    potential_biases: List[str]
    recommended_use: str  # "primary", "supporting", "verify", "avoid"


class QualityEvaluator:
    """
    Evaluate the quality of information sources.

    Combines domain-based heuristics with LLM-based content analysis.
    """

    def __init__(self, llm_client=None, language: str = "ja"):
        """
        Initialize QualityEvaluator.

        Args:
            llm_client: Optional LLM client for content-based evaluation
            language: Language for evaluation notes
        """
        self.llm = llm_client
        self.language = language

    def evaluate_url(self, url: str) -> Dict[str, Any]:
        """
        Evaluate quality based on URL/domain alone.

        Args:
            url: Source URL

        Returns:
            Dictionary with preliminary quality assessment
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
        except Exception:
            return {
                "source_type": SourceType.UNKNOWN,
                "quality_category": QualityCategory.UNVERIFIED,
                "domain_authority": 0.0,
            }

        # Check authoritative domains
        for pattern in AUTHORITATIVE_DOMAINS:
            if re.search(pattern, domain):
                return {
                    "source_type": self._detect_source_type_from_domain(domain),
                    "quality_category": QualityCategory.AUTHORITATIVE,
                    "domain_authority": 0.95,
                }

        # Check high quality domains
        for pattern in HIGH_QUALITY_DOMAINS:
            if re.search(pattern, domain):
                return {
                    "source_type": self._detect_source_type_from_domain(domain),
                    "quality_category": QualityCategory.HIGH,
                    "domain_authority": 0.8,
                }

        # Check medium quality domains
        for pattern in MEDIUM_QUALITY_DOMAINS:
            if re.search(pattern, domain):
                return {
                    "source_type": self._detect_source_type_from_domain(domain),
                    "quality_category": QualityCategory.MEDIUM,
                    "domain_authority": 0.6,
                }

        # Check low quality patterns
        for pattern in LOW_QUALITY_PATTERNS:
            if re.search(pattern, domain):
                return {
                    "source_type": self._detect_source_type_from_domain(domain),
                    "quality_category": QualityCategory.LOW,
                    "domain_authority": 0.3,
                }

        # Default to unverified
        return {
            "source_type": self._detect_source_type_from_domain(domain),
            "quality_category": QualityCategory.UNVERIFIED,
            "domain_authority": 0.5,
        }

    def _detect_source_type_from_domain(self, domain: str) -> SourceType:
        """Detect source type from domain."""
        domain = domain.lower()

        # Government
        if any(re.search(p, domain) for p in [r"\.gov", r"\.go\.", r"\.govt\."]):
            return SourceType.OFFICIAL

        # Academic
        if any(re.search(p, domain) for p in [r"\.edu", r"\.ac\.", r"arxiv", r"pubmed"]):
            return SourceType.ACADEMIC

        # News
        news_patterns = ["news", "times", "post", "guardian", "bbc", "cnn", "reuters", "nhk", "asahi", "nikkei"]
        if any(p in domain for p in news_patterns):
            return SourceType.NEWS

        # Wiki
        if "wiki" in domain:
            return SourceType.WIKI

        # Social
        social_patterns = ["twitter", "facebook", "instagram", "tiktok", "reddit", "x.com"]
        if any(p in domain for p in social_patterns):
            return SourceType.SOCIAL

        # Forum
        if any(p in domain for p in ["forum", "quora", "stackoverflow"]):
            return SourceType.FORUM

        # Blog
        blog_patterns = ["blog", "medium", "substack", "wordpress", "blogspot"]
        if any(p in domain for p in blog_patterns):
            return SourceType.BLOG

        # Commercial
        if any(p in domain for p in [".com", ".co.", "shop", "store", "buy"]):
            return SourceType.COMMERCIAL

        return SourceType.UNKNOWN

    def evaluate_content(
        self,
        url: str,
        title: str,
        content: str,
    ) -> QualityEvaluation:
        """
        Evaluate quality based on URL and content.

        Args:
            url: Source URL
            title: Source title
            content: Source content

        Returns:
            QualityEvaluation with full assessment
        """
        # Start with URL-based evaluation
        url_eval = self.evaluate_url(url)

        # Content-based indicators (heuristics)
        indicators = QualityIndicators(
            domain_authority=url_eval.get("domain_authority", 0.5)
        )

        # Check for author
        author_patterns = [
            r"(?:by|author|written by|著者|執筆)[:\s]+([A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+)",
            r"(?:reporter|correspondent|記者)[:\s]+",
        ]
        for pattern in author_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                indicators.has_author = True
                break

        # Check for date
        date_patterns = [
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}",
            r"\d{4}年\d{1,2}月\d{1,2}日",
        ]
        for pattern in date_patterns:
            if re.search(pattern, content):
                indicators.has_date = True
                break

        # Check for citations/references
        citation_patterns = [
            r"\[\d+\]",  # [1], [2], etc.
            r"\((?:19|20)\d{2}\)",  # (2023), (1999), etc.
            r"(?:references|bibliography|参考文献|出典)",
            r"(?:doi|DOI):\s*\d",
        ]
        for pattern in citation_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                indicators.has_citations = True
                break

        # Check for professional tone (absence of informal markers)
        informal_markers = [
            r"!{2,}",  # Multiple exclamation marks
            r"(?:lol|omg|wtf|lmao)",
            r"(?:www|笑|草)",
            r"(?:click here|今すぐ|無料)",  # Clickbait
        ]
        indicators.has_professional_tone = not any(
            re.search(p, content, re.IGNORECASE) for p in informal_markers
        )

        # Content depth (simple heuristic based on length and structure)
        word_count = len(content.split())
        has_headers = bool(re.search(r"(?:^|\n)#+\s|<h[1-6]>|(?:^|\n)[\d]+\.\s", content))
        has_lists = bool(re.search(r"(?:^|\n)[-*]\s|<[uo]l>", content))

        if word_count > 2000 and has_headers:
            indicators.content_depth = 0.8
        elif word_count > 1000:
            indicators.content_depth = 0.6
        elif word_count > 500:
            indicators.content_depth = 0.4
        else:
            indicators.content_depth = 0.2

        # Use LLM for deeper evaluation if available
        if self.llm:
            llm_eval = self._evaluate_with_llm(url, title, content)
            if llm_eval:
                # Merge LLM evaluation with heuristics
                indicators.is_primary_source = llm_eval.get("is_primary_source", False)
                indicators.is_peer_reviewed = llm_eval.get("is_peer_reviewed", False)
                indicators.factual_consistency = llm_eval.get("factual_consistency", 0.5)

                potential_biases = llm_eval.get("potential_biases", [])
                quality_notes = llm_eval.get("evaluation_notes", "")
                recommended_use = llm_eval.get("recommended_use", "verify")
            else:
                potential_biases = []
                quality_notes = "Heuristic evaluation only"
                recommended_use = "verify"
        else:
            potential_biases = []
            quality_notes = "Heuristic evaluation only (no LLM)"
            recommended_use = "verify"

        # Calculate final quality score and category
        quality_score = indicators.calculate_score()

        # Determine final category (can upgrade/downgrade from URL-based)
        base_category = url_eval.get("quality_category", QualityCategory.UNVERIFIED)

        if quality_score >= 0.8 and base_category != QualityCategory.LOW:
            final_category = QualityCategory.AUTHORITATIVE
        elif quality_score >= 0.6:
            final_category = max(base_category, QualityCategory.HIGH, key=self._category_rank)
        elif quality_score >= 0.4:
            final_category = max(base_category, QualityCategory.MEDIUM, key=self._category_rank)
        elif quality_score >= 0.2:
            final_category = QualityCategory.LOW
        else:
            final_category = base_category

        return QualityEvaluation(
            quality_category=final_category,
            source_type=url_eval.get("source_type", SourceType.UNKNOWN),
            quality_indicators=indicators,
            confidence=0.7 if self.llm else 0.5,
            quality_notes=quality_notes,
            potential_biases=potential_biases,
            recommended_use=recommended_use,
        )

    def _category_rank(self, category: QualityCategory) -> int:
        """Get ranking for quality category (higher is better)."""
        ranks = {
            QualityCategory.AUTHORITATIVE: 4,
            QualityCategory.HIGH: 3,
            QualityCategory.MEDIUM: 2,
            QualityCategory.LOW: 1,
            QualityCategory.UNVERIFIED: 0,
        }
        return ranks.get(category, 0)

    def _evaluate_with_llm(
        self,
        url: str,
        title: str,
        content: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM for deeper quality evaluation.

        Args:
            url: Source URL
            title: Source title
            content: Source content (truncated)

        Returns:
            LLM evaluation dictionary or None
        """
        if not self.llm:
            return None

        lang_instruction = (
            "Respond in Japanese." if self.language == "ja"
            else f"Respond in {self.language}."
        )

        prompt = f"""Evaluate the quality and reliability of this information source:

URL: {url}
Title: {title}

Content (first 3000 characters):
{content[:3000]}

{lang_instruction}

Analyze and return as JSON:
{{
    "source_type": "official/academic/news/blog/social/commercial/wiki/forum/unknown",
    "is_primary_source": true/false,
    "is_peer_reviewed": true/false,
    "factual_consistency": 0.0-1.0,
    "potential_biases": ["bias description 1", "bias description 2"],
    "recommended_use": "primary/supporting/verify/avoid",
    "evaluation_notes": "Brief notes about source quality and reliability"
}}

Consider:
- Is this a primary or secondary source?
- Does it cite other sources?
- Is there potential commercial or political bias?
- Is the information verifiable?
- Is the author/organization credible?"""

        try:
            response = self.llm.generate(prompt)
            content = response.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except Exception:
            pass

        return None

    def evaluate_evidence(self, evidence: Evidence) -> QualityEvaluation:
        """
        Evaluate quality of an Evidence object.

        Args:
            evidence: Evidence object to evaluate

        Returns:
            QualityEvaluation
        """
        return self.evaluate_content(
            url=evidence.url,
            title=evidence.title,
            content=evidence.content_excerpt,
        )

    def evaluate_and_update_locker(
        self,
        locker: EvidenceLocker,
        update_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate all evidence in a locker and update quality fields.

        Args:
            locker: EvidenceLocker to process
            update_existing: Whether to re-evaluate already evaluated evidence

        Returns:
            Summary statistics
        """
        evaluated_count = 0
        skipped_count = 0

        for evidence in locker.get_all_evidence():
            # Skip already evaluated unless update_existing is True
            if not update_existing and evidence.quality_category != QualityCategory.UNVERIFIED:
                skipped_count += 1
                continue

            evaluation = self.evaluate_evidence(evidence)

            locker.update_quality(
                evidence.id,
                quality_category=evaluation.quality_category,
                source_type=evaluation.source_type,
                quality_indicators=evaluation.quality_indicators,
                quality_notes=evaluation.quality_notes,
                potential_biases=evaluation.potential_biases,
            )

            evaluated_count += 1

        return {
            "evaluated_count": evaluated_count,
            "skipped_count": skipped_count,
            "quality_statistics": locker.get_quality_statistics(),
        }


def categorize_by_quality(
    locker: EvidenceLocker,
) -> Dict[QualityCategory, List[Evidence]]:
    """
    Organize evidence by quality category.

    Args:
        locker: EvidenceLocker with evidence

    Returns:
        Dictionary mapping quality categories to evidence lists
    """
    result = {cat: [] for cat in QualityCategory}

    for evidence in locker.get_all_evidence():
        result[evidence.quality_category].append(evidence)

    return result


def get_quality_summary(locker: EvidenceLocker, language: str = "ja") -> str:
    """
    Generate a human-readable quality summary.

    Args:
        locker: EvidenceLocker with evidence
        language: Output language

    Returns:
        Formatted summary string
    """
    stats = locker.get_quality_statistics()

    if language == "ja":
        lines = [
            "== 情報品質サマリー ==",
            f"総エビデンス数: {stats['total_evidence']}",
            f"高品質情報の割合: {stats['high_quality_percentage']}%",
            f"平均品質スコア: {stats['average_quality_score']}",
            "",
            "品質カテゴリ分布:",
        ]

        category_labels = {
            "authoritative": "権威的",
            "high": "高品質",
            "medium": "中品質",
            "low": "低品質",
            "unverified": "未検証",
        }

        for cat, count in stats["quality_distribution"].items():
            label = category_labels.get(cat, cat)
            lines.append(f"  {label}: {count}")

        lines.extend([
            "",
            "ソースタイプ分布:",
        ])

        source_labels = {
            "official": "公式",
            "academic": "学術",
            "news": "ニュース",
            "blog": "ブログ",
            "social": "SNS",
            "commercial": "商用",
            "wiki": "Wiki",
            "forum": "フォーラム",
            "unknown": "不明",
        }

        for st, count in stats["source_type_distribution"].items():
            label = source_labels.get(st, st)
            lines.append(f"  {label}: {count}")
    else:
        lines = [
            "== Quality Summary ==",
            f"Total Evidence: {stats['total_evidence']}",
            f"High Quality Percentage: {stats['high_quality_percentage']}%",
            f"Average Quality Score: {stats['average_quality_score']}",
            "",
            "Quality Distribution:",
        ]

        for cat, count in stats["quality_distribution"].items():
            lines.append(f"  {cat.title()}: {count}")

        lines.extend([
            "",
            "Source Type Distribution:",
        ])

        for st, count in stats["source_type_distribution"].items():
            lines.append(f"  {st.title()}: {count}")

    return "\n".join(lines)
