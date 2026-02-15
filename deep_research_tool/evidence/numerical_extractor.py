"""
Numerical Data Extractor for Deep Research Tool.

Extracts structured numerical data during the research phase for
intelligent chart generation and analysis.
"""

import re
import json
import uuid
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path


logger = logging.getLogger(__name__)


class DataType(str, Enum):
    """Types of numerical data."""
    CURRENCY = "currency"           # Money values ($100M, ¥10億)
    PERCENTAGE = "percentage"       # Percentages (30%, 0.3)
    COUNT = "count"                 # Counts/quantities (1000 users)
    RATIO = "ratio"                 # Ratios (1:3, 2.5x)
    RATE = "rate"                   # Rates (5% CAGR, 10% growth)
    TIME_SERIES = "time_series"     # Year/date associated values
    MEASUREMENT = "measurement"     # Physical measurements
    RANKING = "ranking"             # Rankings (1st, top 3)
    SCORE = "score"                 # Scores/indices (NPS: 45)
    OTHER = "other"


class MetricCategory(str, Enum):
    """Categories for metrics to enable grouping."""
    MARKET_SIZE = "market_size"
    GROWTH_RATE = "growth_rate"
    MARKET_SHARE = "market_share"
    REVENUE = "revenue"
    PROFIT = "profit"
    USER_COUNT = "user_count"
    PRICE = "price"
    PERFORMANCE = "performance"
    FORECAST = "forecast"
    COMPARISON = "comparison"
    OTHER = "other"


@dataclass
class NumericalDataPoint:
    """A single extracted numerical data point."""

    # Identification
    data_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Core data
    value: float = 0.0
    raw_text: str = ""              # Original text (e.g., "$10.5 billion")
    normalized_value: float = 0.0   # Normalized to base unit (e.g., 10500000000)
    unit: str = ""                  # Unit (USD, %, users, etc.)

    # Context
    metric_name: str = ""           # What this measures (e.g., "market size", "revenue")
    subject: str = ""               # What entity this is about (e.g., "AI market", "Company X")
    data_type: DataType = DataType.OTHER
    category: MetricCategory = MetricCategory.OTHER

    # Time context
    year: Optional[int] = None
    quarter: Optional[str] = None   # Q1, Q2, Q3, Q4
    date_context: str = ""          # "2024", "Q3 2024", "2020-2025"
    is_forecast: bool = False       # Whether this is a prediction

    # Source and reliability
    source_url: str = ""
    source_title: str = ""
    evidence_id: str = ""           # Link to EvidenceLocker
    section_id: str = ""            # Research section this belongs to

    # Confidence scoring
    extraction_confidence: float = 0.8  # How confident we are in extraction
    source_reliability: float = 0.7     # Source credibility (from verification)
    combined_confidence: float = 0.0    # Computed: extraction * reliability

    # Relationships
    related_data_ids: List[str] = field(default_factory=list)  # Links to related data points
    is_derived: bool = False        # Whether this was calculated (e.g., CAGR)
    derived_from: List[str] = field(default_factory=list)  # Source data IDs if derived

    def __post_init__(self):
        """Calculate combined confidence after init."""
        self.combined_confidence = self.extraction_confidence * self.source_reliability

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data_id": self.data_id,
            "value": self.value,
            "raw_text": self.raw_text,
            "normalized_value": self.normalized_value,
            "unit": self.unit,
            "metric_name": self.metric_name,
            "subject": self.subject,
            "data_type": self.data_type.value,
            "category": self.category.value,
            "year": self.year,
            "quarter": self.quarter,
            "date_context": self.date_context,
            "is_forecast": self.is_forecast,
            "source_url": self.source_url,
            "source_title": self.source_title,
            "evidence_id": self.evidence_id,
            "section_id": self.section_id,
            "extraction_confidence": self.extraction_confidence,
            "source_reliability": self.source_reliability,
            "combined_confidence": self.combined_confidence,
            "related_data_ids": self.related_data_ids,
            "is_derived": self.is_derived,
            "derived_from": self.derived_from,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NumericalDataPoint":
        """Create from dictionary."""
        return cls(
            data_id=data.get("data_id", str(uuid.uuid4())[:8]),
            value=data.get("value", 0.0),
            raw_text=data.get("raw_text", ""),
            normalized_value=data.get("normalized_value", 0.0),
            unit=data.get("unit", ""),
            metric_name=data.get("metric_name", ""),
            subject=data.get("subject", ""),
            data_type=DataType(data.get("data_type", "other")),
            category=MetricCategory(data.get("category", "other")),
            year=data.get("year"),
            quarter=data.get("quarter"),
            date_context=data.get("date_context", ""),
            is_forecast=data.get("is_forecast", False),
            source_url=data.get("source_url", ""),
            source_title=data.get("source_title", ""),
            evidence_id=data.get("evidence_id", ""),
            section_id=data.get("section_id", ""),
            extraction_confidence=data.get("extraction_confidence", 0.8),
            source_reliability=data.get("source_reliability", 0.7),
            related_data_ids=data.get("related_data_ids", []),
            is_derived=data.get("is_derived", False),
            derived_from=data.get("derived_from", []),
        )


@dataclass
class NumericalDataStore:
    """
    Store for all extracted numerical data from research.

    Provides methods for querying, grouping, and analyzing data points.
    """

    data_points: List[NumericalDataPoint] = field(default_factory=list)
    research_topic: str = ""

    def add(self, data_point: NumericalDataPoint) -> None:
        """Add a data point to the store."""
        self.data_points.append(data_point)

    def add_many(self, data_points: List[NumericalDataPoint]) -> None:
        """Add multiple data points."""
        self.data_points.extend(data_points)

    def get_by_id(self, data_id: str) -> Optional[NumericalDataPoint]:
        """Get a data point by ID."""
        for dp in self.data_points:
            if dp.data_id == data_id:
                return dp
        return None

    def get_by_section(self, section_id: str) -> List[NumericalDataPoint]:
        """Get all data points for a section."""
        return [dp for dp in self.data_points if dp.section_id == section_id]

    def get_by_category(self, category: MetricCategory) -> List[NumericalDataPoint]:
        """Get all data points in a category."""
        return [dp for dp in self.data_points if dp.category == category]

    def get_by_subject(self, subject: str) -> List[NumericalDataPoint]:
        """Get all data points for a subject (fuzzy match)."""
        subject_lower = subject.lower()
        return [
            dp for dp in self.data_points
            if subject_lower in dp.subject.lower()
        ]

    def get_time_series(
        self,
        subject: str = None,
        metric_name: str = None,
        min_points: int = 2,
    ) -> List[List[NumericalDataPoint]]:
        """
        Get time series data (grouped by subject+metric).

        Returns list of time series, each sorted by year.
        """
        # Filter by criteria
        filtered = [
            dp for dp in self.data_points
            if dp.year is not None
        ]

        if subject:
            subject_lower = subject.lower()
            filtered = [dp for dp in filtered if subject_lower in dp.subject.lower()]

        if metric_name:
            metric_lower = metric_name.lower()
            filtered = [dp for dp in filtered if metric_lower in dp.metric_name.lower()]

        # Group by (subject, metric_name, unit)
        groups: Dict[Tuple[str, str, str], List[NumericalDataPoint]] = {}
        for dp in filtered:
            key = (dp.subject.lower(), dp.metric_name.lower(), dp.unit)
            if key not in groups:
                groups[key] = []
            groups[key].append(dp)

        # Filter by min_points and sort each by year
        result = []
        for series in groups.values():
            if len(series) >= min_points:
                series.sort(key=lambda x: (x.year or 0, x.quarter or ""))
                result.append(series)

        return result

    def get_comparison_data(
        self,
        metric_name: str = None,
        year: int = None,
        min_items: int = 2,
    ) -> List[NumericalDataPoint]:
        """
        Get data suitable for comparison charts.

        Returns data points with same metric/year but different subjects.
        """
        filtered = self.data_points.copy()

        if metric_name:
            metric_lower = metric_name.lower()
            filtered = [dp for dp in filtered if metric_lower in dp.metric_name.lower()]

        if year:
            filtered = [dp for dp in filtered if dp.year == year]

        # Group by (metric_name, year, unit) and return groups with multiple subjects
        groups: Dict[Tuple[str, int, str], List[NumericalDataPoint]] = {}
        for dp in filtered:
            key = (dp.metric_name.lower(), dp.year or 0, dp.unit)
            if key not in groups:
                groups[key] = []
            groups[key].append(dp)

        # Find best comparison group
        best_group = []
        for group in groups.values():
            subjects = set(dp.subject for dp in group)
            if len(subjects) >= min_items and len(group) > len(best_group):
                best_group = group

        return best_group

    def get_high_confidence(self, threshold: float = 0.6) -> List[NumericalDataPoint]:
        """Get data points above confidence threshold."""
        return [
            dp for dp in self.data_points
            if dp.combined_confidence >= threshold
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the data store."""
        if not self.data_points:
            return {
                "total_points": 0,
                "by_category": {},
                "by_type": {},
                "time_series_count": 0,
                "avg_confidence": 0.0,
            }

        by_category = {}
        for dp in self.data_points:
            cat = dp.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        by_type = {}
        for dp in self.data_points:
            dtype = dp.data_type.value
            by_type[dtype] = by_type.get(dtype, 0) + 1

        time_series = self.get_time_series()

        return {
            "total_points": len(self.data_points),
            "by_category": by_category,
            "by_type": by_type,
            "time_series_count": len(time_series),
            "avg_confidence": sum(dp.combined_confidence for dp in self.data_points) / len(self.data_points),
            "derived_count": sum(1 for dp in self.data_points if dp.is_derived),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "research_topic": self.research_topic,
            "data_points": [dp.to_dict() for dp in self.data_points],
            "statistics": self.get_statistics(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NumericalDataStore":
        """Create from dictionary."""
        store = cls(research_topic=data.get("research_topic", ""))
        for dp_data in data.get("data_points", []):
            store.add(NumericalDataPoint.from_dict(dp_data))
        return store

    def save_to_json(self, filepath: Path) -> None:
        """Save to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_json(cls, filepath: Path) -> "NumericalDataStore":
        """Load from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class NumericalDataExtractor:
    """
    Extracts numerical data from text content during research.

    Uses LLM for intelligent extraction with context understanding,
    with pattern-based fallback for reliability.
    """

    # Patterns for basic extraction
    CURRENCY_PATTERNS = [
        # $10.5 billion, $100M, ¥10億
        r'[\$€¥£]\s*([\d,]+\.?\d*)\s*(billion|million|trillion|万|億|兆)?',
        r'([\d,]+\.?\d*)\s*(billion|million|trillion|万|億|兆)?\s*(dollars|ドル|円|euros|ユーロ)',
        r'USD\s*([\d,]+\.?\d*)\s*(B|M|K)?',
    ]

    PERCENTAGE_PATTERNS = [
        r'([\d,]+\.?\d*)\s*[%％]',
        r'([\d,]+\.?\d*)\s*percent',
    ]

    YEAR_PATTERNS = [
        r'\b(19|20)\d{2}\b',
        r'(FY|CY)\s*\'?\d{2,4}',
    ]

    GROWTH_PATTERNS = [
        r'CAGR\s*(?:of)?\s*([\d,]+\.?\d*)\s*[%％]',
        r'([\d,]+\.?\d*)\s*[%％]\s*(?:annual|yearly|year-over-year|YoY)\s*growth',
        r'grow(?:th|ing)?\s*(?:of|by|at)?\s*([\d,]+\.?\d*)\s*[%％]',
    ]

    def __init__(
        self,
        llm_client=None,
        language: str = "ja",
        min_confidence: float = 0.5,
        use_llm: bool = True,
    ):
        """
        Initialize extractor.

        Args:
            llm_client: LLM client for intelligent extraction
            language: Output language
            min_confidence: Minimum confidence to keep data point
            use_llm: Whether to use LLM (False = pattern-only)
        """
        self.llm = llm_client
        self.language = language
        self.min_confidence = min_confidence
        self.use_llm = use_llm and llm_client is not None

    def extract_from_content(
        self,
        content: str,
        source_url: str = "",
        source_title: str = "",
        evidence_id: str = "",
        section_id: str = "",
        source_reliability: float = 0.7,
        research_topic: str = "",
    ) -> List[NumericalDataPoint]:
        """
        Extract numerical data from content.

        Args:
            content: Text content to extract from
            source_url: Source URL
            source_title: Source title
            evidence_id: Evidence ID for linking
            section_id: Section ID this belongs to
            source_reliability: Source reliability score (from verification)
            research_topic: Research topic for context

        Returns:
            List of extracted NumericalDataPoint objects
        """
        if not content or len(content.strip()) < 10:
            return []

        data_points = []

        # Try LLM extraction first (more accurate)
        if self.use_llm:
            llm_points = self._extract_with_llm(
                content=content,
                source_url=source_url,
                source_title=source_title,
                evidence_id=evidence_id,
                section_id=section_id,
                source_reliability=source_reliability,
                research_topic=research_topic,
            )
            data_points.extend(llm_points)

        # Always run pattern extraction as backup/supplement
        pattern_points = self._extract_with_patterns(
            content=content,
            source_url=source_url,
            source_title=source_title,
            evidence_id=evidence_id,
            section_id=section_id,
            source_reliability=source_reliability,
        )

        # Merge pattern results (avoid duplicates)
        existing_values = {(dp.normalized_value, dp.year, dp.subject) for dp in data_points}
        for pp in pattern_points:
            key = (pp.normalized_value, pp.year, pp.subject)
            if key not in existing_values:
                data_points.append(pp)

        # Filter by confidence
        data_points = [
            dp for dp in data_points
            if dp.combined_confidence >= self.min_confidence
        ]

        return data_points

    def _extract_with_llm(
        self,
        content: str,
        source_url: str,
        source_title: str,
        evidence_id: str,
        section_id: str,
        source_reliability: float,
        research_topic: str,
    ) -> List[NumericalDataPoint]:
        """Extract using LLM for better context understanding."""

        # Truncate very long content
        max_content = 6000
        if len(content) > max_content:
            content = content[:max_content] + "..."

        if self.language == "ja":
            prompt = f"""以下のテキストから、グラフや表の作成に使える数値データを抽出してください。

研究トピック: {research_topic}

【抽出対象】
- 市場規模、売上、利益などの金額
- 成長率、シェア、割合などのパーセンテージ
- ユーザー数、販売数などの数量
- ランキング、スコア

【各データポイントについて以下を特定】
1. value: 数値（正規化した数値、例: 10億 → 10, 単位は別途）
2. raw_text: 元のテキスト表現（例: "10億ドル"）
3. unit: 単位（USD, 億円, %, ユーザー等）
4. metric_name: 何の指標か（例: "市場規模", "売上高", "成長率"）
5. subject: 何についてか（例: "AI市場", "企業A", "製品X"）
6. year: 年（あれば、例: 2024）
7. is_forecast: 予測値かどうか（true/false）
8. data_type: currency/percentage/count/ratio/rate/time_series/other
9. category: market_size/growth_rate/market_share/revenue/profit/user_count/price/performance/forecast/comparison/other
10. confidence: 抽出の確信度（0.0-1.0）

JSON配列形式で出力してください。数値データがない場合は空配列[]を返してください。

テキスト:
{content}

出力形式:
```json
[
  {{
    "value": 10,
    "raw_text": "10億ドル",
    "unit": "billion USD",
    "metric_name": "市場規模",
    "subject": "生成AI市場",
    "year": 2024,
    "is_forecast": false,
    "data_type": "currency",
    "category": "market_size",
    "confidence": 0.9
  }}
]
```"""
        else:
            prompt = f"""Extract numerical data suitable for charts and tables from the following text.

Research Topic: {research_topic}

【Data to Extract】
- Market size, revenue, profit (currency)
- Growth rates, shares, percentages
- User counts, sales quantities
- Rankings, scores

【For each data point, identify】
1. value: Numeric value (normalized, e.g., 10 billion → 10, unit separate)
2. raw_text: Original text representation
3. unit: Unit (USD, %, users, etc.)
4. metric_name: What metric (e.g., "market size", "revenue", "growth rate")
5. subject: What entity (e.g., "AI market", "Company A", "Product X")
6. year: Year (if available)
7. is_forecast: Whether this is a forecast (true/false)
8. data_type: currency/percentage/count/ratio/rate/time_series/other
9. category: market_size/growth_rate/market_share/revenue/profit/user_count/price/performance/forecast/comparison/other
10. confidence: Extraction confidence (0.0-1.0)

Output as JSON array. Return empty array [] if no numerical data found.

Text:
{content}

Output format:
```json
[
  {{
    "value": 10,
    "raw_text": "$10 billion",
    "unit": "billion USD",
    "metric_name": "market size",
    "subject": "Generative AI market",
    "year": 2024,
    "is_forecast": false,
    "data_type": "currency",
    "category": "market_size",
    "confidence": 0.9
  }}
]
```"""

        try:
            response = self.llm.generate(prompt)
            response_text = response.content.strip()

            # Parse JSON from response
            extracted = self._parse_json_array(response_text)

            data_points = []
            for item in extracted:
                dp = NumericalDataPoint(
                    value=float(item.get("value", 0)),
                    raw_text=item.get("raw_text", ""),
                    normalized_value=self._normalize_value(
                        float(item.get("value", 0)),
                        item.get("unit", "")
                    ),
                    unit=item.get("unit", ""),
                    metric_name=item.get("metric_name", ""),
                    subject=item.get("subject", ""),
                    data_type=DataType(item.get("data_type", "other")),
                    category=MetricCategory(item.get("category", "other")),
                    year=item.get("year"),
                    date_context=str(item.get("year", "")) if item.get("year") else "",
                    is_forecast=item.get("is_forecast", False),
                    source_url=source_url,
                    source_title=source_title,
                    evidence_id=evidence_id,
                    section_id=section_id,
                    extraction_confidence=float(item.get("confidence", 0.8)),
                    source_reliability=source_reliability,
                )
                data_points.append(dp)

            return data_points

        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")
            return []

    def _extract_with_patterns(
        self,
        content: str,
        source_url: str,
        source_title: str,
        evidence_id: str,
        section_id: str,
        source_reliability: float,
    ) -> List[NumericalDataPoint]:
        """Extract using regex patterns as fallback."""
        data_points = []

        # Extract years for context
        years = set(re.findall(r'\b(20\d{2})\b', content))
        default_year = max(int(y) for y in years) if years else None

        # Currency extraction
        for pattern in self.CURRENCY_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                try:
                    # Get surrounding context (50 chars each side)
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 50)
                    context = content[start:end]

                    # Find year in context
                    year_match = re.search(r'\b(20\d{2})\b', context)
                    year = int(year_match.group(1)) if year_match else default_year

                    raw_text = match.group(0)
                    value = self._parse_currency_value(raw_text)

                    if value > 0:
                        dp = NumericalDataPoint(
                            value=value,
                            raw_text=raw_text,
                            normalized_value=value,
                            unit=self._detect_currency_unit(raw_text),
                            metric_name="",  # Pattern can't determine this
                            subject="",
                            data_type=DataType.CURRENCY,
                            category=MetricCategory.OTHER,
                            year=year,
                            source_url=source_url,
                            source_title=source_title,
                            evidence_id=evidence_id,
                            section_id=section_id,
                            extraction_confidence=0.6,  # Lower confidence for patterns
                            source_reliability=source_reliability,
                        )
                        data_points.append(dp)
                except Exception:
                    continue

        # Percentage extraction
        for pattern in self.PERCENTAGE_PATTERNS:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                try:
                    raw_text = match.group(0)
                    value = float(match.group(1).replace(',', ''))

                    # Get context for year
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 50)
                    context = content[start:end]
                    year_match = re.search(r'\b(20\d{2})\b', context)
                    year = int(year_match.group(1)) if year_match else default_year

                    # Detect if it's growth rate
                    is_growth = any(kw in context.lower() for kw in [
                        'growth', 'cagr', '成長', '増加', 'increase', 'yoy'
                    ])

                    dp = NumericalDataPoint(
                        value=value,
                        raw_text=raw_text,
                        normalized_value=value / 100,  # Normalize to decimal
                        unit="%",
                        metric_name="growth rate" if is_growth else "",
                        data_type=DataType.RATE if is_growth else DataType.PERCENTAGE,
                        category=MetricCategory.GROWTH_RATE if is_growth else MetricCategory.OTHER,
                        year=year,
                        source_url=source_url,
                        source_title=source_title,
                        evidence_id=evidence_id,
                        section_id=section_id,
                        extraction_confidence=0.5,
                        source_reliability=source_reliability,
                    )
                    data_points.append(dp)
                except Exception:
                    continue

        return data_points

    def _parse_json_array(self, text: str) -> List[Dict]:
        """Parse JSON array from LLM response."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from code block
        code_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if code_match:
            try:
                return json.loads(code_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding array brackets
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return []

    def _normalize_value(self, value: float, unit: str) -> float:
        """Normalize value to base unit."""
        unit_lower = unit.lower()

        multipliers = {
            'trillion': 1e12, '兆': 1e12,
            'billion': 1e9, '億': 1e8,
            'million': 1e6, '百万': 1e6,
            'thousand': 1e3, '千': 1e3, '万': 1e4,
            'k': 1e3, 'm': 1e6, 'b': 1e9,
        }

        for key, mult in multipliers.items():
            if key in unit_lower:
                return value * mult

        return value

    def _parse_currency_value(self, text: str) -> float:
        """Parse currency value from text."""
        # Remove currency symbols
        text = re.sub(r'[\$€¥£]', '', text)

        # Find number
        num_match = re.search(r'([\d,]+\.?\d*)', text)
        if not num_match:
            return 0.0

        value = float(num_match.group(1).replace(',', ''))

        # Apply multiplier
        text_lower = text.lower()
        if 'trillion' in text_lower or '兆' in text:
            value *= 1e12
        elif 'billion' in text_lower or '億' in text:
            value *= 1e9 if 'billion' in text_lower else 1e8
        elif 'million' in text_lower or '百万' in text:
            value *= 1e6
        elif '万' in text:
            value *= 1e4

        return value

    def _detect_currency_unit(self, text: str) -> str:
        """Detect currency unit from text."""
        if '$' in text or 'dollar' in text.lower():
            if 'billion' in text.lower():
                return 'billion USD'
            elif 'million' in text.lower():
                return 'million USD'
            return 'USD'
        elif '¥' in text or '円' in text:
            if '兆' in text:
                return '兆円'
            elif '億' in text:
                return '億円'
            elif '万' in text:
                return '万円'
            return '円'
        elif '€' in text or 'euro' in text.lower():
            return 'EUR'
        return 'USD'  # Default


class DerivedMetricsCalculator:
    """
    Calculates derived metrics from extracted data.

    Includes CAGR, growth rates, market share, averages, etc.
    """

    def __init__(self, store: NumericalDataStore):
        """Initialize with data store."""
        self.store = store

    def calculate_all(self) -> List[NumericalDataPoint]:
        """Calculate all possible derived metrics."""
        derived = []

        # Calculate CAGRs for time series
        derived.extend(self.calculate_cagrs())

        # Calculate market shares if we have total and parts
        derived.extend(self.calculate_market_shares())

        # Calculate growth rates for consecutive years
        derived.extend(self.calculate_yoy_growth())

        return derived

    def calculate_cagrs(self) -> List[NumericalDataPoint]:
        """Calculate CAGR for all time series."""
        derived = []

        time_series = self.store.get_time_series(min_points=2)

        for series in time_series:
            if len(series) < 2:
                continue

            # Get first and last points
            first = series[0]
            last = series[-1]

            if first.year is None or last.year is None:
                continue

            years = last.year - first.year
            if years <= 0 or first.normalized_value <= 0:
                continue

            # Calculate CAGR
            try:
                cagr = (
                    (last.normalized_value / first.normalized_value) ** (1 / years) - 1
                ) * 100

                dp = NumericalDataPoint(
                    value=round(cagr, 2),
                    raw_text=f"CAGR {first.year}-{last.year}: {cagr:.1f}%",
                    normalized_value=cagr / 100,
                    unit="%",
                    metric_name=f"{first.metric_name} CAGR",
                    subject=first.subject,
                    data_type=DataType.RATE,
                    category=MetricCategory.GROWTH_RATE,
                    date_context=f"{first.year}-{last.year}",
                    source_url=first.source_url,
                    section_id=first.section_id,
                    extraction_confidence=0.9,
                    source_reliability=min(first.source_reliability, last.source_reliability),
                    is_derived=True,
                    derived_from=[first.data_id, last.data_id],
                )
                derived.append(dp)

            except Exception:
                continue

        return derived

    def calculate_yoy_growth(self) -> List[NumericalDataPoint]:
        """Calculate year-over-year growth rates."""
        derived = []

        time_series = self.store.get_time_series(min_points=2)

        for series in time_series:
            for i in range(1, len(series)):
                prev = series[i - 1]
                curr = series[i]

                if prev.year is None or curr.year is None:
                    continue
                if curr.year - prev.year != 1:
                    continue
                if prev.normalized_value <= 0:
                    continue

                try:
                    growth = (
                        (curr.normalized_value - prev.normalized_value)
                        / prev.normalized_value
                        * 100
                    )

                    dp = NumericalDataPoint(
                        value=round(growth, 2),
                        raw_text=f"YoY Growth {curr.year}: {growth:.1f}%",
                        normalized_value=growth / 100,
                        unit="%",
                        metric_name=f"{curr.metric_name} YoY Growth",
                        subject=curr.subject,
                        data_type=DataType.RATE,
                        category=MetricCategory.GROWTH_RATE,
                        year=curr.year,
                        date_context=str(curr.year),
                        source_url=curr.source_url,
                        section_id=curr.section_id,
                        extraction_confidence=0.9,
                        source_reliability=min(prev.source_reliability, curr.source_reliability),
                        is_derived=True,
                        derived_from=[prev.data_id, curr.data_id],
                    )
                    derived.append(dp)

                except Exception:
                    continue

        return derived

    def calculate_market_shares(self) -> List[NumericalDataPoint]:
        """
        Calculate market shares if we have total market and individual values.
        """
        derived = []

        # Find potential total market data
        totals = [
            dp for dp in self.store.data_points
            if dp.category == MetricCategory.MARKET_SIZE
            and any(kw in dp.subject.lower() for kw in ['total', '全体', 'market', '市場'])
        ]

        for total in totals:
            if total.normalized_value <= 0:
                continue

            # Find parts with same year and unit
            parts = [
                dp for dp in self.store.data_points
                if dp.year == total.year
                and dp.unit == total.unit
                and dp.data_id != total.data_id
                and dp.category in (MetricCategory.MARKET_SIZE, MetricCategory.REVENUE)
                and dp.normalized_value < total.normalized_value
            ]

            for part in parts:
                try:
                    share = (part.normalized_value / total.normalized_value) * 100

                    dp = NumericalDataPoint(
                        value=round(share, 2),
                        raw_text=f"{part.subject} Market Share: {share:.1f}%",
                        normalized_value=share / 100,
                        unit="%",
                        metric_name="Market Share",
                        subject=part.subject,
                        data_type=DataType.PERCENTAGE,
                        category=MetricCategory.MARKET_SHARE,
                        year=total.year,
                        date_context=str(total.year) if total.year else "",
                        source_url=part.source_url,
                        section_id=part.section_id,
                        extraction_confidence=0.85,
                        source_reliability=min(part.source_reliability, total.source_reliability),
                        is_derived=True,
                        derived_from=[part.data_id, total.data_id],
                    )
                    derived.append(dp)

                except Exception:
                    continue

        return derived

    def interpolate_missing_years(
        self,
        series: List[NumericalDataPoint],
    ) -> List[NumericalDataPoint]:
        """
        Interpolate missing years in a time series.

        Uses linear interpolation for gaps.
        """
        if len(series) < 2:
            return []

        derived = []
        series = sorted(series, key=lambda x: x.year or 0)

        for i in range(len(series) - 1):
            curr = series[i]
            next_dp = series[i + 1]

            if curr.year is None or next_dp.year is None:
                continue

            gap = next_dp.year - curr.year
            if gap <= 1:
                continue

            # Linear interpolation
            slope = (next_dp.normalized_value - curr.normalized_value) / gap

            for year_offset in range(1, gap):
                year = curr.year + year_offset
                value = curr.normalized_value + slope * year_offset

                dp = NumericalDataPoint(
                    value=value,
                    raw_text=f"Interpolated for {year}",
                    normalized_value=value,
                    unit=curr.unit,
                    metric_name=curr.metric_name,
                    subject=curr.subject,
                    data_type=curr.data_type,
                    category=curr.category,
                    year=year,
                    date_context=str(year),
                    section_id=curr.section_id,
                    extraction_confidence=0.5,  # Lower confidence for interpolated
                    source_reliability=min(curr.source_reliability, next_dp.source_reliability),
                    is_derived=True,
                    derived_from=[curr.data_id, next_dp.data_id],
                )
                derived.append(dp)

        return derived
