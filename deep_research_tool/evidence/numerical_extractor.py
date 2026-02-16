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
    MEASUREMENT = "measurement"     # Physical measurements (generic)
    RANKING = "ranking"             # Rankings (1st, top 3)
    SCORE = "score"                 # Scores/indices (NPS: 45)
    # Engineering / scientific data types
    STRESS = "stress"               # Pa, MPa, GPa, psi
    STRAIN = "strain"               # Dimensionless or %
    FORCE = "force"                 # N, kN, kgf, lbf
    ENERGY = "energy"               # J, kJ, MJ, cal, eV
    POWER = "power"                 # W, kW, MW, hp
    TEMPERATURE = "temperature"     # K, °C, °F
    PRESSURE = "pressure"           # Pa, bar, atm, psi
    LENGTH = "length"               # m, mm, μm, nm, in, ft
    MASS = "mass"                   # kg, g, mg, lb, oz
    VELOCITY = "velocity"           # m/s, km/h, mph
    DENSITY = "density"             # kg/m³, g/cm³
    FREQUENCY = "frequency"         # Hz, kHz, MHz, GHz
    VOLTAGE = "voltage"             # V, mV, kV
    CURRENT = "current"             # A, mA, μA
    RESISTANCE = "resistance"       # Ω, kΩ, MΩ
    CAPACITANCE = "capacitance"     # F, μF, nF, pF
    VISCOSITY = "viscosity"         # Pa·s, mPa·s, cP
    THERMAL_CONDUCTIVITY = "thermal_conductivity"  # W/(m·K)
    ARBITRARY = "arbitrary"         # a.u., arb., dimensionless relative values
    DIMENSIONLESS = "dimensionless" # Pure numbers, ratios without units
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
    # Engineering / scientific categories
    MECHANICAL_PROPERTY = "mechanical_property"       # Tensile strength, hardness, etc.
    THERMAL_PROPERTY = "thermal_property"             # Thermal conductivity, heat capacity
    ELECTRICAL_PROPERTY = "electrical_property"       # Conductivity, resistance, etc.
    MATERIAL_PROPERTY = "material_property"           # Density, porosity, etc.
    PHYSICAL_CONSTANT = "physical_constant"           # Speed of light, etc.
    ENVIRONMENTAL = "environmental"                   # Temperature, pressure, humidity
    CHEMICAL_PROPERTY = "chemical_property"           # pH, concentration, etc.
    DIMENSIONAL = "dimensional"                       # Sizes, lengths, areas, volumes
    ENERGY_PROPERTY = "energy_property"               # Energy, power, efficiency
    RELATIVE_VALUE = "relative_value"                 # Arbitrary units, normalized values
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


class CurrencyNormalizer:
    """
    Normalizes currency symbols, names, and codes to ISO 4217 codes.

    Handles:
    - Symbols: $, €, ¥, £, etc.
    - Names: dollar, ドル, euro, ユーロ, etc.
    - Codes: USD, EUR, JPY, GBP, etc.
    - Full-width numbers: １２３４５ → 12345
    - Thousand separators: 1,000 → 1000
    """

    # Currency normalization mappings → ISO 4217 code
    CURRENCY_MAPPINGS: Dict[str, str] = {
        # US Dollar
        '$': 'USD',
        'US$': 'USD',
        'USD': 'USD',
        'dollar': 'USD',
        'dollars': 'USD',
        'ドル': 'USD',
        '米ドル': 'USD',
        'アメリカドル': 'USD',
        # Euro
        '€': 'EUR',
        'EUR': 'EUR',
        'euro': 'EUR',
        'euros': 'EUR',
        'ユーロ': 'EUR',
        # Japanese Yen
        '¥': 'JPY',
        '￥': 'JPY',
        'JPY': 'JPY',
        'yen': 'JPY',
        '円': 'JPY',
        '日本円': 'JPY',
        # British Pound
        '£': 'GBP',
        'GBP': 'GBP',
        'pound': 'GBP',
        'pounds': 'GBP',
        'ポンド': 'GBP',
        '英ポンド': 'GBP',
        # Chinese Yuan
        '元': 'CNY',
        '人民元': 'CNY',
        'CNY': 'CNY',
        'RMB': 'CNY',
        'yuan': 'CNY',
        # Korean Won
        '₩': 'KRW',
        'KRW': 'KRW',
        'won': 'KRW',
        'ウォン': 'KRW',
        '韓国ウォン': 'KRW',
        # Swiss Franc
        'CHF': 'CHF',
        'franc': 'CHF',
        'francs': 'CHF',
        'フラン': 'CHF',
        # Australian Dollar
        'A$': 'AUD',
        'AU$': 'AUD',
        'AUD': 'AUD',
        '豪ドル': 'AUD',
        # Canadian Dollar
        'C$': 'CAD',
        'CA$': 'CAD',
        'CAD': 'CAD',
        'カナダドル': 'CAD',
        # Indian Rupee
        '₹': 'INR',
        'INR': 'INR',
        'rupee': 'INR',
        'rupees': 'INR',
        'ルピー': 'INR',
        # Brazilian Real
        'R$': 'BRL',
        'BRL': 'BRL',
        'real': 'BRL',
        'レアル': 'BRL',
        # Russian Ruble
        '₽': 'RUB',
        'RUB': 'RUB',
        'ruble': 'RUB',
        'rubles': 'RUB',
        'ルーブル': 'RUB',
        # Singapore Dollar
        'S$': 'SGD',
        'SG$': 'SGD',
        'SGD': 'SGD',
        'シンガポールドル': 'SGD',
        # Hong Kong Dollar
        'HK$': 'HKD',
        'HKD': 'HKD',
        '香港ドル': 'HKD',
        # Taiwan Dollar
        'NT$': 'TWD',
        'TWD': 'TWD',
        '台湾ドル': 'TWD',
        # Thai Baht
        '฿': 'THB',
        'THB': 'THB',
        'baht': 'THB',
        'バーツ': 'THB',
    }

    # Full-width to half-width number mapping
    FULLWIDTH_NUMBERS: Dict[str, str] = {
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
        '．': '.', '，': ',',
    }

    @classmethod
    def normalize_currency(cls, text: str) -> Optional[str]:
        """
        Normalize currency text to ISO 4217 code.

        Args:
            text: Currency symbol, name, or code (e.g., "€", "ユーロ", "EUR")

        Returns:
            ISO 4217 code (e.g., "EUR") or None if not recognized
        """
        if not text:
            return None

        # Clean the text
        cleaned = text.strip().lower()

        # Direct lookup (case-insensitive for names)
        for key, iso_code in cls.CURRENCY_MAPPINGS.items():
            if cleaned == key.lower():
                return iso_code

        # Try original case for symbols
        original = text.strip()
        if original in cls.CURRENCY_MAPPINGS:
            return cls.CURRENCY_MAPPINGS[original]

        return None

    @classmethod
    def normalize_number(cls, text: str) -> str:
        """
        Normalize number format.

        Handles:
        - Full-width numbers: １２３４５ → 12345
        - Thousand separators: 1,000 → 1000
        - Spaces in numbers: 1 000 → 1000

        Args:
            text: Number string (e.g., "１，０００", "1,000", "1 000")

        Returns:
            Normalized number string (e.g., "1000")
        """
        result = text

        # Convert full-width to half-width
        for fw, hw in cls.FULLWIDTH_NUMBERS.items():
            result = result.replace(fw, hw)

        # Remove thousand separators (comma)
        result = result.replace(',', '')

        # Remove space separators (European style)
        result = re.sub(r'(\d)\s+(\d)', r'\1\2', result)

        return result

    @classmethod
    def parse_number(cls, text: str) -> Optional[float]:
        """
        Parse a number string with normalization.

        Args:
            text: Number string (e.g., "１，０００.５", "1,000.5")

        Returns:
            Float value or None if parsing failed
        """
        try:
            normalized = cls.normalize_number(text)
            return float(normalized)
        except ValueError:
            return None

    @classmethod
    def get_currency_from_text(cls, text: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract currency code and value from text.

        Args:
            text: Text containing currency (e.g., "10€", "100ユーロ", "€50")

        Returns:
            (ISO code, value) tuple, e.g., ("EUR", 10.0)
        """
        # Try symbol at start: $100, €50, ¥1000
        for symbol in ['$', 'US$', 'A$', 'C$', 'S$', 'HK$', 'NT$', 'R$',
                       '€', '£', '¥', '￥', '₩', '₹', '₽', '฿']:
            if text.startswith(symbol):
                iso_code = cls.CURRENCY_MAPPINGS.get(symbol)
                if iso_code:
                    num_text = text[len(symbol):].strip()
                    value = cls.parse_number(num_text)
                    if value is not None:
                        return iso_code, value

        # Try symbol at end: 100€, 50£
        for symbol in ['€', '£', '¥', '￥', '₩', '₹', '₽', '฿']:
            if text.endswith(symbol):
                iso_code = cls.CURRENCY_MAPPINGS.get(symbol)
                if iso_code:
                    num_text = text[:-len(symbol)].strip()
                    value = cls.parse_number(num_text)
                    if value is not None:
                        return iso_code, value

        # Try Japanese/Korean names at end: 100円, 100ドル, 100ユーロ
        for name in ['円', 'ドル', '米ドル', 'ユーロ', 'ポンド', '元', '人民元',
                     'ウォン', 'フラン', '豪ドル', 'カナダドル', 'ルピー',
                     'レアル', 'ルーブル', 'バーツ']:
            if text.endswith(name):
                iso_code = cls.CURRENCY_MAPPINGS.get(name)
                if iso_code:
                    num_text = text[:-len(name)].strip()
                    value = cls.parse_number(num_text)
                    if value is not None:
                        return iso_code, value

        return None, None


class UnitConverter:
    """
    Comprehensive unit conversion system with three levels:

    Level 1: SI prefix normalization (GPa → Pa × 10⁹)
    Level 2: Same-dimension unit conversion (psi → Pa, cal → J)
    Level 3: Dimensional analysis via pint library (Pa = kg·m⁻¹·s⁻²)
    """

    # SI prefix multipliers
    SI_PREFIXES: Dict[str, float] = {
        'Y': 1e24,   'yotta': 1e24,
        'Z': 1e21,   'zetta': 1e21,
        'E': 1e18,   'exa': 1e18,
        'P': 1e15,   'peta': 1e15,
        'T': 1e12,   'tera': 1e12,
        'G': 1e9,    'giga': 1e9,
        'M': 1e6,    'mega': 1e6,
        'k': 1e3,    'kilo': 1e3,
        'h': 1e2,    'hecto': 1e2,
        'da': 1e1,   'deca': 1e1,
        'd': 1e-1,   'deci': 1e-1,
        'c': 1e-2,   'centi': 1e-2,
        'm': 1e-3,   'milli': 1e-3,
        'μ': 1e-6,   'u': 1e-6,   'micro': 1e-6,
        'n': 1e-9,   'nano': 1e-9,
        'p': 1e-12,  'pico': 1e-12,
        'f': 1e-15,  'femto': 1e-15,
        'a': 1e-18,  'atto': 1e-18,
    }

    # Base SI units and their symbols for prefix detection
    SI_BASE_UNITS: Dict[str, str] = {
        'Pa': 'pascal', 'N': 'newton', 'J': 'joule', 'W': 'watt',
        'Hz': 'hertz', 'V': 'volt', 'A': 'ampere', 'Ω': 'ohm',
        'F': 'farad', 'H': 'henry', 'S': 'siemens', 'T': 'tesla',
        'Wb': 'weber', 'lm': 'lumen', 'lx': 'lux', 'Bq': 'becquerel',
        'Gy': 'gray', 'Sv': 'sievert', 'kat': 'katal',
        'm': 'meter', 'g': 'gram', 's': 'second', 'K': 'kelvin',
        'mol': 'mole', 'cd': 'candela', 'L': 'liter',
        'eV': 'electronvolt', 'bar': 'bar', 'b': 'byte',
    }

    # Same-dimension conversion tables (Level 2)
    # All values convert TO the SI base unit
    CONVERSION_TABLES: Dict[str, Dict[str, float]] = {
        # Pressure → Pa
        'pressure': {
            'Pa': 1.0,
            'hPa': 1e2,
            'kPa': 1e3,
            'MPa': 1e6,
            'GPa': 1e9,
            'bar': 1e5,
            'mbar': 1e2,
            'atm': 101325.0,
            'torr': 133.322,
            'mmHg': 133.322,
            'psi': 6894.757,
            'ksi': 6894757.0,
            'psf': 47.88026,
        },
        # Force → N
        'force': {
            'N': 1.0,
            'kN': 1e3,
            'MN': 1e6,
            'GN': 1e9,
            'mN': 1e-3,
            'μN': 1e-6,
            'dyn': 1e-5,
            'kgf': 9.80665,
            'gf': 9.80665e-3,
            'lbf': 4.44822,
            'ozf': 0.278014,
            'kip': 4448.22,
        },
        # Energy → J
        'energy': {
            'J': 1.0,
            'kJ': 1e3,
            'MJ': 1e6,
            'GJ': 1e9,
            'mJ': 1e-3,
            'cal': 4.184,
            'kcal': 4184.0,
            'Wh': 3600.0,
            'kWh': 3.6e6,
            'MWh': 3.6e9,
            'eV': 1.602176634e-19,
            'keV': 1.602176634e-16,
            'MeV': 1.602176634e-13,
            'GeV': 1.602176634e-10,
            'BTU': 1055.06,
            'erg': 1e-7,
            'ft·lbf': 1.35582,
        },
        # Power → W
        'power': {
            'W': 1.0,
            'kW': 1e3,
            'MW': 1e6,
            'GW': 1e9,
            'mW': 1e-3,
            'μW': 1e-6,
            'hp': 745.7,
            'PS': 735.499,
            'BTU/h': 0.293071,
        },
        # Length → m
        'length': {
            'm': 1.0,
            'km': 1e3,
            'cm': 1e-2,
            'mm': 1e-3,
            'μm': 1e-6,
            'nm': 1e-9,
            'pm': 1e-12,
            'Å': 1e-10,
            'in': 0.0254,
            'ft': 0.3048,
            'yd': 0.9144,
            'mi': 1609.344,
            'mil': 2.54e-5,
            'thou': 2.54e-5,
        },
        # Mass → kg
        'mass': {
            'kg': 1.0,
            'g': 1e-3,
            'mg': 1e-6,
            'μg': 1e-9,
            'ng': 1e-12,
            't': 1e3,
            'lb': 0.453592,
            'oz': 0.0283495,
            'grain': 6.47989e-5,
            'slug': 14.5939,
        },
        # Temperature (special: offset + scale, stored as K)
        # Handled separately via convert_temperature()
        # Frequency → Hz
        'frequency': {
            'Hz': 1.0,
            'kHz': 1e3,
            'MHz': 1e6,
            'GHz': 1e9,
            'THz': 1e12,
            'rpm': 1.0 / 60.0,
        },
        # Voltage → V
        'voltage': {
            'V': 1.0,
            'kV': 1e3,
            'MV': 1e6,
            'mV': 1e-3,
            'μV': 1e-6,
        },
        # Current → A
        'current': {
            'A': 1.0,
            'kA': 1e3,
            'mA': 1e-3,
            'μA': 1e-6,
            'nA': 1e-9,
        },
        # Resistance → Ω
        'resistance': {
            'Ω': 1.0,
            'ohm': 1.0,
            'kΩ': 1e3,
            'MΩ': 1e6,
            'GΩ': 1e9,
            'mΩ': 1e-3,
        },
        # Capacitance → F
        'capacitance': {
            'F': 1.0,
            'mF': 1e-3,
            'μF': 1e-6,
            'nF': 1e-9,
            'pF': 1e-12,
        },
        # Viscosity → Pa·s
        'viscosity': {
            'Pa·s': 1.0,
            'Pa*s': 1.0,
            'mPa·s': 1e-3,
            'mPa*s': 1e-3,
            'cP': 1e-3,
            'P': 0.1,
            'cSt': 1e-6,  # kinematic, approximate
        },
        # Density → kg/m³
        'density': {
            'kg/m³': 1.0,
            'kg/m3': 1.0,
            'g/cm³': 1e3,
            'g/cm3': 1e3,
            'g/mL': 1e3,
            'kg/L': 1e3,
            'lb/ft³': 16.0185,
            'lb/ft3': 16.0185,
            'lb/in³': 27679.9,
            'lb/in3': 27679.9,
        },
        # Velocity → m/s
        'velocity': {
            'm/s': 1.0,
            'km/h': 1.0 / 3.6,
            'km/s': 1e3,
            'mph': 0.44704,
            'ft/s': 0.3048,
            'knot': 0.514444,
            'kn': 0.514444,
        },
        # Thermal conductivity → W/(m·K)
        'thermal_conductivity': {
            'W/(m·K)': 1.0,
            'W/(m*K)': 1.0,
            'W/mK': 1.0,
            'W/(m·°C)': 1.0,
            'BTU/(h·ft·°F)': 1.730735,
            'cal/(s·cm·°C)': 418.4,
        },
        # Area → m²
        'area': {
            'm²': 1.0,
            'm2': 1.0,
            'cm²': 1e-4,
            'cm2': 1e-4,
            'mm²': 1e-6,
            'mm2': 1e-6,
            'km²': 1e6,
            'km2': 1e6,
            'ha': 1e4,
            'in²': 6.4516e-4,
            'in2': 6.4516e-4,
            'ft²': 0.092903,
            'ft2': 0.092903,
            'acre': 4046.86,
        },
        # Volume → m³
        'volume': {
            'm³': 1.0,
            'm3': 1.0,
            'cm³': 1e-6,
            'cm3': 1e-6,
            'cc': 1e-6,
            'mm³': 1e-9,
            'mm3': 1e-9,
            'L': 1e-3,
            'mL': 1e-6,
            'μL': 1e-9,
            'gal': 3.78541e-3,
            'ft³': 0.0283168,
            'ft3': 0.0283168,
            'in³': 1.6387e-5,
            'in3': 1.6387e-5,
        },
    }

    # Map unit → dimension name for lookup
    _unit_to_dimension: Dict[str, str] = {}

    # DataType mapping for dimensions
    DIMENSION_TO_DATATYPE: Dict[str, DataType] = {
        'pressure': DataType.PRESSURE,
        'force': DataType.FORCE,
        'energy': DataType.ENERGY,
        'power': DataType.POWER,
        'length': DataType.LENGTH,
        'mass': DataType.MASS,
        'frequency': DataType.FREQUENCY,
        'voltage': DataType.VOLTAGE,
        'current': DataType.CURRENT,
        'resistance': DataType.RESISTANCE,
        'capacitance': DataType.CAPACITANCE,
        'viscosity': DataType.VISCOSITY,
        'density': DataType.DENSITY,
        'velocity': DataType.VELOCITY,
        'thermal_conductivity': DataType.THERMAL_CONDUCTIVITY,
        'area': DataType.MEASUREMENT,
        'volume': DataType.MEASUREMENT,
    }

    # MetricCategory mapping for dimensions
    DIMENSION_TO_CATEGORY: Dict[str, MetricCategory] = {
        'pressure': MetricCategory.MECHANICAL_PROPERTY,
        'force': MetricCategory.MECHANICAL_PROPERTY,
        'energy': MetricCategory.ENERGY_PROPERTY,
        'power': MetricCategory.ENERGY_PROPERTY,
        'length': MetricCategory.DIMENSIONAL,
        'mass': MetricCategory.MATERIAL_PROPERTY,
        'frequency': MetricCategory.ELECTRICAL_PROPERTY,
        'voltage': MetricCategory.ELECTRICAL_PROPERTY,
        'current': MetricCategory.ELECTRICAL_PROPERTY,
        'resistance': MetricCategory.ELECTRICAL_PROPERTY,
        'capacitance': MetricCategory.ELECTRICAL_PROPERTY,
        'viscosity': MetricCategory.MATERIAL_PROPERTY,
        'density': MetricCategory.MATERIAL_PROPERTY,
        'velocity': MetricCategory.MECHANICAL_PROPERTY,
        'thermal_conductivity': MetricCategory.THERMAL_PROPERTY,
        'area': MetricCategory.DIMENSIONAL,
        'volume': MetricCategory.DIMENSIONAL,
    }

    def __init__(self, enable_pint: bool = True):
        """
        Initialize UnitConverter.

        Args:
            enable_pint: Whether to enable pint for dimensional analysis (Level 3)
        """
        # Build reverse lookup: unit string → dimension
        self._unit_to_dimension = {}
        for dimension, units in self.CONVERSION_TABLES.items():
            for unit_str in units:
                self._unit_to_dimension[unit_str] = dimension

        # Initialize pint if available and enabled
        self._ureg = None
        if enable_pint:
            try:
                import pint
                self._ureg = pint.UnitRegistry()
                logger.debug("pint UnitRegistry initialized for dimensional analysis")
            except ImportError:
                logger.info("pint not installed; Level 3 dimensional analysis disabled")

    def normalize_si_prefix(self, value: float, unit: str) -> Tuple[float, str]:
        """
        Level 1: Normalize SI prefix to base unit.

        Args:
            value: Numeric value (e.g., 10)
            unit: Unit with prefix (e.g., "GPa")

        Returns:
            (normalized_value, base_unit) e.g., (10e9, "Pa")
        """
        if not unit:
            return value, unit

        # Try matching prefix + base unit
        for base_symbol in sorted(self.SI_BASE_UNITS.keys(), key=len, reverse=True):
            if unit.endswith(base_symbol) and len(unit) > len(base_symbol):
                prefix = unit[:-len(base_symbol)]
                if prefix in self.SI_PREFIXES:
                    multiplier = self.SI_PREFIXES[prefix]
                    return value * multiplier, base_symbol

        return value, unit

    def convert_to_si_base(self, value: float, unit: str) -> Tuple[float, str, str]:
        """
        Level 2: Convert to SI base unit using conversion tables.

        Args:
            value: Numeric value
            unit: Unit string (e.g., "psi", "kgf", "cal")

        Returns:
            (converted_value, si_base_unit, dimension) e.g., (6894.757, "Pa", "pressure")
        """
        # First try direct lookup in conversion tables
        dimension = self._unit_to_dimension.get(unit)
        if dimension:
            table = self.CONVERSION_TABLES[dimension]
            factor = table.get(unit, 1.0)
            # Find the base unit (factor == 1.0)
            base_unit = unit
            for u, f in table.items():
                if f == 1.0:
                    base_unit = u
                    break
            return value * factor, base_unit, dimension

        # Try with SI prefix normalization first, then lookup
        norm_value, norm_unit = self.normalize_si_prefix(value, unit)
        dimension = self._unit_to_dimension.get(norm_unit)
        if dimension:
            table = self.CONVERSION_TABLES[dimension]
            factor = table.get(norm_unit, 1.0)
            base_unit = norm_unit
            for u, f in table.items():
                if f == 1.0:
                    base_unit = u
                    break
            return norm_value * factor, base_unit, dimension

        return value, unit, ""

    def convert_unit(
        self, value: float, from_unit: str, to_unit: str
    ) -> Optional[float]:
        """
        Convert between any two compatible units.

        Uses Level 2 tables first, falls back to pint (Level 3).

        Args:
            value: Numeric value
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value or None if conversion failed
        """
        # Level 2: Try table-based conversion
        from_dim = self._unit_to_dimension.get(from_unit)
        to_dim = self._unit_to_dimension.get(to_unit)

        if from_dim and to_dim and from_dim == to_dim:
            table = self.CONVERSION_TABLES[from_dim]
            from_factor = table.get(from_unit, None)
            to_factor = table.get(to_unit, None)
            if from_factor is not None and to_factor is not None:
                # Convert: from → base → to
                return value * from_factor / to_factor

        # Level 3: Use pint for dimensional analysis
        if self._ureg is not None:
            try:
                quantity = self._ureg.Quantity(value, self._parse_pint_unit(from_unit))
                result = quantity.to(self._parse_pint_unit(to_unit))
                return result.magnitude
            except Exception as e:
                logger.debug(f"pint conversion failed ({from_unit} → {to_unit}): {e}")

        return None

    def decompose_to_base_si(self, value: float, unit: str) -> Optional[Dict[str, Any]]:
        """
        Level 3: Decompose compound unit to base SI dimensions using pint.

        e.g., "Pa" → "kg·m⁻¹·s⁻²"

        Args:
            value: Numeric value
            unit: Unit string

        Returns:
            Dict with decomposed info, or None if pint unavailable
        """
        if self._ureg is None:
            return None

        try:
            pint_unit = self._parse_pint_unit(unit)
            quantity = self._ureg.Quantity(value, pint_unit)
            base = quantity.to_base_units()
            return {
                'value': float(base.magnitude),
                'unit': str(base.units),
                'dimensionality': str(base.dimensionality),
                'original_value': value,
                'original_unit': unit,
            }
        except Exception as e:
            logger.debug(f"pint decomposition failed for {unit}: {e}")
            return None

    def get_dimension(self, unit: str) -> str:
        """Get the physical dimension for a unit string."""
        # Direct lookup
        dim = self._unit_to_dimension.get(unit)
        if dim:
            return dim

        # Try after prefix normalization
        _, base_unit = self.normalize_si_prefix(1.0, unit)
        dim = self._unit_to_dimension.get(base_unit)
        if dim:
            return dim

        # Use pint for unknown units
        if self._ureg is not None:
            try:
                pint_unit = self._parse_pint_unit(unit)
                q = self._ureg.Quantity(1, pint_unit)
                dim_str = str(q.dimensionality)
                # Map pint dimensionality to our dimension names
                dim_map = {
                    '[length] ** 2 / [time] ** 2': 'energy',
                    '[mass] / [length] / [time] ** 2': 'pressure',
                    '[mass] * [length] / [time] ** 2': 'force',
                    '[mass] * [length] ** 2 / [time] ** 2': 'energy',
                    '[mass] * [length] ** 2 / [time] ** 3': 'power',
                    '[length]': 'length',
                    '[mass]': 'mass',
                    '1 / [time]': 'frequency',
                }
                return dim_map.get(dim_str, "")
            except Exception:
                pass

        return ""

    def get_data_type(self, unit: str) -> DataType:
        """Get DataType for a unit."""
        dim = self.get_dimension(unit)
        return self.DIMENSION_TO_DATATYPE.get(dim, DataType.MEASUREMENT)

    def get_metric_category(self, unit: str) -> MetricCategory:
        """Get MetricCategory for a unit."""
        dim = self.get_dimension(unit)
        return self.DIMENSION_TO_CATEGORY.get(dim, MetricCategory.OTHER)

    def convert_temperature(
        self, value: float, from_unit: str, to_unit: str = "K"
    ) -> Optional[float]:
        """
        Convert temperature (special case: offset conversions).

        Args:
            value: Temperature value
            from_unit: Source unit (K, °C, C, °F, F)
            to_unit: Target unit (default: K)

        Returns:
            Converted value or None
        """
        # Normalize unit names
        from_u = from_unit.replace("°", "").strip().upper()
        to_u = to_unit.replace("°", "").strip().upper()

        # Convert to Kelvin first
        if from_u == 'K':
            kelvin = value
        elif from_u == 'C':
            kelvin = value + 273.15
        elif from_u == 'F':
            kelvin = (value - 32) * 5 / 9 + 273.15
        else:
            return None

        # Convert from Kelvin to target
        if to_u == 'K':
            return kelvin
        elif to_u == 'C':
            return kelvin - 273.15
        elif to_u == 'F':
            return (kelvin - 273.15) * 9 / 5 + 32
        else:
            return None

    def _parse_pint_unit(self, unit: str) -> str:
        """
        Convert our unit notation to pint-compatible notation.

        Handles common replacements like · → *, ² → **2, etc.
        """
        # Replace special characters
        pint_unit = unit
        pint_unit = pint_unit.replace('·', ' * ')
        pint_unit = pint_unit.replace('⁻¹', '**-1')
        pint_unit = pint_unit.replace('⁻²', '**-2')
        pint_unit = pint_unit.replace('⁻³', '**-3')
        pint_unit = pint_unit.replace('²', '**2')
        pint_unit = pint_unit.replace('³', '**3')
        pint_unit = pint_unit.replace('μ', 'u')
        pint_unit = pint_unit.replace('Ω', 'ohm')

        # Handle special compound units
        special_map = {
            'Pa*s': 'pascal * second',
            'Pa·s': 'pascal * second',
            'mPa·s': 'millipascal * second',
            'mPa*s': 'millipascal * second',
            'W/(m·K)': 'watt / meter / kelvin',
            'W/(m*K)': 'watt / meter / kelvin',
            'W/mK': 'watt / meter / kelvin',
            'kg/m³': 'kilogram / meter**3',
            'kg/m3': 'kilogram / meter**3',
            'g/cm³': 'gram / centimeter**3',
            'g/cm3': 'gram / centimeter**3',
            'g/mL': 'gram / milliliter',
            'kg/L': 'kilogram / liter',
            'kgf': 'kilogram * force',
            'lbf': 'pound * force',
            'ft·lbf': 'foot * pound_force',
        }

        if unit in special_map:
            return special_map[unit]

        return pint_unit

    def is_compatible(self, unit1: str, unit2: str) -> bool:
        """Check if two units are dimensionally compatible."""
        dim1 = self.get_dimension(unit1)
        dim2 = self.get_dimension(unit2)

        if dim1 and dim2:
            return dim1 == dim2

        # Try pint
        if self._ureg is not None:
            try:
                q1 = self._ureg.Quantity(1, self._parse_pint_unit(unit1))
                q2 = self._ureg.Quantity(1, self._parse_pint_unit(unit2))
                return q1.is_compatible_with(q2)
            except Exception:
                pass

        return False


class NumericalDataExtractor:
    """
    Extracts numerical data from text content during research.

    Uses LLM for intelligent extraction with context understanding,
    with pattern-based fallback for reliability.
    Supports both business/financial and scientific/engineering data.
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

    # Scientific/engineering measurement patterns
    # Format: (pattern, capturing_group_for_value, unit_group_or_fixed_unit, data_type)
    MEASUREMENT_PATTERNS = [
        # Pressure: 10 GPa, 100 MPa, 14.7 psi, 1.013 bar, 1 atm
        (r'([\d,]+\.?\d*)\s*(Y|Z|E|P|T|G|M|k|h|da|d|c|m|μ|u|n|p|f|a)?Pa\b', 1, 'Pa', DataType.PRESSURE),
        (r'([\d,]+\.?\d*)\s*(psi|ksi|bar|mbar|atm|torr|mmHg)\b', 1, None, DataType.PRESSURE),
        # Force: 10 kN, 5 kgf, 100 lbf
        (r'([\d,]+\.?\d*)\s*(Y|Z|E|P|T|G|M|k|h|da|d|c|m|μ|u|n|p|f|a)?N\b', 1, 'N', DataType.FORCE),
        (r'([\d,]+\.?\d*)\s*(kgf|gf|lbf|ozf|kip|dyn)\b', 1, None, DataType.FORCE),
        # Energy: 500 kJ, 100 cal, 3.6 MJ
        (r'([\d,]+\.?\d*)\s*(Y|Z|E|P|T|G|M|k|h|da|d|c|m|μ|u|n|p|f|a)?J\b', 1, 'J', DataType.ENERGY),
        (r'([\d,]+\.?\d*)\s*(cal|kcal|BTU|erg|Wh|kWh|MWh)\b', 1, None, DataType.ENERGY),
        (r'([\d,]+\.?\d*)\s*(Y|Z|E|P|T|G|M|k|h|da|d|c|m|μ|u|n|p|f|a)?eV\b', 1, 'eV', DataType.ENERGY),
        # Power: 100 W, 1.5 kW, 500 MW, 10 hp
        (r'([\d,]+\.?\d*)\s*(Y|Z|E|P|T|G|M|k|h|da|d|c|m|μ|u|n|p|f|a)?W\b', 1, 'W', DataType.POWER),
        (r'([\d,]+\.?\d*)\s*(hp|PS)\b', 1, None, DataType.POWER),
        # Frequency: 2.4 GHz, 50 Hz, 100 kHz
        (r'([\d,]+\.?\d*)\s*(Y|Z|E|P|T|G|M|k|h|da|d|c|m|μ|u|n|p|f|a)?Hz\b', 1, 'Hz', DataType.FREQUENCY),
        (r'([\d,]+\.?\d*)\s*rpm\b', 1, 'rpm', DataType.FREQUENCY),
        # Voltage: 3.3 V, 220 V, 12 kV
        (r'([\d,]+\.?\d*)\s*(Y|Z|E|P|T|G|M|k|h|da|d|c|m|μ|u|n|p|f|a)?V\b', 1, 'V', DataType.VOLTAGE),
        # Current: 10 mA, 2 A, 100 μA (negative lookahead to avoid A.U./a.u.)
        (r'([\d,]+\.?\d*)\s*(Y|Z|E|P|T|G|M|k|h|da|d|c|m|μ|u|n|p|f|a)?A\b(?!\.?[Uu]\.?)', 1, 'A', DataType.CURRENT),
        # Resistance: 100 Ω, 4.7 kΩ, 1 MΩ
        (r'([\d,]+\.?\d*)\s*(Y|Z|E|P|T|G|M|k|h|da|d|c|m|μ|u|n|p|f|a)?(?:Ω|ohm)\b', 1, 'Ω', DataType.RESISTANCE),
        # Capacitance: 100 μF, 10 nF, 22 pF
        (r'([\d,]+\.?\d*)\s*(m|μ|u|n|p|f)?F\b', 1, 'F', DataType.CAPACITANCE),
        # Temperature: 300 K, 25 °C, 77 °F
        (r'([\d,]+\.?\d*)\s*°?K\b', 1, 'K', DataType.TEMPERATURE),
        (r'([\d,]+\.?\d*)\s*°C\b', 1, '°C', DataType.TEMPERATURE),
        (r'([\d,]+\.?\d*)\s*°F\b', 1, '°F', DataType.TEMPERATURE),
        # Length: 10 nm, 1.5 mm, 100 μm, 5 Å
        (r'([\d,]+\.?\d*)\s*(Y|Z|E|P|T|G|M|k|h|da|d|c|m|μ|u|n|p|f|a)?m\b(?!/)', 1, 'm', DataType.LENGTH),
        (r'([\d,]+\.?\d*)\s*Å\b', 1, 'Å', DataType.LENGTH),
        (r'([\d,]+\.?\d*)\s*(in|ft|yd|mi|mil|thou)\b', 1, None, DataType.LENGTH),
        # Mass: 5 kg, 100 mg, 2.5 g
        (r'([\d,]+\.?\d*)\s*(k|m|μ|u|n|p)?g\b', 1, 'g', DataType.MASS),
        (r'([\d,]+\.?\d*)\s*(lb|oz|slug|grain)\b', 1, None, DataType.MASS),
        # Density: 7.8 g/cm³, 1000 kg/m³
        (r'([\d,]+\.?\d*)\s*(kg/m[³3]|g/cm[³3]|g/mL|kg/L|lb/ft[³3]|lb/in[³3])\b', 1, None, DataType.DENSITY),
        # Velocity: 340 m/s, 100 km/h
        (r'([\d,]+\.?\d*)\s*(m/s|km/h|km/s|mph|ft/s|knot|kn)\b', 1, None, DataType.VELOCITY),
        # Viscosity: 1 Pa·s, 10 mPa·s, 1 cP
        (r'([\d,]+\.?\d*)\s*(m)?Pa[·*]s\b', 1, None, DataType.VISCOSITY),
        (r'([\d,]+\.?\d*)\s*(cP|P|cSt)\b', 1, None, DataType.VISCOSITY),
        # Thermal conductivity: 400 W/(m·K)
        (r'([\d,]+\.?\d*)\s*W/\(?m[·*]?K\)?\b', 1, 'W/(m·K)', DataType.THERMAL_CONDUCTIVITY),
        # Arbitrary/relative units: 100 a.u., 0.5 arb., 1.2 rel.
        (r'([\d,]+\.?\d*)\s*a\.?u\.?\b', 1, 'a.u.', DataType.ARBITRARY),
        (r'([\d,]+\.?\d*)\s*A\.?U\.?\b', 1, 'a.u.', DataType.ARBITRARY),
        (r'([\d,]+\.?\d*)\s*arb\.?\s*(?:units?)?', 1, 'arb.', DataType.ARBITRARY),
        (r'([\d,]+\.?\d*)\s*(?:arbitrary\s*units?)', 1, 'a.u.', DataType.ARBITRARY),
        (r'([\d,]+\.?\d*)\s*rel\.?\b', 1, 'rel.', DataType.ARBITRARY),
        (r'([\d,]+\.?\d*)\s*norm\.?\b', 1, 'norm.', DataType.ARBITRARY),
        # Japanese: 100単位, 100ユニット (no \b for Japanese)
        (r'([\d,]+\.?\d*)\s*単位', 1, '単位', DataType.ARBITRARY),
        (r'([\d,]+\.?\d*)\s*ユニット', 1, 'unit', DataType.ARBITRARY),
        # Generic units: 100 units, 50 U (common in biology/chemistry)
        (r'([\d,]+\.?\d*)\s*units?\b', 1, 'unit', DataType.ARBITRARY),
        (r'([\d,]+\.?\d*)\s*U\b(?!/)', 1, 'U', DataType.ARBITRARY),  # (?!/) to avoid m/s etc
        (r'([\d,]+\.?\d*)\s*IU\b', 1, 'IU', DataType.ARBITRARY),  # International Units
    ]

    def __init__(
        self,
        llm_client=None,
        language: str = "ja",
        min_confidence: float = 0.5,
        use_llm: bool = True,
        enable_unit_conversion: bool = True,
        enable_pint: bool = True,
    ):
        """
        Initialize extractor.

        Args:
            llm_client: LLM client for intelligent extraction
            language: Output language
            min_confidence: Minimum confidence to keep data point
            use_llm: Whether to use LLM (False = pattern-only)
            enable_unit_conversion: Whether to enable unit conversion
            enable_pint: Whether to enable pint dimensional analysis
        """
        self.llm = llm_client
        self.language = language
        self.min_confidence = min_confidence
        self.use_llm = use_llm and llm_client is not None
        self.unit_converter = UnitConverter(enable_pint=enable_pint) if enable_unit_conversion else None

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

【抽出対象 - ビジネス/金融】
- 市場規模、売上、利益などの金額
- 成長率、シェア、割合などのパーセンテージ
- ユーザー数、販売数などの数量
- ランキング、スコア

【抽出対象 - 科学/工学】
- 力学的数値（引張強度 MPa/GPa、ヤング率、硬度 HV/HRC）
- 物理量（圧力 Pa/bar/atm、力 N/kN、エネルギー J/kJ）
- 電気的特性（電圧 V、電流 A、抵抗 Ω、静電容量 F）
- 材料特性（密度 kg/m³、熱伝導率 W/(m·K)、粘度 Pa·s）
- 寸法・サイズ（長さ nm/μm/mm/m、面積、体積）
- 温度（K、°C、°F）
- 周波数（Hz、kHz、GHz）
- 任意単位（a.u., arb., rel., norm.）- スペクトル強度等の相対値
- 単位付きの全ての定量データ

【重要】SI接頭辞付き単位（GPa、kN、MHz等）もそのまま抽出してください。

【各データポイントについて以下を特定】
1. value: 数値（数字のみ、例: 10 GPaの場合→ 10）
2. raw_text: 元のテキスト表現（例: "10 GPa"）
3. unit: 単位（GPa, MPa, Pa, kN, N, kJ, J, W/(m·K), kg/m³, %, USD等）
4. metric_name: 何の指標か（例: "引張強度", "ヤング率", "市場規模", "密度"）
5. subject: 何についてか（例: "アルミニウム合金", "AI市場", "SUS304"）
6. year: 年（あれば、例: 2024）
7. is_forecast: 予測値かどうか（true/false）
8. data_type: currency/percentage/count/ratio/rate/time_series/measurement/stress/strain/force/energy/power/temperature/pressure/length/mass/velocity/density/frequency/voltage/current/resistance/capacitance/viscosity/thermal_conductivity/arbitrary/dimensionless/other
9. category: market_size/growth_rate/market_share/revenue/profit/user_count/price/performance/forecast/comparison/mechanical_property/thermal_property/electrical_property/material_property/physical_constant/environmental/chemical_property/dimensional/energy_property/relative_value/other
10. confidence: 抽出の確信度（0.0-1.0）

JSON配列形式で出力してください。数値データがない場合は空配列[]を返してください。

テキスト:
{content}

出力形式:
```json
[
  {{
    "value": 10,
    "raw_text": "10 GPa",
    "unit": "GPa",
    "metric_name": "ヤング率",
    "subject": "チタン合金",
    "year": null,
    "is_forecast": false,
    "data_type": "stress",
    "category": "mechanical_property",
    "confidence": 0.9
  }},
  {{
    "value": 50,
    "raw_text": "50億ドル",
    "unit": "billion USD",
    "metric_name": "市場規模",
    "subject": "半導体市場",
    "year": 2024,
    "is_forecast": false,
    "data_type": "currency",
    "category": "market_size",
    "confidence": 0.85
  }}
]
```"""
        else:
            prompt = f"""Extract numerical data suitable for charts and tables from the following text.

Research Topic: {research_topic}

【Data to Extract - Business/Financial】
- Market size, revenue, profit (currency)
- Growth rates, shares, percentages
- User counts, sales quantities
- Rankings, scores

【Data to Extract - Science/Engineering】
- Mechanical values (tensile strength MPa/GPa, Young's modulus, hardness HV/HRC)
- Physical quantities (pressure Pa/bar/atm, force N/kN, energy J/kJ)
- Electrical properties (voltage V, current A, resistance Ω, capacitance F)
- Material properties (density kg/m³, thermal conductivity W/(m·K), viscosity Pa·s)
- Dimensions/sizes (length nm/μm/mm/m, area, volume)
- Temperature (K, °C, °F)
- Frequency (Hz, kHz, GHz)
- Arbitrary units (a.u., arb., rel., norm.) - relative values like spectral intensity
- All quantitative data with units

【Important】Extract values with SI prefixed units (GPa, kN, MHz, etc.) as-is.

【For each data point, identify】
1. value: Numeric value only (e.g., for 10 GPa → 10)
2. raw_text: Original text representation (e.g., "10 GPa")
3. unit: Unit (GPa, MPa, Pa, kN, N, kJ, J, W/(m·K), kg/m³, %, USD, etc.)
4. metric_name: What metric (e.g., "tensile strength", "Young's modulus", "market size", "density")
5. subject: What entity (e.g., "aluminum alloy", "AI market", "SUS304")
6. year: Year (if available)
7. is_forecast: Whether this is a forecast (true/false)
8. data_type: currency/percentage/count/ratio/rate/time_series/measurement/stress/strain/force/energy/power/temperature/pressure/length/mass/velocity/density/frequency/voltage/current/resistance/capacitance/viscosity/thermal_conductivity/arbitrary/dimensionless/other
9. category: market_size/growth_rate/market_share/revenue/profit/user_count/price/performance/forecast/comparison/mechanical_property/thermal_property/electrical_property/material_property/physical_constant/environmental/chemical_property/dimensional/energy_property/relative_value/other
10. confidence: Extraction confidence (0.0-1.0)

Output as JSON array. Return empty array [] if no numerical data found.

Text:
{content}

Output format:
```json
[
  {{
    "value": 10,
    "raw_text": "10 GPa",
    "unit": "GPa",
    "metric_name": "Young's modulus",
    "subject": "titanium alloy",
    "year": null,
    "is_forecast": false,
    "data_type": "stress",
    "category": "mechanical_property",
    "confidence": 0.9
  }},
  {{
    "value": 50,
    "raw_text": "$50 billion",
    "unit": "billion USD",
    "metric_name": "market size",
    "subject": "semiconductor market",
    "year": 2024,
    "is_forecast": false,
    "data_type": "currency",
    "category": "market_size",
    "confidence": 0.85
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
                try:
                    raw_unit = item.get("unit", "")
                    raw_value = float(item.get("value", 0))
                    data_type_str = item.get("data_type", "other")
                    category_str = item.get("category", "other")

                    # Normalize value using unit converter
                    normalized_value = self._normalize_value(raw_value, raw_unit)

                    # Validate enum values
                    try:
                        data_type = DataType(data_type_str)
                    except ValueError:
                        data_type = DataType.OTHER

                    try:
                        category = MetricCategory(category_str)
                    except ValueError:
                        category = MetricCategory.OTHER

                    dp = NumericalDataPoint(
                        value=raw_value,
                        raw_text=item.get("raw_text", ""),
                        normalized_value=normalized_value,
                        unit=raw_unit,
                        metric_name=item.get("metric_name", ""),
                        subject=item.get("subject", ""),
                        data_type=data_type,
                        category=category,
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
                except Exception as e:
                    logger.debug(f"Failed to parse LLM data point: {e}")
                    continue

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

        # Normalize full-width numbers to half-width for pattern matching
        # Keep original content for raw_text extraction
        normalized_content = CurrencyNormalizer.normalize_number(content)

        # Extract years for context
        years = set(re.findall(r'\b(20\d{2})\b', normalized_content))
        default_year = max(int(y) for y in years) if years else None

        # Currency extraction (use normalized_content for matching)
        for pattern in self.CURRENCY_PATTERNS:
            for match in re.finditer(pattern, normalized_content, re.IGNORECASE):
                try:
                    # Get surrounding context (50 chars each side)
                    start = max(0, match.start() - 50)
                    end = min(len(normalized_content), match.end() + 50)
                    context = normalized_content[start:end]

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

        # Percentage extraction (use normalized_content for matching)
        for pattern in self.PERCENTAGE_PATTERNS:
            for match in re.finditer(pattern, normalized_content, re.IGNORECASE):
                try:
                    raw_text = match.group(0)
                    # Number already normalized, no need for replace(',', '')
                    value = float(match.group(1))

                    # Get context for year
                    start = max(0, match.start() - 50)
                    end = min(len(normalized_content), match.end() + 50)
                    context = normalized_content[start:end]
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

        # Scientific/engineering measurement extraction (use normalized_content)
        data_points.extend(self._extract_measurements(
            content=normalized_content,
            source_url=source_url,
            source_title=source_title,
            evidence_id=evidence_id,
            section_id=section_id,
            source_reliability=source_reliability,
            default_year=default_year,
        ))

        return data_points

    def _extract_measurements(
        self,
        content: str,
        source_url: str,
        source_title: str,
        evidence_id: str,
        section_id: str,
        source_reliability: float,
        default_year: Optional[int] = None,
    ) -> List[NumericalDataPoint]:
        """
        Extract scientific/engineering measurements using patterns.

        Handles SI prefixed units, non-SI units, and compound units.
        Expects content to be pre-normalized (commas removed, full-width converted).
        """
        data_points = []

        for pattern_tuple in self.MEASUREMENT_PATTERNS:
            pattern, val_group, base_unit, data_type = pattern_tuple

            for match in re.finditer(pattern, content):
                try:
                    raw_text = match.group(0)
                    # Content is already normalized, no need for replace(',', '')
                    value_str = match.group(val_group)
                    value = float(value_str)

                    # Determine the full unit string
                    if base_unit is not None:
                        # Pattern has SI prefix group (group 2)
                        prefix = match.group(2) if match.lastindex >= 2 and match.group(2) else ""
                        unit = prefix + base_unit
                    else:
                        # Unit is captured directly in the match
                        # Find the unit part after the number (content is normalized, no commas)
                        unit_match = re.search(r'[\d]+\.?\d*\s*(.*)', raw_text)
                        unit = unit_match.group(1).strip() if unit_match else ""

                    if not unit:
                        continue

                    # Normalize to base SI unit
                    normalized_value = self._normalize_value(value, unit)

                    # Get context
                    start = max(0, match.start() - 80)
                    end = min(len(content), match.end() + 80)
                    context = content[start:end]

                    # Find year in context
                    year_match = re.search(r'\b(20\d{2})\b', context)
                    year = int(year_match.group(1)) if year_match else default_year

                    # Determine category from data type or unit converter
                    if data_type == DataType.ARBITRARY:
                        category = MetricCategory.RELATIVE_VALUE
                    elif self.unit_converter:
                        category = self.unit_converter.get_metric_category(unit)
                    else:
                        category = MetricCategory.OTHER

                    dp = NumericalDataPoint(
                        value=value,
                        raw_text=raw_text,
                        normalized_value=normalized_value,
                        unit=unit,
                        metric_name="",  # Pattern can't determine this
                        subject="",
                        data_type=data_type,
                        category=category,
                        year=year,
                        source_url=source_url,
                        source_title=source_title,
                        evidence_id=evidence_id,
                        section_id=section_id,
                        extraction_confidence=0.65,  # Pattern-based measurement extraction
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
        """
        Normalize value to base SI unit.

        Uses UnitConverter if available, falls back to simple multipliers.
        """
        if self.unit_converter:
            converted, base_unit, dimension = self.unit_converter.convert_to_si_base(value, unit)
            if dimension:
                return converted

        # Fallback: simple business multipliers
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
        """Parse currency value from text with full normalization."""
        # Normalize full-width numbers first
        normalized_text = CurrencyNormalizer.normalize_number(text)

        # Remove currency symbols
        normalized_text = re.sub(r'[\$€¥£￥₩₹₽฿]', '', normalized_text)

        # Find number (now without commas after normalization)
        num_match = re.search(r'([\d]+\.?\d*)', normalized_text)
        if not num_match:
            return 0.0

        value = float(num_match.group(1))

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

    def _detect_currency_unit(self, text: str, normalize: bool = True) -> str:
        """
        Detect currency unit from text.

        Args:
            text: Text containing currency (e.g., "10€", "100ユーロ", "$50 billion")
            normalize: If True, return ISO 4217 code (EUR). If False, return with scale (billion USD).

        Returns:
            Currency unit string, normalized to ISO code if normalize=True
        """
        text_lower = text.lower()

        # Determine scale multiplier suffix
        scale_suffix = ""
        if not normalize:
            if 'trillion' in text_lower or '兆' in text:
                scale_suffix = " trillion"
            elif 'billion' in text_lower or '億' in text:
                scale_suffix = " billion" if 'billion' in text_lower else "億"
            elif 'million' in text_lower or '百万' in text:
                scale_suffix = " million"
            elif '万' in text:
                scale_suffix = "万"

        # Try to detect currency using CurrencyNormalizer
        for symbol in ['$', 'US$', 'A$', 'C$', 'S$', 'HK$', 'NT$', 'R$',
                       '€', '£', '¥', '￥', '₩', '₹', '₽', '฿']:
            if symbol in text:
                iso_code = CurrencyNormalizer.normalize_currency(symbol)
                if iso_code:
                    if normalize:
                        return iso_code
                    else:
                        # For JPY, use Japanese format
                        if iso_code == 'JPY' and scale_suffix:
                            return scale_suffix + '円'
                        return scale_suffix.strip() + ' ' + iso_code if scale_suffix else iso_code

        # Try currency names (Japanese/English)
        for name in ['ドル', '米ドル', 'dollar', 'dollars']:
            if name in text or name in text_lower:
                if normalize:
                    return 'USD'
                return (scale_suffix.strip() + ' USD').strip() if scale_suffix else 'USD'

        for name in ['ユーロ', 'euro', 'euros']:
            if name in text or name in text_lower:
                if normalize:
                    return 'EUR'
                return (scale_suffix.strip() + ' EUR').strip() if scale_suffix else 'EUR'

        for name in ['円', 'yen']:
            if name in text or name in text_lower:
                if normalize:
                    return 'JPY'
                return scale_suffix + '円' if scale_suffix else '円'

        for name in ['ポンド', 'pound', 'pounds']:
            if name in text or name in text_lower:
                if normalize:
                    return 'GBP'
                return (scale_suffix.strip() + ' GBP').strip() if scale_suffix else 'GBP'

        for name in ['元', '人民元', 'yuan']:
            if name in text or name in text_lower:
                if normalize:
                    return 'CNY'
                return (scale_suffix.strip() + ' CNY').strip() if scale_suffix else 'CNY'

        # Default to USD
        return 'USD'


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
