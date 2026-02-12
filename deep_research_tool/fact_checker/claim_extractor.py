"""
Claim extractor for identifying verifiable claims from sentences.

Uses a hybrid approach:
- Rule-based pattern matching for claim indicators (reliable, fast)
- LLM-based extraction for detailed claim analysis (accurate, slower)

Claim types:
- Factual: Objective statements that can be verified
- Statistical: Claims with numbers, percentages, data
- Quotation: Attributed quotes
- Temporal: Date/time-related claims
- Causal: Cause-effect relationships
- Comparative: Comparisons between entities
"""

import re
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

from .sentence_splitter import Sentence


class ClaimType(str, Enum):
    """Types of verifiable claims."""
    FACTUAL = "factual"
    STATISTICAL = "statistical"
    QUOTATION = "quotation"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    DEFINITION = "definition"
    OPINION = "opinion"  # Not verifiable, but identified
    UNKNOWN = "unknown"


class ClaimStrength(str, Enum):
    """Strength/certainty of the claim."""
    STRONG = "strong"  # Definitive statements
    MODERATE = "moderate"  # Qualified statements
    WEAK = "weak"  # Hedged statements
    UNCERTAIN = "uncertain"


@dataclass
class Claim:
    """A single claim extracted from a sentence."""
    text: str
    claim_type: ClaimType
    strength: ClaimStrength
    source_sentence: Sentence
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    is_verifiable: bool = True
    search_queries: List[str] = field(default_factory=list)
    confidence: float = 0.0  # Extraction confidence
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "claim_type": self.claim_type.value,
            "strength": self.strength.value,
            "source_sentence_index": self.source_sentence.index,
            "source_text": self.source_sentence.text,
            "entities": self.entities,
            "keywords": self.keywords,
            "is_verifiable": self.is_verifiable,
            "search_queries": self.search_queries,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class ClaimExtractor:
    """
    Extract verifiable claims from sentences.

    Uses pattern-based detection to identify claim indicators,
    then LLM for detailed claim extraction and search query generation.
    """

    # Patterns indicating statistical claims
    STATISTICAL_PATTERNS = [
        r'\d+(?:\.\d+)?%',  # Percentages
        r'\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:万|億|兆|百|千|人|円|ドル|個|件|回|年))',  # Japanese numbers
        r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:million|billion|trillion|thousand|hundred)\b',  # English numbers
        r'(?:約|およそ|nearly|about|approximately|around)\s*\d+',  # Approximate numbers
        r'\b(?:increased|decreased|grew|fell|rose|dropped)\s+(?:by\s+)?\d+',  # Growth/decline
        r'(?:増加|減少|上昇|下落|成長).*\d+',  # Japanese growth/decline
    ]

    # Patterns indicating temporal claims
    TEMPORAL_PATTERNS = [
        r'(?:^|(?<=\s)|(?<=[^\d]))(\d{4})年',  # Japanese year (no \b needed)
        r'(?:^|\s)(?:19|20)\d{2}(?:\s|$|[,.\)])',  # Western year
        r'\d{1,2}月\d{1,2}日',  # Japanese date
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?',
        r'\b(?:first|second|third|last|next|previous)\s+(?:year|month|week|day|century|decade)\b',
        r'(?:昨年|今年|来年|先月|今月|来月|昨日|今日|明日)',
    ]

    # Patterns indicating causal claims
    CAUSAL_PATTERNS = [
        r'\b(?:because|due to|caused by|leads to|results in|therefore|consequently|as a result)\b',
        r'(?:なぜなら|ため|によって|原因|結果|したがって|そのため|だから)',
        r'\b(?:if|when|unless|provided that)\b.*(?:then|will|would)',
    ]

    # Patterns indicating comparative claims
    COMPARATIVE_PATTERNS = [
        r'\b(?:more|less|better|worse|higher|lower|larger|smaller|faster|slower)\s+than\b',
        r'\b(?:most|least|best|worst|highest|lowest|largest|smallest)\b',
        r'(?:より|もっと|最も|一番|比べ)',
    ]

    # Patterns indicating quotations
    QUOTATION_PATTERNS = [
        r'[「『""](.*?)[」』""]',  # Quoted text
        r'(?:said|stated|claimed|reported|announced|argued|wrote)(?:\s+that)?',
        r'(?:と述べた|と語った|と発表した|によると|によれば)',
    ]

    # Hedge words indicating weak claims
    HEDGE_WORDS = {
        'en': ['might', 'may', 'could', 'possibly', 'perhaps', 'likely', 'probably',
               'seems', 'appears', 'suggests', 'indicates', 'approximately', 'about',
               'around', 'roughly', 'estimated', 'believed', 'thought'],
        'ja': ['かもしれない', 'らしい', 'ようだ', 'みたい', 'おそらく', '多分',
               'と思われる', 'と考えられる', '推定', '約', 'およそ', '程度'],
    }

    # Strong assertion words
    ASSERTION_WORDS = {
        'en': ['is', 'are', 'was', 'were', 'will', 'must', 'always', 'never',
               'definitely', 'certainly', 'proven', 'confirmed', 'established'],
        'ja': ['である', 'です', 'だ', '必ず', '常に', '絶対に', '確実に', '明らかに'],
    }

    def __init__(
        self,
        llm_client=None,
        use_patterns: bool = True,
        extract_search_queries: bool = True,
        language: str = "auto",
    ):
        """
        Initialize claim extractor.

        Args:
            llm_client: LLM client for detailed extraction
            use_patterns: Use pattern-based pre-filtering
            extract_search_queries: Generate search queries for each claim
            language: Language hint (auto, en, ja)
        """
        self.llm_client = llm_client
        self.use_patterns = use_patterns
        self.extract_search_queries = extract_search_queries
        self.language = language

    def extract_claims(self, sentences: List[Sentence]) -> List[Claim]:
        """
        Extract claims from a list of sentences.

        Args:
            sentences: List of Sentence objects

        Returns:
            List of Claim objects
        """
        claims = []

        for sentence in sentences:
            sentence_claims = self._extract_from_sentence(sentence)
            claims.extend(sentence_claims)

        return claims

    def _extract_from_sentence(self, sentence: Sentence) -> List[Claim]:
        """Extract claims from a single sentence."""
        text = sentence.text

        # Step 1: Pattern-based pre-analysis
        detected_types = self._detect_claim_types(text)
        strength = self._detect_claim_strength(text, sentence.language)

        # Skip if appears to be pure opinion with no verifiable content
        if not detected_types and self._is_pure_opinion(text):
            return [Claim(
                text=text,
                claim_type=ClaimType.OPINION,
                strength=strength,
                source_sentence=sentence,
                is_verifiable=False,
                confidence=0.8,
            )]

        # Step 2: Use LLM for detailed extraction
        if self.llm_client:
            return self._extract_with_llm(sentence, detected_types, strength)

        # Fallback: Create basic claim from pattern analysis
        if detected_types:
            primary_type = detected_types[0]
        else:
            primary_type = ClaimType.FACTUAL

        claim = Claim(
            text=text,
            claim_type=primary_type,
            strength=strength,
            source_sentence=sentence,
            keywords=self._extract_keywords(text),
            is_verifiable=True,
            search_queries=[text[:100]] if self.extract_search_queries else [],
            confidence=0.5,
        )

        return [claim]

    def _detect_claim_types(self, text: str) -> List[ClaimType]:
        """Detect claim types using patterns."""
        detected = []

        has_statistical = False
        has_temporal = False

        # Check statistical patterns
        for pattern in self.STATISTICAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                has_statistical = True
                break

        # Check temporal patterns
        for pattern in self.TEMPORAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                has_temporal = True
                break

        # If both statistical and temporal are detected, check if the number
        # is primarily a year reference (e.g. "2021年に開催") rather than a
        # quantity. Year patterns like "YYYY年に" are temporal, not statistical.
        if has_statistical and has_temporal:
            # Check if the statistical match is only a year reference
            stat_matches = []
            for pattern in self.STATISTICAL_PATTERNS:
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    stat_matches.append(m.group())
            year_only = all(re.match(r'^\d{4}年$', m.strip()) for m in stat_matches)
            if year_only:
                # It's a year reference, prioritize temporal
                detected.append(ClaimType.TEMPORAL)
            else:
                detected.append(ClaimType.STATISTICAL)
                detected.append(ClaimType.TEMPORAL)
        elif has_temporal:
            detected.append(ClaimType.TEMPORAL)
        elif has_statistical:
            detected.append(ClaimType.STATISTICAL)

        # Check causal patterns
        for pattern in self.CAUSAL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(ClaimType.CAUSAL)
                break

        # Check comparative patterns
        for pattern in self.COMPARATIVE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(ClaimType.COMPARATIVE)
                break

        # Check quotation patterns
        for pattern in self.QUOTATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(ClaimType.QUOTATION)
                break

        return detected

    def _detect_claim_strength(self, text: str, language: str) -> ClaimStrength:
        """Detect the strength/certainty of a claim."""
        text_lower = text.lower()

        # Check for hedge words
        hedge_count = 0
        assertion_count = 0

        lang = language if language in ['en', 'ja'] else 'en'

        for word in self.HEDGE_WORDS.get(lang, []):
            if word in text_lower:
                hedge_count += 1

        for word in self.ASSERTION_WORDS.get(lang, []):
            if word in text_lower:
                assertion_count += 1

        if hedge_count > assertion_count:
            return ClaimStrength.WEAK if hedge_count > 1 else ClaimStrength.MODERATE
        elif assertion_count > hedge_count:
            return ClaimStrength.STRONG
        else:
            return ClaimStrength.MODERATE

    def _is_pure_opinion(self, text: str) -> bool:
        """Check if text is purely an opinion without verifiable facts."""
        opinion_indicators = [
            r'\bI (?:think|believe|feel|hope)\b',
            r'\bIn my (?:opinion|view)\b',
            r'(?:思う|考える|感じる|願う)',
            r'(?:私見では|個人的には)',
        ]

        for pattern in opinion_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using simple heuristics."""
        # Remove common stopwords and extract significant words
        # This is a simple implementation; could be enhanced with NLP
        words = re.findall(r'\b[A-Za-z\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]{2,}\b', text)

        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'this', 'that', 'these', 'those', 'it', 'its', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'する', 'いる', 'ある', 'なる', 'できる', 'この', 'その', 'あの',
            'これ', 'それ', 'あれ', 'ため', 'こと', 'もの', 'とき', 'ところ',
        }

        keywords = [w for w in words if w.lower() not in stopwords]
        return keywords[:10]  # Limit to 10 keywords

    def _extract_with_llm(
        self,
        sentence: Sentence,
        detected_types: List[ClaimType],
        strength: ClaimStrength,
    ) -> List[Claim]:
        """Use LLM for detailed claim extraction."""
        types_hint = ", ".join([t.value for t in detected_types]) if detected_types else "unknown"

        prompt = f"""Analyze this sentence and extract verifiable claims.

Sentence: {sentence.text}
Language: {sentence.language}
Pre-detected claim types: {types_hint}
Pre-detected strength: {strength.value}

For each verifiable claim:
1. Extract the specific claim text
2. Classify the claim type (factual, statistical, quotation, temporal, causal, comparative, definition, opinion)
3. Identify key entities (people, organizations, places, products)
4. Generate 2-3 search queries to verify this claim
5. Assess if it's actually verifiable

Return as JSON:
{{
    "claims": [
        {{
            "claim_text": "The specific claim",
            "claim_type": "statistical",
            "entities": ["Entity1", "Entity2"],
            "keywords": ["keyword1", "keyword2"],
            "is_verifiable": true,
            "search_queries": ["search query 1", "search query 2"],
            "verification_focus": "What to look for when verifying"
        }}
    ],
    "overall_verifiability": "high/medium/low/none"
}}

If the sentence contains no verifiable claims (pure opinion), return an empty claims array."""

        response = self.llm_client.generate(prompt)

        try:
            content = response.content
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError):
            # Fallback to pattern-based extraction
            primary_type = detected_types[0] if detected_types else ClaimType.FACTUAL
            return [Claim(
                text=sentence.text,
                claim_type=primary_type,
                strength=strength,
                source_sentence=sentence,
                keywords=self._extract_keywords(sentence.text),
                is_verifiable=True,
                search_queries=[sentence.text[:100]],
                confidence=0.5,
            )]

        claims = []
        for claim_data in data.get('claims', []):
            try:
                claim_type = ClaimType(claim_data.get('claim_type', 'factual'))
            except ValueError:
                claim_type = ClaimType.FACTUAL

            claim = Claim(
                text=claim_data.get('claim_text', sentence.text),
                claim_type=claim_type,
                strength=strength,
                source_sentence=sentence,
                entities=claim_data.get('entities', []),
                keywords=claim_data.get('keywords', []),
                is_verifiable=claim_data.get('is_verifiable', True),
                search_queries=claim_data.get('search_queries', []),
                confidence=0.8,
                metadata={'verification_focus': claim_data.get('verification_focus', '')},
            )
            claims.append(claim)

        # If no claims extracted, create one from the sentence
        if not claims:
            verifiability = data.get('overall_verifiability', 'medium')
            is_verifiable = verifiability != 'none'

            claims.append(Claim(
                text=sentence.text,
                claim_type=ClaimType.OPINION if not is_verifiable else ClaimType.FACTUAL,
                strength=strength,
                source_sentence=sentence,
                keywords=self._extract_keywords(sentence.text),
                is_verifiable=is_verifiable,
                search_queries=[sentence.text[:100]] if is_verifiable else [],
                confidence=0.6,
            ))

        return claims

    def generate_search_queries(self, claim: Claim) -> List[str]:
        """Generate search queries for verifying a claim."""
        if claim.search_queries:
            return claim.search_queries

        queries = []

        # Basic query from claim text
        base_query = claim.text[:100]
        queries.append(base_query)

        # Entity-focused queries
        for entity in claim.entities[:3]:
            queries.append(f"{entity} {' '.join(claim.keywords[:3])}")

        # Type-specific queries
        if claim.claim_type == ClaimType.STATISTICAL:
            # Add "statistics" or "data" to help find sources
            queries.append(f"{base_query} statistics data")

        elif claim.claim_type == ClaimType.TEMPORAL:
            # Extract year/date and add to query
            year_match = re.search(r'\b(19|20)\d{2}\b', claim.text)
            if year_match:
                queries.append(f"{' '.join(claim.keywords[:5])} {year_match.group()}")

        elif claim.claim_type == ClaimType.QUOTATION:
            # Try to find the source of the quote
            queries.append(f"{' '.join(claim.entities[:2])} quote statement")

        return queries[:5]  # Limit to 5 queries
