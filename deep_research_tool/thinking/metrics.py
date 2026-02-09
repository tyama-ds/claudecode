"""
Quantitative metrics for DeepThink.

This module provides statistical and mathematical evaluation of reasoning quality,
using embeddings, similarity measures, and contradiction detection.
"""

import re
import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

# Try to import sentence-transformers, fall back to basic metrics if unavailable
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    np = None


@dataclass
class MetricsConfig:
    """Configuration for metrics calculation."""
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"  # Supports Japanese
    fidelity_threshold: float = 0.7
    coherence_threshold: float = 0.6
    expansion_tolerance: float = 0.2
    deviation_weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)  # semantic, logical, contradiction


class DeepThinkMetrics:
    """
    Quantitative metrics calculator for DeepThink.

    Provides mathematical evaluation of:
    - Source fidelity (fact extraction accuracy)
    - Logical coherence (reasoning validity)
    - Information expansion degree
    - Deviation from original facts
    - Contradiction detection
    - Confidence scoring
    """

    def __init__(self, config: MetricsConfig = None):
        """
        Initialize metrics calculator.

        Args:
            config: Metrics configuration
        """
        self.config = config or MetricsConfig()
        self._embedder = None
        self._embedding_cache: Dict[str, List[float]] = {}

    @property
    def embedder(self):
        """Lazy load embedder."""
        if self._embedder is None and HAS_SENTENCE_TRANSFORMERS:
            self._embedder = SentenceTransformer(self.config.embedding_model)
        return self._embedder

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text, with caching."""
        if not HAS_SENTENCE_TRANSFORMERS or self.embedder is None:
            return None

        # Check cache
        cache_key = text[:500]  # Use first 500 chars as key
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        # Compute embedding
        embedding = self.embedder.encode(text, convert_to_numpy=True)
        self._embedding_cache[cache_key] = embedding.tolist()
        return embedding.tolist()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not HAS_SENTENCE_TRANSFORMERS:
            return self._fallback_similarity(vec1, vec2)

        arr1 = np.array(vec1)
        arr2 = np.array(vec2)
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    def _fallback_similarity(self, vec1, vec2) -> float:
        """Fallback similarity when numpy is not available."""
        # Simple dot product approximation
        if not vec1 or not vec2:
            return 0.0
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _tokenize(self, text: str) -> Set[str]:
        """Simple tokenization for vocabulary comparison."""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split into tokens
        tokens = set(text.split())
        # Remove very short tokens
        return {t for t in tokens if len(t) > 1}

    def _jaccard_coefficient(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard coefficient between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def calc_source_fidelity(self, fact_text: str, source_text: str) -> float:
        """
        Calculate source fidelity - how faithfully facts are extracted from source.

        Uses cosine similarity between embeddings of extracted fact and source.
        Higher values indicate more faithful extraction.

        Args:
            fact_text: Extracted fact text
            source_text: Original source text

        Returns:
            Fidelity score between 0 and 1
        """
        # Get embeddings
        fact_emb = self._get_embedding(fact_text)
        source_emb = self._get_embedding(source_text)

        if fact_emb is None or source_emb is None:
            # Fallback to vocabulary overlap
            fact_tokens = self._tokenize(fact_text)
            source_tokens = self._tokenize(source_text)
            return self._jaccard_coefficient(fact_tokens, source_tokens)

        return self._cosine_similarity(fact_emb, source_emb)

    def calc_logical_coherence(
        self,
        premises: List[str],
        conclusion: str,
        weight_semantic: float = 0.7,
        weight_vocab: float = 0.3
    ) -> float:
        """
        Calculate logical coherence between premises and conclusion.

        Combines semantic similarity (embeddings) and vocabulary overlap (Jaccard).
        Higher values indicate better logical flow.

        Args:
            premises: List of premise statements
            conclusion: Conclusion statement
            weight_semantic: Weight for semantic similarity
            weight_vocab: Weight for vocabulary overlap

        Returns:
            Coherence score between 0 and 1
        """
        if not premises:
            return 0.0

        # Combine premises into single text
        combined_premises = " ".join(premises)

        # Semantic similarity
        premise_emb = self._get_embedding(combined_premises)
        conclusion_emb = self._get_embedding(conclusion)

        if premise_emb is not None and conclusion_emb is not None:
            semantic_sim = self._cosine_similarity(premise_emb, conclusion_emb)
        else:
            semantic_sim = 0.5  # Default when embeddings unavailable

        # Vocabulary overlap (Jaccard)
        premise_tokens = self._tokenize(combined_premises)
        conclusion_tokens = self._tokenize(conclusion)
        vocab_overlap = self._jaccard_coefficient(premise_tokens, conclusion_tokens)

        # Combined score
        return weight_semantic * semantic_sim + weight_vocab * vocab_overlap

    def calc_expansion_degree(self, premises: List[str], conclusion: str) -> float:
        """
        Calculate information expansion degree.

        Measures how much new information is introduced in the conclusion
        that wasn't present in the premises. Lower values indicate more
        conservative conclusions.

        Args:
            premises: List of premise statements
            conclusion: Conclusion statement

        Returns:
            Expansion degree between 0 and 1 (0 = no expansion, 1 = all new)
        """
        if not premises:
            return 1.0  # Full expansion if no premises

        # Combine premises
        combined_premises = " ".join(premises)

        # Tokenize
        premise_tokens = self._tokenize(combined_premises)
        conclusion_tokens = self._tokenize(conclusion)

        if not conclusion_tokens:
            return 0.0

        # Calculate proportion of conclusion tokens not in premises
        new_tokens = conclusion_tokens - premise_tokens
        expansion = len(new_tokens) / len(conclusion_tokens)

        return expansion

    def detect_contradiction(
        self,
        facts: List[str],
        conclusion: str,
        use_patterns: bool = True,
        use_numerical: bool = True
    ) -> Tuple[float, List[str]]:
        """
        Detect contradictions between facts and conclusion.

        Uses pattern matching for negation detection and numerical comparison
        for value conflicts.

        Args:
            facts: List of factual statements
            conclusion: Conclusion to check
            use_patterns: Enable pattern-based detection
            use_numerical: Enable numerical comparison

        Returns:
            Tuple of (contradiction_score, list of detected contradictions)
        """
        contradictions = []
        contradiction_score = 0.0

        # Pattern-based negation detection
        # Each tuple: (negation_pattern, keyword_to_look_for_in_facts)
        negation_keywords = [
            # English patterns: (keyword in negated form, keyword in positive form)
            (r'\bnot\b', None),  # Generic negation - will extract the word after "not"
            (r'\bno\b', None),
            (r'\bnever\b', r'\balways\b'),
            (r'\bimpossible\b', r'\bpossible\b'),
            (r'\bincorrect\b', r'\bcorrect\b'),
            (r'\bfalse\b', r'\btrue\b'),
            (r'\bdecrease\b', r'\bincrease\b'),
            (r'\breduce\b', r'\bincrease\b'),
            (r'\bunavailable\b', r'\bavailable\b'),
            # Japanese patterns
            (r'ではない', r'である'),
            (r'しない', r'する'),
            (r'ない', r'ある'),
            (r'減少', r'増加'),
            (r'低下', r'上昇'),
            (r'不可能', r'可能'),
        ]

        if use_patterns:
            conclusion_lower = conclusion.lower()
            for neg_pattern, pos_pattern in negation_keywords:
                if re.search(neg_pattern, conclusion_lower, re.IGNORECASE):
                    # Check if any fact contains the positive version
                    for fact in facts:
                        fact_lower = fact.lower()
                        if pos_pattern:
                            if re.search(pos_pattern, fact_lower, re.IGNORECASE):
                                contradictions.append(
                                    f"Potential negation conflict: '{fact[:50]}...' vs conclusion"
                                )
                                contradiction_score += 0.3
                                break
                        else:
                            # For generic negations like "not", check word overlap
                            fact_words = set(re.findall(r'\w+', fact_lower))
                            conclusion_words = set(re.findall(r'\w+', conclusion_lower))
                            common = fact_words & conclusion_words
                            if len(common) > 2:  # Significant overlap with negation
                                contradictions.append(
                                    f"Potential negation conflict: '{fact[:50]}...' vs conclusion"
                                )
                                contradiction_score += 0.2
                                break

        # Numerical contradiction detection
        if use_numerical:
            # Extract numbers from facts and conclusion
            fact_numbers = {}
            for fact in facts:
                numbers = re.findall(r'(\w+)[:\s]+([0-9]+(?:\.[0-9]+)?)\s*(%|円|ドル|人|件|個)?', fact)
                for match in numbers:
                    key = match[0].lower()
                    value = float(match[1])
                    unit = match[2] if len(match) > 2 else ''
                    fact_numbers[key] = (value, unit)

            conclusion_numbers = re.findall(
                r'(\w+)[:\s]+([0-9]+(?:\.[0-9]+)?)\s*(%|円|ドル|人|件|個)?',
                conclusion
            )

            for match in conclusion_numbers:
                key = match[0].lower()
                value = float(match[1])

                if key in fact_numbers:
                    fact_value, _ = fact_numbers[key]
                    # Check for significant discrepancy (>50% difference)
                    if fact_value > 0:
                        diff_ratio = abs(value - fact_value) / fact_value
                        if diff_ratio > 0.5:
                            contradictions.append(
                                f"Numerical discrepancy for '{key}': fact={fact_value}, conclusion={value}"
                            )
                            contradiction_score += min(diff_ratio * 0.5, 0.5)

        # Normalize score to 0-1
        contradiction_score = min(contradiction_score, 1.0)

        return contradiction_score, contradictions

    def calc_deviation_score(
        self,
        initial_facts: List[str],
        conclusion: str,
        intermediate_conclusions: List[str] = None,
        weights: Tuple[float, float, float] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive deviation score.

        Combines semantic deviation, logical gap, and contradiction score
        with configurable weights.

        Args:
            initial_facts: Original extracted facts
            conclusion: Final conclusion
            intermediate_conclusions: Intermediate conclusions in reasoning chain
            weights: Tuple of (semantic_weight, logical_weight, contradiction_weight)

        Returns:
            Dictionary with individual scores and weighted total
        """
        weights = weights or self.config.deviation_weights
        alpha, beta, gamma = weights

        # Semantic deviation (1 - similarity)
        combined_facts = " ".join(initial_facts)
        facts_emb = self._get_embedding(combined_facts)
        conclusion_emb = self._get_embedding(conclusion)

        if facts_emb is not None and conclusion_emb is not None:
            semantic_sim = self._cosine_similarity(facts_emb, conclusion_emb)
            semantic_deviation = 1.0 - semantic_sim
        else:
            # Fallback
            fact_tokens = self._tokenize(combined_facts)
            conclusion_tokens = self._tokenize(conclusion)
            jaccard = self._jaccard_coefficient(fact_tokens, conclusion_tokens)
            semantic_deviation = 1.0 - jaccard

        # Logical gap (using intermediate conclusions if available)
        if intermediate_conclusions:
            # Check chain continuity
            all_steps = initial_facts + intermediate_conclusions + [conclusion]
            gaps = []
            for i in range(len(all_steps) - 1):
                coherence = self.calc_logical_coherence([all_steps[i]], all_steps[i + 1])
                gaps.append(1.0 - coherence)
            logical_gap = sum(gaps) / len(gaps) if gaps else 0.0
        else:
            # Direct logical coherence
            coherence = self.calc_logical_coherence(initial_facts, conclusion)
            logical_gap = 1.0 - coherence

        # Contradiction detection
        contradiction_score, _ = self.detect_contradiction(initial_facts, conclusion)

        # Weighted total
        total_deviation = (
            alpha * semantic_deviation +
            beta * logical_gap +
            gamma * contradiction_score
        )

        return {
            "semantic_deviation": semantic_deviation,
            "logical_gap": logical_gap,
            "contradiction_score": contradiction_score,
            "total_deviation": total_deviation,
            "weights_used": {"alpha": alpha, "beta": beta, "gamma": gamma}
        }

    def calc_confidence_score(
        self,
        source_fidelity: float,
        logical_coherence: float,
        expansion_degree: float,
        deviation_score: float,
        expansion_tolerance: float = None
    ) -> float:
        """
        Calculate overall confidence score using sigmoid normalization.

        Combines multiple metrics into a single confidence score between 0 and 1.

        Args:
            source_fidelity: Fidelity to source (0-1)
            logical_coherence: Logical coherence (0-1)
            expansion_degree: Information expansion (0-1)
            deviation_score: Total deviation score (0-1)
            expansion_tolerance: Acceptable expansion level

        Returns:
            Confidence score between 0 and 1
        """
        expansion_tolerance = expansion_tolerance or self.config.expansion_tolerance

        # Calculate expansion penalty
        expansion_penalty = max(0, expansion_degree - expansion_tolerance)

        # Raw confidence based on positive and negative factors
        # Positive: fidelity, coherence
        # Negative: deviation, excess expansion
        raw_score = (
            0.35 * source_fidelity +
            0.35 * logical_coherence +
            0.15 * (1.0 - deviation_score) +
            0.15 * (1.0 - expansion_penalty)
        )

        # Sigmoid normalization for smoother boundaries
        # sigmoid(x) = 1 / (1 + e^(-k*(x-0.5)))
        # k controls steepness
        k = 10  # Steepness factor
        sigmoid_score = 1.0 / (1.0 + math.exp(-k * (raw_score - 0.5)))

        return sigmoid_score

    def evaluate_reasoning_step(
        self,
        premises: List[str],
        conclusion: str,
        source_text: str = None
    ) -> Dict[str, float]:
        """
        Evaluate a single reasoning step comprehensively.

        Args:
            premises: Premise statements
            conclusion: Conclusion from this step
            source_text: Original source text (if available)

        Returns:
            Dictionary with all metric scores
        """
        # Source fidelity (if source available)
        if source_text:
            fidelity = self.calc_source_fidelity(" ".join(premises), source_text)
        else:
            fidelity = 0.8  # Default when source not available

        # Logical coherence
        coherence = self.calc_logical_coherence(premises, conclusion)

        # Expansion degree
        expansion = self.calc_expansion_degree(premises, conclusion)

        # Deviation
        deviation_result = self.calc_deviation_score(premises, conclusion)

        # Confidence
        confidence = self.calc_confidence_score(
            source_fidelity=fidelity,
            logical_coherence=coherence,
            expansion_degree=expansion,
            deviation_score=deviation_result["total_deviation"]
        )

        return {
            "source_fidelity": fidelity,
            "logical_coherence": coherence,
            "expansion_degree": expansion,
            "semantic_deviation": deviation_result["semantic_deviation"],
            "logical_gap": deviation_result["logical_gap"],
            "contradiction_score": deviation_result["contradiction_score"],
            "total_deviation": deviation_result["total_deviation"],
            "confidence_score": confidence
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._embedding_cache.clear()
