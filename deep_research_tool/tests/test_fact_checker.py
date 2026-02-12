"""
Tests for the LLM-based Fact Checker module.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

# Import modules to test
from deep_research_tool.fact_checker.sentence_splitter import (
    SentenceSplitter,
    Sentence,
    SplitMethod,
)
from deep_research_tool.fact_checker.claim_extractor import (
    ClaimExtractor,
    Claim,
    ClaimType,
    ClaimStrength,
)
from deep_research_tool.fact_checker.web_crawler import (
    RecursiveWebCrawler,
    CrawlResult,
    AdFilter,
    SearchEngine,
    SearchResultItem,
)
from deep_research_tool.fact_checker.fact_verifier import (
    FactVerifier,
    VerificationLabel,
    EvidenceStrength,
    SentenceVerificationResult,
    FactCheckReport,
    EvidenceMatch,
    CorrectionSuggestion,
)
from deep_research_tool.fact_checker.fact_checker import (
    FactChecker,
    FactCheckerConfig,
    create_fact_checker,
)


class TestSentenceSplitter:
    """Tests for SentenceSplitter."""

    def test_split_japanese_text(self):
        """Test splitting Japanese text."""
        splitter = SentenceSplitter(method=SplitMethod.RULE_BASED)

        text = "これは最初の文です。これは二番目の文です。これは三番目の文です。"
        sentences = splitter.split(text)

        assert len(sentences) == 3
        assert sentences[0].text == "これは最初の文です。"
        assert sentences[1].text == "これは二番目の文です。"
        assert sentences[2].text == "これは三番目の文です。"

    def test_split_english_text(self):
        """Test splitting English text."""
        splitter = SentenceSplitter(method=SplitMethod.RULE_BASED)

        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        sentences = splitter.split(text)

        assert len(sentences) == 3
        assert "first" in sentences[0].text
        assert "second" in sentences[1].text
        assert "third" in sentences[2].text

    def test_split_mixed_language(self):
        """Test splitting mixed Japanese and English text."""
        splitter = SentenceSplitter(method=SplitMethod.RULE_BASED)

        text = "日本語の文です。This is English. また日本語です。"
        sentences = splitter.split(text)

        assert len(sentences) == 3

    def test_handle_abbreviations(self):
        """Test that abbreviations don't cause incorrect splits."""
        splitter = SentenceSplitter(method=SplitMethod.RULE_BASED)

        text = "Dr. Smith is a professor at MIT. He studies AI."
        sentences = splitter.split(text)

        # Should be 2 sentences, not split at "Dr."
        assert len(sentences) == 2

    def test_extract_reference_urls(self):
        """Test URL extraction from sentences."""
        splitter = SentenceSplitter(method=SplitMethod.RULE_BASED)

        text = "詳細はhttps://example.com/page123を参照してください。"
        sentences = splitter.split(text)

        assert len(sentences) == 1
        assert sentences[0].has_reference
        assert "https://example.com/page123" in sentences[0].reference_urls

    def test_empty_text(self):
        """Test handling empty text."""
        splitter = SentenceSplitter(method=SplitMethod.RULE_BASED)

        sentences = splitter.split("")
        assert len(sentences) == 0

        sentences = splitter.split("   ")
        assert len(sentences) == 0

    def test_min_sentence_length(self):
        """Test minimum sentence length filtering."""
        splitter = SentenceSplitter(method=SplitMethod.RULE_BASED, min_sentence_length=10)

        text = "短い。これは十分に長い文章です。"
        sentences = splitter.split(text)

        # "短い。" (3 chars) should be filtered out, "これは十分に長い文章です。" (13 chars) passes
        assert len(sentences) == 1
        assert "十分に長い" in sentences[0].text


class TestClaimExtractor:
    """Tests for ClaimExtractor."""

    def test_detect_statistical_claim(self):
        """Test detection of statistical claims."""
        extractor = ClaimExtractor(llm_client=None)

        # Create a mock sentence
        sentence = Sentence(
            text="日本の人口は約1億2000万人です。",
            index=0,
            start_pos=0,
            end_pos=100,
            language="ja",
        )

        claims = extractor.extract_claims([sentence])

        assert len(claims) == 1
        assert claims[0].claim_type == ClaimType.STATISTICAL

    def test_detect_temporal_claim(self):
        """Test detection of temporal claims."""
        extractor = ClaimExtractor(llm_client=None)

        sentence = Sentence(
            text="東京オリンピックは2021年に開催されました。",
            index=0,
            start_pos=0,
            end_pos=100,
            language="ja",
        )

        claims = extractor.extract_claims([sentence])

        assert len(claims) == 1
        assert ClaimType.TEMPORAL in [claims[0].claim_type] or claims[0].claim_type == ClaimType.FACTUAL

    def test_detect_causal_claim(self):
        """Test detection of causal claims."""
        extractor = ClaimExtractor(llm_client=None)

        sentence = Sentence(
            text="The economic crisis was caused by poor policy decisions.",
            index=0,
            start_pos=0,
            end_pos=100,
            language="en",
        )

        claims = extractor.extract_claims([sentence])

        assert len(claims) == 1
        # Should detect causal pattern
        detected_types = extractor._detect_claim_types(sentence.text)
        assert ClaimType.CAUSAL in detected_types

    def test_detect_claim_strength(self):
        """Test detection of claim strength."""
        extractor = ClaimExtractor(llm_client=None)

        # Strong claim
        strong_text = "This is definitely true."
        strength = extractor._detect_claim_strength(strong_text, "en")
        assert strength == ClaimStrength.STRONG

        # Weak claim
        weak_text = "This might possibly be true."
        strength = extractor._detect_claim_strength(weak_text, "en")
        assert strength in [ClaimStrength.WEAK, ClaimStrength.MODERATE]

    def test_extract_keywords(self):
        """Test keyword extraction."""
        extractor = ClaimExtractor(llm_client=None)

        text = "Artificial Intelligence is transforming healthcare industry."
        keywords = extractor._extract_keywords(text)

        assert len(keywords) > 0
        assert "Artificial" in keywords or "Intelligence" in keywords


class TestAdFilter:
    """Tests for AdFilter."""

    def test_ad_url_detection(self):
        """Test ad URL detection."""
        filter = AdFilter()

        assert filter.is_ad_url("https://doubleclick.net/ad123")
        assert filter.is_ad_url("https://googlesyndication.com/pagead")
        assert not filter.is_ad_url("https://wikipedia.org/wiki/Test")
        assert not filter.is_ad_url("https://example.com/article")

    def test_clean_text(self):
        """Test text cleaning."""
        filter = AdFilter()

        text = """This is real content.
        Click here to buy now!
        More real content here.
        Limited offer - act fast!
        Final paragraph of content."""

        cleaned = filter.clean_text(text)

        assert "real content" in cleaned
        # Ad-like lines should be removed in strict mode
        # In normal mode, they may remain


class TestCrawlResult:
    """Tests for CrawlResult."""

    def test_content_hash(self):
        """Test content hash generation."""
        result = CrawlResult(
            url="https://example.com",
            title="Test",
            text_content="Full text",
            clean_text="Clean text content",
        )

        assert result.content_hash != ""
        assert len(result.content_hash) == 32  # MD5 hash length

    def test_word_count(self):
        """Test word count calculation."""
        result = CrawlResult(
            url="https://example.com",
            title="Test",
            text_content="Full text",
            clean_text="This is a test with seven words.",
        )

        assert result.word_count == 7


class TestVerificationLabel:
    """Tests for VerificationLabel enum."""

    def test_label_values(self):
        """Test that all expected labels exist."""
        assert VerificationLabel.SUPPORTED.value == "supported"
        assert VerificationLabel.NOT_TRUE.value == "not_true"
        assert VerificationLabel.NOT_VERIFIED.value == "not_verified"
        assert VerificationLabel.SUSPICIOUS.value == "suspicious"
        assert VerificationLabel.PARTIALLY_TRUE.value == "partially_true"
        assert VerificationLabel.OPINION.value == "opinion"


class TestFactCheckReport:
    """Tests for FactCheckReport."""

    def test_calculate_statistics(self):
        """Test statistics calculation."""
        # Create mock results
        claim = Claim(
            text="Test claim",
            claim_type=ClaimType.FACTUAL,
            strength=ClaimStrength.STRONG,
            source_sentence=Sentence(
                text="Test sentence",
                index=0,
                start_pos=0,
                end_pos=50,
            ),
        )

        results = [
            SentenceVerificationResult(
                claim=claim,
                label=VerificationLabel.SUPPORTED,
                confidence=0.9,
                evidence_strength=EvidenceStrength.STRONG,
            ),
            SentenceVerificationResult(
                claim=claim,
                label=VerificationLabel.SUPPORTED,
                confidence=0.8,
                evidence_strength=EvidenceStrength.MODERATE,
            ),
            SentenceVerificationResult(
                claim=claim,
                label=VerificationLabel.NOT_TRUE,
                confidence=0.9,
                evidence_strength=EvidenceStrength.STRONG,
            ),
            SentenceVerificationResult(
                claim=claim,
                label=VerificationLabel.NOT_VERIFIED,
                confidence=0.5,
                evidence_strength=EvidenceStrength.NONE,
            ),
            SentenceVerificationResult(
                claim=claim,
                label=VerificationLabel.OPINION,
                confidence=0.9,
                evidence_strength=EvidenceStrength.NONE,
            ),
        ]

        report = FactCheckReport(
            document_title="Test Report",
            total_sentences=5,
            total_claims=5,
            results=results,
        )

        report.calculate_statistics()

        assert report.supported_count == 2
        assert report.not_true_count == 1
        assert report.not_verified_count == 1
        assert report.opinion_count == 1
        # Accuracy: 2 supported out of 4 verifiable = 0.5
        assert report.overall_accuracy_score == 0.5

    def test_to_dict(self):
        """Test dictionary conversion."""
        report = FactCheckReport(
            document_title="Test",
            total_sentences=10,
            total_claims=5,
        )

        data = report.to_dict()

        assert data["document_title"] == "Test"
        assert data["total_sentences"] == 10
        assert data["total_claims"] == 5
        assert "summary" in data
        assert "created_at" in data


class TestFactCheckerConfig:
    """Tests for FactCheckerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = FactCheckerConfig()

        assert config.extraction_llm_provider == "openai"
        assert config.verification_llm_provider == "openai"
        assert config.search_engine == SearchEngine.GOOGLE
        assert config.output_language == "ja"
        assert config.headless is True

    @patch.dict('os.environ', {
        'OPENAI_API_KEY': 'test-key-123',
        'FACT_CHECKER_SEARCH_ENGINE': 'duckduckgo',
    })
    def test_env_var_loading(self):
        """Test loading from environment variables."""
        config = FactCheckerConfig()

        assert config.extraction_llm_api_key == 'test-key-123'
        assert config.search_engine == 'duckduckgo'

    def test_direct_api_key(self):
        """Test direct API key specification."""
        config = FactCheckerConfig(
            extraction_llm_api_key="direct-key-456",
            verification_llm_api_key="verify-key-789",
        )

        assert config.extraction_llm_api_key == "direct-key-456"
        assert config.verification_llm_api_key == "verify-key-789"

    def test_validation_missing_keys(self):
        """Test validation with missing API keys."""
        config = FactCheckerConfig(
            extraction_llm_api_key=None,
            verification_llm_api_key=None,
        )
        # Clear any env vars that might be set
        config.extraction_llm_api_key = None
        config.verification_llm_api_key = None

        errors = config.validate()

        assert len(errors) >= 2  # Should have errors for both keys

    def test_from_env(self):
        """Test creating config from environment."""
        config = FactCheckerConfig.from_env()

        assert config is not None
        assert isinstance(config, FactCheckerConfig)


class TestFactCheckerIntegration:
    """Integration tests for FactChecker (mocked)."""

    def test_create_fact_checker(self):
        """Test factory function."""
        # This will create a FactChecker but may warn about missing API keys
        checker = create_fact_checker(
            extraction_provider="openai",
            extraction_api_key="test-key",
            verification_provider="openai",
            verification_api_key="test-key",
            search_engine=SearchEngine.DUCKDUCKGO,
            language="en",
        )

        assert checker is not None
        assert checker.config.search_engine == SearchEngine.DUCKDUCKGO
        assert checker.config.output_language == "en"

    def test_sentence_verification_result_to_dict(self):
        """Test SentenceVerificationResult serialization."""
        claim = Claim(
            text="Test claim text",
            claim_type=ClaimType.STATISTICAL,
            strength=ClaimStrength.STRONG,
            source_sentence=Sentence(
                text="Original sentence",
                index=0,
                start_pos=0,
                end_pos=50,
            ),
            search_queries=["test query 1"],
        )

        result = SentenceVerificationResult(
            claim=claim,
            label=VerificationLabel.NOT_TRUE,
            confidence=0.85,
            evidence_strength=EvidenceStrength.STRONG,
            evidence_matches=[
                EvidenceMatch(
                    source_url="https://example.com",
                    source_title="Example Source",
                    relevant_text="Relevant quote",
                    support_type="contradicts",
                    relevance_score=0.9,
                )
            ],
            correction=CorrectionSuggestion(
                original_claim="Test claim text",
                corrected_claim="Corrected claim text",
                correction_type="factual",
                explanation="The original was incorrect",
                confidence=0.8,
                sources=["https://example.com"],
            ),
            reasoning="The evidence contradicts this claim.",
        )

        data = result.to_dict()

        assert data["claim_text"] == "Test claim text"
        assert data["label"] == "not_true"
        assert data["confidence"] == 0.85
        assert len(data["evidence_matches"]) == 1
        assert data["correction"]["corrected"] == "Corrected claim text"


class TestSearchEngine:
    """Tests for SearchEngine constants."""

    def test_engine_values(self):
        """Test search engine values."""
        assert SearchEngine.GOOGLE == "google"
        assert SearchEngine.DUCKDUCKGO == "duckduckgo"
        assert SearchEngine.BING == "bing"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
