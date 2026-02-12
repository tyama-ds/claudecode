"""
LLM-based Fact Checker for sentence-level fact verification.

This module provides comprehensive fact-checking capabilities including:
- Sentence-level text splitting
- Claim extraction from sentences
- Web-based evidence gathering (DuckDuckGo, Google via Selenium)
- LLM-based verification with separate API sessions
- Detailed labeling and correction suggestions
"""

from .sentence_splitter import SentenceSplitter, SplitMethod
from .claim_extractor import ClaimExtractor, Claim, ClaimType
from .web_crawler import RecursiveWebCrawler, CrawlResult, AdFilter
from .fact_verifier import (
    FactVerifier,
    VerificationLabel,
    SentenceVerificationResult,
    FactCheckReport,
)
from .fact_checker import FactChecker, FactCheckerConfig

__all__ = [
    # Sentence splitting
    "SentenceSplitter",
    "SplitMethod",
    # Claim extraction
    "ClaimExtractor",
    "Claim",
    "ClaimType",
    # Web crawling
    "RecursiveWebCrawler",
    "CrawlResult",
    "AdFilter",
    # Verification
    "FactVerifier",
    "VerificationLabel",
    "SentenceVerificationResult",
    "FactCheckReport",
    # Main interface
    "FactChecker",
    "FactCheckerConfig",
]
