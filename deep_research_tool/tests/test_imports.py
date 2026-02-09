"""
Import tests for Deep Research Tool.
Verify that all modules can be imported correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_config_imports():
    """Test config module imports."""
    from deep_research_tool.config import (
        Config,
        APIConfig,
        SearchConfig,
        ResearchConfig,
        ReportConfig,
        LLMProvider,
        SearchMethod,
        ReportFormat,
        create_config,
    )
    assert Config is not None
    assert create_config is not None
    print("config imports: OK")


def test_api_imports():
    """Test API module imports."""
    from deep_research_tool.api import (
        BaseLLMClient,
        LLMResponse,
        get_client,
    )
    from deep_research_tool.api.base import Message, MessageRole
    assert BaseLLMClient is not None
    assert LLMResponse is not None
    print("api imports: OK")


def test_search_imports():
    """Test search module imports."""
    from deep_research_tool.search import (
        BaseSearchClient,
        SearchResult,
        DuckDuckGoSearch,
        SeleniumBrowser,
        get_search_client,
    )
    assert BaseSearchClient is not None
    assert SearchResult is not None
    print("search imports: OK")


def test_evidence_imports():
    """Test evidence module imports."""
    from deep_research_tool.evidence import (
        EvidenceLocker,
        Evidence,
        EvidenceType,
    )
    assert EvidenceLocker is not None
    assert Evidence is not None
    print("evidence imports: OK")


def test_research_imports():
    """Test research module imports."""
    from deep_research_tool.research import (
        QueryGenerator,
        ResearchPlan,
        TableOfContents,
        ContentExtractor,
        ExtractedContent,
        Researcher,
        ResearchSession,
        ResearchState,
    )
    assert QueryGenerator is not None
    assert Researcher is not None
    print("research imports: OK")


def test_verification_imports():
    """Test verification module imports."""
    from deep_research_tool.verification import (
        Verifier,
        VerificationResult,
        ClaimVerification,
    )
    assert Verifier is not None
    assert VerificationResult is not None
    print("verification imports: OK")


def test_report_imports():
    """Test report module imports."""
    from deep_research_tool.report import (
        ReportGenerator,
        ReportFormat,
    )
    assert ReportGenerator is not None
    print("report imports: OK")


def test_utils_imports():
    """Test utils module imports."""
    from deep_research_tool.utils import (
        DocumentReader,
        DocumentContent,
        setup_logging,
        format_timestamp,
        truncate_text,
    )
    assert DocumentReader is not None
    assert setup_logging is not None
    print("utils imports: OK")


def test_main_imports():
    """Test main module imports."""
    from deep_research_tool.main import (
        DeepResearchTool,
        run_research,
    )
    assert DeepResearchTool is not None
    assert run_research is not None
    print("main imports: OK")


def test_thinking_imports():
    """Test thinking module imports."""
    from deep_research_tool.thinking import (
        DeepThinkConfig,
        DeepThinkProcessor,
        DeepThinkMetrics,
        MetricsConfig,
        LogicValidator,
        ValidationConfig,
        ValidationLevel,
        ReasoningStep,
        ReasoningChain,
        ReasoningType,
        ConsistencyMode,
        ConsistencyCheckResult,
        DeepThinkResult,
        ExtensionConfig,
        DeepThinkExtension,
        SectionExtension,
        ReportExtension,
        DeepThinkMixin,
        create_extension_for_module,
    )
    assert DeepThinkConfig is not None
    assert DeepThinkProcessor is not None
    assert DeepThinkMetrics is not None
    print("thinking imports: OK")


def test_package_imports():
    """Test package-level imports."""
    from deep_research_tool import (
        Config,
        ResearchConfig,
        DeepResearchTool,
    )
    assert Config is not None
    assert DeepResearchTool is not None
    print("package imports: OK")


if __name__ == "__main__":
    test_config_imports()
    test_api_imports()
    test_search_imports()
    test_evidence_imports()
    test_research_imports()
    test_verification_imports()
    test_report_imports()
    test_utils_imports()
    test_main_imports()
    test_thinking_imports()
    test_package_imports()
    print("\nAll import tests passed!")
