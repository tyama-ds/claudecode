"""
Example usage of the LLM-based Fact Checker.

This module demonstrates how to use the FactChecker for sentence-level
fact verification with various configuration options.
"""

import os
from pathlib import Path

# Set up API key (or use environment variable)
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"
# os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"

from deep_research_tool.fact_checker import (
    FactChecker,
    FactCheckerConfig,
    create_fact_checker,
    quick_fact_check,
    SplitMethod,
)
from deep_research_tool.fact_checker.web_crawler import SearchEngine


def example_1_basic_usage():
    """
    Example 1: Basic fact-checking with default settings.

    Uses OpenAI API and Google search by default.
    API keys are loaded from environment variables:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY
    """
    print("=" * 60)
    print("Example 1: Basic Fact-Checking")
    print("=" * 60)

    # Sample text to fact-check (Japanese)
    text = """
    東京は日本の首都であり、人口は約1400万人です。
    東京オリンピックは2020年に開催されました。
    富士山の高さは3776メートルで、日本で最も高い山です。
    日本の面積は約38万平方キロメートルです。
    """

    # Create fact checker with default settings
    config = FactCheckerConfig()
    checker = FactChecker(config=config)

    try:
        # Perform fact-check
        report = checker.check_text(text, document_title="日本に関する事実")

        # Print summary
        print(f"\n📊 検証結果サマリー:")
        print(f"  - 総文数: {report.total_sentences}")
        print(f"  - 総主張数: {report.total_claims}")
        print(f"  - 裏付けあり: {report.supported_count}")
        print(f"  - 事実と異なる: {report.not_true_count}")
        print(f"  - 未検証: {report.not_verified_count}")
        print(f"  - 正確性スコア: {report.overall_accuracy_score:.1%}")

        # Export report
        html_path = checker.export_report(report, format="html")
        print(f"\n📄 HTMLレポート: {html_path}")

    finally:
        checker.close()


def example_2_custom_configuration():
    """
    Example 2: Fact-checking with custom configuration.

    Demonstrates:
    - Using Anthropic API for verification
    - Using DuckDuckGo for search
    - Custom crawling settings
    """
    print("\n" + "=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)

    # Create custom configuration
    config = FactCheckerConfig(
        # Use Anthropic Claude for both extraction and verification
        extraction_llm_provider="anthropic",
        extraction_llm_model="claude-3-5-sonnet-20241022",
        verification_llm_provider="anthropic",
        verification_llm_model="claude-3-5-sonnet-20241022",

        # Use DuckDuckGo instead of Google
        search_engine=SearchEngine.DUCKDUCKGO,

        # Crawling settings
        max_search_results=3,
        max_crawl_depth=1,
        max_pages_per_depth=2,

        # Enable strict ad filtering
        strict_ad_filtering=True,

        # Output in English
        output_language="en",
    )

    # Sample text (English)
    text = """
    Python was created by Guido van Rossum and first released in 1991.
    Python 3.0 was released in December 2008.
    According to the TIOBE Index, Python has been the most popular programming language since 2021.
    """

    checker = FactChecker(config=config)

    try:
        report = checker.check_text(text, document_title="Python Facts")

        print(f"\n📊 Verification Summary:")
        print(f"  - Total sentences: {report.total_sentences}")
        print(f"  - Total claims: {report.total_claims}")
        print(f"  - Supported: {report.supported_count}")
        print(f"  - Not true: {report.not_true_count}")
        print(f"  - Not verified: {report.not_verified_count}")
        print(f"  - Accuracy score: {report.overall_accuracy_score:.1%}")

        # Print detailed results
        for result in report.results:
            print(f"\n  Claim: {result.claim.text[:60]}...")
            print(f"  Label: {result.label.value}")
            print(f"  Confidence: {result.confidence:.1%}")
            if result.correction:
                print(f"  Correction: {result.correction.corrected_claim}")

    finally:
        checker.close()


def example_3_factory_function():
    """
    Example 3: Using the factory function for quick setup.
    """
    print("\n" + "=" * 60)
    print("Example 3: Factory Function")
    print("=" * 60)

    # Create fact checker using factory function
    checker = create_fact_checker(
        extraction_provider="openai",
        verification_provider="openai",
        search_engine=SearchEngine.GOOGLE,
        language="ja",
        headless=True,
        max_search_results=5,
    )

    text = "地球から太陽までの距離は約1億5000万キロメートルです。光は太陽から地球まで約8分で届きます。"

    try:
        report = checker.check_text(text)

        print(f"\n📊 結果: {report.supported_count}件が検証されました")

    finally:
        checker.close()


def example_4_quick_fact_check():
    """
    Example 4: One-liner quick fact check.
    """
    print("\n" + "=" * 60)
    print("Example 4: Quick Fact Check")
    print("=" * 60)

    text = "アインシュタインは1905年に特殊相対性理論を発表しました。"

    # Quick fact check with minimal configuration
    report = quick_fact_check(
        text=text,
        provider="openai",
        search_engine=SearchEngine.DUCKDUCKGO,
        language="ja",
    )

    print(f"\n📊 Quick check result: {report.overall_accuracy_score:.1%} accuracy")


def example_5_file_checking():
    """
    Example 5: Fact-checking a file.
    """
    print("\n" + "=" * 60)
    print("Example 5: File Fact-Checking")
    print("=" * 60)

    # Create a sample file
    sample_dir = Path("./sample_documents")
    sample_dir.mkdir(exist_ok=True)

    sample_file = sample_dir / "sample_article.txt"
    sample_file.write_text("""
    気候変動に関する事実

    地球の平均気温は産業革命以降、約1.1度上昇しています。
    2023年は観測史上最も暑い年でした。
    北極の氷は過去40年間で約40%減少しました。
    パリ協定は2015年に採択され、気温上昇を1.5度以内に抑えることを目指しています。
    """, encoding="utf-8")

    print(f"Sample file created: {sample_file}")

    config = FactCheckerConfig(
        output_dir=sample_dir / "reports",
    )
    checker = FactChecker(config=config)

    try:
        report = checker.check_file(str(sample_file), document_title="気候変動ファクトチェック")

        print(f"\n📊 File check results:")
        print(f"  - Accuracy: {report.overall_accuracy_score:.1%}")

        # Export both HTML and JSON
        html_path = checker.export_report(report, format="html")
        json_path = checker.export_report(report, format="json")

        print(f"  - HTML report: {html_path}")
        print(f"  - JSON report: {json_path}")

    finally:
        checker.close()


def example_6_progress_callback():
    """
    Example 6: Using progress callback for real-time updates.
    """
    print("\n" + "=" * 60)
    print("Example 6: Progress Callback")
    print("=" * 60)

    def progress_handler(message: str, progress: float):
        """Custom progress handler."""
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r[{bar}] {progress:.0%} - {message}", end="", flush=True)
        if progress >= 1.0:
            print()  # New line at completion

    config = FactCheckerConfig(
        progress_callback=progress_handler,
        max_search_results=2,  # Fewer results for faster demo
    )

    text = "日本の国花は桜です。富士山は活火山です。"

    checker = FactChecker(config=config)

    try:
        report = checker.check_text(text)
        print(f"\n✅ Complete! Accuracy: {report.overall_accuracy_score:.1%}")
    finally:
        checker.close()


def example_7_mixed_provider():
    """
    Example 7: Using different providers for extraction and verification.

    This demonstrates the separate session architecture where:
    - Extraction uses OpenAI (faster, cheaper)
    - Verification uses Anthropic Claude (more thorough)
    """
    print("\n" + "=" * 60)
    print("Example 7: Mixed Provider Configuration")
    print("=" * 60)

    config = FactCheckerConfig(
        # Use OpenAI for faster claim extraction
        extraction_llm_provider="openai",
        extraction_llm_model="gpt-4o-mini",

        # Use Anthropic for more thorough verification
        verification_llm_provider="anthropic",
        verification_llm_model="claude-3-5-sonnet-20241022",

        # Lower temperature for more consistent verification
        llm_temperature=0.2,
    )

    text = """
    The Great Wall of China is visible from space with the naked eye.
    Napoleon Bonaparte was short for his time.
    We only use 10% of our brains.
    """

    checker = FactChecker(config=config)

    try:
        report = checker.check_text(text, document_title="Common Misconceptions Check")

        print(f"\n📊 Results for common misconceptions:")
        for result in report.results:
            emoji = "✓" if result.label.value == "supported" else "✗" if result.label.value == "not_true" else "?"
            print(f"  {emoji} {result.claim.text[:50]}... -> {result.label.value}")

    finally:
        checker.close()


def main():
    """Run all examples."""
    print("\n🔍 LLM-based Fact Checker - Usage Examples\n")

    # Note: These examples require API keys to be set
    # Uncomment and run individual examples as needed

    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key and not anthropic_key:
        print("⚠️  Warning: No API keys found in environment variables.")
        print("   Set OPENAI_API_KEY and/or ANTHROPIC_API_KEY to run examples.")
        print("\n   Example:")
        print('   export OPENAI_API_KEY="your-key-here"')
        print('   export ANTHROPIC_API_KEY="your-key-here"')
        return

    print(f"✓ OpenAI API key: {'Found' if openai_key else 'Not set'}")
    print(f"✓ Anthropic API key: {'Found' if anthropic_key else 'Not set'}")

    # Run examples (uncomment as needed)
    # example_1_basic_usage()
    # example_2_custom_configuration()
    # example_3_factory_function()
    # example_4_quick_fact_check()
    # example_5_file_checking()
    # example_6_progress_callback()
    # example_7_mixed_provider()

    print("\n✅ Examples ready to run. Uncomment desired examples in main().")


if __name__ == "__main__":
    main()
