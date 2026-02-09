#!/usr/bin/env python3
"""
Basic usage example for Deep Research Tool.

This script demonstrates how to use the Deep Research Tool
for automated research tasks.

Usage:
    python basic_usage.py

Requirements:
    - Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
    - Install dependencies: pip install -r requirements.txt
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deep_research_tool2 import DeepResearchTool, Config
from deep_research_tool2.config import create_config


def example_basic_research():
    """Basic research example with default settings."""
    print("=" * 60)
    print("Example 1: Basic Research")
    print("=" * 60)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return

    # Create configuration
    config = create_config(
        provider="openai",  # or "anthropic"
        research_iterations=3,
        output_format="markdown",
        output_dir="./output",
    )

    # Create tool instance
    tool = DeepResearchTool(config)

    # Run research
    result = tool.run(
        query="AI trends in healthcare 2024",
        requirements="Focus on recent developments and practical applications",
    )

    print(f"\nResearch completed!")
    print(f"Session ID: {result['session_id']}")
    print(f"Report: {result['report_path']}")
    print(f"Evidence: {result['evidence_json']}")


def example_with_documents():
    """Research example with additional documents."""
    print("=" * 60)
    print("Example 2: Research with Additional Documents")
    print("=" * 60)

    config = create_config(
        provider="openai",
        research_iterations=3,
        output_format="docx",
        additional_documents=["reference.pdf"],  # Add your documents here
    )

    tool = DeepResearchTool(config)

    result = tool.run(
        query="Market analysis of renewable energy",
        requirements="Include comparison with previous reports",
    )

    print(f"\nReport generated: {result['report_path']}")


def example_quick_research():
    """Quick research without full report generation."""
    print("=" * 60)
    print("Example 3: Quick Research")
    print("=" * 60)

    config = Config()
    tool = DeepResearchTool(config)

    # Quick research for fact-checking
    result = tool.quick_research(
        query="Latest developments in quantum computing",
        max_results=5,
    )

    print(f"\nQuery: {result['query']}")
    print(f"Results found: {result['results_count']}")
    print(f"\nSummary:\n{result['summary']}")


def example_with_verification():
    """Research with hallucination verification."""
    print("=" * 60)
    print("Example 4: Research with Verification")
    print("=" * 60)

    config = create_config(
        provider="openai",
        research_iterations=3,
        enable_verification=True,
        output_format="html",
    )

    tool = DeepResearchTool(config)

    def progress_callback(message: str, percentage: float):
        print(f"[{percentage:5.1f}%] {message}")

    result = tool.run(
        query="Climate change impact on agriculture",
        requirements="Focus on scientific evidence and data",
        progress_callback=progress_callback,
    )

    print(f"\nVerification report: {result.get('verification_html', 'N/A')}")


if __name__ == "__main__":
    print("\nDeep Research Tool - Usage Examples")
    print("=" * 60)

    # Run basic example
    example_basic_research()

    # Uncomment to run other examples:
    # example_with_documents()
    # example_quick_research()
    # example_with_verification()
