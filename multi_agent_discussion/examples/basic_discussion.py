#!/usr/bin/env python3
"""
Basic example of running a multi-agent discussion.

This example demonstrates how to use the MultiAgentDiscussion tool
with default settings.

Usage:
    export OPENAI_API_KEY="your-api-key"
    python basic_discussion.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from multi_agent_discussion import run_discussion


def main():
    """Run a basic multi-agent discussion."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        print("Please set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    # Define the discussion topic
    topic = "人工知能は人間の仕事を奪うのか、それとも新しい仕事を創出するのか"

    print(f"議論トピック: {topic}")
    print("=" * 50)

    # Progress callback to show status
    def on_progress(status: str, progress: float):
        print(f"[{progress*100:.0f}%] {status}")

    # Message callback to show each message
    def on_message(message):
        print(f"\n[{message.agent_name}]")
        print(message.content)
        print("-" * 30)

    # Run the discussion
    result = run_discussion(
        topic=topic,
        provider="openai",
        max_rounds=3,
        progress_callback=on_progress,
        message_callback=on_message,
    )

    # Print results
    print("\n" + "=" * 50)
    print("議論完了!")
    print(f"セッションID: {result['session_id']}")
    print(f"ラウンド数: {result['rounds']}")
    print(f"メッセージ数: {result['message_count']}")

    if result["evaluation"]:
        print("\n評価結果:")
        print(f"  サマリー: {result['evaluation']['summary']}")
        print(f"  品質スコア: {result['evaluation']['quality_score']:.2f}")

    if result["session_path"]:
        print(f"\nセッションファイル: {result['session_path']}")


if __name__ == "__main__":
    main()
