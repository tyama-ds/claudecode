#!/usr/bin/env python3
"""
Example: Research-backed multi-agent discussion.

This example demonstrates how to set up a discussion where each participant
searches the web for information related to their specialty before responding.

Usage:
    export OPENAI_API_KEY="your-api-key"
    python research_discussion.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from multi_agent_discussion import (
    Config,
    LLMConfig,
    AgentConfig,
    DiscussionConfig,
    AgentRole,
    LLMProvider,
    MultiAgentDiscussion,
    run_discussion,
)


def main():
    """Run a research-backed multi-agent discussion."""

    topic = "量子コンピューティング分野の日本における未来は？"

    print(f"議論トピック: {topic}")
    print("=" * 60)
    print("\n各エージェントが情報収集しながら議論を行います...\n")

    # ===== 方法1: 詳細な設定 =====
    config = Config(
        llm=LLMConfig(provider=LLMProvider.OPENAI),
        discussion=DiscussionConfig(
            topic=topic,
            max_rounds=3,
            enable_evaluation=True,
        ),
        agents=[
            # コーディネーター（モデレーター）
            AgentConfig(
                name="コーディネーター",
                role=AgentRole.MODERATOR,
                persona="各専門家の意見を統合し、議論を生産的な方向に導くコーディネーター",
            ),

            # 技術担当（情報収集付き）
            AgentConfig(
                name="技術担当",
                role=AgentRole.RESEARCH_PARTICIPANT,
                persona="量子コンピューティングの技術動向に詳しい技術専門家。量子ビット、量子ゲート、量子アルゴリズムなどの技術的側面を調査・分析する。",
                search_config={
                    "region": "jp-jp",
                    "max_queries_per_turn": 2,
                    "max_results_per_query": 5,
                },
            ),

            # 特許担当（情報収集付き）
            AgentConfig(
                name="特許担当",
                role=AgentRole.RESEARCH_PARTICIPANT,
                persona="量子コンピューティング関連の特許・知的財産に詳しい専門家。特許出願動向、主要プレイヤーの知財戦略を調査・分析する。",
                search_config={
                    "region": "jp-jp",
                    "max_queries_per_turn": 2,
                },
            ),

            # 市場担当（情報収集付き）
            AgentConfig(
                name="市場担当",
                role=AgentRole.RESEARCH_PARTICIPANT,
                persona="量子コンピューティングの市場動向・ビジネス展開に詳しい市場アナリスト。市場規模予測、主要企業の動向、投資トレンドを調査・分析する。",
                search_config={
                    "region": "jp-jp",
                    "max_queries_per_turn": 2,
                },
            ),

            # 法規制担当（情報収集付き）
            AgentConfig(
                name="法規制担当",
                role=AgentRole.RESEARCH_PARTICIPANT,
                persona="量子コンピューティングに関する法規制・政策に詳しい法律専門家。各国の規制動向、セキュリティ要件、政府の支援策を調査・分析する。",
                search_config={
                    "region": "jp-jp",
                    "max_queries_per_turn": 2,
                },
            ),

            # 評価者
            AgentConfig(
                name="評価者",
                role=AgentRole.EVALUATOR,
            ),
        ],
    )

    # 議論を実行
    # discussion = MultiAgentDiscussion(config)
    # result = discussion.run(
    #     message_callback=lambda msg: print(f"\n[{msg.agent_name}]\n{msg.content}\n{'='*40}")
    # )

    print("設定例を表示しました。")
    print("実際に実行するには、APIキーを設定してコメントを外してください。")


def simple_research_discussion():
    """
    シンプルな情報収集付き議論の例。
    run_discussion() の enable_search=True で簡単に有効化できます。
    """
    print("\n" + "=" * 60)
    print("シンプルな情報収集付き議論の例")
    print("=" * 60 + "\n")

    topic = "再生可能エネルギーの日本における将来性"

    # カスタムペルソナでエージェントを指定
    participant_personas = [
        {
            "name": "技術アナリスト",
            "persona": "再生可能エネルギー技術の専門家。太陽光、風力、水素などの技術動向を調査する。",
        },
        {
            "name": "政策アナリスト",
            "persona": "エネルギー政策の専門家。政府の施策、規制、補助金制度を調査する。",
        },
        {
            "name": "投資アナリスト",
            "persona": "クリーンテック投資の専門家。市場動向、投資トレンド、企業動向を調査する。",
        },
    ]

    # 実行（enable_search=True で情報収集を有効化）
    # result = run_discussion(
    #     topic=topic,
    #     provider="openai",
    #     participant_personas=participant_personas,
    #     max_rounds=3,
    #     enable_search=True,
    #     search_config={"region": "jp-jp", "max_queries_per_turn": 2},
    #     message_callback=lambda msg: print(f"\n[{msg.agent_name}]\n{msg.content}\n{'-'*40}"),
    # )

    print(f"トピック: {topic}")
    print("\nエージェント構成:")
    for p in participant_personas:
        print(f"  - {p['name']}: {p['persona'][:50]}...")

    print("\n実行するには:")
    print("  1. OPENAI_API_KEY を設定")
    print("  2. deep_research_tool をインストール")
    print("  3. コメントアウトされた run_discussion() を実行")


if __name__ == "__main__":
    main()
    simple_research_discussion()
