#!/usr/bin/env python3
"""
Example of running a multi-agent discussion with custom personas.

This example shows how to define custom participant personas
for more specialized discussions.

Usage:
    export OPENAI_API_KEY="your-api-key"
    python custom_personas.py
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from multi_agent_discussion import (
    Config,
    LLMConfig,
    AgentConfig,
    DiscussionConfig,
    AgentRole,
    LLMProvider,
    MultiAgentDiscussion,
)


def main():
    """Run a discussion with custom expert personas."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set")
        sys.exit(1)

    # Define custom personas for a tech ethics discussion
    topic = "SNSのアルゴリズムによる情報フィルタリングは民主主義に有害か"

    # Create LLM configuration
    llm_config = LLMConfig(
        provider=LLMProvider.OPENAI,
        openai_model="gpt-4o-mini",
        temperature=0.8,  # Higher temperature for more diverse opinions
    )

    # Create discussion configuration
    discussion_config = DiscussionConfig(
        topic=topic,
        max_rounds=4,
        enable_evaluation=True,
        save_session=True,
    )

    # Define custom agents
    agents = [
        # Moderator
        AgentConfig(
            name="司会者",
            role=AgentRole.MODERATOR,
            system_prompt=(
                "あなたは技術と社会に関する専門的な議論の司会者です。"
                "参加者の多様な視点を引き出し、建設的な対話を促進してください。"
                "技術的な正確性と社会的影響の両面から議論を導いてください。"
            ),
        ),
        # Tech expert
        AgentConfig(
            name="テクノロジスト",
            role=AgentRole.PARTICIPANT,
            persona=(
                "IT企業でアルゴリズム開発に携わるエンジニア。"
                "技術的な仕組みと可能性について詳しく説明できる。"
                "技術の進歩による利便性向上を重視する立場。"
            ),
        ),
        # Social scientist
        AgentConfig(
            name="社会学者",
            role=AgentRole.PARTICIPANT,
            persona=(
                "メディアと社会の関係を研究する大学教授。"
                "情報の偏りが社会に与える影響を研究している。"
                "データに基づいた客観的な分析を重視する。"
            ),
        ),
        # Privacy advocate
        AgentConfig(
            name="市民活動家",
            role=AgentRole.PARTICIPANT,
            persona=(
                "デジタル権利と市民の自由を守る活動家。"
                "プライバシーとユーザーの自律性を重視。"
                "大企業による情報操作に批判的な立場。"
            ),
        ),
        # Business perspective
        AgentConfig(
            name="メディア経営者",
            role=AgentRole.PARTICIPANT,
            persona=(
                "オンラインメディア企業の経営者。"
                "ビジネスモデルとユーザーエンゲージメントの観点から発言。"
                "規制と自由市場のバランスを考える立場。"
            ),
        ),
        # Evaluator
        AgentConfig(
            name="分析官",
            role=AgentRole.EVALUATOR,
            system_prompt=(
                "あなたは議論の分析官です。"
                "各参加者の論点を整理し、議論の質を評価してください。"
                "論理的な整合性、根拠の妥当性、多様な視点の包含を評価基準とします。"
            ),
        ),
    ]

    # Create configuration
    config = Config(
        llm=llm_config,
        discussion=discussion_config,
        agents=agents,
    )

    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print(f"議論トピック: {topic}")
    print("=" * 50)
    print("\n参加者:")
    for agent in agents:
        if agent.role == AgentRole.PARTICIPANT:
            print(f"  - {agent.name}: {agent.persona[:50]}...")
    print()

    # Create and run discussion
    discussion = MultiAgentDiscussion(config)

    def on_message(message):
        print(f"\n[{message.agent_name}]")
        print(message.content[:500] + "..." if len(message.content) > 500 else message.content)
        print("-" * 30)

    result = discussion.run(message_callback=on_message)

    # Print evaluation
    print("\n" + "=" * 50)
    print("議論完了!")

    if result["evaluation"]:
        eval_data = result["evaluation"]
        print("\n【評価結果】")
        print(f"品質スコア: {eval_data['quality_score']:.2f}")
        print(f"\nサマリー:\n{eval_data['summary']}")

        if eval_data.get("key_points"):
            print("\n主要論点:")
            for i, point in enumerate(eval_data["key_points"], 1):
                print(f"  {i}. {point}")

        if eval_data.get("consensus_areas"):
            print("\n合意点:")
            for area in eval_data["consensus_areas"]:
                print(f"  ✓ {area}")

        if eval_data.get("disagreement_areas"):
            print("\n相違点:")
            for area in eval_data["disagreement_areas"]:
                print(f"  ✗ {area}")


if __name__ == "__main__":
    main()
