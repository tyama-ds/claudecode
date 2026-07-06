#!/usr/bin/env python3
"""
Example of running a multi-agent discussion with different LLM providers.

This example shows how to configure different agents with different LLM providers
(e.g., ChatGPT, Claude, Gemini, Llama, Grok).

Usage:
    # Set API keys for providers you want to use
    export OPENAI_API_KEY="your-openai-key"
    export ANTHROPIC_API_KEY="your-anthropic-key"
    export GOOGLE_API_KEY="your-google-key"
    export XAI_API_KEY="your-xai-key"

    # For Ollama, make sure it's running locally
    # ollama serve

    python multi_provider_discussion.py
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
)


def main():
    """Run a discussion with multiple LLM providers."""

    topic = "AIアシスタントの将来と人間との共存について"

    print(f"議論トピック: {topic}")
    print("=" * 60)
    print("\n各エージェントに異なるLLMプロバイダーを設定しています:\n")

    # ===== エージェントごとに異なるLLMを設定 =====

    # モデレーター: ChatGPT (OpenAI)
    moderator_llm = LLMConfig(
        provider=LLMProvider.OPENAI,
        openai_model="gpt-5-mini",
    )

    # 参加者1: Claude (Anthropic)
    participant1_llm = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        anthropic_model="claude-3-5-sonnet-20241022",
        temperature=0.7,
    )

    # 参加者2: Gemini (Google)
    participant2_llm = LLMConfig(
        provider=LLMProvider.GOOGLE,
        google_model="gemini-1.5-flash",
        temperature=0.7,
    )

    # 参加者3: Llama (Ollama - ローカル)
    participant3_llm = LLMConfig(
        provider=LLMProvider.OLLAMA,
        ollama_model="llama3.2",
        ollama_base_url="http://localhost:11434",
        temperature=0.7,
    )

    # 評価者: Grok (xAI)
    evaluator_llm = LLMConfig(
        provider=LLMProvider.XAI,
        xai_model="grok-beta",
        temperature=0.3,  # Lower temperature for evaluation
    )

    # ===== エージェント設定 =====
    agents = [
        # モデレーター (ChatGPT)
        AgentConfig(
            name="司会者 (GPT-5)",
            role=AgentRole.MODERATOR,
            llm_config=moderator_llm,
        ),

        # 参加者1 (Claude)
        AgentConfig(
            name="テクノロジスト (Claude)",
            role=AgentRole.PARTICIPANT,
            persona="AI技術の専門家。技術的な可能性と限界について詳しい。",
            llm_config=participant1_llm,
        ),

        # 参加者2 (Gemini)
        AgentConfig(
            name="倫理学者 (Gemini)",
            role=AgentRole.PARTICIPANT,
            persona="AI倫理の研究者。社会的影響と倫理的課題に関心がある。",
            llm_config=participant2_llm,
        ),

        # 参加者3 (Llama)
        AgentConfig(
            name="実務家 (Llama)",
            role=AgentRole.PARTICIPANT,
            persona="企業でAI導入を推進する実務家。現場の視点から意見を述べる。",
            llm_config=participant3_llm,
        ),

        # 評価者 (Grok)
        AgentConfig(
            name="分析官 (Grok)",
            role=AgentRole.EVALUATOR,
            llm_config=evaluator_llm,
        ),
    ]

    # 使用するプロバイダーを表示
    print("エージェント構成:")
    for agent in agents:
        provider = agent.llm_config.provider.value if agent.llm_config else "default"
        model = agent.llm_config.get_model() if agent.llm_config else "default"
        print(f"  - {agent.name}: {provider} ({model})")
    print()

    # ===== 設定の作成 =====
    # グローバルLLM設定（エージェント固有の設定がない場合のフォールバック）
    default_llm = LLMConfig(provider=LLMProvider.OPENAI)

    config = Config(
        llm=default_llm,
        discussion=DiscussionConfig(
            topic=topic,
            max_rounds=3,
            enable_evaluation=True,
        ),
        agents=agents,
    )

    # バリデーション（APIキーのチェック）
    # 注意: 実際に使用するプロバイダーのAPIキーのみ必要
    print("注意: 全てのプロバイダーを使用するには、対応するAPIキーが必要です。")
    print("利用可能なプロバイダーのみで実行してください。\n")

    # 実際に実行する場合はコメントを外してください
    # discussion = MultiAgentDiscussion(config)
    # result = discussion.run()

    print("設定例を表示しました。")
    print("実際に実行するには、必要なAPIキーを設定してください。")


def simple_multi_provider_example():
    """
    シンプルな2プロバイダーの例（OpenAI + Anthropic）
    """
    from multi_agent_discussion import run_discussion

    # カスタムペルソナでエージェントを作成
    # 各エージェントに異なるLLMを設定するには、Configを直接作成します

    topic = "リモートワークの是非"

    # ChatGPT用のLLM設定
    openai_llm = LLMConfig(
        provider=LLMProvider.OPENAI,
        openai_model="gpt-5-mini",
    )

    # Claude用のLLM設定
    anthropic_llm = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        anthropic_model="claude-3-5-sonnet-20241022",
    )

    agents = [
        AgentConfig(
            name="司会者",
            role=AgentRole.MODERATOR,
            llm_config=openai_llm,  # ChatGPTを使用
        ),
        AgentConfig(
            name="賛成派 (Claude)",
            role=AgentRole.PARTICIPANT,
            persona="リモートワークを推進する立場",
            llm_config=anthropic_llm,  # Claudeを使用
        ),
        AgentConfig(
            name="反対派 (GPT)",
            role=AgentRole.PARTICIPANT,
            persona="オフィス勤務を推進する立場",
            llm_config=openai_llm,  # ChatGPTを使用
        ),
        AgentConfig(
            name="評価者",
            role=AgentRole.EVALUATOR,
            llm_config=anthropic_llm,  # Claudeを使用
        ),
    ]

    config = Config(
        llm=openai_llm,  # デフォルト
        discussion=DiscussionConfig(topic=topic, max_rounds=2),
        agents=agents,
    )

    # 実行
    # discussion = MultiAgentDiscussion(config)
    # result = discussion.run()

    print("\n=== シンプルな2プロバイダー例 ===")
    print(f"トピック: {topic}")
    for agent in agents:
        provider = agent.llm_config.provider.value
        print(f"  {agent.name}: {provider}")


if __name__ == "__main__":
    main()
    print("\n" + "=" * 60 + "\n")
    simple_multi_provider_example()
