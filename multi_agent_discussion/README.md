# Multi-Agent Discussion Tool

複数のAIエージェントが一つの主題について議論を行うフレームワーク。
`deep_research_tool` と連携するオーケストレーションパイプラインも搭載。

## 機能

- **マルチエージェント議論**: モデレーター・参加者・評価者の役割を持つエージェントが議論を実行
- **マルチプロバイダー対応**: OpenAI (ChatGPT), Anthropic (Claude), Google (Gemini), Ollama (Llama), xAI (Grok)
- **プロキシ対応**: 全プロバイダーで HTTP/HTTPS プロキシを利用可能
- **オーケストレーター**: `deep_research_tool` と組み合わせた多段階パイプライン（統合・議論・洗練・競争評価）
- **セッション管理**: 議論の保存・読み込み・リプレイ
- **CLI**: Rich を使ったターミナルUI

## セットアップ

### 依存パッケージ

```bash
pip install openai anthropic google-generativeai httpx click rich
```

### 環境変数

使用するプロバイダーに応じてAPIキーを設定:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AI..."
export XAI_API_KEY="xai-..."

# プロキシ（任意）
export HTTPS_PROXY="http://proxy.example.com:8080"
```

Ollama はローカルで動作するため、APIキー不要:

```bash
ollama serve  # デフォルト: http://localhost:11434
```

### スクリプト/Jupyter上でAPIキーを指定

環境変数を使わず、コード内で直接APIキーを指定することも可能です:

```python
from multi_agent_discussion import LLMConfig, LLMProvider, run_discussion

# OpenAI
llm = LLMConfig(
    provider=LLMProvider.OPENAI,
    openai_api_key="sk-...",
)

# Anthropic
llm = LLMConfig(
    provider=LLMProvider.ANTHROPIC,
    anthropic_api_key="sk-ant-...",
)

# Google
llm = LLMConfig(
    provider=LLMProvider.GOOGLE,
    google_api_key="AI...",
)

# xAI
llm = LLMConfig(
    provider=LLMProvider.XAI,
    xai_api_key="xai-...",
)

# Ollama（APIキー不要、エンドポイントのみ）
llm = LLMConfig(
    provider=LLMProvider.OLLAMA,
    ollama_base_url="http://localhost:11434",
)

# 議論実行
result = run_discussion(
    topic="リモートワークの是非",
    llm_config=llm,
)
```

**APIキーパラメータ一覧:**

| プロバイダー | パラメータ |
|---|---|
| OpenAI | `openai_api_key` |
| Anthropic | `anthropic_api_key` |
| Google | `google_api_key` |
| xAI | `xai_api_key` |
| Ollama | 不要（`ollama_base_url` でエンドポイント指定） |

環境変数が設定されている場合は環境変数が優先されます。

---

## 基本的な使い方

### 1. 最もシンプルな議論

```python
from multi_agent_discussion import run_discussion

result = run_discussion(
    topic="リモートワークの是非",
    provider="openai",
    max_rounds=3,
)

print(result["transcript"])
```

デフォルトで「賛成派」「反対派」「中立派」の3エージェントが議論します。

### 2. カスタムペルソナを指定

```python
from multi_agent_discussion import run_discussion

result = run_discussion(
    topic="日本の教育改革について",
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    participant_personas=[
        {"name": "教育者", "persona": "30年の教職経験を持つベテラン教師"},
        {"name": "保護者", "persona": "小中学生の子供を持つ保護者"},
        {"name": "教育研究者", "persona": "比較教育学の研究者"},
    ],
    max_rounds=4,
)
```

### 3. エージェントごとに異なるLLMを使用

```python
from multi_agent_discussion import (
    Config, LLMConfig, AgentConfig, DiscussionConfig,
    AgentRole, LLMProvider, MultiAgentDiscussion,
)

config = Config(
    llm=LLMConfig(provider=LLMProvider.OPENAI),  # デフォルト
    discussion=DiscussionConfig(
        topic="AIアシスタントの将来",
        max_rounds=3,
    ),
    agents=[
        AgentConfig(
            name="司会者",
            role=AgentRole.MODERATOR,
            llm_config=LLMConfig(provider=LLMProvider.OPENAI, openai_model="gpt-4o"),
        ),
        AgentConfig(
            name="技術者 (Claude)",
            role=AgentRole.PARTICIPANT,
            persona="AI技術の専門家",
            llm_config=LLMConfig(provider=LLMProvider.ANTHROPIC),
        ),
        AgentConfig(
            name="倫理学者 (Gemini)",
            role=AgentRole.PARTICIPANT,
            persona="AI倫理の研究者",
            llm_config=LLMConfig(provider=LLMProvider.GOOGLE),
        ),
        AgentConfig(
            name="実務家 (Llama)",
            role=AgentRole.PARTICIPANT,
            persona="AI導入を推進する企業担当者",
            llm_config=LLMConfig(
                provider=LLMProvider.OLLAMA,
                ollama_model="llama3.2",
            ),
        ),
        AgentConfig(
            name="評価者",
            role=AgentRole.EVALUATOR,
            llm_config=LLMConfig(provider=LLMProvider.XAI),
        ),
    ],
)

discussion = MultiAgentDiscussion(config)
result = discussion.run()
```

### 4. プロキシ経由でアクセス

```python
from multi_agent_discussion import LLMConfig, LLMProvider

llm = LLMConfig(
    provider=LLMProvider.OPENAI,
    proxy_url="http://proxy.example.com:8080",
)
```

`HTTPS_PROXY` / `HTTP_PROXY` 環境変数が設定されていれば自動的に読み込まれます。

### 5. コールバックでリアルタイム表示

```python
from multi_agent_discussion import run_discussion

def on_progress(status, progress):
    print(f"[{progress*100:.0f}%] {status}")

def on_message(message):
    print(f"\n--- {message.agent_name} ---")
    print(message.content)

result = run_discussion(
    topic="宇宙開発の民営化",
    progress_callback=on_progress,
    message_callback=on_message,
)
```

### 6. セッションの保存と読み込み

議論結果は自動的に `./discussion_sessions/` に保存されます。

```python
from multi_agent_discussion import DiscussionSession
from pathlib import Path

# 読み込み
session = DiscussionSession.load(Path("discussion_sessions/session_abc123.json"))

# トランスクリプト生成
print(session.generate_transcript())

# メタ情報
print(f"トピック: {session.topic}")
print(f"ラウンド数: {len(session.rounds)}")
print(f"メッセージ数: {session.message_count}")
```

### 7. 情報収集付き議論（Research Participant）

各参加者が議論中にウェブ検索で情報を収集し、根拠に基づいた議論を行います。

```python
from multi_agent_discussion import run_discussion

# enable_search=True で情報収集を有効化
result = run_discussion(
    topic="量子コンピューティング分野の日本における未来は？",
    participant_personas=[
        {"name": "技術担当", "persona": "量子コンピューティングの技術動向に詳しい専門家"},
        {"name": "市場担当", "persona": "市場動向・ビジネス展開に詳しいアナリスト"},
        {"name": "法規制担当", "persona": "法規制・政策に詳しい専門家"},
    ],
    enable_search=True,
    search_config={"region": "jp-jp", "max_queries_per_turn": 2},
    max_rounds=3,
)
```

詳細な設定:

```python
from multi_agent_discussion import (
    Config, LLMConfig, AgentConfig, DiscussionConfig,
    AgentRole, LLMProvider, MultiAgentDiscussion,
)

config = Config(
    llm=LLMConfig(provider=LLMProvider.OPENAI),
    discussion=DiscussionConfig(topic="AIの未来", max_rounds=3),
    agents=[
        AgentConfig(
            name="コーディネーター",
            role=AgentRole.MODERATOR,
            persona="各専門家の意見を統合し議論を導くコーディネーター",
        ),
        AgentConfig(
            name="技術担当",
            role=AgentRole.RESEARCH_PARTICIPANT,  # 情報収集付き
            persona="技術動向に詳しい専門家",
            search_config={
                "region": "jp-jp",
                "max_queries_per_turn": 2,
                "max_results_per_query": 5,
            },
        ),
        AgentConfig(
            name="特許担当",
            role=AgentRole.RESEARCH_PARTICIPANT,
            persona="特許・知的財産に詳しい専門家",
            search_config={"region": "jp-jp"},
        ),
        AgentConfig(name="評価者", role=AgentRole.EVALUATOR),
    ],
)

discussion = MultiAgentDiscussion(config)
result = discussion.run()
```

**検索設定オプション:**

| パラメータ | デフォルト | 説明 |
|---|---|---|
| `enabled` | `True` | 検索機能の有効/無効 |
| `search_method` | `"duckduckgo"` | 検索方法（`duckduckgo` or `selenium`） |
| `max_queries_per_turn` | `3` | 1ターンあたりの最大クエリ数 |
| `max_results_per_query` | `5` | 1クエリあたりの最大結果数 |
| `region` | `"jp-jp"` | 検索リージョン |

---

## CLI

```bash
# 議論を開始
python -m multi_agent_discussion.cli discuss "AIの倫理的な問題について"
python -m multi_agent_discussion.cli discuss "リモートワークの是非" --max-rounds 5
python -m multi_agent_discussion.cli discuss "教育改革" -p anthropic -m claude-3-5-sonnet-20241022

# セッション一覧
python -m multi_agent_discussion.cli list-sessions

# セッション情報
python -m multi_agent_discussion.cli info session_abc123.json

# エクスポート
python -m multi_agent_discussion.cli export session_abc123.json -f markdown -o output.md

# リプレイ
python -m multi_agent_discussion.cli replay session_abc123.json
```

---

## オーケストレーター（パイプライン）

`deep_research_tool` と連携し、調査 → 議論/統合/洗練/競争評価 → レポート生成を自動化するパイプラインフレームワーク。

### プリセットパイプライン

5つのプリセットが用意されています:

| プリセット | フロー | 説明 |
|---|---|---|
| `multi_perspective` | 調査 → 統合 → レポート | 複数視点から調査し統合 |
| `debate_research` | 調査 → 議論 → ファクトチェック → レポート | 調査結果を議論で検証 |
| `iterative_refinement` | 調査 → 統合 → 洗練 → ファクトチェック → レポート | ライター・レビューアーで反復洗練 |
| `competitive_analysis` | 調査 → 競争評価 → 洗練 → レポート | 各結果を評価し最良を選択 |
| `full` | 全7ステージ | 最も包括的なパイプライン |

### プリセットを使う

```python
from multi_agent_discussion.orchestrator import Pipeline

# 多角的統合パイプライン
pipeline = Pipeline.preset("multi_perspective", "量子コンピュータの実用化")
context = pipeline.run()
print(context.final_report)

# 議論型調査パイプライン
pipeline = Pipeline.preset("debate_research", "再生可能エネルギーの未来")
context = pipeline.run()
```

### カスタムパイプラインの構築

```python
from multi_agent_discussion.orchestrator import (
    Pipeline,
    OrchestratorConfig,
    ResearchAgentConfig,
    create_orchestrator_config,
)
from multi_agent_discussion.orchestrator.stages import (
    ParallelResearchStage,
    SynthesisStage,
    RefinementStage,
    FactCheckStage,
    ReportStage,
)

# 設定を作成
config = create_orchestrator_config(
    topic="自動運転技術の社会実装",
    perspectives=[
        {"name": "技術者", "perspective": "技術的な実現可能性"},
        {"name": "法律家", "perspective": "法規制と責任の観点"},
        {"name": "都市計画者", "perspective": "交通インフラへの影響"},
    ],
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    proxy_url="http://proxy:8080",  # プロキシ対応
)

# ステージを自由に組み合わせ
pipeline = Pipeline(topic=config.topic, config=config)
pipeline.add_stage(ParallelResearchStage(name="調査", config=config))
pipeline.add_stage(SynthesisStage(name="統合", config=config))
pipeline.add_stage(RefinementStage(name="洗練", config=config))
pipeline.add_stage(FactCheckStage(name="検証", config=config))
pipeline.add_stage(ReportStage(name="レポート", config=config))

# 進捗コールバック
pipeline.set_progress_callback(lambda status, progress: print(f"[{progress*100:.0f}%] {status}"))

# 実行
context = pipeline.run()
print(context.final_report)
```

### 既存の調査ファイルを読み込む

`deep_research_tool` で生成済みのセッションファイルをパイプラインに取り込めます:

```python
from multi_agent_discussion.orchestrator import (
    Pipeline,
    OrchestratorConfig,
    ResearchAgentConfig,
    SynthesisStage,
    ReportStage,
    ParallelResearchStage,
)
from multi_agent_discussion.config import LLMConfig

config = OrchestratorConfig(
    topic="AIの倫理",
    llm_config=LLMConfig(),
    research_agents=[
        ResearchAgentConfig(
            name="技術調査",
            perspective="",  # ファイルからの読み込み時は空でOK
            from_file="./sessions/tech_research.json",
        ),
        ResearchAgentConfig(
            name="社会調査",
            perspective="",
            from_file="./sessions/social_research.json",
        ),
    ],
)

pipeline = Pipeline(topic=config.topic, config=config)
pipeline.add_stage(ParallelResearchStage(name="読込", config=config))
pipeline.add_stage(SynthesisStage(name="統合", config=config))
pipeline.add_stage(ReportStage(name="レポート", config=config))

context = pipeline.run()
```

### パイプラインの中間結果

各ステージの結果は `PipelineContext` に蓄積されます:

```python
context = pipeline.run()

# 各種結果にアクセス
context.research_results       # 調査結果（エージェント名 → ResearchResult）
context.synthesized_report     # 統合レポート
context.refined_reports        # 洗練の各イテレーション結果
context.discussion_transcript  # 議論のトランスクリプト
context.competitive_rankings   # 競争評価のランキング
context.fact_check_results     # ファクトチェック結果
context.final_report           # 最終レポート
context.stage_results          # 各ステージの実行記録

# JSON で保存・復元
context.save(output_dir="./results")
restored = PipelineContext.load(Path("./results/pipeline_xxx.json"))
```

---

## パイプラインステージ一覧

| ステージ | クラス | 説明 |
|---|---|---|
| 調査 | `ParallelResearchStage` | deep_research_tool で並列調査（フォールバック: LLM直接利用） |
| 統合 | `SynthesisStage` | 複数の調査結果を一つのレポートに統合 |
| 洗練 | `RefinementStage` | ライター・レビューアーの反復ループで品質向上 |
| 議論 | `DebateStage` | MultiAgentDiscussion を使ったエージェント間議論 |
| 競争評価 | `CompetitiveStage` | 各結果をスコアリングし最良を選択/マージ |
| ファクトチェック | `FactCheckStage` | 事実検証（deep_research_tool またはLLMフォールバック） |
| レポート | `ReportStage` | 全結果を最終レポートに編纂（Markdown/HTML） |

---

## プロジェクト構成

```
multi_agent_discussion/
├── __init__.py              # パッケージエントリポイント
├── config.py                # 設定（LLMConfig, AgentConfig, etc.）
├── main.py                  # MultiAgentDiscussion オーケストレーター
├── cli.py                   # CLIインターフェース
├── agents/
│   ├── __init__.py          # エージェントファクトリ
│   ├── base.py              # BaseAgent（LLMクライアント生成・プロキシ対応）
│   ├── moderator.py         # モデレーターエージェント
│   ├── participant.py       # 参加者エージェント
│   ├── research_participant.py  # 情報収集付き参加者エージェント
│   ├── search_mixin.py      # 検索機能ミックスイン
│   └── evaluator.py         # 評価者エージェント
├── conversation/
│   ├── __init__.py
│   ├── message.py           # Message, Turn, Round
│   └── session.py           # DiscussionSession（保存・読込・トランスクリプト）
├── orchestrator/
│   ├── __init__.py
│   ├── config.py            # OrchestratorConfig
│   ├── context.py           # PipelineContext
│   ├── pipeline.py          # Pipeline エンジン
│   ├── presets.py           # プリセットパイプライン
│   └── stages/
│       ├── base.py          # BaseStage
│       ├── research.py      # ParallelResearchStage
│       ├── synthesis.py     # SynthesisStage
│       ├── refinement.py    # RefinementStage
│       ├── debate.py        # DebateStage
│       ├── competitive.py   # CompetitiveStage
│       ├── fact_check.py    # FactCheckStage
│       └── report.py        # ReportStage
├── examples/
│   ├── basic_discussion.py
│   ├── custom_personas.py
│   ├── research_discussion.py  # 情報収集付き議論の例
│   └── multi_provider_discussion.py
└── tests/
    ├── test_config.py
    ├── test_session.py
    ├── test_imports.py
    ├── test_orchestrator.py
    └── test_research_agent.py  # 情報収集機能のテスト
```

## テスト

```bash
python -m pytest multi_agent_discussion/tests/ -v
```
