# Agent Orchestrator（日本語ガイド）

> English version: [README.md](README.md)

2つのコーディングエージェント — **Codex**（OpenAI `codex` CLI）と **Claude Code**
（Anthropic `claude` CLI）— をタスク上で**協調**させ、その様子をブラウザで**ライブ表示**する
オーケストレーター。さらに **ローカルLLM**（Ollama / LM Studio）や、ホスト型の
**Claude / GPT API** も参加者として追加できます。

```
┌──────────────┐   イベント (SSE)    ┌──────────────────────────┐
│   Web UI     │ ◀───────────────── │  オーケストレーター        │
│ (ブラウザ)    │  ─────────────────▶ │  ＋ 協調ストラテジー        │
└──────────────┘   実行 / 停止        └────────────┬─────────────┘
                                                   │ 共通インターフェース
                        ┌──────────────────────────┼───────────────────────────┐
                        ▼              ▼            ▼            ▼               ▼
                    Claude Code      Codex     Claude (API)  GPT (API)     ローカルLLM
                      (CLI)         (CLI)     anthropic SDK  openai SDK   OpenAI互換
```

## どこでも動く理由

オーケストレーターのコア **と** Web サーバは **Python 標準ライブラリのみ**で実装しています
（フレームワーク不要・ビルド不要・ネイティブバイナリ無し）。そのため `pip install` 無しで
すぐ起動でき、アンチウイルスに引っかかる実行ファイルもありません。ブラウザで開くだけです。
ホスト型 API のアダプタは公式 SDK（`anthropic` / `openai`）を**遅延 import** するので、
実際に選んだときだけ必要になります。

**Python 3.10 以上**が必要です。

## クイックスタート

```bash
# リポジトリのルートで実行:
python -m agent_orchestrator serve
# → ブラウザで http://127.0.0.1:8765/ を開く
```

あとは: タスクを入力 →**ストラテジー**を選択 → 各**ロール**にバックエンドを割り当て →
**Run collaboration** を押す。エージェントの応答（ターン）が起きた順にトランスクリプトへ
流れていきます。

> 何もインストールしていなくても **Mock（オフライン）** はすぐ使えます。`claude` CLI が
> PATH にあれば **Claude Code (CLI)** も即利用可能です。API バックエンドを使うには
> 下記の SDK をインストールしてください。

## 協調ストラテジー

| ストラテジー | ロール | 動作 |
|---|---|---|
| **Implementer + Reviewer** | implementer, reviewer | 一方が実装、他方がレビュー。承認またはラウンド上限まで反復。 |
| **Debate / Consensus** | debater A, debater B | 双方が異なる立場で議論し、最後に統合ターンで結論へ収束。 |
| **Planner + Executor** | planner, executor | 一方が計画、他方が実行。各ラウンドで計画を調整。 |

## バックエンド（アダプタ）

| バックエンド | 必要なもの |
|---|---|
| Mock（オフライン） | なし — 決定的な出力。デモ・テスト用 |
| Claude Code (CLI) | `claude` CLI が PATH 上にあること |
| Codex (CLI) | `codex` CLI が PATH 上にあること |
| Claude (API) | `pip install anthropic` ＋ `ANTHROPIC_API_KEY` |
| GPT (API) | `pip install openai` ＋ `OPENAI_API_KEY` |
| ローカルLLM | `pip install openai` ＋ OpenAI 互換のローカルサーバ（例: Ollama） |

任意の設定は `.env` に書けます（[.env.example](.env.example) 参照）。

```bash
pip install -r agent_orchestrator/requirements.txt   # API バックエンドを使う場合のみ
```

## ヘッドレス（ターミナル）での実行

```bash
# この環境で利用可能なバックエンド／ストラテジーを一覧:
python -m agent_orchestrator agents

# 協調をターミナルで実行:
python -m agent_orchestrator run \
    --task "順序を保ったままリストを重複排除する関数を書いて。" \
    --strategy implementer_reviewer --rounds 2 \
    --agent implementer=claude_code --agent reviewer=codex
```

## 設定（環境変数）

| 変数 | 用途 | 既定値 |
|---|---|---|
| `ORCHESTRATOR_HOST` / `ORCHESTRATOR_PORT` | サーバの待受 | `127.0.0.1` / `8765` |
| `ORCHESTRATOR_MAX_TOKENS` | API 生成の最大トークン | `4096` |
| `ORCHESTRATOR_CLI_TIMEOUT` | CLI 1ターンのタイムアウト(秒) | `600` |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | Claude (API) | — / `claude-opus-4-8` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | GPT (API) | — / `gpt-4o` |
| `LOCAL_LLM_BASE_URL` / `LOCAL_LLM_MODEL` | ローカルLLM | `http://localhost:11434/v1` / `llama3.1` |

## 拡張方法

メソッドを1つ実装するだけで新しい参加者を追加できます:

```python
from agent_orchestrator.adapters.base import AgentAdapter

class MyAdapter(AgentAdapter):
    kind = "my_backend"
    def _generate(self, prompt, system, history):
        return my_llm_call(system, history, prompt)
```

`agent_orchestrator/adapters/__init__.py` の `_BUILDERS` と `_CATALOG_META` に登録すれば、
UI に自動で表示されます。新しい協調パターンは `orchestrator/strategies.py` の
`Strategy` のサブクラスとして追加します。

## テスト

```bash
python -m unittest discover -s agent_orchestrator/tests -v
```

## ディレクトリ構成

```
agent_orchestrator/
├── config.py            # 列挙型＋設定（環境変数駆動）
├── cli.py / __main__.py # serve / run / agents
├── adapters/            # mock / cli_agent / api_agent ＋ レジストリ
├── orchestrator/        # events, session, strategies, engine
├── server/              # 標準ライブラリの HTTP＋SSE サーバ
│   └── static/          # index.html, style.css, app.js
└── tests/               # unittest 一式（mock アダプタで動作）
```

## よくある質問

- **API キーが無くても動く？** はい。Mock バックエンド（と、入っていれば `claude` CLI）で
  すべての機能を体験できます。
- **外部に通信する？** コア／UI は通信しません。通信が発生するのは、API／ローカルLLM
  バックエンドを選んで実行したときだけです。
- **ポートを変えたい。** `python -m agent_orchestrator serve --port 9000` か
  `ORCHESTRATOR_PORT` で変更できます。
