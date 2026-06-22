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
ホスト型 API のアダプタも **SDK 不使用**で、標準ライブラリ（`urllib`）でプロバイダの REST API を
直接呼びます。よって**コンパイル済み依存は一切なく、インストールも不要** — API バックエンドに
必要なのは API キーだけです。`.exe`・インストーラ・コンパイル済み wheel が無いため、
**監査の厳しい／ロックダウンされた端末でも扱いやすい**設計です。

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
| **Round-robin（自由対話）** | agent A, B, C | 複数エージェントが全会話を共有しつつ自由に議論し、最後に結論を統合。 |
| **Panel + Judge（3者＋審判）** | contender A, B, C, judge | 3体が異なる主張を提示し、公平な審判が評価して裁定を下す。 |
| **Doc authoring（文章共同制作）** | writer, editor | 共有**Artifact**（文書）を writer が起草・改稿し、editor が各版を批評。承認まで反復。 |
| **Code authoring（コード共同制作）** | implementer, reviewer | 単一の**Artifact**（コード）を implementer が編集し、reviewer が各版を批評。承認まで反復。 |
| **Workspace build（実ファイル開発）** | implementer, reviewer | **実ディレクトリ**上で implementer がファイルを作成/編集し、reviewer が差分を批評。承認まで反復。 |
| **Custom（自由定義）** | 自分で定義 (2〜5体) | 参加者ごとにバックエンド・モデル・人格を自由に設定し、議論して結論を統合。 |

authoring 系は共有**Artifact**（1つの育つドキュメント/コード）を作ります。トランスクリプト上部の
専用パネルに表示され、**版管理・Preview/Diff 切替・コピー・ダウンロード**が可能です。編集役は
更新後の全文を `<ARTIFACT>…</ARTIFACT>` タグで出力し、レビュー役は指摘のみを返します。

**Workspace build** はさらに一歩進み、**ディスク上の実ディレクトリ**へ書き込みます。実装役は
変更ファイルを `<FILE path="…">…全文…</FILE>` で出力し、オーケストレーターがワークスペース
直下に限定して書き込み（`..` や絶対パスは拒否）、ファイルごとの unified diff を計算して
Workspace パネル（ファイル一覧＋色付き差分）に表示します。変更は作業ツリーに残し、ステージ／
コミットは行いません（レビューと commit はユーザーが実施）。ワークスペースは既定でサーバ起動
ディレクトリ、実行ごとに上書き指定可能です。（これはバックエンド非依存の Phase 2。CLI 自身に
よる実ファイル編集は今後の予定です。）

**各ロールごとに**「バックエンド・モデル・人格(system prompt)」を個別に指定できます
（UI上で編集可）。1回の実行で Claude Code・Codex・GPT・ローカルモデルを**混在**させたり、
任意のロールの指示文を上書きできます。

すべてのストラテジーは**スクラッチパッド**（チーム共有の黒板）を持ち、各エージェントは
`NOTE:` 行を書くことで共有メモを残せます。トランスクリプト上部に固定表示され、協調の
共有状態が常に見えます。

**会話の文脈**は可能な限りネイティブに共有されます。対応バックエンド（CLI／API）には
会話を構造化された **history（メッセージ列）** として渡し（自分の発言=assistant、他者=user）、
非対応のバックエンドは自動的に「トランスクリプトをプロンプトに埋め込む」方式へフォール
バックします。各ターンには `ctx: history` / `ctx: prompt` のタグが付き、どちらの方式かが分かります。

## バックエンド（アダプタ）

| バックエンド | 必要なもの |
|---|---|
| Mock（オフライン） | なし — 決定的な出力。デモ・テスト用 |
| Claude Code (CLI) | `claude` CLI が PATH 上にあること |
| Codex (CLI) | `codex` CLI が PATH 上にあること |
| Claude (API) | `ANTHROPIC_API_KEY`（インストール不要・標準ライブラリHTTP） |
| GPT (API) | `OPENAI_API_KEY`（インストール不要・標準ライブラリHTTP） |
| ローカルLLM | OpenAI 互換のローカルサーバ（例: Ollama）（インストール不要） |

任意の設定は `.env` に書けます（[.env.example](.env.example) 参照）。**インストールするパッケージは
ありません** — 使うホスト型バックエンドの API キーだけ用意してください。

API キー・モデル・Base URL・HTTP(S) **プロキシ**は、アプリ内の **Settings（⚙）** からも入力できます
（対応する環境変数が未設定のときに使用。値はこのローカルサーバのメモリ内のみに保持し、再表示はしません）。
Base URL を変えれば任意の **OpenAI 互換プロバイダ**に向けられます。

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
| `LOCAL_LLM_USE_PROXY` | ローカルLLMをproxy経由で送信（`1`/`true` で有効。既定は直結） | 無効 |

ローカルLLM はエンドポイントが localhost のことが多いため**既定で直結**します。proxy 経由に
したい場合は、Settings（⚙）の *Local LLM* で **Send via proxy** にチェック（または
`LOCAL_LLM_USE_PROXY=1`）してください。

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
