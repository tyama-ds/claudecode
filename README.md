# Deep Research Tool

AIを活用した自動リサーチツール。OpenAI/Anthropic APIとWeb検索を組み合わせて、包括的な調査レポートを自動生成します。

## 特徴

- **マルチLLMサポート**: OpenAI (GPT-4o-mini, GPT-5等) および Anthropic (Claude Opus, Sonnet, Haiku) に対応
- **柔軟なWeb検索**: DuckDuckGo API または Selenium（動的サイト対応）による情報収集
- **自動リサーチループ**: クエリ連想→検索→情報抽出を設定回数繰り返し
- **Evidence Locker**: 全ての参照元を追跡し、引用として管理
- **ハルシネーション検証**: 生成内容の信頼性を自動検証
- **複数形式での出力**: Markdown, DOCX, PDF, HTML形式でレポート生成
- **追加文書サポート**: 既存のPDFやレポートを参照として入力可能

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/your-repo/deep-research-tool.git
cd deep-research-tool

# 依存関係をインストール
pip install -e .

# Seleniumを使用する場合
pip install -e ".[selenium]"

# 全ての機能をインストール
pip install -e ".[all]"
```

## 使用方法

### 環境変数の設定

```bash
# OpenAI APIを使用する場合
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic APIを使用する場合
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### CLIでの使用

```bash
# 基本的なリサーチ
deep-research research "AI trends in healthcare 2024"

# オプションを指定
deep-research research "Renewable energy market analysis" \
    --provider anthropic \
    --model claude-3-5-sonnet \
    --iterations 5 \
    --output-format docx \
    --output-dir ./reports

# 追加文書を含める
deep-research research "Market comparison" \
    --documents previous_report.pdf \
    --documents reference_paper.docx

# 検証レポートを生成
deep-research verify ./output/report.md --strictness high
```

### Pythonコードでの使用

```python
from deep_research_tool import DeepResearchTool
from deep_research_tool.config import create_config

# 設定を作成
config = create_config(
    provider="openai",
    research_iterations=3,
    output_format="markdown",
)

# ツールを初期化
tool = DeepResearchTool(config)

# リサーチを実行
result = tool.run(
    query="日本のAI産業の現状と将来展望",
    requirements="最新のトレンドと主要企業の動向を含めること",
)

print(f"レポート: {result['report_path']}")
print(f"エビデンス: {result['evidence_json']}")
```

### 簡易関数での使用

```python
from deep_research_tool import run_research

result = run_research(
    query="Climate change impacts on agriculture",
    provider="anthropic",
    iterations=5,
    output_format="docx",
    verbose=True,
)
```

## プロジェクト構造

```
deep_research_tool/
├── __init__.py          # パッケージ初期化
├── config.py            # 設定管理
├── main.py              # メインインターフェース
├── cli.py               # CLIインターフェース
├── api/                 # LLM APIクライアント
│   ├── base.py
│   ├── openai_client.py
│   └── anthropic_client.py
├── search/              # Web検索モジュール
│   ├── base.py
│   ├── duckduckgo.py
│   └── selenium_browser.py
├── research/            # リサーチコアロジック
│   ├── query_generator.py
│   ├── content_extractor.py
│   └── researcher.py
├── verification/        # ハルシネーション検証
│   └── verifier.py
├── evidence/            # Evidence Locker
│   └── locker.py
├── report/              # レポート生成
│   └── generator.py
├── utils/               # ユーティリティ
│   ├── document_reader.py
│   └── helpers.py
└── examples/            # 使用例
```

## リサーチフロー

1. **クエリ分析**: 入力されたクエリを分析し、調査計画と目次を作成
2. **検索クエリ生成**: 目次の各セクションに対する検索クエリを生成
3. **Web検索・情報抽出**: 検索結果から関連情報を抽出
4. **リサーチループ**: 情報の不足部分を特定し、追加クエリで補完（設定回数繰り返し）
5. **情報統合**: 収集した情報をセクションごとに統合
6. **検証**: ハルシネーションリスクのある内容を特定
7. **レポート生成**: 指定形式でレポートを出力

## 出力ファイル

- **レポート**: `output/reports/research_report_[session_id].[format]`
- **エビデンス (JSON)**: `output/evidence/evidence_[session_id].json`
- **エビデンス (CSV)**: `output/evidence/evidence_[session_id].csv`
- **検証レポート**: `output/verification_[session_id].html`
- **セッション情報**: `output/session_[session_id].json`

## 設定オプション

| オプション | 説明 | デフォルト |
|-----------|------|----------|
| `provider` | LLMプロバイダー (openai/anthropic) | openai |
| `model` | 使用するモデル名 | gpt-4o-mini |
| `search_method` | 検索方法 (duckduckgo/selenium) | duckduckgo |
| `research_iterations` | リサーチループ回数 | 3 |
| `output_format` | 出力形式 (markdown/docx/pdf/html) | markdown |
| `enable_verification` | 検証機能の有効化 | True |
| `verification_strictness` | 検証の厳密さ (low/medium/high) | medium |

## ライセンス

MIT License

## 貢献

プルリクエストや課題報告を歓迎します。
