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
- **図表自動生成**: レポートに図や表を自動で追加
- **出力量調整**: ページ数・文字数を指定してレポートの長さを制御

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

## 環境変数の設定

```bash
# OpenAI APIを使用する場合
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic APIを使用する場合
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

---

## 基本機能：リサーチとレポート生成

### 概要

Deep Research Toolの基本機能は、指定したテーマについて自動でWeb検索を行い、収集した情報を整理してレポートを生成することです。

### 使用イメージ

```
ユーザー: 「日本のEV市場の現状と将来展望について調査してほしい」
    ↓
Deep Research Tool:
    1. テーマを分析し、調査計画を作成
    2. 「日本 EV 市場規模」「電気自動車 販売台数 推移」などのクエリで検索
    3. 検索結果から関連情報を抽出
    4. 情報の不足部分を特定し、追加検索を実行
    5. 収集した情報を構造化してレポートにまとめる
    ↓
出力: 10ページ程度の包括的な調査レポート
```

### CLIでの使用

```bash
# 基本的なリサーチ
deep-research research "日本のEV市場の現状と将来展望"

# オプションを指定
deep-research research "再生可能エネルギー市場分析" \
    --provider anthropic \
    --model claude-3-5-sonnet \
    --iterations 5 \
    --output-format docx \
    --output-dir ./reports

# 追加文書を含める
deep-research research "競合他社分析" \
    --documents previous_report.pdf \
    --documents reference_data.xlsx
```

### Pythonでの使用

```python
from deep_research_tool import DeepResearchTool
from deep_research_tool.config import create_config

# 設定を作成
config = create_config(
    provider="anthropic",
    research_iterations=5,
    output_format="docx",
)

# ツールを初期化してリサーチ実行
tool = DeepResearchTool(config)
result = tool.run(
    query="日本のAI産業の現状と将来展望",
    requirements="最新のトレンドと主要企業の動向を含めること",
)

print(f"レポート: {result['report_path']}")
print(f"エビデンス: {result['evidence_json']}")
```

### 出力ファイル

| ファイル | パス | 説明 |
|---------|------|------|
| レポート | `output/reports/research_report_[session_id].[format]` | 生成されたレポート |
| エビデンス (JSON) | `output/evidence/evidence_[session_id].json` | 参照元の詳細情報 |
| エビデンス (CSV) | `output/evidence/evidence_[session_id].csv` | 参照元一覧（Excel対応） |
| セッション情報 | `output/session_[session_id].json` | リサーチプロセスの記録 |

---

## 追加機能1：ファクトチェック（ハルシネーション検証）

### 概要

生成されたレポートの内容が、収集したエビデンスに基づいているかを自動検証します。AIが生成した内容が「事実」なのか「推測」なのかを判定し、信頼性の低い部分を特定します。

### 使用イメージ

```
生成されたレポート内容:
「トヨタは2025年までに30車種のEVを発売予定で、
 2030年の世界販売目標は350万台である」
    ↓
ファクトチェック結果:
[VERIFIED] 「30車種のEV発売」- エビデンスで確認済み
           出典: トヨタ公式プレスリリース (2021年12月)
[UNVERIFIED] 「350万台」- エビデンスでは数値が異なる
           発見した数値: 550万台 (2023年最新計画)
           → 要確認
```

### CLIでの使用

```bash
# リサーチ完了後、検証レポートを生成
deep-research verify ./output/session_abc123.json

# 検証の厳密さを指定
deep-research verify ./output/session_abc123.json --strictness high

# 出力先を指定
deep-research verify ./output/session_abc123.json \
    --output ./verification_report.html
```

### Pythonでの使用

```python
from deep_research_tool.verification import HallucinationVerifier
from deep_research_tool.research.researcher import ResearchSession
from deep_research_tool.evidence.locker import EvidenceLocker

# セッションとエビデンスを読み込み
session = ResearchSession.load("./output/session_abc123.json")
evidence = EvidenceLocker.load_from_json("./output/evidence/evidence_abc123.json")

# 検証を実行
verifier = HallucinationVerifier(llm_client, strictness="high")
result = verifier.verify_report(session, evidence)

# 結果を確認
print(f"検証済み項目: {result.verified_count}")
print(f"未検証項目: {result.unverified_count}")
print(f"全体信頼度: {result.overall_confidence}")

# 問題のある箇所を表示
for finding in result.findings:
    if finding.status == "unverified":
        print(f"要確認: {finding.claim}")
        print(f"理由: {finding.reason}")
```

### 検証結果の見方

| ステータス | 意味 | 対応方法 |
|-----------|------|---------|
| `VERIFIED` | エビデンスで確認済み | そのまま使用可能 |
| `PARTIALLY_VERIFIED` | 一部確認済み | 詳細を確認推奨 |
| `UNVERIFIED` | エビデンスで確認できず | 追加調査または削除を検討 |
| `CONTRADICTED` | エビデンスと矛盾 | 修正が必要 |

---

## 追加機能2：図表の自動追加

### 概要

生成されたレポートに、図（画像）や表（データテーブル）を自動で追加します。参照元のWebページから関連画像を取得したり、テキスト内の数値データから表やグラフを自動生成したりします。

### 使用イメージ

```
レポート内容:
「EV市場規模は2020年に1兆円、2021年に1.5兆円、
 2022年に2兆円と急成長している...」
    ↓
図表追加後:
「EV市場規模は2020年に1兆円、2021年に1.5兆円、
 2022年に2兆円と急成長している...」

 [表1: EV市場規模の推移]
 | 年    | 市場規模 |
 |-------|---------|
 | 2020  | 1兆円   |
 | 2021  | 1.5兆円 |
 | 2022  | 2兆円   |

 [図1: EV市場規模の推移グラフ]
 （自動生成された棒グラフ）

 [図2: 主要EVメーカーのシェア]
 （参照元から取得した画像、出典付き）
```

### CLIでの使用

```bash
# レポート生成後に図表を追加
deep-research add-figures ./output/session_abc123.json

# オプションを指定
deep-research add-figures ./output/session_abc123.json \
    --include-images \        # Webページから画像を取得
    --include-tables \        # 数値データから表を生成
    --include-charts \        # 表からグラフを自動生成
    --max-images 3            # セクションあたりの最大画像数

# 特定のレポートファイルに追加
deep-research add-figures ./output/session_abc123.json \
    --report ./output/reports/my_report.md

# LLMを使用してより高度な分析
deep-research add-figures ./output/session_abc123.json \
    --provider anthropic
```

### Pythonでの使用

```python
from deep_research_tool.report.figure_table_generator import (
    FigureTableGenerator,
    add_figures_to_report,
)
from deep_research_tool.research.researcher import ResearchSession
from deep_research_tool.evidence.locker import EvidenceLocker

# セッションとエビデンスを読み込み
session = ResearchSession.load("./output/session_abc123.json")
evidence = EvidenceLocker.load_from_json("./output/evidence/evidence_abc123.json")

# 図表ジェネレーターを作成
generator = FigureTableGenerator(
    output_dir="./output/figures",
    language="ja",
    max_images_per_section=2,
)

# 図表を生成
collection = generator.generate_figures_and_tables(
    session=session,
    evidence_locker=evidence,
    include_images=True,
    include_tables=True,
    include_charts=True,
)

print(f"生成された図: {len(collection.figures)}")
print(f"生成された表: {len(collection.tables)}")
print(f"生成されたグラフ: {len(collection.charts)}")

# Markdownレポートに追加
with open("./output/reports/report.md", "r") as f:
    content = f.read()

updated_content = generator.add_figures_to_markdown(content, collection)

with open("./output/reports/report_with_figures.md", "w") as f:
    f.write(updated_content)
```

### 生成される図表の種類

| 種類 | 説明 | 生成条件 |
|-----|------|---------|
| 画像 | 参照元Webページから取得 | 関連性の高い画像が存在する場合 |
| 表 | テキストから数値を抽出 | 年次データや比較データがある場合 |
| 折れ線グラフ | 時系列データを可視化 | 3点以上の時系列データがある場合 |
| 棒グラフ | 比較データを可視化 | カテゴリ別比較データがある場合 |

---

## 追加機能3：出力量の調整（ページ数・文字数指定）

### 概要

生成するレポートのページ数や文字数を指定できます。指定した量に近づくように、コンテンツの量を調整します。

- **目標より多い場合**: 冗長な部分を削減し、要点を絞る
- **目標より少ない場合**: 追加のリサーチを実行してコンテンツを拡充

### 使用イメージ

```
例1: 10ページのレポートを指定
ユーザー: 「10ページ程度でまとめてほしい」
    ↓
現在の生成量: 15ページ分
    ↓
調整: 各セクションの内容を要約し、重要度の低い情報を削減
    ↓
出力: 約10ページのレポート

例2: 20ページのレポートを指定
ユーザー: 「20ページ以上の詳細レポートが欲しい」
    ↓
現在の生成量: 8ページ分
    ↓
調整: 追加リサーチを実行し、各セクションを深堀り
    ↓
出力: 約20ページの詳細レポート
```

### CLIでの使用

```bash
# ページ数を指定
deep-research research "市場分析" --target-pages 10

# 文字数を指定
deep-research research "技術調査" --target-characters 25000

# 他のオプションと組み合わせ
deep-research research "競合分析レポート" \
    --provider anthropic \
    --iterations 5 \
    --output-format docx \
    --target-pages 15

# 既存のセッションからレポート再生成時に指定
deep-research report ./output/session_abc123.json \
    --format pdf \
    --target-pages 10
```

### Pythonでの使用

```python
from deep_research_tool import run_research

# ページ数を指定してリサーチ
result = run_research(
    query="日本の半導体産業の現状",
    provider="anthropic",
    iterations=5,
    output_format="pdf",
    target_pages=15,  # 約15ページを目標
)

# または文字数で指定
result = run_research(
    query="AIトレンド分析",
    provider="openai",
    target_characters=30000,  # 約30,000文字を目標
)
```

### 詳細な制御（レポート生成時）

```python
from deep_research_tool.report import ReportGenerator, ReportFormat
from deep_research_tool.research.researcher import ResearchSession
from deep_research_tool.evidence.locker import EvidenceLocker

# セッションとエビデンスを読み込み
session = ResearchSession.load("./output/session_abc123.json")
evidence = EvidenceLocker.load_from_json("./output/evidence/evidence_abc123.json")

# レポートジェネレーターを作成
generator = ReportGenerator(output_dir="./output/reports")

# 現在の長さを確認
length_info = generator.get_length_info(session)
print(f"現在の文字数: {length_info.total_characters:,}")
print(f"推定ページ数: {length_info.estimated_pages:.1f}")

# ページ数を指定してレポート生成
report_path = generator.generate_report(
    session=session,
    evidence_locker=evidence,
    format=ReportFormat.PDF,
    target_pages=10,
)
```

### 調整の仕組み

| 状態 | 調整方法 |
|-----|---------|
| 目標の90-110%以内 | 調整なし |
| 目標の110%超過 | 文章を要約・削減（文境界を考慮） |
| 目標の90%未満 | 追加リサーチを実行して拡充 |

---

## 設定オプション一覧

| オプション | 説明 | デフォルト |
|-----------|------|----------|
| `provider` | LLMプロバイダー (openai/anthropic) | openai |
| `model` | 使用するモデル名 | gpt-4o-mini |
| `search_method` | 検索方法 (duckduckgo/selenium) | duckduckgo |
| `research_iterations` | リサーチループ回数 | 3 |
| `output_format` | 出力形式 (markdown/docx/pdf/html) | markdown |
| `enable_verification` | 検証機能の有効化 | True |
| `verification_strictness` | 検証の厳密さ (low/medium/high) | medium |
| `target_pages` | 目標ページ数 | None (無制限) |
| `target_characters` | 目標文字数 | None (無制限) |

---

## リサーチフロー

```
1. クエリ分析
   └─ 入力されたクエリを分析し、調査計画と目次を作成

2. 検索クエリ生成
   └─ 目次の各セクションに対する検索クエリを生成

3. Web検索・情報抽出
   └─ 検索結果から関連情報を抽出

4. リサーチループ
   └─ 情報の不足部分を特定し、追加クエリで補完（設定回数繰り返し）

5. 出力量調整（オプション）
   └─ 目標ページ数/文字数に合わせてコンテンツを調整

6. 情報統合
   └─ 収集した情報をセクションごとに統合

7. 検証（オプション）
   └─ ハルシネーションリスクのある内容を特定

8. レポート生成
   └─ 指定形式でレポートを出力

9. 図表追加（オプション）
   └─ レポートに図や表を追加
```

---

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
│   ├── locker.py
│   └── quality_evaluator.py
├── report/              # レポート生成
│   ├── generator.py
│   ├── length_controller.py
│   └── figure_table_generator.py
└── utils/               # ユーティリティ
    ├── document_reader.py
    └── helpers.py
```

---

## ライセンス

MIT License

## 貢献

プルリクエストや課題報告を歓迎します。
