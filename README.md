# Deep Research Tool

AIを活用した自動リサーチツール。OpenAI/Anthropic APIとWeb検索を組み合わせて、包括的な調査レポートを自動生成します。

## 特徴

- **マルチLLMサポート**: OpenAI (GPT-4o-mini, GPT-5等)、Anthropic (Claude Opus, Sonnet, Haiku)、**ローカルLLM (Ollama, vLLM)** に対応
- **柔軟なWeb検索**: DuckDuckGo API または Selenium（動的サイト対応）による情報収集
- **自動リサーチループ**: クエリ連想→検索→情報抽出を設定回数繰り返し
- **Evidence Locker**: 全ての参照元を追跡し、引用として管理
- **ハルシネーション検証**: 生成内容の信頼性を自動検証
- **複数形式での出力**: Markdown, DOCX, PDF, HTML形式でレポート生成
- **追加文書サポート**: 既存のPDFやレポートを参照として入力可能
- **図表自動生成**: レポートに図や表を自動で追加
- **Extended Mode**: 検索結果のサイトを深くクローリングして詳細情報を収集
- **出力量調整**: ページ数・文字数を指定してレポートの長さを制御
- **段階的コンテンツ生成**: Multi-Pass Synthesisによる高品質なレポート生成
- **論理的整合性チェック**: レポート全体の論理的な流れを自動検証
- **トークン使用量追跡**: API呼び出しのトークン消費を詳細に追跡
- **DeepThink推論強化**: 複雑なトピックに対する深い推論機能
- **多言語検索**: 複数言語での同時検索と結果統合
- **ローカルデータ分析**: PDFやExcelなどのローカルファイルを直接分析
- **データサルベージ**: エラー時のデータ復旧とCSVエクスポート（日本語エンコーディング対応）
- **高速クロールモード**: 並列フェッチとバッチ/並列LLM評価で情報収集を高速化

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

# ローカルLLMを使用する場合
export LOCAL_LLM_BASE_URL="http://localhost:11434"  # Ollama
# export LOCAL_LLM_BASE_URL="http://localhost:8000"  # vLLM
export LOCAL_LLM_API_KEY=""  # 認証が必要な場合のみ
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
from deep_research_tool import DeepResearchTool, create_config

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

## 追加機能3：Extended Mode（深層サイトクローリング）

### 概要

Extended Modeは、通常の検索に加えて、検索結果のWebサイトを深くクローリングし、より詳細で包括的な情報を収集する機能です。検索結果のURLだけでなく、そのサイト内の関連ページも探索して情報を収集します。

### 使用イメージ

```
通常モード:
検索クエリ: "日本のEV市場"
    ↓
検索結果: 10件のURL
    ↓
各URLのページ内容を取得
    ↓
レポート生成

Extended Mode:
検索クエリ: "日本のEV市場"
    ↓
検索結果: 10件のURL
    ↓
各URLのサイトをクローリング（最大50ページ）
    ├─ site1.com/ev-market → /ev-market/japan → /ev-market/trends
    ├─ site2.com/industry → /industry/auto → /industry/ev
    └─ ...
    ↓
関連性の高いページから情報を抽出
    ↓
発見したトピックから追加クエリを生成
    ↓
より詳細で包括的なレポート生成
```

### 安全対策（無限ループ防止）

Extended Modeは以下の安全対策を実装しています：

| 制限 | 値 | 説明 |
|-----|---|------|
| グローバル上限 | **50ページ** | 全体で最大50ページまで |
| 1サイトあたり | 10ページ | デフォルト設定 |
| 最大深度 | 2階層 | シードURLから2リンク先まで |
| 最大サイト数 | 3サイト | 1検索あたり |
| 同一ドメイン制限 | あり | 外部リンクは追跡しない |

### クローリングの仕組み

```
BFS（幅優先探索）アルゴリズム:

シードURL (深度0)
    ├─ /page1 (深度1) → 関連度スコア 0.8 → 採用
    │     ├─ /page1/sub1 (深度2) → 関連度スコア 0.3 → 不採用
    │     └─ /page1/sub2 (深度2) → 関連度スコア 0.7 → 採用
    ├─ /page2 (深度1) → 関連度スコア 0.2 → 不採用
    └─ /page3 (深度1) → 関連度スコア 0.9 → 採用
          └─ 深度2ページ...

処理フロー:
1. 訪問済みURLをセットで管理（重複アクセス防止）
2. 深度がmax_depthを超えたらスキップ
3. ページ数がmax_pagesを超えたら終了
4. グローバル上限(50)に達したら即座に終了
5. 関連度スコアが閾値未満のページは結果から除外
```

### CLIでの使用

```bash
# Extended Modeを有効化
deep-research research "市場分析" --extended-mode

# クローリング設定をカスタマイズ
deep-research research "技術調査" \
    --extended-mode \
    --crawl-max-pages 15 \
    --crawl-max-depth 3 \
    --crawl-max-sites 5

# 他のオプションと組み合わせ
deep-research research "競合分析" \
    --extended-mode \
    --provider anthropic \
    --iterations 5 \
    --output-format pdf \
    --target-pages 20
```

### Pythonでの使用

```python
from deep_research_tool import run_research

# Extended Modeでリサーチ
result = run_research(
    query="日本のAI産業の動向",
    provider="anthropic",
    iterations=5,
    output_format="pdf",
    extended_mode=True,          # Extended Mode有効化
    crawl_max_pages=15,          # 1サイトあたり最大15ページ
    crawl_max_depth=2,           # 最大深度2
    crawl_max_sites=5,           # 最大5サイトをクローリング
)

print(f"レポート: {result['report_path']}")
```

### 詳細な設定（Config使用）

```python
from deep_research_tool import DeepResearchTool, create_config

config = create_config(
    provider="anthropic",
    research_iterations=5,
    output_format="docx",

    # Extended Mode設定
    extended_mode=True,
    crawl_max_pages=10,      # 1サイトあたり最大10ページ
    crawl_max_depth=2,       # シードURLから最大2階層
    crawl_max_sites=3,       # 最大3サイトをクローリング
)

tool = DeepResearchTool(config)
result = tool.run(
    query="再生可能エネルギー市場",
    requirements="詳細な市場データと事例を含める",
)
```

### Extended Modeが適しているケース

| ケース | 説明 |
|-------|------|
| 専門的な調査 | 特定サイト内の詳細情報が必要な場合 |
| 包括的なレポート | より多くの情報源が必要な場合 |
| ニッチなトピック | 検索結果だけでは情報が不足する場合 |
| 企業サイトの調査 | 企業の公式サイト内を深く調査したい場合 |

### 注意事項

- クローリングには時間がかかります（通常の2-3倍）
- 対象サイトのrobots.txtを尊重します
- 過度なアクセスを避けるため、リクエスト間に0.5秒の遅延を設けています
- グローバル上限（50ページ）は安全のため変更できません

---

## 追加機能4：出力量の調整（ページ数・文字数指定）

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

## 追加機能5：多言語検索（Multilingual Search）

### 概要

複数の言語で同時に検索を行い、より広範な情報を収集する機能です。検索クエリをLLMで自動翻訳し、各言語で並列検索を実行。結果を重複排除・統合してレポートに反映します。

### 使用イメージ

```
検索クエリ: "日本のEV市場の現状"
    ↓
【クエリ翻訳】
├─ 日本語: "日本のEV市場の現状"（原文）
├─ English: "Current state of Japan's EV market"
├─ 中文: "日本电动汽车市场现状"
└─ 한국어: "일본 전기차 시장 현황"
    ↓
【並列検索】
├─ 日本語検索 → 10件の結果
├─ 英語検索 → 10件の結果
├─ 中国語検索 → 10件の結果
└─ 韓国語検索 → 10件の結果
    ↓
【重複排除・統合】
├─ URL重複を除去
├─ タイトル類似度で重複検出
└─ 言語重み付けでスコアリング
    ↓
結果: 30件のユニークな検索結果（言語横断）
```

### 対応言語

| コード | 言語 | ネイティブ名 |
|--------|------|-------------|
| `ja` | Japanese | 日本語 |
| `en` | English | English |
| `zh` | Chinese | 中文 |
| `ko` | Korean | 한국어 |
| `de` | German | Deutsch |
| `fr` | French | Français |
| `es` | Spanish | Español |
| `pt` | Portuguese | Português |
| `ru` | Russian | Русский |
| `it` | Italian | Italiano |
| `nl` | Dutch | Nederlands |
| `pl` | Polish | Polski |
| `ar` | Arabic | العربية |
| `hi` | Hindi | हिन्दी |
| `th` | Thai | ไทย |
| `vi` | Vietnamese | Tiếng Việt |

### CLIでの使用

```bash
# 日本語と英語で検索（デフォルト）
deep-research research "市場分析" --multilingual

# 検索言語を指定
deep-research research "AI技術動向" \
    --multilingual \
    --search-languages ja,en,zh,ko

# 言語あたりの結果数を指定
deep-research research "競合分析" \
    --multilingual \
    --search-languages ja,en,de,fr \
    --results-per-language 15
```

### Pythonでの使用

```python
from deep_research_tool import run_research

# 多言語検索でリサーチ
result = run_research(
    query="再生可能エネルギー市場の最新動向",
    provider="anthropic",
    iterations=5,
    output_format="pdf",
    multilingual=True,                    # 多言語検索を有効化
    search_languages=["ja", "en", "zh"],  # 検索言語
    results_per_language=10,              # 言語あたりの結果数
    translate_results=True,               # 結果を出力言語に翻訳
)

print(f"レポート: {result['report_path']}")
```

### 詳細な設定（Config使用）

```python
from deep_research_tool import DeepResearchTool, create_config

config = create_config(
    provider="anthropic",
    research_iterations=5,
    output_format="docx",

    # 多言語検索設定
    multilingual=True,
    search_languages=["ja", "en", "zh", "ko"],
    results_per_language=10,
    query_translation="llm",     # "llm" または "none"
    translate_results=True,      # 結果を翻訳するか
)

tool = DeepResearchTool(config)
result = tool.run(
    query="グローバルAI規制の動向",
    requirements="各国の規制状況を比較すること",
)
```

### 設定パラメータ

| パラメータ | 説明 | デフォルト |
|-----------|------|----------|
| `multilingual` | 多言語検索の有効化 | `False` |
| `search_languages` | 検索言語リスト | `["ja", "en"]` |
| `results_per_language` | 言語あたりの結果数 | `10` |
| `query_translation` | クエリ翻訳方法（`llm`/`none`） | `"llm"` |
| `translate_results` | 結果を出力言語に翻訳 | `True` |
| `dedup_threshold` | 重複判定の閾値（0.0-1.0） | `0.85` |
| `max_concurrent_searches` | 同時検索数 | `3` |

### 多言語検索が有効なケース

| ケース | 説明 |
|-------|------|
| グローバルトピック | 複数国で異なる視点の情報が存在する場合 |
| 学術・技術調査 | 英語文献と母国語文献の両方が必要な場合 |
| 市場調査 | 各国市場の情報を横断的に収集したい場合 |
| 規制・法令調査 | 各国の法規制を比較分析したい場合 |
| 競合分析 | グローバル企業の各地域での動向を調査する場合 |

### 注意事項

- クエリ翻訳にはLLM APIを使用するため、追加のトークンを消費します
- 検索言語数に比例して検索時間が増加します（並列処理で軽減）
- 一部の言語では検索エンジンの対応が限定的な場合があります
- 翻訳精度は使用するLLMモデルに依存します

---

## 追加機能6：ローカルLLMサポート

### 概要

ローカルで動作するLLMサーバー（Ollama、vLLM等）を使用してリサーチを実行できます。API料金を気にせず、機密データを外部に送信せずに処理が可能です。

### 対応バックエンド・モデル

| バックエンド | 対応モデル | デフォルトポート |
|-------------|-----------|-----------------|
| **Ollama** | Llama 3.1/3.2/2, CodeLlama, Mistral, Mixtral, Phi, Qwen, Gemma | 11434 |
| **vLLM** | gpt-oss-20b, gpt-oss-120b, その他OpenAI互換モデル | 8000 |
| **OpenAI互換** | 任意のOpenAI API互換サーバー | 8000 |

### 使用方法

#### Ollama + Llama3.1

```python
from deep_research_tool import run_research

result = run_research(
    query="調査テーマ",
    provider="local",
    local_model="llama3.1:8b",
    local_backend="ollama",
    local_base_url="http://localhost:11434",
)
```

#### vLLM + gpt-oss-20b

```python
from deep_research_tool import run_research

result = run_research(
    query="調査テーマ",
    provider="local",
    local_model="gpt-oss-20b",
    local_backend="vllm",
    local_base_url="http://localhost:8000",
)
```

### ローカルデータのみで分析（Web検索なし）

```python
from pathlib import Path
from deep_research_tool.api import LocalLLMClient
from deep_research_tool.research.researcher import Researcher
from deep_research_tool.evidence.locker import EvidenceLocker, Evidence
from deep_research_tool.utils.document_reader import read_document

# LLMクライアント作成
llm = LocalLLMClient(
    model="llama3.1:8b",
    backend="ollama",
    base_url="http://localhost:11434",
)

# ローカルファイルをエビデンスとして読み込み
locker = EvidenceLocker()
for file_path in Path("./data").rglob("*.*"):
    try:
        content = read_document(str(file_path))
        locker.add_evidence(Evidence(
            url=f"file://{file_path.absolute()}",
            title=file_path.name,
            content_excerpt=content[:2000],
            full_content=content,
            source_type="local_file",
            relevance_score=1.0,
        ))
    except Exception:
        pass

# Web検索をスキップしてローカルデータのみで分析
researcher = Researcher(llm_client=llm, language="ja")
session = researcher.run_research(
    query="ローカルデータの分析",
    evidence_locker=locker,
    skip_search=True,
)
```

### LocalLLMClientの直接使用

```python
from deep_research_tool.api import LocalLLMClient

client = LocalLLMClient(
    model="llama3.1:8b",
    backend="ollama",
    base_url="http://localhost:11434",
)

# サーバー接続確認
if client.is_available():
    print("✓ 接続OK")

    # 利用可能なモデル一覧
    models = client.list_models()
    print(f"利用可能モデル: {models}")

    # テキスト生成
    response = client.generate("日本の経済状況を要約してください")
    print(response.content)
```

### 対応モデル一覧

```python
# Llama系（Ollama）
"llama3.1:8b", "llama3.1:70b", "llama3.2:3b", "llama2:7b", "codellama:7b"

# GPT-OSS系（vLLM）
"gpt-oss-20b", "gpt-oss-120b"

# その他
"mistral:7b", "mixtral:8x7b", "phi3:mini", "qwen2:7b", "gemma2:9b"
```

---

## 追加機能7：データサルベージと復旧

### 概要

接続切れやエラーで中断したリサーチセッションのデータを復旧するための機能です。`salvage.ipynb` ノートブックを使用して、メモリ上のデータやファイルからサルベージできます。

### 使用方法

```python
# salvage.ipynbを開いてセルを実行

# === 1. メモリ上のオブジェクト検索 ===
# カーネルが再起動されていなければ、セッションやエビデンスを復旧可能

# === 2. トークン使用量の確認 ===
from deep_research_tool.api.base import get_token_stats
stats = get_token_stats()
print(stats.get_summary("ja"))

# === 3. 報告書本文の抽出 ===
# サルベージしたセッションから報告書を抽出
extract_report_text(salvaged['sessions'][0], 'report.md')

# === 4. CSVエクスポート（日本語エンコーディング対応）===
# Excelで開く場合: utf-8-sig
# 古いWindowsアプリ: cp932
convert_json_to_csv("salvage_xxx/evidences.json", encoding="cp932")
```

### CSVエンコーディングオプション

| エンコーディング | 説明 | 用途 |
|----------------|------|------|
| `utf-8-sig` | BOM付きUTF-8 | Excel（推奨） |
| `utf-8` | UTF-8 | 汎用 |
| `cp932` | Windows日本語 | 古いWindowsアプリ |
| `shift_jis` | Shift_JIS | レガシーシステム |

### 報告書抽出機能

```python
# Markdown形式で保存
extract_report_text(session, 'report.md', format='markdown')

# テキスト形式（cp932エンコーディング）
extract_report_text(session, 'report.txt', encoding='cp932', format='txt')

# Word文書（python-docx必要）
extract_report_text(session, 'report.docx', format='docx')
```

---

## 追加機能8：高速クロールモード（Fast Crawl Mode）

### 概要

情報収集プロセスを高速化するための最適化モードです。従来の順次処理に対して、並列フェッチと効率的なLLM評価を組み合わせることで、リサーチ時間を大幅に短縮できます。

### 処理フロー比較

```
【標準モード（standard）】
検索 → ページ1取得 → LLM評価 → ページ2取得 → LLM評価 → ...（順次処理）

【高速バッチモード（fast_batch）】
検索 → [ページ1,2,3,4,5を並列取得] → [5ページをまとめてLLM評価] → ...
        ↑ 並列HTTP fetch            ↑ 1回のLLMコールで複数評価

【高速並列モード（fast_parallel）】
検索 → [ページ1,2,3,4,5を並列取得] → [LLM評価を並列実行] → ...
        ↑ 並列HTTP fetch            ↑ 複数のLLMコールを同時実行
```

### モードの選択指針

| モード | 特徴 | 推奨ケース |
|-------|------|-----------|
| `standard` | 順次処理、最も安定 | デフォルト、確実性重視 |
| `fast_batch` | 並列fetch + バッチLLM評価 | トークンコスト削減、中程度の高速化 |
| `fast_parallel` | 並列fetch + 並列LLM評価 | 最速、レート制限に余裕がある場合 |

### CLIでの使用

```bash
# 高速バッチモード（推奨）
deep-research research "市場分析" --crawl-mode fast_batch

# 高速並列モード（最速）
deep-research research "技術調査" --crawl-mode fast_parallel

# ワーカー数とバッチサイズをカスタマイズ
deep-research research "競合分析" \
    --crawl-mode fast_batch \
    --fast-crawl-workers 15 \
    --fast-crawl-batch-size 8
```

### Pythonでの使用

```python
from deep_research_tool import run_research

# 高速バッチモード
result = run_research(
    query="AIトレンド分析 2024",
    provider="anthropic",
    crawl_mode="fast_batch",      # fast_batch または fast_parallel
    fast_crawl_workers=10,        # 並列HTTPワーカー数
    fast_crawl_batch_size=5,      # バッチあたりのページ数
)

# 高速並列モード（最速）
result = run_research(
    query="市場動向調査",
    crawl_mode="fast_parallel",
    fast_crawl_workers=15,
)

print(f"レポート: {result['report_path']}")
```

### 詳細な設定（Config使用）

```python
from deep_research_tool import DeepResearchTool, create_config

config = create_config(
    provider="anthropic",
    research_iterations=5,
    output_format="pdf",

    # 高速クロールモード設定
    crawl_mode="fast_batch",     # standard / fast_batch / fast_parallel
    fast_crawl_workers=10,       # 並列フェッチのワーカー数
    fast_crawl_batch_size=5,     # バッチ評価時のページ数
)

tool = DeepResearchTool(config)
result = tool.run(
    query="再生可能エネルギー市場",
    requirements="最新データと予測を含める",
)
```

### 設定パラメータ

| パラメータ | 説明 | デフォルト |
|-----------|------|----------|
| `crawl_mode` | クロールモード（standard/fast_batch/fast_parallel） | `standard` |
| `fast_crawl_workers` | 並列HTTPフェッチのワーカー数 | `10` |
| `fast_crawl_batch_size` | バッチ評価時の1バッチあたりページ数 | `5` |

### パフォーマンス特性

| モード | 速度 | トークン効率 | 安定性 |
|-------|------|------------|--------|
| standard | ★★☆ | ★★★ | ★★★ |
| fast_batch | ★★★ | ★★★ | ★★★ |
| fast_parallel | ★★★★ | ★★☆ | ★★☆ |

### 注意事項

- `fast_parallel`モードはAPIのレート制限に注意が必要です
- `fast_batch`モードはトークン効率が良く、多くのケースで推奨されます
- ネットワークが不安定な環境では`standard`モードが安定します
- Extended Modeとの併用も可能です

### FastCrawlerの独立使用（情報収集のみ）

FastCrawlerはリサーチ全体のワークフローから独立して使用できます。Web検索と情報収集だけを行い、結果を自分で処理したい場合に便利です。

#### 基本的な使い方

```python
from deep_research_tool.research.fast_crawler import FastCrawler, EvaluationMode
from deep_research_tool.search import get_search_client
from deep_research_tool.api import get_client

# クライアントを作成
search_client = get_search_client(method="duckduckgo")
llm_client = get_client(provider="openai")

# FastCrawlerを作成
crawler = FastCrawler(
    search_client=search_client,
    llm_client=llm_client,
    evaluation_mode=EvaluationMode.BATCH,  # BATCH / PARALLEL / SEQUENTIAL
    max_workers=10,        # 並列フェッチのワーカー数
    batch_size=5,          # バッチ評価時のページ数
    language="ja",
)

# クロール実行
result = crawler.crawl_and_evaluate(
    queries=["日本のEV市場 2024", "電気自動車 販売台数"],
    section_context="日本のEV市場の現状",
    max_pages_per_query=5,
    min_relevance_score=0.3,
)

# 結果を処理
print(f"取得ページ数: {result.pages_fetched}")
print(f"フィルタ済み: {result.pages_filtered}")
print(f"関連ページ数: {len(result.pages)}")
print(f"フェッチ時間: {result.total_fetch_time:.1f}秒")
print(f"評価時間: {result.total_eval_time:.1f}秒")

# 各ページの情報を取得
for page in result.pages:
    print(f"\n--- {page.title} ---")
    print(f"URL: {page.url}")
    print(f"関連度: {page.relevance_score:.2f}")
    print(f"要点: {page.key_points}")
    print(f"要約: {page.processed_content[:200]}...")
```

#### ファクトリー関数を使った簡易作成

```python
from deep_research_tool.research.fast_crawler import create_fast_crawler
from deep_research_tool.search import get_search_client
from deep_research_tool.api import get_client

search_client = get_search_client(method="duckduckgo")
llm_client = get_client(provider="openai")

# ファクトリー関数で作成
crawler = create_fast_crawler(
    search_client=search_client,
    llm_client=llm_client,
    mode="batch",      # "batch" / "parallel" / "sequential"
    language="ja",
    max_workers=15,
    batch_size=8,
)

result = crawler.crawl_and_evaluate(
    queries=["量子コンピュータ 最新動向"],
    section_context="量子コンピュータの技術動向",
)
```

#### コンテンツフィルターのカスタマイズ

```python
from deep_research_tool.research.fast_crawler import FastCrawler, EvaluationMode
from deep_research_tool.evidence.content_filter import (
    ContentFilter,
    ContentFilterConfig,
    create_strict_filter,
)

# 厳格なフィルター（広告サイトを強力に排除）
strict_filter = create_strict_filter()

# カスタムフィルター
custom_filter = ContentFilter(ContentFilterConfig(
    min_content_length=500,
    max_ad_ratio=0.1,
    blocked_domains=["spam-site.com", "ad-network.net"],
    whitelisted_domains=["trusted-source.org"],
))

crawler = FastCrawler(
    search_client=search_client,
    llm_client=llm_client,
    evaluation_mode=EvaluationMode.BATCH,
    content_filter=custom_filter,  # カスタムフィルターを使用
)
```

#### プログレスコールバック

```python
def progress_callback(message: str, current: int, total: int):
    """進捗を表示するコールバック"""
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"[{percentage:5.1f}%] {message}")

result = crawler.crawl_and_evaluate(
    queries=["検索クエリ1", "検索クエリ2"],
    section_context="調査テーマ",
    progress_callback=progress_callback,
)
```

#### 大量クエリの一括処理

```python
# 複数のテーマについて一括でクロール
themes = [
    ("EV市場", ["日本 EV市場 2024", "電気自動車 普及率"]),
    ("自動運転", ["自動運転技術 最新", "レベル4 自動運転 日本"]),
    ("水素自動車", ["FCV 燃料電池車 トヨタ", "水素ステーション 整備"]),
]

all_results = {}
for theme_name, queries in themes:
    result = crawler.crawl_and_evaluate(
        queries=queries,
        section_context=theme_name,
        max_pages_per_query=3,
    )
    all_results[theme_name] = result
    print(f"{theme_name}: {len(result.pages)}ページ取得")

# 結果をCSVで保存
import csv
with open("crawl_results.csv", "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["テーマ", "URL", "タイトル", "関連度", "要約"])
    for theme, result in all_results.items():
        for page in result.pages:
            writer.writerow([
                theme,
                page.url,
                page.title,
                f"{page.relevance_score:.2f}",
                page.processed_content[:500],
            ])
```

#### CrawlResult / EvaluatedPage の構造

```python
# CrawlResult の属性
result.pages              # List[EvaluatedPage] - 関連性のあるページ一覧
result.total_fetch_time   # float - フェッチ総時間（秒）
result.total_eval_time    # float - 評価総時間（秒）
result.pages_fetched      # int - フェッチしたページ数
result.pages_filtered     # int - フィルタで除外されたページ数
result.pages_evaluated    # int - 評価したページ数
result.errors             # List[str] - エラーメッセージ一覧

# EvaluatedPage の属性
page.url                  # str - ページURL
page.title                # str - ページタイトル
page.snippet              # str - 検索結果のスニペット
page.content              # str - ページの全文コンテンツ
page.relevance_score      # float - 関連度スコア (0.0-1.0)
page.processed_content    # str - LLMが生成した要約
page.key_points           # List[str] - 抽出された要点
page.fetch_time           # float - フェッチ時間（秒）
page.evaluation_time      # float - 評価時間（秒）
page.metadata             # Dict - メタデータ（検索クエリ等）
```

### 検索エンジンの選択（DuckDuckGo / Selenium）

```python
from deep_research_tool.search import get_search_client

# DuckDuckGo（高速、APIベース）
search_duckduckgo = get_search_client(method="duckduckgo", max_results=10)

# Selenium（動的サイト対応、ブラウザベース）
search_selenium = get_search_client(
    method="selenium",
    headless=True,         # ヘッドレスモード
    browser="chrome",      # chrome / firefox
    max_results=10,
)

# FastCrawlerで使用
crawler = FastCrawler(
    search_client=search_selenium,  # Seleniumを使用
    llm_client=llm_client,
    evaluation_mode=EvaluationMode.BATCH,
)
```

### 検索結果数の調整

```python
# 検索クエリあたりの取得ページ数を調整
result = crawler.crawl_and_evaluate(
    queries=["AI市場動向 2024"],
    section_context="AI市場の概況",
    max_pages_per_query=10,     # デフォルト3、最大10程度推奨
    min_relevance_score=0.2,    # 関連度閾値（低いほど多く取得）
)
```

### サイト内深層クローリング（SiteCrawler）

SiteCrawlerを使うと、検索結果のページだけでなく、そのサイト内のリンクを辿って深層ページも収集できます。

```python
from deep_research_tool.research.site_crawler import SiteCrawler
from deep_research_tool.search import get_search_client
from deep_research_tool.api import get_client

# クライアント作成
search_client = get_search_client(method="selenium", headless=True)
llm_client = get_client(provider="openai")

# SiteCrawler作成
site_crawler = SiteCrawler(
    search_client=search_client,
    llm_client=llm_client,       # LLM使用でより正確な関連度評価
    max_pages=15,                # 1サイトあたり最大ページ数
    max_depth=2,                 # シードURLからの最大深度
    relevance_threshold=0.3,     # 関連度閾値
    delay_between_requests=0.5,  # リクエスト間隔（秒）
    language="ja",
)

# 単一サイトのクローリング
result = site_crawler.crawl_site(
    seed_url="https://example.com/ev-market",
    research_topic="日本のEV市場",
    keywords=["電気自動車", "EV", "市場規模", "普及率"],
    section_context="市場概況",
)

print(f"クロール済み: {result.pages_crawled}ページ")
print(f"関連ページ: {result.pages_relevant}ページ")
print(f"発見トピック: {result.discovered_topics}")
print(f"追加クエリ提案: {result.suggested_queries}")

# 取得したページを処理
for page in result.crawled_pages:
    print(f"\n[{page.depth}] {page.title}")
    print(f"  URL: {page.url}")
    print(f"  関連度: {page.relevance_score:.2f}")
    print(f"  リンク数: {len(page.links)}")
```

### FastCrawler + SiteCrawler の組み合わせ

検索 → 検索結果ページ取得 → サイト内深層クロール のワークフロー:

```python
from deep_research_tool.research.fast_crawler import FastCrawler, EvaluationMode
from deep_research_tool.research.site_crawler import SiteCrawler
from deep_research_tool.search import get_search_client
from deep_research_tool.api import get_client

# クライアント
search_client = get_search_client(method="selenium", headless=True)
llm_client = get_client(provider="openai")

# Step 1: FastCrawlerで検索結果を取得・評価
fast_crawler = FastCrawler(
    search_client=search_client,
    llm_client=llm_client,
    evaluation_mode=EvaluationMode.BATCH,
)

search_result = fast_crawler.crawl_and_evaluate(
    queries=["量子コンピュータ 技術動向 2024"],
    section_context="量子コンピュータの最新技術",
    max_pages_per_query=5,
)

print(f"検索結果: {len(search_result.pages)}件の関連ページ")

# Step 2: 高関連度のページについてサイト内深層クロール
site_crawler = SiteCrawler(
    search_client=search_client,
    llm_client=llm_client,
    max_pages=10,
    max_depth=2,
)

all_deep_pages = []
for page in search_result.pages:
    if page.relevance_score >= 0.6:  # 高関連度のみ深層クロール
        print(f"\n深層クロール開始: {page.url}")
        deep_result = site_crawler.crawl_site(
            seed_url=page.url,
            research_topic="量子コンピュータ",
            keywords=["量子", "qubit", "超電導", "誤り訂正"],
        )
        all_deep_pages.extend(deep_result.crawled_pages)
        print(f"  → {deep_result.pages_relevant}ページ取得")

print(f"\n合計: {len(all_deep_pages)}ページの深層情報を取得")
```

### 情報収集結果のエビデンスロッカーへの保存

```python
from deep_research_tool.evidence.locker import EvidenceLocker, EvidenceType

# エビデンスロッカー作成
locker = EvidenceLocker(output_dir="./output/evidence")

# FastCrawlerの結果を保存
for page in search_result.pages:
    locker.add_evidence(
        url=page.url,
        title=page.title,
        content_excerpt=page.processed_content or page.content[:500],
        evidence_type=EvidenceType.WEB_PAGE,
        search_query=page.metadata.get("query", ""),
        relevance_score=page.relevance_score,
    )

# SiteCrawlerの結果を保存
for page in all_deep_pages:
    locker.add_evidence(
        url=page.url,
        title=page.title,
        content_excerpt=page.content[:500],
        evidence_type=EvidenceType.WEB_PAGE,
        relevance_score=page.relevance_score,
    )

# エクスポート
locker.export_to_json()   # JSON形式
locker.export_to_csv()    # CSV形式（Excel対応）

print(f"保存済み: {len(locker.get_all_evidence())}件のエビデンス")
```

### 完全なワークフロー例

```python
"""
完全な情報収集ワークフロー:
1. 複数クエリで検索
2. 検索結果を並列フェッチ・評価
3. 高関連度ページについてサイト内深層クロール
4. 全結果をエビデンスロッカーに保存
5. CSVエクスポート
"""

from deep_research_tool.research.fast_crawler import FastCrawler, EvaluationMode
from deep_research_tool.research.site_crawler import SiteCrawler
from deep_research_tool.search import get_search_client
from deep_research_tool.api import get_client
from deep_research_tool.evidence.locker import EvidenceLocker, EvidenceType

def comprehensive_research(
    topic: str,
    queries: list,
    provider: str = "openai",
    search_method: str = "duckduckgo",
    deep_crawl: bool = True,
    output_dir: str = "./output",
):
    """包括的な情報収集を実行"""

    # クライアント初期化
    search = get_search_client(method=search_method, max_results=10)
    llm = get_client(provider=provider)
    locker = EvidenceLocker(output_dir=f"{output_dir}/evidence")

    # FastCrawler
    fast_crawler = FastCrawler(
        search_client=search,
        llm_client=llm,
        evaluation_mode=EvaluationMode.BATCH,
        max_workers=10,
    )

    print(f"=== 検索フェーズ: {len(queries)}クエリ ===")
    search_result = fast_crawler.crawl_and_evaluate(
        queries=queries,
        section_context=topic,
        max_pages_per_query=5,
        min_relevance_score=0.2,
    )

    # 検索結果を保存
    for page in search_result.pages:
        locker.add_evidence(
            url=page.url,
            title=page.title,
            content_excerpt=page.processed_content or page.content[:500],
            evidence_type=EvidenceType.WEB_PAGE,
            search_query=page.metadata.get("query", ""),
            relevance_score=page.relevance_score,
        )

    print(f"検索結果: {len(search_result.pages)}件")

    # 深層クロール
    if deep_crawl:
        site_crawler = SiteCrawler(
            search_client=search,
            llm_client=llm,
            max_pages=10,
            max_depth=2,
        )

        high_relevance = [p for p in search_result.pages if p.relevance_score >= 0.5]
        print(f"\n=== 深層クロールフェーズ: {len(high_relevance)}サイト ===")

        for page in high_relevance[:3]:  # 上位3サイトのみ
            try:
                deep_result = site_crawler.crawl_site(
                    seed_url=page.url,
                    research_topic=topic,
                )
                for dp in deep_result.crawled_pages:
                    locker.add_evidence(
                        url=dp.url,
                        title=dp.title,
                        content_excerpt=dp.content[:500],
                        evidence_type=EvidenceType.WEB_PAGE,
                        relevance_score=dp.relevance_score,
                    )
                print(f"  {page.url[:50]}... → {deep_result.pages_relevant}ページ")
            except Exception as e:
                print(f"  エラー: {e}")

    # エクスポート
    json_path = locker.export_to_json()
    csv_path = locker.export_to_csv()

    total = len(locker.get_all_evidence())
    print(f"\n=== 完了: {total}件のエビデンス ===")
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")

    return locker

# 使用例
locker = comprehensive_research(
    topic="日本のEV市場の現状",
    queries=[
        "日本 EV市場 2024 市場規模",
        "電気自動車 販売台数 推移",
        "EV 充電インフラ 整備状況",
    ],
    provider="openai",
    search_method="duckduckgo",  # または "selenium"
    deep_crawl=True,
)
```

---

## 詳細ワークフロー：クエリ入力からレポート出力まで

以下は、ユーザーがリサーチクエリを入力してからレポートが出力されるまでの詳細なワークフローです。

### 全体フロー図

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ユーザー入力                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ クエリ: "日本のEV市場の現状と将来展望"                               │   │
│  │ 要件: "最新のトレンドと主要企業を含める"                             │   │
│  │ オプション: --target-pages 10 --output-format pdf                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Phase 1: 調査計画作成                                │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ QueryGenerator.create_research_plan()                              │    │
│  │  ├─ クエリを分析                                                   │    │
│  │  ├─ 調査の範囲と目的を特定                                         │    │
│  │  ├─ 目次(Table of Contents)を生成                                  │    │
│  │  │   1. 日本のEV市場の概要                                         │    │
│  │  │   2. 市場規模と成長推移                                         │    │
│  │  │   3. 主要プレイヤーと競争環境                                   │    │
│  │  │   4. 政策・規制動向                                             │    │
│  │  │   5. 将来展望と課題                                             │    │
│  │  └─ 初期検索クエリを生成                                           │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Phase 2: リサーチループ（セクションごと）               │
│                                                                             │
│  ┌─ セクション 1: 日本のEV市場の概要 ─────────────────────────────────┐    │
│  │                                                                    │    │
│  │  【イテレーション 1】                                              │    │
│  │   ├─ 検索クエリ実行: "日本 EV市場 概要 2024"                       │    │
│  │   │   └─ DuckDuckGo/Selenium で Web検索                           │    │
│  │   ├─ 検索結果取得 (上位10件)                                       │    │
│  │   │   ├─ https://example.com/ev-market-japan...                   │    │
│  │   │   ├─ https://news.example.org/ev-trend...                     │    │
│  │   │   └─ ...                                                      │    │
│  │   ├─ ページコンテンツ取得                                          │    │
│  │   │   └─ HTML → テキスト変換、画像URL抽出                         │    │
│  │   ├─ 関連情報抽出 (ContentExtractor)                               │    │
│  │   │   ├─ LLMで関連性スコア算出                                    │    │
│  │   │   ├─ 重要な情報を抽出・要約                                   │    │
│  │   │   └─ Evidence Lockerに登録                                    │    │
│  │   └─ 情報ギャップ特定                                              │    │
│  │       └─ "充電インフラの情報が不足"                                │    │
│  │                                                                    │    │
│  │  【イテレーション 2】                                              │    │
│  │   ├─ フォローアップクエリ生成                                      │    │
│  │   │   └─ "日本 EV 充電インフラ 整備状況"                          │    │
│  │   ├─ 検索・抽出を繰り返し                                          │    │
│  │   └─ 情報ギャップ再評価                                            │    │
│  │                                                                    │    │
│  │  【イテレーション 3】 ... （min_iterations まで繰り返し）          │    │
│  │                                                                    │    │
│  │  【セクションコンテンツ統合】                                      │    │
│  │   └─ 収集した情報をLLMで統合・構造化                               │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─ セクション 2: 市場規模と成長推移 ─────────────────────────────────┐    │
│  │  ... (同様のプロセス)                                              │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌─ セクション 3〜5 ──────────────────────────────────────────────────┐    │
│  │  ... (各セクションで同様のプロセス)                                │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Phase 3: 出力量調整（オプション）                      │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ ContentLengthController.get_expansion_requirement()                │    │
│  │                                                                    │    │
│  │  現在の文字数: 15,000文字 (約6ページ)                              │    │
│  │  目標: 10ページ (約25,000文字)                                     │    │
│  │  判定: 拡張が必要 (目標の60%)                                      │    │
│  │                                                                    │    │
│  │  【コンテンツ拡張処理】                                            │    │
│  │   ├─ 拡張対象セクション選定                                        │    │
│  │   │   └─ 優先度: 低信頼度 > ギャップあり > 短いセクション          │    │
│  │   ├─ 追加リサーチ実行                                              │    │
│  │   │   ├─ より詳細なデータを検索                                   │    │
│  │   │   ├─ 専門家の分析を検索                                       │    │
│  │   │   └─ 具体的な事例を検索                                       │    │
│  │   └─ 既存コンテンツとマージ                                        │    │
│  │       └─ LLMで自然に統合、重複を排除                               │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Phase 4: 全体統合・要約作成                           │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Researcher._synthesize_findings()                                  │    │
│  │  ├─ 全セクションの要約を統合                                       │    │
│  │  ├─ エグゼクティブサマリー生成                                     │    │
│  │  ├─ 主要な発見事項をリスト化                                       │    │
│  │  ├─ 推奨事項を整理                                                 │    │
│  │  └─ 全体の信頼度を評価                                             │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Phase 5: ハルシネーション検証（オプション）             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ Verifier.verify_content()                                          │    │
│  │  ├─ レポート内の主張を抽出                                         │    │
│  │  ├─ 各主張をEvidence Lockerと照合                                  │    │
│  │  │   ├─ [VERIFIED] "EV販売台数は前年比30%増"                      │    │
│  │  │   │   └─ 出典: 経済産業省統計 (evidence_001)                   │    │
│  │  │   ├─ [UNVERIFIED] "2030年までに50%シェア"                      │    │
│  │  │   │   └─ エビデンスで確認できず                                │    │
│  │  │   └─ ...                                                       │    │
│  │  └─ 検証レポートHTML生成                                           │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Phase 6: レポート生成                                │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ ReportGenerator.generate_report()                                  │    │
│  │  ├─ Markdownフォーマットで構造化                                   │    │
│  │  │   ├─ タイトル・概要                                            │    │
│  │  │   ├─ 目次                                                      │    │
│  │  │   ├─ エグゼクティブサマリー                                    │    │
│  │  │   ├─ 各セクション本文                                          │    │
│  │  │   ├─ 結論・推奨事項                                            │    │
│  │  │   └─ 参考文献リスト                                            │    │
│  │  ├─ 出力形式に変換                                                 │    │
│  │  │   ├─ PDF: reportlabで生成                                      │    │
│  │  │   ├─ DOCX: python-docxで生成                                   │    │
│  │  │   ├─ HTML: テンプレートで生成                                  │    │
│  │  │   └─ Markdown: そのまま出力                                    │    │
│  │  └─ 目標ページ数に調整（必要に応じて縮小）                         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Phase 7: 図表追加（オプション・後処理）                 │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ FigureTableGenerator.generate_figures_and_tables()                 │    │
│  │  ├─ 各セクションを分析                                             │    │
│  │  ├─ 画像抽出                                                       │    │
│  │  │   ├─ Evidence Lockerから関連URL取得                            │    │
│  │  │   ├─ Webページから画像をダウンロード                           │    │
│  │  │   └─ 関連性を評価して選択                                      │    │
│  │  ├─ 表データ抽出                                                   │    │
│  │  │   ├─ テキストから数値データをパターンマッチ                    │    │
│  │  │   ├─ 時系列データを検出                                        │    │
│  │  │   └─ 表形式に整形                                              │    │
│  │  ├─ グラフ生成                                                     │    │
│  │  │   ├─ matplotlibで折れ線/棒グラフ作成                           │    │
│  │  │   └─ 日本語フォント対応                                        │    │
│  │  └─ レポートに挿入                                                 │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              出力ファイル                                   │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │ output/                                                            │    │
│  │  ├─ reports/                                                       │    │
│  │  │   └─ research_report_abc123.pdf    ← メインレポート            │    │
│  │  ├─ evidence/                                                      │    │
│  │  │   ├─ evidence_abc123.json          ← エビデンス詳細(JSON)      │    │
│  │  │   └─ evidence_abc123.csv           ← エビデンス一覧(CSV)       │    │
│  │  ├─ figures/                                                       │    │
│  │  │   ├─ chart_1_市場規模推移.png      ← 生成されたグラフ          │    │
│  │  │   └─ image_2_ev_lineup.jpg         ← 取得した画像              │    │
│  │  ├─ session_abc123.json               ← セッション情報            │    │
│  │  └─ verification_abc123.html          ← 検証レポート              │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 各フェーズの詳細

#### Phase 1: 調査計画作成

| 処理 | 説明 | 使用コンポーネント |
|-----|------|------------------|
| クエリ分析 | ユーザーのクエリと要件から調査の目的・範囲を特定 | `QueryGenerator` |
| 目次生成 | 調査トピックに適した章立てを自動生成 | LLM |
| 検索クエリ生成 | 各セクションに対応する初期検索クエリを作成 | LLM |

#### Phase 2: リサーチループ

各セクションに対して以下を繰り返し実行：

| ステップ | 処理内容 | 回数 |
|---------|---------|------|
| 検索実行 | 検索クエリでWeb検索を実行 | クエリ数 × 3 |
| コンテンツ取得 | 上位結果のページ内容を取得 | 結果数 × 3 |
| 情報抽出 | LLMで関連情報を抽出・評価 | ページごと |
| ギャップ分析 | 不足している情報を特定 | イテレーションごと |
| フォローアップ | 不足情報に対する追加クエリを生成 | ギャップごと |
| 統合 | セクション内の情報を統合 | セクションごと |

#### Phase 3: 出力量調整

目標ページ数/文字数が指定された場合の処理：

| 状態 | 処理 |
|-----|------|
| 目標の90%未満 | 追加リサーチを実行してコンテンツを拡充 |
| 目標の90-110% | 調整なし |
| 目標の110%超過 | 文境界を考慮してコンテンツを削減 |

**拡張時の優先セクション選定基準：**
1. 信頼度が「低」のセクション
2. 情報ギャップが多いセクション
3. 平均より短いセクション

#### Phase 4: 全体統合

| 処理 | 出力 |
|-----|------|
| エグゼクティブサマリー生成 | 300-500語の要約 |
| 主要発見事項リスト化 | 箇条書きリスト |
| 推奨事項整理 | 今後のアクション提案 |
| 信頼度評価 | high/medium/low |

#### Phase 5: ハルシネーション検証

| 検証レベル | 説明 |
|-----------|------|
| `low` | 明らかな矛盾のみチェック |
| `medium` | 数値・日付・固有名詞を重点的にチェック |
| `high` | すべての主張をエビデンスと照合 |

#### Phase 6: レポート生成

| 形式 | 特徴 |
|-----|------|
| Markdown | 軽量、バージョン管理に適する |
| PDF | 印刷・配布に適する、ページ制御可能 |
| DOCX | 編集可能、Microsoft Office互換 |
| HTML | ブラウザで閲覧、インタラクティブ |

#### Phase 7: 図表追加

| 図表タイプ | 生成条件 |
|-----------|---------|
| 参照画像 | エビデンスURLに関連画像が存在 |
| データ表 | 数値データ（年次推移、比較）が検出 |
| 折れ線グラフ | 3点以上の時系列データ |
| 棒グラフ | カテゴリ比較データ |

---

## 設定オプション一覧

| オプション | 説明 | デフォルト |
|-----------|------|----------|
| `provider` | LLMプロバイダー (openai/anthropic/local) | openai |
| `model` | 使用するモデル名 | gpt-4o-mini |
| `search_method` | 検索方法 (duckduckgo/selenium) | duckduckgo |
| `research_iterations` | リサーチループ回数 | 3 |
| `output_format` | 出力形式 (markdown/docx/pdf/html) | markdown |
| `enable_verification` | 検証機能の有効化 | True |
| `verification_strictness` | 検証の厳密さ (low/medium/high) | medium |
| `target_pages` | 目標ページ数 | None (無制限) |
| `target_characters` | 目標文字数 | None (無制限) |
| `extended_mode` | Extended Mode有効化 | False |
| `crawl_max_pages` | 1サイトあたり最大クローリングページ数 | 10 |
| `crawl_max_depth` | シードURLからの最大深度 | 2 |
| `crawl_max_sites` | クローリング対象の最大サイト数 | 3 |
| `http_proxy` | HTTPプロキシURL | None |
| `https_proxy` | HTTPSプロキシURL | None |
| `verify_ssl` | SSL証明書検証 | True |
| `multilingual` | 多言語検索の有効化 | False |
| `search_languages` | 検索言語リスト | ["ja", "en"] |
| `results_per_language` | 言語あたりの結果数 | 10 |
| `query_translation` | クエリ翻訳方法 (llm/none) | llm |
| `translate_results` | 結果を出力言語に翻訳 | True |
| `use_enhanced_synthesis` | Multi-Pass Synthesis有効化 | True |
| `local_model` | ローカルLLMモデル名 | llama3.1:8b |
| `local_backend` | ローカルLLMバックエンド (ollama/vllm/openai_compatible) | ollama |
| `local_base_url` | ローカルLLMサーバーURL | http://localhost:11434 |
| `crawl_mode` | 高速クロールモード (standard/fast_batch/fast_parallel) | standard |
| `fast_crawl_workers` | 並列HTTPフェッチのワーカー数 | 10 |
| `fast_crawl_batch_size` | バッチ評価時の1バッチあたりページ数 | 5 |

---

## プロキシ設定

企業ネットワークなどプロキシ経由でインターネットにアクセスする環境向けの設定です。

### 使用方法

#### 方法1: `run_research`で直接指定

```python
from deep_research_tool import run_research

result = run_research(
    query="市場分析レポート",
    provider="openai",
    api_key="sk-your-api-key",
    http_proxy="http://proxy.company.com:8080",
    https_proxy="http://proxy.company.com:8080",
    verify_ssl=True,  # 自己署名証明書の場合はFalse
)
```

#### 方法2: `create_config`で詳細設定

```python
from deep_research_tool import DeepResearchTool, create_config

config = create_config(
    provider="anthropic",
    anthropic_api_key="sk-ant-your-key",
    http_proxy="http://proxy.company.com:8080",
    https_proxy="http://proxy.company.com:8080",
    proxy_username="user",      # 認証が必要な場合
    proxy_password="password",
    verify_ssl=False,           # 自己署名証明書の場合
)

tool = DeepResearchTool(config)
result = tool.run(query="AI動向調査")
```

#### 方法3: 環境変数（自動読み込み）

```python
import os

# 環境変数を設定
os.environ["HTTP_PROXY"] = "http://proxy.company.com:8080"
os.environ["HTTPS_PROXY"] = "http://proxy.company.com:8080"

# 環境変数から自動的に読み込まれる
from deep_research_tool import run_research

result = run_research(
    query="市場分析",
    api_key="sk-your-key",
)
```

### プロキシ設定パラメータ

| パラメータ | 説明 | 例 |
|-----------|------|---|
| `http_proxy` | HTTPプロキシURL | `http://proxy:8080` |
| `https_proxy` | HTTPSプロキシURL | `http://proxy:8080` |
| `proxy_username` | プロキシ認証ユーザー名 | `user123` |
| `proxy_password` | プロキシ認証パスワード | `pass456` |
| `verify_ssl` | SSL証明書検証 | `True` / `False` |

### 注意事項

- `verify_ssl=False`は自己署名証明書を使用するプロキシ環境でのみ使用してください
- 認証情報（`proxy_username`, `proxy_password`）はコードに直接書かず、環境変数や設定ファイルから読み込むことを推奨します
- プロキシ設定はLLM API（OpenAI/Anthropic）とWeb検索（DuckDuckGo）の両方に適用されます

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
│   ├── anthropic_client.py
│   └── local_client.py  # ローカルLLM (Ollama/vLLM)
├── search/              # Web検索モジュール
│   ├── base.py
│   ├── duckduckgo.py
│   └── selenium_browser.py
├── research/            # リサーチコアロジック
│   ├── query_generator.py
│   ├── content_extractor.py
│   ├── researcher.py
│   ├── site_crawler.py  # Extended Mode用サイトクローラー
│   └── fast_crawler.py  # 高速クロールモード用並列クローラー
├── verification/        # ハルシネーション検証
│   └── verifier.py
├── evidence/            # Evidence Locker
│   ├── locker.py
│   └── quality_evaluator.py
├── report/              # レポート生成
│   ├── generator.py
│   ├── length_controller.py
│   └── figure_table_generator.py
├── utils/               # ユーティリティ
│   ├── document_reader.py
│   └── helpers.py
└── salvage.ipynb        # データサルベージ用ノートブック
```

---

## ライセンス

MIT License

## 貢献

プルリクエストや課題報告を歓迎します。
