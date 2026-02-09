# Deep Research Tool - Jupyter Notebook使用例

このドキュメントでは、Jupyter Notebook/Labでの使用方法を説明します。

## セットアップ

```python
# 必要なパッケージをインストール
!pip install -e /path/to/deep_research_tool

# または、パスを追加
import sys
sys.path.insert(0, '/path/to/deep_research_tool')
```

## 基本的な使用方法

### 1. 設定の作成

```python
from deep_research_tool import DeepResearchTool
from deep_research_tool.config import create_config
import os

# 環境変数を設定（または.envファイルを使用）
os.environ["OPENAI_API_KEY"] = "your-api-key"
# または
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"

# 設定を作成
config = create_config(
    provider="openai",  # または "anthropic"
    model="gpt-4o-mini",  # オプション
    search_method="duckduckgo",  # または "selenium"
    research_iterations=3,  # リサーチループ回数
    output_format="markdown",  # "docx", "pdf", "html"
    output_dir="./research_output",
)
```

### 2. リサーチの実行

```python
# ツールインスタンスを作成
tool = DeepResearchTool(config)

# 進捗表示用コールバック
def progress(message, percentage):
    if percentage >= 0:
        print(f"[{percentage:5.1f}%] {message}")

# リサーチを実行
result = tool.run(
    query="日本のAI産業の現状と将来展望",
    requirements="最新のトレンドと主要企業の動向を含めること",
    progress_callback=progress,
)

print(f"レポート: {result['report_path']}")
print(f"エビデンス: {result['evidence_json']}")
```

### 3. 追加文書を含むリサーチ

```python
# PDFや既存レポートを参照として追加
config = create_config(
    provider="anthropic",
    anthropic_api_key="your-key",
    research_iterations=5,
    output_format="docx",
    additional_documents=[
        "reference_paper.pdf",
        "previous_report.docx",
    ],
)

tool = DeepResearchTool(config)
result = tool.run(
    query="再生可能エネルギー市場分析",
    requirements="前回レポートとの比較を含める",
)
```

### 4. クイックリサーチ（簡易調査）

```python
# フルレポート生成なしの簡易調査
result = tool.quick_research(
    query="量子コンピューティングの最新動向",
    max_results=5,
)

print("概要:")
print(result['summary'])
print("\nソース:")
for source in result['sources']:
    print(f"  - {source['title']}: {source['url']}")
```

## 高度な使用方法

### Selenium（動的サイト対応）の使用

```python
config = create_config(
    provider="openai",
    search_method="selenium",
    headless=True,  # ヘッドレスモード
)

tool = DeepResearchTool(config)
# 動的なJavaScriptレンダリングが必要なサイトも処理可能
```

### 検証機能のカスタマイズ

```python
from deep_research_tool.verification import Verifier

# 検証の厳密さを設定
config = create_config(
    enable_verification=True,
    verification_strictness="high",  # "low", "medium", "high"
)

# 検証結果の取得
result = tool.run(query="...")
verification = result.get('verification_result')

if verification:
    print(f"信頼性スコア: {verification.overall_reliability_score:.1%}")
    print(f"ハルシネーションリスク: {verification.hallucination_risk_count}件")
```

### Evidence Lockerの直接使用

```python
from deep_research_tool.evidence import EvidenceLocker

# 既存のエビデンスファイルを読み込み
locker = EvidenceLocker.load_from_json("./output/evidence_xyz.json")

# 統計情報を取得
stats = locker.get_statistics()
print(f"総エビデンス数: {stats['total_evidence']}")
print(f"ユニークドメイン数: {stats['unique_domains']}")

# 参考文献リストを生成
locker.export_bibliography(style="apa")
```

### レポート形式の変更

```python
from deep_research_tool.report import ReportGenerator, ReportFormat
from deep_research_tool.research.researcher import ResearchSession

# 既存セッションからレポートを再生成
session = ResearchSession.load("./output/session_xyz.json")
locker = EvidenceLocker.load_from_json("./output/evidence_xyz.json")

generator = ReportGenerator(output_dir="./new_reports")

# 異なる形式で出力
for format in [ReportFormat.MARKDOWN, ReportFormat.DOCX, ReportFormat.PDF]:
    path = generator.generate_report(session, locker, format=format)
    print(f"Generated: {path}")
```

## トラブルシューティング

### APIキーエラー

```python
# 環境変数が設定されているか確認
import os
print("OpenAI:", "Set" if os.getenv("OPENAI_API_KEY") else "Not set")
print("Anthropic:", "Set" if os.getenv("ANTHROPIC_API_KEY") else "Not set")
```

### Selenium関連エラー

```python
# ChromeDriverが正しくインストールされているか確認
from webdriver_manager.chrome import ChromeDriverManager
driver_path = ChromeDriverManager().install()
print(f"ChromeDriver: {driver_path}")
```

### メモリ不足

```python
# 大規模リサーチの場合、イテレーション数を制限
config = create_config(
    research_iterations=2,  # 最小限に設定
    max_results=5,  # 検索結果数を制限
)
```
