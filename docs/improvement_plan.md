# deep_research_tool 改善計画
# open_deep_research との比較に基づく抜本的見直し

**作成日**: 2026-03-03
**実装日**: 2026-03-03
**対象バージョン**: deep_research_tool 現行版 → research/v2 パッケージとして実装
**参照**: [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research)
**状態**: 実装済み

---

## 概要

open_deep_research（LangChain公式）との比較分析により、3つの改善を `research/v2/` パッケージとして実装した。
`report/v2/`, `report/v3/` と同じパターンで、既存V1コードは変更せずに新バージョンとして追加。

### 改善の優先順位と実装状態

| 優先度 | 改善項目 | 実装ファイル | 状態 |
|--------|----------|-------------|------|
| **P1** | Think Tool（戦略的思考ステップ） | `research/v2/reflector.py` | 実装済み |
| **P2** | ユーザー事前確認フロー | `research/v2/clarifier.py` | 実装済み |
| **P3** | セクション並行調査（asyncio） | `research/v2/async_orchestrator.py` | 実装済み |
| **統合** | ResearcherV2 (V1を継承) | `research/v2/researcher.py` | 実装済み |
| **--** | Tavily検索バックエンド | -- | オーナー判断により除外 |
| **--** | MCP統合 | -- | 将来対応 |
| **--** | LangGraph移行 | -- | 非推奨 |

---

## 実装内容

### ディレクトリ構成

```
deep_research_tool/research/
├── __init__.py              # V1 exports (変更なし)
├── researcher.py            # V1 Researcher (変更なし)
├── query_generator.py       # (変更なし)
├── content_extractor.py     # (変更なし)
├── site_crawler.py          # (変更なし)
├── fast_crawler.py          # (変更なし)
├── manual_researcher.py     # (変更なし)
└── v2/
    ├── __init__.py           # V2 exports
    ├── reflector.py          # ResearchReflector (Think Tool)
    ├── clarifier.py          # ResearchClarifier (事前確認)
    ├── async_orchestrator.py # AsyncResearchOrchestrator (並行調査)
    └── researcher.py         # ResearcherV2 (V1を継承)
```

### P1: Think Tool (ResearchReflector)

**ファイル**: `research/v2/reflector.py`
**クラス**: `ResearchReflector`

open_deep_researchの`think_tool`を参考に、検索ループ内にメタ認知的思考ステップを導入。

**仕組み**:
- `reflect_on_section()`: セクション調査の中間振り返り（イテレーション2以降に発火）
- `reflect_on_overall()`: 全セクション完了後の全体振り返り
- LLMに構造化JSONで回答させ、`ReflectionResult`にパース

**判断ロジック**:
```
coverage_score >= 0.8 かつ quality_score >= 0.7 → stop_research: true
見落としが重大 → should_pivot: true + recommended_queries で方向修正
```

**V1との違い**:
- V1の`identify_gaps`は「不足情報の検出」に特化
- Think Toolは「研究全体の戦略的評価」（網羅性、品質、方向性の三軸）

### P2: ユーザー事前確認フロー (ResearchClarifier)

**ファイル**: `research/v2/clarifier.py`
**クラス**: `ResearchClarifier`

**仕組み**:
- `analyze_query()`: クエリの曖昧さを5観点（スコープ、時間軸、地域、技術レベル、深さ）で分析
- `merge_clarification()`: ユーザーの回答を要件に統合
- `main.py`の`run()`冒頭で呼び出し、`v2_enable_clarification=True`時のみ発火

### P3: セクション並行調査 (AsyncResearchOrchestrator)

**ファイル**: `research/v2/async_orchestrator.py`
**クラス**: `AsyncResearchOrchestrator`

**仕組み**:
- `analyze_dependencies()`: セクション間の親子関係を分析し、並行実行可能なグループに分割
- `process_sections_parallel()`: asyncio.to_threadで同期APIをラップし、semaphoreで並行数制御
- 同レベルのセクション（1.1, 1.2, 1.3）は並行可能、親子関係は順次

### ResearcherV2 (統合クラス)

**ファイル**: `research/v2/researcher.py`
**クラス**: `ResearcherV2(Researcher)`

V1のResearcherを継承し、以下をオーバーライド:
- `_conduct_research_loop()`: 並行/順次の分岐、全体振り返り追加
- `_process_section_v2()`: Think Tool統合版のセクション処理
- `_execute_search_iteration()`: V1のインナーループを抽出・再利用

---

## 設定方法

### Config経由

```python
from deep_research_tool import create_config

config = create_config(
    # ... 既存設定 ...
    researcher_version="v2",
    researcher_v2_enable_think_tool=True,
    researcher_v2_think_tool_start_iteration=2,
    researcher_v2_enable_parallel=False,
    researcher_v2_max_concurrent_sections=3,
    researcher_v2_enable_clarification=False,
)
```

### 直接使用

```python
from deep_research_tool.research.v2 import ResearcherV2

researcher = ResearcherV2(
    llm_client=llm,
    search_client=search,
    enable_think_tool=True,
    enable_parallel=True,
    max_concurrent_sections=3,
)
session = researcher.conduct_research("AI market analysis 2025")
```

---

## 変更されたファイル

| ファイル | 変更内容 |
|---------|---------|
| `config.py` | `ResearcherVersion` enum追加、`ResearchConfig` にV2設定追加、`create_config()` にパラメータ追加 |
| `main.py` | `ResearcherV2`/`ResearchClarifier` import、`_create_researcher()` ヘルパー追加、`run()` に確認フロー追加 |

---

## 非推奨とした変更

1. **Tavily検索バックエンド**: オーナー判断により除外（DuckDuckGoで十分）
2. **LangGraph全面移行**: 37,000行の書き換えリスク。asyncio+Strategyパターンで代替
3. **レポート生成機構の変更**: open_deep_researchの単一LLMコール方式は現行より劣る
