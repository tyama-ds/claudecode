# open_deep_research vs deep_research_tool 比較分析レポート

**作成日**: 2026-03-03
**参照**: [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research)

---

## 1. アーキテクチャ全体比較

| 観点 | open_deep_research | deep_research_tool |
|------|--------------------|--------------------|
| **フレームワーク** | LangGraph（状態グラフ） | 独自実装（Pythonクラスベース） |
| **設計パターン** | Supervisor-Worker マルチエージェント | 単一オーケストレーター型 |
| **ソースコード規模** | ~5ファイル（約2,000行） | ~83ファイル（約37,000行） |
| **LLMプロバイダー** | LangChainの`init_chat_model`で統一 | 独自クライアント（OpenAI, Anthropic, Ollama/vLLM） |
| **構造化出力** | Pydanticモデル + tool_calling | JSON形式のプロンプト指示 |
| **状態管理** | LangGraph State（MessagesState拡張） | 独自ResearchSession + EvidenceLocker |
| **MCP対応** | あり（MCPサーバー統合） | なし |
| **デプロイ** | LangGraph Platform / Open Agent Platform対応 | CLI / GUI ローカル実行 |

---

## 2. 検索機構の詳細比較

### 2.1 open_deep_research の検索機構

**アーキテクチャ: Supervisor → Researcher(s) → Search**

```
ユーザークエリ
  ↓
clarify_with_user() → 必要なら質問を返す
  ↓
write_research_brief() → 研究課題を構造化
  ↓
supervisor() ←──────────────────────┐
  ├─ think_tool: 戦略的思考        │
  ├─ ConductResearch: 調査委譲     │
  │   ↓                            │
  │   researcher() [最大5並行]      │
  │   ├─ tavily_search (2-5回)     │
  │   ├─ MCP tools                 │
  │   └─ think_tool                │
  │   ↓                            │
  │   compress_research()          │
  │   ↓                            │
  │   結果をsupervisorに返す ──────┘
  └─ ResearchComplete: 完了宣言
```

**検索API**: Tavily（デフォルト）、Anthropic/OpenAI内蔵検索、MCP servers

**クエリ生成**: Supervisorが`ConductResearch`で自然言語のトピックを委譲。自動分割・精緻化はない。

**結果処理**: GPT-4.1-miniで25-30%に要約 → 「notes」としてSupervisorに返す。

**特徴的な点**:
- `think_tool`: 「何もしない」ツール。LLMに「立ち止まって考える」機会を与える
- 最大5つのResearcherが並行実行可能（LangGraphの`Send` API）
- トークン制限のプロバイダー別エラーハンドリング

### 2.2 deep_research_tool の検索機構

**アーキテクチャ: Orchestrator → QueryGenerator → Search → ContentExtractor**

```
ユーザークエリ
  ↓
QueryGenerator.create_research_plan()
  ├─ 目次(ToC)構造を自動生成
  ├─ 25+個の検索クエリを生成
  ├─ 複雑クエリを自動分割（split_complex_queries）
  └─ ToC品質バリデーション
  ↓
FOR EACH セクション in ToC:
  ├─ 検索 → コンテンツ抽出 → ギャップ特定 → フォローアップ
  └─ セクション内容合成
```

**検索API**: DuckDuckGo（メイン、無料）、Selenium、13言語並列検索、SiteCrawler、FastCrawler

**クエリ生成**: LLMが25+個を一括生成。One-Query-One-Topic原則（40文字以内）。自動簡略化3段階。

**結果処理**: チャンク分割（6000文字、500文字オーバーラップ）→ バッチ/並列/逐次の3モード関連性評価。

### 2.3 検索機構の主要差分

| 観点 | open_deep_research | deep_research_tool |
|------|--------------------|--------------------|
| **検索API** | Tavily（有料）、MCP | DuckDuckGo（無料）、Selenium、多言語 |
| **クエリ生成** | Supervisorが自然言語で委譲 | LLMが25+個を一括生成 + 自動分割 |
| **並行性** | 最大5 Researcher並行 | ThreadPoolExecutorでページ並行 |
| **関連性評価** | なし（Supervisor判断） | バッチ/並列/逐次の3モード |
| **ファイル対応** | Webページのみ | PDF/DOCX/XLSX/PPTX/CSV |
| **多言語** | なし | 13言語並列検索 |
| **Think Tool** | あり | なし（→ V2で追加） |

---

## 3. レポート作成機構の詳細比較

### 3.1 open_deep_research のレポート作成

**単一LLMコール方式**: 全Researcherの圧縮ノートを結合 → 1回のLLMコール → Markdown出力

品質管理: **なし**（トークン制限時10%縮小リトライのみ）

### 3.2 deep_research_tool のレポート作成

**3世代ジェネレーター + 品質管理パイプライン**:

- V2: 章ごと順次生成（コンテキスト継承）→ 4層一貫性チェック → Two-Phase修正
- V3: python-docx直接生成 → 図表/グラフ自動挿入 → プロフェッショナル文書出力

品質管理: 用語統一、文体統一、矛盾検出、重複検出、事実追跡、ソース品質評価

### 3.3 レポート作成機構の主要差分

| 観点 | open_deep_research | deep_research_tool |
|------|--------------------|--------------------|
| **生成方式** | 単一LLMコール | 章ごと順次生成（コンテキスト継承） |
| **品質管理** | なし | 用語/文体/矛盾/重複の4層チェック |
| **グラフ・図表** | なし | ChartAnalyzer + matplotlib（7種） |
| **出力形式** | Markdownのみ | DOCX, MD, PDF, HTML |
| **引用管理** | `[Title](URL)`インライン | EvidenceLocker（品質評価、BibTeX出力） |

---

## 4. ベンチマーク

### open_deep_research
- **Deep Research Bench** #6（RACE スコア: 0.4344）
- GPT-5使用時: 0.4943（最高）、Claude Sonnet 4: 0.4401、GPT-4.1: 0.4309

### deep_research_tool
- 公開ベンチマーク結果なし
- 内蔵のファクトチェック・幻覚検出・一貫性検査が品質担保

---

## 5. 結論

### open_deep_researchの優位点（deep_research_toolが取り込んだ点）
1. **Think Tool**: → `research/v2/reflector.py` として実装
2. **ユーザー事前確認**: → `research/v2/clarifier.py` として実装
3. **並行調査**: → `research/v2/async_orchestrator.py` として実装

### deep_research_toolの優位点（維持すべき点）
1. 検索の深さと多様性（多言語、サイト深掘り、多ファイル形式）
2. レポート品質管理（4層チェック、Two-Phase、用語集）
3. データ可視化（ChartAnalyzer、7種のグラフ）
4. 出力の多様性（DOCX/MD/PDF/HTML）
5. ソース管理（EvidenceLocker、完全な監査証跡）
