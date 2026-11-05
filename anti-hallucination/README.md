# Anti-hallucination

過去レポートと独立した正解データを入力し、モデルごとの数値穴埋め、同一候補へのファクトチェック、チェック前後の結果を比較するローカルアプリです。Python 3.11以降の標準ライブラリだけで動きます。


数値実験のブラウザー画面に加え、引用・対象条件・情報保持を検査して局所修正を選別するCLIを含みます。ブラウザー画面の実験と、文書修正用CLIは別の入口です。

この公開版にはソース、架空サンプル、合成テストを収録しています。過去のDeep Research原文、会話URL、スクリーンショット、実験結果、認証情報は収録していません。

## 起動

リポジトリのルートから `cd anti-hallucination` で移動して実行します。外部パッケージのインストールは不要です。

```powershell
python app.py serve
```

[http://127.0.0.1:8788](http://127.0.0.1:8788) を開いてください。`launch.ps1` でも起動できます。ポート変更は `python app.py serve --port 8789`。

画面には最初から架空の2文書・6項目と、2種類の模擬モデルが読み込まれます。「実験を実行」でAPIキーなしに一連の動作を確認できます。**デモの結果は実際のAIの性能を表していません。** デモ処理は共通資料の明示行を読み、正解台帳にはアクセスしません。

## 実データで回す

1. `samples/reports.json` を参考に、レポート本文と根拠資料をJSONにします。
2. `samples/gold.json` を参考に、正解値・単位・許容誤差・過去の候補を用意します。
3. `samples/models.live.example.json` をコピーして、使うモデルだけ残し、APIで利用可能な実際のモデルID・接続先を設定します。
4. APIキーの環境変数を設定した同じターミナルでサーバーを起動します。
5. 画面で3つのJSONを読み込み、「入力を検証」してから「実験を実行」を押します。

初回はレポート1〜2件から数値を5〜10件選び、2モデル・`repeats: 1` で入出力を確認すると始めやすくなります。これは接続と採点を確認する予備実験の規模で、精度を保証する件数ではありません。穴埋めの呼び出し数は「モデル数×項目数×反復回数」、チェックの呼び出し数は「モデル数×重複除去後の候補数×反復回数」です。チェックには正しい対照値や過去の値も含むため、穴埋めより呼び出しが多くなる場合があります。

キーはJSONやブラウザーには入力しません。例えばPowerShellでは、履歴にキーを書かずに設定できます。

```powershell
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new('', (Read-Host 'OpenAI API key' -AsSecureString)).Password
python app.py serve
```

他サービスも設定の `api_key_env` と同名の環境変数を使います。実API実行時は、数値を伏せたレポート本文、共通資料、検証する候補値が設定先APIへ送信されます。正解台帳そのものや、候補の生成元・正誤ラベルは送信しません。正しい値を含む対照候補は、正解と明かさずチェック対象に含めます。

## 入力1：レポートと根拠資料

```json
{
  "reports": [
    {
      "id": "report-001",
      "title": "A社調査（説明用の架空例）",
      "as_of": "2024-06-30",
      "text": "A社の2024年3月期の連結売上高は118.0億円だった。",
      "sources": [
        {
          "id": "source-001",
          "title": "決算資料（説明用の架空例）",
          "text": "2024年3月期の連結売上高は120.5億円。",
          "published_at": "2024-05-01"
        }
      ]
    }
  ]
}
```

資料本文は実際の出典から転記し、必要な表・注記も含めます。URLを設定しても、この初期版はリンクを取得せず `text` を使います。PDF/DOCXの直接読込やWeb検索は含みません。

対象年度は文章に残します。`as_of` は評価の基準日、`published_at` は根拠資料の公表日です。日付は `YYYY-MM-DD` とし、`published_at` が `as_of` より後の資料は入力エラーになります。`published_at` の省略は許可しますが、その資料の時点は利用者が確認してください。基準日後の改訂値と当時の値を混ぜず、同じ資料を全モデルに渡してください。現在の情報を使う実験では `as_of` を今回の基準日にし、過去時点の実験とは分けて集計します。

## 入力2：正解データ

```json
{
  "items": [
    {
      "id": "number-001",
      "report_id": "report-001",
      "placeholder": "{{number-001}}",
      "mask_text": "118.0",
      "statement": "A社の2024年3月期の連結売上高は {{value}} 億円だった。",
      "value": "120.5",
      "unit": "億円",
      "tolerance": "0.05",
      "accepted_units": {"百万円": "0.01"},
      "source_ids": ["source-001"],
      "candidates": [
        {"value": "118.0", "unit": "億円", "origin": "historical", "historical_verdict": "supported"}
      ]
    }
  ]
}
```

- `mask_text` は**原レポートに書かれた数字の表記**です。符号・小数・3桁ごとのカンマ区切りを指定でき、単位は含めません。本文中でその表記に完全一致する数値トークンをすべて `placeholder` に置換します。`1,205` と `1205` は別の表記です。根拠資料は置換しません。
- 文書中の同じ表記が別の項目にも使われる場合は、対象箇所を人が `{{number-001}}` などに置換し、`mask_text` を省いてください。数値項目の選定・抽出は人が行う初期版です。
- 既に `placeholder` にした箇所と、未置換の `mask_text` が同じ文書に共存する場合も入力エラーにします。再掲や表記違いも含め、どの数値を伏せるか確認してから手動でマスクしてください。
- `statement` は検証する主張です。数値箇所を `{{value}}` にし、年度・対象・条件を明示します。正解値や当時の判定をこの文に書かないでください。
- `value` は根拠で確認した正解です。過去のAI判定を無条件に正解にしません。
- 数値はJSON文字列とし、`value` にカンマや単位を混ぜません。数値比較には `Decimal` を使います。
- `tolerance` は**正解の単位での絶対許容誤差**です。例では ±0.05億円。厳密一致は `"0"`。許容誤差と単位換算規則は答えとは別の公開比較ルールとして全モデルに渡し、判定と採点の基準を揃えます。
- `accepted_units` は代替単位から正解単位への倍率です。12050百万円 × 0.01 = 120.5億円。％とポイントは同じ単位として登録しないでください。
- `source_ids` は引用元として認める資料IDです。
- `candidates` は過去の値などの追加候補です。`historical_verdict` は保存・確認用メタデータであり、正解ラベルには使いません。

この架空例の `historical_verdict: "supported"` は、過去のAIが誤った118.0億円を支持したという設定です。今回の正解120.5億円とは区別しています。候補がなければ `candidates: []` とできます。

入力JSONはファイル選択か画面内のエディターで編集できます。正解はブラウザー画面・ローカル保存・ZIPに含まれますが、穴埋めモデルの入力には入りません。

## 入力3：モデル設定

```json
{
  "models": [
    {
      "id": "model-a",
      "provider": "openai_responses",
      "model": "YOUR_AVAILABLE_MODEL_ID",
      "api_key_env": "OPENAI_API_KEY",
      "base_url": "https://api.openai.com/v1",
      "timeout_seconds": 90,
      "max_output_tokens": 2400,
      "input_cost_per_million": null,
      "output_cost_per_million": null
    }
  ],
  "repeats": 1
}
```

`models` に比較対象を追加します。同じAPIで異なるモデルIDを使うこともできます。

| provider | 接続方式 |
|---|---|
| `openai_responses` | OpenAI Responses API、JSON Schema形式の応答 |
| `openai_chat` | OpenAI互換 Chat Completions、JSON Schema形式の応答 |
| `anthropic` | Anthropic Messages、構造化出力 |
| `demo` | ローカルの模擬モデル。`model` は `careful` または `noisy` |

使用するAPI・モデルが構造化出力をサポートする必要があります。未対応の場合はエラーとして記録し、別モデルや非構造化出力に黙って切り替えません。実際に利用可能なモデル・単価は契約先で確認してください。

単価を設定すると、入力/出力トークン数からUSDの概算費用を計算します。未設定は「未設定」で、0円とは扱いません。キャッシュ割引・検索料金・プロバイダー固有の課金には未対応です。実API呼び出しの自動再試行はしません。失敗も母数に残します。

## 実験の動き

1. 全モデル×全項目×反復回数で穴埋めを実行。
2. 正しい対照値、過去の候補、各モデルが生成した数値から、項目・数値・単位で重複を除いた共通候補集合を作成。
3. 全チェックモデルが同じ候補集合を同じ根拠で判定。生成とは独立したAPIリクエストを使用。
4. 数値の採点、判定の採点、生成×判定の全組合せの集計を保存。

正解・過去判定・モデル名はチェック用の候補には含めません。初期版は修正を行わず、「支持できる」とされた回答だけを通した場合の効果を計算します。停止すると現在の1呼び出しが終わった後に止まり、未実行件数を残した部分結果を保存します。

## 結果の読み方

| 画面 | 指標 |
|---|---|
| 穴埋め | 全出題数を分母に正答・誤答・保留・実行失敗（途中停止は未実行も表示） |
| ファクトチェック | 誤り検出率、正答の誤指摘率、保留率、失敗率、通過率、通過結果中の誤り率 |
| チェックの効果 | 元の全出題数に対する正しい回答の提供率、誤った回答の提供率、未提供率 |
| 回答・根拠・ログ | モデル、問題、回答、引用、採点、エラーの詳細 |

通過が0件のとき、通過結果中の誤り率は算出不能（`null` / `—`）です。「保留」は誤りを検出した成功には数えません。

**引用文字列一致は、引用が指定資料に存在するかの機械検査です。意味的に主張を支持しているかの自動採点ではありません。** 正しい引用・計算式・年度や対象の読み違いは、詳細を見て別途人が確認してください。正解データにある問題の文脈と数値・単位を固定した採点であり、文書全体の事実性は評価しません。

共通候補集合には正しい対照候補を加えるため、自然な誤答率とは異なります。チェック性能の比較と、実際の穴埋め出力に対する「チェックの効果」を分けて読んでください。

旧モデルを再実行できるなら同条件の比較ができます。過去の自由生成結果しかない場合は、現在の穴埋め結果との違いをモデル単独の改善とは解釈しません。過去候補と判定は保存されますが、入力の条件差を補正した歴史的比較は初期版に含みません。

初期版では信頼区間・検定・データ分割の自動化を行いません。モデルの順位を確定する前に、開発に使っていないレポートを用意し、文書単位のばらつきも評価してください。

## 保存とCLI

結果は `runs/<実行ID>/` に保存されます。画面の「結果一式を保存」で、入力（正解含む）・生ログ・CSV・集計JSONをZIPにできます。APIキーは保存しません。ローカルの結果フォルダーはGitの追跡対象から除外しています。

```powershell
python app.py run --reports samples/reports.json --gold samples/gold.json --models samples/models.demo.json
```

入力検証のみ：

```powershell
python app.py run --reports samples/reports.json --gold samples/gold.json --models samples/models.demo.json --validate-only
```

実データ用の `data/` と `models.local.json` もGitの追跡対象から除外しています。

## 動作確認

```powershell
python -m unittest discover -s tests -v
```

テストでは単位換算・許容誤差、正解を渡さない入力構築、API応答の検証、失敗・保留・取消の集計、比較実験とHTTP画面の接続を確認します。実APIの品質・可用性の検証には、利用者のAPI設定での実行が必要です。

API実装の参照： [OpenAI Text generation](https://developers.openai.com/api/docs/guides/text)、[Anthropic Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)。


## 文書修正を検査するCLI

`claim_guard_v7.py` は、候補・主張台帳・独立したレビューを受け取り、許可された修正だけを適用します。

- 主張ごとの数量・指標・対象・時点・比較・断定・引用を記録し、原文の全ブロックの割当を検査します。
- 正しい情報の保持、必要な条件・引用の追加、引用番号と本文の境界、修正間の依存関係を確認します。
- 確認不能な事項を特定の修正だけに限定でき、残りと文書全体との相互作用を確認できたとき、その修正と依存する修正を保留します。残りの全文を別担当が再確認してから採用します。
- `unknown` を誤りや成功へ読み替えません。追加確認は最大1回で、候補の書き直し・再有効化は行いません。

`ledger_consistency.py` は、必要な取得済み根拠を持たない `supported` を `unknown` に下げ、必要なら主張全体も `uncertain` に下げます。原文・根拠・候補を変更せず、元台帳と変更ログを保存します。意味的な正しさを新たに保証する処理ではありません。

```powershell
python -X utf8 claim_guard_v7.py describe --output guard_contract.json
python -X utf8 claim_guard_v7.py blocks --original original_report.json --output block_packet.json
python -X utf8 ledger_consistency.py normalize --original original_report.json --candidate candidate.raw.json --ledger ledger.raw.json --output-dir normalized
python -X utf8 claim_guard_v7.py prepare --original original_report.json --candidate candidate.raw.json --ledger normalized/ledger.normalized.json --output-dir guard_input
python -X utf8 claim_guard_v7.py --help
```

`describe` が出力する契約で入力・応答JSONの構造を確認してください。`preview` → 初回全文確認 → 必要なら `recover` → 別担当の再確認 → `apply` の順に進めます。各サブコマンドの引数は `--help` で確認できます。

**修正用CLIはLLMやWeb検索を呼び出しません。** 候補、台帳、レビューを作成するモデル呼出・検索は別途必要です。数値実験の `app.py` はモデル設定に応じてAPIを呼び出します。ブラウザーだけでDeep Research生成から修正まで自動実行するアプリではありません。

## 構成

| パス | 内容 |
|---|---|
| `app.py`, `web/` | 数値穴埋め・同一候補の判定・採点を比較するローカル画面 |
| `factcheck/` | 入力検証、API接続、採点、局所修正と採否検査 |
| `claim_guard*.py`, `ledger_consistency.py` | 修正候補・台帳・保持レビューの検証CLI |
| `pilot.py`, `*_report.py`, `replication*.py`, `paired*.py`, `*_trial_ops.py` | 保存応答の取込、比較試験の検証、HTML集計 |
| `samples/` | 架空入力、模擬モデル設定、実API設定の雛形 |
| `tests/` | 合成データを使う単体・結合テスト |

比較試験CLIの既定の `pilot/...` フォルダーや過去の保存応答は同梱していません。試験の設定と入力を別途用意し、対応する `--root` / `--output` を指定してください。保存記録の再集計はLLMの再実行ではありません。

過去の実データを必要とする回帰テスト1件は、データがなければ明示的にスキップします。APIのテストは模擬応答を使い、実モデルの精度や実APIへの接続成功を保証しません。

## 共通候補で比較するときの担当モデル向け指示

`instructions/` は同じ候補集合をA/Bで比較する実験担当者向けの雛形です。実験結果や実データは含みません。

| 指示 | 担当 |
|---|---|
| `repair_v1.txt`, `repair_v2.txt` | 同一の指示で候補・主張台帳を作る |
| `guard.txt` | 修正単位の必要性・情報保持を確認する |
| `final_review.txt`, `final_rereview.txt` | 選別後の全文をA/Bの符号で確認する |
| `reference.txt` | 原文のみから独立した問題台帳を作る |
| `post.txt` | 共通の問題台帳を使ってA/Bの修正後を採点する |

`contracts/v7.json` / `v6.json` は各ゲートの契約、`ledger_normalization.json` は台帳整合化の契約です。`shared_candidate_contract.json` は候補・主張台帳・修正単位レビュー用の旧共通契約で、v7の全文確認契約ではありません。

全文確認指示は `{packet_binding_sha256, reviews}` のA/B共同応答を求めます。`uncertainty_trial_ops.py` のパケット・選択ID・ハッシュへの結び付けが必要であり、応答をそのまま単体の `claim_guard_v7.py apply` に渡すことはできません。`post.txt` は両群の候補と発見記録が共通である比較用です。単体CLIの応答構造は `describe` で確認してください。
