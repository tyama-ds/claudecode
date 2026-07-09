# 🤖 自動売買バックエンド(auカブコム kabuステーションAPI)

StockLensの戦略シグナルを使って**実際に発注する**Node製バックエンドです。
静的アプリ本体とは完全に分離されており、これを起動しない限り一切発注しません。

> ⚠️ **本物の資金が動きます。** 必ず「戦略・売買シグナル」タブのバックテスト／
> ペーパートレードで戦略を検証し、まずは **検証環境** と **少額** で試してください。
> 本コードは雛形であり、発注パラメータや約定挙動はご自身の口座で必ず確認を。

## 二重ロック(安全設計)

実弾発注には**次の両方**が必要です。片方でも欠ければ自動的にドライラン(発注なし)になります。

1. 設定ファイルで `"dryRun": false`
2. 環境変数 `KABU_LIVE=1`

さらにキルスイッチとして、`trading/HALT` というファイルを作ると即座に全発注を停止します。

## 前提

- auカブコム証券の口座 + **kabuステーション**アプリが起動し、API利用が有効(APIモードON)
- ローカルにAPIが立つ:本番 `localhost:18080` / 検証 `localhost:18081`
- Node.js 18以上(グローバル`fetch`使用)

## セットアップ

```bash
cd stock-tracker
cp trading/config.example.json trading/config.json   # 設定を編集

# 秘密情報は環境変数で(config.jsonには書かない)
export KABU_API_PASSWORD="＜kabuステーションのAPIパスワード＞"
export KABU_TRADE_PASSWORD="＜取引パスワード＞"
```

## 実行

```bash
# 1) まず模擬ブローカーでロジック確認(発注なし・kabuステーション不要)
node trading/run.js --broker mock --once

# 2) 実口座に接続してドライラン(現在値取得＋シグナル判定のみ・発注なし)
#    config.json: "broker":"kabus", "env":"verification", "dryRun":true
node trading/run.js --once

# 3) 検証環境でごく少額ライブ(二重ロック)
#    config.json: "dryRun": false
KABU_LIVE=1 node trading/run.js

# 緊急停止(別ターミナルで)
touch trading/HALT
```

## 設定(config.json)

| キー | 説明 |
|---|---|
| `broker` | `"mock"`(模擬) / `"kabus"`(実API) |
| `env` | `"verification"`(検証:18081) / `"production"`(本番:18080) |
| `dryRun` | `true`で発注しない(既定)。ライブは`false`＋`KABU_LIVE=1` |
| `pollSeconds` | ポーリング間隔(秒)。`--once`で1回のみ |
| `strategy` / `params` | 使う戦略(`maCross`/`rsiReversal`)とパラメータ。`js/strategy.js`と共通 |
| `symbols` | 対象銘柄 `{code, exchange, qty, accountType}`(exchange 1=東証, accountType 4=特定) |
| `risk` | `maxPositionValueYen`/`maxOrderQty`/`maxOrdersPerDay`/`maxDailyLossYen` |

## リスク管理(毎発注前にチェック)

- 1回の発注数量上限 / 建玉評価額上限 / 当日発注回数上限 / 当日損失上限
- 当日損失上限に達すると自動停止(`state.json`に記録)
- `HALT`ファイルによるキルスイッチ
- 同一銘柄・同日の二重発注防止

## 構成

```
trading/
├── run.js              # 起動スクリプト(CLI)
├── engine.js           # 戦略判定→リスクチェック→発注/ドライラン
├── risk.js             # リスク管理・キルスイッチ・当日状態
├── brokers/
│   ├── kabus.js        # kabuステーションAPIアダプタ(実発注)
│   └── mock.js         # 模擬ブローカー(テスト用・実弾なし)
├── config.example.json # 設定サンプル(config.jsonにコピーして使う)
├── config.json         # 実設定(gitignore)
├── state.json          # 当日の発注回数・損益・停止状態(gitignore)
├── HALT                # 存在すると全発注停止(gitignore)
└── logs/               # 実行ログ(gitignore)
```

## 免責

本コードは学習・検証用の雛形です。売買の結果生じる損失について作者は責任を負いません。
自己資金の自動売買は原則自由ですが、証券会社の利用規約と税務(申告分離課税等)をご確認ください。
他人の資金の運用・有償の投資助言には金融商品取引業の登録が必要です。
