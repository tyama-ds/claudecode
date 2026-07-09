#!/usr/bin/env node
/* ==========================================================================
   run.js — 自動売買エンジンの起動スクリプト

   使い方:
     node trading/run.js                       # config.json を使用(既定ドライラン)
     node trading/run.js --config myconf.json  # 設定ファイル指定
     node trading/run.js --once                # 1回だけ評価して終了
     node trading/run.js --broker mock         # 模擬ブローカーで動作確認

   ライブ発注(実弾)を行うには、以下の【両方】が必要(二重ロック):
     1) 設定ファイルで "dryRun": false
     2) 環境変数 KABU_LIVE=1
   さらに KABU_API_PASSWORD / KABU_TRADE_PASSWORD の設定が必要。

   キルスイッチ: trading/HALT というファイルを作ると即座に全発注を停止。
   ========================================================================== */

const fs = require("fs");
const path = require("path");
const { Engine, loadHistory } = require("./engine.js");
const KabusBroker = require("./brokers/kabus.js");
const MockBroker = require("./brokers/mock.js");

function parseArgs(argv) {
  const args = { config: path.join(__dirname, "config.json") };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--once") args.once = true;
    else if (a === "--config") args.config = argv[++i];
    else if (a === "--broker") args.broker = argv[++i];
  }
  return args;
}

function loadConfig(file) {
  if (!fs.existsSync(file)) {
    console.error(`設定ファイルがありません: ${file}`);
    console.error("trading/config.example.json をコピーして config.json を作成してください。");
    process.exit(1);
  }
  return JSON.parse(fs.readFileSync(file, "utf8"));
}

function buildBroker(config, override) {
  const kind = override || config.broker || "mock";
  if (kind === "kabus") return new KabusBroker({ env: config.env });
  if (kind === "mock") {
    // 履歴の最新終値を模擬現在値として渡す
    const hist = loadHistory();
    const priceMap = {};
    for (const [code, series] of Object.entries(hist)) {
      if (series.length) priceMap[code] = series[series.length - 1].close;
    }
    return new MockBroker({ priceMap });
  }
  throw new Error("不明なbroker: " + kind);
}

async function main() {
  const args = parseArgs(process.argv);
  const config = loadConfig(args.config);
  if (args.once) config.once = true;
  const broker = buildBroker(config, args.broker);
  const engine = new Engine(config, broker);
  await engine.run();
}

main().catch(e => { console.error("致命的エラー:", e.message); process.exit(1); });
