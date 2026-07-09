/* ==========================================================================
   engine.js — 自動売買エンジン(戦略判定 → リスクチェック → 発注/ドライラン)

   1周期の流れ(各銘柄):
     1. 履歴の日次終値を読み込む(research/jquants_data.json 優先、なければ
        data/stock-data.js)
     2. ブローカーから現在値を取得し、当日バーとして追加
     3. strategy.js でシグナル生成、最新バーのシグナルを判定
     4. 現在の建玉を確認し、行動(買い/売り/何もしない)を決定
     5. リスク管理を通過したら、ドライランならログのみ、ライブなら発注

   安全既定: dryRun=true。ライブ発注には config.dryRun=false かつ
   環境変数 KABU_LIVE=1 の両方が必要(二重ロック)。
   ========================================================================== */

const fs = require("fs");
const path = require("path");
const Strategy = require("../js/strategy.js");
const { RiskManager } = require("./risk.js");

const ROOT = path.join(__dirname, "..");
const LOG_DIR = path.join(__dirname, "logs");

function jstNow() {
  return new Date(Date.now() + 9 * 3600 * 1000).toISOString().replace("T", " ").slice(0, 19) + " JST";
}
function todayIso() {
  return new Date(Date.now() + 9 * 3600 * 1000).toISOString().slice(0, 10);
}

function log(msg, obj) {
  const line = `[${jstNow()}] ${msg}` + (obj ? " " + JSON.stringify(obj) : "");
  console.log(line);
  try {
    fs.mkdirSync(LOG_DIR, { recursive: true });
    fs.appendFileSync(path.join(LOG_DIR, `engine-${todayIso()}.log`), line + "\n");
  } catch { /* ignore */ }
}

/* ---------- 履歴データ読み込み ---------- */
function loadHistory() {
  const jqPath = path.join(ROOT, "research", "jquants_data.json");
  const out = {};
  if (fs.existsSync(jqPath)) {
    const jq = JSON.parse(fs.readFileSync(jqPath, "utf8"));
    for (const [code, s] of Object.entries(jq.symbols || {})) {
      if (s.prices) out[code] = s.prices.map(p => ({ date: p.date, close: p.close }));
    }
  }
  // 不足分は data/stock-data.js から補完(参考値)
  const dataPath = path.join(ROOT, "data", "stock-data.js");
  if (fs.existsSync(dataPath)) {
    const src = fs.readFileSync(dataPath, "utf8");
    const data = JSON.parse(src.slice(src.indexOf("{"), src.lastIndexOf(";")));
    for (const [code, s] of Object.entries(data.symbols || {})) {
      if (!out[code] && s.prices) out[code] = s.prices.map(p => ({ date: p.date, close: p.close }));
    }
  }
  return out;
}

class Engine {
  constructor(config, broker) {
    this.config = config;
    this.broker = broker;
    this.risk = new RiskManager(config.risk || {});
    this.history = loadHistory();
    this.live = config.dryRun === false && process.env.KABU_LIVE === "1";
    this.actedToday = {}; // 同一銘柄・同日の二重発注防止(約定反映前の連続ポーリング対策)
  }

  async start() {
    log(`エンジン起動: broker=${this.broker.name()} strategy=${this.config.strategy} ` +
      `mode=${this.live ? "★ライブ発注★" : "ドライラン(発注なし)"}`);
    if (this.config.dryRun === false && !this.live) {
      log("⚠️ config.dryRun=false ですが 環境変数 KABU_LIVE=1 が未設定のため、安全のためドライランで動作します。");
    }
    await this.broker.connect();
    log("ブローカー接続OK", this.risk.summary());
  }

  async tick() {
    if (this.risk.isHalted()) { log("⛔ 停止中(キルスイッチ/損失上限)。発注を行いません。"); return; }
    for (const sym of this.config.symbols) {
      try { await this.evaluateSymbol(sym); }
      catch (e) { log(`銘柄 ${sym.code} の処理でエラー: ${e.message}`); }
    }
  }

  async evaluateSymbol(sym) {
    const hist = this.history[sym.code];
    if (!hist || hist.length < 80) { log(`${sym.code}: 履歴データ不足のためスキップ`); return; }

    const { price } = await this.broker.getPrice(sym.code, sym.exchange);
    if (price == null) { log(`${sym.code}: 現在値が取得できずスキップ`); return; }

    // 当日バーを追加(同日が既にあれば置換)して最新シグナルを判定
    const series = hist.slice();
    const today = todayIso();
    if (series[series.length - 1].date === today) series[series.length - 1] = { date: today, close: price };
    else series.push({ date: today, close: price });

    const signals = Strategy.generateSignals(series, this.config.strategy, this.config.params);
    const todaySignal = signals[signals.length - 1].signal;

    const pos = await this.broker.getPosition(sym.code);
    let action = null;
    if (todaySignal === "buy" && pos.qty === 0) action = "buy";
    else if (todaySignal === "sell" && pos.qty > 0) action = "sell";

    if (!action) {
      log(`${sym.code}: シグナル=${todaySignal || "なし"} 建玉=${pos.qty}株 → 行動なし(価格 ${price})`);
      return;
    }

    const akey = `${sym.code}:${today}`;
    if (this.live && this.actedToday[akey]) {
      log(`${sym.code}: 本日は既に${this.actedToday[akey]}を発注済み → 二重発注防止でスキップ`);
      return;
    }

    const qty = action === "sell" ? pos.qty : sym.qty;
    const rc = this.risk.check({ side: action, qty, price });
    if (!rc.ok) { log(`${sym.code}: ${action} をリスク管理でブロック → ${rc.reason}`); return; }

    if (!this.live) {
      log(`🟡 [ドライラン] ${sym.code} ${action} ${qty}株 @成行(現在値 ${price}) を発注する想定(実発注なし)`);
      return;
    }

    // ★ライブ発注★
    log(`🔴 [ライブ] ${sym.code} ${action} ${qty}株 を成行発注します`);
    const r = await this.broker.sendOrder({ code: sym.code, exchange: sym.exchange, side: action, qty, accountType: sym.accountType });
    this.actedToday[akey] = action;
    this.risk.recordOrder();
    log(`約定依頼を送信: orderId=${r.orderId}`, { code: sym.code, action, qty });
    if (action === "sell" && pos.avgPrice) {
      const pnl = (price - pos.avgPrice) * qty;
      this.risk.recordPnL(pnl);
      log(`概算実現損益 ¥${Math.round(pnl).toLocaleString()}(当日累計 ¥${Math.round(this.risk.state.realizedPnLToday).toLocaleString()})`);
    }
  }

  async run() {
    await this.start();
    await this.tick();
    const poll = this.config.pollSeconds || 0;
    if (poll > 0 && !this.config.once) {
      log(`${poll}秒間隔でポーリングします(Ctrl+Cで停止、HALTファイル設置で発注停止)`);
      this._timer = setInterval(() => this.tick().catch(e => log("tickエラー: " + e.message)), poll * 1000);
    } else {
      log("1回のみ実行して終了(--once または pollSeconds=0)");
    }
  }
}

module.exports = { Engine, loadHistory };
