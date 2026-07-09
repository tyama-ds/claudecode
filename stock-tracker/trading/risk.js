/* ==========================================================================
   risk.js — リスク管理(発注前の安全チェック)
   毎回の発注前に全チェックを通す。1つでも不合格なら発注しない。
   状態(当日の発注回数・実現損益・停止フラグ)は state.json に永続化。
   ========================================================================== */

const fs = require("fs");
const path = require("path");

const STATE_FILE = path.join(__dirname, "state.json");
const HALT_FILE = path.join(__dirname, "HALT"); // このファイルを置くと全発注停止(キルスイッチ)

function today() {
  // JSTの日付(UTC+9)
  const d = new Date(Date.now() + 9 * 3600 * 1000);
  return d.toISOString().slice(0, 10);
}

function loadState() {
  try {
    const s = JSON.parse(fs.readFileSync(STATE_FILE, "utf8"));
    if (s.day !== today()) return freshState();  // 日付が変わったらリセット
    return s;
  } catch { return freshState(); }
}
function freshState() { return { day: today(), ordersToday: 0, realizedPnLToday: 0, halted: false }; }
function saveState(s) { fs.writeFileSync(STATE_FILE, JSON.stringify(s, null, 2)); }

class RiskManager {
  constructor(cfg = {}) {
    this.limits = {
      maxPositionValueYen: cfg.maxPositionValueYen ?? 500_000,
      maxOrderQty: cfg.maxOrderQty ?? 100,
      maxOrdersPerDay: cfg.maxOrdersPerDay ?? 10,
      maxDailyLossYen: cfg.maxDailyLossYen ?? 30_000,
    };
    this.state = loadState();
  }

  // キルスイッチ: HALTファイルの存在、または当日停止フラグ
  isHalted() {
    return fs.existsSync(HALT_FILE) || this.state.halted;
  }

  // 発注可否チェック。{ok:true} か {ok:false, reason} を返す。
  check({ side, qty, price }) {
    if (this.isHalted()) return { ok: false, reason: "停止中(HALTファイルまたは当日損失上限超過)" };
    if (this.state.ordersToday >= this.limits.maxOrdersPerDay)
      return { ok: false, reason: `当日発注上限(${this.limits.maxOrdersPerDay}回)に到達` };
    if (qty > this.limits.maxOrderQty)
      return { ok: false, reason: `1回の発注数量上限(${this.limits.maxOrderQty}株)を超過` };
    if (side === "buy" && price != null && qty * price > this.limits.maxPositionValueYen)
      return { ok: false, reason: `建玉評価額上限(¥${this.limits.maxPositionValueYen.toLocaleString()})を超過` };
    if (this.state.realizedPnLToday <= -this.limits.maxDailyLossYen)
      return { ok: false, reason: `当日損失上限(¥${this.limits.maxDailyLossYen.toLocaleString()})に到達` };
    return { ok: true };
  }

  // 発注が成立したら記録
  recordOrder() {
    this.state.ordersToday += 1;
    saveState(this.state);
  }

  // 決済損益を記録。損失上限に達したら当日停止。
  recordPnL(pnl) {
    this.state.realizedPnLToday += pnl;
    if (this.state.realizedPnLToday <= -this.limits.maxDailyLossYen) {
      this.state.halted = true;
    }
    saveState(this.state);
  }

  summary() {
    return { ...this.state, limits: this.limits, halted: this.isHalted() };
  }
}

module.exports = { RiskManager, HALT_FILE, STATE_FILE };
