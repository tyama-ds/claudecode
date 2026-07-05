#!/usr/bin/env node
/* ==========================================================================
   generate-data.js — 収集データ(research/*.json)から data/stock-data.js を生成

   リサーチで収集した実在の「月次株価アンカー」「イベント」「財務指標」を元に、
   アンカー点を通る日次終値・出来高系列を決定論的に合成する。
   (日次の細かい値は補間+ノイズによる参考値。アンカー・イベント・財務は実データ)

   usage: node scripts/generate-data.js
   ========================================================================== */

const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const RESEARCH = path.join(ROOT, "research");
const OUT = path.join(ROOT, "data", "stock-data.js");

/* ---------- 銘柄メタデータ ---------- */
const SYMBOL_META = {
  "7203": { group: "jp", market: "JP", marketName: "東証プライム", currency: "JPY", sector: "自動車", aliases: ["トヨタ", "TOYOTA", "toyota"], baseVolume: 28000000 },
  "6758": { group: "jp", market: "JP", marketName: "東証プライム", currency: "JPY", sector: "電気機器・エンタメ", aliases: ["ソニー", "SONY", "sony"], baseVolume: 12000000 },
  "5401": { group: "jp", market: "JP", marketName: "東証プライム", currency: "JPY", sector: "鉄鋼", aliases: ["日本製鉄", "日鉄", "新日鉄"], baseVolume: 9000000 },
  "8306": { group: "jp", market: "JP", marketName: "東証プライム", currency: "JPY", sector: "銀行", aliases: ["三菱UFJ", "MUFG", "三菱ＵＦＪ"], baseVolume: 45000000 },
  "8035": { group: "jp", market: "JP", marketName: "東証プライム", currency: "JPY", sector: "半導体製造装置", aliases: ["東京エレクトロン", "東エレク", "TEL"], baseVolume: 4000000 },
  "9983": { group: "jp", market: "JP", marketName: "東証プライム", currency: "JPY", sector: "小売(アパレル)", aliases: ["ファーストリテイリング", "ファストリ", "ユニクロ"], baseVolume: 900000 },
  "1321": { group: "etf", market: "JP", marketName: "東証ETF", currency: "JPY", type: "etf", index: "日経平均株価", aliases: ["日経225ETF", "日経ETF"], baseVolume: 600000 },
  "1306": { group: "etf", market: "JP", marketName: "東証ETF", currency: "JPY", type: "etf", index: "TOPIX", aliases: ["TOPIX ETF", "トピックスETF"], baseVolume: 1200000 },
  "SPY": { group: "etf", market: "US", marketName: "NYSE Arca", currency: "USD", type: "etf", index: "S&P 500", aliases: ["S&P500 ETF", "SPDR"], baseVolume: 70000000 },
  "QQQ": { group: "etf", market: "US", marketName: "NASDAQ", currency: "USD", type: "etf", index: "NASDAQ-100", aliases: ["ナスダック100 ETF", "インベスコQQQ"], baseVolume: 45000000 },
  "AAPL": { group: "us", market: "US", marketName: "NASDAQ", currency: "USD", sector: "テクノロジー", aliases: ["アップル", "Apple", "apple"], baseVolume: 55000000 },
  "NVDA": { group: "us", market: "US", marketName: "NASDAQ", currency: "USD", sector: "半導体", aliases: ["エヌビディア", "NVIDIA", "nvidia"], baseVolume: 250000000 },
  "MSFT": { group: "us", market: "US", marketName: "NASDAQ", currency: "USD", sector: "テクノロジー", aliases: ["マイクロソフト", "Microsoft"], baseVolume: 22000000 },
  "GOOGL": { group: "us", market: "US", marketName: "NASDAQ", currency: "USD", sector: "テクノロジー", aliases: ["グーグル", "アルファベット", "Google", "Alphabet"], baseVolume: 30000000 },
  "AMZN": { group: "us", market: "US", marketName: "NASDAQ", currency: "USD", sector: "Eコマース・クラウド", aliases: ["アマゾン", "Amazon", "amazon"], baseVolume: 40000000 },
  "TSLA": { group: "us", market: "US", marketName: "NASDAQ", currency: "USD", sector: "EV・自動車", aliases: ["テスラ", "Tesla", "tesla"], baseVolume: 90000000 },
};

const DATA_START = "2025-07-01";
const DATA_END = "2026-07-03"; // 直近の金曜

/* ---------- 決定論的PRNG (mulberry32) ---------- */
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function seedFrom(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}

/* ---------- 営業日リスト(土日除外) ---------- */
function tradingDays(start, end) {
  const days = [];
  const d = new Date(start + "T00:00:00Z");
  const e = new Date(end + "T00:00:00Z");
  while (d <= e) {
    const dow = d.getUTCDay();
    if (dow !== 0 && dow !== 6) days.push(d.toISOString().slice(0, 10));
    d.setUTCDate(d.getUTCDate() + 1);
  }
  return days;
}

/* ---------- アンカー補間 + ノイズで日次系列を合成 ---------- */
function synthesizeSeries(code, anchors, events, marketEvents, meta) {
  const days = tradingDays(DATA_START, DATA_END);
  const rand = mulberry32(seedFrom(code));

  const sorted = [...anchors].filter(a => a && a.close != null).sort((a, b) => a.date.localeCompare(b.date));
  if (sorted.length < 2) throw new Error(`${code}: 株価アンカーが不足しています (${sorted.length}点)`);

  const t = iso => new Date(iso + "T00:00:00Z").getTime();

  // 基準パス: アンカー間を対数空間で線形補間
  function basePath(iso) {
    const x = t(iso);
    for (let i = 0; i < sorted.length - 1; i++) {
      const a = sorted[i], b = sorted[i + 1];
      if (x >= t(a.date) && x <= t(b.date)) {
        const r = (x - t(a.date)) / Math.max(1, t(b.date) - t(a.date));
        return Math.exp(Math.log(a.close) * (1 - r) + Math.log(b.close) * r);
      }
    }
    return x < t(sorted[0].date) ? sorted[0].close : sorted[sorted.length - 1].close;
  }

  // イベント日 → ジャンプ率のマップ(土日のイベントは翌営業日に反映)
  const jumps = new Map();
  const snapToTrading = iso => {
    const d = new Date(iso + "T00:00:00Z");
    while ([0, 6].includes(d.getUTCDay())) d.setUTCDate(d.getUTCDate() + 1);
    return d.toISOString().slice(0, 10);
  };
  const addJump = (date, pct) => {
    pct = Math.max(-6, Math.min(6, pct)); // 1日のジャンプは±6%に制限
    const day = snapToTrading(date);
    jumps.set(day, (jumps.get(day) || 0) + pct);
  };
  for (const ev of events || []) {
    const mag = parseMagnitude(ev.magnitude);
    const pct = mag != null ? mag : (ev.impact === "up" ? 1.5 + rand() * 2 : ev.impact === "down" ? -(1.5 + rand() * 2.5) : 0);
    addJump(ev.date, pct);
  }
  for (const ev of marketEvents || []) {
    if (ev.market && ev.market !== meta.market && !["GLOBAL", "BOTH"].includes(ev.market)) continue;
    const mag = parseMagnitude(ev.magnitude);
    const base = mag != null ? mag * 0.7 : (ev.impact === "up" ? 1.0 : ev.impact === "down" ? -1.3 : 0);
    addJump(ev.date, base * (0.7 + rand() * 0.6));
  }

  // 日次ゆらぎ(平均回帰付きランダムウォーク)を基準パスに重ねる
  const dailyVol = meta.group === "etf" ? 0.006 : meta.market === "US" && (code === "NVDA" || code === "TSLA") ? 0.022 : 0.012;
  const prices = [];
  let dev = 0;        // 基準パスからの累積乖離(対数)
  let jumpEffect = 0; // イベントによる乖離(%)。時間とともに減衰しアンカーパスへ回帰
  for (const day of days) {
    dev = dev * 0.90 + (rand() * 2 - 1) * dailyVol;
    const jump = jumps.get(day);
    if (jump) jumpEffect += jump;
    let close = basePath(day) * Math.exp(dev) * (1 + jumpEffect / 100);
    jumpEffect *= 0.94;
    prices.push({ date: day, close });
  }

  // アンカー日の値がアンカーに一致するよう、アンカー間で対数比を線形補間して全体を滑らかに補正
  const anchorIdx = [];
  for (const a of sorted) {
    const idx = nearestIndex(prices, a.date);
    if (idx >= 0) anchorIdx.push({ idx, logRatio: Math.log(a.close / prices[idx].close) });
  }
  anchorIdx.sort((a, b) => a.idx - b.idx);
  if (anchorIdx.length) {
    for (let i = 0; i < prices.length; i++) {
      let lr;
      if (i <= anchorIdx[0].idx) lr = anchorIdx[0].logRatio;
      else if (i >= anchorIdx[anchorIdx.length - 1].idx) lr = anchorIdx[anchorIdx.length - 1].logRatio;
      else {
        for (let k = 0; k < anchorIdx.length - 1; k++) {
          const a = anchorIdx[k], b = anchorIdx[k + 1];
          if (i >= a.idx && i <= b.idx) {
            const r = (i - a.idx) / Math.max(1, b.idx - a.idx);
            lr = a.logRatio * (1 - r) + b.logRatio * r;
            break;
          }
        }
      }
      prices[i].close *= Math.exp(lr);
    }
  }

  // 出来高: ベース + ノイズ + イベント日スパイク
  for (let i = 0; i < prices.length; i++) {
    const p = prices[i];
    let vol = meta.baseVolume * (0.7 + rand() * 0.6);
    const chg = i > 0 ? Math.abs(p.close - prices[i - 1].close) / prices[i - 1].close : 0;
    vol *= 1 + chg * 40; // 値動きが大きい日は出来高増
    if (jumps.has(p.date)) vol *= 1.8 + rand();
    p.volume = Math.round(vol);
    p.close = round(p.close, meta.currency);
  }
  return prices;
}

function parseMagnitude(s) {
  if (!s) return null;
  const m = String(s).match(/([+\-−▲▼]?)(\d+(?:\.\d+)?)\s*%/);
  if (!m) return null;
  const v = parseFloat(m[2]);
  return /[\-−▼]/.test(m[1]) ? -v : v;
}

function nearestIndex(prices, date) {
  const x = new Date(date + "T00:00:00Z").getTime();
  let best = -1, bd = Infinity;
  for (let i = 0; i < prices.length; i++) {
    const d = Math.abs(new Date(prices[i].date + "T00:00:00Z").getTime() - x);
    if (d < bd) { bd = d; best = i; }
  }
  return bd <= 5 * 86400000 ? best : -1;
}

function round(v, currency) {
  if (currency === "USD") return Math.round(v * 100) / 100;
  return v >= 5000 ? Math.round(v) : Math.round(v * 10) / 10; // 円は概ね整数
}

/* ---------- メイン ---------- */
function loadJson(file) {
  const p = path.join(RESEARCH, file);
  if (!fs.existsSync(p)) { console.warn(`⚠ ${file} が見つかりません — スキップ`); return null; }
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

function main() {
  const jp = loadJson("research_jp.json");
  const us = loadJson("research_us.json");
  const etf = loadJson("research_etf_market.json");

  const marketEvents = (etf?.marketEvents || []).sort((a, b) => a.date.localeCompare(b.date));

  const symbols = {};
  const sources = [
    ...Object.entries(jp?.symbols || {}),
    ...Object.entries(us?.symbols || {}),
    ...Object.entries(etf?.etfs || {}).map(([c, v]) => [c, { ...v, isEtf: true }]),
  ];

  for (const [code, r] of sources) {
    const meta = SYMBOL_META[code];
    if (!meta) { console.warn(`⚠ 未定義の銘柄コード: ${code} — スキップ`); continue; }
    const events = (r.events || []).sort((a, b) => a.date.localeCompare(b.date));
    let prices;
    try {
      prices = synthesizeSeries(code, r.priceAnchors || [], events, marketEvents, meta);
    } catch (e) {
      console.error(`✗ ${code}: ${e.message}`);
      continue;
    }
    const financials = r.isEtf
      ? { expenseRatio: r.expenseRatio ?? null, distributionYield: r.distributionYield ?? null, week52High: r.week52High ?? null, week52Low: r.week52Low ?? null }
      : { ...(r.financials || {}), week52High: r.week52High ?? null, week52Low: r.week52Low ?? null };

    symbols[code] = {
      name: r.name,
      ...meta,
      baseVolume: undefined,
      events,
      financials,
      prices,
    };
    delete symbols[code].baseVolume;
    console.log(`✓ ${code} ${r.name}: ${prices.length}営業日 / イベント${events.length}件`);
  }

  const data = {
    meta: {
      appName: "StockLens",
      collectedAt: new Date().toISOString().slice(0, 10),
      dataStart: DATA_START,
      dataEnd: DATA_END,
      defaultSymbol: "7203",
      note: "株価アンカー・イベント・財務指標はWeb収集した実データ。日次系列はアンカー点を通るよう補間合成した参考値。",
    },
    symbols,
    marketEvents,
  };

  const js = "// 自動生成ファイル — 編集しないでください。scripts/generate-data.js で再生成します。\n" +
    "window.STOCK_DATA = " + JSON.stringify(data) + ";\n";
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, js);
  console.log(`\n→ ${path.relative(ROOT, OUT)} を生成しました (${(js.length / 1024).toFixed(0)} KB, 銘柄${Object.keys(symbols).length}件, 市場イベント${marketEvents.length}件)`);
}

main();
