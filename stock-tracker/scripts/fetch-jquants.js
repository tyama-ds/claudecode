#!/usr/bin/env node
/* ==========================================================================
   fetch-jquants.js — J-Quants API(JPX公式)から日本株の実データを取得

   取得内容:
     - /prices/daily_quotes : 調整済み日足(終値・出来高)
     - /fins/statements     : 直近決算からROE・自己資本比率・EPS/BPS等を算出

   出力: research/jquants_data.json
     (この後 `node scripts/generate-data.js` を実行すると data/stock-data.js に
      反映され、該当銘柄は合成系列ではなく実データの日足で表示されます)

   認証情報は環境変数で渡す(コード・リポジトリに含めないこと):
     export JQUANTS_MAIL_ADDRESS="you@example.com"
     export JQUANTS_PASSWORD="********"
   または取得済みリフレッシュトークンを直接:
     export JQUANTS_REFRESH_TOKEN="..."

   プラン対応:
     契約プランを自動判定し、利用可能なエンドポイントだけを取得します
     (未契約のものは403を検知してスキップ。どのプランでもエラーなく完走):
       全プラン     : /prices/daily_quotes, /fins/statements
       Light以上    : /indices/topix           → TOPIX比較チャート
       Standard以上 : /markets/weekly_margin_interest → 信用残トレンド
       Premium      : /fins/dividend           → 配当履歴
       Premium      : /fins/fs_details         → 営業CF等の詳細財務

   注意:
     - J-QuantsのFreeプランは約12週間遅延・過去2年分の日足です(Light以上は
       前営業日まで)。本スクリプトは取得できた範囲をそのまま保存します。
     - J-Quantsの利用規約上、取得データの再配布は制限されています。出力JSONは
       .gitignore に登録済みで、リポジトリにはコミットされません。
   ========================================================================== */

const fs = require("fs");
const path = require("path");

const BASE = "https://api.jquants.com/v1";
const ROOT = path.join(__dirname, "..");
const OUT = path.join(ROOT, "research", "jquants_data.json");

// アプリ収録の国内銘柄(4桁コード)。J-Quants APIは5桁コード(末尾0)を使う。
const JP_CODES = ["7203", "6758", "5401", "8306", "8035", "9983", "1321", "1306"];

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function api(pathname, params = {}, idToken = null, method = "GET", body = null) {
  const url = new URL(BASE + pathname);
  for (const [k, v] of Object.entries(params)) if (v != null) url.searchParams.set(k, v);
  for (let attempt = 1; ; attempt++) {
    const res = await fetch(url, {
      method,
      headers: {
        ...(idToken ? { Authorization: `Bearer ${idToken}` } : {}),
        ...(body ? { "Content-Type": "application/json" } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
    });
    if (res.status === 429 && attempt <= 5) { // レート制限
      await sleep(1500 * attempt);
      continue;
    }
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`${method} ${pathname} → HTTP ${res.status}: ${text.slice(0, 300)}`);
    }
    return res.json();
  }
}

/* ---------- 認証 ---------- */
async function authenticate() {
  let refreshToken = process.env.JQUANTS_REFRESH_TOKEN;
  if (!refreshToken) {
    const mail = process.env.JQUANTS_MAIL_ADDRESS;
    const pass = process.env.JQUANTS_PASSWORD;
    if (!mail || !pass) {
      console.error(
        "認証情報がありません。以下のいずれかを設定してください:\n" +
        "  export JQUANTS_MAIL_ADDRESS=... ; export JQUANTS_PASSWORD=...\n" +
        "  export JQUANTS_REFRESH_TOKEN=...\n" +
        "アカウント登録(Freeプランあり): https://jpx-jquants.com/"
      );
      process.exit(1);
    }
    const r = await api("/token/auth_user", {}, null, "POST", { mailaddress: mail, password: pass });
    refreshToken = r.refreshToken;
  }
  const r2 = await api("/token/auth_refresh", { refreshtoken: refreshToken }, null, "POST");
  return r2.idToken;
}

/* ---------- 日足(ページネーション対応) ---------- */
async function fetchDailyQuotes(idToken, code5, from, to) {
  const quotes = [];
  let paginationKey = null;
  do {
    const r = await api("/prices/daily_quotes",
      { code: code5, from, to, pagination_key: paginationKey }, idToken);
    quotes.push(...(r.daily_quotes || []));
    paginationKey = r.pagination_key || null;
  } while (paginationKey);
  // 調整済み終値・出来高のみ抽出
  return quotes
    .filter(q => q.AdjustmentClose != null)
    .map(q => ({
      date: q.Date,
      close: q.AdjustmentClose,
      volume: q.AdjustmentVolume ?? q.Volume ?? 0,
    }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

/* ---------- 財務諸表 → 指標算出 ---------- */
async function fetchFinancials(idToken, code5, lastClose) {
  const r = await api("/fins/statements", { code: code5 }, idToken).catch(() => null);
  const list = (r && r.statements) || [];
  // 直近の通期(FY)実績を優先、なければ直近四半期
  const fy = list.filter(s => s.TypeOfCurrentPeriod === "FY").sort((a, b) => (b.CurrentPeriodEndDate || "").localeCompare(a.CurrentPeriodEndDate || ""));
  const s = fy[0] || list.sort((a, b) => (b.CurrentPeriodEndDate || "").localeCompare(a.CurrentPeriodEndDate || ""))[0];
  if (!s) return null;

  const num = v => (v === "" || v == null ? null : Number(v));
  const equity = num(s.Equity);
  const totalAssets = num(s.TotalAssets);
  const profit = num(s.Profit);
  const eps = num(s.EarningsPerShare);
  const bps = num(s.BookValuePerShare);
  const dps = num(s.ResultDividendPerShareAnnual) ?? num(s.ForecastDividendPerShareAnnual);

  const out = { asOf: `J-Quants ${s.CurrentPeriodEndDate || ""} (${s.TypeOfCurrentPeriod || ""})` };
  if (equity && totalAssets) out.equityRatio = round(equity / totalAssets * 100, 1);
  if (profit && equity) out.roe = round(profit / equity * 100, 1);
  if (eps && lastClose) out.per = round(lastClose / eps, 1);
  if (bps && lastClose) out.pbr = round(lastClose / bps, 2);
  if (dps && lastClose) out.dividendYield = round(dps / lastClose * 100, 2);
  const revenue = num(s.NetSales);
  const opIncome = num(s.OperatingProfit);
  if (revenue) out.revenue = Math.round(revenue / 1e8);         // 億円
  if (opIncome) out.operatingIncome = Math.round(opIncome / 1e8);
  if (profit) out.netIncome = Math.round(profit / 1e8);
  return out;
}

function round(v, d) { const p = 10 ** d; return Math.round(v * p) / p; }

/* ---------- 上位プラン向けエンドポイント ---------- */
// 未契約(401/403)を検知したら false を返し、以降そのエンドポイントはスキップ
function isSubscriptionError(e) {
  return /HTTP (401|403)/.test(e.message) || /subscription|not authorized/i.test(e.message);
}

// Light以上: TOPIX指数の日足
async function fetchTopix(idToken, from, to) {
  const series = [];
  let paginationKey = null;
  do {
    const r = await api("/indices/topix", { from, to, pagination_key: paginationKey }, idToken);
    series.push(...(r.topix || []));
    paginationKey = r.pagination_key || null;
  } while (paginationKey);
  return series
    .filter(x => x.Close != null)
    .map(x => ({ date: x.Date, close: x.Close }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

// Standard以上: 銘柄別の信用取引週末残高(信用買残・売残)
async function fetchMarginInterest(idToken, code5, from, to) {
  const rows = [];
  let paginationKey = null;
  do {
    const r = await api("/markets/weekly_margin_interest",
      { code: code5, from, to, pagination_key: paginationKey }, idToken);
    rows.push(...(r.weekly_margin_interest || []));
    paginationKey = r.pagination_key || null;
  } while (paginationKey);
  return rows
    .filter(x => x.LongMarginTradeVolume != null || x.ShortMarginTradeVolume != null)
    .map(x => ({
      date: x.Date,
      long: x.LongMarginTradeVolume ?? 0,   // 信用買残(株)
      short: x.ShortMarginTradeVolume ?? 0, // 信用売残(株)
    }))
    .sort((a, b) => a.date.localeCompare(b.date));
}

// Premium: 配当履歴(1株あたり配当金の推移)
async function fetchDividends(idToken, code5) {
  const rows = [];
  let paginationKey = null;
  do {
    const r = await api("/fins/dividend", { code: code5, pagination_key: paginationKey }, idToken);
    rows.push(...(r.dividend || []));
    paginationKey = r.pagination_key || null;
  } while (paginationKey);
  return rows
    .filter(x => x.GrossDividendRate != null && x.GrossDividendRate !== "" && x.RecordDate)
    .map(x => ({
      recordDate: x.RecordDate,
      dividend: Number(x.GrossDividendRate),
      term: x.InterimFinalCode === "1" ? "中間" : x.InterimFinalCode === "2" ? "期末" : "",
      forecast: x.ForecastResultCode === "1", // 1=予想, 2=実績 想定
    }))
    .sort((a, b) => a.recordDate.localeCompare(b.recordDate))
    .slice(-12); // 直近12回分
}

// Premium: 詳細財務諸表から営業CF等を抽出
async function fetchFsDetails(idToken, code5) {
  const r = await api("/fins/fs_details", { code: code5 }, idToken);
  const list = (r.fs_details || []).sort((a, b) =>
    (b.DisclosedDate || "").localeCompare(a.DisclosedDate || ""));
  const fs = list[0]?.FinancialStatement;
  if (!fs) return null;
  // 会計基準によりキー名が異なるため部分一致で探す
  const find = pattern => {
    for (const [k, v] of Object.entries(fs)) {
      if (pattern.test(k) && v !== "" && v != null && !isNaN(Number(v))) return Number(v);
    }
    return null;
  };
  const out = {};
  const opCF = find(/cash flows? from .*operating activities/i);
  if (opCF != null) out.operatingCF = Math.round(opCF / 1e8); // 億円
  return Object.keys(out).length ? out : null;
}

/* ---------- メイン ---------- */
async function main() {
  console.log("J-Quants API 認証中…");
  const idToken = await authenticate();
  console.log("認証OK\n");

  // 取得範囲: 過去2年(プランの制約内で取れた分だけ返る)
  const to = new Date().toISOString().slice(0, 10);
  const from = new Date(Date.now() - 730 * 86400000).toISOString().slice(0, 10);

  const result = { fetchedAt: new Date().toISOString(), source: "J-Quants API (JPX)", symbols: {} };
  // 拡張エンドポイントの利用可否(未契約を検知したら false にして以降スキップ)
  const features = { topix: true, marginInterest: true, dividend: true, fsDetails: true };

  /* --- Light以上: TOPIX指数 --- */
  try {
    const topix = await fetchTopix(idToken, from, to);
    if (topix.length) {
      result.topix = topix;
      console.log(`✦ TOPIX指数: ${topix.length}件 (${topix[0].date}〜${topix[topix.length - 1].date})`);
    }
  } catch (e) {
    features.topix = false;
    if (isSubscriptionError(e)) console.log("・TOPIX指数: 未契約のためスキップ(Light以上で利用可)");
    else console.log(`・TOPIX指数: 取得失敗 (${e.message})`);
  }

  for (const code of JP_CODES) {
    const code5 = code + "0";
    process.stdout.write(`${code} を取得中… `);
    try {
      const prices = await fetchDailyQuotes(idToken, code5, from, to);
      if (!prices.length) { console.log("日足0件(プランの提供範囲外の可能性)"); continue; }
      const financials = await fetchFinancials(idToken, code5, prices[prices.length - 1].close);
      const sym = { prices, financials };
      const parts = [`日足${prices.length}件 (${prices[0].date}〜${prices[prices.length - 1].date})`,
        financials ? "財務OK" : "財務なし"];
      const isEtf = code === "1321" || code === "1306";

      /* --- Standard以上: 信用残(個別株のみ) --- */
      if (features.marginInterest && !isEtf) {
        try {
          const mi = await fetchMarginInterest(idToken, code5, from, to);
          if (mi.length) { sym.marginInterest = mi; parts.push(`信用残${mi.length}週`); }
        } catch (e) {
          if (isSubscriptionError(e)) { features.marginInterest = false; parts.push("信用残:未契約(Standard以上)"); }
          else parts.push(`信用残:失敗`);
        }
      }
      /* --- Premium: 配当履歴・詳細財務(個別株のみ) --- */
      if (features.dividend && !isEtf) {
        try {
          const div = await fetchDividends(idToken, code5);
          if (div.length) { sym.dividends = div; parts.push(`配当${div.length}件`); }
        } catch (e) {
          if (isSubscriptionError(e)) { features.dividend = false; parts.push("配当履歴:未契約(Premium)"); }
          else parts.push("配当履歴:失敗");
        }
      }
      if (features.fsDetails && !isEtf) {
        try {
          const fsx = await fetchFsDetails(idToken, code5);
          if (fsx) { sym.extraFinancials = fsx; parts.push("詳細財務OK"); }
        } catch (e) {
          if (isSubscriptionError(e)) { features.fsDetails = false; parts.push("詳細財務:未契約(Premium)"); }
          else parts.push("詳細財務:失敗");
        }
      }

      result.symbols[code] = sym;
      console.log(parts.join(" / "));
    } catch (e) {
      console.log(`失敗: ${e.message}`);
    }
    await sleep(300); // 行儀よく
  }

  result.planFeatures = features;
  const tier = features.fsDetails || features.dividend ? "Premium相当"
    : features.marginInterest ? "Standard相当"
    : features.topix ? "Light相当" : "Free相当";
  console.log(`\n検出プラン: ${tier}(利用可能: ${Object.entries(features).filter(([, v]) => v).map(([k]) => k).join(", ") || "基本APIのみ"})`);

  if (!Object.keys(result.symbols).length) {
    console.error("\n1銘柄も取得できませんでした。プラン・認証情報を確認してください。");
    process.exit(1);
  }
  fs.writeFileSync(OUT, JSON.stringify(result));
  console.log(`\n→ ${path.relative(ROOT, OUT)} に保存しました。`);
  console.log("次に実行: node scripts/generate-data.js  (data/stock-data.js に反映)");
}

main().catch(e => { console.error("エラー:", e.message); process.exit(1); });
