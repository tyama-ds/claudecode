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

/* ---------- メイン ---------- */
async function main() {
  console.log("J-Quants API 認証中…");
  const idToken = await authenticate();
  console.log("認証OK\n");

  // 取得範囲: 過去2年(プランの制約内で取れた分だけ返る)
  const to = new Date().toISOString().slice(0, 10);
  const from = new Date(Date.now() - 730 * 86400000).toISOString().slice(0, 10);

  const result = { fetchedAt: new Date().toISOString(), source: "J-Quants API (JPX)", symbols: {} };
  for (const code of JP_CODES) {
    const code5 = code + "0";
    process.stdout.write(`${code} を取得中… `);
    try {
      const prices = await fetchDailyQuotes(idToken, code5, from, to);
      if (!prices.length) { console.log("日足0件(プランの提供範囲外の可能性)"); continue; }
      const financials = await fetchFinancials(idToken, code5, prices[prices.length - 1].close);
      result.symbols[code] = { prices, financials };
      console.log(`日足${prices.length}件 (${prices[0].date}〜${prices[prices.length - 1].date})` +
        (financials ? " / 財務OK" : " / 財務なし"));
    } catch (e) {
      console.log(`失敗: ${e.message}`);
    }
    await sleep(300); // 行儀よく
  }

  if (!Object.keys(result.symbols).length) {
    console.error("\n1銘柄も取得できませんでした。プラン・認証情報を確認してください。");
    process.exit(1);
  }
  fs.writeFileSync(OUT, JSON.stringify(result));
  console.log(`\n→ ${path.relative(ROOT, OUT)} に保存しました。`);
  console.log("次に実行: node scripts/generate-data.js  (data/stock-data.js に反映)");
}

main().catch(e => { console.error("エラー:", e.message); process.exit(1); });
