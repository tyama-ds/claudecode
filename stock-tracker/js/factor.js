/* ==========================================================================
   factor.js — ファクター分析
   個別株ユニバース(ETF除く)横断で6ファクターをzスコア化し、
   偏差値(50±10)・スタイル判定・ランキングを提供する。
   ========================================================================== */

const Factor = (() => {

  const FACTORS = [
    { key: "value", label: "バリュー", desc: "PER・PBRの低さ(割安度)" },
    { key: "quality", label: "クオリティ", desc: "ROE・自己資本比率の高さ" },
    { key: "momentum", label: "モメンタム", desc: "直近12ヶ月・3ヶ月の株価上昇率" },
    { key: "lowvol", label: "低ボラティリティ", desc: "日々の値動きの穏やかさ" },
    { key: "dividend", label: "配当", desc: "配当利回りの高さ" },
    { key: "size", label: "サイズ", desc: "時価総額の大きさ(大型ほど高)" },
  ];

  let cache = null;

  /* ---------- 生指標の計算 ---------- */
  function returns(prices, days) {
    const n = prices.length;
    if (n <= days) return null;
    return (prices[n - 1].close - prices[n - 1 - days].close) / prices[n - 1 - days].close;
  }

  function dailyVolatility(prices) {
    const rets = [];
    for (let i = 1; i < prices.length; i++) {
      rets.push(Math.log(prices[i].close / prices[i - 1].close));
    }
    const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
    const varr = rets.reduce((a, r) => a + (r - mean) ** 2, 0) / rets.length;
    return Math.sqrt(varr);
  }

  // "40兆円" "8兆円" "$4.5T" "$500B" などを兆円換算の数値に変換(比較用の近似)
  function parseMarketCap(s) {
    if (s == null) return null;
    if (typeof s === "number") return s;
    const str = String(s).replace(/[,\s]/g, "");
    let m;
    if ((m = str.match(/([\d.]+)兆円?/))) return parseFloat(m[1]);
    if ((m = str.match(/([\d.]+)億円?/))) return parseFloat(m[1]) / 10000;
    if ((m = str.match(/\$([\d.]+)T/i))) return parseFloat(m[1]) * 150; // 1T$ ≒ 150兆円
    if ((m = str.match(/\$([\d.]+)B/i))) return parseFloat(m[1]) * 0.15;
    return null;
  }

  function rawMetrics(sym) {
    const f = sym.financials || {};
    return {
      invPer: f.per > 0 ? 1 / f.per : null,
      invPbr: f.pbr > 0 ? 1 / f.pbr : null,
      roe: f.roe ?? null,
      equityRatio: f.equityRatio ?? null,
      ret12m: returns(sym.prices, 250),
      ret3m: returns(sym.prices, 63),
      negVol: -dailyVolatility(sym.prices),
      dividendYield: f.dividendYield ?? null,
      logCap: (() => { const c = parseMarketCap(f.marketCap); return c ? Math.log(c) : null; })(),
    };
  }

  /* ---------- zスコア → 偏差値 ---------- */
  function zScores(values) {
    const v = values.filter(x => x != null);
    if (v.length < 2) return values.map(() => null);
    const mean = v.reduce((a, b) => a + b, 0) / v.length;
    const sd = Math.sqrt(v.reduce((a, x) => a + (x - mean) ** 2, 0) / v.length) || 1;
    return values.map(x => (x == null ? null : (x - mean) / sd));
  }

  function hensachi(z) {
    if (z == null) return null;
    return Math.round(Math.max(20, Math.min(80, 50 + 10 * z)) * 10) / 10;
  }

  function avg(arr) {
    const v = arr.filter(x => x != null);
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  }

  /* ---------- 全銘柄のファクタースコア(偏差値)を計算 ---------- */
  function computeAll() {
    if (cache) return cache;
    const codes = Object.keys(STOCK_DATA.symbols).filter(c => STOCK_DATA.symbols[c].type !== "etf");
    const raw = codes.map(c => rawMetrics(STOCK_DATA.symbols[c]));

    const z = {};
    for (const k of ["invPer", "invPbr", "roe", "equityRatio", "ret12m", "ret3m", "negVol", "dividendYield", "logCap"]) {
      z[k] = zScores(raw.map(r => r[k]));
    }

    const result = {};
    codes.forEach((code, i) => {
      result[code] = {
        code,
        name: STOCK_DATA.symbols[code].name,
        market: STOCK_DATA.symbols[code].market,
        value: hensachi(avg([z.invPer[i], z.invPbr[i]])),
        quality: hensachi(avg([z.roe[i], z.equityRatio[i]])),
        momentum: hensachi(avg([z.ret12m[i], z.ret3m[i]])),
        lowvol: hensachi(z.negVol[i]),
        dividend: hensachi(z.dividendYield[i]),
        size: hensachi(z.logCap[i]),
      };
    });
    cache = result;
    return result;
  }

  /* ---------- スタイル判定 ---------- */
  function classifyStyle(code) {
    const s = computeAll()[code];
    if (!s) return null;
    const tags = [];
    tags.push(s.size >= 55 ? "大型" : s.size <= 45 ? "中型" : "準大型");
    if (s.value >= 58) tags.push("バリュー");
    else if (s.value <= 42 && s.momentum >= 52) tags.push("グロース");
    if (s.quality >= 58) tags.push("クオリティ");
    if (s.dividend >= 58) tags.push("高配当");
    if (s.lowvol >= 58) tags.push("ディフェンシブ");
    if (s.momentum >= 60) tags.push("高モメンタム");
    if (tags.length === 1) tags.push("ブレンド");

    // 解説文
    const parts = [];
    if (s.value >= 58) parts.push("指標面の割安さ");
    if (s.value <= 42) parts.push("成長期待の織り込み(割高寄り)");
    if (s.quality >= 58) parts.push("高い資本効率・財務の質");
    if (s.momentum >= 58) parts.push("強い株価トレンド");
    if (s.momentum <= 42) parts.push("弱い株価トレンド");
    if (s.dividend >= 58) parts.push("厚い配当");
    if (s.lowvol >= 58) parts.push("値動きの安定性");
    if (s.lowvol <= 42) parts.push("値動きの大きさ");
    const comment = parts.length
      ? `この銘柄の特徴は${parts.join("、")}です。`
      : "各ファクターとも市場平均並みで、際立った偏りのないバランス型です。";

    return { tags, comment, scores: s };
  }

  /* ---------- ランキング ---------- */
  function ranking(factorKey) {
    const all = Object.values(computeAll());
    return all
      .filter(s => s[factorKey] != null)
      .sort((a, b) => b[factorKey] - a[factorKey]);
  }

  return { FACTORS, computeAll, classifyStyle, ranking };
})();
