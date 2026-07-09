/* ==========================================================================
   strategy.js — 売買シグナル生成(発注なし・分析専用)
   価格系列から売買シグナルを生成する。実弾の発注は一切行わない。
   ブラウザ(window.Strategy)でもNode(module.exports)でも使える。
   ========================================================================== */

(function (root) {

  /* ---------- テクニカル指標 ---------- */
  function sma(closes, window) {
    const out = new Array(closes.length).fill(null);
    let sum = 0;
    for (let i = 0; i < closes.length; i++) {
      sum += closes[i];
      if (i >= window) sum -= closes[i - window];
      if (i >= window - 1) out[i] = sum / window;
    }
    return out;
  }

  // Wilder's RSI
  function rsi(closes, period = 14) {
    const out = new Array(closes.length).fill(null);
    if (closes.length <= period) return out;
    let avgGain = 0, avgLoss = 0;
    for (let i = 1; i <= period; i++) {
      const d = closes[i] - closes[i - 1];
      if (d >= 0) avgGain += d; else avgLoss -= d;
    }
    avgGain /= period; avgLoss /= period;
    out[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
    for (let i = period + 1; i < closes.length; i++) {
      const d = closes[i] - closes[i - 1];
      const gain = d > 0 ? d : 0, loss = d < 0 ? -d : 0;
      avgGain = (avgGain * (period - 1) + gain) / period;
      avgLoss = (avgLoss * (period - 1) + loss) / period;
      out[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
    }
    return out;
  }

  /* ---------- 戦略定義 ---------- */
  // 各戦略は generateSignals(prices, params) を実装し、
  // 各日について signal('buy'|'sell'|null) と補助指標を返す。
  const STRATEGIES = {
    maCross: {
      label: "移動平均クロス",
      desc: "短期MAが長期MAを上抜けで買い(ゴールデンクロス)、下抜けで売り(デッドクロス)",
      params: [
        { key: "short", label: "短期MA", default: 25, min: 3, max: 100 },
        { key: "long", label: "長期MA", default: 75, min: 5, max: 200 },
      ],
      generate(prices, p) {
        const closes = prices.map(d => d.close);
        const s = sma(closes, p.short), l = sma(closes, p.long);
        return prices.map((d, i) => {
          let signal = null;
          if (i > 0 && s[i] != null && l[i] != null && s[i - 1] != null && l[i - 1] != null) {
            if (s[i - 1] <= l[i - 1] && s[i] > l[i]) signal = "buy";
            else if (s[i - 1] >= l[i - 1] && s[i] < l[i]) signal = "sell";
          }
          return { date: d.date, close: d.close, signal, indicators: { short: s[i], long: l[i] } };
        });
      },
    },
    rsiReversal: {
      label: "RSI逆張り",
      desc: "RSIが売られすぎ水準を上抜けたら買い、買われすぎ水準を下抜けたら売り",
      params: [
        { key: "period", label: "RSI期間", default: 14, min: 5, max: 30 },
        { key: "buy", label: "買い水準(以下で売られすぎ)", default: 30, min: 10, max: 45 },
        { key: "sell", label: "売り水準(以上で買われすぎ)", default: 70, min: 55, max: 90 },
      ],
      generate(prices, p) {
        const closes = prices.map(d => d.close);
        const r = rsi(closes, p.period);
        return prices.map((d, i) => {
          let signal = null;
          if (i > 0 && r[i] != null && r[i - 1] != null) {
            if (r[i - 1] <= p.buy && r[i] > p.buy) signal = "buy";       // 売られすぎから回復
            else if (r[i - 1] >= p.sell && r[i] < p.sell) signal = "sell"; // 買われすぎから下落
          }
          return { date: d.date, close: d.close, signal, indicators: { rsi: r[i] } };
        });
      },
    },
  };

  function resolveParams(strategyKey, overrides = {}) {
    const strat = STRATEGIES[strategyKey];
    const p = {};
    for (const def of strat.params) {
      const v = overrides[def.key];
      p[def.key] = (v == null || isNaN(v)) ? def.default : Number(v);
    }
    return p;
  }

  function generateSignals(prices, strategyKey, overrides) {
    const strat = STRATEGIES[strategyKey];
    if (!strat) throw new Error("unknown strategy: " + strategyKey);
    return strat.generate(prices, resolveParams(strategyKey, overrides));
  }

  // 直近の「行動すべきシグナル」(最後のbuy/sell)と現在の推奨を返す
  function currentSignal(signalSeries) {
    let last = null;
    for (let i = signalSeries.length - 1; i >= 0; i--) {
      if (signalSeries[i].signal) { last = { ...signalSeries[i], idxFromEnd: signalSeries.length - 1 - i }; break; }
    }
    return last;
  }

  const api = { sma, rsi, STRATEGIES, generateSignals, currentSignal, resolveParams };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Strategy = api;

})(typeof window !== "undefined" ? window : globalThis);
