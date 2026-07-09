/* ==========================================================================
   backtest.js — バックテスト(過去データで戦略を検証・発注なし)
   ロングオンリー・全額イン/アウトの単純モデル。売買手数料を考慮。
   ブラウザ(window.Backtest)/Node(module.exports)両対応。
   ========================================================================== */

(function (root) {

  // signals: [{date, close, signal}], opts: {initialCash, feeRate}
  function run(signals, opts = {}) {
    const initialCash = opts.initialCash ?? 1_000_000;
    const feeRate = opts.feeRate ?? 0.0005; // 片道0.05%(実際は証券会社による)
    if (!signals || signals.length < 2) return null;

    let cash = initialCash, shares = 0, entryPrice = null, entryDate = null;
    const trades = [];
    const equity = [];

    for (const bar of signals) {
      // シグナルに従って建玉変更
      if (bar.signal === "buy" && shares === 0) {
        shares = (cash * (1 - feeRate)) / bar.close;
        entryPrice = bar.close; entryDate = bar.date; cash = 0;
      } else if (bar.signal === "sell" && shares > 0) {
        cash = shares * bar.close * (1 - feeRate);
        const ret = (bar.close - entryPrice) / entryPrice;
        trades.push({ entryDate, exitDate: bar.date, entryPrice, exitPrice: bar.close, return: ret });
        shares = 0; entryPrice = null; entryDate = null;
      }
      equity.push({ date: bar.date, value: cash + shares * bar.close });
    }
    // 最終日に建玉が残っていれば時価評価(未決済)
    const last = signals[signals.length - 1];
    let openTrade = null;
    if (shares > 0) {
      openTrade = { entryDate, entryPrice, currentPrice: last.close, return: (last.close - entryPrice) / entryPrice };
    }

    return { trades, equity, openTrade, metrics: metrics(equity, trades, signals, initialCash) };
  }

  function metrics(equity, trades, signals, initialCash) {
    const finalEquity = equity[equity.length - 1].value;
    const totalReturn = (finalEquity - initialCash) / initialCash;

    // 買い持ち比較
    const first = signals[0].close, lastC = signals[signals.length - 1].close;
    const buyHold = (lastC - first) / first;

    // 最大ドローダウン
    let peak = -Infinity, maxDD = 0;
    for (const e of equity) {
      if (e.value > peak) peak = e.value;
      const dd = (e.value - peak) / peak;
      if (dd < maxDD) maxDD = dd;
    }

    // 勝率
    const closed = trades.length;
    const wins = trades.filter(t => t.return > 0).length;
    const winRate = closed ? wins / closed : null;

    // シャープレシオ(日次リターン・年率化)
    const rets = [];
    for (let i = 1; i < equity.length; i++) {
      const prev = equity[i - 1].value;
      if (prev > 0) rets.push((equity[i].value - prev) / prev);
    }
    let sharpe = null;
    if (rets.length > 2) {
      const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
      const sd = Math.sqrt(rets.reduce((a, r) => a + (r - mean) ** 2, 0) / rets.length);
      sharpe = sd > 0 ? (mean / sd) * Math.sqrt(252) : null;
    }

    return { finalEquity, totalReturn, buyHold, maxDrawdown: maxDD, winRate, numTrades: closed, sharpe, initialCash };
  }

  const api = { run };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Backtest = api;

})(typeof window !== "undefined" ? window : globalThis);
