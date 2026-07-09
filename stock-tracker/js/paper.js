/* ==========================================================================
   paper.js — ペーパートレード(仮想口座・実弾なし)
   仮想資金の売買をlocalStorageに保存。実際の発注は一切行わない。
   ロングオンリー・全額イン/アウト(バックテストと同じ単純モデル)。
   ========================================================================== */

const Paper = (() => {
  const INITIAL_CASH = 1_000_000;
  const FEE_RATE = 0.0005;
  const KEY = code => `stocklens-paper-${code}`;

  function load(code) {
    try {
      const raw = localStorage.getItem(KEY(code));
      if (raw) return JSON.parse(raw);
    } catch (e) { /* ignore */ }
    return { cash: INITIAL_CASH, shares: 0, avgPrice: null, realizedPnL: 0, log: [], createdAt: null };
  }

  function save(code, state) {
    try { localStorage.setItem(KEY(code), JSON.stringify(state)); } catch (e) { /* ignore */ }
  }

  function reset(code) {
    localStorage.removeItem(KEY(code));
    return load(code);
  }

  // action: 'buy' | 'sell'。price/date は約定価格・約定日(=最新データ)。
  // 成功時は更新後stateを返し、実行不可(建玉なしで売り等)なら {error} を返す。
  function execute(code, action, price, date, note = "") {
    const s = load(code);
    if (action === "buy") {
      if (s.shares > 0) return { error: "すでに建玉があります(このモデルは全額イン/アウトです)" };
      if (s.cash <= 0) return { error: "現金がありません" };
      const shares = (s.cash * (1 - FEE_RATE)) / price;
      s.shares = shares; s.avgPrice = price; s.cash = 0;
      s.createdAt = s.createdAt || date;
      s.log.unshift({ date, action: "買い", price, shares: Math.round(shares), note });
    } else if (action === "sell") {
      if (s.shares <= 0) return { error: "建玉がありません" };
      const proceeds = s.shares * price * (1 - FEE_RATE);
      const pnl = (price - s.avgPrice) * s.shares - s.shares * price * FEE_RATE;
      s.realizedPnL += pnl;
      s.log.unshift({ date, action: "売り", price, shares: Math.round(s.shares), note, pnl: Math.round(pnl) });
      s.cash += proceeds; s.shares = 0; s.avgPrice = null;
    } else {
      return { error: "不明なアクション" };
    }
    save(code, s);
    return s;
  }

  function summary(code, currentPrice) {
    const s = load(code);
    const marketValue = s.shares * currentPrice;
    const totalValue = s.cash + marketValue;
    const unrealizedPnL = s.shares > 0 ? (currentPrice - s.avgPrice) * s.shares : 0;
    const totalReturn = (totalValue - INITIAL_CASH) / INITIAL_CASH;
    return { ...s, currentPrice, marketValue, totalValue, unrealizedPnL, totalReturn, initialCash: INITIAL_CASH };
  }

  return { load, reset, execute, summary, INITIAL_CASH };
})();
