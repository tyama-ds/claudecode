/* ==========================================================================
   brokers/mock.js — 模擬ブローカー(テスト用・実弾なし)
   kabuステーションが無くてもエンジンの動作確認ができるよう、
   価格は履歴データの最新終値、建玉・現金はメモリ内で管理する。
   ========================================================================== */

class MockBroker {
  constructor(opts = {}) {
    this.priceMap = opts.priceMap || {}; // { code: latestClose }
    this.cash = opts.cash ?? 1_000_000;
    this.positions = {};                  // { code: {qty, avgPrice} }
    this.orders = [];
  }
  async connect() { return true; }
  async getPrice(code) { return { price: this.priceMap[code] ?? null, raw: { mock: true } }; }
  async getCash() { return this.cash; }
  async getPosition(code) {
    const p = this.positions[code];
    return { qty: p ? p.qty : 0, avgPrice: p ? p.avgPrice : null, raw: p || null };
  }
  async sendOrder({ code, side, qty }) {
    const price = this.priceMap[code] ?? 0;
    if (side === "buy") {
      this.cash -= price * qty;
      this.positions[code] = { qty, avgPrice: price };
    } else {
      this.cash += price * qty;
      delete this.positions[code];
    }
    const orderId = "MOCK-" + (this.orders.length + 1);
    this.orders.push({ orderId, code, side, qty, price });
    return { orderId, raw: { mock: true } };
  }
  name() { return "MockBroker(テスト用・実弾なし)"; }
}

module.exports = MockBroker;
