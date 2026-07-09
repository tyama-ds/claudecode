/* ==========================================================================
   brokers/kabus.js — auカブコム証券 kabuステーションAPI アダプタ

   前提:
     - PC上で「kabuステーション」アプリが起動し、API利用が有効(APIモードON)
     - ローカルにREST APIが立つ: 本番 http://localhost:18080/kabusapi
                                   検証 http://localhost:18081/kabusapi
   認証:
     - APIパスワード(kabuステーションのAPI設定)で /token を叩きトークン取得
     - 発注時は別途「取引パスワード」が必要(sendorderのPassword)
   秘密情報は環境変数で渡す(コードに書かない):
     KABU_API_PASSWORD    … APIパスワード(/token用)
     KABU_TRADE_PASSWORD  … 取引パスワード(発注用)

   注意: 発注パラメータ(DelivType/FundType/AccountType等)は口座区分により
   異なる。必ず「検証環境」(env:"verification")と少額で挙動を確認すること。
   ========================================================================== */

const ENDPOINTS = {
  production: "http://localhost:18080/kabusapi",
  verification: "http://localhost:18081/kabusapi",
};

class KabusBroker {
  constructor(opts = {}) {
    this.base = ENDPOINTS[opts.env === "production" ? "production" : "verification"];
    this.apiPassword = process.env.KABU_API_PASSWORD;
    this.tradePassword = process.env.KABU_TRADE_PASSWORD;
    this.token = null;
  }

  async _req(method, path, { body, auth = true } = {}) {
    const headers = { "Content-Type": "application/json" };
    if (auth) {
      if (!this.token) throw new Error("未認証です。connect() を先に呼んでください。");
      headers["X-API-KEY"] = this.token;
    }
    const res = await fetch(this.base + path, {
      method, headers, body: body ? JSON.stringify(body) : undefined,
    });
    const text = await res.text();
    let json; try { json = text ? JSON.parse(text) : {}; } catch { json = { raw: text }; }
    if (!res.ok) {
      throw new Error(`kabus ${method} ${path} → HTTP ${res.status}: ${JSON.stringify(json).slice(0, 300)}`);
    }
    return json;
  }

  async connect() {
    if (!this.apiPassword) throw new Error("環境変数 KABU_API_PASSWORD が未設定です。");
    const r = await this._req("POST", "/token", { auth: false, body: { APIPassword: this.apiPassword } });
    if (!r.Token) throw new Error("トークン取得に失敗: " + JSON.stringify(r));
    this.token = r.Token;
    return true;
  }

  // 現在値(板情報)。code="7203", exchange=1(東証)
  async getPrice(code, exchange = 1) {
    const r = await this._req("GET", `/board/${code}@${exchange}`);
    // CurrentPrice が無い場合(引け後等)は BidPrice/AskPrice や前日終値でフォールバック
    const price = r.CurrentPrice ?? r.PreviousClose ?? r.BidPrice ?? r.AskPrice ?? null;
    return { price, raw: r };
  }

  // 買付余力
  async getCash() {
    const r = await this._req("GET", "/wallet/cash");
    return r.StockAccountWallet != null ? Number(r.StockAccountWallet) : null;
  }

  // 現物保有(product=1: 現物)
  async getPosition(code) {
    const list = await this._req("GET", "/positions?product=1");
    const rows = (Array.isArray(list) ? list : []).filter(p => String(p.Symbol) === String(code));
    const qty = rows.reduce((a, p) => a + (Number(p.LeavesQty) || 0), 0);
    const avg = rows.length ? Number(rows[0].Price) : null;
    return { qty, avgPrice: avg, raw: rows };
  }

  // 現物 成行注文。side: "buy" | "sell"
  async sendOrder({ code, exchange = 1, side, qty, accountType = 4 }) {
    if (!this.tradePassword) throw new Error("環境変数 KABU_TRADE_PASSWORD が未設定です。");
    const isBuy = side === "buy";
    const body = {
      Password: this.tradePassword,
      Symbol: String(code),
      Exchange: exchange,        // 1=東証
      SecurityType: 1,           // 1=株式
      Side: isBuy ? "2" : "1",   // "2"=買, "1"=売
      CashMargin: 1,             // 1=現物
      DelivType: isBuy ? 2 : 0,  // 現物買=2(お預り金), 現物売=0
      FundType: isBuy ? "AA" : "  ", // 現物買="AA"(信用代用), 現物売=空白2文字
      AccountType: accountType,  // 2=一般, 4=特定, 12=法人
      Qty: qty,
      FrontOrderType: 10,        // 10=成行
      Price: 0,                  // 成行は0
      ExpireDay: 0,              // 0=当日
    };
    const r = await this._req("POST", "/sendorder", { body });
    if (r.Result !== 0 && r.Result != null) {
      throw new Error("発注エラー: " + JSON.stringify(r));
    }
    return { orderId: r.OrderId, raw: r };
  }

  name() { return `kabuステーションAPI(${this.base})`; }
}

module.exports = KabusBroker;
