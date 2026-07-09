/* ==========================================================================
   app.js — メイン制御
   ========================================================================== */

(() => {
  const $ = sel => document.querySelector(sel);
  const state = {
    code: null,
    period: "1Y",
    showMA25: true,
    showMA75: false,
    showEvents: true,
    showTopix: false,
    marketFilter: "all",
  };

  const GROUPS = [
    { key: "jp", label: "🇯🇵 日経225 銘柄" },
    { key: "etf", label: "📦 ETF" },
    { key: "us", label: "🇺🇸 米国株 (S&P500 / NASDAQ)" },
  ];
  const PERIOD_DAYS = { "1M": 22, "3M": 66, "6M": 132, "1Y": 100000 };

  /* ---------- テーマ ---------- */
  function initTheme() {
    const saved = localStorage.getItem("stocklens-theme");
    const prefers = window.matchMedia("(prefers-color-scheme: dark)").matches;
    setTheme(saved || (prefers ? "dark" : "light"));
    $("#themeToggle").addEventListener("click", () => {
      setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
    });
  }
  function setTheme(t) {
    document.documentElement.dataset.theme = t;
    $("#themeToggle").textContent = t === "dark" ? "☀️" : "🌙";
    localStorage.setItem("stocklens-theme", t);
    if (state.code) renderChart();
  }

  /* ---------- サイドバー ---------- */
  function renderSidebar(filter = "") {
    const list = $("#symbolList");
    list.innerHTML = "";
    const f = filter.trim().toLowerCase();
    for (const g of GROUPS) {
      const codes = Object.keys(STOCK_DATA.symbols).filter(c => {
        const s = STOCK_DATA.symbols[c];
        if (s.group !== g.key) return false;
        if (!f) return true;
        return (s.name + c + (s.aliases || []).join("")).toLowerCase().includes(f);
      });
      if (!codes.length) continue;
      const label = document.createElement("div");
      label.className = "symbol-group-label";
      label.textContent = g.label;
      list.appendChild(label);
      for (const code of codes) {
        const s = STOCK_DATA.symbols[code];
        const last = s.prices[s.prices.length - 1];
        const prev = s.prices[s.prices.length - 2];
        const chg = ((last.close - prev.close) / prev.close) * 100;
        const btn = document.createElement("button");
        btn.className = "symbol-item" + (code === state.code ? " active" : "");
        btn.dataset.code = code;
        btn.innerHTML = `
          <span><span class="s-name">${s.name}</span><span class="s-code">${code} · ${s.market}</span></span>
          <span class="s-right">
            <span class="s-price">${StockChart.fmtNum(last.close, s.currency)}</span>
            <span class="s-chg ${chg >= 0 ? "up" : "down"}">${chg >= 0 ? "▲" : "▼"}${Math.abs(chg).toFixed(2)}%</span>
          </span>`;
        btn.addEventListener("click", () => selectSymbol(code));
        list.appendChild(btn);
      }
    }
  }

  /* ---------- 銘柄選択 ---------- */
  function selectSymbol(code) {
    state.code = code;
    document.querySelectorAll(".symbol-item").forEach(b =>
      b.classList.toggle("active", b.dataset.code === code));
    renderHeader();
    renderChart();
    renderNews();
    renderAnalysis();
    if ($("#panel-factor").classList.contains("active")) renderFactor();
    if ($("#panel-strategy").classList.contains("active")) renderStrategy();
  }

  function renderHeader() {
    const s = STOCK_DATA.symbols[state.code];
    const last = s.prices[s.prices.length - 1];
    const prev = s.prices[s.prices.length - 2];
    const chg = last.close - prev.close;
    const pct = (chg / prev.close) * 100;
    const jq = s.dataSource === "jquants";
    $("#stockName").innerHTML = `${s.name} <span class="src-badge ${jq ? "real" : ""}">${jq ? "📡 J-Quants実データ" : "📝 参考値"}</span>`;
    $("#stockMeta").textContent = `${state.code} · ${s.marketName || s.market}${s.sector ? " · " + s.sector : ""} · 最終データ: ${StockChart.fmtDate(last.date)}`;
    $("#chartNote").textContent = jq
      ? "株価・出来高はJ-Quants API(JPX公式)の調整済み実データです。イベントマーカー ● をクリックすると関連情報を表示します。"
      : "出来高・株価は収集データに基づく参考値です。イベントマーカー ● をクリックすると関連情報を表示します。";
    $("#stockPrice").textContent = StockChart.fmtNum(last.close, s.currency);
    const chEl = $("#stockChange");
    chEl.textContent = `${chg >= 0 ? "+" : "-"}${StockChart.fmtNum(Math.abs(chg), s.currency)} (${pct >= 0 ? "+" : ""}${pct.toFixed(2)}%) 前日比`;
    chEl.className = "stock-change " + (chg >= 0 ? "up" : "down");
  }

  /* ---------- チャート ---------- */
  function renderChart() {
    const s = STOCK_DATA.symbols[state.code];
    const days = PERIOD_DAYS[state.period];
    const data = s.prices.slice(-days);
    const topixAvailable = !!(STOCK_DATA.topix && s.market === "JP");
    $("#topixToggleWrap").hidden = !topixAvailable;
    StockChart.renderPriceChart($("#priceChart"), $("#volumeChart"), $("#chartTooltip"), {
      data,
      fullData: s.prices,
      events: s.events || [],
      currency: s.currency,
      showMA25: state.showMA25,
      showMA75: state.showMA75,
      showEvents: state.showEvents,
      overlay: topixAvailable && state.showTopix ? { series: STOCK_DATA.topix } : null,
      onEventClick: showEventModal,
    });
    renderLegend(topixAvailable && state.showTopix);
  }

  function renderLegend(showTopix = false) {
    const items = [`<span class="legend-item"><span class="legend-swatch" style="background:var(--series-1)"></span>終値</span>`];
    if (showTopix) items.push(`<span class="legend-item"><span class="legend-swatch" style="background:var(--series-2)"></span>TOPIX(表示起点で正規化・破線)</span>`);
    if (state.showMA25) items.push(`<span class="legend-item"><span class="legend-swatch" style="background:var(--series-3)"></span>MA25(25日移動平均)</span>`);
    if (state.showMA75) items.push(`<span class="legend-item"><span class="legend-swatch" style="background:var(--series-5)"></span>MA75(75日移動平均)</span>`);
    if (state.showEvents) items.push(`<span class="legend-item"><span class="legend-dot" style="background:var(--up)"></span>好材料 <span class="legend-dot" style="background:var(--down)"></span>悪材料 <span class="legend-dot" style="background:var(--ink-muted)"></span>中立イベント</span>`);
    $("#chartLegend").innerHTML = items.join("");
  }

  /* ---------- イベントモーダル ---------- */
  function showEventModal(ev) {
    $("#modalBody").innerHTML = `
      <h3>${impactIcon(ev.impact)} ${ev.title}</h3>
      <div class="m-date">${StockChart.fmtDate(ev.date)}${ev.magnitude ? " · " + ev.magnitude : ""}</div>
      <div class="m-body">${ev.summary}</div>
      ${ev.source ? `<div class="m-src">情報源: ${ev.source}(LLMクローラー収集・要約)</div>` : ""}`;
    $("#modalBackdrop").hidden = false;
  }
  function impactIcon(i) { return i === "up" ? "📈" : i === "down" ? "📉" : "📌"; }

  /* ---------- ニュースタイムライン ---------- */
  function renderNews() {
    const s = STOCK_DATA.symbols[state.code];
    const evs = [...(s.events || [])].sort((a, b) => b.date.localeCompare(a.date));
    $("#newsCount").textContent = `${evs.length}件`;
    $("#newsTimeline").innerHTML = evs.length ? evs.map(ev => `
      <div class="tl-item ${ev.impact || ""}">
        <div class="tl-date">${StockChart.fmtDate(ev.date)}</div>
        <div class="tl-title">${ev.title}</div>
        <div class="tl-summary">${ev.summary}</div>
        <div class="tl-tags">
          <span class="tl-tag impact-${ev.impact}">${ev.impact === "up" ? "📈 好材料" : ev.impact === "down" ? "📉 悪材料" : "📌 中立"}</span>
          ${ev.source ? `<span class="tl-tag">出典: ${ev.source}</span>` : ""}
        </div>
      </div>`).join("")
      : '<p style="color:var(--ink-muted)">この銘柄のニュースはまだ収集されていません。</p>';
  }

  /* ---------- 市場全体タイムライン ---------- */
  function renderMarket() {
    const evs = [...(STOCK_DATA.marketEvents || [])]
      .filter(ev => state.marketFilter === "all" || ev.market === state.marketFilter || ev.market === "BOTH")
      .sort((a, b) => b.date.localeCompare(a.date));
    $("#marketTimeline").innerHTML = evs.map(ev => `
      <div class="tl-item ${ev.impact || ""}">
        <div class="tl-date">${StockChart.fmtDate(ev.date)} ${ev.market === "JP" ? "🇯🇵" : ev.market === "US" ? "🇺🇸" : "🌐"}</div>
        <div class="tl-title">${ev.title}</div>
        <div class="tl-summary">${ev.summary}</div>
        <div class="tl-tags">
          ${ev.magnitude ? `<span class="tl-tag impact-${ev.impact}">${ev.magnitude}</span>` : ""}
          ${ev.source ? `<span class="tl-tag">出典: ${ev.source}</span>` : ""}
        </div>
      </div>`).join("");
  }

  /* ---------- 財務分析 ---------- */
  function renderAnalysis() {
    const s = STOCK_DATA.symbols[state.code];
    const f = s.financials || {};

    // 指標タイル
    const tiles = [];
    const isEtf = s.type === "etf";
    if (isEtf) {
      addTile(tiles, "連動指数", s.index || "—", "", "");
      addTile(tiles, "経費率", f.expenseRatio, "%", "低いほど低コスト");
      addTile(tiles, "分配金利回り", f.distributionYield, "%", "");
      addTile(tiles, "52週高値", f.week52High != null ? StockChart.fmtNum(f.week52High, s.currency) : null, "", "");
      addTile(tiles, "52週安値", f.week52Low != null ? StockChart.fmtNum(f.week52Low, s.currency) : null, "", "");
    } else {
      addTile(tiles, "PER", f.per, "倍", "株価収益率・低いほど割安", f.per != null && f.per < (s.market === "US" ? 20 : 13) ? "good" : "");
      addTile(tiles, "PBR", f.pbr, "倍", "株価純資産倍率", f.pbr != null && f.pbr < 1 ? "good" : "");
      addTile(tiles, "ROE", f.roe, "%", "自己資本利益率・高いほど◎", f.roe != null && f.roe >= 12 ? "good" : f.roe != null && f.roe < 5 ? "bad" : "");
      addTile(tiles, "配当利回り", f.dividendYield, "%", "", f.dividendYield != null && f.dividendYield >= 3 ? "good" : "");
      addTile(tiles, "自己資本比率", f.equityRatio, "%", "高いほど財務健全", f.equityRatio != null && f.equityRatio >= 50 ? "good" : "");
      addTile(tiles, "時価総額", f.marketCap, "", "");
      if (f.revenue != null) addTile(tiles, "売上高", fmtFin(f.revenue, s), "", `${f.asOf || ""}時点`);
      if (f.netIncome != null) addTile(tiles, "純利益", fmtFin(f.netIncome, s), "", "");
      if (f.operatingCF != null) addTile(tiles, "営業CF", fmtFin(f.operatingCF, s), "", "本業の現金創出力", f.operatingCF > 0 ? "good" : "bad");
    }
    $("#metricTiles").innerHTML = tiles.join("");

    // レーダー
    const scores = Analysis.computeScores(s);
    const radarEl = $("#radarChart");
    if (scores) {
      StockChart.renderRadar(radarEl, scores);
    } else {
      radarEl.innerHTML = '<p style="color:var(--ink-muted);padding:30px;text-align:center;font-size:.8rem">ETFはインデックス連動型のため<br>個別財務スコアは対象外です</p>';
    }

    // J-Quants拡張データ: 信用残トレンド(Standard以上)
    const mi = s.marginInterest;
    $("#marginCard").hidden = !(mi && mi.length >= 2);
    if (mi && mi.length >= 2) {
      const fmtShares = v => v >= 1e6 ? (v / 1e6).toFixed(1) + "百万株" : Math.round(v / 1e4) + "万株";
      StockChart.renderMiniLines($("#marginChart"), mi.map(m => m.date), [
        { label: "信用買残", cssVar: "--series-1", values: mi.map(m => m.long) },
        { label: "信用売残", cssVar: "--series-2", values: mi.map(m => m.short) },
      ], { fmt: fmtShares });
      const last = mi[mi.length - 1];
      const ratio = last.short > 0 ? (last.long / last.short).toFixed(2) : "—";
      $("#marginLegend").innerHTML =
        `<span class="legend-item"><span class="legend-swatch" style="background:var(--series-1)"></span>信用買残 ${fmtShares(last.long)}</span>` +
        `<span class="legend-item"><span class="legend-swatch" style="background:var(--series-2)"></span>信用売残 ${fmtShares(last.short)}</span>` +
        `<span class="legend-item">信用倍率 ${ratio}倍</span>`;
    }

    // J-Quants拡張データ: 配当推移(Premium)
    const divs = s.dividends;
    $("#dividendCard").hidden = !(divs && divs.length);
    if (divs && divs.length) {
      const maxD = Math.max(...divs.map(d => d.dividend)) || 1;
      $("#dividendChart").innerHTML = divs.map(d => {
        const [yy, mm] = d.recordDate.split("-");
        return `<div class="div-bar ${d.forecast ? "forecast" : ""}" title="${d.recordDate} ${d.term}${d.forecast ? "(予想)" : ""}">
          <span class="db-val">${d.dividend}</span>
          <span class="db-fill" style="height:${Math.max(4, (d.dividend / maxD) * 100)}%"></span>
          <span class="db-label">${yy.slice(2)}/${Number(mm)}<br>${d.term}${d.forecast ? "予" : ""}</span>
        </div>`;
      }).join("");
    }

    // アドバイス
    const adv = Analysis.buildAdvice(s);
    $("#adviceBox").innerHTML =
      `<div class="advice-verdict">${adv.icon} ${adv.verdict}</div>` +
      adv.sections.map(sec => `<div class="advice-section"><h4>${sec.h}</h4><p>${sec.p}</p></div>`).join("");
  }

  function addTile(arr, label, value, unit, hint, cls = "") {
    arr.push(`<div class="metric-tile ${cls}">
      <div class="m-label">${label}</div>
      <div class="m-value">${value != null ? value : "—"}<span class="m-unit">${value != null ? unit : ""}</span></div>
      ${hint ? `<div class="m-hint">${hint}</div>` : ""}
    </div>`);
  }
  function fmtFin(v, s) {
    return s.market === "US" ? `$${v}B` : `${v.toLocaleString()}億円`;
  }

  /* ---------- ファクター分析 ---------- */
  const FACTOR_AXES = { x: "value", y: "momentum", rank: "momentum" };

  function initFactor() {
    const optHtml = Factor.FACTORS.map(f => `<option value="${f.key}">${f.label}</option>`).join("");
    for (const [id, key] of [["factorX", FACTOR_AXES.x], ["factorY", FACTOR_AXES.y], ["factorRank", FACTOR_AXES.rank]]) {
      const sel = $("#" + id);
      sel.innerHTML = optHtml;
      sel.value = key;
    }
    $("#factorX").addEventListener("change", e => { FACTOR_AXES.x = e.target.value; renderFactor(); });
    $("#factorY").addEventListener("change", e => { FACTOR_AXES.y = e.target.value; renderFactor(); });
    $("#factorRank").addEventListener("change", e => { FACTOR_AXES.rank = e.target.value; renderFactor(); });
  }

  function renderFactor() {
    const s = STOCK_DATA.symbols[state.code];
    const isEtf = s.type === "etf";
    $("#factorStockName").textContent = s.name;

    /* プロファイル(横バー) */
    const profileEl = $("#factorProfile");
    const styleEl = $("#factorStyle");
    if (isEtf) {
      profileEl.innerHTML = '<p style="color:var(--ink-muted);font-size:.8rem">ETFはインデックス連動型のためファクター分析の対象外です。個別株を選択してください。</p>';
      styleEl.innerHTML = `<p style="font-size:.85rem;color:var(--ink-2)">📦 このETFは<b>${s.index || "指数"}</b>への分散投資商品です。下のファクターマップで構成市場の個別銘柄の位置づけを確認できます。</p>`;
    } else {
      const style = Factor.classifyStyle(state.code);
      profileEl.innerHTML = Factor.FACTORS.map(f => {
        const v = style.scores[f.key];
        const w = v == null ? 0 : ((Math.max(20, Math.min(80, v)) - 20) / 60) * 100;
        const strong = v != null && v >= 58, weak = v != null && v <= 42;
        return `<div class="fbar-row" title="${f.desc}">
          <span class="fbar-label">${f.label}</span>
          <span class="fbar-track"><span class="fbar-mid"></span><span class="fbar-fill ${strong ? "strong" : weak ? "weak" : ""}" style="width:${w}%"></span></span>
          <span class="fbar-value ${strong ? "strong" : weak ? "weak" : ""}">${v ?? "—"}</span>
        </div>`;
      }).join("");
      styleEl.innerHTML = `
        <div class="style-tags">${style.tags.map(t => `<span class="style-tag">${t}</span>`).join("")}</div>
        <p style="font-size:.83rem;color:var(--ink-2);margin-top:10px">${style.comment}</p>
        <p style="font-size:.72rem;color:var(--ink-muted);margin-top:8px">偏差値58以上を「強い」、42以下を「弱い」として判定しています。</p>`;
    }

    /* 散布図 */
    const all = Factor.computeAll();
    const fx = Factor.FACTORS.find(f => f.key === FACTOR_AXES.x);
    const fy = Factor.FACTORS.find(f => f.key === FACTOR_AXES.y);
    const points = Object.values(all).map(p => ({ code: p.code, name: p.name, x: p[FACTOR_AXES.x], y: p[FACTOR_AXES.y] }));
    StockChart.renderScatter($("#factorScatter"), points, {
      xLabel: fx.label, yLabel: fy.label,
      highlightCode: isEtf ? null : state.code,
      onPointClick: selectSymbol,
    });

    /* ランキング */
    const rank = Factor.ranking(FACTOR_AXES.rank);
    const fr = Factor.FACTORS.find(f => f.key === FACTOR_AXES.rank);
    $("#factorRanking").innerHTML = `
      <table class="rank-table">
        <thead><tr><th>#</th><th>銘柄</th><th>${fr.label}</th><th>偏差値バー</th></tr></thead>
        <tbody>${rank.map((r, i) => `
          <tr class="${r.code === state.code ? "current" : ""}" data-code="${r.code}">
            <td>${i + 1}</td>
            <td>${r.name}<span class="rank-code">${r.code}</span></td>
            <td class="rank-val">${r[FACTOR_AXES.rank]}</td>
            <td><span class="rank-bar"><span style="width:${((r[FACTOR_AXES.rank] - 20) / 60) * 100}%"></span></span></td>
          </tr>`).join("")}
        </tbody>
      </table>`;
    document.querySelectorAll(".rank-table tbody tr").forEach(tr =>
      tr.addEventListener("click", () => selectSymbol(tr.dataset.code)));
  }

  /* ---------- 戦略・売買シグナル ---------- */
  const stratState = { key: "maCross", params: {} };

  function initStrategy() {
    const sel = $("#stratSelect");
    sel.innerHTML = Object.entries(Strategy.STRATEGIES)
      .map(([k, s]) => `<option value="${k}">${s.label}</option>`).join("");
    sel.value = stratState.key;
    sel.addEventListener("change", e => { stratState.key = e.target.value; stratState.params = {}; renderStratParams(); renderStrategy(); });
    renderStratParams();
    $("#paperExec").addEventListener("click", paperExecute);
    $("#paperReset").addEventListener("click", () => {
      if (confirm("この銘柄のペーパー口座をリセットしますか?")) { Paper.reset(state.code); renderStrategy(); }
    });
  }

  function renderStratParams() {
    const strat = Strategy.STRATEGIES[stratState.key];
    $("#stratDesc").textContent = strat.desc;
    $("#stratParams").innerHTML = strat.params.map(p => {
      const val = stratState.params[p.key] ?? p.default;
      return `<label class="strat-param">${p.label}
        <input type="number" data-pkey="${p.key}" value="${val}" min="${p.min}" max="${p.max}">
      </label>`;
    }).join("");
    document.querySelectorAll("#stratParams input").forEach(inp =>
      inp.addEventListener("change", e => {
        stratState.params[e.target.dataset.pkey] = Number(e.target.value);
        renderStrategy();
      }));
  }

  function renderStrategy() {
    const s = STOCK_DATA.symbols[state.code];
    const prices = s.prices;
    const signals = Strategy.generateSignals(prices, stratState.key, stratState.params);
    const cur = Strategy.currentSignal(signals);
    const last = prices[prices.length - 1];

    /* 現在のシグナル */
    renderSignalBox(s, signals, cur, last);

    /* バックテスト */
    const bt = Backtest.run(signals, { initialCash: Paper.INITIAL_CASH });
    renderBacktest(s, bt);

    /* ペーパートレード */
    renderPaper(s, last, cur, signals);
  }

  function renderSignalBox(s, signals, cur, last) {
    // 直近バーのシグナル(当日の新規シグナル)と、最後に出たシグナル
    const todaySig = signals[signals.length - 1].signal;
    const ind = signals[signals.length - 1].indicators || {};
    let indText = "";
    if (ind.short != null) indText = `短期MA ${Math.round(ind.short).toLocaleString()} / 長期MA ${Math.round(ind.long).toLocaleString()}`;
    else if (ind.rsi != null) indText = `RSI ${ind.rsi.toFixed(1)}`;

    let verdict, icon, cls, detail;
    if (todaySig === "buy") { verdict = "買いシグナル"; icon = "🟢"; cls = "buy"; detail = "最新データで買いシグナルが点灯しました。"; }
    else if (todaySig === "sell") { verdict = "売りシグナル"; icon = "🔴"; cls = "sell"; detail = "最新データで売りシグナルが点灯しました。"; }
    else if (cur && cur.signal === "buy") { verdict = "買い継続(ホールド)"; icon = "🔵"; cls = "hold"; detail = `直近の買いシグナル(${StockChart.fmtDate(cur.date)})以降、保有継続のゾーンです。`; }
    else if (cur && cur.signal === "sell") { verdict = "様子見(ノーポジション)"; icon = "⚪"; cls = "hold"; detail = `直近の売りシグナル(${StockChart.fmtDate(cur.date)})以降、待機のゾーンです。`; }
    else { verdict = "シグナルなし"; icon = "⚪"; cls = "hold"; detail = "まだ売買シグナルが発生していません。"; }

    $("#signalBox").innerHTML = `
      <div class="signal-verdict ${cls}">${icon} ${verdict}</div>
      <p class="signal-detail">${detail}</p>
      <div class="signal-meta">
        <span>銘柄: <b>${s.name}</b></span>
        <span>最新終値: <b>${StockChart.fmtNum(last.close, s.currency)}</b>(${StockChart.fmtDate(last.date)})</span>
        ${indText ? `<span>指標: <b>${indText}</b></span>` : ""}
      </div>`;
  }

  function renderBacktest(s, bt) {
    if (!bt) { $("#backtestTiles").innerHTML = "<p style='color:var(--ink-muted)'>データ不足</p>"; return; }
    const m = bt.metrics;
    const pct = v => (v == null ? "—" : (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "%");
    const tiles = [];
    const beatBH = m.totalReturn > m.buyHold;
    addTile(tiles, "戦略リターン", pct(m.totalReturn), "", "初期資金比", m.totalReturn >= 0 ? "good" : "bad");
    addTile(tiles, "買い持ち比較", pct(m.buyHold), "", "同期間の単純保有", "");
    addTile(tiles, "対買い持ち", (beatBH ? "勝ち " : "負け ") + pct(m.totalReturn - m.buyHold), "", "超過リターン", beatBH ? "good" : "bad");
    addTile(tiles, "最大ドローダウン", pct(m.maxDrawdown), "", "最大の下落幅", "bad");
    addTile(tiles, "勝率", m.winRate == null ? "—" : (m.winRate * 100).toFixed(0) + "%", "", `${m.numTrades}回の決済`, m.winRate != null && m.winRate >= 0.5 ? "good" : "");
    addTile(tiles, "シャープレシオ", m.sharpe == null ? "—" : m.sharpe.toFixed(2), "", "リスク調整後(年率)", m.sharpe != null && m.sharpe >= 1 ? "good" : "");
    $("#backtestTiles").innerHTML = tiles.join("");

    /* 資産推移(戦略 vs 買い持ち、初期資金=100に正規化) */
    const base = m.initialCash;
    const first = bt.equity[0].value;
    const stratIdx = bt.equity.map(e => (e.value / base) * 100);
    const firstClose = s.prices[s.prices.length - bt.equity.length].close;
    const bhIdx = bt.equity.map((e, i) => {
      const price = s.prices[s.prices.length - bt.equity.length + i].close;
      return (price / firstClose) * 100;
    });
    StockChart.renderMiniLines($("#equityChart"), bt.equity.map(e => e.date), [
      { label: "戦略", cssVar: "--series-1", values: stratIdx },
      { label: "買い持ち", cssVar: "--series-3", values: bhIdx },
    ], { fmt: v => v.toFixed(0) });
    $("#equityLegend").innerHTML =
      `<span class="legend-item"><span class="legend-swatch" style="background:var(--series-1)"></span>戦略(初期100)</span>` +
      `<span class="legend-item"><span class="legend-swatch" style="background:var(--series-3)"></span>買い持ち(初期100)</span>`;

    /* 売買履歴 */
    const rows = bt.trades.slice().reverse();
    let html = `<table class="rank-table"><thead><tr><th>建玉</th><th>手仕舞い</th><th>損益率</th></tr></thead><tbody>`;
    if (bt.openTrade) {
      html += `<tr><td>${StockChart.fmtDate(bt.openTrade.entryDate)}<span class="rank-code">建玉中</span></td><td>—(時価)</td><td class="${bt.openTrade.return >= 0 ? "pl-up" : "pl-down"}">${pct(bt.openTrade.return)}</td></tr>`;
    }
    html += rows.map(t =>
      `<tr><td>${StockChart.fmtDate(t.entryDate)}</td><td>${StockChart.fmtDate(t.exitDate)}</td><td class="${t.return >= 0 ? "pl-up" : "pl-down"}">${pct(t.return)}</td></tr>`
    ).join("");
    if (!rows.length && !bt.openTrade) html += `<tr><td colspan="3" style="color:var(--ink-muted)">この期間に売買はありませんでした</td></tr>`;
    html += "</tbody></table>";
    $("#tradeTable").innerHTML = html;
  }

  function renderPaper(s, last, cur, signals) {
    const sum = Paper.summary(state.code, last.close);
    const yen = v => "¥" + Math.round(v).toLocaleString();
    const pct = v => (v >= 0 ? "+" : "") + (v * 100).toFixed(2) + "%";
    const todaySig = signals[signals.length - 1].signal;

    $("#paperBox").innerHTML = `
      <div class="paper-value ${sum.totalReturn >= 0 ? "up" : "down"}">
        <span class="pv-label">総資産</span>
        <span class="pv-num">${yen(sum.totalValue)}</span>
        <span class="pv-ret">${pct(sum.totalReturn)}</span>
      </div>
      <div class="paper-rows">
        <div><span>現金</span><b>${yen(sum.cash)}</b></div>
        <div><span>建玉</span><b>${sum.shares > 0 ? Math.round(sum.shares).toLocaleString() + "株 @ " + StockChart.fmtNum(sum.avgPrice, s.currency) : "なし"}</b></div>
        <div><span>評価損益(含み)</span><b class="${sum.unrealizedPnL >= 0 ? "pl-up" : "pl-down"}">${sum.shares > 0 ? yen(sum.unrealizedPnL) : "—"}</b></div>
        <div><span>実現損益</span><b class="${sum.realizedPnL >= 0 ? "pl-up" : "pl-down"}">${yen(sum.realizedPnL)}</b></div>
      </div>`;

    // 執行ボタンの状態(全額イン/アウトモデル)
    const btn = $("#paperExec");
    let canExec = false, btnLabel = "現在のシグナルで執行";
    if (todaySig === "buy" && sum.shares === 0) { canExec = true; btnLabel = `🟢 買い執行(${yen(sum.cash)}分)`; }
    else if (todaySig === "sell" && sum.shares > 0) { canExec = true; btnLabel = "🔴 売り執行(全株)"; }
    else if (sum.shares === 0 && cur && cur.signal === "buy") { canExec = true; btnLabel = "🟢 直近買いシグナルで新規建玉"; }
    else if (sum.shares > 0 && cur && cur.signal === "sell") { canExec = true; btnLabel = "🔴 直近売りシグナルで手仕舞い"; }
    else btnLabel = sum.shares > 0 ? "売りシグナル待ち(保有中)" : "買いシグナル待ち";
    btn.textContent = btnLabel;
    btn.disabled = !canExec;

    // ログ
    $("#paperLog").innerHTML = sum.log.length
      ? `<table class="rank-table"><thead><tr><th>日付</th><th>売買</th><th>価格</th><th>株数</th><th>損益</th></tr></thead><tbody>` +
        sum.log.map(l => `<tr><td>${StockChart.fmtDate(l.date)}</td><td>${l.action}</td><td>${StockChart.fmtNum(l.price, s.currency)}</td><td>${l.shares.toLocaleString()}</td><td class="${l.pnl == null ? "" : l.pnl >= 0 ? "pl-up" : "pl-down"}">${l.pnl == null ? "—" : yen(l.pnl)}</td></tr>`).join("") +
        `</tbody></table>`
      : `<p style="color:var(--ink-muted);font-size:.78rem">まだ取引履歴はありません。シグナルが出たら「執行」で仮想売買を記録できます。</p>`;
  }

  function paperExecute() {
    const s = STOCK_DATA.symbols[state.code];
    const last = s.prices[s.prices.length - 1];
    const signals = Strategy.generateSignals(s.prices, stratState.key, stratState.params);
    const cur = Strategy.currentSignal(signals);
    const todaySig = signals[signals.length - 1].signal;
    const sum = Paper.summary(state.code, last.close);

    let action = null;
    if ((todaySig === "buy" || (cur && cur.signal === "buy")) && sum.shares === 0) action = "buy";
    else if ((todaySig === "sell" || (cur && cur.signal === "sell")) && sum.shares > 0) action = "sell";
    if (!action) return;

    const res = Paper.execute(state.code, action, last.close, last.date,
      `${Strategy.STRATEGIES[stratState.key].label}`);
    if (res.error) { alert(res.error); return; }
    renderStrategy();
  }

  /* ---------- タブ ---------- */
  function initTabs() {
    document.querySelectorAll(".tab").forEach(tab => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
        tab.classList.add("active");
        $("#panel-" + tab.dataset.tab).classList.add("active");
        if (tab.dataset.tab === "market") renderMarket();
        if (tab.dataset.tab === "factor") renderFactor();
        if (tab.dataset.tab === "strategy") renderStrategy();
      });
    });
    document.querySelectorAll("#marketFilter .chip").forEach(chip => {
      chip.addEventListener("click", () => {
        document.querySelectorAll("#marketFilter .chip").forEach(c => c.classList.remove("active"));
        chip.classList.add("active");
        state.marketFilter = chip.dataset.market;
        renderMarket();
      });
    });
  }

  /* ---------- チャートコントロール ---------- */
  function initChartControls() {
    document.querySelectorAll("#periodBtns button").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll("#periodBtns button").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.period = btn.dataset.period;
        renderChart();
      });
    });
    $("#toggleMA25").addEventListener("change", e => { state.showMA25 = e.target.checked; renderChart(); });
    $("#toggleMA75").addEventListener("change", e => { state.showMA75 = e.target.checked; renderChart(); });
    $("#toggleEvents").addEventListener("change", e => { state.showEvents = e.target.checked; renderChart(); });
    $("#toggleTopix").addEventListener("change", e => { state.showTopix = e.target.checked; renderChart(); });
    let resizeTimer;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (!state.code) return;
        renderChart();
        if ($("#panel-factor").classList.contains("active")) renderFactor();
      }, 150);
    });
  }

  /* ---------- Q&A ---------- */
  const SUGGESTS = [
    "2025年12月3日ごろに株価が大きく変動しているけどこの理由は何?",
    "この銘柄は割安?",
    "この銘柄のファクター分析をして",
    "モメンタムが強い銘柄は?",
    "最近のニュースを教えて",
    "見解・アドバイスを教えて",
  ];

  function initQA() {
    const panel = $("#qaPanel"), backdrop = $("#qaBackdrop");
    const open = () => { panel.classList.add("open"); panel.setAttribute("aria-hidden", "false"); backdrop.hidden = false; $("#qaText").focus(); };
    const close = () => { panel.classList.remove("open"); panel.setAttribute("aria-hidden", "true"); backdrop.hidden = true; };
    $("#qaFab").addEventListener("click", open);
    $("#qaClose").addEventListener("click", close);
    backdrop.addEventListener("click", close);

    $("#qaSuggests").innerHTML = SUGGESTS.map(s => `<button class="qa-suggest">${s}</button>`).join("");
    document.querySelectorAll(".qa-suggest").forEach(b =>
      b.addEventListener("click", () => ask(b.textContent)));

    $("#qaForm").addEventListener("submit", e => {
      e.preventDefault();
      const q = $("#qaText").value.trim();
      if (q) { ask(q); $("#qaText").value = ""; }
    });

    addBot(`こんにちは!収集済みの株価・ニュース・財務データからご質問にお答えします。<br>下の候補をタップするか、自由に質問してください。`);
  }

  function ask(q) {
    addMsg(q, "user");
    const typing = addTyping();
    setTimeout(() => {
      typing.remove();
      const res = QA.answer(q, state.code);
      addBot(res.html);
    }, 450 + Math.random() * 350);
  }

  function addMsg(html, cls) {
    const div = document.createElement("div");
    div.className = "qa-msg " + cls;
    if (cls === "user") div.textContent = html; else div.innerHTML = html;
    $("#qaMessages").appendChild(div);
    $("#qaMessages").scrollTop = $("#qaMessages").scrollHeight;
    return div;
  }
  const addBot = html => addMsg(html, "bot");
  function addTyping() {
    return addMsg('<span class="typing"><span></span><span></span><span></span></span>', "bot");
  }

  /* ---------- モーダル・検索 ---------- */
  function initMisc() {
    $("#modalClose").addEventListener("click", () => $("#modalBackdrop").hidden = true);
    $("#modalBackdrop").addEventListener("click", e => {
      if (e.target === $("#modalBackdrop")) $("#modalBackdrop").hidden = true;
    });
    document.addEventListener("keydown", e => {
      if (e.key === "Escape") {
        $("#modalBackdrop").hidden = true;
        $("#qaPanel").classList.remove("open");
        $("#qaBackdrop").hidden = true;
      }
    });
    $("#symbolSearch").addEventListener("input", e => renderSidebar(e.target.value));
    $("#dataAsOf").textContent = `データ収集: ${STOCK_DATA.meta.collectedAt}`;
  }

  /* ---------- 起動 ---------- */
  initTheme();
  initTabs();
  initFactor();
  initStrategy();
  initChartControls();
  initQA();
  initMisc();
  renderSidebar();
  selectSymbol(STOCK_DATA.meta.defaultSymbol || Object.keys(STOCK_DATA.symbols)[0]);
})();
