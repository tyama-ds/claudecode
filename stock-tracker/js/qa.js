/* ==========================================================================
   qa.js — Q&Aエンジン
   収集済みデータ(銘柄イベント・市場イベント・財務・株価)から
   日本語の質問に回答する。
   ========================================================================== */

const QA = (() => {

  /* ---------- 日付表現の解析 ---------- */
  // "2025年12月3日", "2025/12/03", "12月3日ごろ", "2025年12月上旬" などに対応
  function parseDates(q) {
    const results = [];
    let m;

    // YYYY年M月D日 / YYYY/M/D / YYYY-M-D
    const reFull = /(\d{4})[年\/\-](\d{1,2})[月\/\-](\d{1,2})日?/g;
    while ((m = reFull.exec(q)) !== null) {
      results.push({ date: iso(m[1], m[2], m[3]), span: 5 });
    }
    // YYYY年M月上旬/中旬/下旬
    const rePart = /(\d{4})年(\d{1,2})月(上旬|中旬|下旬)/g;
    while ((m = rePart.exec(q)) !== null) {
      const day = m[3] === "上旬" ? 5 : m[3] === "中旬" ? 15 : 25;
      results.push({ date: iso(m[1], m[2], day), span: 7 });
    }
    // YYYY年M月(日なし)
    if (!results.length) {
      const reMonth = /(\d{4})年(\d{1,2})月/g;
      while ((m = reMonth.exec(q)) !== null) {
        results.push({ date: iso(m[1], m[2], 15), span: 17 });
      }
    }
    // M月D日(年なし → データ期間内で直近の該当日を推定)
    if (!results.length) {
      const reMD = /(\d{1,2})月(\d{1,2})日/g;
      while ((m = reMD.exec(q)) !== null) {
        const meta = STOCK_DATA.meta;
        const endY = Number(meta.dataEnd.slice(0, 4));
        for (const yy of [endY, endY - 1]) {
          const d = iso(yy, m[1], m[2]);
          if (d >= meta.dataStart && d <= meta.dataEnd) { results.push({ date: d, span: 5 }); break; }
        }
      }
    }
    return results;
  }

  function iso(y, mo, d) {
    return `${y}-${String(mo).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
  }

  function daysBetween(a, b) {
    return Math.abs(new Date(a).getTime() - new Date(b).getTime()) / 86400000;
  }

  /* ---------- 銘柄の解析 ---------- */
  function findSymbols(q) {
    const found = [];
    for (const [code, s] of Object.entries(STOCK_DATA.symbols)) {
      const names = [s.name, ...(s.aliases || []), code];
      if (names.some(n => n && q.includes(n))) found.push(code);
    }
    return found;
  }

  /* ---------- 意図判定 ---------- */
  function detectIntent(q) {
    if (/(なぜ|理由|要因|原因|何があった|どうして|変動|急落|急騰|下がっ|上がっ|下落|上昇|動い)/.test(q) && parseDates(q).length) return "why-move";
    if (/(なぜ|理由|要因|原因)/.test(q)) return "why-general";
    if (/(ファクター|バリュー|グロース|モメンタム|ボラティリティ|クオリティ|スタイル|偏差値)/.test(q)) return "factor";
    if (/(いくら|株価は|価格|終値)/.test(q)) return "price";
    if (/(PER|PBR|ROE|配当|利回り|財務|指標|割安|割高)/i.test(q)) return "financials";
    if (/(買い|売り|おすすめ|投資すべき|どう思う|見解|評価|アドバイス)/.test(q)) return "advice";
    if (/(ニュース|イベント|情報|最近|何があ)/.test(q)) return "news";
    if (/(比較|どっち|どちら)/.test(q)) return "compare";
    if (parseDates(q).length) return "why-move";
    return "unknown";
  }

  /* ---------- イベント検索 ---------- */
  function eventsNear(date, span, symbolCode = null) {
    const hits = [];
    // 銘柄イベント
    const symbols = symbolCode ? [symbolCode] : Object.keys(STOCK_DATA.symbols);
    for (const code of symbols) {
      const s = STOCK_DATA.symbols[code];
      for (const ev of (s.events || [])) {
        const d = daysBetween(ev.date, date);
        if (d <= span) hits.push({ ...ev, symbol: s.name, code, dist: d, scope: "symbol" });
      }
    }
    // 市場イベント
    for (const ev of (STOCK_DATA.marketEvents || [])) {
      const d = daysBetween(ev.date, date);
      if (d <= span) hits.push({ ...ev, dist: d, scope: "market" });
    }
    hits.sort((a, b) => a.dist - b.dist || (b.scope === "market" ? 1 : -1));
    return hits;
  }

  function priceMoveAround(code, date, span) {
    const s = STOCK_DATA.symbols[code];
    if (!s || !s.prices) return null;
    const t = new Date(date).getTime();
    const win = s.prices.filter(p => Math.abs(new Date(p.date).getTime() - t) <= span * 86400000);
    if (win.length < 2) return null;
    const first = win[0], last = win[win.length - 1];
    const chg = ((last.close - first.close) / first.close) * 100;
    // 期間内最大の1日変動
    let maxDay = null;
    for (let i = 1; i < win.length; i++) {
      const d = ((win[i].close - win[i - 1].close) / win[i - 1].close) * 100;
      if (!maxDay || Math.abs(d) > Math.abs(maxDay.chg)) maxDay = { date: win[i].date, chg: d };
    }
    return { first, last, chg, maxDay, currency: s.currency };
  }

  /* ---------- 回答生成 ---------- */
  function answer(q, currentCode) {
    const intent = detectIntent(q);
    const dates = parseDates(q);
    let codes = findSymbols(q);
    if (!codes.length && currentCode) codes = [currentCode];

    switch (intent) {
      case "why-move": return answerWhyMove(q, dates, codes, currentCode);
      case "price": return answerPrice(dates, codes);
      case "factor": return answerFactor(q, codes, currentCode);
      case "financials": return answerFinancials(codes);
      case "advice": return answerAdvice(codes);
      case "news": return answerNews(codes);
      case "compare": return answerCompare(codes);
      default:
        return {
          html: `<p>すみません、質問をうまく理解できませんでした。以下のような聞き方ができます:</p>
          <ul>
            <li>「<b>2025年12月3日ごろに株価が大きく変動した理由は?</b>」</li>
            <li>「トヨタの最近のニュースは?」</li>
            <li>「NVIDIAのPERは?」「日本製鉄は割安?」</li>
            <li>「ソニーの見解を教えて」</li>
          </ul>`
        };
    }
  }

  function answerWhyMove(q, dates, codes, currentCode) {
    if (!dates.length) {
      return { html: "<p>いつ頃の変動についてお調べしますか?「2025年12月3日ごろ」のように日付を含めて質問してください。</p>" };
    }
    const { date, span } = dates[0];
    const code = codes[0] || currentCode;
    const sym = STOCK_DATA.symbols[code];
    const evs = eventsNear(date, span, code);
    const marketEvs = eventsNear(date, span).filter(e => e.scope === "market");
    const move = code ? priceMoveAround(code, date, Math.max(span, 4)) : null;

    let html = "";
    // 実際の値動き
    if (move && sym) {
      const dir = move.chg >= 0 ? "上昇" : "下落";
      const cls = move.chg >= 0 ? "var(--up)" : "var(--down)";
      html += `<h4>📊 ${sym.name} の値動き(${StockChart.fmtDate(date)}前後)</h4>
        <p>${StockChart.fmtDate(move.first.date)} → ${StockChart.fmtDate(move.last.date)} で
        <b style="color:${cls}">${move.chg >= 0 ? "+" : ""}${move.chg.toFixed(1)}% ${dir}</b>しました。`;
      if (move.maxDay && Math.abs(move.maxDay.chg) >= 1.5) {
        html += ` 特に ${StockChart.fmtDate(move.maxDay.date)} は1日で <b>${move.maxDay.chg >= 0 ? "+" : ""}${move.maxDay.chg.toFixed(1)}%</b> 動いています。`;
      }
      html += "</p>";
    }

    const symbolEvs = evs.filter(e => e.scope === "symbol").slice(0, 3);
    const mktEvs = marketEvs.slice(0, 3);

    if (symbolEvs.length || mktEvs.length) {
      html += `<h4>💡 考えられる要因</h4><ul>`;
      for (const ev of symbolEvs) {
        html += `<li><b>${StockChart.fmtDate(ev.date)}【${ev.symbol}】${ev.title}</b><br>${ev.summary}</li>`;
      }
      for (const ev of mktEvs) {
        html += `<li><b>${StockChart.fmtDate(ev.date)}【市場全体${ev.market === "JP" ? "・日本" : ev.market === "US" ? "・米国" : ""}】${ev.title}</b><br>${ev.summary}${ev.magnitude ? `(${ev.magnitude})` : ""}</li>`;
      }
      html += "</ul>";
      const srcs = [...new Set([...symbolEvs, ...mktEvs].map(e => e.source).filter(Boolean))];
      if (srcs.length) html += `<span class="qa-src">情報源: ${srcs.join(" / ")}(LLMクローラー収集)</span>`;
    } else {
      html += `<p>${StockChart.fmtDate(date)}前後の収集データには、該当するイベントが見つかりませんでした。この期間のニュース収集を依頼していただければデータを更新できます。</p>`;
    }
    return { html };
  }

  function answerPrice(dates, codes) {
    if (!codes.length) return { html: "<p>どの銘柄の株価をお調べしますか?銘柄名を含めて質問してください。</p>" };
    const s = STOCK_DATA.symbols[codes[0]];
    let target = s.prices[s.prices.length - 1];
    let label = "直近";
    if (dates.length) {
      const t = new Date(dates[0].date).getTime();
      target = s.prices.reduce((best, p) =>
        Math.abs(new Date(p.date).getTime() - t) < Math.abs(new Date(best.date).getTime() - t) ? p : best);
      label = StockChart.fmtDate(target.date) + "時点";
    }
    return {
      html: `<p><b>${s.name}</b>(${codes[0]})の${label}の終値は <b>${StockChart.fmtNum(target.close, s.currency)}</b> です。</p>`
    };
  }

  function answerFinancials(codes) {
    if (!codes.length) return { html: "<p>どの銘柄の指標をお調べしますか?</p>" };
    const s = STOCK_DATA.symbols[codes[0]];
    const f = s.financials;
    if (!f) return { html: `<p>${s.name}の財務データは未収集です。</p>` };
    if (s.type === "etf") {
      return { html: `<p><b>${s.name}</b>はETFです。経費率: ${f.expenseRatio != null ? f.expenseRatio + "%" : "—"}、分配金利回り: ${f.distributionYield != null ? f.distributionYield + "%" : "—"}。</p>` };
    }
    const rows = [
      ["PER", f.per, "倍"], ["PBR", f.pbr, "倍"], ["ROE", f.roe, "%"],
      ["配当利回り", f.dividendYield, "%"], ["自己資本比率", f.equityRatio, "%"],
    ].filter(r => r[1] != null);
    let html = `<h4>📊 ${s.name} の主要指標</h4><ul>` +
      rows.map(r => `<li>${r[0]}: <b>${r[1]}${r[2]}</b></li>`).join("") + "</ul>";
    const adv = Analysis.buildAdvice(s);
    const valSec = adv.sections.find(x => x.h.includes("バリュエーション"));
    if (valSec) html += `<p>${valSec.p}</p>`;
    html += `<span class="qa-src">データ時点: ${f.asOf || STOCK_DATA.meta.collectedAt}</span>`;
    return { html };
  }

  function answerAdvice(codes) {
    if (!codes.length) return { html: "<p>どの銘柄についての見解をお求めですか?</p>" };
    const s = STOCK_DATA.symbols[codes[0]];
    const adv = Analysis.buildAdvice(s);
    let html = `<h4>${adv.icon} ${s.name} — ${adv.verdict}</h4>`;
    for (const sec of adv.sections.slice(0, 4)) html += `<p><b>${sec.h}</b><br>${sec.p}</p>`;
    html += `<span class="qa-src">※ 参考見解であり投資助言ではありません。詳細は「財務分析」タブをご覧ください。</span>`;
    return { html };
  }

  function answerFactor(q, codes, currentCode) {
    // 「◯◯が強い銘柄は?」のようなランキング質問
    const rankFactor = Factor.FACTORS.find(f => q.includes(f.label));
    const asksRanking = /(強い銘柄|高い銘柄|上位|ランキング|どの銘柄|一番|トップ)/.test(q);
    if (rankFactor && asksRanking) {
      const top = Factor.ranking(rankFactor.key).slice(0, 3);
      return {
        html: `<h4>🧮 ${rankFactor.label}ファクター 上位銘柄</h4><ul>` +
          top.map((r, i) => `<li>${i + 1}位: <b>${r.name}</b>(${r.code}) — 偏差値 ${r[rankFactor.key]}</li>`).join("") +
          `</ul><p>${rankFactor.desc}を全個別株横断で偏差値化した順位です。詳細は「ファクター分析」タブをご覧ください。</p>`
      };
    }
    // 個別銘柄のスタイル質問(「トヨタはバリュー株?」等)
    const code = codes[0] || currentCode;
    const sym = STOCK_DATA.symbols[code];
    if (!sym || sym.type === "etf") {
      return { html: "<p>ETFはファクター分析の対象外です。個別株の銘柄名を含めて質問してください(例:「トヨタはバリュー株?」「モメンタムが強い銘柄は?」)。</p>" };
    }
    const style = Factor.classifyStyle(code);
    const rows = Factor.FACTORS.map(f => {
      const v = style.scores[f.key];
      const mark = v == null ? "" : v >= 58 ? " 💪" : v <= 42 ? " ⤵" : "";
      return `<li>${f.label}: <b>${v ?? "—"}</b>${mark}</li>`;
    });
    return {
      html: `<h4>🧮 ${sym.name} のファクター分析</h4>
        <p>スタイル判定: <b>${style.tags.join("・")}</b></p>
        <ul>${rows.join("")}</ul>
        <p>${style.comment}</p>
        <span class="qa-src">偏差値50=収録個別株の平均。詳細は「ファクター分析」タブへ。</span>`
    };
  }

  function answerNews(codes) {
    if (!codes.length) return { html: "<p>どの銘柄のニュースをお調べしますか?</p>" };
    const s = STOCK_DATA.symbols[codes[0]];
    const evs = (s.events || []).slice(-4).reverse();
    if (!evs.length) return { html: `<p>${s.name}のニュースは未収集です。</p>` };
    let html = `<h4>📰 ${s.name} の最近のイベント</h4><ul>` +
      evs.map(ev => `<li><b>${StockChart.fmtDate(ev.date)} ${ev.title}</b><br>${ev.summary}</li>`).join("") + "</ul>";
    html += `<span class="qa-src">LLMクローラーによる収集データより</span>`;
    return { html };
  }

  function answerCompare(codes) {
    if (codes.length < 2) return { html: "<p>比較する2銘柄を含めて質問してください(例: 「トヨタとソニーどっちが割安?」)。</p>" };
    const rows = codes.slice(0, 3).map(c => {
      const s = STOCK_DATA.symbols[c];
      const f = s.financials || {};
      return `<li><b>${s.name}</b>: PER ${f.per ?? "—"}倍 / PBR ${f.pbr ?? "—"}倍 / ROE ${f.roe ?? "—"}% / 配当 ${f.dividendYield ?? "—"}%</li>`;
    });
    return { html: `<h4>⚖️ 指標比較</h4><ul>${rows.join("")}</ul><p>一般にPER・PBRが低いほど割安、ROEが高いほど資本効率が良いとされます。</p>` };
  }

  return { answer };
})();
