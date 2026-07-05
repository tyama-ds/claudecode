/* ==========================================================================
   chart.js — SVG 株価チャート & レーダーチャート(依存ライブラリなし)
   ========================================================================== */

const StockChart = (() => {
  const NS = "http://www.w3.org/2000/svg";

  function el(name, attrs = {}, parent = null) {
    const e = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function fmtNum(v, currency) {
    const opts = v >= 1000 ? { maximumFractionDigits: 0 } : { maximumFractionDigits: 2 };
    const s = v.toLocaleString("ja-JP", opts);
    return currency === "USD" ? "$" + s : "¥" + s;
  }

  function fmtDate(iso) {
    const [y, m, d] = iso.split("-");
    return `${y}/${Number(m)}/${Number(d)}`;
  }

  function movingAverage(data, window) {
    const out = new Array(data.length).fill(null);
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      sum += data[i].close;
      if (i >= window) sum -= data[i - window].close;
      if (i >= window - 1) out[i] = sum / window;
    }
    return out;
  }

  /* ---------- メイン株価チャート ---------- */
  // opts: { data:[{date,close,volume}], events:[], showMA25, showMA75, showEvents,
  //         currency, onEventClick(event) }
  function renderPriceChart(container, volContainer, tooltip, opts) {
    const { data, events, currency } = opts;
    container.innerHTML = "";
    volContainer.innerHTML = "";
    if (!data || data.length < 2) {
      container.innerHTML = '<p style="color:var(--ink-muted);padding:40px;text-align:center">データがありません</p>';
      return;
    }

    const W = Math.max(container.clientWidth, 320);
    const H = 320, VH = 70;
    const pad = { top: 14, right: 64, bottom: 24, left: 10 };
    const plotW = W - pad.left - pad.right;
    const plotH = H - pad.top - pad.bottom;

    const closes = data.map(d => d.close);
    const ma25 = movingAverage(opts.fullData || data, 25).slice(-(data.length));
    const ma75 = movingAverage(opts.fullData || data, 75).slice(-(data.length));

    let lo = Math.min(...closes), hi = Math.max(...closes);
    if (opts.showMA25) { ma25.forEach(v => { if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }); }
    if (opts.showMA75) { ma75.forEach(v => { if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }); }
    const range = hi - lo || 1;
    lo -= range * 0.06; hi += range * 0.06;

    const x = i => pad.left + (i / (data.length - 1)) * plotW;
    const y = v => pad.top + (1 - (v - lo) / (hi - lo)) * plotH;

    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H }, container);

    /* grid + y-axis labels (right) */
    const ticks = 5;
    for (let t = 0; t <= ticks; t++) {
      const v = lo + ((hi - lo) * t) / ticks;
      const yy = y(v);
      el("line", { x1: pad.left, x2: pad.left + plotW, y1: yy, y2: yy, stroke: "var(--grid)", "stroke-width": 1 }, svg);
      const label = el("text", {
        x: pad.left + plotW + 8, y: yy + 4, "font-size": 11,
        fill: "var(--ink-muted)", "font-variant-numeric": "tabular-nums"
      }, svg);
      label.textContent = fmtNum(v, currency);
    }

    /* x-axis labels: ~6 evenly spaced */
    const nLabels = Math.min(6, data.length);
    for (let t = 0; t < nLabels; t++) {
      const i = Math.round((t / (nLabels - 1)) * (data.length - 1));
      const label = el("text", {
        x: x(i), y: H - 6, "font-size": 11, "text-anchor": "middle",
        fill: "var(--ink-muted)", "font-variant-numeric": "tabular-nums"
      }, svg);
      const [yy, mm] = data[i].date.split("-");
      label.textContent = `${yy.slice(2)}/${Number(mm)}`;
    }

    /* area fill */
    let areaPath = `M ${x(0)} ${y(closes[0])}`;
    for (let i = 1; i < data.length; i++) areaPath += ` L ${x(i)} ${y(closes[i])}`;
    areaPath += ` L ${x(data.length - 1)} ${pad.top + plotH} L ${x(0)} ${pad.top + plotH} Z`;
    el("path", { d: areaPath, fill: "var(--series-1-fill)" }, svg);

    /* price line */
    let linePath = `M ${x(0)} ${y(closes[0])}`;
    for (let i = 1; i < data.length; i++) linePath += ` L ${x(i)} ${y(closes[i])}`;
    el("path", { d: linePath, fill: "none", stroke: "var(--series-1)", "stroke-width": 2, "stroke-linejoin": "round" }, svg);

    /* MA lines */
    function drawMA(ma, color) {
      let p = "", started = false;
      for (let i = 0; i < data.length; i++) {
        if (ma[i] == null) continue;
        p += (started ? " L" : "M") + ` ${x(i)} ${y(ma[i])}`;
        started = true;
      }
      if (p) el("path", { d: p, fill: "none", stroke: color, "stroke-width": 1.5, "stroke-dasharray": "none", opacity: .9 }, svg);
    }
    if (opts.showMA25) drawMA(ma25, "var(--series-3)");
    if (opts.showMA75) drawMA(ma75, "var(--series-5)");

    /* event markers */
    const dateIndex = new Map(data.map((d, i) => [d.date, i]));
    const shownEvents = [];
    if (opts.showEvents && events) {
      for (const ev of events) {
        // snap to nearest trading day within 4 days
        let idx = dateIndex.get(ev.date);
        if (idx == null) {
          const evT = new Date(ev.date).getTime();
          let best = -1, bestDiff = Infinity;
          for (let i = 0; i < data.length; i++) {
            const diff = Math.abs(new Date(data[i].date).getTime() - evT);
            if (diff < bestDiff) { bestDiff = diff; best = i; }
          }
          if (bestDiff <= 4 * 86400000) idx = best;
        }
        if (idx == null) continue;
        shownEvents.push({ ...ev, idx });
        const color = ev.impact === "up" ? "var(--up)" : ev.impact === "down" ? "var(--down)" : "var(--ink-muted)";
        const c = el("circle", {
          cx: x(idx), cy: y(closes[idx]), r: 5.5,
          fill: color, stroke: "var(--surface)", "stroke-width": 2,
          class: "event-marker"
        }, svg);
        c.addEventListener("click", (e) => { e.stopPropagation(); opts.onEventClick && opts.onEventClick(ev); });
      }
    }

    /* ---------- volume chart ---------- */
    const vsvg = el("svg", { viewBox: `0 0 ${W} ${VH}`, width: "100%", height: VH }, volContainer);
    const maxVol = Math.max(...data.map(d => d.volume || 0)) || 1;
    const barW = Math.max(1, (plotW / data.length) - 1);
    for (let i = 0; i < data.length; i++) {
      const vh = ((data[i].volume || 0) / maxVol) * (VH - 14);
      const up = i === 0 || data[i].close >= data[i - 1].close;
      el("rect", {
        x: x(i) - barW / 2, y: VH - vh, width: barW, height: vh, rx: 1,
        fill: up ? "var(--up)" : "var(--down)", opacity: .45
      }, vsvg);
    }
    const volLabel = el("text", { x: pad.left, y: 11, "font-size": 10, fill: "var(--ink-muted)" }, vsvg);
    volLabel.textContent = "出来高";

    /* ---------- crosshair + tooltip ---------- */
    const cross = el("line", { y1: pad.top, y2: pad.top + plotH, stroke: "var(--axis)", "stroke-width": 1, "stroke-dasharray": "3 3", visibility: "hidden" }, svg);
    const dot = el("circle", { r: 4.5, fill: "var(--series-1)", stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }, svg);
    const hitRect = el("rect", { x: pad.left, y: 0, width: plotW, height: H, fill: "transparent" }, svg);

    function onMove(clientX) {
      const rect = svg.getBoundingClientRect();
      const scale = W / rect.width;
      const px = (clientX - rect.left) * scale;
      let i = Math.round(((px - pad.left) / plotW) * (data.length - 1));
      i = Math.max(0, Math.min(data.length - 1, i));
      const d = data[i];
      cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i));
      cross.setAttribute("visibility", "visible");
      dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(d.close));
      dot.setAttribute("visibility", "visible");

      const chg = i > 0 ? ((d.close - data[i - 1].close) / data[i - 1].close) * 100 : 0;
      const evHere = shownEvents.filter(ev => Math.abs(ev.idx - i) <= 0);
      tooltip.innerHTML = `
        <div class="tt-date">${fmtDate(d.date)}</div>
        <div class="tt-row"><span class="k">終値</span><span>${fmtNum(d.close, currency)}</span></div>
        <div class="tt-row"><span class="k">前日比</span><span style="color:${chg >= 0 ? "var(--up)" : "var(--down)"}">${chg >= 0 ? "+" : ""}${chg.toFixed(2)}%</span></div>
        <div class="tt-row"><span class="k">出来高</span><span>${(d.volume || 0).toLocaleString()}</span></div>
        ${opts.showMA25 && ma25[i] != null ? `<div class="tt-row"><span class="k">MA25</span><span>${fmtNum(ma25[i], currency)}</span></div>` : ""}
        ${evHere.length ? `<div class="tt-event">● ${evHere[0].title}</div>` : ""}
      `;
      tooltip.hidden = false;
      const wrapRect = container.parentElement.getBoundingClientRect();
      const ttW = tooltip.offsetWidth;
      let left = ((x(i) / W) * rect.width) + 14;
      if (left + ttW > wrapRect.width - 8) left = ((x(i) / W) * rect.width) - ttW - 14;
      tooltip.style.left = left + "px";
      tooltip.style.top = Math.max(4, (y(d.close) / H) * rect.height - 40) + "px";
    }
    function onLeave() {
      cross.setAttribute("visibility", "hidden");
      dot.setAttribute("visibility", "hidden");
      tooltip.hidden = true;
    }
    hitRect.addEventListener("mousemove", e => onMove(e.clientX));
    hitRect.addEventListener("mouseleave", onLeave);
    hitRect.addEventListener("touchmove", e => { onMove(e.touches[0].clientX); }, { passive: true });
    hitRect.addEventListener("touchend", onLeave);
  }

  /* ---------- レーダーチャート(財務スコア) ---------- */
  // scores: [{label, score(0-5)}]
  function renderRadar(container, scores) {
    container.innerHTML = "";
    const W = 340, H = 300, cx = W / 2, cy = H / 2 + 6, R = 100;
    const n = scores.length;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, container);

    const angle = i => (Math.PI * 2 * i) / n - Math.PI / 2;
    const pt = (i, r) => [cx + Math.cos(angle(i)) * r, cy + Math.sin(angle(i)) * r];

    /* rings */
    for (let ring = 1; ring <= 5; ring++) {
      const r = (R * ring) / 5;
      let d = "";
      for (let i = 0; i <= n; i++) {
        const [px, py] = pt(i % n, r);
        d += (i === 0 ? "M" : "L") + `${px} ${py}`;
      }
      el("path", { d: d + "Z", fill: "none", stroke: "var(--grid)", "stroke-width": 1 }, svg);
    }
    /* spokes + labels */
    for (let i = 0; i < n; i++) {
      const [px, py] = pt(i, R);
      el("line", { x1: cx, y1: cy, x2: px, y2: py, stroke: "var(--grid)", "stroke-width": 1 }, svg);
      const [lx, ly] = pt(i, R + 24);
      const t = el("text", {
        x: lx, y: ly + 4, "font-size": 11.5, "font-weight": 600,
        "text-anchor": "middle", fill: "var(--ink-2)"
      }, svg);
      t.textContent = scores[i].label;
      const t2 = el("text", {
        x: lx, y: ly + 18, "font-size": 10, "text-anchor": "middle",
        fill: "var(--ink-muted)", "font-variant-numeric": "tabular-nums"
      }, svg);
      t2.textContent = scores[i].score.toFixed(1);
    }
    /* data polygon */
    let d = "";
    for (let i = 0; i <= n; i++) {
      const [px, py] = pt(i % n, (R * Math.max(0.05, scores[i % n].score)) / 5);
      d += (i === 0 ? "M" : "L") + `${px} ${py}`;
    }
    el("path", { d: d + "Z", fill: "var(--series-1-fill)", stroke: "var(--series-1)", "stroke-width": 2, "stroke-linejoin": "round" }, svg);
    for (let i = 0; i < n; i++) {
      const [px, py] = pt(i, (R * Math.max(0.05, scores[i].score)) / 5);
      el("circle", { cx: px, cy: py, r: 4, fill: "var(--series-1)", stroke: "var(--surface)", "stroke-width": 2 }, svg);
    }
  }

  /* ---------- ファクター散布図 ---------- */
  // points: [{code, name, x, y}] (偏差値20-80), highlightCode: 強調銘柄
  function renderScatter(container, points, opts) {
    container.innerHTML = "";
    const W = Math.max(container.clientWidth, 320), H = 380;
    const pad = { top: 20, right: 20, bottom: 44, left: 46 };
    const plotW = W - pad.left - pad.right, plotH = H - pad.top - pad.bottom;
    const LO = 18, HI = 82;
    const x = v => pad.left + ((v - LO) / (HI - LO)) * plotW;
    const y = v => pad.top + (1 - (v - LO) / (HI - LO)) * plotH;

    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%" }, container);

    /* grid */
    for (let v = 20; v <= 80; v += 10) {
      el("line", { x1: x(v), x2: x(v), y1: pad.top, y2: pad.top + plotH, stroke: "var(--grid)", "stroke-width": 1 }, svg);
      el("line", { y1: y(v), y2: y(v), x1: pad.left, x2: pad.left + plotW, stroke: "var(--grid)", "stroke-width": 1 }, svg);
      const tx = el("text", { x: x(v), y: H - 26, "font-size": 10, "text-anchor": "middle", fill: "var(--ink-muted)" }, svg);
      tx.textContent = v;
      const ty = el("text", { x: pad.left - 8, y: y(v) + 3, "font-size": 10, "text-anchor": "end", fill: "var(--ink-muted)" }, svg);
      ty.textContent = v;
    }
    /* 平均線(50) */
    el("line", { x1: x(50), x2: x(50), y1: pad.top, y2: pad.top + plotH, stroke: "var(--axis)", "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);
    el("line", { y1: y(50), y2: y(50), x1: pad.left, x2: pad.left + plotW, stroke: "var(--axis)", "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);

    /* 軸ラベル */
    const xl = el("text", { x: pad.left + plotW / 2, y: H - 6, "font-size": 12, "font-weight": 700, "text-anchor": "middle", fill: "var(--ink-2)" }, svg);
    xl.textContent = `${opts.xLabel} →`;
    const yl = el("text", { x: 12, y: pad.top + plotH / 2, "font-size": 12, "font-weight": 700, "text-anchor": "middle", fill: "var(--ink-2)", transform: `rotate(-90 12 ${pad.top + plotH / 2})` }, svg);
    yl.textContent = `${opts.yLabel} →`;

    /* 点 + ラベル(重なる場合は下側に退避、それでも重なれば省略しホバーで表示) */
    const placedLabels = [];
    const drawOrder = [...points].sort((a, b) => (a.code === opts.highlightCode ? 1 : 0) - (b.code === opts.highlightCode ? 1 : 0));
    for (const p of drawOrder) {
      if (p.x == null || p.y == null) continue;
      const hl = p.code === opts.highlightCode;
      if (hl) {
        el("circle", { cx: x(p.x), cy: y(p.y), r: 13, fill: "none", stroke: "var(--down)", "stroke-width": 2, opacity: .8 }, svg);
      }
      const c = el("circle", {
        cx: x(p.x), cy: y(p.y), r: hl ? 8 : 6,
        fill: hl ? "var(--down)" : "var(--series-1)",
        stroke: "var(--surface)", "stroke-width": 2,
        class: "scatter-dot", "data-code": p.code, cursor: "pointer"
      }, svg);
      const title = el("title", {}, c);
      title.textContent = `${p.name}\n${opts.xLabel}: ${p.x} / ${opts.yLabel}: ${p.y}`;
      if (opts.onPointClick) c.addEventListener("click", () => opts.onPointClick(p.code));

      const label = p.name.length > 8 ? p.name.slice(0, 7) + "…" : p.name;
      const fs = hl ? 11.5 : 10;
      const w = label.length * fs * 0.95; // 全角想定の概算幅
      const overlaps = (bx, by) => placedLabels.some(b =>
        Math.abs(bx - b.x) < (w + b.w) / 2 && Math.abs(by - b.y) < fs + 3);
      let ly = y(p.y) - (hl ? 16 : 11);            // まず上側
      if (overlaps(x(p.x), ly)) ly = y(p.y) + (hl ? 24 : 19); // だめなら下側
      if (overlaps(x(p.x), ly) && !hl) continue;    // それでも重なれば省略(ホバーで確認可)
      placedLabels.push({ x: x(p.x), y: ly, w });
      const t = el("text", {
        x: x(p.x), y: ly, "font-size": fs,
        "font-weight": hl ? 800 : 600, "text-anchor": "middle",
        fill: hl ? "var(--ink)" : "var(--ink-2)", "pointer-events": "none"
      }, svg);
      t.textContent = label;
    }
  }

  return { renderPriceChart, renderRadar, renderScatter, fmtNum, fmtDate };
})();
