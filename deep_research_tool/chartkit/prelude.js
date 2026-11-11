// chartkit prelude — fixed rendering rules (no LLM-written code runs here).
// SPEC is defined by the template: an array of resolved panels.
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const ink = '#898781', grid = dark ? '#2c2c2a' : '#e1e0d9', hollow = dark ? '#1a1a19' : '#fcfcfb';
const ROLE = { emphasis: '#2a78d6', context: '#888780', context2: '#B4B2A9', negative: '#e34948' };
const CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
const SEG = [['#85B7EB', '#042C53'], ['#F0997B', '#4A1B0C'], ['#B4B2A9', '#2C2C2A'], ['#A9D8C2', '#0E3D2A']];
// every point style can be drawn hollow ('cross' / 'star' are only strokes)
const SHAPES = ['circle', 'rect', 'triangle', 'rectRot', 'rectRounded', 'circle'];
// dashes mean a forecast (never a second series)
const FORECAST_DASH = [6, 4];
// notes under the legend (render_png.py prints the same words)
const NOTE_STRIPE = '斜線＝換算・導出・推定・二次', NOTE_HOLLOW = '白抜き＝換算・導出・推定・二次', NOTE_FORECAST = '破線・白抜き＝予測値';
Chart.defaults.color = ink; Chart.defaults.font.size = 12;

function stripe(color) {
  const c = document.createElement('canvas'); c.width = c.height = 8;
  const g = c.getContext('2d'); g.fillStyle = color; g.fillRect(0, 0, 8, 8);
  g.strokeStyle = 'rgba(255,255,255,.55)'; g.lineWidth = 2; g.beginPath();
  g.moveTo(-2, 8); g.lineTo(8, -2); g.moveTo(2, 12); g.lineTo(12, 2); g.stroke();
  return g.createPattern(c, 'repeat');
}
// a forecast bar: hollow, dotted in its colour (forecastOutline draws its dashed outline)
function dots(color) {
  const c = document.createElement('canvas'); c.width = c.height = 6;
  const g = c.getContext('2d'); g.fillStyle = hollow; g.fillRect(0, 0, 6, 6);
  g.fillStyle = color; g.beginPath(); g.arc(3, 3, 1, 0, 2 * Math.PI); g.fill();
  return g.createPattern(c, 'repeat');
}
const published = (b) => b == null || b === '公表';
const forecastOf = (s) => (s.values || []).map((_, j) => !!(s.forecast || [])[j]);
const fillFor = (colors, basis, fc) => colors.map((col, i) => fc[i] ? dots(col) : published(basis[i]) ? col : stripe(col));
// a line chart's value axis starts at 0 only when the values are near it (the
// smallest at most a fifth of the largest) — the PNG's rule (_line_floor)
function lineFromZero(p) {
  const ys = [];
  (p.series || []).forEach(s => (s.values || []).forEach(v => {
    const y = v && typeof v === 'object' ? v.y : v; if (typeof y === 'number') ys.push(y); }));
  if (!ys.length) return true;
  const lo = Math.min(...ys), hi = Math.max(...ys);
  return !(hi > 0 && hi > lo && lo > 0.2 * hi);
}
// "%" values get a sign only on change panels (growth / YoY / negatives);
// shares and ratios stay unsigned ("56%", not "+56%")
const GROWTH_RE = /前年比|前期比|成長|伸び|増減|変化|増加率|growth|yoy|change|cagr/i;
const signedPct = (p) => (p.series || []).some(s => (s.values || []).some(v => typeof v === 'number' && v < 0))
  || GROWTH_RE.test(p.title || '');
// as the PNG's _fmt: thousands separators, one decimal, 3 significant
// digits below 1 (0.04 never prints as 0.0)
const num = (v) => Number.isInteger(v) ? v.toLocaleString('en-US')
  : Math.abs(v) < 1 ? String(Number(Number(v).toPrecision(3)))
  : Number(v).toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
const fmtOf = (p) => p.unit === '%' ? (sg => v => (sg && v > 0 ? '+' : '') + v + '%')(signedPct(p)) : num;

// the rectangle a bar element covers (canvas pixels)
function barBox(el, horiz) {
  const p = el.getProps(['x', 'y', 'base', 'width', 'height'], true);
  return horiz ? { l: Math.min(p.x, p.base), r: Math.max(p.x, p.base), t: p.y - p.height / 2, b: p.y + p.height / 2 }
               : { l: p.x - p.width / 2, r: p.x + p.width / 2, t: Math.min(p.y, p.base), b: Math.max(p.y, p.base) };
}
const hits = (a, b) => a.l < b.r - 1 && b.l < a.r - 1 && a.t < b.b - 1 && b.t < a.b - 1;
// value labels, the emphasised series first: a label that would sit on another
// series' bar or on a label already drawn is left out (the emphasised series
// keeps all of its labels) — the PNG's rule
const valueLabels = { id: 'vl', afterDatasetsDraw(c) {
  const x = c.ctx, horiz = c.options.indexAxis === 'y'; x.save(); x.font = '11px sans-serif'; x.fillStyle = ink;
  const sets = c.data.datasets, metas = sets.map((_, i) => c.getDatasetMeta(i));
  const bars = metas.map(m => m.data.map(el => barBox(el, horiz)));
  const order = sets.map((_, i) => i).sort((a, b) => (sets[a].role === 'emphasis' ? 0 : 1) - (sets[b].role === 'emphasis' ? 0 : 1));
  const kept = [];
  order.forEach(i => { const ds = sets[i]; metas[i].data.forEach((el, j) => {
    const v = ds.data[j]; if (v == null) return; const t = (ds.fmt || (n => n))(v);
    const w = x.measureText(t).width, h = 12; let tx, ty, box;
    if (horiz) { tx = v < 0 ? el.x - 4 : el.x + 4; ty = el.y;
      box = { l: v < 0 ? tx - w : tx, r: v < 0 ? tx : tx + w, t: ty - h / 2, b: ty + h / 2 }; }
    else { tx = el.x; ty = v < 0 ? el.y + 3 : el.y - 3;
      box = { l: tx - w / 2, r: tx + w / 2, t: v < 0 ? ty : ty - h, b: v < 0 ? ty + h : ty }; }
    const blocked = bars.some((bs, k) => k !== i && bs.some(b => hits(box, b))) || kept.some(k => hits(box, k));
    if (blocked && ds.role !== 'emphasis') return;
    kept.push(box);
    if (horiz) { x.textAlign = v < 0 ? 'right' : 'left'; x.textBaseline = 'middle'; }
    else { x.textAlign = 'center'; x.textBaseline = v < 0 ? 'top' : 'bottom'; }
    x.fillText(t, tx, ty);
  }); }); x.restore();
}};
const totalLabels = { id: 'tot', afterDatasetsDraw(c) {
  const x = c.ctx; x.save(); x.font = '11px sans-serif'; x.fillStyle = ink; x.textAlign = 'center'; x.textBaseline = 'bottom';
  c.data.labels.forEach((_, j) => { let s = 0, top = null;
    c.data.datasets.forEach((ds, i) => { const v = ds.data[j]; if (v == null) return; s += v;
      const el = c.getDatasetMeta(i).data[j]; if (!top || el.y < top.y) top = el; });
    if (top) x.fillText(num(s), top.x, top.y - 3); }); x.restore();
}};
const segmentLabels = { id: 'seg', afterDatasetsDraw(c) {
  const x = c.ctx; x.save(); x.font = '12px sans-serif'; x.textAlign = 'center'; x.textBaseline = 'middle';
  c.data.datasets.forEach((ds, i) => { const el = c.getDatasetMeta(i).data[0]; if (!el || ds.data[0] == null) return;
    x.fillStyle = ds.lc; x.fillText(ds.label + ' ' + ds.data[0] + (c.$unit === '%' ? '%' : ''), (el.base + el.x) / 2, el.y); }); x.restore();
}};
// a forecast bar's dashed outline (a Chart.js bar has no dashed border)
const forecastOutline = { id: 'fco', afterDatasetsDraw(c) {
  const x = c.ctx, horiz = c.options.indexAxis === 'y'; x.save(); x.setLineDash([4, 3]); x.lineWidth = 1.5;
  c.data.datasets.forEach((ds, i) => { if (!ds.fc) return;
    c.getDatasetMeta(i).data.forEach((el, j) => { if (!ds.fc[j] || ds.data[j] == null) return;
      const b = barBox(el, horiz); x.strokeStyle = (ds.edge || [])[j] || ds.borderColor;
      x.strokeRect(b.l, b.t, b.r - b.l, b.b - b.t); }); });
  x.restore();
}};

function buildLegend(p, datasets) {
  const lg = document.getElementById('lg_' + p.id); if (!lg) return;
  // one series: the title names it — no legend box (the notes stay)
  const items = datasets.length > 1
    ? datasets.map(d => '<span><span class="sw" style="background:' + d.legendColor + '"></span>' + esc(d.label) + '</span>') : [];
  const fc = p.series.map(forecastOf);
  // not 公表 and not a forecast: striped on a bar, hollow on a line
  if (p.series.some((s, i) => (s.basis || []).some((b, j) => !published(b) && !fc[i][j])))
    items.push('<span>' + (p.form === 'line_xy' ? NOTE_HOLLOW : NOTE_STRIPE) + '</span>');
  if (fc.some(f => f.some(Boolean))) items.push('<span>' + NOTE_FORECAST + '</span>');
  lg.innerHTML = items.join('');
}
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function renderPanel(p) {
  const horiz = p.form.endsWith('hbar'), stacked = p.form.startsWith('stacked'), line = p.form === 'line_xy';
  const segRow = p.form === 'stacked_hbar', several = p.series.length > 1;
  const datasets = p.series.map((s, i) => {
    const base = line || s.role === 'part' ? (segRow ? SEG[i % SEG.length][0] : CAT[i % CAT.length]) : ROLE[s.role] || ROLE.context;
    const basis = s.basis || [], fc = forecastOf(s);
    let colors;
    if (p.highlight && p.highlight.length) colors = p.labels.map(l => p.highlight.includes(l) ? ROLE.emphasis : ROLE.context);
    else colors = (line ? [] : s.values).map(v => (typeof v === 'number' && v < 0 && p.series.length === 1) ? ROLE.negative : base);
    const ds = { label: s.name, data: s.values, legendColor: base, role: s.role, fc, borderColor: base, fmt: fmtOf(p) };
    if (line) {
      // actual values: a solid line in the series' own colour; a segment into
      // or out of a forecast is dashed; a forecast point, or one that is not
      // 公表, is hollow; the emphasised series is drawn thicker
      const open = (s.values || []).map((_, j) => fc[j] || !published(basis[j]));
      Object.assign(ds, { backgroundColor: base, borderWidth: s.role === 'emphasis' && several ? 2.6 : 2,
        pointStyle: SHAPES[i % SHAPES.length], pointRadius: 5, pointBorderWidth: 1.5,
        pointBackgroundColor: open.map(o => o ? hollow : base), pointBorderColor: base,
        tension: 0, showLine: true, order: s.role === 'emphasis' ? 0 : 1,
        segment: { borderDash: ctx => (fc[ctx.p0DataIndex] || fc[ctx.p1DataIndex]) ? FORECAST_DASH : undefined } });
    } else {
      // bar-only options (a line dataset has no bars to size)
      Object.assign(ds, { backgroundColor: fillFor(colors, basis, fc), edge: colors, borderWidth: 0,
        borderRadius: stacked ? 0 : 4, maxBarThickness: segRow ? 28 : 24,
        barPercentage: 0.9, categoryPercentage: several && !stacked ? 0.8 : 0.7 });
    }
    if (segRow) ds.lc = SEG[i % SEG.length][1];
    return ds;
  });
  const scales = line
    ? { x: { type: 'linear', grid: { display: false }, ticks: { stepSize: 1, callback: v => String(v) } },
        y: { beginAtZero: lineFromZero(p), grace: '5%', suggestedMax: p.y_max, grid: { color: grid }, border: { display: false }, title: { display: !!p.unit, text: p.unit } } }
    : segRow
      ? { x: { stacked: true, min: 0, max: p.y_max || undefined, display: false }, y: { stacked: true, display: false } }
      : { x: { stacked, grid: { display: horiz, color: grid }, border: { display: false }, ticks: { autoSkip: false }, max: horiz ? p.y_max : undefined, beginAtZero: true },
          y: { stacked, beginAtZero: true, grid: { display: !horiz, color: grid }, border: { display: false }, suggestedMax: horiz ? undefined : p.y_max,
               title: { display: !!p.unit && !horiz, text: p.unit } } };
  const chart = new Chart(document.getElementById('cv_' + p.id), {
    type: line ? 'line' : 'bar', data: { labels: line ? undefined : p.labels, datasets },
    options: { indexAxis: horiz ? 'y' : 'x', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales },
    plugins: segRow ? [segmentLabels, forecastOutline] : stacked ? [totalLabels, forecastOutline] : line ? [] : [valueLabels, forecastOutline]
  });
  chart.$unit = p.unit;
  buildLegend(p, datasets);
}
SPEC.forEach(p => { try { renderPanel(p); } catch (e) { console.error('panel ' + p.id, e); } });
