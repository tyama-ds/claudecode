// chartkit prelude — fixed rendering rules (no LLM-written code runs here).
// SPEC is defined by the template: an array of resolved panels.
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
const ink = '#898781', grid = dark ? '#2c2c2a' : '#e1e0d9';
const ROLE = { emphasis: '#2a78d6', context: '#888780', context2: '#B4B2A9', negative: '#e34948' };
const CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'];
const SEG = [['#85B7EB', '#042C53'], ['#F0997B', '#4A1B0C'], ['#B4B2A9', '#2C2C2A'], ['#A9D8C2', '#0E3D2A']];
const SHAPES = ['circle', 'rect', 'triangle', 'rectRot', 'star', 'cross'];
const DASHES = [[], [6, 4], [2, 3], [8, 4, 2, 4], [4, 4], [1, 2]];
Chart.defaults.color = ink; Chart.defaults.font.size = 12;

function stripe(color) {
  const c = document.createElement('canvas'); c.width = c.height = 8;
  const g = c.getContext('2d'); g.fillStyle = color; g.fillRect(0, 0, 8, 8);
  g.strokeStyle = 'rgba(255,255,255,.55)'; g.lineWidth = 2; g.beginPath();
  g.moveTo(-2, 8); g.lineTo(8, -2); g.moveTo(2, 12); g.lineTo(12, 2); g.stroke();
  return g.createPattern(c, 'repeat');
}
const published = (b) => b == null || b === '公表';
const fillFor = (colors, basis) => colors.map((col, i) => published(basis[i]) ? col : stripe(col));
// "%" values get a sign only on change panels (growth / YoY / negatives);
// shares and ratios stay unsigned ("56%", not "+56%")
const GROWTH_RE = /前年比|前期比|成長|伸び|増減|変化|増加率|growth|yoy|change|cagr/i;
const signedPct = (p) => (p.series || []).some(s => (s.values || []).some(v => typeof v === 'number' && v < 0))
  || GROWTH_RE.test(p.title || '');
const fmtOf = (p) => p.unit === '%' ? (sg => v => (sg && v > 0 ? '+' : '') + v + '%')(signedPct(p))
  : (v => Number.isInteger(v) ? v.toLocaleString() : Number(v).toFixed(1));

const valueLabels = { id: 'vl', afterDatasetsDraw(c) {
  const x = c.ctx, horiz = c.options.indexAxis === 'y'; x.save(); x.font = '11px sans-serif'; x.fillStyle = ink;
  c.data.datasets.forEach((ds, i) => c.getDatasetMeta(i).data.forEach((el, j) => {
    const v = ds.data[j]; if (v == null) return; const t = (ds.fmt || (n => n))(v);
    if (horiz) { x.textAlign = 'left'; x.textBaseline = 'middle'; x.fillText(t, el.x + 4, el.y); }
    else { x.textAlign = 'center'; x.textBaseline = v < 0 ? 'top' : 'bottom'; x.fillText(t, el.x, v < 0 ? el.y + 3 : el.y - 3); }
  })); x.restore();
}};
const totalLabels = { id: 'tot', afterDatasetsDraw(c) {
  const x = c.ctx; x.save(); x.font = '11px sans-serif'; x.fillStyle = ink; x.textAlign = 'center'; x.textBaseline = 'bottom';
  c.data.labels.forEach((_, j) => { let s = 0, top = null;
    c.data.datasets.forEach((ds, i) => { const v = ds.data[j]; if (v == null) return; s += v;
      const el = c.getDatasetMeta(i).data[j]; if (!top || el.y < top.y) top = el; });
    if (top) x.fillText(Number.isInteger(s) ? s.toLocaleString() : s.toFixed(1), top.x, top.y - 3); }); x.restore();
}};
const segmentLabels = { id: 'seg', afterDatasetsDraw(c) {
  const x = c.ctx; x.save(); x.font = '12px sans-serif'; x.textAlign = 'center'; x.textBaseline = 'middle';
  c.data.datasets.forEach((ds, i) => { const el = c.getDatasetMeta(i).data[0]; if (!el || ds.data[0] == null) return;
    x.fillStyle = ds.lc; x.fillText(ds.label + ' ' + ds.data[0] + (c.$unit === '%' ? '%' : ''), (el.base + el.x) / 2, el.y); }); x.restore();
}};

function buildLegend(p, datasets) {
  const lg = document.getElementById('lg_' + p.id); if (!lg) return;
  const items = datasets.map(d => '<span><span class="sw" style="background:' + d.legendColor + '"></span>' + esc(d.label) + '</span>');
  const anyStriped = p.series.some(s => (s.basis || []).some(b => !published(b)));
  if (anyStriped) items.push('<span>斜線＝換算・導出・推定・二次</span>');
  lg.innerHTML = items.join('');
}
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function renderPanel(p) {
  const horiz = p.form.endsWith('hbar'), stacked = p.form.startsWith('stacked'), line = p.form === 'line_xy';
  const segRow = p.form === 'stacked_hbar';
  const datasets = p.series.map((s, i) => {
    const base = line || s.role === 'part' ? (segRow ? SEG[i % SEG.length][0] : CAT[i % CAT.length]) : ROLE[s.role] || ROLE.context;
    let colors;
    if (p.highlight && p.highlight.length) colors = p.labels.map(l => p.highlight.includes(l) ? ROLE.emphasis : ROLE.context);
    else colors = (line ? [] : s.values).map(v => (typeof v === 'number' && v < 0 && p.series.length === 1) ? ROLE.negative : base);
    const ds = { label: s.name, data: s.values, legendColor: base,
      backgroundColor: line ? base : fillFor(colors, s.basis || []), borderColor: base,
      borderRadius: stacked ? 0 : 4, maxBarThickness: segRow ? 28 : 24, borderWidth: line ? 2 : 0,
      pointStyle: SHAPES[i % SHAPES.length], pointRadius: 5, borderDash: DASHES[i % DASHES.length], tension: 0, showLine: true,
      fmt: fmtOf(p) };
    if (segRow) ds.lc = SEG[i % SEG.length][1];
    return ds;
  });
  const scales = line
    ? { x: { type: 'linear', grid: { display: false }, ticks: { stepSize: 1, callback: v => String(v) } },
        y: { beginAtZero: true, suggestedMax: p.y_max, grid: { color: grid }, border: { display: false }, title: { display: !!p.unit, text: p.unit } } }
    : segRow
      ? { x: { stacked: true, min: 0, max: p.y_max || undefined, display: false }, y: { stacked: true, display: false } }
      : { x: { stacked, grid: { display: horiz, color: grid }, border: { display: false }, ticks: { autoSkip: false }, max: horiz ? p.y_max : undefined, beginAtZero: true },
          y: { stacked, beginAtZero: true, grid: { display: !horiz, color: grid }, border: { display: false }, suggestedMax: horiz ? undefined : p.y_max,
               title: { display: !!p.unit && !horiz, text: p.unit } } };
  const chart = new Chart(document.getElementById('cv_' + p.id), {
    type: line ? 'line' : 'bar', data: { labels: line ? undefined : p.labels, datasets },
    options: { indexAxis: horiz ? 'y' : 'x', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales },
    plugins: segRow ? [segmentLabels] : stacked ? [totalLabels] : line ? [] : [valueLabels]
  });
  chart.$unit = p.unit;
  buildLegend(p, datasets);
}
SPEC.forEach(p => { try { renderPanel(p); } catch (e) { console.error('panel ' + p.id, e); } });
