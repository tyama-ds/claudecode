/* Agent Orchestrator — frontend logic (vanilla JS, no dependencies). */

const state = {
  agents: [],        // catalog of selectable backends
  strategies: [],    // strategy metadata
  es: null,          // active EventSource
  sessionId: null,
  cards: {},         // role-round -> DOM node (thinking -> filled)
  connState: "idle",
  artifact: { versions: [], view: "preview" },
  workspace: { files: {}, order: [], selected: null },
  team: { workers: {}, order: [], conductor: "", round: 0 },
};

const $ = (sel) => document.querySelector(sel);

// Which accent bucket a role belongs to (drives the card colour).
function accentFor(role) {
  if (["implementer", "agent_a", "planner", "conductor"].includes(role)) return "acc-a";
  if (["reviewer", "agent_b", "executor"].includes(role)) return "acc-b";
  const m = /^(?:agent|worker)_(\d+)$/.exec(role); // custom / team members
  if (m) return ["acc-a", "acc-b", "acc-c"][(parseInt(m[1], 10) - 1) % 3];
  return "acc-c"; // synthesizer / judge / anything else
}

// Preference order for auto-selecting a backend per role bucket.
function defaultAgentFor(role) {
  const a = ["claude_code", "anthropic", "codex", "openai", "local", "mock"];
  const b = ["codex", "openai", "claude_code", "anthropic", "local", "mock"];
  const order = accentFor(role) === "acc-b" ? b : a;
  for (const id of order) {
    const found = state.agents.find((x) => x.id === id && x.available);
    if (found) return id;
  }
  return "mock";
}

// Monogram for an agent avatar, e.g. "Claude Code" -> "CC", "Codex" -> "CX".
function initials(name) {
  const words = (name || "?").replace(/[()]/g, "").trim().split(/\s+/);
  if (words.length >= 2) return (words[0][0] + words[1][0]).toUpperCase();
  const w = words[0] || "?";
  return (w.length > 1 ? w[0] + w[1] : w[0]).toUpperCase();
}

// -- minimal, safe markdown rendering --------------------------------------
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}
function renderMarkdown(text) {
  let s = escapeHtml(text || "");
  s = s.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) =>
    `<pre><code>${code.replace(/\n$/, "")}</code></pre>`);
  s = s.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  const parts = s.split(/(<pre>[\s\S]*?<\/pre>)/g);
  return parts.map((p) => {
    if (p.startsWith("<pre>")) return p;
    return p.split(/\n{2,}/).filter(Boolean)
      .map((para) => `<p>${para.replace(/\n/g, "<br>")}</p>`).join("");
  }).join("");
}

// -- catalog & role rendering ----------------------------------------------
async function loadCatalog() {
  const res = await fetch("/api/catalog");
  const data = await res.json();
  state.agents = data.agents;
  state.strategies = data.strategies;

  const sel = $("#strategy");
  sel.innerHTML = "";
  for (const st of state.strategies) {
    const opt = document.createElement("option");
    opt.value = st.name;
    opt.textContent = st.name.replace(/_/g, " ");
    sel.appendChild(opt);
  }
  sel.addEventListener("change", renderRoles);
  renderRoles();
}

function currentStrategy() {
  return state.strategies.find((s) => s.name === $("#strategy").value) || state.strategies[0];
}

function renderRoles() {
  const st = currentStrategy();
  if (!st) return;
  $("#strategy-desc").textContent = st.description;
  $("#rounds").value = st.default_rounds;
  $("#workspace-opts").hidden = st.name !== "workspace_build";

  const wrap = $("#roles");
  wrap.innerHTML = "";
  if (st.name === "conductor_team") { renderConductorBuilder(wrap); return; }
  if (st.custom) { renderCustomBuilder(wrap); return; }
  st.roles.forEach((role) => wrap.appendChild(roleCard(role.key, role.label, role.system || "", false)));
}

// Conductor team: one conductor, 2–4 workers (add/remove), one reviewer.
const CONDUCTOR_DEFAULTS = {
  conductor: "Leads the team: decomposes the task, assigns subtasks, holds workers " +
    "accountable (calling out anyone who doesn't deliver), and integrates the result.",
  worker: "Carries out the specific assignment the conductor gives, concretely and in full.",
  reviewer: "Reviews each worker's output against its assignment and reports back to the conductor.",
};
function renderConductorBuilder(wrap) {
  wrap.appendChild(roleCard("conductor", "Conductor", CONDUCTOR_DEFAULTS.conductor, false));

  const holder = document.createElement("div");
  holder.id = "workers"; holder.className = "roles";
  wrap.appendChild(holder);
  const add = document.createElement("button");
  add.type = "button"; add.className = "btn ghost add-worker";
  add.textContent = "+ Add worker";
  add.addEventListener("click", () => addWorker(holder));
  wrap.appendChild(add);
  addWorker(holder);
  addWorker(holder);

  wrap.appendChild(roleCard("reviewer", "Reviewer", CONDUCTOR_DEFAULTS.reviewer, false));
}

function addWorker(holder) {
  if (holder.children.length >= 4) return;
  const i = holder.children.length + 1;
  holder.appendChild(roleCard("worker_" + i, "Worker " + i, CONDUCTOR_DEFAULTS.worker, true));
  relabelWorkers(holder);
}

function relabelWorkers(holder) {
  Array.from(holder.children).forEach((card, i) => {
    const key = "worker_" + (i + 1);
    const bucket = accentFor(key);
    card.className = "role " + (bucket === "acc-b" ? "b" : bucket === "acc-c" ? "c" : "");
    card.querySelector(".role-name").childNodes[0].nodeValue = "Worker " + (i + 1);
    card.querySelector("select").dataset.role = key;
  });
  const add = document.querySelector(".add-worker");
  if (add) add.disabled = holder.children.length >= 4;
}

// Build one role card: backend picker + optional model + editable persona.
function roleCard(key, label, defaultSystem, removable) {
  const bucket = accentFor(key);
  const box = document.createElement("div");
  box.className = "role " + (bucket === "acc-b" ? "b" : bucket === "acc-c" ? "c" : "");

  const name = document.createElement("div");
  name.className = "role-name";
  name.appendChild(document.createTextNode(label));
  if (removable) {
    const rm = document.createElement("button");
    rm.type = "button"; rm.className = "role-remove"; rm.textContent = "×"; rm.title = "Remove";
    rm.addEventListener("click", () => {
      const h = box.parentElement; box.remove();
      (h && h.id === "workers" ? relabelWorkers : relabelParticipants)(h);
    });
    name.appendChild(rm);
  }
  box.appendChild(name);

  const sw = document.createElement("div");
  sw.className = "select-wrap";
  const select = document.createElement("select");
  select.dataset.role = key;
  const preferred = defaultAgentFor(key);
  for (const ag of state.agents) {
    const opt = document.createElement("option");
    opt.value = ag.id;
    opt.textContent = ag.available ? ag.label : `${ag.label} — ${ag.reason}`;
    opt.disabled = !ag.available;
    if (ag.id === preferred) opt.selected = true;
    select.appendChild(opt);
  }
  sw.appendChild(select);
  box.appendChild(sw);

  const model = document.createElement("input");
  model.type = "text"; model.className = "model-in";
  model.placeholder = "model (optional — uses default)";
  box.appendChild(model);

  const det = document.createElement("details");
  det.className = "persona";
  const sum = document.createElement("summary");
  sum.textContent = "Persona";
  det.appendChild(sum);
  const ta = document.createElement("textarea");
  ta.className = "persona-in"; ta.rows = 3;
  ta.placeholder = defaultSystem
    ? "Leave blank to use the default persona below"
    : "Describe this participant's role / character (optional)";
  det.appendChild(ta);
  if (defaultSystem) {
    const hint = document.createElement("div");
    hint.className = "persona-default";
    hint.textContent = "Default: " + defaultSystem;
    det.appendChild(hint);
  }
  box.appendChild(det);
  return box;
}

// Custom strategy: a builder for 2–5 freely-defined participants.
function renderCustomBuilder(wrap) {
  const holder = document.createElement("div");
  holder.id = "participants"; holder.className = "roles";
  wrap.appendChild(holder);
  const add = document.createElement("button");
  add.type = "button"; add.className = "btn ghost add-participant";
  add.textContent = "+ Add participant";
  add.addEventListener("click", () => addParticipant(holder));
  wrap.appendChild(add);
  addParticipant(holder);
  addParticipant(holder);
}

function addParticipant(holder) {
  if (holder.children.length >= 5) return;
  const i = holder.children.length;
  holder.appendChild(roleCard("agent_" + (i + 1), "Participant " + (i + 1), "", true));
  relabelParticipants(holder);
}

function relabelParticipants(holder) {
  Array.from(holder.children).forEach((card, i) => {
    const key = "agent_" + (i + 1);
    const bucket = accentFor(key);
    card.className = "role " + (bucket === "acc-b" ? "b" : bucket === "acc-c" ? "c" : "");
    card.querySelector(".role-name").childNodes[0].nodeValue = "Participant " + (i + 1);
    card.querySelector("select").dataset.role = key;
  });
  const add = document.querySelector(".add-participant");
  if (add) add.disabled = holder.children.length >= 5;
}

function collectRole(card) {
  const out = { id: card.querySelector("select").value };
  const m = card.querySelector(".model-in");
  if (m && m.value.trim()) out.model = m.value.trim();
  const s = card.querySelector(".persona-in");
  if (s && s.value.trim()) out.system = s.value.trim();
  return out;
}

function rolesPayload() {
  const st = currentStrategy();
  const roles = {};
  if (st && st.custom) {
    const order = [];
    document.querySelectorAll("#participants > .role").forEach((card, i) => {
      const key = "agent_" + (i + 1);
      order.push(key);
      roles[key] = collectRole(card);
    });
    return { roles, role_order: order };
  }
  if (st && st.name === "conductor_team") {
    const fixed = Array.from(document.querySelectorAll("#roles > .role")); // [conductor, reviewer]
    const order = ["conductor"];
    roles.conductor = collectRole(fixed[0]);
    document.querySelectorAll("#workers > .role").forEach((card, i) => {
      const key = "worker_" + (i + 1);
      order.push(key);
      roles[key] = collectRole(card);
    });
    roles.reviewer = collectRole(fixed[fixed.length - 1]);
    order.push("reviewer");
    return { roles, role_order: order };
  }
  document.querySelectorAll("#roles > .role").forEach((card) => {
    roles[card.querySelector("select").dataset.role] = collectRole(card);
  });
  return { roles };
}

// -- run / stream ----------------------------------------------------------
function setConn(label, cls) {
  state.connState = cls;
  const el = $("#conn");
  el.className = "conn " + cls;
  el.querySelector(".conn-label").textContent = label;
}
function setStatus(msg, isErr) {
  const el = $("#status");
  el.textContent = msg || "";
  el.className = "status" + (isErr ? " err" : "");
}

async function run() {
  const task = $("#task").value.trim();
  if (!task) { setStatus("Please enter a task.", true); return; }

  const rp = rolesPayload();
  const payload = {
    task,
    strategy: $("#strategy").value,
    rounds: parseInt($("#rounds").value, 10) || 2,
    roles: rp.roles,
  };
  if (rp.role_order) payload.role_order = rp.role_order;
  const refDir = $("#reference-dir").value.trim();
  if (refDir) payload.reference_dir = refDir;
  if ($("#strategy").value === "workspace_build") {
    const ws = $("#workspace-dir").value.trim();
    if (ws) payload.workspace = ws;
    payload.create_dir = $("#workspace-init").checked;
  }

  setStatus("Starting…");
  $("#run").disabled = true;
  let res, data;
  try {
    res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    data = await res.json();
  } catch (e) {
    setStatus("Network error: " + e, true);
    $("#run").disabled = false;
    return;
  }
  if (!res.ok) {
    let msg = data.error || "Failed to start.";
    if (data.details) msg += " (" + data.details.map((d) => `${d.role}: ${d.reason}`).join("; ") + ")";
    setStatus(msg, true);
    $("#run").disabled = false;
    return;
  }

  state.sessionId = data.session_id;
  state.cards = {};
  state.artifact = { versions: [], view: "preview" };
  state.workspace = { files: {}, order: [], selected: null };
  state.team = { workers: {}, order: [], conductor: "", round: 0 };
  $("#team").hidden = true;
  $("#team-roster").innerHTML = "";
  $("#stream").innerHTML = "";
  $("#sp-list").innerHTML = "";
  $("#scratchpad").hidden = true;
  $("#artifact").hidden = true;
  $("#artifact-body").innerHTML = "";
  $("#workspace").hidden = true;
  $("#ws-files").innerHTML = "";
  $("#ws-diff").innerHTML = "";
  $("#stop").disabled = false;
  openStream(data.session_id);
}

function openStream(id) {
  if (state.es) state.es.close();
  const es = new EventSource(`/api/stream/${id}`);
  state.es = es;
  es.onmessage = (ev) => {
    let evt;
    try { evt = JSON.parse(ev.data); } catch { return; }
    handleEvent(evt);
  };
  es.onerror = () => {
    if (state.connState === "running") setConn("disconnected", "error");
  };
}

async function stop() {
  if (!state.sessionId) return;
  await fetch(`/api/stop/${state.sessionId}`, { method: "POST" });
  setStatus("Stop requested…");
}

// -- event handling --------------------------------------------------------
function handleEvent(evt) {
  const { type, data } = evt;
  if (type === "session_start") {
    setConn("running", "running");
    setStatus("Collaboration running…");
    const agents = Object.entries(data.agents).map(([r, n]) => `${r}=${n}`).join("  ·  ");
    const extras = [];
    if (data.references) extras.push(`${data.references} reference file(s)`);
    if (data.workspace_created === "created") extras.push("dir created");
    $("#meta").textContent =
      `${data.strategy} · ${data.rounds} rounds · ${agents}` +
      (extras.length ? ` · ${extras.join(" · ")}` : "");
    $("#artifact-ext").value = data.strategy === "code_authoring" ? ".py" : ".md";
    if (data.workspace) {
      $("#workspace-path").textContent =
        data.workspace + (data.workspace_created === "created" ? "  (created)" : "");
    }
    if (data.strategy === "conductor_team") seedTeam(data.agents);
  } else if (type === "artifact") {
    handleArtifact(data);
  } else if (type === "workspace_edit") {
    handleWorkspaceEdit(data);
  } else if (type === "worker_status") {
    handleWorkerStatus(data);
  } else if (type === "turn_start") {
    addThinkingCard(data);
  } else if (type === "turn_end") {
    fillCard(data);
  } else if (type === "status") {
    addNote(data.message);
  } else if (type === "scratchpad") {
    renderScratchpad(data.notes);
  } else if (type === "result") {
    addResult(data.content);
  } else if (type === "error") {
    setStatus(data.message, true);
    addNote("⚠ " + data.message);
  } else if (type === "session_end") {
    finish(data.status);
  }
}

// One-click copy of an agent's raw (un-rendered) output.
function makeCopyBtn(text) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "copy-btn";
  b.title = "Copy output";
  b.textContent = "Copy";
  b.addEventListener("click", (e) => { e.stopPropagation(); copyText(text, b); });
  return b;
}
function copyText(text, btn) {
  const flash = () => {
    btn.textContent = "Copied";
    btn.classList.add("copied");
    setTimeout(() => { btn.textContent = "Copy"; btn.classList.remove("copied"); }, 1200);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(flash).catch(() => fallbackCopy(text, flash));
  } else {
    fallbackCopy(text, flash);
  }
}
function fallbackCopy(text, flash) {
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  try { document.execCommand("copy"); flash(); } catch (e) { /* ignore */ }
  document.body.removeChild(ta);
}

function cardKey(d) { return `${d.role}-${d.round}`; }

function addThinkingCard(d) {
  const node = document.createElement("div");
  node.className = `turn thinking ${accentFor(d.role)}`;
  node.innerHTML = `
    <div class="turn-head">
      <span class="avatar">${escapeHtml(initials(d.agent))}</span>
      <span class="who">
        <span class="agent">${escapeHtml(d.agent)}</span>
        <span class="role-tag">${escapeHtml(d.role)}</span>
      </span>
      <span class="round">round ${d.round}</span>
    </div>
    <div class="turn-body">thinking</div>`;
  $("#stream").appendChild(node);
  state.cards[cardKey(d)] = node;
  node.scrollIntoView({ behavior: "smooth", block: "end" });
}

function fillCard(d) {
  const node = state.cards[cardKey(d)] || (() => { addThinkingCard(d); return state.cards[cardKey(d)]; })();
  node.classList.remove("thinking");
  if (!d.ok) node.classList.add("failed");
  if (d.duration != null) {
    const dur = document.createElement("span");
    dur.className = "dur";
    dur.textContent = `${d.duration}s`;
    node.querySelector(".turn-head").appendChild(dur);
  }
  if (d.via) {
    const tag = document.createElement("span");
    tag.className = "via via-" + d.via;
    tag.textContent = "ctx: " + d.via;
    tag.title = d.via === "history"
      ? "Context shared as structured message history"
      : "Context embedded in the prompt (fallback)";
    node.querySelector(".turn-head").appendChild(tag);
  }
  if (d.ok && d.content) {
    node.querySelector(".turn-head").appendChild(makeCopyBtn(d.content));
  }
  node.querySelector(".turn-body").innerHTML = d.ok ? renderMarkdown(d.content) : escapeHtml(d.content);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
}

function addNote(msg) {
  const el = document.createElement("div");
  el.className = "note";
  el.textContent = msg;
  $("#stream").appendChild(el);
}

function renderScratchpad(notes) {
  const list = $("#sp-list");
  list.innerHTML = "";
  (notes || []).forEach((n) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="sp-author">${escapeHtml(n.author)}</span>${escapeHtml(n.text)}`;
    list.appendChild(li);
  });
  $("#scratchpad").hidden = !(notes && notes.length);
}

// -- artifact (shared evolving document / code) ----------------------------
function handleArtifact(d) {
  const a = state.artifact;
  a.versions.push(d);
  a.content = d.content;
  a.prev = a.versions.length > 1 ? a.versions[a.versions.length - 2].content : "";
  $("#artifact").hidden = false;
  renderArtifact();
}

function setSeg(which) {
  $("#artifact-view-preview").classList.toggle("active", which === "preview");
  $("#artifact-view-diff").classList.toggle("active", which === "diff");
}

function renderArtifact() {
  const a = state.artifact;
  if (!a.versions.length) return;
  const v = a.versions[a.versions.length - 1];
  $("#artifact-meta").textContent = `v${v.version} · ${v.author} [${v.role}] · round ${v.round}`;
  const body = $("#artifact-body");
  if (a.view === "diff") {
    body.className = "artifact-body diff";
    body.innerHTML = renderDiff(a.prev || "", a.content || "");
  } else {
    body.className = "artifact-body";
    const ext = $("#artifact-ext").value;
    body.innerHTML = (ext === ".md" || ext === ".txt")
      ? renderMarkdown(a.content || "")
      : `<pre><code>${escapeHtml(a.content || "")}</code></pre>`;
  }
}

// Minimal LCS line diff for the Diff view.
function renderDiff(oldText, newText) {
  const a = oldText.split("\n"), b = newText.split("\n");
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const rows = [];
  const push = (cls, sign, line) =>
    rows.push(`<div class="dline ${cls}">${escapeHtml(sign + " " + line)}</div>`);
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { push("d-ctx", " ", a[i]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { push("d-del", "-", a[i]); i++; }
    else { push("d-add", "+", b[j]); j++; }
  }
  while (i < n) push("d-del", "-", a[i++]);
  while (j < m) push("d-add", "+", b[j++]);
  return rows.join("") || '<div class="dline d-ctx">(no changes)</div>';
}

function downloadArtifact() {
  const blob = new Blob([state.artifact.content || ""], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "artifact" + $("#artifact-ext").value;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

// -- workspace (real files edited on disk) ---------------------------------
function handleWorkspaceEdit(d) {
  const w = state.workspace;
  if (!(d.path in w.files)) w.order.push(d.path);
  w.files[d.path] = d;
  if (!w.selected || w.selected === d.path) w.selected = d.path;
  $("#workspace").hidden = false;
  renderWorkspace();
}

function renderWorkspace() {
  const w = state.workspace;
  const list = $("#ws-files");
  list.innerHTML = "";
  w.order.forEach((path) => {
    const f = w.files[path];
    const li = document.createElement("li");
    li.className = "ws-file" + (path === w.selected ? " active" : "");
    li.innerHTML =
      `<span class="ws-dot ${f.status}"></span>` +
      `<span class="ws-name">${escapeHtml(path)}</span>` +
      `<span class="ws-stat"><span class="d-add">+${f.additions}</span> ` +
      `<span class="d-del">-${f.deletions}</span></span>`;
    li.addEventListener("click", () => { w.selected = path; renderWorkspace(); });
    list.appendChild(li);
  });
  const sel = w.files[w.selected];
  $("#ws-diff").innerHTML = sel ? renderUnifiedDiff(sel.diff) : "";
}

// -- conductor team roster -------------------------------------------------
const TEAM_BADGES = {
  idle:      { label: "idle",      icon: "○" },
  assigned:  { label: "assigned",  icon: "→" },
  delivered: { label: "delivered", icon: "•" },
  ok:        { label: "approved",  icon: "✓" },
  warned:    { label: "called out", icon: "⚠" },
};

function seedTeam(agents) {
  const t = state.team;
  t.conductor = (agents && agents.conductor) || "Conductor";
  t.order = Object.keys(agents || {}).filter((k) => k.startsWith("worker"))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  t.order.forEach((k) => { t.workers[k] = { name: agents[k], status: "idle", note: "", round: 0 }; });
  $("#team").hidden = t.order.length === 0;
  renderTeam();
}

function handleWorkerStatus(d) {
  const t = state.team;
  if (!t.order.includes(d.worker)) { t.order.push(d.worker); }
  t.workers[d.worker] = { name: d.name, status: d.status, note: d.note || "", round: d.round };
  if (d.round > t.round) t.round = d.round;
  $("#team").hidden = false;
  renderTeam();
}

function renderTeam() {
  const t = state.team;
  $("#team-meta").textContent =
    `conductor: ${t.conductor}` + (t.round ? `  ·  round ${t.round}` : "");
  const list = $("#team-roster");
  list.innerHTML = "";
  t.order.forEach((key) => {
    const w = t.workers[key];
    const b = TEAM_BADGES[w.status] || TEAM_BADGES.idle;
    const li = document.createElement("li");
    li.className = "team-row st-" + w.status;
    li.innerHTML =
      `<span class="team-ava">${escapeHtml(initials(w.name))}</span>` +
      `<span class="team-main"><span class="team-name">${escapeHtml(w.name)} ` +
      `<small>${escapeHtml(key)}</small></span>` +
      (w.note ? `<span class="team-note">${escapeHtml(w.note)}</span>` : "") +
      `</span>` +
      `<span class="team-badge st-${w.status}">${b.icon} ${b.label}</span>`;
    list.appendChild(li);
  });
}

// Colorize a unified diff produced by difflib.
function renderUnifiedDiff(diff) {
  if (!diff) return '<div class="dline d-ctx">(no textual changes)</div>';
  return diff.split("\n").map((ln) => {
    let cls = "d-ctx";
    if (ln.startsWith("+") && !ln.startsWith("+++")) cls = "d-add";
    else if (ln.startsWith("-") && !ln.startsWith("---")) cls = "d-del";
    else if (ln.startsWith("@@")) cls = "d-hunk";
    else if (ln.startsWith("+++") || ln.startsWith("---")) cls = "d-meta";
    return `<div class="dline ${cls}">${escapeHtml(ln)}</div>`;
  }).join("");
}

function addResult(content) {
  const el = document.createElement("div");
  el.className = "result-card";
  const head = document.createElement("div");
  head.className = "result-head";
  const h3 = document.createElement("h3");
  h3.textContent = "Final deliverable";
  head.appendChild(h3);
  head.appendChild(makeCopyBtn(content));
  const body = document.createElement("div");
  body.className = "turn-body";
  body.innerHTML = renderMarkdown(content);
  el.appendChild(head);
  el.appendChild(body);
  $("#stream").appendChild(el);
  el.scrollIntoView({ behavior: "smooth", block: "end" });
}

function finish(status) {
  if (state.es) { state.es.close(); state.es = null; }
  $("#run").disabled = false;
  $("#stop").disabled = true;
  if (status === "done") { setConn("done", "done"); setStatus("Done."); }
  else if (status === "stopped") { setConn("stopped", "idle"); setStatus("Stopped."); }
  else { setConn("error", "error"); }
}

// -- theme (light / dark) --------------------------------------------------
function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  const btn = $("#theme-toggle");
  if (btn) btn.textContent = t === "light" ? "☀" : "☾";
}
function initTheme() {
  applyTheme(localStorage.getItem("ao-theme") || "dark");
  $("#theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    localStorage.setItem("ao-theme", next);
    applyTheme(next);
  });
}

// -- settings --------------------------------------------------------------
function openSettings() { $("#settings-overlay").hidden = false; loadSettings(); }
function closeSettings() { $("#settings-overlay").hidden = true; }

const SET_FIELDS = {
  anthropic: ["set-anthropic-key", "set-anthropic-model", "set-anthropic-base"],
  openai: ["set-openai-key", "set-openai-model", "set-openai-base"],
  local: ["set-local-key", "set-local-model", "set-local-base"],
};

async function loadSettings() {
  try {
    const d = await (await fetch("/api/settings")).json();
    $("#set-proxy").value = d.proxy || "";
    $("#set-proxy").placeholder = d.proxy_from_env
      ? "set via environment" : "http://proxy.corp:8080 (blank = direct)";
    for (const [prov, [keyId, modelId, baseId]] of Object.entries(SET_FIELDS)) {
      const p = d.providers[prov];
      $("#" + modelId).value = p.model || "";
      $("#" + baseId).value = p.base_url || "";
      const k = $("#" + keyId);
      k.value = "";
      k.placeholder = p.key_from_env ? "set via environment"
        : (p.key_set ? "saved (hidden)" : "not set");
    }
    $("#set-local-proxy").checked = !!(d.providers.local && d.providers.local.use_proxy);
  } catch (e) { $("#settings-status").textContent = "Failed to load: " + e; }
}

async function saveSettings() {
  const v = (id) => $("#" + id).value;
  const payload = {
    proxy: v("set-proxy"),
    anthropic_api_key: v("set-anthropic-key"), anthropic_model: v("set-anthropic-model"),
    anthropic_base_url: v("set-anthropic-base"),
    openai_api_key: v("set-openai-key"), openai_model: v("set-openai-model"),
    openai_base_url: v("set-openai-base"),
    local_api_key: v("set-local-key"), local_model: v("set-local-model"),
    local_base_url: v("set-local-base"),
    local_use_proxy: $("#set-local-proxy").checked,
  };
  const st = $("#settings-status");
  st.textContent = "Saving…";
  try {
    await fetch("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await refreshAvailability();
    await loadSettings();
    st.textContent = "Saved ✓";
  } catch (e) { st.textContent = "Save failed: " + e; }
}

// Re-check backend availability after settings change, preserving selections.
async function refreshAvailability() {
  const data = await (await fetch("/api/catalog")).json();
  state.agents = data.agents;
  document.querySelectorAll("#roles select, #participants select").forEach((sel) => {
    Array.from(sel.options).forEach((opt) => {
      const ag = state.agents.find((a) => a.id === opt.value);
      if (ag) {
        opt.disabled = !ag.available;
        opt.textContent = ag.available ? ag.label : `${ag.label} — ${ag.reason}`;
      }
    });
  });
}

// -- boot ------------------------------------------------------------------
initTheme();
$("#run").addEventListener("click", run);
$("#stop").addEventListener("click", stop);
$("#settings-open").addEventListener("click", openSettings);
$("#settings-close").addEventListener("click", closeSettings);
$("#settings-save").addEventListener("click", saveSettings);
$("#settings-overlay").addEventListener("click", (e) => {
  if (e.target.id === "settings-overlay") closeSettings();
});
$("#artifact-view-preview").addEventListener("click", () => { state.artifact.view = "preview"; setSeg("preview"); renderArtifact(); });
$("#artifact-view-diff").addEventListener("click", () => { state.artifact.view = "diff"; setSeg("diff"); renderArtifact(); });
$("#artifact-ext").addEventListener("change", renderArtifact);
$("#artifact-copy").addEventListener("click", (e) => copyText(state.artifact.content || "", e.currentTarget));
$("#artifact-download").addEventListener("click", downloadArtifact);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("#settings-overlay").hidden) closeSettings();
});
loadCatalog().catch((e) => setStatus("Failed to load catalog: " + e, true));
