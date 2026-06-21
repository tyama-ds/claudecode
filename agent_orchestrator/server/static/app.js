/* Agent Orchestrator — frontend logic (vanilla JS, no dependencies). */

const state = {
  agents: [],        // catalog of selectable backends
  strategies: [],    // strategy metadata
  es: null,          // active EventSource
  sessionId: null,
  cards: {},         // role-round -> DOM node (thinking -> filled)
  connState: "idle",
};

const $ = (sel) => document.querySelector(sel);

// Which accent bucket a role belongs to (drives the card colour).
function accentFor(role) {
  if (["implementer", "agent_a", "planner"].includes(role)) return "acc-a";
  if (["reviewer", "agent_b", "executor"].includes(role)) return "acc-b";
  return "acc-c"; // synthesizer / anything else
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

  const wrap = $("#roles");
  wrap.innerHTML = "";
  st.roles.forEach((role) => {
    const box = document.createElement("div");
    box.className = "role " + (accentFor(role.key) === "acc-b" ? "b" : "");
    const name = document.createElement("div");
    name.className = "role-name";
    name.textContent = role.label;
    const wrapSel = document.createElement("div");
    wrapSel.className = "select-wrap";
    const select = document.createElement("select");
    select.dataset.role = role.key;
    const preferred = defaultAgentFor(role.key);
    for (const ag of state.agents) {
      const opt = document.createElement("option");
      opt.value = ag.id;
      opt.textContent = ag.available ? ag.label : `${ag.label} — ${ag.reason}`;
      opt.disabled = !ag.available;
      if (ag.id === preferred) opt.selected = true;
      select.appendChild(opt);
    }
    wrapSel.appendChild(select);
    box.appendChild(name);
    box.appendChild(wrapSel);
    wrap.appendChild(box);
  });
}

function rolesPayload() {
  const roles = {};
  document.querySelectorAll("#roles select").forEach((s) => {
    roles[s.dataset.role] = { id: s.value };
  });
  return roles;
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

  const payload = {
    task,
    strategy: $("#strategy").value,
    rounds: parseInt($("#rounds").value, 10) || 2,
    roles: rolesPayload(),
  };

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
  $("#stream").innerHTML = "";
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
    $("#meta").textContent = `${data.strategy} · ${data.rounds} rounds · ${agents}`;
  } else if (type === "turn_start") {
    addThinkingCard(data);
  } else if (type === "turn_end") {
    fillCard(data);
  } else if (type === "status") {
    addNote(data.message);
  } else if (type === "result") {
    addResult(data.content);
  } else if (type === "error") {
    setStatus(data.message, true);
    addNote("⚠ " + data.message);
  } else if (type === "session_end") {
    finish(data.status);
  }
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
  node.querySelector(".turn-body").innerHTML = d.ok ? renderMarkdown(d.content) : escapeHtml(d.content);
  node.scrollIntoView({ behavior: "smooth", block: "end" });
}

function addNote(msg) {
  const el = document.createElement("div");
  el.className = "note";
  el.textContent = msg;
  $("#stream").appendChild(el);
}

function addResult(content) {
  const el = document.createElement("div");
  el.className = "result-card";
  el.innerHTML = `<h3>Final deliverable</h3><div class="turn-body">${renderMarkdown(content)}</div>`;
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

// -- boot ------------------------------------------------------------------
$("#run").addEventListener("click", run);
$("#stop").addEventListener("click", stop);
loadCatalog().catch((e) => setStatus("Failed to load catalog: " + e, true));
