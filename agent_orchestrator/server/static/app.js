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
  board: { order: [], roles: {} },   // live who's-doing-what panel
  graph: { pos: null, names: {}, sustained: [], strategy: "" },  // interaction graph
  tab: "stream",     // active feed tab
  log: [],           // finished turns, for the Export button
  lang: localStorage.getItem("ao-lang") || "en",
};

const $ = (sel) => document.querySelector(sel);

// -- i18n (EN default, JA toggle) -------------------------------------------
const JA = {
  // compose rail
  "Compose": "実行設定",
  "Task": "タスク",
  "Describe the task for the agents to collaborate on…": "エージェントに協働してもらうタスクを記述…",
  "Strategy": "戦略",
  "Rounds": "ラウンド数",
  "Workspace": "ワークスペース",
  "Directory": "ディレクトリ",
  "blank = server's launch directory": "空欄＝サーバ起動ディレクトリ",
  "Create the directory if it doesn't exist": "ディレクトリが無ければ作成",
  "(no git)": "（git なし）",
  "Reference directory": "参照ディレクトリ",
  "(optional, read-only)": "（任意・読み取り専用）",
  "a local folder whose files the agents may consult": "エージェントに読ませたいローカルフォルダ",
  "Run collaboration": "コラボレーションを実行",
  "Stop": "停止",
  "Tip: Ctrl+Enter (⌘+Enter) in the task box runs it": "ヒント: タスク欄で Ctrl+Enter（⌘+Enter）でも実行できます",
  // feed
  "Transcript": "トランスクリプト",
  "Artifact": "成果物",
  "Team": "チーム",
  "Shared scratchpad": "共有メモ",
  "Preview": "プレビュー",
  "Diff": "差分",
  "Copy": "コピー",
  "Copied": "コピー済",
  "Download": "ダウンロード",
  "Export": "書き出し",
  "Latest": "最新へ",
  "Two agents, one task.": "2つのエージェント、1つのタスク。",
  "Final deliverable": "最終成果物",
  "round": "ラウンド",
  "thinking": "考え中",
  // history
  "Recent sessions": "最近のセッション",
  "No sessions yet.": "セッションはまだありません。",
  // statuses
  "Please enter a task.": "タスクを入力してください。",
  "Starting…": "開始中…",
  "Collaboration running…": "コラボレーション実行中…",
  "Done.": "完了しました。",
  "Stopped.": "停止しました。",
  "Stop requested…": "停止をリクエストしました…",
  "Session ended with an error.": "セッションはエラーで終了しました。",
  "Failed to start.": "開始に失敗しました。",
  "Network error: ": "ネットワークエラー: ",
  // interaction graph captions
  "designing…": "設計中…",
  "implementing…": "実装中…",
  "reviewing…": "レビュー中…",
  "working…": "作業中…",
  "assign": "指示",
  "deliver": "提出",
  "approve": "承認 ✓",
  "request changes": "修正依頼",
  "call out": "指摘",
  "propose design": "設計を提案",
  "discuss design": "設計を議論",
  // agent board
  "Agents": "エージェント",
  "waiting": "待機",
  "working": "作業中",
  "done": "完了",
  "failed": "失敗",
  "design": "設計",
  "implement": "実装",
  "review": "レビュー",
  // role builder
  "Persona": "ペルソナ",
  "Implementer": "実装役",
  "+ Add reviewer": "+ レビュアーを追加",
  "model (optional — uses default)": "モデル（任意・未入力なら既定値）",
  "Leave blank to use the default persona below": "空欄なら下記の既定ペルソナを使用",
  "Describe this participant's role / character (optional)": "この参加者の役割・キャラクターを記述（任意）",
  "Default: ": "既定: ",
  "+ Add worker": "+ ワーカーを追加",
  "+ Add participant": "+ 参加者を追加",
  "Worker": "ワーカー",
  "Participant": "参加者",
  "Conductor": "コンダクター",
  "Reviewer": "レビュアー",
  // strategy groups
  "Discuss & decide": "議論・検討",
  "Author together": "共同作成",
  "Team play": "チーム編成",
  // team badges
  "idle": "待機",
  "assigned": "割当済",
  "delivered": "提出済",
  "approved": "承認",
  "called out": "要注意",
  // settings modal
  "Settings": "設定",
  "Used when the matching environment variable isn't set. Keys are held in memory for this local server only and are never shown back. Works with any OpenAI-compatible provider via its base URL.":
    "対応する環境変数が未設定のときに使われます。キーはこのローカルサーバのメモリ上にのみ保持され、再表示されません。Base URL を指定すれば OpenAI 互換の任意のプロバイダで使えます。",
  "Network": "ネットワーク",
  "HTTP(S) proxy": "HTTP(S) プロキシ",
  "API key": "APIキー",
  "API key (optional)": "APIキー（任意）",
  "Model": "モデル",
  "Base URL": "ベースURL",
  "Send via proxy": "プロキシ経由で送信",
  "(off = direct connection)": "（オフ＝直接接続）",
  "Save": "保存",
  "Saved ✓": "保存しました ✓",
  "Saving…": "保存中…",
  // footer
  "Runs on the Python standard library — no installs, no telemetry.":
    "Python 標準ライブラリのみで動作 — インストール不要・テレメトリなし。",
};

const EMPTY_COPY = {
  en: "Configure the collaboration on the left and press <em>Run</em>. " +
      "Each agent's turn streams in here as it happens.",
  ja: "左側でコラボレーションを設定して<em>実行</em>を押してください。" +
      "各エージェントのターンがここにリアルタイムで表示されます。",
};

function t(s) { return state.lang === "ja" && JA[s] ? JA[s] : s; }

function applyLang() {
  document.documentElement.lang = state.lang;
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    if (!el.dataset.en) el.dataset.en = el.textContent.replace(/\s+/g, " ").trim();
    el.textContent = t(el.dataset.en);
  });
  document.querySelectorAll("[data-i18n-ph]").forEach((el) => {
    if (!el.dataset.enPh) el.dataset.enPh = el.getAttribute("placeholder") || "";
    el.setAttribute("placeholder", t(el.dataset.enPh));
  });
  const copy = $("#empty-copy");
  if (copy) copy.innerHTML = EMPTY_COPY[state.lang] || EMPTY_COPY.en;
  $("#lang-toggle").textContent = state.lang === "ja" ? "EN" : "JA";
}

function toggleLang() {
  state.lang = state.lang === "ja" ? "en" : "ja";
  localStorage.setItem("ao-lang", state.lang);
  applyLang();
  renderRoles(); // rebuild role cards with translated labels
}

// Which accent bucket a role belongs to (drives the card colour).
function accentFor(role) {
  if (["implementer", "agent_a", "planner", "conductor"].includes(role)) return "acc-a";
  if (["reviewer", "agent_b", "executor"].includes(role)) return "acc-b";
  const rv = /^reviewer_(\d+)$/.exec(role); // workspace_build review panel
  if (rv) return ["acc-b", "acc-c"][(parseInt(rv[1], 10) - 1) % 2];
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
// Human grouping of the strategy list (anything unknown lands in the first group).
const STRATEGY_GROUPS = [
  ["Discuss & decide", ["implementer_reviewer", "debate_consensus", "planner_executor",
    "round_robin", "panel_judge", "custom"]],
  ["Author together", ["doc_authoring", "code_authoring", "workspace_build"]],
  ["Team play", ["conductor_team"]],
];

async function loadCatalog() {
  const res = await fetch("/api/catalog");
  const data = await res.json();
  state.agents = data.agents;
  state.strategies = data.strategies;

  const sel = $("#strategy");
  sel.innerHTML = "";
  const grouped = new Set(STRATEGY_GROUPS.flatMap(([, names]) => names));
  for (const [label, names] of STRATEGY_GROUPS) {
    const members = state.strategies.filter((s) => names.includes(s.name));
    const extras = label === STRATEGY_GROUPS[0][0]
      ? state.strategies.filter((s) => !grouped.has(s.name)) : [];
    if (!members.length && !extras.length) continue;
    const og = document.createElement("optgroup");
    og.label = t(label);
    for (const st of [...members, ...extras]) {
      const opt = document.createElement("option");
      opt.value = st.name;
      opt.textContent = st.name.replace(/_/g, " ");
      opt.title = st.description;
      og.appendChild(opt);
    }
    sel.appendChild(og);
  }
  sel.addEventListener("change", () => { renderRoles(); persistForm(); });
  restoreForm("strategy");
  renderRoles();
  restoreForm("fields");
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
  if (st.name === "workspace_build") { renderWorkspaceBuilder(wrap, st); return; }
  if (st.custom) { renderCustomBuilder(wrap); return; }
  st.roles.forEach((role) => wrap.appendChild(roleCard(role.key, role.label, role.system || "", false)));
}

// Workspace build: one implementer + 1–3 reviewers (add/remove).
function renderWorkspaceBuilder(wrap, st) {
  const meta = Object.fromEntries((st.roles || []).map((r) => [r.key, r]));
  const implSys = (meta.implementer || {}).system || "";
  const revSys = (meta.reviewer || {}).system || "";
  wrap.appendChild(roleCard("implementer", t("Implementer"), implSys, false));

  const holder = document.createElement("div");
  holder.id = "reviewers"; holder.className = "roles";
  wrap.appendChild(holder);
  const add = document.createElement("button");
  add.type = "button"; add.className = "btn ghost add-reviewer";
  add.textContent = t("+ Add reviewer");
  add.addEventListener("click", () => addReviewer(holder, revSys));
  wrap.appendChild(add);
  addReviewer(holder, revSys);
}

function addReviewer(holder, sys) {
  if (holder.children.length >= 3) return;
  const i = holder.children.length + 1;
  holder.appendChild(roleCard("reviewer_" + i, t("Reviewer") + " " + i, sys, true));
  relabelReviewers(holder);
}

function relabelReviewers(holder) {
  Array.from(holder.children).forEach((card, i) => {
    const key = "reviewer_" + (i + 1);
    const bucket = accentFor(key);
    card.className = "role " + (bucket === "acc-b" ? "b" : bucket === "acc-c" ? "c" : "");
    card.querySelector(".role-name").childNodes[0].nodeValue = t("Reviewer") + " " + (i + 1);
    card.querySelector("select").dataset.role = key;
  });
  const add = document.querySelector(".add-reviewer");
  if (add) add.disabled = holder.children.length >= 3;
}

// Conductor team: one conductor, 2–4 workers (add/remove), one reviewer.
const CONDUCTOR_DEFAULTS = {
  conductor: "Leads the team: decomposes the task, assigns subtasks, holds workers " +
    "accountable (calling out anyone who doesn't deliver), and integrates the result.",
  worker: "Carries out the specific assignment the conductor gives, concretely and in full.",
  reviewer: "Reviews each worker's output against its assignment and reports back to the conductor.",
};
function renderConductorBuilder(wrap) {
  wrap.appendChild(roleCard("conductor", t("Conductor"), CONDUCTOR_DEFAULTS.conductor, false));

  const holder = document.createElement("div");
  holder.id = "workers"; holder.className = "roles";
  wrap.appendChild(holder);
  const add = document.createElement("button");
  add.type = "button"; add.className = "btn ghost add-worker";
  add.textContent = t("+ Add worker");
  add.addEventListener("click", () => addWorker(holder));
  wrap.appendChild(add);
  addWorker(holder);
  addWorker(holder);

  wrap.appendChild(roleCard("reviewer", t("Reviewer"), CONDUCTOR_DEFAULTS.reviewer, false));
}

function addWorker(holder) {
  if (holder.children.length >= 4) return;
  const i = holder.children.length + 1;
  holder.appendChild(roleCard("worker_" + i, t("Worker") + " " + i, CONDUCTOR_DEFAULTS.worker, true));
  relabelWorkers(holder);
}

function relabelWorkers(holder) {
  Array.from(holder.children).forEach((card, i) => {
    const key = "worker_" + (i + 1);
    const bucket = accentFor(key);
    card.className = "role " + (bucket === "acc-b" ? "b" : bucket === "acc-c" ? "c" : "");
    card.querySelector(".role-name").childNodes[0].nodeValue = t("Worker") + " " + (i + 1);
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
      const h = box.parentElement;
      if (h && h.id === "reviewers" && h.children.length <= 1) return; // keep ≥1 reviewer
      box.remove();
      if (h && h.id === "workers") relabelWorkers(h);
      else if (h && h.id === "reviewers") relabelReviewers(h);
      else relabelParticipants(h);
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
  model.placeholder = t("model (optional — uses default)");
  box.appendChild(model);

  const det = document.createElement("details");
  det.className = "persona";
  const sum = document.createElement("summary");
  sum.textContent = t("Persona");
  det.appendChild(sum);
  const ta = document.createElement("textarea");
  ta.className = "persona-in"; ta.rows = 3;
  ta.placeholder = defaultSystem
    ? t("Leave blank to use the default persona below")
    : t("Describe this participant's role / character (optional)");
  det.appendChild(ta);
  if (defaultSystem) {
    const hint = document.createElement("div");
    hint.className = "persona-default";
    hint.textContent = t("Default: ") + defaultSystem;
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
  add.textContent = t("+ Add participant");
  add.addEventListener("click", () => addParticipant(holder));
  wrap.appendChild(add);
  addParticipant(holder);
  addParticipant(holder);
}

function addParticipant(holder) {
  if (holder.children.length >= 5) return;
  const i = holder.children.length;
  holder.appendChild(roleCard("agent_" + (i + 1), t("Participant") + " " + (i + 1), "", true));
  relabelParticipants(holder);
}

function relabelParticipants(holder) {
  Array.from(holder.children).forEach((card, i) => {
    const key = "agent_" + (i + 1);
    const bucket = accentFor(key);
    card.className = "role " + (bucket === "acc-b" ? "b" : bucket === "acc-c" ? "c" : "");
    card.querySelector(".role-name").childNodes[0].nodeValue = t("Participant") + " " + (i + 1);
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
  if (st && st.name === "workspace_build") {
    const order = ["implementer"];
    roles.implementer = collectRole(document.querySelector("#roles > .role"));
    document.querySelectorAll("#reviewers > .role").forEach((card, i) => {
      const key = "reviewer_" + (i + 1);
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

// -- form persistence (task, strategy, dirs survive a reload) ---------------
const PERSIST_FIELDS = ["task", "rounds", "workspace-dir", "reference-dir"];

function persistForm() {
  const data = { strategy: $("#strategy").value };
  PERSIST_FIELDS.forEach((id) => { data[id] = $("#" + id).value; });
  data["workspace-init"] = $("#workspace-init").checked;
  try { localStorage.setItem("ao-form", JSON.stringify(data)); } catch { /* ignore */ }
}

function restoreForm(phase) {
  let data;
  try { data = JSON.parse(localStorage.getItem("ao-form") || "{}"); } catch { return; }
  if (phase === "strategy") {
    if (data.strategy) $("#strategy").value = data.strategy;
    if (!$("#strategy").value && state.strategies[0]) $("#strategy").value = state.strategies[0].name;
    return;
  }
  // phase "fields": after renderRoles(), which resets rounds to the default
  PERSIST_FIELDS.forEach((id) => { if (data[id] != null && data[id] !== "") $("#" + id).value = data[id]; });
  if (data["workspace-init"] != null) $("#workspace-init").checked = data["workspace-init"];
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

// Reset every per-run view. Also called on session_start so an SSE reconnect
// (which replays the whole event backlog) doesn't duplicate the transcript.
function resetRunUI() {
  state.cards = {};
  state.log = [];
  state.artifact = { versions: [], view: "preview" };
  state.workspace = { files: {}, order: [], selected: null };
  state.team = { workers: {}, order: [], conductor: "", round: 0 };
  ["artifact", "workspace", "team"].forEach((name) => {
    const b = document.querySelector(`.ftab[data-tab="${name}"]`);
    if (b) { b.hidden = true; b.classList.remove("badge"); }
  });
  setTab("stream");
  $("#stream").innerHTML = "";
  $("#sp-list").innerHTML = "";
  $("#scratchpad").hidden = true;
  $("#artifact-body").innerHTML = "";
  $("#ws-files").innerHTML = "";
  $("#ws-diff").innerHTML = "";
  $("#team-roster").innerHTML = "";
  $("#meta").textContent = "";
  $("#export-md").hidden = true;
  setSeg("preview");
  resetBoard();
}

async function run() {
  const task = $("#task").value.trim();
  if (!task) { setStatus(t("Please enter a task."), true); return; }

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
  persistForm();

  setStatus(t("Starting…"));
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
    setStatus(t("Network error: ") + e, true);
    $("#run").disabled = false;
    return;
  }
  if (!res.ok) {
    let msg = data.error || t("Failed to start.");
    if (data.details) msg += " (" + data.details.map((d) => `${d.role}: ${d.reason}`).join("; ") + ")";
    setStatus(msg, true);
    $("#run").disabled = false;
    return;
  }

  state.sessionId = data.session_id;
  resetRunUI();
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
    if (state.connState === "running") setConn("reconnecting", "error");
  };
}

// Reopen a past (or still-running) session from the history list; the event
// bus replays its full backlog, so the whole transcript is reconstructed.
function openSession(id) {
  state.sessionId = id;
  resetRunUI();
  $("#stop").disabled = false; // session_end in the replay re-disables it
  setStatus("");
  openStream(id);
}

async function stop() {
  if (!state.sessionId) return;
  await fetch(`/api/stop/${state.sessionId}`, { method: "POST" });
  setStatus(t("Stop requested…"));
}

// -- event handling --------------------------------------------------------
function handleEvent(evt) {
  const { type, data } = evt;
  if (type === "session_start") {
    resetRunUI(); // idempotent replays: never duplicate the transcript
    setConn("running", "running");
    setStatus(t("Collaboration running…"));
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
    seedBoard(data.agents);
    seedGraph(data.agents, data.strategy);
  } else if (type === "artifact") {
    handleArtifact(data);
  } else if (type === "workspace_edit") {
    handleWorkspaceEdit(data);
    graphEdit(data);
  } else if (type === "worker_status") {
    handleWorkerStatus(data);
    graphWorker(data);
  } else if (type === "turn_start") {
    addThinkingCard(data);
    boardTurn(data, "start");
    graphTurnStart(data);
  } else if (type === "turn_end") {
    fillCard(data);
    boardTurn(data, "end");
    graphTurnEnd(data);
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

// -- live interaction graph (who is talking to whom, animated) ---------------
const SVGNS = "http://www.w3.org/2000/svg";

function svgEl(name, attrs) {
  const el = document.createElementNS(SVGNS, name);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

// Lay out the participants: workspace hub for workspace_build, conductor hub
// for conductor_team, otherwise a ring. "__ws__" is the workspace node.
function seedGraph(agents, strategy) {
  const g = state.graph = { pos: {}, names: { ...(agents || {}) }, sustained: [],
                            strategy };
  const roles = Object.keys(agents || {});
  if (roles.length < 2) { $("#graph").hidden = true; g.pos = null; return; }
  const W = 220, cx = W / 2, cy = 88;
  const place = (role, x, y) => { g.pos[role] = { x: Math.round(x), y: Math.round(y) }; };
  if (strategy === "workspace_build" && roles.includes("implementer")) {
    g.names.__ws__ = "workspace";
    const reviewers = roles.filter((r) => r !== "implementer");
    place("__ws__", cx, cy);
    place("implementer", 32, cy);
    const n = reviewers.length;
    reviewers.forEach((r, i) => {
      const a = n === 1 ? 0 : (i / (n - 1) - 0.5) * 1.7; // fan on the right
      place(r, cx + 76 * Math.cos(a), cy + 62 * Math.sin(a));
    });
  } else if (roles.includes("conductor")) {
    place("conductor", cx, cy);
    const rest = roles.filter((r) => r !== "conductor");
    rest.forEach((r, i) => {
      const a = -Math.PI / 2 + (i * 2 * Math.PI) / rest.length;
      place(r, cx + 76 * Math.cos(a), cy + 62 * Math.sin(a));
    });
  } else {
    const start = roles.length === 2 ? 0 : -Math.PI / 2;
    roles.forEach((r, i) => {
      const a = start + (i * 2 * Math.PI) / roles.length;
      place(r, cx + 74 * Math.cos(a), cy + 60 * Math.sin(a));
    });
  }
  renderGraph();
  $("#graph-caption").textContent = "";
  $("#graph").hidden = false;
}

function renderGraph() {
  const g = state.graph;
  const svg = $("#graph-svg");
  svg.innerHTML = "";
  const roles = Object.keys(g.pos);
  // static edges: spokes to the hub, or a full mesh for rings
  const hub = g.pos.__ws__ ? "__ws__" : (g.pos.conductor ? "conductor" : null);
  const edges = [];
  if (hub) roles.filter((r) => r !== hub).forEach((r) => edges.push([hub, r]));
  else for (let i = 0; i < roles.length; i++)
    for (let j = i + 1; j < roles.length; j++) edges.push([roles[i], roles[j]]);
  edges.forEach(([a, b]) => svg.appendChild(svgEl("line", {
    x1: g.pos[a].x, y1: g.pos[a].y, x2: g.pos[b].x, y2: g.pos[b].y, class: "gedge",
  })));
  roles.forEach((role) => {
    const p = g.pos[role];
    const isWs = role === "__ws__";
    const grp = svgEl("g", {
      class: `gnode ${isWs ? "gws" : accentFor(role)}`,
      "data-role": role, transform: `translate(${p.x},${p.y})`,
    });
    grp.appendChild(svgEl("circle", { r: 19, class: "gring" }));
    grp.appendChild(svgEl("circle", { r: 15, class: "gbody" }));
    const txt = svgEl("text", { class: "gtext", "text-anchor": "middle", dy: isWs ? "4.5" : "3.5" });
    txt.textContent = isWs ? "📁" : initials(g.names[role]);
    grp.appendChild(txt);
    const lbl = svgEl("text", { class: "glabel", "text-anchor": "middle", y: 31 });
    lbl.textContent = isWs ? "workspace" : role;
    grp.appendChild(lbl);
    svg.appendChild(grp);
  });
}

function setGraphActive(role) {
  document.querySelectorAll("#graph-svg .gnode").forEach((n) =>
    n.classList.toggle("active", n.dataset.role === role));
}

// Animate a dot travelling from one node to another (loops=Infinity keeps it
// flowing until the returned cancel function is called).
function graphPulse(from, to, cls = "", loops = 1) {
  const g = state.graph;
  const svg = $("#graph-svg");
  if (!svg || !g.pos || !g.pos[from]) return () => {};
  const stops = [];
  (Array.isArray(to) ? to : [to]).forEach((dst) => {
    if (!g.pos[dst] || dst === from) return;
    const p1 = g.pos[from], p2 = g.pos[dst];
    const line = svgEl("line", {
      x1: p1.x, y1: p1.y, x2: p2.x, y2: p2.y, class: "gline-hot " + cls,
    });
    svg.insertBefore(line, svg.querySelector(".gnode"));
    const dot = svgEl("circle", { r: 3.4, cx: p1.x, cy: p1.y, class: "gdot " + cls });
    svg.appendChild(dot);
    const t0 = performance.now(), dur = 900;
    let raf = 0, done = false;
    const stop = () => {
      if (done) return;
      done = true; cancelAnimationFrame(raf); dot.remove(); line.remove();
    };
    const step = (now) => {
      if (done) return;
      const k = (now - t0) / dur;
      if (loops !== Infinity && k >= loops) { stop(); return; }
      const f = k % 1;
      dot.setAttribute("cx", p1.x + (p2.x - p1.x) * f);
      dot.setAttribute("cy", p1.y + (p2.y - p1.y) * f);
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    stops.push(stop);
  });
  return () => stops.forEach((s) => s());
}

function clearSustainedPulses() {
  state.graph.sustained.forEach((cancel) => cancel());
  state.graph.sustained = [];
}

function graphCaption(text) {
  const el = $("#graph-caption");
  el.textContent = text;
  el.classList.remove("flash");
  void el.offsetWidth; // restart the entrance animation
  el.classList.add("flash");
}

function graphTurnStart(d) {
  const g = state.graph;
  if (!g.pos || !g.pos[d.role]) return;
  setGraphActive(d.role);
  clearSustainedPulses();
  const others = Object.keys(g.pos).filter((r) => r !== d.role && r !== "__ws__");
  const name = g.names[d.role] || d.role;
  if (d.action === "implement" && g.pos.__ws__) {
    g.sustained.push(graphPulse(d.role, "__ws__", "", Infinity));
    graphCaption(`${name} · ${t("implementing…")}`);
  } else if (d.action === "review" && g.pos.__ws__) {
    g.sustained.push(graphPulse("__ws__", d.role, "", Infinity));
    graphCaption(`${name} · ${t("reviewing…")}`);
  } else if (d.action === "design") {
    graphPulse(d.role, others, "soft");
    graphCaption(`${name} · ${t("designing…")}`);
  } else {
    graphPulse(d.role, others, "soft");
    graphCaption(`${name} · ${t("working…")}`);
  }
}

function graphTurnEnd(d) {
  const g = state.graph;
  if (!g.pos) return;
  clearSustainedPulses();
  setGraphActive(null);
  if (!g.pos[d.role]) return;
  const names = g.names;
  if (d.action === "design" && d.role !== "implementer" && g.pos.implementer) {
    graphPulse(d.role, "implementer", "soft");
    graphCaption(`${names[d.role]} → ${names.implementer} · ${t("discuss design")}`);
  } else if (d.action === "design" && d.role === "implementer") {
    graphCaption(`${names[d.role]} · ${t("propose design")}`);
  } else if (d.action === "review" && d.ok && g.pos.implementer) {
    const up = (d.content || "").toUpperCase();
    if (up.includes("REQUEST CHANGES")) {
      graphPulse(d.role, "implementer", "bad");
      graphCaption(`${names[d.role]} → ${names.implementer} · ${t("request changes")}`);
    } else if (up.includes("APPROVE")) {
      graphPulse(d.role, "implementer", "good");
      graphCaption(`${names[d.role]} → ${names.implementer} · ${t("approve")}`);
    }
  }
}

// Conductor-team arrows come straight from worker_status events.
function graphWorker(d) {
  const g = state.graph;
  if (!g.pos || !g.pos.conductor || !g.pos[d.worker]) return;
  const map = {
    assigned:  ["conductor", d.worker, "", "assign"],
    delivered: [d.worker, "conductor", "soft", "deliver"],
    ok:        ["conductor", d.worker, "good", "approve"],
    warned:    ["conductor", d.worker, "bad", "call out"],
  };
  const m = map[d.status];
  if (!m) return;
  graphPulse(m[0], m[1], m[2]);
  graphCaption(`${g.names[m[0]] || m[0]} → ${g.names[m[1]] || m[1]} · ${t(m[3])}`);
}

// A file landing in the workspace: pulse author → workspace with the path.
function graphEdit(d) {
  const g = state.graph;
  if (!g.pos || !g.pos.__ws__ || !g.pos[d.role]) return;
  graphPulse(d.role, "__ws__", "good");
  graphCaption(`${d.author} → 📁 ${d.path}`);
}

function resetGraph() {
  clearSustainedPulses();
  state.graph = { pos: null, names: {}, sustained: [], strategy: "" };
  $("#graph").hidden = true;
  $("#graph-svg").innerHTML = "";
  $("#graph-caption").textContent = "";
}

// -- live agent board (who's doing what, right now) --------------------------
function seedBoard(agents) {
  const b = state.board;
  b.order = Object.keys(agents || {});
  b.roles = {};
  b.order.forEach((r) => {
    b.roles[r] = { name: agents[r], state: "waiting", action: "", round: 0,
                   start: 0, duration: null };
  });
  const show = b.order.length > 0;
  $("#board").hidden = !show;
  document.querySelector(".console").classList.toggle("has-board", show);
  renderBoard();
}

function boardTurn(d, phase) {
  const e = state.board.roles[d.role];
  if (!e) return;
  if (phase === "start") {
    e.state = "working"; e.action = d.action || ""; e.round = d.round;
    e.start = Date.now(); e.duration = null;
  } else {
    e.state = d.ok ? "done" : "failed";
    if (d.action) e.action = d.action;
    e.round = d.round; e.duration = d.duration; e.start = 0;
  }
  renderBoard();
}

function renderBoard() {
  const list = $("#board-list");
  list.innerHTML = "";
  state.board.order.forEach((role) => {
    const e = state.board.roles[role];
    const li = document.createElement("li");
    li.className = `brow st-${e.state} ${accentFor(role)}`;
    const bits = [role];
    if (e.action) bits.push(t(e.action));
    if (e.round) bits.push("r" + e.round);
    const time = e.duration != null ? " · " + e.duration + "s" : "";
    li.innerHTML =
      `<span class="board-ava">${escapeHtml(initials(e.name))}</span>` +
      `<span class="board-main">` +
      `<span class="board-top"><span class="board-name">${escapeHtml(e.name)}</span>` +
      `<span class="board-state"><span class="bdot"></span>${escapeHtml(t(e.state))}</span></span>` +
      `<span class="board-sub">${escapeHtml(bits.join(" · "))}` +
      `<span class="board-time" data-role="${escapeHtml(role)}">${escapeHtml(time)}</span>` +
      `</span></span>`;
    list.appendChild(li);
  });
}

function resetBoard() {
  state.board = { order: [], roles: {} };
  $("#board").hidden = true;
  $("#board-list").innerHTML = "";
  document.querySelector(".console").classList.remove("has-board");
  resetGraph();
}

// -- feed tabs ---------------------------------------------------------------
const TAB_PANES = { stream: "#tab-stream", artifact: "#tab-artifact",
  workspace: "#tab-workspace", team: "#tab-team" };

function setTab(name) {
  state.tab = name;
  document.querySelectorAll(".ftab").forEach((b) => {
    const active = b.dataset.tab === name;
    b.classList.toggle("active", active);
    if (active) b.classList.remove("badge");
  });
  for (const [tab, sel] of Object.entries(TAB_PANES)) $(sel).hidden = tab !== name;
}

// Show a tab button once its pane has content; dot-badge it if not focused.
function revealTab(name) {
  const btn = document.querySelector(`.ftab[data-tab="${name}"]`);
  if (!btn) return;
  btn.hidden = false;
  if (state.tab !== name) btn.classList.add("badge");
}

// -- smart autoscroll --------------------------------------------------------
function nearBottom() {
  return window.innerHeight + window.scrollY
    >= document.documentElement.scrollHeight - 240;
}
function autoScroll(node) {
  if (nearBottom()) node.scrollIntoView({ behavior: "smooth", block: "end" });
  else $("#jump-latest").hidden = false;
}

// One-click copy of an agent's raw (un-rendered) output.
function makeCopyBtn(text) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "copy-btn";
  b.title = "Copy output";
  b.textContent = t("Copy");
  b.addEventListener("click", (e) => { e.stopPropagation(); copyText(text, b); });
  return b;
}
function copyText(text, btn) {
  const flash = () => {
    btn.textContent = t("Copied");
    btn.classList.add("copied");
    setTimeout(() => { btn.textContent = t("Copy"); btn.classList.remove("copied"); }, 1200);
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
  node.dataset.start = Date.now();
  node.innerHTML = `
    <div class="turn-head">
      <span class="avatar">${escapeHtml(initials(d.agent))}</span>
      <span class="who">
        <span class="agent">${escapeHtml(d.agent)}</span>
        <span class="role-tag">${escapeHtml(d.role)}</span>
      </span>
      <span class="round">${t("round")} ${d.round}</span>
      <span class="dur"></span>
    </div>
    <div class="turn-body">${t("thinking")}</div>`;
  $("#stream").appendChild(node);
  state.cards[cardKey(d)] = node;
  autoScroll(node);
}

// Live elapsed-seconds counter on every in-progress turn and board row.
setInterval(() => {
  document.querySelectorAll(".turn.thinking").forEach((n) => {
    const t0 = parseInt(n.dataset.start || "0", 10);
    if (t0) n.querySelector(".dur").textContent = Math.round((Date.now() - t0) / 1000) + "s";
  });
  for (const role of state.board.order) {
    const e = state.board.roles[role];
    if (e.state !== "working" || !e.start) continue;
    const el = document.querySelector(`.board-time[data-role="${role}"]`);
    if (el) el.textContent = " · " + Math.round((Date.now() - e.start) / 1000) + "s";
  }
}, 1000);

function fillCard(d) {
  const node = state.cards[cardKey(d)] || (() => { addThinkingCard(d); return state.cards[cardKey(d)]; })();
  node.classList.remove("thinking");
  if (!d.ok) node.classList.add("failed");
  node.querySelector(".dur").textContent = d.duration != null ? `${d.duration}s` : "";
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
  state.log.push({ agent: d.agent, role: d.role, round: d.round, content: d.content, ok: d.ok });
  $("#export-md").hidden = false;
  autoScroll(node);
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
  revealTab("artifact");
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
  $("#artifact-meta").textContent = `v${v.version} · ${v.author} [${v.role}] · ${t("round")} ${v.round}`;
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

function downloadBlob(text, filename) {
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function downloadArtifact() {
  downloadBlob(state.artifact.content || "", "artifact" + $("#artifact-ext").value);
}

// Export the whole transcript as a markdown file.
function exportTranscript() {
  const lines = [`# Agent Orchestrator — transcript`, ""];
  const meta = $("#meta").textContent;
  if (meta) lines.push(`> ${meta}`, "");
  for (const e of state.log) {
    if (e.final) lines.push(`## ${t("Final deliverable")}`, "", e.content, "");
    else lines.push(`## ${e.agent} [${e.role}] — ${t("round")} ${e.round}${e.ok ? "" : " (failed)"}`,
      "", e.content, "");
  }
  downloadBlob(lines.join("\n"), `transcript-${state.sessionId || "session"}.md`);
}

// -- workspace (real files edited on disk) ---------------------------------
function handleWorkspaceEdit(d) {
  const w = state.workspace;
  if (!(d.path in w.files)) w.order.push(d.path);
  w.files[d.path] = d;
  if (!w.selected || w.selected === d.path) w.selected = d.path;
  revealTab("workspace");
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
  if (t.order.length) revealTab("team");
  renderTeam();
}

function handleWorkerStatus(d) {
  const t = state.team;
  if (!t.order.includes(d.worker)) { t.order.push(d.worker); }
  t.workers[d.worker] = { name: d.name, status: d.status, note: d.note || "", round: d.round };
  if (d.round > t.round) t.round = d.round;
  revealTab("team");
  renderTeam();
}

function renderTeam() {
  const tm = state.team;
  $("#team-meta").textContent =
    `conductor: ${tm.conductor}` + (tm.round ? `  ·  ${t("round")} ${tm.round}` : "");
  const list = $("#team-roster");
  list.innerHTML = "";
  tm.order.forEach((key) => {
    const w = tm.workers[key];
    const b = TEAM_BADGES[w.status] || TEAM_BADGES.idle;
    const li = document.createElement("li");
    li.className = "team-row st-" + w.status;
    li.innerHTML =
      `<span class="team-ava">${escapeHtml(initials(w.name))}</span>` +
      `<span class="team-main"><span class="team-name">${escapeHtml(w.name)} ` +
      `<small>${escapeHtml(key)}</small></span>` +
      (w.note ? `<span class="team-note">${escapeHtml(w.note)}</span>` : "") +
      `</span>` +
      `<span class="team-badge st-${w.status}">${b.icon} ${t(b.label)}</span>`;
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
  h3.textContent = t("Final deliverable");
  head.appendChild(h3);
  head.appendChild(makeCopyBtn(content));
  const body = document.createElement("div");
  body.className = "turn-body";
  body.innerHTML = renderMarkdown(content);
  el.appendChild(head);
  el.appendChild(body);
  $("#stream").appendChild(el);
  state.log.push({ final: true, content });
  $("#export-md").hidden = false;
  autoScroll(el);
}

function finish(status) {
  if (state.es) { state.es.close(); state.es = null; }
  clearSustainedPulses();
  setGraphActive(null);
  $("#run").disabled = false;
  $("#stop").disabled = true;
  if (status === "done") { setConn("done", "done"); setStatus(t("Done.")); }
  else if (status === "stopped") { setConn("stopped", "idle"); setStatus(t("Stopped.")); }
  else { setConn("error", "error"); setStatus(t("Session ended with an error."), true); }
}

// -- session history ---------------------------------------------------------
async function toggleHistory() {
  const pop = $("#history-pop");
  if (!pop.hidden) { pop.hidden = true; return; }
  const list = $("#history-list");
  list.innerHTML = "";
  try {
    const d = await (await fetch("/api/sessions")).json();
    if (!d.sessions.length) {
      list.innerHTML = `<li class="history-empty">${escapeHtml(t("No sessions yet."))}</li>`;
    }
    d.sessions.forEach((s) => {
      const li = document.createElement("li");
      li.className = "history-item";
      const when = new Date(s.created * 1000).toLocaleTimeString();
      li.innerHTML =
        `<span class="hist-status st-${escapeHtml(s.status)}"></span>` +
        `<span class="hist-main"><span class="hist-task">${escapeHtml(s.task)}</span>` +
        `<span class="hist-sub">${escapeHtml(s.strategy)} · ${when} · ${escapeHtml(s.status)}</span></span>`;
      li.addEventListener("click", () => { pop.hidden = true; openSession(s.id); });
      list.appendChild(li);
    });
  } catch (e) {
    list.innerHTML = `<li class="history-empty">${escapeHtml(String(e))}</li>`;
  }
  pop.hidden = false;
}

// -- theme (auto / light / dark) ---------------------------------------------
const THEME_ICONS = { auto: "◐", light: "☀", dark: "☾" };

function themeMode() { return localStorage.getItem("ao-theme") || "auto"; }

function applyTheme() {
  const mode = themeMode();
  const resolved = mode !== "auto" ? mode
    : (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  document.documentElement.dataset.theme = resolved;
  const btn = $("#theme-toggle");
  btn.textContent = THEME_ICONS[mode] || THEME_ICONS.auto;
  btn.title = "Theme: " + mode;
}

function cycleTheme() {
  const order = ["auto", "light", "dark"];
  const next = order[(order.indexOf(themeMode()) + 1) % order.length];
  localStorage.setItem("ao-theme", next);
  applyTheme();
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
  st.textContent = t("Saving…");
  try {
    await fetch("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await refreshAvailability();
    await loadSettings();
    st.textContent = t("Saved ✓");
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
applyTheme();
window.matchMedia("(prefers-color-scheme: light)")
  .addEventListener("change", () => { if (themeMode() === "auto") applyTheme(); });
applyLang();

$("#run").addEventListener("click", run);
$("#stop").addEventListener("click", stop);
$("#theme-toggle").addEventListener("click", cycleTheme);
$("#lang-toggle").addEventListener("click", toggleLang);
$("#history-open").addEventListener("click", toggleHistory);
$("#export-md").addEventListener("click", exportTranscript);
$("#settings-open").addEventListener("click", openSettings);
$("#settings-close").addEventListener("click", closeSettings);
$("#settings-save").addEventListener("click", saveSettings);
$("#settings-overlay").addEventListener("click", (e) => {
  if (e.target.id === "settings-overlay") closeSettings();
});
document.querySelectorAll(".ftab").forEach((b) =>
  b.addEventListener("click", () => setTab(b.dataset.tab)));
$("#jump-latest").addEventListener("click", () => {
  window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "smooth" });
  $("#jump-latest").hidden = true;
});
window.addEventListener("scroll", () => { if (nearBottom()) $("#jump-latest").hidden = true; });
$("#artifact-view-preview").addEventListener("click", () => { state.artifact.view = "preview"; setSeg("preview"); renderArtifact(); });
$("#artifact-view-diff").addEventListener("click", () => { state.artifact.view = "diff"; setSeg("diff"); renderArtifact(); });
$("#artifact-ext").addEventListener("change", renderArtifact);
$("#artifact-copy").addEventListener("click", (e) => copyText(state.artifact.content || "", e.currentTarget));
$("#artifact-download").addEventListener("click", downloadArtifact);

// Ctrl+Enter / Cmd+Enter in the task box runs the collaboration.
$("#task").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !$("#run").disabled) run();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (!$("#settings-overlay").hidden) closeSettings();
    else if (!$("#history-pop").hidden) $("#history-pop").hidden = true;
  }
});
document.addEventListener("click", (e) => {
  const pop = $("#history-pop");
  if (!pop.hidden && !pop.contains(e.target) && e.target.id !== "history-open") pop.hidden = true;
});

// Form values survive a reload.
["task", "rounds", "workspace-dir", "reference-dir"].forEach((id) =>
  $("#" + id).addEventListener("input", persistForm));
$("#workspace-init").addEventListener("change", persistForm);

loadCatalog().catch((e) => setStatus("Failed to load catalog: " + e, true));
