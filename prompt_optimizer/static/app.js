// ── Prompt Optimization Agent - Frontend ────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── State ───────────────────────────────────────────────────────

let techniques = [];
let models = { openai: [], anthropic: [] };
let selectedTechniques = new Set();

// ── Init ────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async () => {
    await Promise.all([loadTechniques(), loadModels()]);
    setupEventListeners();
});

// ── Data Loading ────────────────────────────────────────────────

async function loadTechniques() {
    try {
        const res = await fetch("/api/techniques");
        const data = await res.json();
        techniques = data.techniques;
        renderTechniques();
    } catch (err) {
        console.error("Failed to load techniques:", err);
    }
}

async function loadModels() {
    try {
        const res = await fetch("/api/models");
        const data = await res.json();
        models = data.models;
        updateModelSelect();
    } catch (err) {
        console.error("Failed to load models:", err);
    }
}

// ── Rendering ───────────────────────────────────────────────────

function renderTechniques() {
    const list = $("#technique-list");
    list.innerHTML = "";

    techniques.forEach((t) => {
        const item = document.createElement("div");
        item.className = "technique-item";
        item.dataset.id = t.id;

        const catClass = `cat-${t.category}`;

        item.innerHTML = `
            <input type="checkbox" id="tech-${t.id}" value="${t.id}">
            <div class="technique-info">
                <div class="technique-name">${t.name_ja}</div>
                <div class="technique-desc">${t.description_ja}</div>
            </div>
            <span class="technique-category ${catClass}">${t.category}</span>
        `;

        // Click entire item to toggle
        item.addEventListener("click", (e) => {
            if (e.target.tagName === "INPUT") return; // Let checkbox handle itself
            const cb = item.querySelector("input[type=checkbox]");
            cb.checked = !cb.checked;
            toggleTechnique(t.id, cb.checked);
        });

        const cb = item.querySelector("input[type=checkbox]");
        cb.addEventListener("change", () => {
            toggleTechnique(t.id, cb.checked);
        });

        list.appendChild(item);
    });

    updateTechniqueCount();
}

function toggleTechnique(id, checked) {
    if (checked) {
        selectedTechniques.add(id);
    } else {
        selectedTechniques.delete(id);
    }

    // Update visual
    const item = $(`.technique-item[data-id="${id}"]`);
    if (item) {
        item.classList.toggle("selected", checked);
    }

    updateTechniqueCount();
}

function updateTechniqueCount() {
    $("#technique-count").textContent = `${selectedTechniques.size} selected`;
}

function updateModelSelect() {
    const provider = $("#provider").value;
    const modelSelect = $("#model");
    const providerModels = models[provider] || [];

    modelSelect.innerHTML = "";
    providerModels.forEach((m) => {
        const opt = document.createElement("option");
        opt.value = m.id;
        opt.textContent = m.name;
        modelSelect.appendChild(opt);
    });
}

// ── Event Listeners ─────────────────────────────────────────────

function setupEventListeners() {
    // Provider change → update model list
    $("#provider").addEventListener("change", updateModelSelect);

    // Temperature slider
    $("#temperature").addEventListener("input", (e) => {
        $("#temp-value").textContent = e.target.value;
    });

    // API key toggle
    $("#toggle-key").addEventListener("click", () => {
        const input = $("#api-key");
        input.type = input.type === "password" ? "text" : "password";
    });

    // Select all / clear
    $("#select-all").addEventListener("click", () => {
        techniques.forEach((t) => {
            selectedTechniques.add(t.id);
            const cb = $(`#tech-${t.id}`);
            if (cb) cb.checked = true;
            const item = $(`.technique-item[data-id="${t.id}"]`);
            if (item) item.classList.add("selected");
        });
        updateTechniqueCount();
    });

    $("#select-none").addEventListener("click", () => {
        selectedTechniques.clear();
        $$(".technique-item input[type=checkbox]").forEach((cb) => (cb.checked = false));
        $$(".technique-item").forEach((item) => item.classList.remove("selected"));
        updateTechniqueCount();
    });

    // Optimize button
    $("#optimize-btn").addEventListener("click", optimizePrompt);

    // Copy button
    $("#copy-btn").addEventListener("click", copyOptimized);
}

// ── Optimize ────────────────────────────────────────────────────

async function optimizePrompt() {
    const originalPrompt = $("#original-prompt").value.trim();
    const apiKey = $("#api-key").value.trim();
    const provider = $("#provider").value;
    const model = $("#model").value;
    const language = $("#language").value;
    const temperature = parseFloat($("#temperature").value);
    const includeAnalysis = $("#include-analysis").checked;

    // Validation
    if (!originalPrompt) {
        showError("Please enter a prompt to optimize.");
        return;
    }
    if (!apiKey) {
        showError("Please enter your API key.");
        return;
    }
    if (selectedTechniques.size === 0) {
        showError("Please select at least one PE technique.");
        return;
    }

    // UI state
    const btn = $("#optimize-btn");
    btn.disabled = true;
    btn.querySelector(".btn-text").style.display = "none";
    btn.querySelector(".btn-loading").style.display = "inline";
    hideError();
    $("#output-section").style.display = "none";

    try {
        const res = await fetch("/api/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                original_prompt: originalPrompt,
                technique_ids: Array.from(selectedTechniques),
                provider,
                model,
                api_key: apiKey,
                language,
                temperature,
                include_analysis: includeAnalysis,
            }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        showResult(data);
    } catch (err) {
        showError(err.message);
    } finally {
        btn.disabled = false;
        btn.querySelector(".btn-text").style.display = "inline";
        btn.querySelector(".btn-loading").style.display = "none";
    }
}

// ── Display Results ─────────────────────────────────────────────

function showResult(data) {
    $("#output-section").style.display = "block";
    $("#optimized-output").textContent = data.optimized_prompt;

    // Analysis (render simple markdown)
    if (data.analysis) {
        $("#analysis-section").style.display = "block";
        $("#analysis-output").innerHTML = renderMarkdown(data.analysis);
    } else {
        $("#analysis-section").style.display = "none";
    }

    // Meta info
    const usage = data.usage || {};
    $("#meta-info").innerHTML = `
        <span>Model: <strong>${data.model_used}</strong></span>
        <span>Techniques: <strong>${data.techniques_applied.length}</strong></span>
        <span>Tokens: <strong>${(usage.total_tokens || 0).toLocaleString()}</strong></span>
        <span>Prompt tokens: ${(usage.prompt_tokens || 0).toLocaleString()}</span>
        <span>Completion tokens: ${(usage.completion_tokens || 0).toLocaleString()}</span>
    `;
}

function renderMarkdown(text) {
    // Simple markdown renderer
    let html = text
        // Escape HTML
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        // Headers
        .replace(/^### (.+)$/gm, "<h3>$1</h3>")
        .replace(/^## (.+)$/gm, "<h2>$1</h2>")
        .replace(/^# (.+)$/gm, "<h1>$1</h1>")
        // Bold
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        // Inline code
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        // Bullet lists
        .replace(/^- (.+)$/gm, "<li>$1</li>")
        // Numbered lists
        .replace(/^\d+\. (.+)$/gm, "<li>$1</li>")
        // Wrap consecutive <li> in <ul>
        .replace(/((?:<li>.*<\/li>\n?)+)/g, "<ul>$1</ul>")
        // Paragraphs (double newline)
        .replace(/\n\n/g, "</p><p>")
        // Single newlines
        .replace(/\n/g, "<br>");

    return `<p>${html}</p>`;
}

// ── Copy ────────────────────────────────────────────────────────

async function copyOptimized() {
    const text = $("#optimized-output").textContent;
    try {
        await navigator.clipboard.writeText(text);
        const btn = $("#copy-btn");
        btn.textContent = "Copied!";
        btn.classList.add("copied");
        setTimeout(() => {
            btn.textContent = "Copy";
            btn.classList.remove("copied");
        }, 2000);
    } catch {
        // Fallback
        const textarea = document.createElement("textarea");
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
    }
}

// ── Error Handling ──────────────────────────────────────────────

function showError(msg) {
    const box = $("#error-box");
    box.textContent = msg;
    box.style.display = "block";
}

function hideError() {
    $("#error-box").style.display = "none";
}
