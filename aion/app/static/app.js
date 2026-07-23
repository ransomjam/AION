"use strict";

// ── DOM helpers ───────────────────────────────────────────────────────────────
const el = (tag, attrs = {}, ...kids) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return node;
};

async function apiPost(path, body = {}) {
  const res = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}
const apiGet = async (p) => (await fetch(p)).json();

const main = () => document.getElementById("main");
function toast(msg, kind = "error") {
  main().prepend(el("div", { class: "card" }, el("div", { class: kind === "error" ? "error" : "pill" }, msg)));
}

// ── App state ─────────────────────────────────────────────────────────────────
const state = { registry: null, labId: null, project: null };

// ── Charts (browser-native, no dependencies) ──────────────────────────────────
function barChart(items, { label, value, max }) {
  const top = max || Math.max(1, ...items.map((i) => i[value]));
  return el("div", {}, ...items.map((i) =>
    el("div", { class: "barrow" },
      el("div", { class: "lbl" }, String(i[label])),
      el("div", { class: "track" }, el("div", { class: "fill", style: `width:${(i[value] / top) * 100}%` })),
      el("div", { class: "val" }, String(i[value])))));
}

function sparkline(points) {
  // points: [{documents, unique_tokens}] -> polyline over normalized coords
  if (points.length < 2) return el("div", { class: "pill" }, "Add more documents to see growth.");
  const W = 600, H = 120, pad = 6;
  const xs = points.map((p) => p.documents), ys = points.map((p) => p.unique_tokens);
  const xmin = Math.min(...xs), xmax = Math.max(...xs), ymax = Math.max(...ys) || 1;
  const sx = (x) => pad + ((x - xmin) / Math.max(1, xmax - xmin)) * (W - 2 * pad);
  const sy = (y) => H - pad - (y / ymax) * (H - 2 * pad);
  const d = points.map((p) => `${sx(p.documents).toFixed(1)},${sy(p.unique_tokens).toFixed(1)}`).join(" ");
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("class", "spark"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  const poly = document.createElementNS(ns, "polyline");
  poly.setAttribute("points", d); poly.setAttribute("fill", "none");
  poly.setAttribute("stroke", "#6ea8fe"); poly.setAttribute("stroke-width", "2");
  svg.append(poly);
  return svg;
}

const stat = (n, label) => el("div", { class: "stat" },
  el("div", { class: "n" }, String(n)), el("div", { class: "l" }, label));

// ── Panels registry (id -> render fn) ─────────────────────────────────────────
const PANELS = {};

// Project Explorer -------------------------------------------------------------
PANELS.projects = async function () {
  const m = main(); m.replaceChildren();
  m.append(el("div", { class: "panel-head" },
    el("h1", {}, "Project Explorer"),
    el("p", {}, "Every dataset, tokenizer, and model belongs to a project. Choose one, or create a new one.")));

  const name = el("input", { type: "text", placeholder: "e.g. TinyGPT" });
  const desc = el("input", { type: "text", placeholder: "one-line description" });
  const lang = el("input", { type: "text", placeholder: "language (optional)" });
  const create = async () => {
    try {
      await apiPost("/api/projects/create", { name: name.value, description: desc.value, language: lang.value });
      PANELS.projects();
    } catch (e) { toast(e.message); }
  };
  m.append(el("div", { class: "card" }, el("h2", {}, "New project"),
    el("div", { class: "row" },
      el("div", {}, el("label", {}, "Name"), name),
      el("div", {}, el("label", {}, "Description"), desc),
      el("div", {}, el("label", {}, "Language"), lang)),
    el("div", { class: "row" }, el("div", {}, el("button", { class: "run", onclick: create }, "Create project")))));

  const listCard = el("div", { class: "card" }, el("h2", {}, "Projects"), el("div", {}, "Loading…"));
  m.append(listCard);
  try {
    const { projects } = await apiPost("/api/projects/list");
    listCard.replaceChildren(el("h2", {}, `Projects (${projects.length})`),
      projects.length
        ? el("div", { class: "grid-cards" }, ...projects.map(projectCard))
        : el("div", { class: "notice" }, "No projects yet. Create your first one above."));
  } catch (e) { toast(e.message); }
};

function projectCard(p) {
  return el("div", { class: "project-card", onclick: () => openProject(p.id) },
    el("div", { class: "row-line", style: "display:flex;justify-content:space-between;align-items:center" },
      el("h3", {}, p.name),
      el("span", { class: "badge-status " + (p.status === "archived" ? "badge-archived" : "badge-active") }, p.status)),
    el("div", { class: "meta" }, p.description || "no description"),
    el("div", { class: "meta" }, (p.language ? p.language + " · " : "") + "updated " + (p.updated_at || "").slice(0, 10)),
    p.tags && p.tags.length ? el("div", { class: "tags" }, ...p.tags.map((t) => el("span", { class: "tag" }, t))) : null);
}

async function openProject(id) {
  try {
    const { project } = await apiPost("/api/projects/get", { id });
    state.project = project;
    select("data");            // enter the first project lab
  } catch (e) { toast(e.message); }
}

// Scratch labs -----------------------------------------------------------------
PANELS.tokenizer = function () {
  const m = main(); m.replaceChildren();
  m.append(el("div", { class: "panel-head" }, el("h1", {}, "Tokenizer Lab"),
    el("p", {}, "Scratch tool — normalize and tokenize text. Nothing here is saved.")));
  const input = el("textarea", { rows: "3" }, "Confirm your PIN or your MoMo account will be BLOCKED!! Café ﬁle 123");
  const cCase = el("input", { type: "checkbox", checked: "" });
  const cAcc = el("input", { type: "checkbox", checked: "" });
  const out = el("div", {});
  const run = async () => {
    try {
      out.replaceChildren(); renderTokenize(out, await apiPost("/api/tokenize",
        { text: input.value, casefold: cCase.checked, accents: cAcc.checked }));
    } catch (e) { toast(e.message); }
  };
  m.append(el("div", { class: "card" }, el("h2", {}, "Input"), el("label", {}, "Text"), input,
    el("div", { class: "row" }, el("div", { class: "checks" },
      el("label", { class: "check" }, cCase, "case-fold"),
      el("label", { class: "check" }, cAcc, "strip accents")),
      el("div", {}, el("button", { class: "run", onclick: run }, "Tokenize")))));
  m.append(out); run();
};
function renderTokenize(out, r) {
  const chip = (t) => el("span", { class: "chip" + (/^[\wÀ-﾿]+$/u.test(t) ? "" : " punct") }, t);
  out.replaceChildren(
    el("div", { class: "card" }, el("h2", {}, "Pipeline"),
      el("div", { class: "stages" }, ...r.stages.map((s) =>
        el("div", { class: "stage" }, el("div", { class: "k" }, s.name), el("div", { class: "v" }, s.value || "∅"))))),
    el("div", { class: "card" }, el("h2", {}, `Tokens (${r.tokens.length})`),
      el("div", { class: "chips" }, ...r.tokens.map(chip))),
    el("div", { class: "card" }, el("h2", {}, "Statistics"), el("div", { class: "stats" },
      stat(r.stats.char_count, "characters"), stat(r.stats.token_count, "tokens"),
      stat(r.stats.unique_tokens, "unique"), stat(r.stats.type_token_ratio, "type/token"))));
}

PANELS.vocabulary = function () {
  const m = main(); m.replaceChildren();
  m.append(el("div", { class: "panel-head" }, el("h1", {}, "Vocabulary Lab"),
    el("p", {}, "Scratch tool — build a vocabulary from a corpus. One document per line.")));
  const corpus = el("textarea", { rows: "6" }, "the cat sat on the mat\nthe dog sat on the log\nthe the the pin pin otp");
  const minFreq = el("input", { type: "number", value: "1", min: "1" });
  const probe = el("input", { type: "text", value: "the cat flew" });
  const out = el("div", {});
  const run = async () => {
    try {
      out.replaceChildren(); renderVocab(out, await apiPost("/api/vocabulary",
        { corpus: corpus.value.split("\n"), min_freq: Number(minFreq.value) || 1, probe: probe.value }));
    } catch (e) { toast(e.message); }
  };
  m.append(el("div", { class: "card" }, el("h2", {}, "Corpus"), el("label", {}, "Documents (one per line)"), corpus,
    el("div", { class: "row" }, el("div", {}, el("label", {}, "min frequency"), minFreq),
      el("div", {}, el("label", {}, "probe text"), probe),
      el("div", {}, el("button", { class: "run", onclick: run }, "Build")))));
  m.append(out); run();
};
function renderVocab(out, r) {
  const rows = r.entries.map((e) => el("tr", e.special ? { class: "special" } : {},
    el("td", {}, String(e.id)), el("td", {}, e.token), el("td", {}, e.special ? "—" : String(e.frequency))));
  const cards = [el("div", { class: "card" }, el("h2", {}, "Summary"), el("div", { class: "stats" },
    stat(r.stats.vocab_size, "vocab size"), stat(r.stats.corpus_tokens, "corpus tokens"),
    stat(r.stats.documents, "documents"), stat(r.stats.specials.length, "specials"))),
    el("div", { class: "card" }, el("h2", {}, "Token ↔ id"), el("div", { class: "tablewrap" }, el("table", {},
      el("thead", {}, el("tr", {}, el("th", {}, "id"), el("th", {}, "token"), el("th", {}, "freq"))),
      el("tbody", {}, ...rows))))];
  if (r.probe) {
    const p = r.probe;
    cards.push(el("div", { class: "card" }, el("h2", {}, "Probe"),
      el("div", { class: "chips" }, ...p.tokens.map((t, i) =>
        el("span", { class: "chip" + (p.unknown.includes(t) ? " unk" : "") }, `${t}→${p.ids[i]}`))),
      el("p", { class: "hint" }, `ids: [${p.ids.join(", ")}]`)));
  }
  out.replaceChildren(...cards);
}

// Data Lab (project-scoped) ----------------------------------------------------
const dataState = { datasetId: null, tab: "documents" };

PANELS.data = async function () {
  const m = main(); m.replaceChildren();
  if (!state.project) { requireProject(m, "Data Lab"); return; }
  m.append(el("div", { class: "panel-head" }, el("h1", {}, "Data Lab"),
    el("p", {}, `Datasets in project “${state.project.name}”. The root of every training pipeline.`)));
  const cols = el("div", { class: "lab-cols" });
  m.append(cols);
  await renderDataLab(cols);
};

async function renderDataLab(cols) {
  const left = el("div", {});
  const right = el("div", {});
  cols.replaceChildren(left, right);

  // dataset list + create
  const dsName = el("input", { type: "text", placeholder: "dataset name" });
  const createDs = async () => {
    try { await apiPost("/api/datasets/create", { project_id: state.project.id, name: dsName.value });
      dataState.datasetId = null; renderDataLab(cols); } catch (e) { toast(e.message); }
  };
  let datasets = [];
  try { datasets = (await apiPost("/api/datasets/list", { project_id: state.project.id })).datasets; }
  catch (e) { toast(e.message); }

  left.replaceChildren(el("div", { class: "card" }, el("h2", {}, "Datasets"),
    el("div", { class: "list-panel" }, ...(datasets.length ? datasets.map((d) =>
      el("button", { class: "list-item" + (d.id === dataState.datasetId ? " active" : ""),
        onclick: () => { dataState.datasetId = d.id; dataState.tab = "documents"; renderDataLab(cols); } },
        el("div", {}, d.name), el("div", { class: "sub" }, `${d.document_count} docs`))) : [el("div", { class: "pill" }, "none yet")])),
    el("label", { style: "margin-top:12px" }, "New dataset"), dsName,
    el("div", { style: "margin-top:8px" }, el("button", { class: "run", onclick: createDs }, "Create"))));

  if (!dataState.datasetId) {
    right.replaceChildren(el("div", { class: "notice" }, "Select or create a dataset to import and analyze documents."));
    return;
  }
  await renderDataset(right, cols);
}

async function renderDataset(right, cols) {
  const pid = state.project.id, did = dataState.datasetId;
  let ds;
  try { ds = await apiPost("/api/datasets/get", { project_id: pid, dataset_id: did }); }
  catch (e) { toast(e.message); return; }

  const tabs = ["documents", "search", "analyze"];
  const tabBar = el("div", { class: "tabs" }, ...tabs.map((t) =>
    el("button", { class: "tab" + (t === dataState.tab ? " active" : ""),
      onclick: () => { dataState.tab = t; renderDataset(right, cols); } }, t[0].toUpperCase() + t.slice(1))));

  const head = el("div", { class: "panel-head" },
    el("h1", {}, ds.dataset.name),
    el("p", {}, `${ds.dataset.document_count} documents · id ${ds.dataset.id}`));
  const body = el("div", {});
  right.replaceChildren(head, tabBar, body);

  if (dataState.tab === "documents") await renderDocuments(body, ds, cols);
  else if (dataState.tab === "search") renderSearch(body);
  else await renderAnalyze(body);
}

async function renderDocuments(body, ds, cols) {
  const pid = state.project.id, did = dataState.datasetId;
  const text = el("textarea", { rows: "4", placeholder: "paste text to import…" });
  const split = el("select", {},
    el("option", { value: "one" }, "one document"),
    el("option", { value: "lines" }, "one per line"),
    el("option", { value: "paragraphs" }, "one per paragraph"));
  const doImport = async () => {
    try { const r = await apiPost("/api/datasets/import_text",
      { project_id: pid, dataset_id: did, text: text.value, split: split.value });
      toast(`Imported ${r.job.result.imported} document(s).`, "info"); dataState.tab = "documents";
      await renderDataLab(cols); } catch (e) { toast(e.message); }
  };
  const importCard = el("div", { class: "card" }, el("h2", {}, "Import text"), text,
    el("div", { class: "row" }, el("div", {}, el("label", {}, "split as"), split),
      el("div", {}, el("button", { class: "run", onclick: doImport }, "Import"))));

  const rows = [];
  for (const id of ds.document_ids.slice(0, 200)) {
    const view = async () => {
      try { const d = await apiPost("/api/documents/get", { project_id: pid, dataset_id: did, doc_id: id });
        openEditor(id, d.text, cols); } catch (e) { toast(e.message); }
    };
    const del = async () => {
      try { await apiPost("/api/documents/delete", { project_id: pid, dataset_id: did, doc_id: id });
        await renderDataLab(cols); } catch (e) { toast(e.message); }
    };
    rows.push(el("div", { class: "doc-row" },
      el("div", { class: "id" }, String(id).padStart(6, "0")),
      el("div", { class: "prev pill" }, "click view to open"),
      el("div", { class: "inline-actions" },
        el("button", { onclick: view }, "view/edit"), el("button", { onclick: del }, "delete"))));
  }
  const listCard = el("div", { class: "card" }, el("h2", {}, `Documents (${ds.document_ids.length})`),
    rows.length ? el("div", {}, ...rows) : el("div", { class: "pill" }, "empty — import some text above"));
  body.replaceChildren(importCard, el("div", { id: "editor-slot" }), listCard);
}

function openEditor(id, textValue, cols) {
  const pid = state.project.id, did = dataState.datasetId;
  const ta = el("textarea", { rows: "6" }, textValue);
  const save = async () => {
    try { await apiPost("/api/documents/edit", { project_id: pid, dataset_id: did, doc_id: id, text: ta.value });
      toast("Saved.", "info"); await renderDataLab(cols); } catch (e) { toast(e.message); }
  };
  const slot = document.getElementById("editor-slot");
  if (slot) slot.replaceChildren(el("div", { class: "card" },
    el("h2", {}, `Document ${String(id).padStart(6, "0")}`), ta,
    el("div", { style: "margin-top:8px" }, el("button", { class: "run", onclick: save }, "Save"))));
}

function renderSearch(body) {
  const pid = state.project.id, did = dataState.datasetId;
  const q = el("input", { type: "text", placeholder: "search text or regex…" });
  const cs = el("input", { type: "checkbox" });
  const rx = el("input", { type: "checkbox" });
  const out = el("div", {});
  const run = async () => {
    try {
      const r = await apiPost("/api/datasets/search",
        { project_id: pid, dataset_id: did, query: q.value, case_sensitive: cs.checked, regex: rx.checked });
      out.replaceChildren(el("div", { class: "card" }, el("h2", {}, `${r.match_count} match(es)${r.truncated ? " (truncated)" : ""}`),
        ...(r.matches.length ? r.matches.map((mm) => el("div", { class: "doc-row" },
          el("div", { class: "id" }, String(mm.doc_id).padStart(6, "0")),
          el("div", { class: "prev" }, mm.snippet), el("div", {}, ""))) : [el("div", { class: "pill" }, "no matches")])));
    } catch (e) { toast(e.message); }
  };
  body.replaceChildren(el("div", { class: "card" }, el("h2", {}, "Search documents"),
    el("div", { class: "row" }, el("div", {}, el("label", {}, "query"), q),
      el("div", { class: "checks" }, el("label", { class: "check" }, cs, "case-sensitive"),
        el("label", { class: "check" }, rx, "regex")),
      el("div", {}, el("button", { class: "run", onclick: run }, "Search")))), out);
}

async function renderAnalyze(body) {
  const pid = state.project.id, did = dataState.datasetId;
  const out = el("div", {});
  const run = async () => {
    out.replaceChildren(el("div", { class: "pill" }, "Analyzing…"));
    try { renderAnalysis(out, await apiPost("/api/datasets/analyze", { project_id: pid, dataset_id: did })); }
    catch (e) { toast(e.message); out.replaceChildren(); }
  };
  body.replaceChildren(el("div", { class: "card" }, el("h2", {}, "Analysis"),
    el("p", { class: "hint" }, "Statistics and quality are computed via a job and cached by dataset fingerprint."),
    el("button", { class: "run", onclick: run }, "Run analysis")), out);
  run();
}

function renderAnalysis(out, r) {
  const s = r.statistics, q = r.quality;
  if (!s) { out.replaceChildren(el("div", { class: "pill" }, "No statistics (empty dataset?).")); return; }
  const dl = s.document_length;
  const cards = [
    el("div", { class: "card" }, el("h2", {}, "Overview"), el("div", { class: "stats" },
      stat(s.documents, "documents"), stat(s.tokens, "tokens"), stat(s.unique_tokens, "unique tokens"),
      stat(s.type_token_ratio, "type/token"), stat(s.characters, "characters"),
      stat(dl.avg_tokens ?? 0, "avg tokens/doc"), stat(dl.max_tokens ?? 0, "longest (tok)"),
      stat(dl.min_tokens ?? 0, "shortest (tok)"))),
    el("div", { class: "card" }, el("h2", {}, "Vocabulary growth"), sparkline(s.vocabulary_growth)),
    el("div", { class: "card" }, el("h2", {}, "Top tokens"),
      barChart(s.top_tokens.slice(0, 15).map(([t, n]) => ({ t, n })), { label: "t", value: "n" })),
    el("div", { class: "card" }, el("h2", {}, "Document length distribution"),
      barChart(s.length_distribution, { label: "range", value: "count" })),
    el("div", { class: "card" }, el("h2", {}, "Top characters"),
      barChart(s.top_characters.slice(0, 12), { label: "char", value: "count" })),
  ];
  if (q) {
    const i = q.issues;
    cards.push(el("div", { class: "card" }, el("h2", {}, "Quality"), el("div", { class: "stats" },
      stat(i.empty.length, "empty"), stat(i.whitespace_only.length, "whitespace"),
      stat(i.duplicate_documents, "duplicates"), stat(i.very_short.length, "very short"),
      stat(i.very_long.length, "very long"),
      stat(q.normalization.needs_normalization, "need normalize")),
      el("div", { style: "margin-top:12px" }, el("h2", {}, "Character composition"),
        barChart(Object.entries(q.character_categories).map(([k, v]) => ({ k, v })), { label: "k", value: "v" })),
      el("p", { class: "hint" }, "Language hint: " + JSON.stringify(q.language_hint))));
  }
  out.replaceChildren(...cards);
}

// BPE Lab (project-scoped) ---------------------------------------------------
const bpeState = { tokenizerId: null };

PANELS.bpe = async function () {
  const m = main(); m.replaceChildren();
  if (!state.project) { requireProject(m, "BPE Lab"); return; }
  m.append(el("div", { class: "panel-head" },
    el("h1", {}, "BPE Lab"),
    el("p", {}, `Byte-Level BPE tokenizers for project \u201c${state.project.name}\u201d.`)));
  const cols = el("div", { class: "lab-cols" });
  m.append(cols);
  await renderBpeLab(cols);
};

async function renderBpeLab(cols) {
  const left = el("div", {});
  const right = el("div", {});
  cols.replaceChildren(left, right);

  // ── left: tokenizer list + train form ──────────────────────────────────────
  let datasets = [], tokenizers = [];
  try { datasets = (await apiPost("/api/datasets/list", { project_id: state.project.id })).datasets; } catch (e) { /**/ }
  try { tokenizers = (await apiPost("/api/tokenizers/list", { project_id: state.project.id })).tokenizers; } catch (e) { /**/ }

  const dsSelect = el("select", {}, ...(
    datasets.length
      ? datasets.map((d) => el("option", { value: d.id }, d.name))
      : [el("option", { value: "" }, "— no datasets —")]
  ));
  const tokName = el("input", { type: "text", placeholder: "e.g. bpe-1k" });
  const vocabSize = el("input", { type: "number", value: "1000", min: "260" });
  const trainStatus = el("div", { class: "pill" });

  const doTrain = async () => {
    if (!dsSelect.value) { toast("Select a dataset first."); return; }
    trainStatus.textContent = "Training\u2026 (this may take a moment)";
    try {
      const r = await apiPost("/api/tokenizers/train", {
        project_id: state.project.id,
        dataset_id: dsSelect.value,
        name: tokName.value || "bpe",
        vocab_size: Number(vocabSize.value) || 1000,
      });
      const job = r.job;
      if (job.status === "failed") { toast("Training failed: " + job.error); trainStatus.textContent = ""; return; }
      bpeState.tokenizerId = job.result?.tokenizer_id || null;
      trainStatus.textContent = "";
      await renderBpeLab(cols);
    } catch (e) { toast(e.message); trainStatus.textContent = ""; }
  };

  const listItems = tokenizers.length
    ? tokenizers.map((t) => {
        const isDefault = t.status === "default";
        return el("button", {
          class: "list-item" + (t.id === bpeState.tokenizerId ? " active" : ""),
          onclick: () => { bpeState.tokenizerId = t.id; renderBpeLab(cols); },
        },
          el("div", { style: "display:flex;justify-content:space-between;align-items:center" },
            el("div", {}, t.name),
            isDefault ? el("span", { class: "badge-active badge-status" }, "default") : null),
          el("div", { class: "sub" }, `${t.vocab_size} tokens · ${t.algorithm}`));
      })
    : [el("div", { class: "pill" }, "none yet")];

  left.replaceChildren(
    el("div", { class: "card" }, el("h2", {}, "Tokenizers"),
      el("div", { class: "list-panel" }, ...listItems)),
    el("div", { class: "card" }, el("h2", {}, "Train new tokenizer"),
      el("label", {}, "Dataset"), dsSelect,
      el("label", { style: "margin-top:10px" }, "Name"), tokName,
      el("label", { style: "margin-top:10px" }, "Vocabulary size"), vocabSize,
      el("div", { class: "row", style: "margin-top:14px" },
        el("div", {}, el("button", { class: "run", onclick: doTrain }, "Train")),
        el("div", {}, trainStatus))));

  // ── right: tokenizer detail ────────────────────────────────────────────────
  if (!bpeState.tokenizerId) {
    right.replaceChildren(el("div", { class: "notice" }, "Select or train a tokenizer to inspect it."));
    return;
  }
  await renderTokenizerDetail(right, cols);
}

async function renderTokenizerDetail(right, cols) {
  const pid = state.project.id, tid = bpeState.tokenizerId;
  let data;
  try { data = await apiPost("/api/tokenizers/get", { project_id: pid, tokenizer_id: tid }); }
  catch (e) { toast(e.message); return; }

  const { manifest: mf, statistics: stats } = data;
  const isDefault = mf.status === "default";

  const setDefault = async () => {
    try { await apiPost("/api/tokenizers/set_default", { project_id: pid, tokenizer_id: tid });
      await renderBpeLab(cols); } catch (e) { toast(e.message); }
  };
  const doDelete = async () => {
    if (!confirm(`Delete tokenizer \u201c${mf.name}\u201d?`)) return;
    try { await apiPost("/api/tokenizers/delete", { project_id: pid, tokenizer_id: tid });
      bpeState.tokenizerId = null; await renderBpeLab(cols); } catch (e) { toast(e.message); }
  };

  const tabs = ["overview", "merges", "charts", "encode"];
  const tabState = { t: right._bpeTab || "overview" };
  right._bpeTab = tabState.t;

  const renderTabs = () => {
    const tabBar = el("div", { class: "tabs" }, ...tabs.map((t) =>
      el("button", { class: "tab" + (t === tabState.t ? " active" : ""),
        onclick: () => { tabState.t = t; right._bpeTab = t; renderBody(); } },
        t[0].toUpperCase() + t.slice(1))));
    return tabBar;
  };

  const body = el("div", {});
  const renderBody = () => {
    if (tabState.t === "overview") renderBpeOverview(body, mf, stats);
    else if (tabState.t === "merges") renderBpeMerges(body, stats);
    else if (tabState.t === "charts") renderBpeCharts(body, stats);
    else renderBpeEncode(body, pid, tid);
  };

  const actions = el("div", { class: "row", style: "margin-bottom:14px" },
    el("div", {}, isDefault
      ? el("span", { class: "badge-active badge-status" }, "\u2713 project default")
      : el("button", { class: "ghost", onclick: setDefault }, "Set as default")),
    el("div", {}, el("button", { class: "ghost", onclick: doDelete }, "Delete")));

  right.replaceChildren(
    el("div", { class: "panel-head" },
      el("h1", {}, mf.name),
      el("p", {}, `${mf.algorithm} · ${mf.vocab_size} tokens · ${mf.merge_count} merges`)),
    actions,
    renderTabs(),
    body);
  renderBody();
}

function renderBpeOverview(body, mf, stats) {
  const m = stats || {};
  body.replaceChildren(
    el("div", { class: "card" }, el("h2", {}, "Training metrics"), el("div", { class: "stats" },
      stat(mf.vocab_size, "vocab size"),
      stat(mf.merge_count, "merges"),
      stat(m.base_vocab_size ?? 260, "base vocab"),
      stat(m.compression_ratio ?? "\u2014", "compression ratio"),
      stat(m.avg_tokens_per_word ?? "\u2014", "avg tokens/word"),
      stat(m.training_time_s != null ? m.training_time_s + "s" : "\u2014", "training time"))),
    el("div", { class: "card" }, el("h2", {}, "Manifest"),
      el("div", { class: "tablewrap" }, el("table", {},
        el("tbody", {},
          ...Object.entries({
            id: mf.id, algorithm: mf.algorithm,
            dataset: mf.dataset_id, status: mf.status,
            created: (mf.created_at || "").slice(0, 19).replace("T", " "),
            "vocab fingerprint": (mf.vocabulary_fingerprint || "").slice(0, 16),
            "dataset fingerprint": (mf.dataset_fingerprint || "").slice(0, 16),
          }).map(([k, v]) => el("tr", {}, el("td", { style: "color:var(--text-dim);width:160px" }, k), el("td", {}, String(v)))))))));
}

function renderBpeMerges(body, stats) {
  if (!stats || !stats.merge_history || !stats.merge_history.length) {
    body.replaceChildren(el("div", { class: "pill" }, "No merge history available."));
    return;
  }
  const history = stats.merge_history;
  const stepIdx = { v: Math.min(9, history.length - 1) };

  const slider = el("input", { type: "range", min: "0", max: String(history.length - 1), value: String(stepIdx.v), style: "width:100%" });
  const stepInfo = el("div", { class: "card" });

  const renderStep = () => {
    const s = history[stepIdx.v];
    stepInfo.replaceChildren(
      el("h2", {}, `Merge #${s.rank + 1} of ${history.length}`),
      el("div", { class: "stats" },
        stat(`\u201c${s.pair[0]}\u201d + \u201c${s.pair[1]}\u201d`, "pair merged"),
        stat(`\u201c${s.new_token}\u201d`, "new token"),
        stat(s.pair_freq, "pair frequency"),
        stat(s.vocab_size, "vocab size after"),
        stat(s.corpus_tokens, "corpus tokens after")));
  };

  slider.addEventListener("input", () => { stepIdx.v = Number(slider.value); renderStep(); });

  const rows = history.slice(0, 200).map((s) =>
    el("tr", {},
      el("td", {}, String(s.rank + 1)),
      el("td", {}, `\u201c${s.pair[0]}\u201d + \u201c${s.pair[1]}\u201d`),
      el("td", {}, `\u201c${s.new_token}\u201d`),
      el("td", {}, String(s.pair_freq)),
      el("td", {}, String(s.vocab_size))));

  body.replaceChildren(
    el("div", { class: "card" }, el("h2", {}, "Merge stepper"),
      el("label", {}, `Step: ${stepIdx.v + 1} / ${history.length}`), slider,
      stepInfo),
    el("div", { class: "card" }, el("h2", {}, `Merge history (first ${rows.length})`),
      el("div", { class: "tablewrap" }, el("table", {},
        el("thead", {}, el("tr", {},
          el("th", {}, "#"), el("th", {}, "pair"), el("th", {}, "new token"),
          el("th", {}, "freq"), el("th", {}, "vocab size"))),
        el("tbody", {}, ...rows)))));
  renderStep();
}

function lineChart(points, { xKey, yKey, color = "#4fd1c5", label = "" }) {
  if (points.length < 2) return el("div", { class: "pill" }, "Not enough data.");
  const W = 560, H = 140, padL = 48, padB = 28, padT = 10, padR = 10;
  const xs = points.map((p) => p[xKey]), ys = points.map((p) => p[yKey]);
  const xmin = xs[0], xmax = xs[xs.length - 1];
  const ymin = Math.min(...ys), ymax = Math.max(...ys) || 1;
  const sx = (x) => padL + ((x - xmin) / Math.max(1, xmax - xmin)) * (W - padL - padR);
  const sy = (y) => H - padB - ((y - ymin) / Math.max(1, ymax - ymin)) * (H - padT - padB);
  const pts = points.map((p) => `${sx(p[xKey]).toFixed(1)},${sy(p[yKey]).toFixed(1)}`).join(" ");
  const ns = "http://www.w3.org/2000/svg";
  const mkEl = (tag, attrs) => { const e = document.createElementNS(ns, tag); Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v)); return e; };
  const svg = mkEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "spark", preserveAspectRatio: "none" });
  // axes
  const axis = mkEl("polyline", { points: `${padL},${padT} ${padL},${H - padB} ${W - padR},${H - padB}`, fill: "none", stroke: "#2a333f", "stroke-width": "1" });
  svg.append(axis);
  // y labels
  [ymin, ymax].forEach((v, i) => {
    const t = mkEl("text", { x: padL - 4, y: i === 0 ? H - padB : padT + 8, "text-anchor": "end", fill: "#8b98a8", "font-size": "10" });
    t.textContent = Math.round(v);
    svg.append(t);
  });
  // x labels
  [xmin, xmax].forEach((v, i) => {
    const t = mkEl("text", { x: i === 0 ? padL : W - padR, y: H - 4, "text-anchor": i === 0 ? "start" : "end", fill: "#8b98a8", "font-size": "10" });
    t.textContent = Math.round(v);
    svg.append(t);
  });
  const line = mkEl("polyline", { points: pts, fill: "none", stroke: color, "stroke-width": "2" });
  svg.append(line);
  if (label) { const t = mkEl("text", { x: padL + 4, y: padT + 10, fill: color, "font-size": "10" }); t.textContent = label; svg.append(t); }
  return svg;
}

function renderBpeCharts(body, stats) {
  if (!stats || !stats.merge_history || !stats.merge_history.length) {
    body.replaceChildren(el("div", { class: "pill" }, "No statistics available."));
    return;
  }
  const h = stats.merge_history;
  // Sample every N steps for performance
  const sample = (arr, n) => arr.filter((_, i) => i % Math.max(1, Math.floor(arr.length / n)) === 0);
  const sampled = sample(h, 200);

  const vocabPoints = sampled.map((s) => ({ rank: s.rank, vocab_size: s.vocab_size }));
  const comprPoints = sampled.map((s) => ({ rank: s.rank, corpus_tokens: s.corpus_tokens }));

  // Top 15 merges by pair_freq
  const topMerges = [...h].sort((a, b) => b.pair_freq - a.pair_freq).slice(0, 15)
    .map((s) => ({ pair: `\u201c${s.pair[0]}\u201d+\u201c${s.pair[1]}\u201d`, freq: s.pair_freq }));

  body.replaceChildren(
    el("div", { class: "card" }, el("h2", {}, "Vocabulary growth"),
      lineChart(vocabPoints, { xKey: "rank", yKey: "vocab_size", color: "#4fd1c5", label: "vocab size" })),
    el("div", { class: "card" }, el("h2", {}, "Corpus token count (compression)"),
      lineChart(comprPoints, { xKey: "rank", yKey: "corpus_tokens", color: "#6ea8fe", label: "corpus tokens" })),
    el("div", { class: "card" }, el("h2", {}, "Top 15 merges by frequency"),
      barChart(topMerges, { label: "pair", value: "freq" })));
}

function renderBpeEncode(body, pid, tid) {
  const input = el("textarea", { rows: "3", placeholder: "Type text to encode\u2026" }, "the quick brown fox");
  const out = el("div", {});
  const run = async () => {
    try {
      const r = await apiPost("/api/tokenizers/encode", { project_id: pid, tokenizer_id: tid, text: input.value });
      out.replaceChildren(
        el("div", { class: "card" }, el("h2", {}, `${r.token_count} tokens`),
          el("div", { class: "chips" }, ...r.ids.map((id) => el("span", { class: "chip" }, String(id)))),
          el("p", { class: "hint" }, `decoded: ${r.decoded}`)));
    } catch (e) { toast(e.message); }
  };
  body.replaceChildren(
    el("div", { class: "card" }, el("h2", {}, "Live encode"),
      el("label", {}, "Text"), input,
      el("div", { style: "margin-top:10px" }, el("button", { class: "run", onclick: run }, "Encode"))),
    out);
  run();
}

// Embedding Lab (project-scoped) ----------------------------------------------
const embState = { embeddingId: null };

PANELS.embedding = async function () {
  const m = main(); m.replaceChildren();
  if (!state.project) { requireProject(m, "Embedding Lab"); return; }
  m.append(el("div", { class: "panel-head" },
    el("h1", {}, "Embedding Lab"),
    el("p", {}, `Word2Vec CBOW embeddings for project \u201c${state.project.name}\u201d.`)));
  const cols = el("div", { class: "lab-cols" });
  m.append(cols);
  await renderEmbLab(cols);
};

async function renderEmbLab(cols) {
  const left = el("div", {});
  const right = el("div", {});
  cols.replaceChildren(left, right);

  let datasets = [], tokenizers = [], embeddings = [];
  try { datasets = (await apiPost("/api/datasets/list", { project_id: state.project.id })).datasets; } catch (_) {}
  try { tokenizers = (await apiPost("/api/tokenizers/list", { project_id: state.project.id })).tokenizers; } catch (_) {}
  try { embeddings = (await apiPost("/api/embeddings/list", { project_id: state.project.id })).embeddings; } catch (_) {}

  const dsSelect = el("select", {}, ...(
    datasets.length ? datasets.map((d) => el("option", { value: d.id }, d.name))
      : [el("option", { value: "" }, "\u2014 no datasets \u2014")]));
  const tokSelect = el("select", {}, ...(
    tokenizers.length ? tokenizers.map((t) => el("option", { value: t.id }, t.name + (t.status === "default" ? " (default)" : "")))
      : [el("option", { value: "" }, "\u2014 no tokenizers \u2014")]));
  const embName  = el("input", { type: "text", placeholder: "e.g. cbow-64" });
  const dimsIn   = el("input", { type: "number", value: "64",  min: "2" });
  const epochsIn = el("input", { type: "number", value: "5",   min: "1" });
  const windowIn = el("input", { type: "number", value: "2",   min: "1" });
  const seedIn   = el("input", { type: "number", value: "42" });
  const trainStatus = el("div", { class: "pill" });

  const doTrain = async () => {
    if (!dsSelect.value) { toast("Select a dataset first."); return; }
    if (!tokSelect.value) { toast("Select a tokenizer first."); return; }
    trainStatus.textContent = "Training\u2026 (may take a moment)";
    try {
      const r = await apiPost("/api/embeddings/train", {
        project_id: state.project.id,
        dataset_id: dsSelect.value,
        tokenizer_id: tokSelect.value,
        name: embName.value || "cbow",
        dims: Number(dimsIn.value) || 64,
        epochs: Number(epochsIn.value) || 5,
        window: Number(windowIn.value) || 2,
        seed: Number(seedIn.value) || 42,
      });
      const job = r.job;
      if (job.status === "failed") { toast("Training failed: " + job.error); trainStatus.textContent = ""; return; }
      embState.embeddingId = job.result?.embedding_id || null;
      trainStatus.textContent = "";
      await renderEmbLab(cols);
    } catch (e) { toast(e.message); trainStatus.textContent = ""; }
  };

  const listItems = embeddings.length
    ? embeddings.map((e) => {
        const isDefault = e.status === "default";
        return el("button", {
          class: "list-item" + (e.id === embState.embeddingId ? " active" : ""),
          onclick: () => { embState.embeddingId = e.id; renderEmbLab(cols); },
        },
          el("div", { style: "display:flex;justify-content:space-between;align-items:center" },
            el("div", {}, e.name),
            isDefault ? el("span", { class: "badge-active badge-status" }, "default") : null),
          el("div", { class: "sub" }, `${e.dims}d \u00b7 ${e.vocab_size} tokens \u00b7 ${e.algorithm}`));
      })
    : [el("div", { class: "pill" }, "none yet")];

  left.replaceChildren(
    el("div", { class: "card" }, el("h2", {}, "Embeddings"),
      el("div", { class: "list-panel" }, ...listItems)),
    el("div", { class: "card" }, el("h2", {}, "Train new embedding"),
      el("label", {}, "Dataset"), dsSelect,
      el("label", { style: "margin-top:10px" }, "Tokenizer"), tokSelect,
      el("label", { style: "margin-top:10px" }, "Name"), embName,
      el("div", { class: "row", style: "margin-top:10px" },
        el("div", {}, el("label", {}, "Dims"), dimsIn),
        el("div", {}, el("label", {}, "Epochs"), epochsIn),
        el("div", {}, el("label", {}, "Window"), windowIn),
        el("div", {}, el("label", {}, "Seed"), seedIn)),
      el("div", { class: "row", style: "margin-top:14px" },
        el("div", {}, el("button", { class: "run", onclick: doTrain }, "Train")),
        el("div", {}, trainStatus))));

  if (!embState.embeddingId) {
    right.replaceChildren(el("div", { class: "notice" }, "Select or train an embedding to inspect it."));
    return;
  }
  await renderEmbDetail(right, cols);
}

async function renderEmbDetail(right, cols) {
  const pid = state.project.id, eid = embState.embeddingId;
  let data;
  try { data = await apiPost("/api/embeddings/get", { project_id: pid, embedding_id: eid }); }
  catch (e) { toast(e.message); return; }

  const { manifest: mf, statistics: stats } = data;
  const isDefault = mf.status === "default";

  const setDefault = async () => {
    try { await apiPost("/api/embeddings/set_default", { project_id: pid, embedding_id: eid });
      await renderEmbLab(cols); } catch (e) { toast(e.message); }
  };
  const doDelete = async () => {
    if (!confirm(`Delete embedding \u201c${mf.name}\u201d?`)) return;
    try { await apiPost("/api/embeddings/delete", { project_id: pid, embedding_id: eid });
      embState.embeddingId = null; await renderEmbLab(cols); } catch (e) { toast(e.message); }
  };

  const tabs = ["overview", "loss", "neighbours", "projection"];
  const tabState = { t: right._embTab || "overview" };
  right._embTab = tabState.t;

  const body = el("div", {});
  const renderBody = () => {
    if (tabState.t === "overview")    renderEmbOverview(body, mf, stats);
    else if (tabState.t === "loss")   renderEmbLoss(body, stats);
    else if (tabState.t === "neighbours") renderEmbNeighbours(body, pid, eid);
    else renderEmbProjection(body, pid, eid);
  };

  const tabBar = el("div", { class: "tabs" }, ...tabs.map((t) =>
    el("button", { class: "tab" + (t === tabState.t ? " active" : ""),
      onclick: () => { tabState.t = t; right._embTab = t; renderBody(); } },
      t[0].toUpperCase() + t.slice(1))));

  const actions = el("div", { class: "row", style: "margin-bottom:14px" },
    el("div", {}, isDefault
      ? el("span", { class: "badge-active badge-status" }, "\u2713 project default")
      : el("button", { class: "ghost", onclick: setDefault }, "Set as default")),
    el("div", {}, el("button", { class: "ghost", onclick: doDelete }, "Delete")));

  right.replaceChildren(
    el("div", { class: "panel-head" },
      el("h1", {}, mf.name),
      el("p", {}, `${mf.algorithm} \u00b7 ${mf.dims}d \u00b7 ${mf.vocab_size} tokens`)),
    actions, tabBar, body);
  renderBody();
}

function renderEmbOverview(body, mf, stats) {
  const m = stats || {};
  body.replaceChildren(
    el("div", { class: "card" }, el("h2", {}, "Training metrics"), el("div", { class: "stats" },
      stat(mf.vocab_size, "vocab size"),
      stat(mf.dims, "dimensions"),
      stat(m.epochs ?? "\u2014", "epochs"),
      stat(m.final_loss != null ? m.final_loss.toFixed(4) : "\u2014", "final loss"),
      stat(m.nn_coherence != null ? m.nn_coherence.toFixed(4) : "\u2014", "NN coherence"),
      stat(m.coverage != null ? (m.coverage * 100).toFixed(1) + "%" : "\u2014", "coverage"),
      stat(m.training_time_s != null ? m.training_time_s + "s" : "\u2014", "training time"))),
    el("div", { class: "card" }, el("h2", {}, "Manifest"),
      el("div", { class: "tablewrap" }, el("table", {},
        el("tbody", {}, ...Object.entries({
          id: mf.id, algorithm: mf.algorithm,
          tokenizer: mf.tokenizer_id, dataset: mf.dataset_id,
          status: mf.status,
          created: (mf.created_at || "").slice(0, 19).replace("T", " "),
          "emb fingerprint": (mf.embedding_fingerprint || "").slice(0, 16),
          "tok fingerprint": (mf.tokenizer_fingerprint || "").slice(0, 16),
        }).map(([k, v]) => el("tr", {},
          el("td", { style: "color:var(--text-dim);width:160px" }, k),
          el("td", {}, String(v)))))))));
}

function renderEmbLoss(body, stats) {
  const history = stats?.loss_history || [];
  if (history.length < 2) {
    body.replaceChildren(el("div", { class: "pill" }, "Not enough loss data."));
    return;
  }
  const points = history.map((loss, i) => ({ epoch: i + 1, loss }));
  body.replaceChildren(
    el("div", { class: "card" }, el("h2", {}, "Training loss per epoch"),
      lineChart(points, { xKey: "epoch", yKey: "loss", color: "#f0a35e", label: "loss" })));
}

function renderEmbNeighbours(body, pid, eid) {
  const input = el("input", { type: "text", placeholder: "Type a word\u2026", value: "the" });
  const nIn = el("input", { type: "number", value: "10", min: "1", max: "50", style: "width:70px" });
  const out = el("div", {});
  const run = async () => {
    try {
      const r = await apiPost("/api/embeddings/query",
        { project_id: pid, embedding_id: eid, text: input.value, n: Number(nIn.value) || 10 });
      if (!r.results.length) { out.replaceChildren(el("div", { class: "pill" }, "No results.")); return; }
      out.replaceChildren(el("div", { class: "card" },
        el("h2", {}, `Nearest neighbours for \u201c${r.text}\u201d`),
        el("div", { class: "tablewrap" }, el("table", {},
          el("thead", {}, el("tr", {}, el("th", {}, "rank"), el("th", {}, "token"), el("th", {}, "similarity"))),
          el("tbody", {}, ...r.results.map((res, i) =>
            el("tr", {},
              el("td", {}, String(i + 1)),
              el("td", {}, res.token || String(res.token_id)),
              el("td", {}, res.similarity.toFixed(4)))))))));
    } catch (e) { toast(e.message); }
  };
  body.replaceChildren(
    el("div", { class: "card" }, el("h2", {}, "Nearest neighbours"),
      el("div", { class: "row" },
        el("div", {}, el("label", {}, "Word"), input),
        el("div", {}, el("label", {}, "n"), nIn),
        el("div", {}, el("button", { class: "run", onclick: run }, "Query"))),
      el("p", { class: "hint" }, "Uses the first token id produced by the tokenizer for the input word.")),
    out);
  run();
}

function renderEmbProjection(body, pid, eid) {
  const status = el("div", { class: "pill" }, "Loading projection\u2026");
  const canvas = el("canvas", { width: "560", height: "400", style: "width:100%;border:1px solid var(--border);border-radius:8px;background:var(--bg)" });
  body.replaceChildren(
    el("div", { class: "card" }, el("h2", {}, "2D PCA projection (top 200 tokens)"),
      el("p", { class: "hint" }, "Cached by embedding fingerprint. First call may take a moment."),
      status, canvas));

  apiPost("/api/embeddings/pca", { project_id: pid, embedding_id: eid, n: 200 })
    .then((r) => {
      status.textContent = "";
      const pts = r.points || [];
      if (!pts.length) { status.textContent = "No projection data."; return; }
      const ctx = canvas.getContext("2d");
      const W = canvas.width, H = canvas.height, pad = 30;
      const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
      const xmin = Math.min(...xs), xmax = Math.max(...xs);
      const ymin = Math.min(...ys), ymax = Math.max(...ys);
      const sx = (x) => pad + ((x - xmin) / Math.max(1e-9, xmax - xmin)) * (W - 2 * pad);
      const sy = (y) => H - pad - ((y - ymin) / Math.max(1e-9, ymax - ymin)) * (H - 2 * pad);
      ctx.clearRect(0, 0, W, H);
      ctx.font = "10px monospace";
      for (const p of pts) {
        const px = sx(p.x), py = sy(p.y);
        ctx.fillStyle = "#4fd1c5";
        ctx.beginPath(); ctx.arc(px, py, 2.5, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = "#8b98a8";
        ctx.fillText(p.token || String(p.token_id), px + 4, py + 4);
      }
    })
    .catch((e) => { status.textContent = "Error: " + e.message; });
}

// ── Coming-soon + require-project ─────────────────────────────────────────────
function comingSoon(labId) {
  const lab = state.registry.labs.find((l) => l.id === labId);
  main().replaceChildren(el("div", { class: "soon-panel" },
    el("div", { class: "big" }, lab?.icon || "🚧"),
    el("h1", {}, (lab?.name || "Lab") + " — Coming Soon"),
    el("p", {}, lab?.description || "")));
}
function requireProject(m, labName) {
  m.replaceChildren(el("div", { class: "panel-head" }, el("h1", {}, labName)),
    el("div", { class: "notice" },
      el("div", {}, `${labName} works inside a project. Open or create one first.`),
      el("div", { style: "margin-top:14px" },
        el("button", { class: "run", onclick: () => select("projects") }, "Go to Project Explorer"))));
}

// ── Navigation ────────────────────────────────────────────────────────────────
function updateContext() {
  const c = document.getElementById("context");
  const lab = state.registry.labs.find((l) => l.id === state.labId);
  if (lab && lab.scope === "scratch") { c.className = "context context-scratch"; c.textContent = "Scratch Mode — nothing is saved"; }
  else if (state.project) { c.className = "context context-project"; c.textContent = "Project: " + state.project.name; }
  else { c.className = "context context-none"; c.textContent = "No project open"; }
}

function renderNav() {
  const nav = document.getElementById("nav");
  const labs = state.registry.labs;
  const groups = [
    ["", labs.filter((l) => l.scope === "home")],
    ["Scratch tools", labs.filter((l) => l.scope === "scratch")],
    ["Project labs", labs.filter((l) => l.scope === "project")],
  ];
  nav.replaceChildren();
  for (const [label, items] of groups) {
    if (!items.length) continue;
    if (label) nav.append(el("div", { class: "nav-group-label" }, label));
    for (const lab of items) {
      const soon = lab.status === "coming_soon";
      nav.append(el("button", {
        class: "nav-item" + (lab.id === state.labId ? " active" : "") + (soon ? " disabled" : ""),
        "data-id": lab.id, onclick: () => !soon && select(lab.id),
      },
        el("div", { class: "row-line" },
          el("span", { class: "ico" }, lab.icon || "•"),
          el("span", { class: "t" }, lab.name),
          soon ? el("span", { class: "soon" }, "soon") : null),
        el("span", { class: "d" }, lab.description)));
    }
  }
}

function select(labId) {
  const lab = state.registry.labs.find((l) => l.id === labId);
  if (!lab || lab.status === "coming_soon") { state.labId = labId; renderNav(); updateContext(); comingSoon(labId); return; }
  state.labId = labId;
  renderNav(); updateContext();
  (PANELS[labId] || (() => comingSoon(labId)))();
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
async function boot() {
  try {
    state.registry = await apiGet("/api/labs");
    document.getElementById("version").textContent = "v" + state.registry.version + " · research OS";
    renderNav();
    select("projects");   // Project Explorer is home
  } catch (e) { toast("Failed to load AION: " + e.message); }
}
boot();
