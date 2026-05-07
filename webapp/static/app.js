/* app.js — Frontend logic for SAP RPT-1 Benchmarking Web App */
"use strict";

// ── Constants ────────────────────────────────────────────────────────────────
const MODEL_COLORS = {
  "XGBoost":           "#f59e0b",
  "LightGBM":          "#10b981",
  "CatBoost":          "#6366f1",
  "TabPFN":            "#3b82f6",
  "SAP-RPT1":          "#ec4899",
  "Voting Ensemble":   "#fbbf24",
  "Stacking Ensemble": "#a78bfa",
};

const MODEL_EMOJIS = {
  "XGBoost":           "🟡",
  "LightGBM":          "🟢",
  "CatBoost":          "🟣",
  "TabPFN":            "🟦",
  "SAP-RPT1":          "🩷",
  "Voting Ensemble":   "🏆",
  "Stacking Ensemble": "✨",
};

const ENSEMBLE_NAMES = ["Voting Ensemble", "Stacking Ensemble"];

// ── DOM refs ─────────────────────────────────────────────────────────────────
const dropZone       = document.getElementById("drop-zone");
const fileInput      = document.getElementById("file-input");
const uploadError    = document.getElementById("upload-error");
const uploadSection  = document.getElementById("upload-section");
const previewSection = document.getElementById("preview-section");
const previewMeta    = document.getElementById("preview-meta");
const targetSelect   = document.getElementById("target-select");
const previewTable   = document.getElementById("preview-table");
const changeFileBtn  = document.getElementById("change-file-btn");
const runBtn         = document.getElementById("run-btn");
const loadingSection = document.getElementById("loading-section");
const resultsSection = document.getElementById("results-section");
const resetBtn       = document.getElementById("reset-btn");

let currentFile = null;
let chartInstances = [];

// ── Drag & Drop ───────────────────────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });

dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave",  () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f) handleFile(f);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

changeFileBtn.addEventListener("click", resetToUpload);
resetBtn.addEventListener("click", resetToUpload);

// ── File handling ─────────────────────────────────────────────────────────────
async function handleFile(file) {
  uploadError.hidden = true;

  if (!file.name.endsWith(".csv")) {
    showError("Please upload a .csv file.");
    return;
  }

  const MAX_MB = 5;
  if (file.size > MAX_MB * 1024 * 1024) {
    showError(`File is too large (${(file.size / 1048576).toFixed(1)} MB). Maximum is ${MAX_MB} MB.`);
    return;
  }

  currentFile = file;

  const fd = new FormData();
  fd.append("file", file);

  try {
    const res = await fetch("/preview", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json();
      showError(err.detail || "Failed to read CSV.");
      return;
    }
    const data = await res.json();
    renderPreview(data, file);
  } catch (e) {
    showError("Network error: " + e.message);
  }
}

function renderPreview(data, file) {
  // Meta badges
  previewMeta.innerHTML = `
    <span class="meta-badge">📄 ${file.name}</span>
    <span class="meta-badge">${data.n_rows.toLocaleString()} rows</span>
    <span class="meta-badge">${data.n_cols} columns</span>
  `;

  // Target column selector
  targetSelect.innerHTML = "";
  data.columns.forEach(col => {
    const opt = document.createElement("option");
    opt.value = col;
    opt.textContent = col;
    if (col === data.default_target) opt.selected = true;
    targetSelect.appendChild(opt);
  });

  // Preview table
  const cols = data.columns;
  let thead = "<thead><tr>" + cols.map(c => `<th class="${c === data.default_target ? 'target-col' : ''}">${esc(c)}</th>`).join("") + "</tr></thead>";
  let tbody = "<tbody>" + data.preview.map(row =>
    "<tr>" + cols.map(c => `<td class="${c === data.default_target ? 'target-col' : ''}">${esc(String(row[c] ?? ""))}</td>`).join("") + "</tr>"
  ).join("") + "</tbody>";
  previewTable.innerHTML = thead + tbody;

  // Highlight target column on select change
  targetSelect.addEventListener("change", () => highlightTarget(targetSelect.value, cols));

  uploadSection.hidden = true;
  previewSection.hidden = false;
}

function highlightTarget(targetCol, cols) {
  previewTable.querySelectorAll("th, td").forEach(el => el.classList.remove("target-col"));
  const idx = cols.indexOf(targetCol);
  if (idx < 0) return;
  previewTable.querySelectorAll("tr").forEach(row => {
    const cells = row.querySelectorAll("th, td");
    if (cells[idx]) cells[idx].classList.add("target-col");
  });
}

// ── Run benchmark ─────────────────────────────────────────────────────────────
runBtn.addEventListener("click", async () => {
  if (!currentFile) return;

  previewSection.hidden = true;
  loadingSection.hidden = false;

  // Animate loader steps
  const steps = ["step-xgb", "step-lgb", "step-cat", "step-tabpfn", "step-sap", "step-vote", "step-stack"];
  const delays = [0, 150, 300, 450, 600, 750, 900];
  let stepIdx = 0;
  const stepTimer = setInterval(() => {
    if (stepIdx > 0) {
      document.getElementById(steps[stepIdx - 1])?.classList.remove("active");
      document.getElementById(steps[stepIdx - 1])?.classList.add("done");
    }
    if (stepIdx < steps.length) {
      document.getElementById(steps[stepIdx])?.classList.add("active");
      stepIdx++;
    } else {
      clearInterval(stepTimer);
    }
  }, 1400);

  const fd = new FormData();
  fd.append("file", currentFile);
  fd.append("target_col", targetSelect.value);

  try {
    const res = await fetch("/benchmark", { method: "POST", body: fd });
    if (!res.ok) {
      const err = await res.json();
      clearInterval(stepTimer);
      loadingSection.hidden = true;
      previewSection.hidden = false;
      showError(err.detail || "Benchmarking failed.");
      return;
    }
    const data = await res.json();
    clearInterval(stepTimer);
    loadingSection.hidden = true;
    renderResults(data);
  } catch (e) {
    clearInterval(stepTimer);
    loadingSection.hidden = true;
    previewSection.hidden = false;
    showError("Network error: " + e.message);
  }
});

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(data) {
  const { dataset_info, task, results, recommendation, n_folds } = data;
  const isCLF = task === "classification";
  const primaryKey = isCLF ? "roc_auc" : "r2";
  const primaryLabel = isCLF ? "ROC-AUC" : "R²";

  // ── Info bar
  const taskBadge = isCLF
    ? `<span class="info-tag">🏷 Classification</span>`
    : `<span class="info-tag green">📈 Regression</span>`;
  document.getElementById("info-bar").innerHTML = `
    <span class="info-tag">📄 ${esc(currentFile.name)}</span>
    ${taskBadge}
    <span class="info-tag">${dataset_info.n_samples.toLocaleString()} samples</span>
    <span class="info-tag">${dataset_info.n_features} features</span>
    <span class="info-tag">Target: <strong>${esc(dataset_info.target_col)}</strong></span>
    ${isCLF ? `<span class="info-tag pink">${dataset_info.n_classes} classes</span>` : ""}
    <span class="info-tag">${n_folds}-Fold CV</span>
  `;

  // ── KPI cards
  const kpiGrid = document.getElementById("kpi-grid");
  kpiGrid.innerHTML = "";

  const validModels = Object.entries(results).filter(([, v]) => !v.error);
  const bestEntry   = validModels.reduce((best, [name, v]) =>
    (v.mean[primaryKey] || 0) > (best[1].mean[primaryKey] || 0) ? [name, v] : best
  , validModels[0]);

  const kpis = [
    {
      label: "Best Model",
      value: bestEntry[0],
      sub:   `${primaryLabel}: ${fmt(bestEntry[1].mean[primaryKey])}`,
      color: MODEL_COLORS[bestEntry[0]],
    },
    {
      label: `Best ${primaryLabel}`,
      value: fmt(bestEntry[1].mean[primaryKey]),
      sub:   `± ${fmt(bestEntry[1].std[primaryKey])} std`,
      color: "#818cf8",
    },
    {
      label: "Models Evaluated",
      value: validModels.length,
      sub:   `${n_folds}-fold cross-validation`,
      color: "#10b981",
    },
    {
      label: "Dataset Size",
      value: dataset_info.n_samples.toLocaleString(),
      sub:   `${dataset_info.n_features} features · ${isCLF ? dataset_info.n_classes + " classes" : "regression"}`,
      color: "#f59e0b",
    },
  ];

  kpis.forEach(k => {
    const card = document.createElement("div");
    card.className = "kpi-card";
    card.style.setProperty("--accent-bar", k.color);
    card.innerHTML = `
      <div class="kpi-label">${k.label}</div>
      <div class="kpi-value" style="color:${k.color}">${esc(String(k.value))}</div>
      <div class="kpi-sub">${k.sub}</div>
    `;
    kpiGrid.appendChild(card);
  });

  // ── Legend
  const legendEl = document.getElementById("legend");
  legendEl.innerHTML = Object.entries(MODEL_COLORS).map(([name, color]) =>
    `<div class="legend-item">
      <div class="legend-dot" style="background:${color}"></div>
      <span>${name}${results[name]?.label ? ` (${results[name].label})` : ""}</span>
    </div>`
  ).join("");

  // ── Charts
  chartInstances.forEach(c => c.destroy());
  chartInstances = [];
  const chartsGrid = document.getElementById("charts-grid");
  chartsGrid.innerHTML = "";

  const metricsToChart = isCLF
    ? [["roc_auc", "ROC-AUC"], ["accuracy", "Accuracy"], ["f1_macro", "F1-Macro"]]
    : [["r2", "R²"], ["mae", "MAE"], ["rmse", "RMSE"]];

  metricsToChart.forEach(([key, label]) => {
    const modelNames = Object.keys(results).filter(n => !results[n].error && results[n].mean[key] != null);
    if (!modelNames.length) return;

    const vals  = modelNames.map(n => roundN(results[n].mean[key], 4));
    const errs  = modelNames.map(n => roundN(results[n].std[key] || 0, 4));
    const bgs   = modelNames.map(n => (MODEL_COLORS[n] || "#888") + "cc");
    const bords = modelNames.map(n => MODEL_COLORS[n] || "#888");

    const card = document.createElement("div");
    card.className = "chart-card";
    const canvasId = `chart-${key}`;
    card.innerHTML = `
      <h4>${label}</h4>
      <div class="chart-sub">${label} (mean ± std over ${n_folds} folds)</div>
      <canvas id="${canvasId}"></canvas>
    `;
    chartsGrid.appendChild(card);

    const minVal = Math.min(...vals);
    const maxVal = Math.max(...vals);
    const pad    = Math.max((maxVal - minVal) * 0.15, 0.02);

    const inst = new Chart(document.getElementById(canvasId), {
      type: "bar",
      data: {
        labels: modelNames,
        datasets: [{
          label,
          data: vals,
          backgroundColor: bgs,
          borderColor:     bords,
          borderWidth: 2,
          borderRadius: 8,
        }],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: ctx => `${label}: ${ctx.parsed.y.toFixed(4)} ± ${errs[ctx.dataIndex].toFixed(4)}`,
            },
          },
        },
        scales: {
          y: {
            min: Math.max(key === "roc_auc" || key === "accuracy" ? 0 : -Infinity, minVal - pad),
            max: key === "roc_auc" || key === "accuracy" ? Math.min(1, maxVal + pad) : maxVal + pad,
            grid: { color: "#1e2a4a" },
            ticks: { color: "#64748b", font: { size: 11 } },
          },
          x: {
            grid: { display: false },
            ticks: { color: "#94a3b8", font: { size: 12 } },
          },
        },
      },
    });
    chartInstances.push(inst);
  });

  // ── Full table
  const thead = document.getElementById("results-thead");
  const tbody = document.getElementById("results-tbody");

  const allMetrics = isCLF
    ? ["accuracy", "f1_macro", "roc_auc", "log_loss", "fit_time"]
    : ["r2", "mae", "rmse", "fit_time"];
  const metricLabels = isCLF
    ? ["Accuracy", "F1-Macro", "ROC-AUC", "Log Loss", "Fit Time"]
    : ["R²", "MAE", "RMSE", "Fit Time"];

  thead.innerHTML = "<tr><th>Model</th>" + metricLabels.map(l => `<th>${l}</th>`).join("") + "</tr>";
  tbody.innerHTML = Object.entries(results).map(([name, d]) => {
    if (d.error) return `<tr><td><span class="model-dot" style="background:${MODEL_COLORS[name] || '#888'}"></span>${name}</td><td colspan="${allMetrics.length}" style="color:#f87171">Error: ${esc(d.error)}</td></tr>`;
    const cells = allMetrics.map(k => {
      const v = d.mean[k];
      if (v == null) return `<td class="mono" style="color:#374151">—</td>`;
      const isTime = k === "fit_time";
      if (isTime) return `<td class="mono" style="color:#94a3b8">${v.toFixed(3)}s</td>`;
      const cls = scoreClass(v, k, task);
      return `<td class="mono ${cls}">${v.toFixed(4)}<span style="color:#374151;font-size:.7em"> ±${(d.std[k]||0).toFixed(3)}</span></td>`;
    }).join("");
    return `<tr><td><span class="model-dot" style="background:${MODEL_COLORS[name] || '#888'}"></span><strong>${name}</strong></td>${cells}</tr>`;
  }).join("");

  // ── Recommendations
  const recGrid = document.getElementById("recommendation-grid");
  recGrid.innerHTML = "";
  const recs = recommendation.recommendations || {};
  const recDefs = [
    { key: "best_overall",      label: "🏆 Best Overall",      winner: true  },
    { key: "production",        label: "🚀 Production Ready",  winner: false },
    { key: "best_accuracy",     label: "🎯 Highest Accuracy",  winner: false },
    { key: "best_speed",        label: "⚡ Fastest Training",  winner: false },
    { key: "best_consistency",  label: "🛡 Most Consistent",   winner: false },
  ];

  recDefs.forEach(({ key, label, winner }) => {
    const rec = recs[key];
    if (!rec) return;
    const color = MODEL_COLORS[rec.model] || "#888";
    const score = rec.score != null
      ? `<div class="rec-score">${recommendation.primary_metric}: ${typeof rec.score === "number" ? rec.score.toFixed(4) : rec.score}</div>`
      : "";
    const card = document.createElement("div");
    card.className = `rec-card${winner ? " winner" : ""}`;
    card.innerHTML = `
      <div class="rec-type">${label}</div>
      <div class="rec-model-name">
        ${winner ? '<span class="rec-trophy">🏆</span>' : ""}
        <span style="color:${color}">${rec.model}</span>
      </div>
      ${score}
      <p class="rec-reason">${esc(rec.reason)}</p>
    `;
    recGrid.appendChild(card);
  });

  // ── Ensemble Analysis section
  renderEnsembleSection(data.ensemble_info || {}, results, recommendation, task);

  // ── Feature Importance (SHAP)
  renderFeatureImportance(data.feature_importance || {}, data.best_model_name || "", task);

  // ── Interactive Playground
  generatePlayground(data.dataset_info.col_dtypes || {}, task, data.best_model_name || "");

  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ── Feature Importance (SHAP) ─────────────────────────────────────────────────
let _importanceChart = null;

function renderFeatureImportance(importance, modelName, task) {
  const title     = document.getElementById("importance-section-title");
  const container = document.getElementById("importance-container");
  const sub       = document.getElementById("importance-chart-sub");

  const entries = Object.entries(importance);
  if (!entries.length) {
    title.hidden     = true;
    container.hidden = true;
    return;
  }
  title.hidden     = false;
  container.hidden = false;

  const top     = entries.slice(0, 15);
  const labels  = top.map(([k]) => k);
  const values  = top.map(([, v]) => v);
  const colors  = labels.map((_, i) => `hsl(${200 + (i / labels.length) * 140}, 70%, 58%)`);

  sub.textContent = `· SHAP values from ${modelName} trained on full dataset`;

  if (_importanceChart) _importanceChart.destroy();
  _importanceChart = new Chart(document.getElementById("importance-chart"), {
    type: "bar",
    data: {
      labels,
      datasets: [{ data: values, backgroundColor: colors.map(c => c + "bb"),
        borderColor: colors, borderWidth: 2, borderRadius: 6 }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: ctx => `Importance: ${ctx.parsed.x.toFixed(5)}` } } },
      scales: {
        x: { min: 0, grid: { color: "#1e2a4a" }, ticks: { color: "#64748b", font: { size: 11 } } },
        y: { grid: { display: false }, ticks: { color: "#94a3b8", font: { size: 12 } } },
      },
    },
  });
}

// ── Interactive Playground ────────────────────────────────────────────────────
function generatePlayground(colDtypes, task, bestModelName) {
  const title     = document.getElementById("playground-section-title");
  const container = document.getElementById("playground-container");
  const form      = document.getElementById("playground-form");

  const cols = Object.keys(colDtypes);
  if (!cols.length) {
    title.hidden     = true;
    container.hidden = true;
    return;
  }
  title.hidden     = false;
  container.hidden = false;
  form.innerHTML   = "";

  cols.forEach(col => {
    const info  = colDtypes[col];
    const field = document.createElement("div");
    field.className = "pg-field";

    let input;
    if (info.type === "numeric") {
      input = document.createElement("input");
      input.type  = "number";
      input.step  = "any";
      input.value = info.mean != null ? Number(info.mean).toFixed(4) : "0";
      if (info.min != null) input.min = info.min;
      if (info.max != null) input.max = info.max;
    } else {
      input = document.createElement("select");
      (info.choices || []).forEach(c => {
        const opt = document.createElement("option");
        opt.value = c; opt.textContent = c;
        input.appendChild(opt);
      });
    }
    input.className    = "pg-input";
    input.id           = `pg-${col}`;
    input.dataset.col  = col;

    field.innerHTML = `<label class="pg-label" for="pg-${esc(col)}">${esc(col)}</label>`;
    field.appendChild(input);
    form.appendChild(field);
  });

  const doPred = debounce(runPlaygroundPrediction, 350);
  form.querySelectorAll(".pg-input").forEach(el => el.addEventListener("input", doPred));
  runPlaygroundPrediction();
}

function debounce(fn, delay) {
  let t;
  return function (...args) { clearTimeout(t); t = setTimeout(() => fn.apply(this, args), delay); };
}

async function runPlaygroundPrediction() {
  const form   = document.getElementById("playground-form");
  const badge  = document.getElementById("badge-value");
  const conf   = document.getElementById("badge-confidence");
  const proba  = document.getElementById("proba-display");

  const row = {};
  form.querySelectorAll(".pg-input").forEach(el => {
    const num = Number(el.value);
    row[el.dataset.col] = Number.isNaN(num) ? el.value : num;
  });

  badge.textContent = "…";
  badge.style.color = "#64748b";

  try {
    const res = await fetch("/predict_single", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(row),
    });
    if (!res.ok) { badge.textContent = "Error"; badge.style.color = "#f87171"; return; }

    const r = await res.json();
    if (r.task === "classification") {
      badge.textContent = r.prediction;
      badge.style.color = "#10b981";
      conf.textContent  = r.confidence != null ? `${(r.confidence * 100).toFixed(1)}% confidence` : "";
      if (r.probabilities) {
        proba.innerHTML = Object.entries(r.probabilities)
          .sort((a, b) => b[1] - a[1])
          .map(([cls, p]) => `
            <div class="proba-row">
              <span class="proba-cls">${esc(String(cls))}</span>
              <div class="proba-track"><div class="proba-fill" style="width:${(p * 100).toFixed(1)}%;background:${p > 0.5 ? "#10b981" : "#6366f1"}"></div></div>
              <span class="proba-pct">${(p * 100).toFixed(1)}%</span>
            </div>`).join("");
      }
    } else {
      badge.textContent = typeof r.prediction === "number" ? r.prediction.toFixed(4) : String(r.prediction);
      badge.style.color = "#f59e0b";
      conf.textContent  = "Regression output";
      proba.innerHTML   = "";
    }
  } catch (e) {
    badge.textContent = "Error";
    badge.style.color = "#f87171";
    conf.textContent  = e.message;
  }
}



// ── Ensemble Analysis renderer ────────────────────────────────────────────────
function renderEnsembleSection(ensembleInfo, results, recommendation, task) {
  const grid  = document.getElementById("ensemble-grid");
  const title = document.getElementById("ensemble-section-title");
  grid.innerHTML = "";

  const entries = Object.entries(ensembleInfo).filter(([name]) => results[name] && !results[name].error);
  if (!entries.length) {
    title.hidden = true;
    grid.hidden  = true;
    return;
  }
  title.hidden = false;
  grid.hidden  = false;

  const primaryKey   = task === "classification" ? "roc_auc" : "r2";
  const primaryLabel = task === "classification" ? "ROC-AUC" : "R²";

  // Find the best individual model score (excluding ensembles) for gain %
  const indivScores = Object.entries(results)
    .filter(([n, v]) => !ENSEMBLE_NAMES.includes(n) && !v.error && v.mean[primaryKey] != null)
    .map(([, v]) => v.mean[primaryKey]);
  const bestIndivScore = indivScores.length ? Math.max(...indivScores) : 0;

  entries.forEach(([name, info]) => {
    const cv    = results[name];
    const score = cv.mean[primaryKey] ?? 0;
    const std   = cv.std[primaryKey]  ?? 0;
    const ft    = cv.mean.fit_time    ?? 0;
    const color = MODEL_COLORS[name] || "#888";
    const gain  = bestIndivScore > 0 ? ((score - bestIndivScore) / bestIndivScore * 100) : 0;
    const gainStr = gain >= 0
      ? `<span class="gain-pos">▲ +${gain.toFixed(2)}% vs best individual</span>`
      : `<span class="gain-neg">▼ ${gain.toFixed(2)}% vs best individual</span>`;

    const componentPills = (info.components || []).map(c =>
      `<span class="comp-pill" style="border-color:${MODEL_COLORS[c] || '#888'};color:${MODEL_COLORS[c] || '#888'}">${c}</span>`
    ).join("");

    const metaTag = info.meta_learner
      ? `<div class="ens-meta">Meta-learner: <strong>${esc(info.meta_learner)}</strong></div>` : "";

    const card = document.createElement("div");
    card.className = "ens-card";
    card.style.setProperty("--ens-color", color);
    card.innerHTML = `
      <div class="ens-header">
        <span class="ens-emoji">${MODEL_EMOJIS[name] || "🧩"}</span>
        <span class="ens-name" style="color:${color}">${name}</span>
        <span class="ens-type-badge">${info.type === "voting" ? "Soft Voting" : "Stacking"}</span>
      </div>
      <div class="ens-score">
        <span class="ens-score-val">${score.toFixed(4)}</span>
        <span class="ens-score-label"> ${primaryLabel} ± ${std.toFixed(3)}</span>
      </div>
      <div class="ens-gain">${gainStr}</div>
      ${metaTag}
      <div class="ens-desc">${esc(info.description || "")}</div>
      <div class="ens-components-label">Component Models</div>
      <div class="ens-components">${componentPills}</div>
      <div class="ens-footer">Avg fit time: ${ft.toFixed(3)}s per fold</div>
    `;
    grid.appendChild(card);
  });
}



// ── Helpers ───────────────────────────────────────────────────────────────────
function resetToUpload() {
  currentFile = null;
  fileInput.value = "";
  uploadError.hidden = true;
  previewSection.hidden = true;
  loadingSection.hidden = true;
  resultsSection.hidden = true;
  uploadSection.hidden = false;
  chartInstances.forEach(c => c.destroy());
  chartInstances = [];
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function showError(msg) {
  uploadError.textContent = msg;
  uploadError.hidden = false;
}

function fmt(v) {
  if (v == null || isNaN(v)) return "—";
  return Number(v).toFixed(4);
}

function roundN(v, n) {
  return Math.round(v * Math.pow(10, n)) / Math.pow(10, n);
}

function esc(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function scoreClass(v, metric, task) {
  if (metric === "fit_time") return "";
  const higherBetter = !["mae", "rmse", "mse", "log_loss"].includes(metric);
  if (!higherBetter) {
    if (v < 0.1)  return "col-excellent";
    if (v < 0.3)  return "col-good";
    if (v < 0.5)  return "col-fair";
    return "col-poor";
  }
  if (metric === "roc_auc" || metric === "accuracy") {
    if (v >= 0.95) return "col-excellent";
    if (v >= 0.88) return "col-good";
    if (v >= 0.75) return "col-fair";
    return "col-poor";
  }
  if (metric === "r2") {
    if (v >= 0.75) return "col-excellent";
    if (v >= 0.5)  return "col-good";
    if (v >= 0.25) return "col-fair";
    return "col-poor";
  }
  if (v >= 0.85) return "col-excellent";
  if (v >= 0.70) return "col-good";
  if (v >= 0.55) return "col-fair";
  return "col-poor";
}
