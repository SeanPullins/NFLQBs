const paths = {
  projections: "model/projections.csv",
  historical: "model/historical_scored.csv",
  indicators: "model/indicators.csv",
  metrics: "model/artifacts/cv_metrics.json",
};

const state = {
  projections: [],
  historical: [],
  indicators: [],
  metrics: null,
  year: "all",
  query: "",
  sortKey: "model_hit_prob",
  sortDir: "desc",
  selectedName: "",
};

const $ = (selector) => document.querySelector(selector);

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"' && inQuotes && next === '"') {
      cell += '"';
      i += 1;
    } else if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      row.push(cell);
      cell = "";
    } else if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") i += 1;
      row.push(cell);
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }

  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }

  const [headers, ...data] = rows;
  return data.map((values) => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])));
}

function toNumber(value) {
  if (value === null || value === undefined || String(value).trim() === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function pct(value, digits = 0) {
  const number = toNumber(value);
  return number === null ? "--" : `${(number * 100).toFixed(digits)}%`;
}

function oneDecimal(value) {
  const number = toNumber(value);
  return number === null ? "--" : number.toFixed(1);
}

function cleanNumber(value) {
  const number = toNumber(value);
  if (number === null) return "";
  return Number.isInteger(number) ? String(number) : number.toFixed(0);
}

function splitIndicators(value) {
  return String(value || "")
    .split(";")
    .map((item) => item.trim())
    .filter(Boolean);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function years() {
  return ["all", ...new Set(state.projections.map((row) => row.draft_season).filter(Boolean))].sort((a, b) => {
    if (a === "all") return -1;
    if (b === "all") return 1;
    return Number(a) - Number(b);
  });
}

function filteredRows() {
  const query = state.query.trim().toLowerCase();
  return state.projections
    .filter((row) => state.year === "all" || row.draft_season === state.year)
    .filter((row) => {
      if (!query) return true;
      return `${row.canonical_name} ${row.college}`.toLowerCase().includes(query);
    })
    .sort((a, b) => {
      const key = state.sortKey;
      const av = ["model_hit_prob", "expected_tier", "draft_season"].includes(key) ? toNumber(a[key]) : a[key];
      const bv = ["model_hit_prob", "expected_tier", "draft_season"].includes(key) ? toNumber(b[key]) : b[key];
      let result = 0;
      if (typeof av === "number" && typeof bv === "number") result = av - bv;
      else result = String(av ?? "").localeCompare(String(bv ?? ""));
      return state.sortDir === "asc" ? result : -result;
    });
}

function renderYearFilters() {
  const host = $("#yearFilters");
  host.innerHTML = years()
    .map((year) => {
      const label = year === "all" ? "All" : year;
      return `<button type="button" data-year="${year}" aria-pressed="${state.year === year}">${label}</button>`;
    })
    .join("");

  host.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.year = button.dataset.year;
      renderAll();
    });
  });
}

function renderTable() {
  const rows = filteredRows();
  const tbody = $("#projectionRows");

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="loading">No quarterbacks match this view.</td></tr>`;
    renderDetail(null);
    return;
  }

  if (!state.selectedName || !rows.some((row) => row.canonical_name === state.selectedName)) {
    state.selectedName = rows[0].canonical_name;
  }

  tbody.innerHTML = rows
    .map((row, index) => {
      const selected = row.canonical_name === state.selectedName ? " selected" : "";
      const prob = Math.max(0, Math.min(100, (toNumber(row.model_hit_prob) ?? 0) * 100));
      const indicators = splitIndicators(row.top_positive_indicators).slice(0, 2);
      const pick = cleanNumber(row.pick);
      const round = cleanNumber(row.round);
      const draftText = round ? `R${round}${pick ? ` / ${pick}` : ""}` : "Undrafted";

      return `
        <tr class="${selected}" tabindex="0" data-name="${escapeHtml(row.canonical_name)}">
          <td class="rank">${index + 1}</td>
          <td>
            <div class="qb-name">
              <strong>${escapeHtml(row.canonical_name)}</strong>
              <span>${escapeHtml(row.college || "Unknown school")}</span>
            </div>
          </td>
          <td>
            <strong>${escapeHtml(row.draft_season)}</strong>
            <div class="muted">${escapeHtml(draftText)}</div>
          </td>
          <td class="prob-cell">
            <div class="prob-row">
              <div class="prob-track" aria-hidden="true">
                <span class="prob-fill" style="--w: ${prob}%"></span>
              </div>
              <strong>${pct(row.model_hit_prob)}</strong>
            </div>
          </td>
          <td>${oneDecimal(row.expected_tier)}</td>
          <td>
            <div class="pill-line">
              ${indicators.map((item) => `<span class="pill">${escapeHtml(item)}</span>`).join("")}
            </div>
          </td>
        </tr>
      `;
    })
    .join("");

  tbody.querySelectorAll("tr[data-name]").forEach((row) => {
    const select = () => {
      state.selectedName = row.dataset.name;
      renderTable();
    };
    row.addEventListener("click", select);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select();
      }
    });
  });

  renderDetail(rows.find((row) => row.canonical_name === state.selectedName));
}

function renderDetail(row) {
  const pane = $("#playerDetail");
  if (!row) {
    pane.innerHTML = `<div class="detail-empty">No profile selected.</div>`;
    return;
  }

  const positives = splitIndicators(row.top_positive_indicators);
  const negatives = splitIndicators(row.top_negative_indicators);
  const round = cleanNumber(row.round);
  const pick = cleanNumber(row.pick);
  const draftText = round ? `Round ${round}${pick ? `, pick ${pick}` : ""}` : "No draft slot";
  const label = row.label_status ? row.label_status.replace("_", " ") : "model score";

  pane.innerHTML = `
    <div class="detail-kicker">${escapeHtml(row.draft_season)} class / ${escapeHtml(label)}</div>
    <h2>${escapeHtml(row.canonical_name)}</h2>
    <p class="detail-school">${escapeHtml(row.college || "Unknown school")} / ${escapeHtml(draftText)}</p>

    <div class="detail-stats">
      <div class="detail-stat"><span>Hit Prob</span><strong>${pct(row.model_hit_prob)}</strong></div>
      <div class="detail-stat"><span>Expected Tier</span><strong>${oneDecimal(row.expected_tier)}</strong></div>
      <div class="detail-stat"><span>Percentile</span><strong>${oneDecimal(row.percentile_vs_history)}</strong></div>
      <div class="detail-stat"><span>Actual Tier</span><strong>${oneDecimal(row.actual_tier_or_projection)}</strong></div>
    </div>

    <div class="detail-block">
      <h3>Positive Indicators</h3>
      <div class="pill-line">${positives.map((item) => `<span class="pill">${escapeHtml(item)}</span>`).join("") || "<p>None listed.</p>"}</div>
    </div>

    <div class="detail-block">
      <h3>Negative Indicators</h3>
      <div class="pill-line">${negatives.map((item) => `<span class="pill">${escapeHtml(item)}</span>`).join("") || "<p>None listed.</p>"}</div>
    </div>

    <div class="detail-block">
      <h3>Data Coverage</h3>
      <p>${escapeHtml(row.data_coverage || "Not listed")}</p>
    </div>

    ${row.scout_note ? `<div class="detail-block"><h3>Scout Note</h3><p>${escapeHtml(row.scout_note)}</p></div>` : ""}
  `;
}

function renderMetrics() {
  const metrics = state.metrics || {};
  const cards = [
    ["Pre-draft AUC", metrics.pre_draft?.auc, "College and combine only"],
    ["Post-draft AUC", metrics.post_draft?.auc, "Adds draft capital"],
    ["Pick-only AUC", metrics.pick_only?.auc, "The market baseline"],
    ["College lift", metrics.college_lift_over_pick, "AUC over pick-only"],
  ];

  $("#metricGrid").innerHTML = cards
    .map(([label, value, note]) => `
      <div class="metric-card">
        <span>${escapeHtml(label)}</span>
        <strong>${value ?? "--"}</strong>
        <p>${escapeHtml(note)}</p>
      </div>
    `)
    .join("");
}

function renderSignals() {
  const topSignals = state.indicators
    .filter((row) => row.group !== "pff")
    .sort((a, b) => (toNumber(b.abs_auc) ?? 0) - (toNumber(a.abs_auc) ?? 0))
    .slice(0, 8);

  $("#signalList").innerHTML = topSignals
    .map((row) => {
      const evidence = String(row.evidence_strength || "signal").toLowerCase();
      const className = evidence.includes("moderate") ? "moderate" : evidence.includes("weak") ? "weak" : "";
      return `
        <div class="signal-row">
          <div>
            <strong>${escapeHtml(row.metric)}</strong>
            <small>${escapeHtml(row.direction)}</small>
          </div>
          <span class="auc">${oneDecimal((toNumber(row.auc) ?? 0) * 100)}</span>
          <span class="evidence ${className}">${escapeHtml(row.evidence_strength || "signal")}</span>
        </div>
      `;
    })
    .join("");
}

function renderHistory() {
  const rows = [...state.historical]
    .sort((a, b) => (toNumber(b.model_hit_prob_pre_draft) ?? 0) - (toNumber(a.model_hit_prob_pre_draft) ?? 0))
    .slice(0, 8);

  $("#historyGrid").innerHTML = rows
    .map((row) => {
      const prob = Math.max(0, Math.min(100, (toNumber(row.model_hit_prob_pre_draft) ?? 0) * 100));
      return `
        <article class="history-card">
          <strong>${escapeHtml(row.canonical_name)}</strong>
          <span>${escapeHtml(row.college)} / ${escapeHtml(row.draft_year)} / Tier ${escapeHtml(row.success_tier)}</span>
          <div class="prob-row">
            <div class="prob-track" aria-hidden="true">
              <span class="prob-fill" style="--w: ${prob}%"></span>
            </div>
            <strong>${pct(row.model_hit_prob_pre_draft)}</strong>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderAll() {
  renderYearFilters();
  renderTable();
  renderMetrics();
  renderSignals();
  renderHistory();
}

function bindEvents() {
  $("#searchInput").addEventListener("input", (event) => {
    state.query = event.target.value;
    renderTable();
  });

  document.querySelectorAll(".sort-button").forEach((button) => {
    button.addEventListener("click", () => {
      const nextKey = button.dataset.sort;
      if (state.sortKey === nextKey) {
        state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
      } else {
        state.sortKey = nextKey;
        state.sortDir = nextKey === "canonical_name" ? "asc" : "desc";
      }
      renderTable();
    });
  });
}

async function loadData() {
  const [projectionText, historicalText, indicatorText, metrics] = await Promise.all([
    fetch(paths.projections).then((response) => response.text()),
    fetch(paths.historical).then((response) => response.text()),
    fetch(paths.indicators).then((response) => response.text()),
    fetch(paths.metrics).then((response) => response.json()),
  ]);

  state.projections = parseCsv(projectionText);
  state.historical = parseCsv(historicalText);
  state.indicators = parseCsv(indicatorText);
  state.metrics = metrics;
}

bindEvents();
loadData()
  .then(renderAll)
  .catch((error) => {
    console.error(error);
    $("#projectionRows").innerHTML = `<tr><td colspan="6" class="loading">Could not load projection data.</td></tr>`;
    $("#playerDetail").innerHTML = `<div class="detail-empty">Could not load model artifacts.</div>`;
  });
