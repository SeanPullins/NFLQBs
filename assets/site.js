const paths = {
  projections: "model/projections.csv",
  historical: "model/historical_scored.csv",
  indicators: "model/indicators.csv",
  metrics: "model/artifacts/cv_metrics.json",
};

const SCORE_MODES = {
  draft: {
    label: "Draft Adj",
    shortLabel: "Hit",
    column: "draft_adjusted_hit_prob",
    note: "Best default after the draft: starts with draft slot, then lets college data raise the score without dropping below the market baseline.",
  },
  pff: {
    label: "PFF Pre",
    shortLabel: "PFF",
    column: "model_hit_prob",
    note: "Pre-draft college lens with PFF charting included; useful for seeing what the data said before NFL teams picked.",
  },
  noPff: {
    label: "No-PFF",
    shortLabel: "No-PFF",
    column: "model_hit_prob_no_pff",
    note: "Same college model with PFF removed; this shows whether charting is doing real work for a player.",
  },
  market: {
    label: "Market",
    shortLabel: "Market",
    column: "model_hit_prob_pick_only",
    note: "Draft slot only: what history says about a QB picked in this range before looking at college details.",
  },
};

const state = {
  projections: [],
  historical: [],
  indicators: [],
  metrics: null,
  year: "all",
  scoreMode: "draft",
  query: "",
  sortKey: "_score",
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

function signedPct(value, digits = 0) {
  const number = toNumber(value);
  if (number === null) return "--";
  const sign = number > 0 ? "+" : "";
  return `${sign}${(number * 100).toFixed(digits)}%`;
}

function signedRank(value) {
  const number = toNumber(value);
  if (number === null) return "--";
  const sign = number > 0 ? "+" : "";
  return `${sign}${number}`;
}

function hitProb(row) {
  return toNumber(row.draft_adjusted_hit_prob) ?? toNumber(row.model_hit_prob);
}

function scoreMode() {
  return SCORE_MODES[state.scoreMode] ?? SCORE_MODES.draft;
}

function scoreValue(row, mode = state.scoreMode) {
  const config = SCORE_MODES[mode] ?? SCORE_MODES.draft;
  return toNumber(row[config.column]) ?? hitProb(row);
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

function plainOutcome(value) {
  const number = toNumber(value);
  if (number === null) return "not enough information";
  if (number >= 0.6) return "a strong starter bet";
  if (number >= 0.35) return "a serious upside bet";
  if (number >= 0.15) return "a risky but live bet";
  if (number >= 0.07) return "a developmental long shot";
  return "a deep long shot";
}

function pffRead(row) {
  const delta = toNumber(row.pff_model_delta);
  if (delta === null) return "PFF is not available enough to change the read.";
  if (delta >= 0.05) return `PFF lifted the profile by ${pct(Math.abs(delta))}.`;
  if (delta <= -0.05) return `PFF lowered the profile by ${pct(Math.abs(delta))}.`;
  return "PFF mostly agrees with the no-PFF college model.";
}

function draftRead(row) {
  const round = cleanNumber(row.round);
  const pick = cleanNumber(row.pick);
  const market = toNumber(row.model_hit_prob_pick_only);
  if (!round) return "There is no draft slot, so the score leans on college and testing data.";
  return `The draft slot, R${round}${pick ? ` / ${pick}` : ""}, gives a market baseline of ${pct(market)}.`;
}

function plainRead(row) {
  return `${row.canonical_name} grades as ${plainOutcome(hitProb(row))}. ${draftRead(row)} ${pffRead(row)}`;
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
  const activeScore = scoreMode();
  const rows = state.projections
    .filter((row) => state.year === "all" || row.draft_season === state.year)
    .filter((row) => {
      if (!query) return true;
      return `${row.canonical_name} ${row.college}`.toLowerCase().includes(query);
    })
    .map((row) => ({ ...row }));

  const rankBy = (key, target) => {
    [...rows]
      .sort((a, b) => {
        const av = toNumber(a[key]) ?? -Infinity;
        const bv = toNumber(b[key]) ?? -Infinity;
        return bv - av || String(a.canonical_name).localeCompare(String(b.canonical_name));
      })
      .forEach((row, index) => {
        row[target] = index + 1;
      });
  };

  rankBy("model_hit_prob", "_pffRank");
  rankBy("model_hit_prob_no_pff", "_noPffRank");
  rankBy("draft_adjusted_hit_prob", "_draftRank");
  rankBy("model_hit_prob_pick_only", "_marketRank");
  rows.forEach((row) => {
    row._rankMove = row._noPffRank - row._pffRank;
    row._score = toNumber(row[activeScore.column]) ?? hitProb(row);
  });

  return rows
    .sort((a, b) => {
      const key = state.sortKey;
      const numericKeys = [
        "model_hit_prob",
        "model_hit_prob_no_pff",
        "model_hit_prob_pick_only",
        "model_hit_prob_post_draft",
        "model_hit_prob_post_draft_pff",
        "draft_adjusted_hit_prob",
        "_score",
        "pff_model_delta",
        "_rankMove",
        "expected_tier",
        "draft_season",
      ];
      const av = numericKeys.includes(key) ? toNumber(a[key]) : a[key];
      const bv = numericKeys.includes(key) ? toNumber(b[key]) : b[key];
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

function renderScoreModes() {
  const host = $("#scoreModes");
  const note = $("#scoreNote");
  host.innerHTML = Object.entries(SCORE_MODES)
    .map(([key, config]) => `
      <button type="button" data-score="${key}" aria-pressed="${state.scoreMode === key}">
        ${escapeHtml(config.label)}
      </button>
    `)
    .join("");

  host.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      state.scoreMode = button.dataset.score;
      state.sortKey = "_score";
      state.sortDir = "desc";
      renderAll();
    });
  });

  if (note) note.textContent = scoreMode().note;
}

function renderBoardSummary(rows) {
  const host = $("#boardSummary");
  if (!rows.length) {
    host.innerHTML = "";
    return;
  }

  const scores = rows.map((row) => toNumber(row._score)).filter((value) => value !== null);
  const avg = scores.length ? scores.reduce((sum, value) => sum + value, 0) / scores.length : null;
  const top = [...rows].sort((a, b) => (toNumber(b._score) ?? -Infinity) - (toNumber(a._score) ?? -Infinity))[0];
  const pffUp = rows.filter((row) => (toNumber(row.pff_model_delta) ?? 0) >= 0.05).length;
  const pffDown = rows.filter((row) => (toNumber(row.pff_model_delta) ?? 0) <= -0.05).length;
  const label = scoreMode().label;

  host.innerHTML = `
    <div class="summary-item"><span>QBs</span><strong>${rows.length}</strong></div>
    <div class="summary-item"><span>Score</span><strong>${escapeHtml(label)}</strong></div>
    <div class="summary-item"><span>Top</span><strong>${escapeHtml(top?.canonical_name || "--")}</strong></div>
    <div class="summary-item"><span>Average</span><strong>${pct(avg)}</strong></div>
    <div class="summary-item"><span>PFF movers</span><strong>${pffUp} up / ${pffDown} down</strong></div>
  `;
}

function renderTable() {
  const rows = filteredRows();
  const tbody = $("#projectionRows");
  const scoreButton = $("#scoreSortButton");
  if (scoreButton) scoreButton.textContent = scoreMode().shortLabel;
  renderBoardSummary(rows);

  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="loading">No quarterbacks match this view.</td></tr>`;
    renderDetail(null);
    return;
  }

  if (!state.selectedName || !rows.some((row) => row.canonical_name === state.selectedName)) {
    state.selectedName = rows[0].canonical_name;
  }

  tbody.innerHTML = rows
    .map((row, index) => {
      const selected = row.canonical_name === state.selectedName ? " selected" : "";
      const prob = Math.max(0, Math.min(100, (toNumber(row._score) ?? 0) * 100));
      const indicators = splitIndicators(row.top_positive_indicators).slice(0, 2);
      const pick = cleanNumber(row.pick);
      const round = cleanNumber(row.round);
      const draftText = round ? `R${round}${pick ? ` / ${pick}` : ""}` : "Undrafted";
      const deltaNumber = toNumber(row.pff_model_delta);
      const deltaClass = deltaNumber === null ? "" : deltaNumber >= 0 ? "good" : "bad";
      const rankMove = toNumber(row._rankMove);
      const rankMoveClass = rankMove === null || rankMove === 0 ? "" : rankMove > 0 ? "good" : "bad";

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
              <strong>${pct(row._score)}</strong>
            </div>
          </td>
          <td><span class="delta ${deltaClass}">${signedPct(row.pff_model_delta)}</span></td>
          <td><span class="rank-delta ${rankMoveClass}">${signedRank(row._rankMove)}</span></td>
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
  const scoreRows = [
    ["Draft Adj", row.draft_adjusted_hit_prob],
    ["PFF Pre", row.model_hit_prob],
    ["No-PFF", row.model_hit_prob_no_pff],
    ["Market", row.model_hit_prob_pick_only],
  ];

  pane.innerHTML = `
    <div class="detail-kicker">${escapeHtml(row.draft_season)} class / ${escapeHtml(label)}</div>
    <h2>${escapeHtml(row.canonical_name)}</h2>
    <p class="detail-school">${escapeHtml(row.college || "Unknown school")} / ${escapeHtml(draftText)}</p>
    <div class="plain-read">
      <span>Plain read</span>
      <p>${escapeHtml(plainRead(row))}</p>
    </div>

    <div class="score-stack">
      ${scoreRows.map(([name, value]) => {
        const number = toNumber(value);
        const width = Math.max(0, Math.min(100, (number ?? 0) * 100));
        return `
          <div class="score-bar">
            <span>${escapeHtml(name)}</span>
            <div class="prob-track" aria-hidden="true">
              <span class="prob-fill" style="--w: ${width}%"></span>
            </div>
            <strong>${pct(number)}</strong>
          </div>
        `;
      }).join("")}
    </div>

    <div class="detail-stats">
      <div class="detail-stat"><span>PFF Delta</span><strong>${signedPct(row.pff_model_delta)}</strong></div>
      <div class="detail-stat"><span>PFF Rank</span><strong>${signedRank(row._rankMove)}</strong></div>
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
    ["Draft Adj AUC", metrics.draft_adjusted_market_guarded?.auc, "Market-guarded hit score"],
    ["Post-draft AUC", metrics.post_draft?.auc, "College plus draft slot"],
    ["Pick-only AUC", metrics.pick_only?.auc, "Draft slot alone"],
    ["PFF Pre AUC", metrics.pre_draft?.auc, "College charting lens"],
    ["PFF Lift", metrics.pff_lift_over_no_pff, "Gain over no-PFF college model"],
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
  renderScoreModes();
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
    $("#projectionRows").innerHTML = `<tr><td colspan="8" class="loading">Could not load projection data.</td></tr>`;
    $("#playerDetail").innerHTML = `<div class="detail-empty">Could not load model artifacts.</div>`;
  });
