const APP_SOURCE = "csv-agent-app";
const LAYOUT_STORAGE_KEY = "csv-agent-explorer-layout";
const DETAIL_STORAGE_KEY = "csv-agent-explorer-details-open";
const SPLIT_BREAKPOINT = 1180;
const DEFAULT_WORKSPACE_PERCENT = 0.58;
const MIN_WORKSPACE_WIDTH = 360;
const MIN_CHAT_WIDTH = 320;

const state = {
  ui: null,
  tableOptions: {
    wrapCells: false,
    compactRows: false,
  },
};

const shell = document.querySelector(".shell");
const workspacePane = document.querySelector(".workspace-pane");
const divider = document.getElementById("shell-divider");
const iframe = document.getElementById("chainlit-frame");
const fileManager = document.getElementById("file-manager");
const summaryPanel = document.getElementById("summary-panel");
const statusBanner = document.getElementById("status-banner");
const emptyState = document.getElementById("empty-state");
const tableView = document.getElementById("table-view");
const chartView = document.getElementById("chart-view");
const tableTab = document.getElementById("table-tab");
const chartTab = document.getElementById("chart-tab");
const syncButton = document.getElementById("sync-btn");

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function safeArray(value) {
  return Array.isArray(value) ? value : [];
}

function getDetailsOpenPreference() {
  return window.localStorage.getItem(DETAIL_STORAGE_KEY) === "true";
}

function setDetailsOpenPreference(isOpen) {
  window.localStorage.setItem(DETAIL_STORAGE_KEY, String(Boolean(isOpen)));
}

function getSavedWorkspaceWidth() {
  const saved = Number.parseFloat(window.localStorage.getItem(LAYOUT_STORAGE_KEY) || "");
  return Number.isFinite(saved) ? saved : null;
}

function saveWorkspaceWidth(widthPx) {
  window.localStorage.setItem(LAYOUT_STORAGE_KEY, String(Math.round(widthPx)));
}

function sendMessageToChainlit(message) {
  if (!iframe?.contentWindow) return;
  const outbound = typeof message === "string" ? message : JSON.stringify(message);
  iframe.contentWindow.postMessage(outbound, window.location.origin);
}

function schedulePlotResize() {
  const resize = () => {
    const plotContainer = document.getElementById("plot-container");
    if (!plotContainer || typeof Plotly === "undefined") return;
    try {
      Plotly.Plots.resize(plotContainer);
    } catch {
      // Ignore transient resize errors while the DOM is still settling.
    }
  };

  window.requestAnimationFrame(resize);
  window.setTimeout(resize, 80);
}

function clampWorkspaceWidth(widthPx) {
  const shellWidth = shell.getBoundingClientRect().width;
  const dividerWidth = divider?.getBoundingClientRect().width || 0;
  const maxWorkspaceWidth = Math.max(MIN_WORKSPACE_WIDTH, shellWidth - MIN_CHAT_WIDTH - dividerWidth);
  return Math.min(Math.max(widthPx, MIN_WORKSPACE_WIDTH), maxWorkspaceWidth);
}

function applyWorkspaceWidth(widthPx, persist = true) {
  if (window.innerWidth <= SPLIT_BREAKPOINT) {
    workspacePane.style.flexBasis = "auto";
    return;
  }

  const nextWidth = clampWorkspaceWidth(widthPx);
  workspacePane.style.flexBasis = `${nextWidth}px`;
  if (persist) {
    saveWorkspaceWidth(nextWidth);
  }
  schedulePlotResize();
}

function initializeResizableShell() {
  if (!divider) return;

  const applySavedOrDefaultWidth = (persist = false) => {
    const savedWidth = getSavedWorkspaceWidth();
    if (savedWidth) {
      applyWorkspaceWidth(savedWidth, persist);
      return;
    }
    const shellWidth = shell.getBoundingClientRect().width;
    applyWorkspaceWidth(shellWidth * DEFAULT_WORKSPACE_PERCENT, persist);
  };

  const onPointerMove = (event) => {
    if (window.innerWidth <= SPLIT_BREAKPOINT) return;
    const shellRect = shell.getBoundingClientRect();
    const nextWidth = event.clientX - shellRect.left;
    applyWorkspaceWidth(nextWidth);
  };

  const stopResize = () => {
    document.body.classList.remove("is-resizing");
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", stopResize);
  };

  divider.addEventListener("pointerdown", (event) => {
    if (window.innerWidth <= SPLIT_BREAKPOINT) return;
    event.preventDefault();
    document.body.classList.add("is-resizing");
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", stopResize, { once: true });
  });

  divider.addEventListener("keydown", (event) => {
    if (window.innerWidth <= SPLIT_BREAKPOINT) return;
    const currentWidth = workspacePane.getBoundingClientRect().width;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      applyWorkspaceWidth(currentWidth - 40);
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      applyWorkspaceWidth(currentWidth + 40);
    }
    if (event.key === "Home") {
      event.preventDefault();
      applyWorkspaceWidth(shell.getBoundingClientRect().width * 0.5);
    }
    if (event.key === "End") {
      event.preventDefault();
      applyWorkspaceWidth(shell.getBoundingClientRect().width * 0.7);
    }
  });

  window.addEventListener("resize", () => {
    if (window.innerWidth <= SPLIT_BREAKPOINT) {
      workspacePane.style.flexBasis = "auto";
      schedulePlotResize();
      return;
    }
    applySavedOrDefaultWidth(false);
  });

  applySavedOrDefaultWidth(false);
}

function renderFileRegistry(fileRegistry) {
  if (!fileManager) return;

  const files = safeArray(fileRegistry?.files);
  const count = Number.parseInt(fileRegistry?.count ?? files.length, 10) || files.length;
  const limit = Number.parseInt(fileRegistry?.limit ?? 5, 10) || 5;
  const canAdd = Boolean(fileRegistry?.can_add ?? count < limit);

  fileManager.innerHTML = `
    <div class="file-manager-header">
      <div>
        <p class="mini-eyebrow">Registered CSV files</p>
        <p class="file-manager-copy">
          ${count ? "Select a file to preview it in the table workspace, or mention its @alias in chat to target it directly." : "Upload one or more CSV files to start. Each file gets an @alias you can use in chat."}
        </p>
      </div>
      <div class="file-manager-actions">
        <span class="file-capacity">${escapeHtml(count)} / ${escapeHtml(limit)} files</span>
        <button
          class="ghost-button compact-button"
          type="button"
          data-file-action="request-upload"
          ${canAdd ? "" : "disabled"}
        >
          Add CSV
        </button>
      </div>
    </div>
    <div class="file-strip ${files.length ? "" : "is-empty"}">
      ${files.length
        ? files
            .map(
              (file) => `
                <article class="file-card ${file.is_active ? "active" : ""}">
                  <button
                    class="file-select-button"
                    type="button"
                    data-file-action="activate-file"
                    data-dataset-key="${escapeHtml(file.dataset_key)}"
                  >
                    <div class="file-card-title-row">
                      <span class="file-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>
                      <div class="file-badges">
                        ${file.is_active ? '<span class="file-badge active-badge">Active</span>' : ""}
                        ${file.has_chart ? '<span class="file-badge">Chart</span>' : ""}
                      </div>
                    </div>
                    <div class="file-card-meta">${escapeHtml(file.rows ?? 0)} rows · ${escapeHtml(file.columns ?? 0)} columns</div>
                    <div class="file-card-footer">
                      <span class="mention-pill">@${escapeHtml(file.mention)}</span>
                      ${file.transformations ? `<span class="file-meta-pill">${escapeHtml(file.transformations)} transform${file.transformations === 1 ? "" : "s"}</span>` : ""}
                    </div>
                  </button>
                  <button
                    class="file-delete-button"
                    type="button"
                    data-file-action="delete-file"
                    data-dataset-key="${escapeHtml(file.dataset_key)}"
                    data-dataset-name="${escapeHtml(file.name)}"
                    aria-label="Delete ${escapeHtml(file.name)}"
                    title="Delete ${escapeHtml(file.name)}"
                  >
                    Delete
                  </button>
                </article>
              `,
            )
            .join("")
        : `
          <div class="file-empty-state">
            <p>No CSV files are registered yet.</p>
            <p>Use <strong>Add CSV</strong> or upload through the chat prompt.</p>
          </div>
        `}
    </div>
  `;
}

function renderSummary(summary, datasetName) {
  const existingDetails = document.querySelector(".details-panel");
  if (existingDetails) existingDetails.remove();

  if (!summary) {
    summaryPanel.innerHTML = '<div class="empty-summary">Upload a CSV in the chat to populate this workspace.</div>';
    return;
  }

  const cards = [
    ["Active file", datasetName || "Unnamed CSV"],
    ["Rows", summary.rows ?? "–"],
    ["Columns", summary.columns ?? "–"],
    ["Missing cells", summary.missing_cells ?? "–"],
    ["Duplicates removed", summary.duplicate_rows_removed ?? "–"],
  ];

  summaryPanel.innerHTML = cards
    .map(
      ([label, value]) => `
        <article class="summary-card">
          <span class="label">${escapeHtml(label)}</span>
          <span class="value">${escapeHtml(value)}</span>
        </article>
      `,
    )
    .join("");

  const numericColumns = safeArray(summary.numeric_columns);
  const datetimeColumns = safeArray(summary.datetime_columns);
  const categoricalColumns = safeArray(summary.categorical_columns);
  const transformations = safeArray(summary.transformations);

  const detailsPanel = document.createElement("details");
  detailsPanel.className = "details-panel";
  detailsPanel.open = getDetailsOpenPreference();

  detailsPanel.innerHTML = `
    <summary>
      <div>
        <div>Dataset details</div>
        <div class="details-summary-copy">Expand to inspect column types and preprocessing notes without shrinking the table.</div>
      </div>
    </summary>
    <div class="details-grid">
      <article class="detail-card">
        <h3>Column types</h3>
        <div class="type-group">
          <h4>Numeric (${escapeHtml(numericColumns.length)})</h4>
          <div class="tag-list tag-scroller">
            ${numericColumns.map((column) => `<span class="tag">${escapeHtml(column)}</span>`).join("") || '<span class="tag">None</span>'}
          </div>
        </div>
        <div class="type-group">
          <h4>Datetime (${escapeHtml(datetimeColumns.length)})</h4>
          <div class="tag-list tag-scroller">
            ${datetimeColumns.map((column) => `<span class="tag">${escapeHtml(column)}</span>`).join("") || '<span class="tag">None</span>'}
          </div>
        </div>
        <div class="type-group">
          <h4>Categorical (${escapeHtml(categoricalColumns.length)})</h4>
          <div class="tag-list tag-scroller">
            ${categoricalColumns.map((column) => `<span class="tag">${escapeHtml(column)}</span>`).join("") || '<span class="tag">None</span>'}
          </div>
        </div>
      </article>
      <article class="detail-card">
        <h3>Column renames</h3>
        <ul>
          ${Object.entries(summary.column_renames || {})
            .slice(0, 12)
            .map(([source, target]) => `<li><code>${escapeHtml(source)}</code> → <code>${escapeHtml(target)}</code></li>`)
            .join("") || "<li>No column renames were needed.</li>"}
        </ul>
      </article>
      <article class="detail-card">
        <h3>Auto-converted columns</h3>
        <ul>
          ${Object.entries(summary.converted_columns || {})
            .slice(0, 12)
            .map(([column, kind]) => `<li><code>${escapeHtml(column)}</code> parsed as <strong>${escapeHtml(kind)}</strong></li>`)
            .join("") || "<li>No additional type conversions were applied.</li>"}
        </ul>
      </article>
      <article class="detail-card">
        <h3>User-requested transformations</h3>
        <ul>
          ${transformations
            .slice(-8)
            .reverse()
            .map((item) => `<li>${escapeHtml(item)}</li>`)
            .join("") || "<li>No additional dataframe transformations have been applied yet.</li>"}
        </ul>
      </article>
    </div>
  `;

  detailsPanel.addEventListener("toggle", () => {
    setDetailsOpenPreference(detailsPanel.open);
  });

  summaryPanel.insertAdjacentElement("afterend", detailsPanel);
}

function renderTable(tablePayload) {
  if (!tablePayload) {
    tableView.innerHTML = "";
    return;
  }

  const columns = safeArray(tablePayload.columns);
  const rows = safeArray(tablePayload.rows);
  const rowNumbers = safeArray(tablePayload.row_numbers);
  const highlightedColumns = new Set(safeArray(tablePayload.highlight_columns));
  const tableClasses = ["data-table"];
  if (state.tableOptions.wrapCells) tableClasses.push("table-wrap");
  if (state.tableOptions.compactRows) tableClasses.push("table-compact");

  const totalRows = Number.parseInt(tablePayload.total_rows ?? rows.length, 10) || rows.length;
  const totalColumns = Number.parseInt(tablePayload.total_columns ?? columns.length, 10) || columns.length;
  const currentPage = Number.parseInt(tablePayload.page ?? 1, 10) || 1;
  const totalPages = Math.max(1, Number.parseInt(tablePayload.total_pages ?? 1, 10) || 1);
  const pageSize = Number.parseInt(tablePayload.page_size ?? rows.length ?? 25, 10) || 25;
  const rowStart = Number.parseInt(tablePayload.page_row_start ?? (rows.length ? 1 : 0), 10) || 0;
  const rowEnd = Number.parseInt(tablePayload.page_row_end ?? rowStart + rows.length - 1, 10) || 0;
  const pageSizeOptions = Array.from(new Set([25, 50, 100, pageSize].filter((value) => Number.isFinite(value) && value > 0))).sort((a, b) => a - b);
  const datasetName = state.ui?.dataset_name || "Active CSV";
  const activeFile = safeArray(state.ui?.file_registry?.files).find((file) => file.is_active);

  const header = ['<th class="row-index-head">#</th>']
    .concat(
      columns.map((column) => {
        const highlighted = highlightedColumns.has(column) ? " is-highlighted" : "";
        return `<th class="${highlighted.trim()}" title="${escapeHtml(column)}">${escapeHtml(column)}</th>`;
      }),
    )
    .join("");

  const body = rows
    .map((row, rowIndex) => {
      const cells = columns
        .map((column) => {
          const raw = row[column];
          const isEmpty = raw === null || raw === undefined || raw === "";
          const value = isEmpty ? '<span class="cell-empty">—</span>' : escapeHtml(raw);
          const title = isEmpty ? "" : ` title="${escapeHtml(raw)}"`;
          const highlighted = highlightedColumns.has(column) ? ' class="is-highlighted"' : "";
          return `<td${highlighted}${title}>${value}</td>`;
        })
        .join("");
      const displayRowNumber = rowNumbers[rowIndex] ?? (rowStart + rowIndex);
      return `<tr><th scope="row" class="row-index">${displayRowNumber}</th>${cells}</tr>`;
    })
    .join("");

  const rowWindowLabel = rowStart && rowEnd ? `Showing rows ${rowStart}-${rowEnd} of ${totalRows}` : `Showing ${rows.length} cleaned rows`;

  tableView.innerHTML = `
    <div class="table-shell">
      <div class="table-meta">
        <span>File: ${escapeHtml(datasetName)}</span>
        ${activeFile?.mention ? `<span>Mention: @${escapeHtml(activeFile.mention)}</span>` : ""}
        <span>${escapeHtml(rowWindowLabel)}</span>
        <span>Page ${escapeHtml(currentPage)} of ${escapeHtml(totalPages)}</span>
        <span>${escapeHtml(totalColumns)} columns</span>
        ${highlightedColumns.size ? `<span class="table-focus-pill">Focused column${highlightedColumns.size > 1 ? "s" : ""}: ${escapeHtml(Array.from(highlightedColumns).join(", "))}</span>` : ""}
      </div>
      <div class="table-tools">
        <div class="table-actions">
          <button
            class="table-option ${state.tableOptions.wrapCells ? "active" : ""}"
            type="button"
            data-table-action="toggle-wrap"
          >
            Wrap cells
          </button>
          <button
            class="table-option ${state.tableOptions.compactRows ? "active" : ""}"
            type="button"
            data-table-action="toggle-compact"
          >
            Compact rows
          </button>
        </div>
        <div class="table-pagination">
          <label class="rows-per-page-control">
            <span>Rows per page</span>
            <select class="page-size-select" data-table-action="set-page-size" aria-label="Rows per page">
              ${pageSizeOptions
                .map((option) => `<option value="${option}" ${option === pageSize ? "selected" : ""}>${option}</option>`)
                .join("")}
            </select>
          </label>
          <div class="pagination-controls" aria-label="Table pagination controls">
            <button class="table-option pagination-button" type="button" data-table-action="goto-first" ${currentPage <= 1 ? "disabled" : ""}>« First</button>
            <button class="table-option pagination-button" type="button" data-table-action="goto-prev" ${currentPage <= 1 ? "disabled" : ""}>‹ Prev</button>
            <span class="pagination-summary">${escapeHtml(currentPage)} / ${escapeHtml(totalPages)}</span>
            <button class="table-option pagination-button" type="button" data-table-action="goto-next" ${currentPage >= totalPages ? "disabled" : ""}>Next ›</button>
            <button class="table-option pagination-button" type="button" data-table-action="goto-last" ${currentPage >= totalPages ? "disabled" : ""}>Last »</button>
          </div>
        </div>
        <span class="table-hint">Use the file strip above to switch CSVs, or mention a file with @alias in chat.</span>
      </div>
      <div class="table-scroll">
        <table class="${tableClasses.join(" ")}">
          <thead>
            <tr>${header}</tr>
          </thead>
          <tbody>${body}</tbody>
        </table>
      </div>
    </div>
  `;
}

function renderChartError(message) {
  const controlsHtml = renderChartControls(state.ui?.chart_controls);
  chartView.innerHTML = `${controlsHtml}
    <div class="chart-error">
      <h3>Chart could not be displayed</h3>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
}

function renderChartControls(chartControls) {
  if (!chartControls?.enabled) return "";

  const numericColumns = safeArray(chartControls.available_numeric_columns);
  const allColumns = safeArray(chartControls.available_columns);
  const yValue = chartControls.y || "";
  const colorValue = chartControls.color || "";

  const yOptions = ['<option value="">(none)</option>']
    .concat(
      numericColumns.map((column) => `<option value="${escapeHtml(column)}" ${column === yValue ? "selected" : ""}>${escapeHtml(column)}</option>`),
    )
    .join("");

  const colorOptions = ['<option value="">(none)</option>']
    .concat(
      allColumns.map((column) => `<option value="${escapeHtml(column)}" ${column === colorValue ? "selected" : ""}>${escapeHtml(column)}</option>`),
    )
    .join("");

  return `
    <section class="chart-controls" aria-label="Chart controls">
      <div class="chart-control-grid">
        <label class="chart-control-field readonly-field">
          <span>Chart type (read-only)</span>
          <input type="text" value="${escapeHtml(chartControls.chart_type || "")}" readonly disabled />
        </label>
        <label class="chart-control-field">
          <span>Y</span>
          <select data-chart-control="y" aria-label="Y column">${yOptions}</select>
        </label>
        <label class="chart-control-field">
          <span>Color</span>
          <select data-chart-control="color" aria-label="Color column">${colorOptions}</select>
        </label>
        <label class="chart-control-field readonly-field">
          <span>Aggregation (read-only)</span>
          <input type="text" value="${escapeHtml(chartControls.aggregation || "(unchanged)")}" readonly disabled />
        </label>
        <label class="chart-control-field readonly-field">
          <span>Top N (read-only)</span>
          <input type="text" value="${escapeHtml(chartControls.top_n ?? "(unchanged)")}" readonly disabled />
        </label>
      </div>
      <div class="chart-control-actions">
        <button class="table-option" type="button" data-chart-action="apply">Apply</button>
        <button class="table-option" type="button" data-chart-action="reset">Reset</button>
      </div>
    </section>
  `;
}

function renderChart(chartPayload) {
  if (!chartPayload || !chartPayload.figure) {
    chartView.innerHTML = "";
    return;
  }

  if (typeof Plotly === "undefined") {
    renderChartError("The Plotly JavaScript bundle did not load in the browser. Refresh the page once after restart; if the problem persists, verify that /static/vendor/plotly.min.js is being served.");
    return;
  }

  const figure = chartPayload.figure && typeof chartPayload.figure === "object" ? chartPayload.figure : {};
  const data = Array.isArray(figure.data) ? figure.data : [];
  const layout = figure.layout && typeof figure.layout === "object" ? { ...figure.layout, autosize: true } : { autosize: true };
  const config = {
    responsive: true,
    displaylogo: false,
    ...(chartPayload.config && typeof chartPayload.config === "object" ? chartPayload.config : {}),
  };

  chartView.innerHTML = renderChartControls(state.ui?.chart_controls);
  const plotContainer = document.createElement("div");
  plotContainer.id = "plot-container";
  plotContainer.className = "plot-container";
  chartView.appendChild(plotContainer);

  Plotly.newPlot(plotContainer, data, layout, config)
    .then(() => {
      schedulePlotResize();
    })
    .catch((error) => {
      const detail = error instanceof Error ? error.message : String(error);
      renderChartError(`Plotly failed to render the figure: ${detail}`);
    });
}

function showActiveView(activeView) {
  const hasTable = Boolean(state.ui?.table);
  const hasChart = Boolean(state.ui?.chart);
  const view = activeView === "chart" && hasChart ? "chart" : hasTable ? "table" : null;

  tableTab.disabled = !hasTable;
  chartTab.disabled = !hasChart;
  tableTab.classList.toggle("active", view === "table");
  chartTab.classList.toggle("active", view === "chart");
  workspacePane.classList.toggle("is-chart-view", view === "chart");

  if (!view) {
    emptyState.classList.remove("hidden");
    tableView.classList.add("hidden");
    chartView.classList.add("hidden");
    return;
  }

  emptyState.classList.add("hidden");
  tableView.classList.toggle("hidden", view !== "table");
  chartView.classList.toggle("hidden", view !== "chart");
  if (view === "chart") {
    schedulePlotResize();
  }
}

function render() {
  const ui = state.ui;
  renderFileRegistry(ui?.file_registry);
  renderSummary(ui?.summary, ui?.dataset_name);
  statusBanner.textContent = ui?.status || "Ready";
  showActiveView(ui?.active_view);
  renderTable(ui?.table);
  renderChart(ui?.chart);
}

function requestSync() {
  sendMessageToChainlit("SYNC_VIEW");
}

function notifyActiveViewChanged(activeView) {
  sendMessageToChainlit({
    type: "ACTIVE_VIEW_CHANGED",
    payload: { active_view: activeView },
  });
}

function requestTablePage(page) {
  sendMessageToChainlit({
    type: "TABLE_PAGE_CHANGED",
    payload: { page },
  });
}

function requestTablePageSize(pageSize) {
  sendMessageToChainlit({
    type: "TABLE_PAGE_SIZE_CHANGED",
    payload: { page_size: pageSize },
  });
}

function requestChartControlsApply(payload) {
  sendMessageToChainlit({
    type: "CHART_CONTROLS_APPLIED",
    payload,
  });
}

function requestChartControlsReset() {
  sendMessageToChainlit({
    type: "CHART_CONTROLS_RESET",
    payload: {},
  });
}

function requestFileUpload() {
  sendMessageToChainlit({ type: "REQUEST_UPLOAD_FILES", payload: {} });
}

function requestDatasetActivation(datasetKey) {
  sendMessageToChainlit({
    type: "ACTIVE_DATASET_CHANGED",
    payload: { dataset_key: datasetKey },
  });
}

function requestDatasetDeletion(datasetKey) {
  sendMessageToChainlit({
    type: "DELETE_DATASET",
    payload: { dataset_key: datasetKey },
  });
}

window.addEventListener("message", (event) => {
  if (event.origin !== window.location.origin) return;

  let payload = event.data;
  if (typeof payload === "string") {
    try {
      payload = JSON.parse(payload);
    } catch {
      return;
    }
  }

  if (!payload || payload.source !== APP_SOURCE || payload.type !== "ui_state") return;
  state.ui = payload.payload || null;
  render();
});

tableTab.addEventListener("click", () => {
  if (!state.ui?.table) return;
  state.ui.active_view = "table";
  render();
  notifyActiveViewChanged("table");
});

chartTab.addEventListener("click", () => {
  if (!state.ui?.chart) return;
  state.ui.active_view = "chart";
  render();
  notifyActiveViewChanged("chart");
});

fileManager.addEventListener("click", (event) => {
  const actionElement = event.target?.closest?.("[data-file-action]");
  const action = actionElement?.dataset?.fileAction;
  if (!action) return;

  if (action === "request-upload") {
    requestFileUpload();
    return;
  }

  const datasetKey = actionElement.dataset.datasetKey;
  if (!datasetKey) return;

  if (action === "activate-file") {
    requestDatasetActivation(datasetKey);
    return;
  }

  if (action === "delete-file") {
    const datasetName = actionElement.dataset.datasetName || "this CSV file";
    const confirmed = window.confirm(`Delete ${datasetName} from this workspace?`);
    if (!confirmed) return;
    requestDatasetDeletion(datasetKey);
  }
});

tableView.addEventListener("click", (event) => {
  const actionElement = event.target?.closest?.("[data-table-action]");
  const action = actionElement?.dataset?.tableAction;
  if (!action) return;

  if (action === "toggle-wrap") {
    state.tableOptions.wrapCells = !state.tableOptions.wrapCells;
    renderTable(state.ui?.table);
    return;
  }
  if (action === "toggle-compact") {
    state.tableOptions.compactRows = !state.tableOptions.compactRows;
    renderTable(state.ui?.table);
    return;
  }

  const tablePayload = state.ui?.table;
  if (!tablePayload) return;

  const currentPage = Number.parseInt(tablePayload.page ?? 1, 10) || 1;
  const totalPages = Math.max(1, Number.parseInt(tablePayload.total_pages ?? 1, 10) || 1);

  if (action === "goto-first" && currentPage > 1) {
    requestTablePage(1);
    return;
  }
  if (action === "goto-prev" && currentPage > 1) {
    requestTablePage(currentPage - 1);
    return;
  }
  if (action === "goto-next" && currentPage < totalPages) {
    requestTablePage(currentPage + 1);
    return;
  }
  if (action === "goto-last" && currentPage < totalPages) {
    requestTablePage(totalPages);
  }
});

tableView.addEventListener("change", (event) => {
  const action = event.target?.dataset?.tableAction;
  if (action !== "set-page-size") return;

  const pageSize = Number.parseInt(event.target.value, 10);
  if (!Number.isFinite(pageSize) || pageSize <= 0) return;
  requestTablePageSize(pageSize);
});

chartView.addEventListener("click", (event) => {
  const action = event.target?.closest?.("[data-chart-action]")?.dataset?.chartAction;
  if (!action) return;
  if (action === "reset") {
    requestChartControlsReset();
    return;
  }
  if (action !== "apply") return;

  const y = chartView.querySelector('[data-chart-control="y"]')?.value ?? "";
  const color = chartView.querySelector('[data-chart-control="color"]')?.value ?? "";
  requestChartControlsApply({ y, color });
});

syncButton.addEventListener("click", requestSync);
iframe.addEventListener("load", () => window.setTimeout(requestSync, 1200));

if (typeof ResizeObserver !== "undefined") {
  const resizeObserver = new ResizeObserver(() => schedulePlotResize());
  resizeObserver.observe(chartView);
  resizeObserver.observe(workspacePane);
}

initializeResizableShell();
render();
