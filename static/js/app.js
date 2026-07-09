const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const dzLabel = document.getElementById("dzLabel");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");
const resultsEl = document.getElementById("results");

let selectedFile = null;

function setStep(active) {
  document.querySelectorAll(".step").forEach((el) => {
    el.classList.toggle("active", el.dataset.step === active);
  });
}

dropzone.addEventListener("click", () => fileInput.click());

["dragover", "dragenter"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  })
);

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelected(file);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFileSelected(fileInput.files[0]);
});

function handleFileSelected(file) {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    uploadStatus.textContent = "Please choose a .csv file.";
    uploadStatus.className = "status error";
    return;
  }
  selectedFile = file;
  dzLabel.textContent = `Selected: ${file.name}`;
  uploadBtn.disabled = false;
  uploadStatus.textContent = "";
}

uploadBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  uploadBtn.disabled = true;
  uploadStatus.textContent = "Running EDA, ML, and Insight agents…";
  uploadStatus.className = "status";
  setStep("eda");
  resultsEl.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const res = await fetch("/api/upload", { method: "POST", body: formData });
    const data = await res.json();

    if (!res.ok) {
      uploadStatus.textContent = data.error || "Upload failed.";
      uploadStatus.className = "status error";
      uploadBtn.disabled = false;
      return;
    }

    renderResults(data);
    uploadStatus.textContent = "Done.";
    uploadStatus.className = "status ok";
  } catch (err) {
    uploadStatus.textContent = `Request failed: ${err}`;
    uploadStatus.className = "status error";
  } finally {
    uploadBtn.disabled = false;
  }
});

function renderResults(data) {
  resultsEl.classList.remove("hidden");
  setStep("insight");

  // Preview table
  const previewEl = document.getElementById("previewTable");
  previewEl.innerHTML = buildTable(data.columns, data.preview);

  // EDA
  document.getElementById("edaSummary").textContent = data.eda_summary;
  renderCorrelation(data.correlation);

  // ML
  const mlMeta = document.getElementById("mlMeta");
  mlMeta.innerHTML = "";
  if (data.problem_type) {
    mlMeta.appendChild(pill(`Problem: ${data.problem_type}`));
  }
  Object.entries(data.metrics || {}).forEach(([k, v]) => {
    mlMeta.appendChild(pill(`${k}: ${Number(v).toFixed(4)}`));
  });
  document.getElementById("mlSummary").textContent = data.ml_summary;
  renderFeatureImportance(data.feature_importance);

  const downloadLink = document.getElementById("downloadModelLink");
  downloadLink.classList.toggle("hidden", !data.model_available);

  // Insight
  document.getElementById("insightSummary").textContent = data.insight_summary;

  // Chat
  document.getElementById("chatLog").innerHTML = "";
  document.getElementById("chatInput").disabled = !data.chat_available;
  if (!data.chat_available) {
    appendChat("assistant", "Chat is unavailable (no GOOGLE_API_KEY configured, or the knowledge base failed to build).");
  }

  setStep("chat");
}

function pill(text) {
  const span = document.createElement("span");
  span.className = "pill";
  span.textContent = text;
  return span;
}

function buildTable(columns, rows) {
  if (!rows || rows.length === 0) return "<p>No rows to preview.</p>";
  let html = "<table class='preview'><thead><tr>";
  columns.forEach((c) => (html += `<th>${escapeHtml(c)}</th>`));
  html += "</tr></thead><tbody>";
  rows.forEach((row) => {
    html += "<tr>";
    columns.forEach((c) => {
      const v = row[c];
      html += `<td>${v === null || v === undefined ? "" : escapeHtml(String(v))}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  return html;
}

function escapeHtml(str) {
  return str
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function renderCorrelation(correlation) {
  const el = document.getElementById("corrChart");
  if (!correlation) {
    el.innerHTML = "";
    return;
  }
  const trace = {
    z: correlation.matrix,
    x: correlation.labels,
    y: correlation.labels,
    type: "heatmap",
    colorscale: [
      [0, "#161923"],
      [0.5, "#2f7f70"],
      [1, "#5ee6c8"],
    ],
    showscale: true,
  };
  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: "#9aa1b2", family: "IBM Plex Mono, monospace", size: 11 },
    margin: { t: 20, l: 90, r: 20, b: 90 },
    height: 420,
  };
  Plotly.newPlot(el, [trace], layout, { responsive: true, displayModeBar: false });
}

function renderFeatureImportance(fi) {
  const el = document.getElementById("fiChart");
  const entries = Object.entries(fi || {}).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) {
    el.innerHTML = "";
    return;
  }
  const trace = {
    x: entries.map((e) => e[1]),
    y: entries.map((e) => e[0]),
    type: "bar",
    orientation: "h",
    marker: { color: "#5ee6c8" },
  };
  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { color: "#9aa1b2", family: "IBM Plex Mono, monospace", size: 11 },
    margin: { t: 20, l: 140, r: 20, b: 40 },
    height: Math.max(220, entries.length * 28),
    title: { text: "Feature importance", font: { size: 12, color: "#5ee6c8" } },
  };
  Plotly.newPlot(el, [trace], layout, { responsive: true, displayModeBar: false });
}

// Chat

const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatLog = document.getElementById("chatLog");

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;

  appendChat("user", question);
  chatInput.value = "";
  const thinkingEl = appendChat("assistant", "Thinking…");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    thinkingEl.textContent = res.ok ? data.answer : (data.error || "Something went wrong.");
  } catch (err) {
    thinkingEl.textContent = `Request failed: ${err}`;
  }
});

function appendChat(role, text) {
  const div = document.createElement("div");
  div.className = `chat-msg ${role}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}
