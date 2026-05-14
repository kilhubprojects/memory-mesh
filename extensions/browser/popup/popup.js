/* global chrome */
"use strict";

const queryEl = document.getElementById("query");
const modeEl = document.getElementById("mode");
const resultsEl = document.getElementById("results");
const dotEl = document.getElementById("status-dot");
const searchBtn = document.getElementById("search-btn");
const optionsLink = document.getElementById("options-link");

// ── Liveness check ────────────────────────────────────────────────────────────

chrome.runtime.sendMessage({ type: "HEALTH" }, (resp) => {
  if (resp && resp.ok) {
    dotEl.className = "dot online";
    dotEl.title = `MemoryMesh v${resp.data.version} — online`;
  } else {
    dotEl.className = "dot offline";
    dotEl.title = "MemoryMesh offline";
  }
});

// ── Search ────────────────────────────────────────────────────────────────────

function doSearch() {
  const query = queryEl.value.trim();
  if (!query) return;
  resultsEl.innerHTML = '<p class="msg">Searching…</p>';

  chrome.runtime.sendMessage(
    { type: "SEARCH", query, topK: 10, mode: modeEl.value },
    (resp) => {
      if (!resp || !resp.ok) {
        resultsEl.innerHTML = `<p class="msg" style="color:red">${resp ? resp.error : "No response"}</p>`;
        return;
      }
      const { hits, duration_ms, total_hits } = resp.data;
      if (total_hits === 0) {
        resultsEl.innerHTML = '<p class="msg">No results found.</p>';
        return;
      }
      const rows = hits
        .map(
          (h) => `<div class="hit">
            <div class="hit-path">${esc(h.path)}<span class="hit-score">${h.score.toFixed(3)}</span></div>
            <div class="hit-preview">${esc(h.preview.substring(0, 250))}</div>
          </div>`
        )
        .join("");
      resultsEl.innerHTML = `<p class="msg">${total_hits} result(s) · ${Math.round(duration_ms)}ms</p>${rows}`;
    }
  );
}

function esc(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

searchBtn.addEventListener("click", doSearch);
queryEl.addEventListener("keydown", (e) => { if (e.key === "Enter") doSearch(); });

// ── Options link ──────────────────────────────────────────────────────────────

optionsLink.addEventListener("click", (e) => {
  e.preventDefault();
  chrome.runtime.openOptionsPage();
});
