/* global chrome */
"use strict";

const DEFAULT_API_URL = "http://127.0.0.1:8765/api";

const apiUrlEl = document.getElementById("apiUrl");
const topKEl   = document.getElementById("topK");
const modeEl   = document.getElementById("mode");
const saveBtn  = document.getElementById("save");
const statusEl = document.getElementById("status");

// Load saved settings
chrome.storage.sync.get(
  { apiUrl: DEFAULT_API_URL, topK: 10, mode: "hybrid" },
  (items) => {
    apiUrlEl.value = items.apiUrl;
    topKEl.value   = items.topK;
    modeEl.value   = items.mode;
  }
);

// Save settings
saveBtn.addEventListener("click", () => {
  chrome.storage.sync.set(
    {
      apiUrl: apiUrlEl.value.trim() || DEFAULT_API_URL,
      topK:   Math.max(1, Math.min(50, parseInt(topKEl.value, 10) || 10)),
      mode:   modeEl.value,
    },
    () => {
      statusEl.textContent = "Settings saved!";
      setTimeout(() => { statusEl.textContent = ""; }, 2000);
    }
  );
});
