/**
 * MemoryMesh background service worker (Manifest V3).
 *
 * Handles extension lifecycle and routes messages from the popup and content
 * scripts to the MemoryMesh REST API.  Caches the API base URL from storage.
 */

const DEFAULT_API_URL = "http://127.0.0.1:8765/api";

/** Retrieve stored API URL or fall back to the default. */
async function getApiUrl() {
  return new Promise((resolve) => {
    chrome.storage.sync.get({ apiUrl: DEFAULT_API_URL }, (items) => {
      resolve(items.apiUrl);
    });
  });
}

/** Forward a search request from popup / content to the REST API. */
async function handleSearch({ query, topK = 10, mode = "hybrid" }) {
  const base = await getApiUrl();
  const params = new URLSearchParams({ q: query, top_k: topK, mode });
  const resp = await fetch(`${base}/search?${params}`);
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  return resp.json();
}

/** Liveness check — called by the popup on open. */
async function handleHealth() {
  const base = await getApiUrl();
  const resp = await fetch(`${base}/health`);
  if (!resp.ok) throw new Error(`API error ${resp.status}`);
  return resp.json();
}

// Message router
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  const dispatch = async () => {
    switch (msg.type) {
      case "SEARCH":
        return handleSearch(msg);
      case "HEALTH":
        return handleHealth();
      default:
        throw new Error(`Unknown message type: ${msg.type}`);
    }
  };

  dispatch()
    .then((result) => sendResponse({ ok: true, data: result }))
    .catch((err) => sendResponse({ ok: false, error: String(err) }));

  return true; // Keep the channel open for the async response.
});
