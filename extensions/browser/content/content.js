/**
 * MemoryMesh content script.
 *
 * Listens for messages from the popup or background service worker and can
 * insert search results into the active page if requested.  Currently only
 * used for page-context text selection forwarding.
 */

"use strict";

// Forward selected text to the popup when the user triggers the extension
// via the context menu (future feature hook).
document.addEventListener("selectionchange", () => {
  // No-op: selection is read in the popup via the messaging API.
});

// Listen for injected search results (optional future feature).
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "INSERT_SNIPPET" && msg.text) {
    // Optional: highlight text on page, future feature.
    console.debug("[MemoryMesh] content script received INSERT_SNIPPET");
  }
});
