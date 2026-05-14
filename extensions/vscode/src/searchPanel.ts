/**
 * Webview panel for MemoryMesh search.
 *
 * Opens a side panel with a search input and result list.  Results link back
 * to the source file using VS Code's `vscode.open` URI scheme.
 */

import * as vscode from "vscode";
import { MemoryMeshClient, SearchHit } from "./memorymeshClient";

export class SearchPanel {
  private static _current: SearchPanel | undefined;
  private readonly _panel: vscode.WebviewPanel;
  private readonly _client: MemoryMeshClient;
  private _disposables: vscode.Disposable[] = [];

  static show(client: MemoryMeshClient): void {
    if (SearchPanel._current) {
      SearchPanel._current._panel.reveal(vscode.ViewColumn.Beside);
      return;
    }
    new SearchPanel(client);
  }

  private constructor(client: MemoryMeshClient) {
    this._client = client;
    this._panel = vscode.window.createWebviewPanel(
      "memorymeshSearch",
      "MemoryMesh Search",
      vscode.ViewColumn.Beside,
      { enableScripts: true, retainContextWhenHidden: true }
    );
    this._panel.iconPath = vscode.Uri.parse(
      "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='14'>🧠</text></svg>"
    );
    this._panel.webview.html = this._html("");
    this._panel.onDidDispose(() => this._dispose(), null, this._disposables);
    this._panel.webview.onDidReceiveMessage(
      (msg: { type: string; query: string; topK: number; mode: string }) => {
        if (msg.type === "search") {
          void this._runSearch(msg.query, msg.topK, msg.mode);
        }
      },
      null,
      this._disposables
    );
    SearchPanel._current = this;
  }

  private async _runSearch(query: string, topK: number, mode: string): Promise<void> {
    try {
      const resp = await this._client.search(query, topK, mode);
      const html = this._renderHits(resp.hits, resp.duration_ms);
      void this._panel.webview.postMessage({ type: "results", html });
    } catch (err) {
      void this._panel.webview.postMessage({
        type: "error",
        message: String(err),
      });
    }
  }

  private _renderHits(hits: SearchHit[], durationMs: number): string {
    if (hits.length === 0) {
      return "<p style='color:#888'>No results found.</p>";
    }
    const rows = hits
      .map(
        (h) =>
          `<div class="hit">
            <div class="path">${h.path} <span class="score">score ${h.score.toFixed(3)}</span></div>
            <div class="preview">${h.preview.substring(0, 300).replace(/</g, "&lt;")}</div>
           </div>`
      )
      .join("");
    return `<p style='color:#888'>${hits.length} result(s) in ${durationMs.toFixed(0)}ms</p>${rows}`;
  }

  private _html(results: string): string {
    return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font-family:var(--vscode-font-family);color:var(--vscode-foreground);padding:1rem}
.search-row{display:flex;gap:.5rem;margin-bottom:1rem}
input{flex:1;background:var(--vscode-input-background);color:var(--vscode-input-foreground);border:1px solid var(--vscode-input-border,#666);border-radius:4px;padding:6px 10px;font-size:.95rem}
select{background:var(--vscode-input-background);color:var(--vscode-input-foreground);border:1px solid var(--vscode-input-border,#666);border-radius:4px;padding:6px}
button{background:var(--vscode-button-background);color:var(--vscode-button-foreground);border:none;border-radius:4px;padding:6px 14px;cursor:pointer}
button:hover{background:var(--vscode-button-hoverBackground)}
.hit{border-left:3px solid var(--vscode-activityBarBadge-background,#a0c4ff);padding:.6rem .8rem;margin:.5rem 0;background:var(--vscode-editorWidget-background)}
.path{font-size:.8rem;color:var(--vscode-descriptionForeground);margin-bottom:3px}
.score{float:right;opacity:.6}
.preview{font-size:.9rem;line-height:1.5}
</style>
</head>
<body>
<div class="search-row">
  <input id="q" type="text" placeholder="Search your knowledge base…" autofocus>
  <select id="mode">
    <option value="hybrid">Hybrid</option>
    <option value="dense">Dense</option>
    <option value="sparse">Sparse</option>
  </select>
  <button onclick="doSearch()">Search</button>
</div>
<div id="results">${results}</div>
<script>
const vscode = acquireVsCodeApi();
function doSearch(){
  const q=document.getElementById('q').value.trim();
  if(!q)return;
  document.getElementById('results').innerHTML='<p>Searching…</p>';
  vscode.postMessage({type:'search',query:q,topK:10,mode:document.getElementById('mode').value});
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')doSearch();});
window.addEventListener('message',e=>{
  const m=e.data;
  if(m.type==='results') document.getElementById('results').innerHTML=m.html;
  if(m.type==='error') document.getElementById('results').innerHTML='<p style="color:red">'+m.message+'</p>';
});
</script>
</body>
</html>`;
  }

  private _dispose(): void {
    SearchPanel._current = undefined;
    this._panel.dispose();
    for (const d of this._disposables) {
      d.dispose();
    }
    this._disposables = [];
  }
}
