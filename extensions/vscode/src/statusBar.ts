/**
 * Status bar item showing MemoryMesh connectivity state.
 *
 * Displays "🧠 MemoryMesh" when the API is reachable, or an error icon when
 * it cannot be reached.  Clicking it runs the search command.
 */

import * as vscode from "vscode";
import { MemoryMeshClient } from "./memorymeshClient";

export class StatusBarManager {
  private readonly _item: vscode.StatusBarItem;
  private readonly _client: MemoryMeshClient;
  private _timer: NodeJS.Timeout | undefined;

  constructor(client: MemoryMeshClient) {
    this._client = client;
    this._item = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
      100
    );
    this._item.command = "memorymesh.search";
    this._item.tooltip = "MemoryMesh — click to search";
    this._item.show();
  }

  /** Start polling the API every 30 s and update the status bar. */
  startPolling(): void {
    void this._refresh();
    this._timer = setInterval(() => void this._refresh(), 30_000);
  }

  dispose(): void {
    clearInterval(this._timer);
    this._item.dispose();
  }

  // ── Private ────────────────────────────────────────────────────────────────

  private async _refresh(): Promise<void> {
    try {
      const h = await this._client.health();
      this._item.text = `🧠 MemoryMesh v${h.version}`;
      this._item.backgroundColor = undefined;
    } catch {
      this._item.text = "$(error) MemoryMesh offline";
      this._item.backgroundColor = new vscode.ThemeColor(
        "statusBarItem.warningBackground"
      );
    }
  }
}
