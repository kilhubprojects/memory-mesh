/**
 * MemoryMesh VS Code extension entry point.
 *
 * Registers commands and activates the status bar.  All heavy lifting is
 * delegated to {@link MemoryMeshClient}, {@link SearchPanel}, and
 * {@link StatusBarManager}.
 */

import * as vscode from "vscode";
import { MemoryMeshClient } from "./memorymeshClient";
import { SearchPanel } from "./searchPanel";
import { StatusBarManager } from "./statusBar";

let _statusBar: StatusBarManager | undefined;

export function activate(context: vscode.ExtensionContext): void {
  const client = new MemoryMeshClient();

  // Status bar
  _statusBar = new StatusBarManager(client);
  _statusBar.startPolling();
  context.subscriptions.push({ dispose: () => _statusBar?.dispose() });

  // Command: search
  context.subscriptions.push(
    vscode.commands.registerCommand("memorymesh.search", () => {
      SearchPanel.show(client);
    })
  );

  // Command: index now
  context.subscriptions.push(
    vscode.commands.registerCommand("memorymesh.indexNow", async () => {
      try {
        await client.indexNow();
        void vscode.window.showInformationMessage("MemoryMesh: indexing started.");
      } catch (err) {
        void vscode.window.showErrorMessage(`MemoryMesh: indexing failed — ${String(err)}`);
      }
    })
  );

  // Command: show status
  context.subscriptions.push(
    vscode.commands.registerCommand("memorymesh.showStatus", async () => {
      try {
        const h = await client.health();
        const srcs = await client.sources();
        const sourceCount = Object.keys(srcs.sources).length;
        void vscode.window.showInformationMessage(
          `MemoryMesh v${h.version} — ${sourceCount} source(s) configured`
        );
      } catch (err) {
        void vscode.window.showErrorMessage(
          `MemoryMesh offline or unreachable: ${String(err)}`
        );
      }
    })
  );
}

export function deactivate(): void {
  _statusBar?.dispose();
}
