/**
 * HTTP client for the MemoryMesh REST API.
 *
 * Wraps `fetch` with typed request/response shapes and a configurable
 * base URL pulled from VS Code workspace settings.
 */

import * as vscode from "vscode";

export interface SearchHit {
  path: string;
  chunk_index: number;
  score: number;
  preview: string;
  file_type: string;
  source: string;
}

export interface SearchResponse {
  mode: string;
  duration_ms: number;
  total_hits: number;
  hits: SearchHit[];
}

export interface HealthResponse {
  status: string;
  version: string;
  ts: number;
}

export interface SourcesResponse {
  sources: Record<
    string,
    { path: string; n_indexed: number; n_errors: number; n_chunks: number }
  >;
}

export class MemoryMeshClient {
  private get baseUrl(): string {
    const cfg = vscode.workspace.getConfiguration("memorymesh");
    return (cfg.get<string>("apiUrl") ?? "http://127.0.0.1:8765/api").replace(/\/$/, "");
  }

  async health(): Promise<HealthResponse> {
    return this._get<HealthResponse>("/health");
  }

  async sources(): Promise<SourcesResponse> {
    return this._get<SourcesResponse>("/sources");
  }

  async search(
    query: string,
    topK: number = 10,
    mode: string = "hybrid",
    modality: string = "all"
  ): Promise<SearchResponse> {
    const params = new URLSearchParams({
      q: query,
      top_k: String(topK),
      mode,
      modality,
    });
    return this._get<SearchResponse>(`/search?${params}`);
  }

  async indexNow(): Promise<{ status: string }> {
    return this._post<{ status: string }>("/index", {});
  }

  // ── Private ────────────────────────────────────────────────────────────────

  private async _get<T>(path: string): Promise<T> {
    const resp = await fetch(`${this.baseUrl}${path}`);
    if (!resp.ok) {
      throw new Error(`MemoryMesh API error ${resp.status}: ${await resp.text()}`);
    }
    return resp.json() as Promise<T>;
  }

  private async _post<T>(path: string, body: unknown): Promise<T> {
    const resp = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      throw new Error(`MemoryMesh API error ${resp.status}: ${await resp.text()}`);
    }
    return resp.json() as Promise<T>;
  }
}
