"""Web dashboard for MemoryMesh (Wave 7).

A lightweight FastAPI + HTMX single-server dashboard running on port 8767
(configurable).  Provides a browser UI for:

* **Sources** - indexed file counts, error counts, chunk totals.
* **Search** - run hybrid search queries and inspect results.
* **Memory tiers** - hot/warm/cold chunk distribution.
* **Entities** - list extracted named entities (when entity extraction is on).
* **Timeline** - recent episodic events.

The dashboard reuses the :class:`~memorymesh.server.app.AppContext` directly
so it has access to all the same data as the MCP tools.

Implementation approach
-----------------------
All HTML is generated server-side and returned as HTMX fragments - no build
step, no JavaScript framework.  Only ``fastapi`` and ``uvicorn`` are required::

    uv add "fastapi[standard]"   # includes uvicorn

The dashboard server runs in a **background thread** (like the health server)
and is started from the CLI ``start`` command when
``server.dashboard.enabled: true``.

Usage
-----
Instantiate and call :meth:`DashboardServer.start`::

    dash = DashboardServer(app_ctx, port=8767)
    dash.start()
    ...
    dash.stop()
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext


_CSS = """
<style>
* { box-sizing: border-box; }
body { font-family: system-ui, sans-serif; margin: 0; background: #f5f5f5; color: #222; }
header { background: #1a1a2e; color: #e0e0e0; padding: 1rem 2rem; display: flex; align-items: center; gap: 1rem; }
header h1 { margin: 0; font-size: 1.4rem; font-weight: 600; }
header span.badge { background: #16213e; border-radius: 6px; padding: 2px 10px; font-size: 0.8rem; }
nav { background: #16213e; padding: 0.4rem 2rem; display: flex; gap: 1.5rem; }
nav a { color: #a0c4ff; text-decoration: none; font-size: 0.9rem; padding: 4px 0; }
nav a:hover { color: #fff; }
main { max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem; }
.card { background: #fff; border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,.08); padding: 1.5rem; margin-bottom: 1.5rem; }
.card h2 { margin: 0 0 1rem; font-size: 1.1rem; color: #333; }
table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
th { text-align: left; border-bottom: 2px solid #e8e8e8; padding: 6px 12px; color: #666; }
td { border-bottom: 1px solid #f0f0f0; padding: 6px 12px; }
tr:last-child td { border-bottom: none; }
.badge-hot  { background: #ff6b6b22; color: #c0392b; border-radius: 4px; padding: 1px 7px; font-size: 0.8rem; }
.badge-warm { background: #f39c1222; color: #d68910; border-radius: 4px; padding: 1px 7px; font-size: 0.8rem; }
.badge-cold { background: #85c1e922; color: #1a5276; border-radius: 4px; padding: 1px 7px; font-size: 0.8rem; }
.search-box { display: flex; gap: 0.7rem; }
.search-box input { flex: 1; padding: 8px 14px; border: 1px solid #ddd; border-radius: 6px; font-size: 0.95rem; }
.search-box button { padding: 8px 20px; background: #1a1a2e; color: #fff; border: none; border-radius: 6px; cursor: pointer; }
.search-box button:hover { background: #16213e; }
.hit { border-left: 3px solid #a0c4ff; padding: 0.8rem 1rem; margin: 0.7rem 0; background: #f9fafc; border-radius: 4px; }
.hit .path { font-size: 0.8rem; color: #666; margin-bottom: 4px; }
.hit .score { float: right; color: #999; font-size: 0.8rem; }
.hit .preview { font-size: 0.9rem; line-height: 1.5; }
#search-results { margin-top: 1.2rem; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 1rem; }
.stat { background: #f0f4ff; border-radius: 8px; padding: 1rem; text-align: center; }
.stat .val { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
.stat .lbl { font-size: 0.8rem; color: #555; margin-top: 4px; }
footer { text-align: center; color: #aaa; font-size: 0.8rem; padding: 2rem; }
</style>
"""

_HTMX_CDN = '<script src="https://unpkg.com/htmx.org@1.9.12" crossorigin="anonymous"></script>'


def _page(title: str, body: str) -> str:
    from memorymesh import __version__

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} - MemoryMesh</title>
{_CSS}
{_HTMX_CDN}
</head>
<body>
<header>
  <h1> MemoryMesh</h1>
  <span class="badge">v{__version__}</span>
</header>
<nav>
  <a href="/">Sources</a>
  <a href="/search">Search</a>
  <a href="/tiers">Memory Tiers</a>
  <a href="/entities">Entities</a>
  <a href="/timeline">Timeline</a>
  <a href="/graph-ui">Graph</a>
  <a href="/health" target="_blank">Health open</a>
</nav>
<main>{body}</main>
<footer>MemoryMesh - all data stays local &nbsp;-&nbsp; <a href="https://github.com/kilhubprojects/memory-mesh" style="color:#bbb">GitHub</a></footer>
</body>
</html>"""


def _make_app(ctx: AppContext) -> object:
    """Create and return a FastAPI application wired to *ctx*.

    Args:
        ctx: The shared :class:`~memorymesh.server.app.AppContext`.

    Returns:
        A ``fastapi.FastAPI`` instance with all routes registered.
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        raise ImportError(
            "The MemoryMesh dashboard requires FastAPI. Install it with: uv add 'fastapi[standard]'"
        ) from exc

    api = FastAPI(title="MemoryMesh Dashboard", docs_url=None, redoc_url=None)

    @api.get("/", response_class=HTMLResponse)
    async def index_page() -> str:
        """Render the dashboard index page with source statistics."""
        stats = _build_source_stats(ctx)
        rows = ""
        for s in stats:
            rows += (
                f"<tr><td><b>{s['name']}</b></td>"
                f"<td>{s['path']}</td>"
                f"<td style='text-align:right'>{s['n_indexed']}</td>"
                f"<td style='text-align:right; color:#c0392b'>{s['n_errors']}</td>"
                f"<td style='text-align:right'>{s['n_chunks']}</td>"
                f"<td style='text-align:right; color:#888'>{s['disk_mb']:.1f} MB</td></tr>"
            )
        table = (
            (
                "<table><thead><tr>"
                "<th>Source</th><th>Path</th><th>Indexed</th>"
                "<th>Errors</th><th>Chunks</th><th>Size</th>"
                f"</tr></thead><tbody>{rows}</tbody></table>"
            )
            if rows
            else "<p style='color:#888'>No sources configured.</p>"
        )

        agg = _aggregate_stats(ctx)
        grid = (
            f"<div class='stat-grid'>"
            f"<div class='stat'><div class='val'>{agg['total_files']}</div><div class='lbl'>Indexed files</div></div>"
            f"<div class='stat'><div class='val'>{agg['total_chunks']}</div><div class='lbl'>Total chunks</div></div>"
            f"<div class='stat'><div class='val'>{agg['total_errors']}</div><div class='lbl'>Parse errors</div></div>"
            f"<div class='stat'><div class='val'>{agg['sources']}</div><div class='lbl'>Sources</div></div>"
            f"</div>"
        )

        body = (
            f"<div class='card'><h2>Index Overview</h2>{grid}</div>"
            f"<div class='card'><h2>Sources</h2>{table}</div>"
        )
        return _page("Sources", body)

    @api.get("/search", response_class=HTMLResponse)
    async def search_page(q: str = "", top_k: int = 10, mode: str = "hybrid") -> str:
        """Render the search page with results for *q*."""
        results_html = ""
        if q.strip():
            results_html = _run_search(ctx, q.strip(), top_k=top_k, mode=mode)

        form = f"""
        <form method="get" action="/search">
          <div class="search-box">
            <input name="q" value="{_esc(q)}" placeholder="Search your knowledge base..." autofocus>
            <select name="mode" style="padding:8px;border:1px solid #ddd;border-radius:6px">
              <option value="hybrid" {"selected" if mode == "hybrid" else ""}>Hybrid</option>
              <option value="dense"  {"selected" if mode == "dense" else ""}>Dense</option>
              <option value="sparse" {"selected" if mode == "sparse" else ""}>Sparse</option>
            </select>
            <button type="submit">Search</button>
          </div>
        </form>
        <div id="search-results">{results_html}</div>
        """
        body = f"<div class='card'><h2>Search</h2>{form}</div>"
        return _page("Search", body)

    @api.get("/tiers", response_class=HTMLResponse)
    async def tiers_page() -> str:
        """Render the memory tier distribution page."""
        from memorymesh.core.models import MemoryTier

        if ctx.tiered_memory is None:
            body = "<div class='card'><p>Memory tiers are not enabled.</p></div>"
            return _page("Memory Tiers", body)

        counts: dict[str, int] = {}
        for tier in MemoryTier:
            try:
                entries = ctx.metadata_store.list_chunks_by_tier(tier, limit=None)
                counts[tier.value] = len(entries)
            except Exception:
                counts[tier.value] = 0

        total = sum(counts.values()) or 1
        grid = "<div class='stat-grid'>"
        for name, count in counts.items():
            pct = 100 * count / total
            grid += (
                f"<div class='stat'>"
                f"<div class='val'><span class='badge-{name}'>{name}</span></div>"
                f"<div class='val' style='font-size:1.5rem'>{count}</div>"
                f"<div class='lbl'>{pct:.1f}%</div>"
                f"</div>"
            )
        grid += "</div>"

        body = f"<div class='card'><h2>Hot / Warm / Cold Chunk Distribution</h2>{grid}</div>"
        return _page("Memory Tiers", body)

    @api.get("/entities", response_class=HTMLResponse)
    async def entities_page() -> str:
        try:
            entities = ctx.metadata_store.list_entities(limit=200)
        except Exception:
            entities = []

        if not entities:
            body = "<div class='card'><p>No entities extracted yet. Enable <code>memory.entity_extraction</code> in config.</p></div>"
            return _page("Entities", body)

        rows = ""
        for e in entities:
            rows += (
                f"<tr><td><b>{_esc(e.name)}</b></td>"
                f"<td><span class='badge-warm'>{_esc(e.entity_type)}</span></td>"
                f"<td style='text-align:right'>{e.mention_count}</td></tr>"
            )

        table = (
            "<table><thead><tr><th>Entity</th><th>Type</th><th>Mentions</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
        body = f"<div class='card'><h2>Extracted Entities ({len(entities)})</h2>{table}</div>"
        return _page("Entities", body)

    @api.get("/timeline", response_class=HTMLResponse)
    async def timeline_page(days: float = 7.0) -> str:
        """Render the episodic timeline page for the last *days* days."""
        if not ctx.config.memory.episodic.enabled:
            body = "<div class='card'><p>Episodic memory is not enabled.</p></div>"
            return _page("Timeline", body)

        since_ts = time.time() - days * 86_400
        try:
            events = ctx.metadata_store.list_episodic_events(since=since_ts, limit=100)
        except Exception:
            events = []

        rows = ""
        for ev in events:
            ts_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ev.timestamp))
            source = _esc(ev.source[:60]) if ev.source else "-"
            n_chunks = len(ev.chunk_ids)
            rows += (
                f"<tr><td>{ts_str}</td>"
                f"<td><span class='badge-warm'>{_esc(ev.event_type)}</span></td>"
                f"<td>{source}</td>"
                f"<td style='text-align:right'>{n_chunks}</td></tr>"
            )

        table = (
            (
                "<table><thead><tr><th>Time</th><th>Type</th><th>Source</th><th>Chunks</th></tr></thead>"
                f"<tbody>{rows}</tbody></table>"
            )
            if rows
            else "<p style='color:#888'>No events in the last 7 days.</p>"
        )

        body = (
            f"<div class='card'><h2>Episodic Timeline - last {int(days)} days</h2>"
            f"<p><a href='/timeline?days=1'>1d</a> &nbsp;"
            f"<a href='/timeline?days=7'>7d</a> &nbsp;"
            f"<a href='/timeline?days=30'>30d</a></p>"
            f"{table}</div>"
        )
        return _page("Timeline", body)

    @api.get("/graph")
    async def graph_data(min_mentions: int = 2, entity_type: str = "") -> dict:
        """Return knowledge graph JSON for the D3 visualisation."""
        etype = entity_type.strip() or None
        return _build_knowledge_graph(ctx, min_mentions=min_mentions, entity_type=etype)

    @api.get("/graph-ui", response_class=HTMLResponse)
    async def graph_ui_page() -> str:
        """Render the D3.js knowledge graph visualisation page."""
        return _page("Knowledge Graph", _build_graph_ui_html())

    @api.get("/health", response_class=HTMLResponse)
    async def health_redirect() -> HTMLResponse:
        return HTMLResponse(
            content='<meta http-equiv="refresh" content="0; url=http://127.0.0.1:8766/health">',
            status_code=302,
        )

    return api


def _build_graph_ui_html() -> str:
    """Return the HTML body fragment for the D3.js knowledge graph page."""
    return """
<div class="card" style="position:relative">
  <h2>Knowledge Graph</h2>
  <div style="display:flex;gap:1rem;align-items:center;margin-bottom:1rem;flex-wrap:wrap">
    <label style="font-size:.9rem">Min mentions:
      <input id="min-mentions" type="number" value="2" min="1" max="50"
             style="width:60px;padding:4px 8px;border:1px solid #ddd;border-radius:4px;margin-left:4px">
    </label>
    <label style="font-size:.9rem">Entity type:
      <input id="entity-type" type="text" placeholder="all" value=""
             style="width:120px;padding:4px 8px;border:1px solid #ddd;border-radius:4px;margin-left:4px">
    </label>
    <button onclick="loadGraph()"
            style="padding:6px 16px;background:#1a1a2e;color:#fff;border:none;border-radius:6px;cursor:pointer">
      Reload
    </button>
    <span id="graph-status" style="color:#888;font-size:.85rem"></span>
  </div>
  <div id="graph-svg-container"
       style="width:100%;height:560px;background:#fafafa;border:1px solid #e8e8e8;border-radius:8px;overflow:hidden;position:relative">
    <svg id="graph-svg" width="100%" height="100%"></svg>
  </div>
  <div id="tooltip"
       style="display:none;position:fixed;background:#1a1a2e;color:#fff;padding:8px 12px;border-radius:6px;font-size:.82rem;pointer-events:none;z-index:999;max-width:220px"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js" crossorigin="anonymous"></script>
<script>
const TYPE_COLORS = {
  PERSON: "#e74c3c",
  ORG: "#3498db",
  GPE: "#2ecc71",
  LOC: "#9b59b6",
  PRODUCT: "#e67e22",
  EVENT: "#1abc9c",
  DATE: "#95a5a6",
  DEFAULT: "#a0c4ff",
};
function typeColor(t) { return TYPE_COLORS[t] || TYPE_COLORS.DEFAULT; }

let simulation = null;

async function loadGraph() {
  const minM = document.getElementById("min-mentions").value || 2;
  const etype = document.getElementById("entity-type").value.trim();
  const status = document.getElementById("graph-status");
  status.textContent = "Loading...";

  const url = `/graph?min_mentions=${minM}${etype ? "&entity_type=" + encodeURIComponent(etype) : ""}`;
  let data;
  try {
    const resp = await fetch(url);
    data = await resp.json();
  } catch (e) {
    status.textContent = "Error: " + e.message;
    return;
  }

  status.textContent = `${data.nodes.length} nodes - ${data.edges.length} edges`;
  drawGraph(data);
}

function drawGraph(data) {
  const container = document.getElementById("graph-svg-container");
  const svg = d3.select("#graph-svg");
  svg.selectAll("*").remove();

  const W = container.clientWidth || 800;
  const H = container.clientHeight || 560;

  if (simulation) simulation.stop();

  const g = svg.append("g");

  svg.call(
    d3.zoom().scaleExtent([0.2, 4]).on("zoom", (e) => g.attr("transform", e.transform))
  );

  const link = g.append("g").attr("stroke", "#ccc").attr("stroke-opacity", 0.7)
    .selectAll("line")
    .data(data.edges)
    .join("line")
    .attr("stroke-width", d => Math.log1p(d.weight) + 1);

  const node = g.append("g").selectAll("circle")
    .data(data.nodes)
    .join("circle")
    .attr("r", d => Math.max(6, Math.min(20, 4 + Math.sqrt(d.mentions) * 2)))
    .attr("fill", d => typeColor(d.type))
    .attr("stroke", "#fff")
    .attr("stroke-width", 1.5)
    .style("cursor", "pointer")
    .call(
      d3.drag()
        .on("start", (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag",  (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on("end",   (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; })
    );

  const label = g.append("g").selectAll("text")
    .data(data.nodes)
    .join("text")
    .text(d => d.label)
    .attr("font-size", "10px")
    .attr("fill", "#333")
    .attr("dy", d => -(Math.max(6, Math.min(20, 4 + Math.sqrt(d.mentions) * 2)) + 3))
    .attr("text-anchor", "middle")
    .style("pointer-events", "none")
    .style("user-select", "none");

  const tooltip = document.getElementById("tooltip");
  node.on("mousemove", (e, d) => {
    tooltip.style.display = "block";
    tooltip.style.left = (e.clientX + 14) + "px";
    tooltip.style.top  = (e.clientY - 10) + "px";
    tooltip.innerHTML = `<b>${d.label}</b><br>Type: ${d.type}<br>Mentions: ${d.mentions}`;
  }).on("mouseleave", () => { tooltip.style.display = "none"; });

  simulation = d3.forceSimulation(data.nodes)
    .force("link",   d3.forceLink(data.edges).id(d => d.id).distance(80))
    .force("charge", d3.forceManyBody().strength(-120))
    .force("center", d3.forceCenter(W / 2, H / 2))
    .force("collision", d3.forceCollide(22))
    .on("tick", () => {
      link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
          .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
      node.attr("cx", d => d.x).attr("cy", d => d.y);
      label.attr("x", d => d.x).attr("y", d => d.y);
    });
}

loadGraph();
</script>
"""


def _build_source_stats(ctx: AppContext) -> list[dict]:
    """Collect per-source indexing statistics from the metadata store."""
    out = []
    for src in ctx.config.sources:
        name = src.name or str(src.path)
        try:
            records = [r for r in ctx.metadata_store.list_files() if r.source_name == name]
        except Exception:
            records = []
        n_indexed = sum(1 for r in records if r.status == "indexed")
        n_errors = sum(1 for r in records if r.status == "parse_error")
        n_chunks = sum(r.n_chunks for r in records if r.status == "indexed")
        disk_bytes = sum(r.size_bytes for r in records)
        out.append(
            {
                "name": name,
                "path": str(src.path),
                "n_indexed": n_indexed,
                "n_errors": n_errors,
                "n_chunks": n_chunks,
                "disk_mb": disk_bytes / 1_048_576,
            }
        )
    return out


def _aggregate_stats(ctx: AppContext) -> dict:
    """Return aggregate index statistics across all sources."""
    try:
        records = ctx.metadata_store.list_files()
    except Exception:
        records = []
    return {
        "total_files": sum(1 for r in records if r.status == "indexed"),
        "total_chunks": sum(r.n_chunks for r in records if r.status == "indexed"),
        "total_errors": sum(1 for r in records if r.status == "parse_error"),
        "sources": len(ctx.config.sources),
    }


def _run_search(ctx: AppContext, query: str, top_k: int, mode: str) -> str:
    """Run a search and return an HTML fragment of results."""
    try:
        response = ctx.engine.search(query, top_k=top_k, mode=mode)
        hits = response.hits
    except Exception as exc:
        return f"<p style='color:red'>Search error: {_esc(str(exc))}</p>"

    if not hits:
        return "<p style='color:#888'>No results found.</p>"

    parts = [f"<p style='color:#888'>{len(hits)} result(s) in {response.duration_ms:.0f}ms</p>"]
    for h in hits:
        parts.append(
            f"<div class='hit'>"
            f"<div class='path'>{_esc(str(h.path))}"
            f"<span class='score'>score {h.score:.3f}</span></div>"
            f"<div class='preview'>{_esc(h.preview[:300])}</div>"
            f"</div>"
        )
    return "".join(parts)


def _esc(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


_GRAPH_CACHE: dict[str, tuple[float, dict]] = {}
_GRAPH_CACHE_TTL = 60.0


def _build_knowledge_graph(
    ctx: AppContext,
    min_mentions: int = 2,
    entity_type: str | None = None,
) -> dict:
    """Build a co-occurrence knowledge graph from the entity index.

    Two entities are connected when they share at least one chunk ID in the
    ``entity_mentions`` table.  The edge weight equals the number of shared
    chunks.  Results are cached for ``_GRAPH_CACHE_TTL`` seconds.

    Args:
        ctx: The shared application context (provides ``metadata_store``).
        min_mentions: Minimum mention count for an entity to appear as a node.
        entity_type: If given, restrict nodes to this entity type.

    Returns:
        Dict with ``"nodes"`` and ``"edges"`` lists ready for JSON serialisation.
    """
    cache_key = f"{min_mentions}:{entity_type or ''}"
    now = time.time()
    if cache_key in _GRAPH_CACHE:
        ts, cached = _GRAPH_CACHE[cache_key]
        if now - ts < _GRAPH_CACHE_TTL:
            return cached

    try:
        entities = ctx.metadata_store.list_entities(
            min_mentions=min_mentions,
            entity_type=entity_type,
            limit=200,
        )
    except Exception:
        entities = []

    # Build entity -> set-of-chunk-IDs mapping
    entity_chunks: dict[str, set[str]] = {}
    for ent in entities:
        try:
            chunk_ids = ctx.metadata_store.get_entity_chunks(ent.name, ent.entity_type)
        except Exception:
            chunk_ids = []
        entity_chunks[ent.name] = set(chunk_ids)

    nodes = [
        {
            "id": ent.name,
            "label": ent.name,
            "type": ent.entity_type,
            "mentions": ent.mention_count,
        }
        for ent in entities
    ]

    # Find co-occurring pairs (O(n^2) but capped at 200 nodes)
    entity_list = list(entities)
    edges: list[dict] = []
    for i, a in enumerate(entity_list):
        for b in entity_list[i + 1 :]:
            shared = entity_chunks[a.name] & entity_chunks[b.name]
            if shared:
                edges.append(
                    {
                        "source": a.name,
                        "target": b.name,
                        "weight": len(shared),
                        "shared_chunks": sorted(shared),
                    }
                )

    result: dict = {"nodes": nodes, "edges": edges}
    _GRAPH_CACHE[cache_key] = (now, result)
    return result


class DashboardServer:
    """Background HTTP server exposing the MemoryMesh web dashboard.

    Args:
        ctx: The fully initialised :class:`AppContext`.
        host: Bind address (default ``"127.0.0.1"``).
        port: TCP port (default ``8767``).
    """

    def __init__(
        self,
        ctx: AppContext,
        host: str = "127.0.0.1",
        port: int = 8767,
    ) -> None:
        self._ctx = ctx
        self._host = host
        self._port = port
        self._server: object | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the dashboard server in a background daemon thread."""
        try:
            import uvicorn  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "Dashboard: 'uvicorn' not installed - dashboard disabled. "
                "Run `uv add 'fastapi[standard]'` to enable it."
            )
            return

        try:
            app = _make_app(self._ctx)
        except ImportError as exc:
            logger.warning(f"Dashboard: {exc}")
            return

        config = uvicorn.Config(
            app,  # type: ignore[arg-type]
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)

        def _run() -> None:
            import asyncio

            asyncio.run(self._server.serve())  # type: ignore[union-attr]

        self._thread = threading.Thread(target=_run, daemon=True, name="memorymesh-dashboard")
        self._thread.start()
        logger.info(f"DashboardServer started on http://{self._host}:{self._port}")

    def stop(self) -> None:
        """Shut down the dashboard server."""
        if self._server is not None:
            with contextlib.suppress(Exception):
                self._server.should_exit = True  # type: ignore[attr-defined,union-attr]
            self._server = None
            logger.debug("DashboardServer stopped")
