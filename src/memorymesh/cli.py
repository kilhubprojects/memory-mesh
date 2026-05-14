"""MemoryMesh CLI - complete Typer application.

Commands
--------
* ``start``   - boot the MCP server (stdio or http daemon).
* ``stop``    - gracefully stop the HTTP daemon via POST /admin/shutdown.
* ``status``  - show indexing statistics for all configured sources.
* ``index``   - trigger an immediate index of a path or all sources.
* ``reindex`` - force re-index (ignores hash cache).
* ``search``  - run a query from the terminal for quick testing.
* ``serve``   - alias for ``start`` (useful in Docker / systemd).
* ``info``    - show version and config paths.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import typer
from loguru import logger

from memorymesh import __version__
from memorymesh.observability.logging import setup_logging

app = typer.Typer(
    name="memorymesh",
    help="MemoryMesh - local MCP hub for personal data.",
    add_completion=False,
    no_args_is_help=True,
)

_DEFAULT_CONFIG: Path | None = None
_DEFAULT_PID = Path("~/.memorymesh/daemon.pid").expanduser()
_DEFAULT_ADMIN_TOKEN_ENV = "MEMORYMESH_ADMIN_TOKEN"


@app.callback()
def _global_options(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        envvar="MEMORYMESH_CONFIG",
        help="Path to config.yaml.  Defaults to ~/.memorymesh/config.yaml.",
        show_default=True,
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        envvar="MEMORYMESH_LOG_LEVEL",
        help="Log level: DEBUG | INFO | WARNING | ERROR | CRITICAL",
        show_default=True,
    ),
) -> None:
    """Global options applied before every command."""
    setup_logging(level=log_level.upper())
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = config


def _load_context(config_path: Path | None) -> tuple:
    """Bootstrap all runtime objects and return ``(mcp, app_ctx)``."""
    from memorymesh.auth.acl import ACLEnforcer
    from memorymesh.auth.identity import IdentityResolver
    from memorymesh.auth.rate_limiter import RateLimiter
    from memorymesh.auth.revocation import RevocationList
    from memorymesh.config import load_config
    from memorymesh.embeddings.sentence_transformers_provider import (
        SentenceTransformersProvider,
    )
    from memorymesh.indexer.file_indexer import FileIndexer
    from memorymesh.observability.audit import AuditLogger
    from memorymesh.search.engine import SearchEngine
    from memorymesh.server.app import AppContext, build_mcp
    from memorymesh.storage.bm25_index import BM25Index
    from memorymesh.storage.metadata_store import MetadataStore
    from memorymesh.storage.tiered import TieredMemoryManager
    from memorymesh.storage.vector_store import ChromaVectorStore

    cfg = load_config(config_path)

    provider = SentenceTransformersProvider(config=cfg.embeddings)

    vector_store = ChromaVectorStore(
        db_path=cfg.storage.vector_db_path,
        embedding_model_id=provider.model_id,
    )
    metadata_store = MetadataStore(cfg.storage.metadata_db_path)
    bm25 = BM25Index(cfg.storage.bm25_pickle_path)
    bm25.load()

    # Optional Ollama LLM client - only instantiated when enabled in config.
    ollama_client = None
    if cfg.ollama.enabled:
        from memorymesh.llm.ollama_client import OllamaClient

        ollama_client = OllamaClient(cfg.ollama)

    indexer = FileIndexer(
        config=cfg,
        vector_store=vector_store,
        metadata_store=metadata_store,
        embedding_provider=provider,
        bm25_index=bm25,
        ollama_client=ollama_client,
    )

    engine = SearchEngine(
        vector_store=vector_store,
        bm25_index=bm25,
        embedding_provider=provider,
        config=cfg.search,
        ollama_config=cfg.ollama,
        metadata_store=metadata_store,
    )

    audit_logger = AuditLogger(cfg.storage.audit_log_path)

    tiered_memory = TieredMemoryManager(
        store=metadata_store,
        tier_config=cfg.memory.tiers,
        forgetting_config=cfg.memory.forgetting,
    )

    identity_resolver = IdentityResolver(cfg.auth)
    acl_enforcer = ACLEnforcer(cfg.auth, identity_resolver)
    rate_limiter = RateLimiter(cfg.auth)
    revocation_list: RevocationList | None = None
    try:
        revocation_db = cfg.storage.base_dir / "revocation.sqlite3"
        revocation_list = RevocationList(revocation_db)
    except Exception as exc:
        logger.debug(f"Revocation list unavailable; continuing without it: {exc}")

    ctx_obj = AppContext(
        config=cfg,
        vector_store=vector_store,
        metadata_store=metadata_store,
        bm25=bm25,
        provider=provider,
        indexer=indexer,
        engine=engine,
        audit_logger=audit_logger,
        ollama_client=ollama_client,
        tiered_memory=tiered_memory,
        identity_resolver=identity_resolver,
        acl_enforcer=acl_enforcer,
        rate_limiter=rate_limiter,
        revocation_list=revocation_list,
    )
    mcp = build_mcp(ctx_obj)
    return mcp, ctx_obj


@app.command()
def start(
    ctx: typer.Context,
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        envvar="MEMORYMESH_TRANSPORT",
        help="Transport: 'stdio' or 'streamable-http'.",
        show_default=True,
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="HTTP listener host (streamable-http only).",
        show_default=True,
    ),
    port: int = typer.Option(
        8765,
        "--port",
        "-p",
        help="HTTP listener port (streamable-http only).",
        show_default=True,
    ),
    detach: bool = typer.Option(
        False,
        "--detach",
        help="Launch daemon in background and return immediately.",
    ),
    rest_api: bool = typer.Option(
        False,
        "--rest-api/--no-rest-api",
        help="Enable the REST API at /api (streamable-http transport only).",
    ),
) -> None:
    """Start the MemoryMesh MCP server.

    Uses stdio transport by default (suitable for Claude Desktop / Cursor).
    Pass ``--transport streamable-http`` to run as an HTTP daemon.
    Add ``--detach`` to launch in the background and return immediately.
    """
    if detach:
        import subprocess

        cmd = [
            sys.executable,
            "-m",
            "memorymesh",
            "start",
            "--transport",
            transport,
            "--host",
            host,
            "--port",
            str(port),
        ]
        config_path: Path | None = ctx.obj["config_path"]
        if config_path:
            cmd += ["--config", str(config_path)]

        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            kwargs["start_new_session"] = True

        proc = subprocess.Popen(cmd, **kwargs)
        typer.echo(f"Daemon started (PID: {proc.pid})")
        return

    config_path = ctx.obj["config_path"]
    mcp, app_ctx = _load_context(config_path)

    # Override transport / http settings from CLI flags.
    # We mutate a fresh copy rather than the frozen AppContext.
    from memorymesh.core.models import ServerConfig, ServerHttpConfig

    server_cfg = ServerConfig(
        transport=transport,  # type: ignore[arg-type]
        http=ServerHttpConfig(host=host, port=port),
    )
    # Build a new AppContext with the overridden server config.
    from dataclasses import replace as dc_replace

    new_config = app_ctx.config.model_copy(update={"server": server_cfg})
    app_ctx = dc_replace(app_ctx, config=new_config)

    app_ctx.indexer.startup()

    from memorymesh.server.health import HealthServer

    health_cfg = getattr(app_ctx.config, "health", None)
    health_host = getattr(health_cfg, "host", "127.0.0.1") if health_cfg else "127.0.0.1"
    health_port = getattr(health_cfg, "port", 8766) if health_cfg else 8766
    health_server = HealthServer(app_ctx, host=health_host, port=health_port)
    health_server.start()

    from memorymesh.server.dashboard import DashboardServer

    dash_server = DashboardServer(app_ctx, host=health_host, port=8767)
    dash_server.start()

    from memorymesh.server.transports import run

    try:
        run(mcp, app_ctx, enable_rest_api=rest_api)
    finally:
        dash_server.stop()
        health_server.stop()
        app_ctx.indexer.shutdown()


@app.command()
def serve(ctx: typer.Context) -> None:
    """Alias for ``start`` - useful in Docker / systemd unit files."""
    # Forward to start with default options.
    ctx.invoke(start)


@app.command()
def stop(
    pid_file: Path = typer.Option(
        _DEFAULT_PID,
        "--pid-file",
        help="Path to the daemon PID file.",
        show_default=True,
    ),
    token: str | None = typer.Option(
        None,
        "--token",
        envvar=_DEFAULT_ADMIN_TOKEN_ENV,
        help="Admin bearer token (MEMORYMESH_ADMIN_TOKEN).",
    ),
    host: str = typer.Option("127.0.0.1", "--host", show_default=True),
    port: int = typer.Option(8765, "--port", "-p", show_default=True),
) -> None:
    """Gracefully stop the HTTP daemon.

    First tries ``POST /admin/shutdown``; falls back to ``psutil`` process
    termination if the HTTP call fails.
    """

    # Try HTTP shutdown first.
    shutdown_ok = _http_shutdown(host, port, token)
    if shutdown_ok:
        typer.echo("MemoryMesh daemon stopped via HTTP shutdown.")
        return

    # Fallback: read PID file and terminate.
    if not pid_file.exists():
        typer.echo(
            f"No PID file found at {pid_file}.  Is the daemon running in HTTP mode?",
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        pid = int(pid_file.read_text().strip())
    except (ValueError, OSError) as exc:
        typer.echo(f"Cannot read PID file: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    _kill_pid(pid, pid_file)


def _http_shutdown(host: str, port: int, token: str | None) -> bool:
    """POST /admin/shutdown; return True on success."""
    try:
        import urllib.request

        url = f"http://{host}:{port}/admin/shutdown"
        headers = {}
        if token:
            headers["X-MemoryMesh-Token"] = token
        req = urllib.request.Request(url, data=b"", headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return bool(resp.status == 200)
    except Exception:
        return False


def _kill_pid(pid: int, pid_file: Path) -> None:
    """Terminate process *pid* with psutil or os.kill fallback."""
    try:
        import psutil

        proc = psutil.Process(pid)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except psutil.TimeoutExpired:
            proc.kill()
        typer.echo(f"MemoryMesh daemon (PID {pid}) stopped.")
    except ImportError:
        import signal as _signal

        try:
            os.kill(pid, _signal.SIGTERM)
            typer.echo(f"Sent SIGTERM to PID {pid}.")
        except ProcessLookupError:
            typer.echo(f"Process {pid} is not running.", err=True)
        except PermissionError as exc:
            typer.echo(f"Cannot terminate PID {pid}: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    except Exception as exc:
        typer.echo(f"Error stopping PID {pid}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        with contextlib.suppress(OSError):
            if pid_file.exists():
                pid_file.unlink()


@app.command()
def status(ctx: typer.Context) -> None:
    """Show indexing statistics for all configured sources."""
    config_path: Path | None = ctx.obj["config_path"]
    from memorymesh.config import load_config
    from memorymesh.storage.metadata_store import MetadataStore

    cfg = load_config(config_path)
    ms = MetadataStore(cfg.storage.metadata_db_path)

    if not cfg.sources:
        typer.echo("No sources configured.  Add sources to your config.yaml.")
        return

    typer.echo(f"MemoryMesh  v{__version__}")
    typer.echo(f"State dir   : {cfg.storage.base_dir}")
    typer.echo("")
    typer.echo(f"{'Source':<30} {'Indexed':>8} {'Errors':>7} {'Chunks':>8}")
    typer.echo("-" * 56)

    for src in cfg.sources:
        name = src.name or str(src.path)
        records = [r for r in ms.list_files() if r.source_name == name]
        n_indexed = sum(1 for r in records if r.status == "indexed")
        n_err = sum(1 for r in records if r.status == "parse_error")
        n_chunks = sum(r.n_chunks for r in records if r.status == "indexed")
        typer.echo(f"{name:<30} {n_indexed:>8} {n_err:>7} {n_chunks:>8}")


@app.command()
def index(
    ctx: typer.Context,
    path: Path | None = typer.Argument(
        None,
        help="File or directory to index.  Omit to index all configured sources.",
    ),
    source_name: str = typer.Option(
        "",
        "--source",
        "-s",
        help="Source name to assign (used with --path).",
    ),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        help="Scan subdirectories recursively (default: True).",
    ),
) -> None:
    """Index a file, directory, or all configured sources."""
    config_path: Path | None = ctx.obj["config_path"]
    _run_index(config_path, path, source_name, force=False, recursive=recursive)


@app.command()
def reindex(
    ctx: typer.Context,
    path: Path | None = typer.Argument(
        None,
        help="File or directory to force re-index.",
    ),
    source_name: str = typer.Option("", "--source", "-s"),
) -> None:
    """Force re-index (ignores the SHA-256 hash cache)."""
    config_path: Path | None = ctx.obj["config_path"]
    _run_index(config_path, path, source_name, force=True)


def _run_index(
    config_path: Path | None,
    path: Path | None,
    source_name: str,
    force: bool,
    recursive: bool = True,
) -> None:
    """Shared logic for index / reindex commands."""
    from memorymesh.config import load_config
    from memorymesh.embeddings.sentence_transformers_provider import (
        SentenceTransformersProvider,
    )
    from memorymesh.indexer.file_indexer import FileIndexer
    from memorymesh.storage.bm25_index import BM25Index
    from memorymesh.storage.metadata_store import MetadataStore
    from memorymesh.storage.vector_store import ChromaVectorStore

    cfg = load_config(config_path)
    provider = SentenceTransformersProvider(config=cfg.embeddings)
    vector_store = ChromaVectorStore(
        db_path=cfg.storage.vector_db_path,
        embedding_model_id=provider.model_id,
    )
    metadata_store = MetadataStore(cfg.storage.metadata_db_path)
    bm25 = BM25Index(cfg.storage.bm25_pickle_path)
    bm25.load()

    indexer = FileIndexer(
        config=cfg,
        vector_store=vector_store,
        metadata_store=metadata_store,
        embedding_provider=provider,
        bm25_index=bm25,
    )

    if path is not None:
        if not path.exists():
            typer.echo(f"Path not found: {path}", err=True)
            raise typer.Exit(code=1)
        if path.is_file():
            result = indexer.index_file(path, source_name=source_name, force=force)
            typer.echo(f"{result.status}  {path}  chunks={result.n_chunks}")
        else:
            results = indexer.index_directory(
                path, source_name=source_name, recursive=recursive, force=force
            )
            _print_index_summary(results)
    else:
        if not cfg.sources:
            typer.echo("No sources configured.", err=True)
            raise typer.Exit(code=1)
        for src in cfg.sources:
            src_name = src.name or str(src.path)
            typer.echo(f"Indexing source: {src_name!r} ({src.path})")
            results = indexer.index_directory(
                src.path,
                source_name=src_name,
                recursive=src.recursive,
                extensions=src.extensions or None,
                ignore_patterns=src.ignore or None,
                force=force,
            )
            _print_index_summary(results)


def _print_index_summary(results: list) -> None:
    n_indexed = sum(1 for r in results if r.status == "indexed")
    n_skipped = sum(1 for r in results if r.status == "skipped")
    n_errors = sum(1 for r in results if r.status == "parse_error")
    n_chunks = sum(r.n_chunks for r in results)
    typer.echo(f"  indexed={n_indexed}  skipped={n_skipped}  errors={n_errors}  chunks={n_chunks}")
    for r in results:
        if r.status == "parse_error":
            typer.echo(f"  ERROR  {r.path}: {r.error}", err=True)


@app.command()
def search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Search query."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results.", show_default=True),
    mode: str = typer.Option(
        "hybrid",
        "--mode",
        "-m",
        help="Search mode: hybrid | dense | sparse.",
        show_default=True,
    ),
    modality: str = typer.Option(
        "all",
        "--modality",
        help="Modality filter: all | text | image.",
        show_default=True,
    ),
) -> None:
    """Run a search query from the terminal."""
    from memorymesh.config import load_config
    from memorymesh.embeddings.sentence_transformers_provider import (
        SentenceTransformersProvider,
    )
    from memorymesh.search.engine import SearchEngine
    from memorymesh.storage.bm25_index import BM25Index
    from memorymesh.storage.metadata_store import MetadataStore
    from memorymesh.storage.vector_store import ChromaVectorStore

    config_path: Path | None = ctx.obj["config_path"]
    cfg = load_config(config_path)

    provider = SentenceTransformersProvider(config=cfg.embeddings)
    vector_store = ChromaVectorStore(
        db_path=cfg.storage.vector_db_path,
        embedding_model_id=provider.model_id,
    )
    bm25 = BM25Index(cfg.storage.bm25_pickle_path)
    bm25.load()
    search_metadata_store = MetadataStore(cfg.storage.metadata_db_path)

    engine = SearchEngine(
        vector_store=vector_store,
        bm25_index=bm25,
        embedding_provider=provider,
        config=cfg.search,
        metadata_store=search_metadata_store,
    )

    response = engine.search(query, top_k=top_k, mode=mode, modality=modality)

    typer.echo(
        f"mode={response.mode}  hits={len(response.hits)}  duration={response.duration_ms:.1f}ms\n"
    )
    for i, hit in enumerate(response.hits, start=1):
        typer.echo(f"[{i}] score={hit.score:.4f}  {hit.path}:{hit.chunk_index}")
        typer.echo(f"    {hit.preview[:120]}")
        typer.echo("")


@app.command()
def sync(
    ctx: typer.Context,
    connector_type: str | None = typer.Argument(
        None,
        help=(
            "Connector type to sync, e.g. 'jira'.  Omit to sync all enabled connectors from config."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Fetch documents but do not write to the index.",
    ),
) -> None:
    """Sync external data sources (connectors) into the index.

    Reads connector settings from ``config.yaml`` and fetches all documents
    from enabled connectors, indexing each one via ``index_parsed_document``.
    """
    from memorymesh.config import load_config
    from memorymesh.connectors.registry import get_connector_classes
    from memorymesh.embeddings.sentence_transformers_provider import (
        SentenceTransformersProvider,
    )
    from memorymesh.indexer.file_indexer import FileIndexer
    from memorymesh.storage.bm25_index import BM25Index
    from memorymesh.storage.metadata_store import MetadataStore
    from memorymesh.storage.vector_store import ChromaVectorStore

    config_path: Path | None = ctx.obj["config_path"]
    cfg = load_config(config_path)

    if not cfg.connectors:
        typer.echo("No connectors configured.  Add connectors to your config.yaml.")
        raise typer.Exit(code=0)

    connectors_to_run = [
        c
        for c in cfg.connectors
        if c.enabled and (connector_type is None or c.type == connector_type)
    ]

    if not connectors_to_run:
        suffix_msg = f" of type '{connector_type}'" if connector_type else ""
        typer.echo(f"No enabled connector found{suffix_msg}.")
        raise typer.Exit(code=1)

    if not dry_run:
        provider = SentenceTransformersProvider(config=cfg.embeddings)
        vector_store = ChromaVectorStore(
            db_path=cfg.storage.vector_db_path,
            embedding_model_id=provider.model_id,
        )
        metadata_store = MetadataStore(cfg.storage.metadata_db_path)
        bm25 = BM25Index(cfg.storage.bm25_pickle_path)
        bm25.load()
        indexer = FileIndexer(
            config=cfg,
            vector_store=vector_store,
            metadata_store=metadata_store,
            embedding_provider=provider,
            bm25_index=bm25,
        )
    else:
        indexer = None

    total_docs = 0
    total_errors = 0

    for conn_cfg in connectors_to_run:
        typer.echo(f"Syncing connector: {conn_cfg.type!r}")
        try:
            cfg_cls, conn_cls = get_connector_classes(conn_cfg.type)
        except KeyError:
            typer.echo(f"  ERROR: unknown connector type '{conn_cfg.type}'", err=True)
            total_errors += 1
            continue

        try:
            config_obj = cfg_cls(**conn_cfg.config)
            connector = conn_cls(config_obj)
        except Exception as exc:
            typer.echo(f"  ERROR: failed to instantiate connector: {exc}", err=True)
            total_errors += 1
            continue

        source_name = getattr(config_obj, "source_name", conn_cfg.type)
        doc_count = 0

        try:
            for doc in connector.fetch_documents():
                doc_count += 1
                if not dry_run and indexer is not None:
                    result = indexer.index_parsed_document(doc, source_name)
                    if result.status == "parse_error":
                        total_errors += 1
        except Exception as exc:
            typer.echo(f"  ERROR: connector fetch failed: {exc}", err=True)
            total_errors += 1

        total_docs += doc_count
        typer.echo(f"  {doc_count} document(s) {'fetched (dry-run)' if dry_run else 'indexed'}")

    suffix = " (dry-run)" if dry_run else ""
    typer.echo(f"\nSync complete{suffix}: {total_docs} document(s), {total_errors} error(s).")

    if not dry_run and indexer is not None:
        bm25.save()


@app.command()
def info() -> None:
    """Show version and basic system information."""
    typer.echo(f"MemoryMesh  v{__version__}")
    typer.echo("Config      : ~/.memorymesh/config.yaml  (default)")
    typer.echo("State dir   : ~/.memorymesh/")
    typer.echo(f"PID file    : {_DEFAULT_PID}  (HTTP daemon)")
    typer.echo("")
    typer.echo("Commands: start  stop  status  index  reindex  search  serve  info")
    typer.echo("          backup  keygen")


@app.command()
def openapi(
    output: Path = typer.Option(
        Path("openapi.json"),
        "--output",
        "-o",
        help="Output path for the OpenAPI JSON schema.",
    ),
) -> None:
    """Export the REST API OpenAPI schema to a JSON file."""
    import json

    # Build a minimal ctx to instantiate the REST API app
    ctx_mock: object = type(
        "_Ctx",
        (),
        {
            "config": type(
                "_Cfg",
                (),
                {
                    "sources": [],
                    "memory": type("_M", (), {"episodic": type("_E", (), {"enabled": False})()})(),
                },
            )(),
            "metadata_store": None,
            "engine": None,
            "tiered_memory": None,
            "indexer": None,
        },
    )()
    from memorymesh.server.rest_api import build_rest_api  # type: ignore[import]

    api = build_rest_api(ctx_mock)  # type: ignore[arg-type]
    schema = api.openapi()
    output.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    typer.echo(f"OpenAPI schema written to: {output}")


@app.command()
def keygen(
    key_file: Path = typer.Option(
        Path("~/.memorymesh/secret.key"),
        "--key-file",
        help="Destination path for the generated Fernet key.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing key file."),
) -> None:
    """Generate a new Fernet encryption key for at-rest encryption."""
    from memorymesh.storage.encryption import EncryptionManager

    resolved = key_file.expanduser().resolve()
    if resolved.exists() and not force:
        typer.echo(
            f"Key file already exists: {resolved}\nUse --force to overwrite.",
            err=True,
        )
        raise typer.Exit(1)

    try:
        EncryptionManager.generate_key(key_file)
        typer.echo(f"Key written to: {resolved}")
        typer.echo("Keep this file secret - it protects your encrypted audit log and backups.")
    except ImportError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc


@app.command()
def backup(
    output: Path = typer.Argument(..., help="Output path for the backup file."),
    config_path: Path = typer.Option(
        Path("~/.memorymesh/config.yaml"),
        "--config",
        help="Path to config.yaml.",
    ),
    encrypt: bool = typer.Option(
        False,
        "--encrypt/--no-encrypt",
        help="Encrypt the backup using the key from config.",
    ),
) -> None:
    """Export a backup of the MemoryMesh metadata database."""
    from memorymesh.config import load_config
    from memorymesh.storage.metadata_store import MetadataStore

    try:
        cfg = load_config(config_path)
    except Exception as exc:
        typer.echo(f"Config error: {exc}", err=True)
        raise typer.Exit(1) from exc

    db_path = cfg.storage.metadata_db_path
    store = MetadataStore(db_path)

    if encrypt:
        from memorymesh.storage.encryption import EncryptionManager

        enc = EncryptionManager.from_config(cfg.encryption)
        if enc is None:
            typer.echo(
                "Encryption is not configured or key file is missing.  "
                "Run `memorymesh keygen` first and set encryption.enabled: true.",
                err=True,
            )
            raise typer.Exit(1)
        dest = store.export_encrypted(output, enc)
        typer.echo(f"Encrypted backup written to: {dest}")
    else:
        import shutil

        output = output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_path, output)
        typer.echo(f"Backup written to: {output}")


@app.command()
def doctor(
    ctx: typer.Context,
    repair: bool = typer.Option(
        False,
        "--repair",
        "--fix",
        help="Repair broken metadata records and orphan vectors where possible.",
    ),
) -> None:
    """Diagnose metadata/vector drift and optionally repair it."""
    from memorymesh.config import load_config
    from memorymesh.core.models import FileRecord
    from memorymesh.storage.metadata_store import MetadataStore

    config_path: Path | None = ctx.obj["config_path"]
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        typer.echo(f"Config error: {exc}", err=True)
        raise typer.Exit(1) from exc

    store = MetadataStore(cfg.storage.metadata_db_path)

    try:
        all_records = store.list_files()
    except Exception as exc:
        typer.echo(f"Cannot read metadata DB: {exc}", err=True)
        raise typer.Exit(1) from exc

    indexed_records = [r for r in all_records if r.status == "indexed"]
    metadata_paths = {r.path for r in indexed_records}
    broken = [r for r in all_records if r.status in ("indexing", "failed")]

    vector_paths: set[str] = set()
    vector_ids_by_path: dict[str, list[str]] = {}
    vector_scan_error: str | None = None

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(cfg.storage.vector_db_path))
        collection = client.get_collection("memorymesh_chunks")
        raw_vectors = collection.get(include=["metadatas"])
        ids = raw_vectors.get("ids", []) or []
        metadatas = raw_vectors.get("metadatas", []) or []
        for chunk_id, meta in zip(ids, metadatas, strict=False):
            if not isinstance(meta, dict):
                continue
            path = str(meta.get("path", "") or "")
            if not path:
                continue
            vector_paths.add(path)
            vector_ids_by_path.setdefault(path, []).append(str(chunk_id))
    except Exception as exc:
        vector_scan_error = str(exc)

    orphan_records = (
        [] if vector_scan_error else [r for r in indexed_records if r.path not in vector_paths]
    )
    orphan_vector_paths = sorted(vector_paths - metadata_paths)

    if not broken and not orphan_records and not orphan_vector_paths:
        typer.echo("No broken records or index drift found.")
        if vector_scan_error:
            typer.echo(f"Vector scan skipped: {vector_scan_error}")
        return

    if broken:
        typer.echo(f"Found {len(broken)} broken metadata record(s):\n")
        for r in broken:
            typer.echo(f"  [{r.status}]  {r.path}")
            if r.error_message:
                typer.echo(f"             error: {r.error_message}")

    if orphan_records:
        typer.echo(f"\nFound {len(orphan_records)} indexed record(s) without vectors:\n")
        for r in orphan_records[:50]:
            typer.echo(f"  [missing_vectors]  {r.path}")

    if orphan_vector_paths:
        typer.echo(f"\nFound {len(orphan_vector_paths)} vector path(s) without metadata:\n")
        for path in orphan_vector_paths[:50]:
            typer.echo(f"  [orphan_vector]  {path}")

    if vector_scan_error:
        typer.echo(f"\nVector scan error: {vector_scan_error}", err=True)

    if repair:
        typer.echo("\nResetting repairable records to 'pending_reindex'...")
        for r in [*broken, *orphan_records]:
            fixed = FileRecord(
                path=r.path,
                source_name=r.source_name,
                sha256=r.sha256,
                mtime=r.mtime,
                size_bytes=r.size_bytes,
                file_type=r.file_type,
                n_chunks=r.n_chunks,
                status="pending_reindex",
                error_message=None,
                indexed_at=r.indexed_at,
                embedding_model_id=r.embedding_model_id,
            )
            try:
                store.upsert_file(fixed)
                typer.echo(f"  queued reindex: {r.path}")
            except Exception as exc:
                typer.echo(f"  failed to queue {r.path}: {exc}", err=True)
        if orphan_vector_paths:
            try:
                import chromadb

                client = chromadb.PersistentClient(path=str(cfg.storage.vector_db_path))
                collection = client.get_collection("memorymesh_chunks")
                orphan_ids = [
                    chunk_id
                    for path in orphan_vector_paths
                    for chunk_id in vector_ids_by_path.get(path, [])
                ]
                if orphan_ids:
                    collection.delete(ids=orphan_ids)
                    typer.echo(f"  deleted {len(orphan_ids)} orphan vector chunk(s)")
            except Exception as exc:
                typer.echo(f"  failed to delete orphan vectors: {exc}", err=True)
        typer.echo("\nDone. Run 'memorymesh index' to reindex the queued files.")
    else:
        typer.echo(
            "\nRun 'memorymesh doctor --repair' to queue missing-vector records "
            "and remove orphan vectors."
        )
        raise typer.Exit(1)
