# ── Build stage ───────────────────────────────────────────────────────────────
# Installs dependencies into a virtual environment that we copy to the final
# image.  Keeps the final image small by excluding the uv tool itself.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy lockfiles and install production dependencies only
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen --python 3.11

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Create a non-root user for security
RUN groupadd --system memorymesh && \
    useradd --system --gid memorymesh --home /home/memorymesh memorymesh && \
    mkdir -p /home/memorymesh/.memorymesh /data && \
    chown -R memorymesh:memorymesh /home/memorymesh /data

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /build/.venv /app/.venv

# Copy application source
COPY --chown=memorymesh:memorymesh src/ ./src/

# Copy example config (users can mount their own at /home/memorymesh/.memorymesh/config.yaml)
COPY --chown=memorymesh:memorymesh config.example.yaml ./config.example.yaml

# Activate the venv on PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# ── Default environment variables ─────────────────────────────────────────────
# Override at runtime: docker run -e MEMORYMESH_LOG_LEVEL=DEBUG ...
ENV MEMORYMESH_LOG_LEVEL=INFO
ENV MEMORYMESH_TRANSPORT=streamable-http
# Where MemoryMesh stores its index, metadata, and logs inside the container.
ENV MEMORYMESH_CONFIG=/home/memorymesh/.memorymesh/config.yaml

# MCP HTTP port
EXPOSE 8765
# Health endpoint port
EXPOSE 8766

# Persist index data and config on a named volume
VOLUME ["/home/memorymesh/.memorymesh", "/data"]

USER memorymesh

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8766/health', timeout=3)"

ENTRYPOINT ["python", "-m", "memorymesh", "start", \
            "--transport", "streamable-http", \
            "--host", "0.0.0.0", \
            "--port", "8765"]
