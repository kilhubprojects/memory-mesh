# Contributing to MemoryMesh

Thank you for your interest in contributing. This document covers everything
you need to go from a fresh clone to an approved pull request.

---

## Prerequisites

- Python 3.11 or 3.12
- [`uv`](https://github.com/astral-sh/uv) — the project's package manager

---

## Setup

```bash
git clone https://github.com/kilhubprojects/memory-mesh.git
cd memory-mesh

# Create virtual environment and install all dependencies (including dev)
uv sync --all-extras

# Install pre-commit hooks (runs ruff on every commit)
uv run pre-commit install
```

---

## Running tests

```bash
# Fast suite — excludes tests that load a real ML model (~30 seconds)
uv run pytest tests/ -m "not slow" -q

# Full suite including slow tests (loads sentence-transformers model, ~2 minutes)
uv run pytest tests/ -q

# With coverage report
uv run pytest tests/ -m "not slow" --cov=src/memorymesh --cov-report=term-missing -q
```

---

## Linting and type checking

```bash
# Lint + auto-fix
uv run ruff check . --fix

# Format
uv run ruff format .

# Type check
uv run mypy src/
```

The CI runs all three. A PR will not be merged if any of these fail.

---

## Commit convention

We use [Conventional Commits](https://www.conventionalcommits.org/). Every
commit message must start with one of these prefixes:

| Prefix | When to use |
|--------|-------------|
| `feat:` | A new feature or capability |
| `fix:` | A bug fix |
| `docs:` | Documentation only |
| `test:` | Adding or fixing tests |
| `chore:` | Build, CI, dependencies, tooling |
| `refactor:` | Code restructuring without behaviour change |
| `perf:` | Performance improvements |

Examples:

```
feat(search): add parent document retriever with configurable window
fix(bm25): handle empty corpus without raising ZeroDivisionError
docs(readme): add Claude Desktop configuration example
test(indexer): cover reconcile() path after dirty shutdown
chore(ci): add macOS to test matrix
```

Breaking changes must include `BREAKING CHANGE:` in the commit footer.

---

## Opening a pull request

1. **Fork** the repo and create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make your changes, following the code rules in `CLAUDE.md`.
3. Add or update tests. The test suite must stay green.
4. Run `uv run ruff check . && uv run mypy src/` locally before pushing.
5. Open a PR against `main`. The CI matrix (Ubuntu / Windows / macOS,
   Python 3.11 + 3.12) must be green before a review is requested.
6. Fill in the PR template — especially the **Test plan** section.

---

## Breaking changes policy

The four MCP tools (`search_memory`, `list_sources`, `get_document`,
`index_now`) are part of the public API and are **backwards-compatible
forever**. Any change to their signatures or return shapes is a breaking
change and requires a major version bump.

Adding new optional fields to responses (like `extended_preview`) is **not**
a breaking change — existing clients silently ignore unknown keys.

---

## Questions?

Open a [GitHub Discussion](https://github.com/kilhubprojects/memory-mesh/discussions)
or file an issue using the bug report template.
