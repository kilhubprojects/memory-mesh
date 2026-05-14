## Summary

<!-- 1-3 bullet points describing what this PR does and why. -->

-
-

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / cleanup
- [ ] Documentation
- [ ] CI / tooling

## Test plan

<!-- Describe how you tested this. CI must be green before requesting review. -->

- [ ] `uv run pytest tests/ -m "not slow" -q` passes locally
- [ ] `uv run ruff check .` passes
- [ ] `uv run mypy src/` passes
- [ ] New behaviour is covered by a test

## Breaking changes

<!-- Does this change the signature or response shape of any MCP tool?
     search_memory / list_sources / get_document / index_now are backwards-compatible forever. -->

- [ ] No breaking changes
- [ ] Breaking change — described below

## Privacy / security checklist

- [ ] No document content is logged (only paths, hashes, counts)
- [ ] No new network calls without opt-in config
- [ ] No new files written outside `~/.memorymesh/` without user configuration
