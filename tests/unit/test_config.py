"""Unit tests for the configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from memorymesh.config import load_config
from memorymesh.core.exceptions import ConfigError


def _write(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(content, encoding="utf-8")
    return p


class TestLoadConfig:
    def test_returns_defaults_when_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import memorymesh.config as cfg_mod

        monkeypatch.setattr(cfg_mod, "_DEFAULT_SEARCH_PATHS", [tmp_path / "nope.yaml"])
        config = load_config(None)
        assert config.embeddings.model == "all-MiniLM-L6-v2"
        assert config.server.transport == "stdio"
        assert len(config.global_ignore) > 0

    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        p = _write(
            tmp_path,
            """
sources:
  - name: notes
    path: ~/Notes
    recursive: true
embeddings:
  model: paraphrase-multilingual-MiniLM-L12-v2
""",
        )
        config = load_config(p)
        assert config.sources[0].name == "notes"
        assert config.embeddings.model == "paraphrase-multilingual-MiniLM-L12-v2"

    def test_tilde_expanded_in_source_path(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "sources:\n  - path: ~/Documents\n")
        config = load_config(p)
        assert not str(config.sources[0].path).startswith("~")
        assert config.sources[0].path == Path.home() / "Documents"

    def test_tilde_expanded_in_base_dir(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "storage:\n  base_dir: ~/.memorymesh\n")
        config = load_config(p)
        assert config.storage.base_dir == Path.home() / ".memorymesh"

    def test_storage_path_properties(self, tmp_path: Path) -> None:
        config = load_config(None)
        assert config.storage.metadata_db_path.name == "metadata.sqlite3"
        assert config.storage.vector_db_path.name == "chroma_db"
        assert config.storage.bm25_pickle_path.name == "bm25.pkl"
        assert config.storage.audit_log_path.name == "audit.jsonl"

    def test_raises_config_error_on_explicit_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "missing.yaml")

    def test_raises_config_error_on_malformed_yaml(self, tmp_path: Path) -> None:
        p = tmp_path / "config.yaml"
        p.write_text("key: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigError, match="YAML parse error"):
            load_config(p)

    def test_raises_config_error_on_invalid_values(self, tmp_path: Path) -> None:
        p = _write(tmp_path, "server:\n  transport: ftp\n")
        with pytest.raises(ConfigError):
            load_config(p)

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        p = tmp_path / "config.yaml"
        p.write_text("", encoding="utf-8")
        config = load_config(p)
        assert config.embeddings.model == "all-MiniLM-L6-v2"

    def test_global_ignore_has_git_entry(self, tmp_path: Path) -> None:
        config = load_config(None)
        assert any(".git" in pattern for pattern in config.global_ignore)

    def test_chunking_defaults(self, tmp_path: Path) -> None:
        config = load_config(None)
        assert config.chunking.default.chunk_size == 512
        assert config.chunking.default.chunk_overlap == 50
        assert config.chunking.code.max_chunk_size == 1024
