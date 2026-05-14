"""Configuration loader for MemoryMesh.

Reads a YAML file, merges it with field defaults from :class:`MemoryMeshConfig`,
and returns a validated model.  The search order for a default config path is:

1. ``./config.yaml`` (current working directory)
2. ``~/.memorymesh/config.yaml``

If neither file exists and no explicit path is given, all defaults are used -
the daemon will start without any indexed sources, which is a valid (empty) state.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger
from pydantic import ValidationError

from memorymesh.core.exceptions import ConfigError
from memorymesh.core.models import MemoryMeshConfig

_DEFAULT_SEARCH_PATHS: list[Path] = [
    Path.cwd() / "config.yaml",
    Path.home() / ".memorymesh" / "config.yaml",
]


def load_config(path: Path | None = None) -> MemoryMeshConfig:
    """Load and validate the MemoryMesh configuration from a YAML file.

    Args:
        path: Explicit path to ``config.yaml``.  When ``None``, the function
            searches :data:`_DEFAULT_SEARCH_PATHS` in order and uses the first
            file that exists.  If no file is found, all defaults apply.

    Returns:
        A fully validated :class:`~memorymesh.core.models.MemoryMeshConfig`.

    Raises:
        ConfigError: If the provided ``path`` does not exist, the YAML is
            malformed, or the parsed data fails Pydantic validation.
    """
    resolved = _resolve_path(path)

    if resolved is None:
        logger.debug("No config.yaml found; using built-in defaults.")
        return MemoryMeshConfig()

    logger.info(f"Loading config from {resolved}")

    raw = _read_yaml(resolved)
    if raw is None:
        logger.warning(f"Config file {resolved} is empty; using built-in defaults.")
        return MemoryMeshConfig()

    if not isinstance(raw, dict):
        raise ConfigError(
            f"Config file {resolved} must be a YAML mapping at the top level, "
            f"got {type(raw).__name__}."
        )

    try:
        config = MemoryMeshConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"Config file {resolved} is invalid:\n{exc}") from exc

    logger.debug(
        f"Config loaded: {len(config.sources)} source(s), "
        f"model={config.embeddings.model!r}, "
        f"transport={config.server.transport!r}"
    )
    return config


def _resolve_path(path: Path | None) -> Path | None:
    """Return a confirmed-existing Path, or None if nothing is found."""
    if path is not None:
        expanded = path.expanduser()
        if not expanded.exists():
            raise ConfigError(f"Config file not found: {expanded}")
        return expanded

    for candidate in _DEFAULT_SEARCH_PATHS:
        if candidate.exists():
            return candidate

    return None


def _read_yaml(path: Path) -> object:
    """Parse a YAML file, raising :class:`ConfigError` on syntax errors."""
    try:
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error in {path}:\n{exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc}") from exc
