"""PII detection and redaction for chunk text.

Uses regex heuristics to detect common PII patterns:
- Email addresses
- Phone numbers (common international formats)
- Social Security Numbers (SSN) — US format
- Credit card numbers (16-digit groups)
- IPv4 addresses

When ``redact=True`` (default), detected spans are replaced with
``[REDACTED]`` placeholders so the chunk text is sanitised before indexing.
When ``redact=False``, any chunk containing PII is dropped entirely.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from memorymesh.core.models import ChunkWithEmbedding

# PII pattern registry — (name, compiled_regex) pairs
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("email", re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
    (
        "phone",
        re.compile(
            r"(?<!\d)(\+?\d{1,3}[\s\-]?)?"
            r"(\(?\d{2,4}\)?[\s\-]?)?"
            r"\d{3,4}[\s\-]?\d{4}(?!\d)"
        ),
    ),
    ("ssn", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    (
        "credit_card",
        re.compile(r"(?<!\d)\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}(?!\d)"),
    ),
    (
        "ipv4",
        re.compile(
            r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?!\d)"
        ),
    ),
]

_REDACTED = "[REDACTED]"


def _has_pii(text: str) -> bool:
    return any(pat.search(text) is not None for _, pat in _PII_PATTERNS)


def _redact(text: str) -> str:
    for _, pat in _PII_PATTERNS:
        text = pat.sub(_REDACTED, text)
    return text


class PIIFilter:
    """Detect and optionally redact PII in chunk text before indexing.

    Args:
        redact: When ``True``, replace PII spans with ``[REDACTED]``.
            When ``False``, drop chunks that contain any PII.
    """

    def __init__(self, redact: bool = True) -> None:
        self._redact = redact

    def filter(self, chunks: list[ChunkWithEmbedding]) -> list[ChunkWithEmbedding]:
        """Return *chunks* with PII handled according to the configured policy.

        Args:
            chunks: Embedded chunks to inspect.

        Returns:
            Filtered/redacted list of chunks safe to index.
        """
        result: list[ChunkWithEmbedding] = []
        for chunk in chunks:
            if not _has_pii(chunk.text):
                result.append(chunk)
                continue
            if self._redact:
                clean_text = _redact(chunk.text)
                redacted = chunk.model_copy(update={"text": clean_text})
                logger.debug(f"PIIFilter: redacted PII in {chunk.path}:{chunk.chunk_index}")
                result.append(redacted)
            else:
                logger.debug(f"PIIFilter: dropped chunk with PII {chunk.path}:{chunk.chunk_index}")
        return result
