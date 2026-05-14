"""PDF parser backed by pypdf with optional OCR fallback.

OCR is only invoked when **all three** conditions are met:
1. ``ocr_config.enabled is True``
2. ``ocr_config.trigger == "empty_text_only"``
3. The native text extraction returned fewer than 50 characters.

Privacy invariant: page text is never logged - only page counts and path.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from memorymesh.core.models import ParsedDocument
from memorymesh.parsing.base import Parser

# Minimum character count for native PDF text to suppress OCR.
_OCR_MIN_CHARS = 50

if TYPE_CHECKING:
    from memorymesh.core.models import OcrConfig


class PdfParser(Parser):
    """Parser for PDF files using pypdf for native text extraction.

    Args:
        ocr_config: OCR settings from the main config.  When ``None`` OCR is
            always disabled.
    """

    def __init__(self, ocr_config: OcrConfig | None = None) -> None:
        self._ocr_config = ocr_config

    @property
    def supported_extensions(self) -> frozenset[str]:
        """File extensions handled by this parser: PDF files."""
        return frozenset({".pdf"})

    def parse(self, path: Path) -> ParsedDocument:
        """Extract text from a PDF.

        Tries native pypdf extraction first.  Falls back to OCR only when
        enabled in config and the native text is near-empty.

        Args:
            path: Absolute path to the PDF file.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument` with extracted text.
        """
        try:
            import pypdf  # deferred - heavy import
        except ImportError:
            msg = "pypdf is not installed; run 'uv add pypdf'"
            logger.error(msg)
            return ParsedDocument(path=path, text="", file_type=".pdf", metadata={"error": msg})

        text = ""
        page_count = 0
        meta: dict[str, object] = {}

        try:
            reader = pypdf.PdfReader(str(path))
            page_count = len(reader.pages)
            parts: list[str] = []
            for page in reader.pages:
                extracted = page.extract_text() or ""
                parts.append(extracted)
            text = "\n".join(parts).strip()
            meta["page_count"] = page_count
            logger.debug(f"PdfParser: {path.name!r} - {page_count} pages, {len(text)} chars")
        except Exception as exc:
            msg = f"pypdf failed to read {path}: {exc}"
            logger.warning(msg)
            meta["error"] = msg
            meta["page_count"] = page_count
            return ParsedDocument(path=path, text="", file_type=".pdf", metadata=meta)

        if self._should_ocr(text):
            text = self._run_ocr(path, text, meta)

        return ParsedDocument(path=path, text=text, file_type=".pdf", metadata=meta)

    def _should_ocr(self, text: str) -> bool:
        """Return True only when OCR is enabled, trigger matches, and text is short."""
        cfg = self._ocr_config
        return (
            cfg is not None
            and cfg.enabled
            and cfg.trigger == "empty_text_only"
            and len(text) < _OCR_MIN_CHARS
        )

    def _run_ocr(self, path: Path, existing_text: str, meta: dict[str, object]) -> str:
        """Attempt OCR via pytesseract; return *existing_text* on any failure."""
        cfg = self._ocr_config
        if cfg is not None:
            try:
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > cfg.max_file_size_mb:
                    logger.info(
                        f"PdfParser: skipping OCR for {path.name!r} "
                        f"({size_mb:.1f} MB > {cfg.max_file_size_mb} MB limit)"
                    )
                    return existing_text
            except OSError as exc:
                logger.debug(f"PdfParser: could not stat {path.name!r} before OCR: {exc}")

        try:
            import pytesseract  # type: ignore[import-not-found]
            from pdf2image import convert_from_path  # type: ignore[import-not-found]
        except ImportError as exc:
            logger.warning(
                f"OCR dependencies not available ({exc}); skipping OCR for {path.name!r}"
            )
            return existing_text

        cfg = self._ocr_config
        langs = "+".join(cfg.languages) if cfg and cfg.languages else "eng"
        try:
            images = convert_from_path(str(path))
            parts = [pytesseract.image_to_string(img, lang=langs) for img in images]
            ocr_text = "\n".join(parts).strip()
            logger.info(f"PdfParser: OCR produced {len(ocr_text)} chars for {path.name!r}")
            meta["ocr"] = True
            return ocr_text
        except Exception as exc:
            logger.warning(f"OCR failed for {path.name!r}: {exc}")
            return existing_text
