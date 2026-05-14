"""Pinterest data export connector for MemoryMesh.

Parses the Pinterest data export CSV (Settings -> Privacy and Data ->
Request Your Data) and yields one
:class:`~memorymesh.core.models.ParsedDocument` per board.

Export format
-------------
A single CSV file named ``pinterest-*.csv`` with columns:
``Pin Link``, ``Board``, ``Note``, ``Title``, ``Description``,
``Image Link``, ``Saved Date``, ``Source Link``.

Features
--------
* **Board grouping** - pins are grouped by the ``Board`` column; each
  board becomes one searchable document.
* **Multi-file** - if ``export_path`` is a directory, all matching
  ``pinterest-*.csv`` files are read.

Usage
-----
::

    connector = PinterestConnector(PinterestConfig(
        export_path=Path("~/Downloads/pinterest-2024-01-01.csv"),
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument


class PinterestConfig(BaseModel):
    """Configuration for a Pinterest data export source.

    Args:
        export_path: Path to a ``pinterest-*.csv`` file or a directory
            containing such files.
        source_name: Name used in the MemoryMesh source registry.
    """

    export_path: Path
    source_name: str = "pinterest"


class PinterestConnector:
    """Parses Pinterest CSV exports and yields per-board ParsedDocuments.

    Args:
        config: Export path and source settings.
    """

    def __init__(self, config: PinterestConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Read all CSV files, group by board, and yield one doc per board.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            board, with ``file_type=".pinterest"`` and metadata containing
            ``board``, ``pin_count``, ``oldest_pin``, and ``newest_pin``.
        """
        path = self._cfg.export_path.expanduser()

        if path.is_dir():
            files = sorted(path.glob("pinterest-*.csv"))
            if not files:
                files = sorted(path.glob("*.csv"))
        elif path.is_file():
            files = [path]
        else:
            logger.warning(f"PinterestConnector: path not found: {path}")
            return

        if not files:
            logger.warning(f"PinterestConnector: no CSV files found in {path}")
            return

        # board -> list of pin dicts
        by_board: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
        for csv_file in files:
            self._read_file(csv_file, by_board)

        total = 0
        source = self._cfg.source_name
        for board, pins in sorted(by_board.items()):
            blocks: list[str] = []
            dates = [p.get("Saved Date", "") for p in pins if p.get("Saved Date")]
            for pin in pins:
                block = (
                    f"Title: {pin.get('Title', '')}\n"
                    f"Description: {pin.get('Description', '')}\n"
                    f"Note: {pin.get('Note', '')}\n"
                    f"Source: {pin.get('Source Link', '')}\n"
                    f"Saved: {pin.get('Saved Date', '')}"
                )
                blocks.append(block)

            safe_board = board.replace("/", "_").replace(" ", "_")
            yield ParsedDocument(
                path=Path(f"pinterest://{safe_board}.pinterest"),
                text="\n\n---\n\n".join(blocks),
                file_type=".pinterest",
                encoding="utf-8",
                metadata={
                    "board": board,
                    "pin_count": len(pins),
                    "oldest_pin": min(dates) if dates else "",
                    "newest_pin": max(dates) if dates else "",
                    "source": source,
                },
            )
            total += 1

        logger.info(f"PinterestConnector: yielded {total} board document(s)")

    def _read_file(
        self,
        csv_file: Path,
        by_board: defaultdict[str, list[dict[str, str]]],
    ) -> None:
        """Parse one CSV file and append pins to *by_board*.

        Args:
            csv_file: Path to the Pinterest export CSV.
            by_board: Accumulator mapping board name -> pin list.
        """
        try:
            text = csv_file.read_text(encoding="utf-8-sig")
        except OSError as exc:
            logger.warning(f"PinterestConnector: cannot read {csv_file}: {exc}")
            return
        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames is None:
            logger.warning(f"PinterestConnector: no header in {csv_file}")
            return
        for row in reader:
            board = (row.get("Board") or "Unknown Board").strip()
            by_board[board].append(dict(row))
