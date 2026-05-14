"""Bank/credit card CSV transaction connector for MemoryMesh.

Parses transaction CSV exports and yields one
:class:`~memorymesh.core.models.ParsedDocument` per calendar month,
containing a plain-text ledger of all transactions for that period.

Features
--------
* **Auto-detection** - tries common column name conventions for date,
  description, and amount fields if not explicitly configured.
* **Multi-file** - if ``csv_path`` is a directory, all ``*.csv`` files
  are read.
* **Flexible date parsing** - tries ``YYYY-MM-DD``, ``DD/MM/YYYY``,
  ``MM/DD/YYYY``, and ``DD-MM-YYYY`` automatically.
* **Brazilian number format** - amounts like ``"1.234,56"`` are correctly
  parsed to ``1234.56``.
* **Monthly grouping** - transactions are grouped by calendar month into
  a single searchable document.
* **Privacy** - individual transaction descriptions are never logged at
  INFO level; only counts and totals are.

Usage
-----
::

    connector = BankCSVConnector(BankCSVConfig(
        csv_path=Path("~/Downloads/nubank_2024.csv"),
        currency="BRL",
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from memorymesh.core.models import ParsedDocument

_DATE_CANDIDATES: list[str] = [
    "Date",
    "Data",
    "Transaction Date",
    "Data Lançamento",
    "Posting Date",
    "date",
    "data",
    "transaction_date",
    "posting_date",
]
_DESC_CANDIDATES: list[str] = [
    "Description",
    "Descrição",
    "Merchant",
    "Memo",
    "Histórico",
    "description",
    "descrição",
    "merchant",
    "memo",
    "histórico",
]
_AMOUNT_CANDIDATES: list[str] = [
    "Amount",
    "Valor",
    "Debit",
    "Credit",
    "Transaction Amount",
    "amount",
    "valor",
    "debit",
    "credit",
    "transaction_amount",
]

_DATE_FORMATS: list[str] = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
]


class BankCSVConfig(BaseModel):
    """Configuration for a bank/credit card CSV source.

    Args:
        csv_path: Path to a single CSV file or a directory of CSV files.
        date_column: Exact column name for the transaction date.  Empty
            string triggers auto-detection.
        description_column: Exact column name for the description.  Empty
            string triggers auto-detection.
        amount_column: Exact column name for the transaction amount.
            Empty string triggers auto-detection.
        currency: ISO 4217 currency code shown in document text.
        source_name: Name used in the MemoryMesh source registry.
    """

    csv_path: Path
    date_column: str = ""
    description_column: str = ""
    amount_column: str = ""
    currency: str = "BRL"
    source_name: str = "bank"


def _detect_column(fieldnames: list[str], candidates: list[str]) -> str | None:
    """Return the first fieldname matching any candidate (case-insensitive).

    Args:
        fieldnames: Actual column names from the CSV header.
        candidates: Ordered list of preferred column name variants.

    Returns:
        Matched fieldname or ``None`` if no match is found.
    """
    lower_map = {f.lower(): f for f in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _parse_date(raw: str) -> datetime | None:
    """Try multiple date formats and return a datetime, or None on failure.

    Args:
        raw: Raw date string from the CSV cell.

    Returns:
        :class:`datetime` without timezone, or ``None``.
    """
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _parse_amount(raw: str) -> float:
    """Parse a numeric amount string, handling comma decimal separators.

    Handles both European (``"1.234,56"``) and US/ISO (``"1234.56"``)
    number formats.

    Args:
        raw: Raw amount string.

    Returns:
        Float value, or 0.0 on parse error.
    """
    cleaned = raw.strip().replace(" ", "")
    if "," in cleaned and "." in cleaned:
        # European format: 1.234,56 -> 1234.56
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


class BankCSVConnector:
    """Reads bank CSVs and yields per-month ParsedDocuments.

    Args:
        config: CSV path, optional column names, and currency settings.
    """

    def __init__(self, config: BankCSVConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Read all CSV files and yield one document per calendar month.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            (source_name, year_month), with ``file_type=".bank"`` and
            metadata containing ``year_month``, ``transaction_count``,
            ``total_debit``, ``total_credit``, and ``currency``.
        """
        csv_path = self._cfg.csv_path.expanduser()

        if csv_path.is_dir():
            files = sorted(csv_path.glob("*.csv"))
        elif csv_path.is_file():
            files = [csv_path]
        else:
            logger.warning(f"BankCSVConnector: path not found: {csv_path}")
            return

        if not files:
            logger.warning(f"BankCSVConnector: no CSV files in {csv_path}")
            return

        # year_month -> list of (date_str, description, amount)
        by_month: defaultdict[str, list[tuple[str, str, float]]] = defaultdict(list)

        for csv_file in files:
            self._read_file(csv_file, by_month)

        total_docs = 0
        for year_month, txns in sorted(by_month.items()):
            doc = self._build_document(year_month, txns)
            if doc is not None:
                yield doc
                total_docs += 1

        logger.info(f"BankCSVConnector: yielded {total_docs} month document(s)")

    def _read_file(
        self,
        csv_file: Path,
        by_month: defaultdict[str, list[tuple[str, str, float]]],
    ) -> None:
        """Parse one CSV file and append rows to *by_month*.

        Args:
            csv_file: Path to the CSV file.
            by_month: Accumulator mapping year_month -> transaction list.
        """
        try:
            text = csv_file.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = csv_file.read_text(encoding="latin-1")
            except OSError as exc:
                logger.warning(f"BankCSVConnector: cannot read {csv_file}: {exc}")
                return
        except OSError as exc:
            logger.warning(f"BankCSVConnector: cannot read {csv_file}: {exc}")
            return

        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames is None:
            logger.warning(f"BankCSVConnector: no header row in {csv_file}")
            return

        fieldnames = list(reader.fieldnames)
        date_col = self._cfg.date_column or _detect_column(fieldnames, _DATE_CANDIDATES)
        desc_col = self._cfg.description_column or _detect_column(fieldnames, _DESC_CANDIDATES)
        amount_col = self._cfg.amount_column or _detect_column(fieldnames, _AMOUNT_CANDIDATES)

        if not date_col or not amount_col:
            logger.warning(
                f"BankCSVConnector: cannot detect date/amount columns in"
                f" {csv_file} (fields: {fieldnames})"
            )
            return

        for row in reader:
            date_raw = row.get(date_col, "").strip()
            dt = _parse_date(date_raw)
            if dt is None:
                continue
            year_month = dt.strftime("%Y-%m")
            description = row.get(desc_col, "").strip() if desc_col else ""
            amount = _parse_amount(row.get(amount_col, "0").strip())
            by_month[year_month].append((dt.strftime("%Y-%m-%d"), description, amount))

    def _build_document(
        self,
        year_month: str,
        txns: list[tuple[str, str, float]],
    ) -> ParsedDocument | None:
        """Assemble a ParsedDocument for one calendar month.

        Args:
            year_month: ``YYYY-MM`` string.
            txns: List of ``(date_str, description, amount)`` tuples.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the transaction list is empty.
        """
        if not txns:
            return None

        currency = self._cfg.currency
        source = self._cfg.source_name
        txns_sorted = sorted(txns, key=lambda t: t[0])

        total_debit = sum(a for _, _, a in txns_sorted if a < 0)
        total_credit = sum(a for _, _, a in txns_sorted if a >= 0)
        n = len(txns_sorted)

        lines = [
            f"Bank transactions - {year_month} ({n} transactions)",
            "",
        ]
        for date_str, desc, amount in txns_sorted:
            lines.append(f"{date_str} | {desc} | {amount:+.2f} {currency}")

        return ParsedDocument(
            path=Path(f"bank://{source}/{year_month}.bank"),
            text="\n".join(lines),
            file_type=".bank",
            encoding="utf-8",
            metadata={
                "year_month": year_month,
                "transaction_count": n,
                "total_debit": round(total_debit, 2),
                "total_credit": round(total_credit, 2),
                "currency": currency,
                "source": source,
            },
        )
