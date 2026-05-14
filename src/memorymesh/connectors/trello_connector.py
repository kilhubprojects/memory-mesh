"""Trello board connector for MemoryMesh.

Fetches all boards, lists, and cards accessible to the authenticated user
from the Trello API and yields one
:class:`~memorymesh.core.models.ParsedDocument` per card.

API reference
-------------
``https://api.trello.com/1``

Authentication
--------------
Trello uses ``key`` + ``token`` query parameters.
Generate credentials at https://trello.com/power-ups/admin.

Features
--------
* **All boards** - fetches every board the token has access to.
* **Card metadata** - includes board name, list name, labels, due date,
  and description in each document.
* **Date filtering** - cards not updated within ``days_past`` are skipped.
* **Closed board/card filtering** - archived items skipped by default.

Usage
-----
::

    connector = TrelloConnector(TrelloConfig(
        api_key=SecretStr("your-api-key"),
        token=SecretStr("your-token"),
        days_past=90,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._http import api_get
from memorymesh.core.models import ParsedDocument

_BASE = "https://api.trello.com/1"


class TrelloConfig(BaseModel):
    """Configuration for a Trello source.

    Args:
        api_key: Trello API key.
        token: Trello user token.
        board_ids: Restrict to these board IDs.  Empty = all boards.
        include_closed: Whether to include archived boards and cards.
        days_past: Only include cards updated within this many days.
            0 = no cutoff.
        source_name: Name used in the MemoryMesh source registry.
    """

    api_key: SecretStr
    token: SecretStr
    board_ids: list[str] = []
    include_closed: bool = False
    days_past: int = 180
    source_name: str = "trello"


class TrelloConnector:
    """Fetches Trello cards and yields one ParsedDocument per card.

    Args:
        config: Trello API credentials and source settings.
    """

    def __init__(self, config: TrelloConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Enumerate boards and cards, yielding one document per card.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            card, with ``file_type=".trello"`` and metadata containing
            ``id``, ``name``, ``board``, ``list_name``, ``labels``,
            ``due``, and ``date_last_activity``.
        """
        auth = self._auth_params()
        cutoff = self._cutoff()
        total = 0

        board_ids = self._cfg.board_ids or self._all_board_ids(auth)

        for board_id in board_ids:
            board_data = self._fetch_board(auth, board_id)
            if board_data is None:
                continue
            board_name = board_data.get("name", board_id)

            lists_data = self._fetch_lists(auth, board_id)
            list_names: dict[str, str] = {
                lst["id"]: lst.get("name", "")
                for lst in lists_data
                if isinstance(lst, dict) and lst.get("id")
            }

            cards = self._fetch_cards(auth, board_id)
            for card in cards:
                if not self._cfg.include_closed and card.get("closed"):
                    continue

                updated = card.get("dateLastActivity", "")
                if cutoff and updated:
                    try:
                        dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                        if dt < cutoff:
                            continue
                    except ValueError as exc:
                        logger.debug(f"TrelloConnector: ignoring unparsable timestamp: {exc}")

                doc = self._build_doc(card, board_name, list_names)
                if doc is not None:
                    yield doc
                    total += 1

        logger.info(f"TrelloConnector: yielded {total} card(s)")

    def _auth_params(self) -> dict[str, str]:
        """Return the key/token query parameters.

        Returns:
            Dict with ``key`` and ``token`` entries.
        """
        return {
            "key": self._cfg.api_key.get_secret_value(),
            "token": self._cfg.token.get_secret_value(),
        }

    def _cutoff(self) -> datetime | None:
        """Return the UTC cutoff datetime.

        Returns:
            Aware :class:`datetime`, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _all_board_ids(self, auth: dict[str, str]) -> list[str]:
        """Fetch all board IDs for the authenticated member.

        Args:
            auth: Auth params dict.

        Returns:
            List of board ID strings.
        """
        url = f"{_BASE}/members/me/boards?{urlencode({**auth, 'fields': 'id,name'})}"
        data = api_get(url, {})
        if not isinstance(data, list):
            return []
        return [b["id"] for b in data if isinstance(b, dict) and b.get("id")]

    def _fetch_board(self, auth: dict[str, str], board_id: str) -> dict[str, Any] | None:
        """Fetch board metadata.

        Args:
            auth: Auth params dict.
            board_id: Trello board ID.

        Returns:
            Board dict or ``None`` on error.
        """
        url = f"{_BASE}/boards/{board_id}?{urlencode({**auth, 'fields': 'id,name,closed'})}"
        data = api_get(url, {})
        return data if isinstance(data, dict) else None

    def _fetch_lists(self, auth: dict[str, str], board_id: str) -> list[dict[str, Any]]:
        """Fetch lists on a board.

        Args:
            auth: Auth params dict.
            board_id: Trello board ID.

        Returns:
            List of list dicts.
        """
        url = f"{_BASE}/boards/{board_id}/lists?{urlencode(auth)}"
        data = api_get(url, {})
        return data if isinstance(data, list) else []

    def _fetch_cards(self, auth: dict[str, str], board_id: str) -> list[dict[str, Any]]:
        """Fetch all cards on a board.

        Args:
            auth: Auth params dict.
            board_id: Trello board ID.

        Returns:
            List of card dicts.
        """
        params = urlencode(
            {
                **auth,
                "fields": "id,name,desc,idList,labels,due,dateLastActivity,closed",
            }
        )
        url = f"{_BASE}/boards/{board_id}/cards?{params}"
        data = api_get(url, {})
        return data if isinstance(data, list) else []

    def _build_doc(
        self,
        card: dict[str, Any],
        board_name: str,
        list_names: dict[str, str],
    ) -> ParsedDocument | None:
        """Convert a Trello card to a ParsedDocument.

        Args:
            card: Raw Trello card dict.
            board_name: Name of the parent board.
            list_names: Mapping of list ID -> list name.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the card ``id`` is missing.
        """
        card_id = card.get("id")
        if not card_id:
            return None

        name = card.get("name", "")
        desc = card.get("desc", "")
        list_id = card.get("idList", "")
        list_name = list_names.get(list_id, list_id)
        due = card.get("due") or ""
        date_last = card.get("dateLastActivity", "")
        labels: list[str] = [
            lbl.get("name", "") for lbl in (card.get("labels") or []) if isinstance(lbl, dict)
        ]

        text_parts = [
            f"{name}",
            f"Board: {board_name}",
            f"List: {list_name}",
        ]
        if labels:
            text_parts.append(f"Labels: {', '.join(labels)}")
        if due:
            text_parts.append(f"Due: {due}")
        if desc:
            text_parts.append(f"\n{desc}")

        return ParsedDocument(
            path=Path(f"trello://{board_name}/{card_id}.trello"),
            text="\n".join(text_parts),
            file_type=".trello",
            encoding="utf-8",
            metadata={
                "id": card_id,
                "name": name,
                "board": board_name,
                "list_name": list_name,
                "labels": labels,
                "due": due,
                "date_last_activity": date_last,
                "source": self._cfg.source_name,
            },
        )
