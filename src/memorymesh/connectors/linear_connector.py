"""Linear issue tracker connector for MemoryMesh.

Fetches issues assigned to the authenticated user from Linear's GraphQL
API and yields one :class:`~memorymesh.core.models.ParsedDocument` per
issue.

API reference
-------------
``https://api.linear.app/graphql``

Authentication
--------------
Linear uses a personal API key passed directly as the ``Authorization``
header value - **no** ``Bearer`` prefix.  Generate a key at
https://linear.app/settings/api.

Features
--------
* **Cursor pagination** - iterates via ``pageInfo.hasNextPage`` /
  ``endCursor`` until all matching issues are fetched.
* **Date filtering** - only issues whose ``updatedAt`` falls within
  ``days_past`` are yielded.
* **Team filtering** - optionally restrict to specific team IDs.
* **State filtering** - optionally restrict to specific state names.
* **Comments** - up to 50 comments per issue are included in the
  document text.

Usage
-----
::

    connector = LinearConnector(LinearConfig(
        api_key=SecretStr("lin_api_..."),
        team_ids=["TEAM-UUID"],
        states=["In Progress", "Done"],
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

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._http import api_post
from memorymesh.core.models import ParsedDocument

_API_URL = "https://api.linear.app/graphql"

_ISSUES_QUERY = """
query($after: String) {
  issues(
    filter: { assignee: { isMe: { eq: true } } }
    first: 50
    after: $after
    orderBy: updatedAt
  ) {
    nodes {
      id identifier title description
      state { name }
      priority
      createdAt updatedAt
      team { id name }
      comments(first: 50) {
        nodes { body createdAt user { name } }
      }
    }
    pageInfo { hasNextPage endCursor }
  }
}
"""


class LinearConfig(BaseModel):
    """Configuration for a Linear issue tracker source.

    Args:
        api_key: Linear personal API key (no ``Bearer`` prefix needed).
            Generate one at https://linear.app/settings/api.
        team_ids: Restrict to issues belonging to these team UUIDs.  An
            empty list includes all teams.
        states: Restrict to these state names (case-insensitive).  An
            empty list includes all states.
        days_past: Only include issues updated within this many days.  0
            = no cutoff.
        source_name: Name used in the MemoryMesh source registry.
    """

    api_key: SecretStr
    team_ids: list[str] = []
    states: list[str] = []
    days_past: int = 180
    source_name: str = "linear"


class LinearConnector:
    """Fetches Linear issues and yields one ParsedDocument per issue.

    Args:
        config: API key, team/state filters, and source settings.
    """

    def __init__(self, config: LinearConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Paginate the Linear GraphQL API and yield issue documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            issue, with ``file_type=".linear"`` and metadata containing
            ``id``, ``identifier``, ``title``, ``state``, ``team``,
            ``priority``, ``created_at``, and ``updated_at``.
        """
        headers = {
            "Authorization": self._cfg.api_key.get_secret_value(),
            "Content-Type": "application/json",
        }
        cutoff = self._cutoff()
        team_ids = set(self._cfg.team_ids)
        states = {s.lower() for s in self._cfg.states}

        after: str | None = None
        total = 0

        while True:
            body: dict[str, Any] = {
                "query": _ISSUES_QUERY,
                "variables": {"after": after},
            }
            data = api_post(_API_URL, headers, body)
            if not isinstance(data, dict):
                break

            issues_data = data.get("data", {}).get("issues", {})
            nodes: list[dict[str, Any]] = issues_data.get("nodes", [])
            page_info: dict[str, Any] = issues_data.get("pageInfo", {})

            for node in nodes:
                updated_at = node.get("updatedAt", "")
                if cutoff is not None and updated_at:
                    try:
                        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                        if dt < cutoff:
                            continue
                    except ValueError as exc:
                        logger.debug(f"LinearConnector: ignoring unparsable timestamp: {exc}")

                team = node.get("team") or {}
                if team_ids and team.get("id") not in team_ids:
                    continue

                state_name = (node.get("state") or {}).get("name", "")
                if states and state_name.lower() not in states:
                    continue

                doc = self._build_doc(node)
                if doc is not None:
                    yield doc
                    total += 1

            if not page_info.get("hasNextPage"):
                break
            after = page_info.get("endCursor")

        logger.info(f"LinearConnector: yielded {total} issue(s)")

    def _cutoff(self) -> datetime | None:
        """Return the UTC cutoff datetime for ``updatedAt`` filtering.

        Returns:
            Aware :class:`datetime`, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _build_doc(self, node: dict[str, Any]) -> ParsedDocument | None:
        """Convert a Linear issue GraphQL node to a ParsedDocument.

        Args:
            node: Raw Linear issue node from the GraphQL response.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the ``identifier`` field is missing.
        """
        identifier = node.get("identifier")
        if not identifier:
            return None

        title = node.get("title", "")
        description = node.get("description") or ""
        team_obj = node.get("team") or {}
        team = team_obj.get("name", "")
        state_name = (node.get("state") or {}).get("name", "")
        priority = node.get("priority", 0)
        created_at = node.get("createdAt", "")
        updated_at = node.get("updatedAt", "")
        issue_id = node.get("id", "")

        comments_nodes: list[dict[str, Any]] = (node.get("comments") or {}).get("nodes", [])
        comment_lines: list[str] = []
        for c in comments_nodes:
            user = (c.get("user") or {}).get("name", "Unknown")
            created = c.get("createdAt", "")
            body = c.get("body", "")
            comment_lines.append(f"{created} {user}: {body}")

        text_parts = [
            f"{identifier}: {title}",
            f"Team: {team}",
            f"State: {state_name}",
            f"Priority: {priority}",
        ]
        if description:
            text_parts.append(f"\n{description}")
        if comment_lines:
            text_parts.append("\n--- Comments ---")
            text_parts.extend(comment_lines)

        return ParsedDocument(
            path=Path(f"linear://{identifier}.linear"),
            text="\n".join(text_parts),
            file_type=".linear",
            encoding="utf-8",
            metadata={
                "id": issue_id,
                "identifier": identifier,
                "title": title,
                "state": state_name,
                "team": team,
                "priority": priority,
                "created_at": created_at,
                "updated_at": updated_at,
                "source": self._cfg.source_name,
            },
        )
