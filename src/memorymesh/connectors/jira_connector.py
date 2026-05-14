"""Jira issue tracker connector for MemoryMesh.

Fetches issues from Jira Cloud via the REST API v3 and yields one
:class:`~memorymesh.core.models.ParsedDocument` per issue.

API reference
-------------
``{base_url}/rest/api/3/search``

Authentication
--------------
HTTP Basic authentication with ``email:api_token`` (base64-encoded).
Generate an API token at https://id.atlassian.com/manage-profile/security/api-tokens.

Features
--------
* **Offset pagination** - iterates via ``startAt`` / ``maxResults`` / ``total``
  until all matching issues are fetched.
* **Project filtering** - optionally restrict to specific project keys via JQL.
* **Status filtering** - optionally restrict to specific status names.
* **Date filtering** - only issues updated within ``days_past`` are included.
* **ADF text extraction** - Atlassian Document Format description bodies are
  recursively flattened to plain text.
* **Comments** - up to 50 comments per issue included in document text.

Usage
-----
::

    connector = JiraConnector(JiraConfig(
        base_url="https://myorg.atlassian.net",
        email="me@example.com",
        api_token=SecretStr("my-token"),
        project_keys=["ENG", "OPS"],
        days_past=90,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._auth import basic_header
from memorymesh.connectors._http import api_get
from memorymesh.core.models import ParsedDocument

_PAGE_SIZE = 50


class JiraConfig(BaseModel):
    """Configuration for a Jira issue source.

    Args:
        base_url: Jira Cloud base URL, e.g. ``https://myorg.atlassian.net``.
        email: Atlassian account email address.
        api_token: Atlassian API token.
        project_keys: Restrict to these project keys.  Empty = all projects.
        statuses: Restrict to these status names (case-insensitive).
            Empty = all statuses.
        days_past: Only include issues updated within this many days.
            0 = no cutoff.
        max_issues: Maximum total issues to fetch.  0 = no limit.
        source_name: Name used in the MemoryMesh source registry.
    """

    base_url: str
    email: str
    api_token: SecretStr
    project_keys: list[str] = []
    statuses: list[str] = []
    days_past: int = 180
    max_issues: int = 1000
    source_name: str = "jira"


class JiraConnector:
    """Fetches Jira issues and yields one ParsedDocument per issue.

    Args:
        config: Jira credentials, project/status filters, and source settings.
    """

    def __init__(self, config: JiraConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Paginate the Jira search API and yield issue documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            issue, with ``file_type=".jira"`` and metadata containing
            ``key``, ``summary``, ``status``, ``project``, ``priority``,
            ``assignee``, ``created``, and ``updated``.
        """
        headers = {
            "Authorization": self._auth_header(),
            "Accept": "application/json",
        }
        jql = self._build_jql()
        start_at = 0
        total_yielded = 0
        limit = self._cfg.max_issues

        while True:
            if limit > 0 and total_yielded >= limit:
                break

            params = urlencode(
                {
                    "jql": jql,
                    "maxResults": _PAGE_SIZE,
                    "startAt": start_at,
                    "fields": (
                        "summary,status,project,priority,assignee,"
                        "created,updated,description,comment"
                    ),
                }
            )
            url = urljoin(
                self._cfg.base_url.rstrip("/") + "/",
                f"rest/api/3/search?{params}",
            )
            data = api_get(url, headers)
            if not isinstance(data, dict):
                break

            issues: list[dict[str, Any]] = data.get("issues", [])
            if not issues:
                break

            for issue in issues:
                if limit > 0 and total_yielded >= limit:
                    break
                doc = self._build_doc(issue)
                if doc is not None:
                    yield doc
                    total_yielded += 1

            start_at += _PAGE_SIZE
            total = data.get("total", 0)
            if start_at >= total:
                break

        logger.info(f"JiraConnector: yielded {total_yielded} issue(s)")

    def _auth_header(self) -> str:
        """Build the HTTP Basic Authorization header value.

        Returns:
            ``Basic <base64(email:token)>`` string.
        """
        return basic_header(
            self._cfg.email,
            self._cfg.api_token.get_secret_value(),
        )["Authorization"]

    def _build_jql(self) -> str:
        """Construct the JQL query from config filters.

        Returns:
            JQL string with project, status, and date constraints.
        """
        clauses: list[str] = []

        if self._cfg.project_keys:
            keys = ", ".join(f'"{k}"' for k in self._cfg.project_keys)
            clauses.append(f"project in ({keys})")

        if self._cfg.statuses:
            statuses = ", ".join(f'"{s}"' for s in self._cfg.statuses)
            clauses.append(f"status in ({statuses})")

        if self._cfg.days_past > 0:
            clauses.append(f"updated >= -{self._cfg.days_past}d")

        clauses.append("ORDER BY updated DESC")
        return " AND ".join([*clauses[:-1], clauses[-1]])

    def _extract_adf_text(self, node: Any) -> str:
        """Recursively extract plain text from an ADF node.

        Args:
            node: ADF node (dict) or primitive value.

        Returns:
            Concatenated plain text from all text leaf nodes.
        """
        if isinstance(node, str):
            return node
        if not isinstance(node, dict):
            return ""
        parts: list[str] = []
        if node.get("type") == "text":
            parts.append(node.get("text", ""))
        for child in node.get("content", []):
            parts.append(self._extract_adf_text(child))
        return " ".join(p for p in parts if p)

    def _build_doc(self, issue: dict[str, Any]) -> ParsedDocument | None:
        """Convert a Jira issue API response to a ParsedDocument.

        Args:
            issue: Raw Jira issue object from the REST API.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the issue ``key`` is missing.
        """
        key = issue.get("key")
        if not key:
            return None

        fields: dict[str, Any] = issue.get("fields") or {}
        summary = fields.get("summary", "")
        status = (fields.get("status") or {}).get("name", "")
        project = (fields.get("project") or {}).get("name", "")
        priority = (fields.get("priority") or {}).get("name", "")
        assignee_obj = fields.get("assignee") or {}
        assignee = assignee_obj.get("displayName", "")
        created = fields.get("created", "")
        updated = fields.get("updated", "")

        description_adf = fields.get("description")
        description = ""
        if isinstance(description_adf, dict):
            description = self._extract_adf_text(description_adf)
        elif isinstance(description_adf, str):
            description = description_adf

        comments_obj = fields.get("comment") or {}
        comment_list: list[dict[str, Any]] = comments_obj.get("comments", [])
        comment_lines: list[str] = []
        for c in comment_list[:50]:
            author = (c.get("author") or {}).get("displayName", "Unknown")
            created_c = c.get("created", "")
            body_adf = c.get("body")
            if isinstance(body_adf, dict):
                body = self._extract_adf_text(body_adf)
            elif isinstance(body_adf, str):
                body = body_adf
            else:
                body = ""
            comment_lines.append(f"{created_c} {author}: {body}")

        text_parts = [
            f"{key}: {summary}",
            f"Project: {project}",
            f"Status: {status}",
            f"Priority: {priority}",
            f"Assignee: {assignee}",
        ]
        if description:
            text_parts.append(f"\n{description}")
        if comment_lines:
            text_parts.append("\n--- Comments ---")
            text_parts.extend(comment_lines)

        return ParsedDocument(
            path=Path(f"jira://{key}.jira"),
            text="\n".join(text_parts),
            file_type=".jira",
            encoding="utf-8",
            metadata={
                "key": key,
                "summary": summary,
                "status": status,
                "project": project,
                "priority": priority,
                "assignee": assignee,
                "created": created,
                "updated": updated,
                "source": self._cfg.source_name,
            },
        )
