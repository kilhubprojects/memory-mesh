"""GitLab connector for MemoryMesh.

Fetches issues and merge requests from GitLab projects via the REST API
and yields one :class:`~memorymesh.core.models.ParsedDocument` per item.

API reference
-------------
``https://gitlab.com/api/v4`` (or self-hosted base URL)

Authentication
--------------
Personal access token passed as the ``PRIVATE-TOKEN`` header.
Generate one at https://gitlab.com/-/profile/personal_access_tokens.

Features
--------
* **Issues and MRs** - both resource types are fetched for each project.
* **Offset pagination** - uses ``page`` / ``per_page`` query params.
* **Project filtering** - fetches only the configured project IDs or paths.
* **Date filtering** - only items updated within ``days_past`` are yielded.
* **State filtering** - optionally restrict to ``opened``, ``closed``, or
  ``merged`` states.

Usage
-----
::

    connector = GitLabConnector(GitLabConfig(
        api_token=SecretStr("glpat-..."),
        projects=["mygroup/myrepo"],
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
from urllib.parse import quote, urlencode

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._http import api_get
from memorymesh.core.models import ParsedDocument

_PAGE_SIZE = 50


class GitLabConfig(BaseModel):
    """Configuration for a GitLab source.

    Args:
        api_token: GitLab personal access token.
        base_url: GitLab instance URL.  Defaults to ``https://gitlab.com``.
        projects: List of project IDs or ``namespace/project`` paths.
            Empty = skip (no projects fetched if empty).
        fetch_issues: Whether to fetch issues.
        fetch_mrs: Whether to fetch merge requests.
        state: Filter by state: ``"opened"``, ``"closed"``, ``"merged"``,
            or ``"all"`` (default).
        days_past: Only include items updated within this many days.
            0 = no cutoff.
        source_name: Name used in the MemoryMesh source registry.
    """

    api_token: SecretStr
    base_url: str = "https://gitlab.com"
    projects: list[str] = []
    fetch_issues: bool = True
    fetch_mrs: bool = True
    state: str = "all"
    days_past: int = 180
    source_name: str = "gitlab"


class GitLabConnector:
    """Fetches GitLab issues and MRs and yields one ParsedDocument each.

    Args:
        config: GitLab credentials, project list, and source settings.
    """

    def __init__(self, config: GitLabConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Iterate projects and fetch issues and/or MRs.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            issue or MR, with ``file_type=".gitlab"`` and metadata
            containing ``iid``, ``title``, ``state``, ``type``,
            ``project``, ``created_at``, and ``updated_at``.
        """
        headers = {"PRIVATE-TOKEN": self._cfg.api_token.get_secret_value()}
        cutoff = self._cutoff()
        total = 0

        for project in self._cfg.projects:
            enc_project = quote(str(project), safe="")
            if self._cfg.fetch_issues:
                for doc in self._fetch_items(headers, enc_project, "issues", cutoff):
                    yield doc
                    total += 1
            if self._cfg.fetch_mrs:
                for doc in self._fetch_items(headers, enc_project, "merge_requests", cutoff):
                    yield doc
                    total += 1

        logger.info(f"GitLabConnector: yielded {total} item(s)")

    def _cutoff(self) -> datetime | None:
        """Return the UTC cutoff datetime.

        Returns:
            Aware :class:`datetime`, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _fetch_items(
        self,
        headers: dict[str, str],
        enc_project: str,
        resource: str,
        cutoff: datetime | None,
    ) -> Iterator[ParsedDocument]:
        """Paginate a project resource (issues or MRs) and yield documents.

        Args:
            headers: Auth headers.
            enc_project: URL-encoded project path or ID.
            resource: ``"issues"`` or ``"merge_requests"``.
            cutoff: UTC datetime cutoff; skip items updated before this.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per item.
        """
        base = self._cfg.base_url.rstrip("/")
        page = 1
        item_type = "issue" if resource == "issues" else "mr"

        while True:
            params = urlencode(
                {
                    "state": self._cfg.state,
                    "per_page": _PAGE_SIZE,
                    "page": page,
                    "order_by": "updated_at",
                    "sort": "desc",
                }
            )
            url = f"{base}/api/v4/projects/{enc_project}/{resource}?{params}"
            data = api_get(url, headers)
            if not isinstance(data, list) or not data:
                break

            for item in data:
                updated_at = item.get("updated_at", "")
                if cutoff and updated_at:
                    try:
                        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                        if dt < cutoff:
                            continue
                    except ValueError as exc:
                        logger.debug(f"GitLabConnector: ignoring unparsable timestamp: {exc}")

                doc = self._build_doc(item, item_type, enc_project)
                if doc is not None:
                    yield doc

            if len(data) < _PAGE_SIZE:
                break
            page += 1

    def _build_doc(
        self,
        item: dict[str, Any],
        item_type: str,
        project: str,
    ) -> ParsedDocument | None:
        """Convert a GitLab issue or MR to a ParsedDocument.

        Args:
            item: Raw GitLab API item dict.
            item_type: ``"issue"`` or ``"mr"``.
            project: URL-encoded project identifier.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the ``iid`` field is missing.
        """
        iid = item.get("iid")
        if iid is None:
            return None

        title = item.get("title", "")
        description = item.get("description") or ""
        state = item.get("state", "")
        created_at = item.get("created_at", "")
        updated_at = item.get("updated_at", "")

        text_parts = [
            f"#{iid}: {title}",
            f"Type: {item_type}",
            f"State: {state}",
            f"Project: {project}",
        ]
        if description:
            text_parts.append(f"\n{description}")

        return ParsedDocument(
            path=Path(f"gitlab://{project}/#{iid}.gitlab"),
            text="\n".join(text_parts),
            file_type=".gitlab",
            encoding="utf-8",
            metadata={
                "iid": iid,
                "title": title,
                "state": state,
                "type": item_type,
                "project": project,
                "created_at": created_at,
                "updated_at": updated_at,
                "source": self._cfg.source_name,
            },
        )
