"""Asana task connector for MemoryMesh.

Fetches tasks from Asana workspaces via the Asana REST API and yields one
:class:`~memorymesh.core.models.ParsedDocument` per task.

API reference
-------------
``https://app.asana.com/api/1.0``

Authentication
--------------
Personal access token passed as ``Authorization: Bearer {token}``.
Generate one at https://app.asana.com/0/my-apps.

Features
--------
* **Workspace enumeration** - fetches all accessible workspaces if none
  are configured.
* **Offset pagination** - uses ``next_page.offset`` for continuation.
* **Date filtering** - tasks not modified within ``days_past`` are skipped.
* **Assignee filter** - optionally restrict to tasks assigned to the
  authenticated user.

Usage
-----
::

    connector = AsanaConnector(AsanaConfig(
        access_token=SecretStr("1/12345:abcdef..."),
        assigned_to_me=True,
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

_BASE = "https://app.asana.com/api/1.0"
_TASK_FIELDS = "gid,name,notes,assignee,completed,due_on,modified_at,created_at,projects,workspace"


class AsanaConfig(BaseModel):
    """Configuration for an Asana task source.

    Args:
        access_token: Asana personal access token.
        workspace_gids: Restrict to these workspace GIDs.  Empty = all.
        project_gids: Restrict to tasks in these project GIDs.  Empty = all.
        assigned_to_me: Only fetch tasks assigned to the token owner.
        include_completed: Whether to include completed tasks.
        days_past: Only include tasks modified within this many days.
            0 = no cutoff.
        source_name: Name used in the MemoryMesh source registry.
    """

    access_token: SecretStr
    workspace_gids: list[str] = []
    project_gids: list[str] = []
    assigned_to_me: bool = True
    include_completed: bool = False
    days_past: int = 180
    source_name: str = "asana"


class AsanaConnector:
    """Fetches Asana tasks and yields one ParsedDocument per task.

    Args:
        config: Asana credentials, workspace/project filters, and source settings.
    """

    def __init__(self, config: AsanaConfig) -> None:
        self._cfg = config

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Enumerate workspaces or projects and yield task documents.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            task, with ``file_type=".asana"`` and metadata containing
            ``gid``, ``name``, ``completed``, ``due_on``, ``assignee``,
            ``modified_at``, and ``created_at``.
        """
        headers = {
            "Authorization": (f"Bearer {self._cfg.access_token.get_secret_value()}"),
        }
        cutoff = self._cutoff()
        total = 0

        if self._cfg.project_gids:
            for pgid in self._cfg.project_gids:
                for doc in self._tasks_for_project(headers, pgid, cutoff):
                    yield doc
                    total += 1
        else:
            workspace_gids = self._cfg.workspace_gids or self._all_workspace_gids(headers)
            for wgid in workspace_gids:
                for doc in self._tasks_for_workspace(headers, wgid, cutoff):
                    yield doc
                    total += 1

        logger.info(f"AsanaConnector: yielded {total} task(s)")

    def _cutoff(self) -> datetime | None:
        """Return the UTC cutoff datetime.

        Returns:
            Aware :class:`datetime`, or ``None`` if no cutoff.
        """
        if self._cfg.days_past <= 0:
            return None
        return datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)

    def _all_workspace_gids(self, headers: dict[str, str]) -> list[str]:
        """Fetch all accessible workspace GIDs.

        Args:
            headers: Auth headers.

        Returns:
            List of workspace GID strings.
        """
        data = api_get(f"{_BASE}/workspaces", headers)
        if not isinstance(data, dict):
            return []
        return [
            w.get("gid", "") for w in data.get("data", []) if isinstance(w, dict) and w.get("gid")
        ]

    def _tasks_for_workspace(
        self,
        headers: dict[str, str],
        workspace_gid: str,
        cutoff: datetime | None,
    ) -> Iterator[ParsedDocument]:
        """Paginate tasks for a workspace.

        Args:
            headers: Auth headers.
            workspace_gid: Workspace GID.
            cutoff: UTC datetime cutoff.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per task.
        """
        params: dict[str, Any] = {
            "workspace": workspace_gid,
            "opt_fields": _TASK_FIELDS,
            "limit": 100,
        }
        if self._cfg.assigned_to_me:
            params["assignee"] = "me"
        if not self._cfg.include_completed:
            params["completed_since"] = "now"

        yield from self._paginate_tasks(headers, f"{_BASE}/tasks", params, cutoff)

    def _tasks_for_project(
        self,
        headers: dict[str, str],
        project_gid: str,
        cutoff: datetime | None,
    ) -> Iterator[ParsedDocument]:
        """Paginate tasks for a project.

        Args:
            headers: Auth headers.
            project_gid: Project GID.
            cutoff: UTC datetime cutoff.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per task.
        """
        params: dict[str, Any] = {
            "opt_fields": _TASK_FIELDS,
            "limit": 100,
        }
        yield from self._paginate_tasks(
            headers,
            f"{_BASE}/projects/{project_gid}/tasks",
            params,
            cutoff,
        )

    def _paginate_tasks(
        self,
        headers: dict[str, str],
        base_url: str,
        params: dict[str, Any],
        cutoff: datetime | None,
    ) -> Iterator[ParsedDocument]:
        """Generic paginator for Asana task lists using offset cursor.

        Args:
            headers: Auth headers.
            base_url: API endpoint URL (without query params).
            params: Initial query parameters.
            cutoff: UTC datetime cutoff.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per task.
        """
        offset: str | None = None
        while True:
            current_params = {**params}
            if offset:
                current_params["offset"] = offset

            url = f"{base_url}?{urlencode(current_params)}"
            data = api_get(url, headers)
            if not isinstance(data, dict):
                break

            tasks: list[dict[str, Any]] = data.get("data", [])
            for task in tasks:
                modified_at = task.get("modified_at", "")
                if cutoff and modified_at:
                    try:
                        dt = datetime.fromisoformat(modified_at.replace("Z", "+00:00"))
                        if dt < cutoff:
                            continue
                    except ValueError as exc:
                        logger.debug(f"AsanaConnector: ignoring unparsable timestamp: {exc}")

                if not self._cfg.include_completed and task.get("completed"):
                    continue

                doc = self._build_doc(task)
                if doc is not None:
                    yield doc

            next_page = data.get("next_page")
            if not next_page:
                break
            offset = next_page.get("offset")
            if not offset:
                break

    def _build_doc(self, task: dict[str, Any]) -> ParsedDocument | None:
        """Convert an Asana task to a ParsedDocument.

        Args:
            task: Raw Asana task dict.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None``
            if the task GID is missing.
        """
        gid = task.get("gid")
        if not gid:
            return None

        name = task.get("name", "")
        notes = task.get("notes") or ""
        completed = task.get("completed", False)
        due_on = task.get("due_on") or ""
        modified_at = task.get("modified_at", "")
        created_at = task.get("created_at", "")
        assignee_obj = task.get("assignee") or {}
        assignee = assignee_obj.get("name", "")

        text_parts = [
            f"{name}",
            f"Completed: {completed}",
            f"Assignee: {assignee}",
        ]
        if due_on:
            text_parts.append(f"Due: {due_on}")
        if notes:
            text_parts.append(f"\n{notes}")

        return ParsedDocument(
            path=Path(f"asana://{gid}.asana"),
            text="\n".join(text_parts),
            file_type=".asana",
            encoding="utf-8",
            metadata={
                "gid": gid,
                "name": name,
                "completed": completed,
                "due_on": due_on,
                "assignee": assignee,
                "modified_at": modified_at,
                "created_at": created_at,
                "source": self._cfg.source_name,
            },
        )
