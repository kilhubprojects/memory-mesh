"""GitHub connector for MemoryMesh.

Fetches issues and pull requests from GitHub repositories and yields each
as a :class:`~memorymesh.core.models.ParsedDocument`.

API reference: https://docs.github.com/en/rest/issues

Features
--------
* **Issues and PRs** - both fetched from the ``/issues`` endpoint
  (GitHub returns PRs there too); distinguished by the ``pull_request`` key.
* **Comments** - optional; fetched per item when ``include_comments=True``.
* **Date filter** - only items whose ``created_at`` is within ``days_past``
  days are yielded.
* **Rate-limit aware** - checks ``X-RateLimit-Remaining``; sleeps until
  reset if fewer than 10 calls remain.
* **Pagination** - iterates pages until GitHub returns an empty list.

Usage
-----
::

    connector = GitHubConnector(GitHubConfig(
        token=SecretStr("ghp_..."),
        repos=["owner/repo"],
        days_past=90,
    ))
    for doc in connector.fetch_documents():
        indexer.index_parsed_document(doc)
"""

from __future__ import annotations

import time
import urllib.parse
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, SecretStr

from memorymesh.connectors._auth import bearer_header
from memorymesh.connectors._http import api_get, api_get_with_headers
from memorymesh.core.models import ParsedDocument

# Pause requests when fewer than this many GitHub API calls remain in the rate-limit window.
_RATE_LIMIT_SAFETY_MARGIN = 10


_BASE = "https://api.github.com"


class GitHubConfig(BaseModel):
    """Configuration for a GitHub repository source.

    Args:
        token: GitHub personal access token or fine-grained token.
        repos: List of ``"owner/repo"`` strings to index.
        include_issues: Yield issues (items without a ``pull_request`` key).
        include_prs: Yield pull requests (items with a ``pull_request`` key).
        include_comments: When ``True``, fetch and append all comments.
        state: Filter by issue/PR state: ``"open"``, ``"closed"``, or
            ``"all"``.
        days_past: Only yield items whose ``created_at`` is within this many
            days.
        max_items_per_repo: Maximum items yielded per repository per run.
        source_name: Name used in the MemoryMesh source registry.
    """

    token: SecretStr
    repos: list[str]
    include_issues: bool = True
    include_prs: bool = True
    include_comments: bool = True
    state: Literal["open", "closed", "all"] = "all"
    days_past: int = 365
    max_items_per_repo: int = 500
    source_name: str = "github"


class GitHubConnector:
    """Fetches GitHub issues and PRs and yields them as ParsedDocuments.

    Args:
        config: GitHub API credentials and source settings.
    """

    def __init__(self, config: GitHubConfig) -> None:
        self._cfg = config
        self._headers = {
            **bearer_header(config.token.get_secret_value()),
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def fetch_documents(self) -> Iterator[ParsedDocument]:
        """Fetch issues/PRs from all configured repositories.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` - one per
            issue/PR, with ``file_type=".github"`` and metadata containing
            ``repo``, ``number``, ``type``, ``state``, ``author``,
            ``created_at``, and ``url``.
        """
        for repo in self._cfg.repos:
            parts = repo.split("/", 1)
            if len(parts) != 2:
                logger.warning(
                    f"GitHubConnector: invalid repo format {repo!r} - expected 'owner/repo'"
                )
                continue
            yield from self._process_repo(parts[0], parts[1])

    def _process_repo(self, owner: str, repo: str) -> Iterator[ParsedDocument]:
        """Fetch and yield all qualifying issues/PRs from one repository.

        Args:
            owner: Repository owner login.
            repo: Repository name.

        Yields:
            :class:`~memorymesh.core.models.ParsedDocument` per item.
        """
        cutoff = datetime.now(tz=UTC) - timedelta(days=self._cfg.days_past)
        cutoff_str = cutoff.isoformat()

        since = urllib.parse.quote(cutoff_str)
        yielded = 0
        page = 1

        logger.info(
            f"GitHubConnector: fetching {owner}/{repo}"
            f" state={self._cfg.state} days_past={self._cfg.days_past}"
        )

        while yielded < self._cfg.max_items_per_repo:
            url = (
                f"{_BASE}/repos/{owner}/{repo}/issues"
                f"?state={self._cfg.state}&per_page=100&page={page}&since={since}"
            )
            data, resp_headers = api_get_with_headers(url, self._headers)
            self._check_rate_limit(resp_headers)

            if not isinstance(data, list) or not data:
                break

            for item in data:
                if yielded >= self._cfg.max_items_per_repo:
                    break
                if item.get("created_at", "") < cutoff_str:
                    continue

                is_pr = "pull_request" in item
                if is_pr and not self._cfg.include_prs:
                    continue
                if not is_pr and not self._cfg.include_issues:
                    continue

                doc = self._build_document(item, owner, repo, is_pr)
                if doc is not None:
                    yield doc
                    yielded += 1

            page += 1

        logger.info(f"GitHubConnector: {owner}/{repo} -> {yielded} item(s)")

    def _fetch_comments(self, owner: str, repo: str, number: int) -> list[str]:
        """Fetch all comments for an issue or PR.

        Args:
            owner: Repository owner login.
            repo: Repository name.
            number: Issue/PR number.

        Returns:
            List of comment body strings (empty strings excluded).
        """
        url = f"{_BASE}/repos/{owner}/{repo}/issues/{number}/comments?per_page=100"
        bodies: list[str] = []
        page = 1

        while True:
            paged_url = f"{url}&page={page}"
            data = api_get(paged_url, self._headers)
            if not isinstance(data, list) or not data:
                break
            for comment in data:
                body = (comment.get("body") or "").strip()
                if body:
                    author = comment.get("user", {}).get("login", "")
                    bodies.append(f"**{author}:** {body}")
            page += 1

        return bodies

    def _build_document(
        self,
        item: dict[str, Any],
        owner: str,
        repo: str,
        is_pr: bool,
    ) -> ParsedDocument | None:
        """Convert a GitHub issue/PR dict to a ParsedDocument.

        Args:
            item: GitHub issue/PR object.
            owner: Repository owner login.
            repo: Repository name.
            is_pr: ``True`` when the item is a pull request.

        Returns:
            :class:`~memorymesh.core.models.ParsedDocument`, or ``None`` if
            the item has no usable content.
        """
        number = item.get("number", 0)
        title = item.get("title", "")
        body = (item.get("body") or "").strip()
        state = item.get("state", "")
        author = item.get("user", {}).get("login", "")
        created_at = item.get("created_at", "")
        html_url = item.get("html_url", "")

        parts: list[str] = [f"#{number}: {title}"]
        if body:
            parts.append("")
            parts.append(body)

        if self._cfg.include_comments and item.get("comments", 0) > 0:
            comment_bodies = self._fetch_comments(owner, repo, number)
            if comment_bodies:
                parts.append("")
                parts.append("--- Comments ---")
                parts.extend(comment_bodies)

        text = "\n".join(parts)
        item_type = "pr" if is_pr else "issue"

        return ParsedDocument(
            path=Path(f"github://{owner}/{repo}/#{number}.github"),
            text=text,
            file_type=".github",
            encoding="utf-8",
            metadata={
                "repo": f"{owner}/{repo}",
                "number": number,
                "type": item_type,
                "state": state,
                "author": author,
                "created_at": created_at,
                "url": html_url,
                "source": self._cfg.source_name,
            },
        )

    def _check_rate_limit(self, resp_headers: dict[str, str]) -> None:
        """Sleep until reset if the GitHub rate limit is nearly exhausted.

        Args:
            resp_headers: Lowercase response headers from the last API call.
        """
        remaining_str = resp_headers.get("x-ratelimit-remaining", "60")
        try:
            remaining = int(remaining_str)
        except ValueError:
            return
        if remaining < _RATE_LIMIT_SAFETY_MARGIN:
            reset_str = resp_headers.get("x-ratelimit-reset", "0")
            try:
                reset_ts = int(reset_str)
            except ValueError:
                reset_ts = 0
            sleep_s = max(0.0, reset_ts - time.time()) + 1.0
            logger.warning(
                f"GitHubConnector: rate limit low ({remaining} remaining), sleeping {sleep_s:.0f}s"
            )
            time.sleep(sleep_s)
