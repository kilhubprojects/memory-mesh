"""Shared auth guard helper for MemoryMesh MCP tools and REST endpoints (Wave 4).

Each MCP tool calls :func:`check_access` at the top of its handler.  The
function resolves the client identity (from a Bearer token or MCP clientId),
checks revocation, enforces rate limits, and runs the requested ACL check.
It returns ``None`` on success or an error ``dict`` that the tool should
return immediately to the caller.

REST endpoints use :func:`check_access_http` which raises
``HTTPException(403)`` instead of returning a dict.

When ``auth.enabled: false`` (the default), every check is a no-op - the
function always returns ``None``.

Identity resolution priority (highest to lowest):
1. Explicit *client_id* kwarg passed by the caller.
2. Value set in :data:`_CURRENT_CLIENT_ID` context variable (set by HTTP
   middleware when it extracts the ``Authorization: Bearer <token>`` header).
3. ``None`` -> anonymous identity with the configured default permission.
"""

from __future__ import annotations

import contextvars
from typing import TYPE_CHECKING, Literal

from loguru import logger

if TYPE_CHECKING:
    from memorymesh.server.app import AppContext

_Action = Literal["read", "index", "delete", "admin"]

# Set by HTTP middleware or MCP request hook with the raw Bearer token /
# clientId extracted from the incoming request.  Defaults to None (anonymous).
_CURRENT_CLIENT_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_CURRENT_CLIENT_ID", default=None
)


def set_client_id(client_id: str | None) -> contextvars.Token[str | None]:
    """Set the client identifier for the current async/thread context.

    Returns a :class:`contextvars.Token` that can be used to restore the
    previous value with :meth:`contextvars.ContextVar.reset`.

    Args:
        client_id: Raw identifier from the request (Bearer token value or
            MCP ``_meta.clientId``).  Pass ``None`` to clear.
    """
    return _CURRENT_CLIENT_ID.set(client_id)


def _extract_bearer(authorization: str | None) -> str | None:
    """Extract the token from an ``Authorization: Bearer <token>`` header.

    Args:
        authorization: Raw header value (e.g. ``"Bearer ghp_abc123"``).
            ``None`` if the header was not present.

    Returns:
        The token string, or ``None`` if the header is absent or malformed.
    """
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip() or None
    return None


def _extract_mcp_client_id(meta: dict | None) -> str | None:
    """Return ``_meta.clientId`` from MCP request metadata when present."""
    if not isinstance(meta, dict):
        return None
    raw = meta.get("clientId")
    if raw is None:
        return None
    client_id = str(raw).strip()
    return client_id or None


def check_access(
    ctx: AppContext,
    action: _Action,
    *,
    source: str | None = None,
    client_id: str | None = None,
    meta: dict | None = None,
) -> dict | None:
    """Perform the full Wave-4 auth stack for a single tool invocation.

    Steps (each step is a no-op when the corresponding component is not
    configured or auth is disabled):

    1. Resolve client identity - from explicit *client_id*, then from the
       :data:`_CURRENT_CLIENT_ID` context variable (set by HTTP middleware),
       then falls back to anonymous.
    2. Revocation check - deny if the client_id is in the revocation list.
    3. Rate limiting - deny if the token bucket is exhausted.
    4. ACL check - deny if the identity lacks the required permission.

    Args:
        ctx: The shared :class:`~memorymesh.server.app.AppContext`.
        action: The operation type - ``"read"``, ``"index"``, ``"delete"``,
            or ``"admin"``.
        source: Optional source name (used by ``check_read`` for source-level
            ACL filtering).
        client_id: Optional explicit client identifier override.  When
            provided, overrides MCP metadata and the context variable.
        meta: Optional MCP request metadata.  If present, ``_meta.clientId`` is
            used before the context variable.

    Returns:
        ``None`` when access is granted, or a ``{"error": str}`` dict to
        return directly from the MCP tool handler.
    """
    # Short-circuit when auth is disabled (default) or no resolver is wired.
    if ctx.identity_resolver is None or not ctx.config.auth.enabled:
        return None

    # Priority: explicit kwarg > MCP metadata > context var > anonymous.
    raw_id = client_id if client_id is not None else _extract_mcp_client_id(meta)
    if raw_id is None:
        raw_id = _CURRENT_CLIENT_ID.get()
    identity = ctx.identity_resolver.resolve(raw_id)

    if ctx.revocation_list is not None:
        try:
            if ctx.revocation_list.is_revoked(identity.client_id):
                return {"error": f"Client {identity.client_id!r} has been revoked."}
        except Exception as exc:
            logger.debug(f"AuthGuard: revocation check skipped after store error: {exc}")

    if ctx.rate_limiter is not None:
        try:
            ctx.rate_limiter.check(identity)
        except Exception as exc:
            from memorymesh.core.exceptions import RateLimitExceededError

            if isinstance(exc, RateLimitExceededError):
                return {"error": f"Rate limit exceeded. Retry after {exc.retry_after_s:.1f}s."}

    if ctx.acl_enforcer is not None:
        try:
            if action == "read":
                ctx.acl_enforcer.check_read(identity, source=source)
            elif action == "index":
                ctx.acl_enforcer.check_index(identity)
            elif action == "delete":
                ctx.acl_enforcer.check_delete(identity)
            elif action == "admin":
                ctx.acl_enforcer.check_admin(identity)
        except Exception as exc:
            from memorymesh.core.exceptions import PermissionDeniedError

            if isinstance(exc, PermissionDeniedError):
                return {"error": str(exc)}

    return None


def check_access_http(
    ctx: AppContext,
    action: _Action,
    *,
    source: str | None = None,
    authorization: str | None = None,
) -> None:
    """Perform the auth stack for a REST endpoint; raise ``HTTPException`` on denial.

    Extracts the Bearer token from *authorization* (the raw
    ``Authorization`` header value), sets the context variable, then calls
    :func:`check_access`.

    Args:
        ctx: The shared :class:`~memorymesh.server.app.AppContext`.
        action: The operation type.
        source: Optional source name for read ACL filtering.
        authorization: Raw ``Authorization`` header value.

    Raises:
        fastapi.HTTPException: 403 when credentials are missing or denied.
    """
    if not ctx.config.auth.enabled:
        return

    from fastapi import HTTPException

    raw_id = _extract_bearer(authorization)

    # No credentials at all -> 401 Unauthorized
    if raw_id is None:
        raise HTTPException(
            status_code=403,
            detail="Authentication required. Provide 'Authorization: Bearer <token>'.",
        )

    # Temporarily set the context var so check_access picks it up
    token = set_client_id(raw_id)
    try:
        result = check_access(ctx, action, source=source, client_id=raw_id)
    finally:
        _CURRENT_CLIENT_ID.reset(token)

    if result is not None:
        raise HTTPException(status_code=403, detail=result.get("error", "Forbidden"))
