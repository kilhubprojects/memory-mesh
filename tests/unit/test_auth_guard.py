"""Unit tests for memorymesh.server.auth_guard.

Covers:
- _extract_bearer: parses / rejects various Authorization header forms
- check_access: disabled path (always None), revocation, rate-limit, ACL checks
- check_access_http: 403 for no credentials/denied, pass-through when
  auth is disabled
- set_client_id / context-variable threading
"""

from __future__ import annotations

import unittest.mock as mock
from typing import Any

import pytest


def _make_ctx(*, enabled: bool = True, agents: list[Any] | None = None) -> mock.MagicMock:
    """Return a minimal AppContext mock with auth configured."""
    from memorymesh.core.models import AgentPermission, AuthConfig, MemoryMeshConfig

    auth = AuthConfig(
        enabled=enabled,
        agents=agents or [],
        default_permission=AgentPermission.read,
    )
    cfg = MemoryMeshConfig(auth=auth)

    ctx = mock.MagicMock()
    ctx.config = cfg
    ctx.identity_resolver = None
    ctx.revocation_list = None
    ctx.rate_limiter = None
    ctx.acl_enforcer = None
    return ctx


class TestExtractBearer:
    """Tests for the Bearer-token extraction helper."""

    def _extract(self, header: str | None) -> str | None:
        from memorymesh.server.auth_guard import _extract_bearer

        return _extract_bearer(header)

    def test_none_header_returns_none(self) -> None:
        assert self._extract(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert self._extract("") is None

    def test_well_formed_bearer(self) -> None:
        assert self._extract("Bearer ghp_abc123") == "ghp_abc123"

    def test_bearer_case_insensitive(self) -> None:
        assert self._extract("bearer my-token") == "my-token"

    def test_bearer_extra_whitespace(self) -> None:
        assert self._extract("Bearer   spaced-token  ") == "spaced-token"

    def test_no_scheme_returns_none(self) -> None:
        assert self._extract("just-a-token") is None

    def test_wrong_scheme_returns_none(self) -> None:
        assert self._extract("Basic dXNlcjpwYXNz") is None

    def test_bearer_with_no_token_returns_none(self) -> None:
        assert self._extract("Bearer ") is None


class TestSetClientId:
    """Tests for context-variable threading."""

    def test_set_and_get(self) -> None:
        from memorymesh.server.auth_guard import _CURRENT_CLIENT_ID, set_client_id

        token = set_client_id("agent-42")
        try:
            assert _CURRENT_CLIENT_ID.get() == "agent-42"
        finally:
            _CURRENT_CLIENT_ID.reset(token)

    def test_reset_restores_previous(self) -> None:
        from memorymesh.server.auth_guard import _CURRENT_CLIENT_ID, set_client_id

        token = set_client_id("agent-A")
        inner = set_client_id("agent-B")
        _CURRENT_CLIENT_ID.reset(inner)
        assert _CURRENT_CLIENT_ID.get() == "agent-A"
        _CURRENT_CLIENT_ID.reset(token)

    def test_set_none_clears(self) -> None:
        from memorymesh.server.auth_guard import _CURRENT_CLIENT_ID, set_client_id

        token = set_client_id(None)
        try:
            assert _CURRENT_CLIENT_ID.get() is None
        finally:
            _CURRENT_CLIENT_ID.reset(token)


class TestCheckAccess:
    """Tests for check_access - the MCP-side auth gate."""

    def test_disabled_auth_always_permits(self) -> None:
        from memorymesh.server.auth_guard import check_access

        ctx = _make_ctx(enabled=False)
        # No identity_resolver wired -> disabled path short-circuits
        assert check_access(ctx, "read") is None
        assert check_access(ctx, "index") is None
        assert check_access(ctx, "delete") is None

    def test_no_resolver_always_permits(self) -> None:
        from memorymesh.server.auth_guard import check_access

        ctx = _make_ctx(enabled=True)
        ctx.identity_resolver = None  # wiring not done -> short-circuit
        assert check_access(ctx, "read") is None

    def test_revoked_client_returns_error(self) -> None:
        from memorymesh.server.auth_guard import check_access

        ctx = _make_ctx(enabled=True)
        identity = mock.MagicMock()
        identity.client_id = "bad-agent"
        ctx.identity_resolver = mock.MagicMock()
        ctx.identity_resolver.resolve.return_value = identity
        ctx.revocation_list = mock.MagicMock()
        ctx.revocation_list.is_revoked.return_value = True

        result = check_access(ctx, "read", client_id="bad-agent")
        assert result is not None
        assert "revoked" in result["error"].lower()

    def test_rate_limit_exceeded_returns_error(self) -> None:
        from memorymesh.core.exceptions import RateLimitExceededError
        from memorymesh.server.auth_guard import check_access

        ctx = _make_ctx(enabled=True)
        identity = mock.MagicMock()
        identity.client_id = "fast-agent"
        ctx.identity_resolver = mock.MagicMock()
        ctx.identity_resolver.resolve.return_value = identity
        ctx.revocation_list = None
        ctx.rate_limiter = mock.MagicMock()
        exc = RateLimitExceededError("fast-agent", retry_after_s=30.0)
        ctx.rate_limiter.check.side_effect = exc

        result = check_access(ctx, "read", client_id="fast-agent")
        assert result is not None
        assert "Rate limit" in result["error"]

    def test_acl_denied_returns_error(self) -> None:
        from memorymesh.core.exceptions import PermissionDeniedError
        from memorymesh.server.auth_guard import check_access

        ctx = _make_ctx(enabled=True)
        identity = mock.MagicMock()
        identity.client_id = "low-perm-agent"
        ctx.identity_resolver = mock.MagicMock()
        ctx.identity_resolver.resolve.return_value = identity
        ctx.revocation_list = None
        ctx.rate_limiter = None
        ctx.acl_enforcer = mock.MagicMock()
        from memorymesh.core.models import AgentPermission

        ctx.acl_enforcer.check_delete.side_effect = PermissionDeniedError(
            "low-perm-agent", AgentPermission.read_index_delete
        )

        result = check_access(ctx, "delete", client_id="low-perm-agent")
        assert result is not None
        assert "error" in result

    def test_all_checks_pass_returns_none(self) -> None:
        from memorymesh.server.auth_guard import check_access

        ctx = _make_ctx(enabled=True)
        identity = mock.MagicMock()
        identity.client_id = "good-agent"
        ctx.identity_resolver = mock.MagicMock()
        ctx.identity_resolver.resolve.return_value = identity
        ctx.revocation_list = mock.MagicMock()
        ctx.revocation_list.is_revoked.return_value = False
        ctx.rate_limiter = mock.MagicMock()  # no exception raised
        ctx.acl_enforcer = mock.MagicMock()  # no exception raised

        assert check_access(ctx, "read", client_id="good-agent") is None

    def test_context_var_used_as_fallback(self) -> None:
        """check_access reads _CURRENT_CLIENT_ID when no explicit client_id given."""
        from memorymesh.server.auth_guard import _CURRENT_CLIENT_ID, check_access, set_client_id

        ctx = _make_ctx(enabled=True)
        identity = mock.MagicMock()
        identity.client_id = "ctx-agent"
        ctx.identity_resolver = mock.MagicMock()
        ctx.identity_resolver.resolve.return_value = identity
        ctx.revocation_list = None
        ctx.rate_limiter = None
        ctx.acl_enforcer = None

        token = set_client_id("ctx-agent")
        try:
            check_access(ctx, "read")
            ctx.identity_resolver.resolve.assert_called_once_with("ctx-agent")
        finally:
            _CURRENT_CLIENT_ID.reset(token)

    def test_mcp_meta_client_id_used_before_context(self) -> None:
        """MCP _meta.clientId is used when no explicit client_id is passed."""
        from memorymesh.server.auth_guard import _CURRENT_CLIENT_ID, check_access, set_client_id

        ctx = _make_ctx(enabled=True)
        identity = mock.MagicMock()
        identity.client_id = "meta-agent"
        ctx.identity_resolver = mock.MagicMock()
        ctx.identity_resolver.resolve.return_value = identity
        ctx.revocation_list = None
        ctx.rate_limiter = None
        ctx.acl_enforcer = None

        token = set_client_id("ctx-agent")
        try:
            check_access(ctx, "read", meta={"clientId": "meta-agent"})
            ctx.identity_resolver.resolve.assert_called_once_with("meta-agent")
        finally:
            _CURRENT_CLIENT_ID.reset(token)


class TestCheckAccessHttp:
    """Tests for check_access_http - the REST-side auth gate."""

    def test_disabled_auth_passes_through(self) -> None:
        """When auth.enabled=False, no exception is raised for any request."""
        from memorymesh.server.auth_guard import check_access_http

        ctx = _make_ctx(enabled=False)
        # Should not raise
        check_access_http(ctx, "read", authorization=None)
        check_access_http(ctx, "index", authorization=None)

    def test_no_credentials_raises_403(self) -> None:
        """Missing Bearer token returns HTTPException 403 when auth is enabled."""
        from fastapi import HTTPException

        from memorymesh.server.auth_guard import check_access_http

        ctx = _make_ctx(enabled=True)

        with pytest.raises(HTTPException) as exc_info:
            check_access_http(ctx, "read", authorization=None)
        assert exc_info.value.status_code == 403

    def test_malformed_authorization_raises_403(self) -> None:
        """Non-Bearer Authorization header returns 403."""
        from fastapi import HTTPException

        from memorymesh.server.auth_guard import check_access_http

        ctx = _make_ctx(enabled=True)

        with pytest.raises(HTTPException) as exc_info:
            check_access_http(ctx, "read", authorization="Basic dXNlcjpwYXNz")
        assert exc_info.value.status_code == 403

    def test_valid_token_but_denied_raises_403(self) -> None:
        """Valid Bearer token that check_access denies -> HTTPException 403."""
        from fastapi import HTTPException

        from memorymesh.server.auth_guard import check_access_http

        ctx = _make_ctx(enabled=True)
        identity = mock.MagicMock()
        identity.client_id = "restricted"
        ctx.identity_resolver = mock.MagicMock()
        ctx.identity_resolver.resolve.return_value = identity
        ctx.revocation_list = None
        ctx.rate_limiter = None
        ctx.acl_enforcer = mock.MagicMock()
        from memorymesh.core.exceptions import PermissionDeniedError
        from memorymesh.core.models import AgentPermission

        ctx.acl_enforcer.check_read.side_effect = PermissionDeniedError(
            "restricted", AgentPermission.read
        )

        with pytest.raises(HTTPException) as exc_info:
            check_access_http(ctx, "read", authorization="Bearer secret-token")
        assert exc_info.value.status_code == 403

    def test_valid_token_permitted_does_not_raise(self) -> None:
        """Valid Bearer token with sufficient permissions -> no exception."""
        from memorymesh.server.auth_guard import check_access_http

        ctx = _make_ctx(enabled=True)
        identity = mock.MagicMock()
        identity.client_id = "good-agent"
        ctx.identity_resolver = mock.MagicMock()
        ctx.identity_resolver.resolve.return_value = identity
        ctx.revocation_list = mock.MagicMock()
        ctx.revocation_list.is_revoked.return_value = False
        ctx.rate_limiter = mock.MagicMock()
        ctx.acl_enforcer = mock.MagicMock()  # no exception

        # Should not raise
        check_access_http(ctx, "read", authorization="Bearer valid-token")

    def test_context_var_reset_after_call(self) -> None:
        """_CURRENT_CLIENT_ID is restored to its prior value after check_access_http."""
        from memorymesh.server.auth_guard import (
            _CURRENT_CLIENT_ID,
            check_access_http,
            set_client_id,
        )

        ctx = _make_ctx(enabled=True)
        identity = mock.MagicMock()
        identity.client_id = "outer-context"
        ctx.identity_resolver = mock.MagicMock()
        ctx.identity_resolver.resolve.return_value = identity
        ctx.revocation_list = None
        ctx.rate_limiter = None
        ctx.acl_enforcer = None

        outer_token = set_client_id("outer-context")
        try:
            check_access_http(ctx, "read", authorization="Bearer inner-token")
            # After the call, context var should be back to "outer-context"
            assert _CURRENT_CLIENT_ID.get() == "outer-context"
        finally:
            _CURRENT_CLIENT_ID.reset(outer_token)
