"""Unit tests for Wave 4 auth layer - identity, ACL, rate limiter, revocation."""

from __future__ import annotations

from pathlib import Path

import pytest

from memorymesh.auth.acl import ACLEnforcer, PermissionDeniedError
from memorymesh.auth.identity import IdentityResolver
from memorymesh.auth.rate_limiter import RateLimiter, RateLimitExceededError
from memorymesh.auth.revocation import RevocationList
from memorymesh.core.models import (
    AgentConfig,
    AgentPermission,
    AuthConfig,
    ClientIdentity,
)


def _auth_disabled() -> AuthConfig:
    return AuthConfig(enabled=False)


def _auth_enabled(*agents: AgentConfig) -> AuthConfig:
    return AuthConfig(
        enabled=True,
        agents=list(agents),
        default_permission=AgentPermission.read,
        default_rate_limit_per_min=10,
    )


def _agent(
    client_id: str = "agent-1",
    permission: AgentPermission = AgentPermission.read,
    sources: list[str] | None = None,
    rate: int = 60,
) -> AgentConfig:
    return AgentConfig(
        client_id=client_id,
        name=client_id,
        permission=permission,
        sources=sources or [],
        rate_limit_per_min=rate,
    )


class TestIdentityResolverDisabled:
    def test_any_client_gets_default_permission(self) -> None:
        resolver = IdentityResolver(_auth_disabled())
        identity = resolver.resolve("some-id")
        assert identity.client_id == "some-id"
        assert identity.is_anonymous is False
        # Rate limiting is off when auth is disabled.
        assert identity.rate_limit_per_min == 0

    def test_none_client_id(self) -> None:
        resolver = IdentityResolver(_auth_disabled())
        identity = resolver.resolve(None)
        assert identity.is_anonymous is True


class TestIdentityResolverEnabled:
    def test_known_agent_resolved(self) -> None:
        a = _agent("claude", AgentPermission.admin, ["notes"])
        resolver = IdentityResolver(_auth_enabled(a))
        identity = resolver.resolve("claude")
        assert identity.permission == AgentPermission.admin
        assert identity.allowed_sources == ["notes"]
        assert identity.is_anonymous is False

    def test_unknown_agent_gets_default(self) -> None:
        resolver = IdentityResolver(_auth_enabled())
        identity = resolver.resolve("unknown-xyz")
        assert identity.permission == AgentPermission.read
        assert identity.rate_limit_per_min == 10

    def test_anonymous_gets_default(self) -> None:
        resolver = IdentityResolver(_auth_enabled())
        identity = resolver.resolve(None)
        assert identity.is_anonymous is True
        assert identity.client_id == "anonymous"


class TestPermissionOrdering:
    def setup_method(self) -> None:
        self._resolver = IdentityResolver(_auth_enabled())

    def _identity(self, perm: AgentPermission) -> ClientIdentity:
        return ClientIdentity(client_id="x", permission=perm, is_anonymous=False)

    def test_admin_has_all_permissions(self) -> None:
        identity = self._identity(AgentPermission.admin)
        for required in AgentPermission:
            assert self._resolver.has_permission(identity, required)

    def test_read_only_lacks_index(self) -> None:
        identity = self._identity(AgentPermission.read)
        assert not self._resolver.has_permission(identity, AgentPermission.read_index)

    def test_read_index_lacks_delete(self) -> None:
        identity = self._identity(AgentPermission.read_index)
        assert not self._resolver.has_permission(identity, AgentPermission.read_index_delete)


class TestSourceAllowlist:
    def test_empty_allowlist_permits_any_source(self) -> None:
        resolver = IdentityResolver(_auth_enabled())
        identity = ClientIdentity(client_id="x", allowed_sources=[], is_anonymous=False)
        assert resolver.can_access_source(identity, "any-source")

    def test_allowlist_restricts_source(self) -> None:
        resolver = IdentityResolver(_auth_enabled())
        identity = ClientIdentity(client_id="x", allowed_sources=["notes"], is_anonymous=False)
        assert resolver.can_access_source(identity, "notes")
        assert not resolver.can_access_source(identity, "email")


class TestACLEnforcerDisabled:
    def test_read_always_passes(self) -> None:
        enforcer = ACLEnforcer(_auth_disabled(), IdentityResolver(_auth_disabled()))
        identity = ClientIdentity(client_id="x", permission=AgentPermission.read, is_anonymous=True)
        # Should not raise.
        enforcer.check_read(identity)

    def test_admin_always_passes(self) -> None:
        enforcer = ACLEnforcer(_auth_disabled(), IdentityResolver(_auth_disabled()))
        identity = ClientIdentity(client_id="x", permission=AgentPermission.read, is_anonymous=True)
        enforcer.check_admin(identity)


class TestACLEnforcerEnabled:
    def setup_method(self) -> None:
        cfg = _auth_enabled(_agent("reader", AgentPermission.read))
        self._resolver = IdentityResolver(cfg)
        self._enforcer = ACLEnforcer(cfg, self._resolver)

    def test_read_permission_passes(self) -> None:
        identity = self._resolver.resolve("reader")
        self._enforcer.check_read(identity)  # Should not raise.

    def test_read_permission_blocks_index(self) -> None:
        identity = self._resolver.resolve("reader")
        with pytest.raises(PermissionDeniedError):
            self._enforcer.check_index(identity)

    def test_source_restriction_enforced(self) -> None:
        cfg = _auth_enabled(_agent("restricted", AgentPermission.read, sources=["notes"]))
        resolver = IdentityResolver(cfg)
        enforcer = ACLEnforcer(cfg, resolver)
        identity = resolver.resolve("restricted")
        with pytest.raises(PermissionDeniedError):
            enforcer.check_read(identity, source="email")

    def test_admin_blocks_read_only(self) -> None:
        identity = self._resolver.resolve("reader")
        with pytest.raises(PermissionDeniedError):
            self._enforcer.check_admin(identity)


class TestRateLimiterDisabled:
    def test_unlimited_when_auth_off(self) -> None:
        limiter = RateLimiter(_auth_disabled())
        identity = ClientIdentity(client_id="x", rate_limit_per_min=0, is_anonymous=False)
        # Should never raise regardless of call count.
        for _ in range(1000):
            limiter.check(identity)


class TestRateLimiterEnabled:
    def setup_method(self) -> None:
        self._cfg = _auth_enabled(_agent("fast", rate=120))
        self._limiter = RateLimiter(self._cfg)

    def _identity(self, rate: int) -> ClientIdentity:
        return ClientIdentity(client_id="test-client", rate_limit_per_min=rate, is_anonymous=False)

    def test_burst_within_capacity_passes(self) -> None:
        identity = self._identity(rate=60)
        # The bucket starts full (capacity = rate_limit_per_min = 60).
        for _ in range(60):
            self._limiter.check(identity)

    def test_exceeding_capacity_raises(self) -> None:
        identity = self._identity(rate=5)
        with pytest.raises(RateLimitExceededError):
            for _ in range(100):
                self._limiter.check(identity)

    def test_retry_after_positive(self) -> None:
        identity = self._identity(rate=1)
        try:
            for _ in range(100):
                self._limiter.check(identity)
        except RateLimitExceededError as exc:
            assert exc.retry_after_s > 0

    def test_reset_clears_bucket(self) -> None:
        identity = self._identity(rate=2)
        # Exhaust.
        try:
            for _ in range(100):
                self._limiter.check(identity)
        except RateLimitExceededError:
            pass
        # Reset should allow new calls.
        self._limiter.reset(identity.client_id)
        self._limiter.check(identity)  # Should not raise.

    def test_zero_rate_is_unlimited(self) -> None:
        limiter = RateLimiter(self._cfg)
        identity = self._identity(rate=0)
        for _ in range(10_000):
            limiter.check(identity)  # Should not raise.


@pytest.fixture()
def revocation(tmp_path: Path) -> RevocationList:
    return RevocationList(tmp_path / "revocation.sqlite3")


class TestRevocationList:
    def test_not_revoked_by_default(self, revocation: RevocationList) -> None:
        assert revocation.is_revoked("alice") is False

    def test_revoke_and_check(self, revocation: RevocationList) -> None:
        revocation.revoke("bad-agent", reason="compromised")
        assert revocation.is_revoked("bad-agent") is True

    def test_unrevoke(self, revocation: RevocationList) -> None:
        revocation.revoke("agent-x")
        revocation.unrevoke("agent-x")
        assert revocation.is_revoked("agent-x") is False

    def test_list_revoked(self, revocation: RevocationList) -> None:
        revocation.revoke("a")
        revocation.revoke("b", reason="old key")
        records = revocation.list_revoked()
        assert len(records) == 2
        ids = {r["client_id"] for r in records}
        assert ids == {"a", "b"}

    def test_duplicate_revoke_is_idempotent(self, revocation: RevocationList) -> None:
        revocation.revoke("dup")
        revocation.revoke("dup", reason="updated reason")
        records = revocation.list_revoked()
        assert len(records) == 1
        assert records[0]["reason"] == "updated reason"

    def test_cache_reflects_revocation(self, revocation: RevocationList) -> None:
        # Force cache population.
        revocation.is_revoked("nothing")
        revocation.revoke("new-villain")
        # Cache should be updated in-process immediately.
        assert revocation.is_revoked("new-villain") is True
