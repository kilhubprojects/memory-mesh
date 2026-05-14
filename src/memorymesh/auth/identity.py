"""Client identity resolution for MemoryMesh (Wave 4).

Resolves an inbound ``client_id`` string — extracted from the
``X-MemoryMesh-Client`` HTTP header or from MCP ``initialize`` metadata — into
a :class:`~memorymesh.core.models.ClientIdentity` with its effective permission
level, source allowlist, and rate limit.

When ``auth.enabled: false`` (the solo-user default) every client gets the
configured ``default_permission`` with no rate limiting.
"""

from __future__ import annotations

from loguru import logger

from memorymesh.core.models import AgentConfig, AgentPermission, AuthConfig, ClientIdentity


class IdentityResolver:
    """Resolves raw client IDs to fully hydrated :class:`ClientIdentity` objects.

    Args:
        config: The ``AuthConfig`` from the root ``MemoryMeshConfig``.

    The resolver builds a lookup dict from ``config.agents`` at construction
    time so all resolution is O(1).
    """

    def __init__(self, config: AuthConfig) -> None:
        self._config = config
        self._agents: dict[str, AgentConfig] = {agent.client_id: agent for agent in config.agents}

    def resolve(self, client_id: str | None) -> ClientIdentity:
        """Resolve *client_id* to a :class:`ClientIdentity`.

        Args:
            client_id: Raw identifier from the request.  ``None`` or empty
                string means the client did not identify itself.

        Returns:
            A :class:`ClientIdentity` with effective permission and limits.
        """
        if not self._config.enabled:
            return ClientIdentity(
                client_id=client_id or "anonymous",
                name="anonymous",
                permission=self._config.default_permission,
                allowed_sources=[],
                rate_limit_per_min=0,  # unlimited when auth is off
                is_anonymous=not bool(client_id),
            )

        if not client_id:
            logger.debug("IdentityResolver: anonymous client (no client_id header)")
            return ClientIdentity(
                client_id="anonymous",
                name="anonymous",
                permission=self._config.default_permission,
                allowed_sources=[],
                rate_limit_per_min=self._config.default_rate_limit_per_min,
                is_anonymous=True,
            )

        agent = self._agents.get(client_id)
        if agent is None:
            logger.warning(
                f"IdentityResolver: unknown client_id={client_id!r}; applying default permission"
            )
            return ClientIdentity(
                client_id=client_id,
                name=client_id,
                permission=self._config.default_permission,
                allowed_sources=[],
                rate_limit_per_min=self._config.default_rate_limit_per_min,
                is_anonymous=False,
            )

        logger.debug(
            f"IdentityResolver: resolved client_id={client_id!r} "
            f"permission={agent.permission.value!r}"
        )
        return ClientIdentity(
            client_id=agent.client_id,
            name=agent.name or agent.client_id,
            permission=agent.permission,
            allowed_sources=list(agent.sources),
            rate_limit_per_min=agent.rate_limit_per_min,
            is_anonymous=False,
        )

    def has_permission(
        self,
        identity: ClientIdentity,
        required: AgentPermission,
    ) -> bool:
        """Return ``True`` if *identity* meets or exceeds *required* permission.

        Permission ordering: ``read`` < ``read+index`` < ``read+index+delete`` < ``admin``.

        Args:
            identity: Resolved client identity.
            required: Minimum permission level needed for the operation.
        """
        _order: dict[AgentPermission, int] = {
            AgentPermission.read: 0,
            AgentPermission.read_index: 1,
            AgentPermission.read_index_delete: 2,
            AgentPermission.admin: 3,
        }
        return _order.get(identity.permission, 0) >= _order.get(required, 0)

    def can_access_source(self, identity: ClientIdentity, source: str) -> bool:
        """Return ``True`` if *identity* is allowed to access *source*.

        An empty ``allowed_sources`` list means "all sources allowed".

        Args:
            identity: Resolved client identity.
            source: Source name to check.
        """
        if not identity.allowed_sources:
            return True
        return source in identity.allowed_sources
