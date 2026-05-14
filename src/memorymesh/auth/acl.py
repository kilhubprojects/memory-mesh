"""ACL (Access Control List) enforcer for MemoryMesh (Wave 4).

Wraps the :class:`~memorymesh.auth.identity.IdentityResolver` and exposes
simple ``check_*`` methods that raise :class:`PermissionDeniedError` on violation.

All checks are no-ops when ``auth.enabled: false``.
"""

from __future__ import annotations

from loguru import logger

from memorymesh.auth.identity import IdentityResolver
from memorymesh.core.exceptions import PermissionDeniedError
from memorymesh.core.models import AgentPermission, AuthConfig, ClientIdentity


class ACLEnforcer:
    """Enforces source-level and operation-level access control.

    Args:
        config: The ``AuthConfig`` from the root ``MemoryMeshConfig``.
        resolver: The identity resolver to use for permission lookups.
    """

    def __init__(self, config: AuthConfig, resolver: IdentityResolver) -> None:
        self._config = config
        self._resolver = resolver

    def check_read(self, identity: ClientIdentity, source: str | None = None) -> None:
        """Assert that *identity* may perform a read operation.

        Args:
            identity: Resolved client identity.
            source: Optional source name being accessed.

        Raises:
            PermissionDeniedError: If the client lacks ``read`` permission or is
                not allowed to access *source*.
        """
        if not self._config.enabled:
            return
        if not self._resolver.has_permission(identity, AgentPermission.read):
            raise PermissionDeniedError(identity.client_id, AgentPermission.read)
        if source and not self._resolver.can_access_source(identity, source):
            raise PermissionDeniedError(
                identity.client_id,
                AgentPermission.read,
                f"source {source!r} not in allowed_sources",
            )
        logger.debug(f"ACL: read granted client={identity.client_id!r} source={source!r}")

    def check_index(self, identity: ClientIdentity) -> None:
        """Assert that *identity* may trigger an indexing operation.

        Args:
            identity: Resolved client identity.

        Raises:
            PermissionDeniedError: If the client lacks ``read+index`` permission.
        """
        if not self._config.enabled:
            return
        if not self._resolver.has_permission(identity, AgentPermission.read_index):
            raise PermissionDeniedError(identity.client_id, AgentPermission.read_index)
        logger.debug(f"ACL: index granted client={identity.client_id!r}")

    def check_delete(self, identity: ClientIdentity) -> None:
        """Assert that *identity* may perform a delete operation.

        Args:
            identity: Resolved client identity.

        Raises:
            PermissionDeniedError: If the client lacks ``read+index+delete`` permission.
        """
        if not self._config.enabled:
            return
        if not self._resolver.has_permission(identity, AgentPermission.read_index_delete):
            raise PermissionDeniedError(identity.client_id, AgentPermission.read_index_delete)
        logger.debug(f"ACL: delete granted client={identity.client_id!r}")

    def check_admin(self, identity: ClientIdentity) -> None:
        """Assert that *identity* has admin-level permission.

        Args:
            identity: Resolved client identity.

        Raises:
            PermissionDeniedError: If the client lacks ``admin`` permission.
        """
        if not self._config.enabled:
            return
        if not self._resolver.has_permission(identity, AgentPermission.admin):
            raise PermissionDeniedError(identity.client_id, AgentPermission.admin)
        logger.debug(f"ACL: admin granted client={identity.client_id!r}")
