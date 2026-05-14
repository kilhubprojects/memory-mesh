"""Wave 4 auth layer for MemoryMesh.

Provides per-client identity resolution, ACL enforcement, token-bucket rate
limiting, and a revocation list — all backed by SQLite via the MetadataStore.

When ``auth.enabled: false`` (the default), every component in this package
returns permissive defaults so the single-user local daemon works with zero
configuration.
"""

from memorymesh.auth.acl import ACLEnforcer, PermissionDeniedError
from memorymesh.auth.identity import IdentityResolver
from memorymesh.auth.rate_limiter import RateLimiter, RateLimitExceededError
from memorymesh.auth.revocation import RevocationList

__all__ = [
    "ACLEnforcer",
    "IdentityResolver",
    "PermissionDeniedError",
    "RateLimitExceededError",
    "RateLimiter",
    "RevocationList",
]
