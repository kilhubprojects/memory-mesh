"""Token-bucket rate limiter for MemoryMesh (Wave 4).

Each client gets an independent token bucket that refills at
``rate_limit_per_min`` tokens per minute.  A rate of ``0`` means unlimited.

The implementation is purely in-process (no Redis, no shared state).  For a
single-node local daemon this is correct; distributed rate limiting is a future
concern if MemoryMesh ever runs as a multi-process service.

All checks are no-ops when ``auth.enabled: false``.
"""

from __future__ import annotations

import threading
import time

from loguru import logger

from memorymesh.core.exceptions import RateLimitExceededError
from memorymesh.core.models import AuthConfig, ClientIdentity


class _Bucket:
    """Single token-bucket state.

    Args:
        rate_per_min: Token refill rate per minute.
        capacity: Burst capacity (defaults to ``rate_per_min``).
    """

    def __init__(self, rate_per_min: int, capacity: int | None = None) -> None:
        self._rate_per_s: float = rate_per_min / 60.0
        self._capacity: float = float(capacity or rate_per_min)
        self._tokens: float = self._capacity
        self._last_refill: float = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> float:
        """Try to consume *tokens* from the bucket.

        Args:
            tokens: Number of tokens to consume (default 1).

        Returns:
            ``0.0`` if the tokens were consumed successfully, or the estimated
            wait time in seconds until enough tokens are available.
        """
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_s)
            self._last_refill = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0
            deficit = tokens - self._tokens
            wait_s = deficit / self._rate_per_s
            return wait_s


class RateLimiter:
    """Per-client token-bucket rate limiter.

    Args:
        config: The ``AuthConfig`` for default limits.
    """

    def __init__(self, config: AuthConfig) -> None:
        self._config = config
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def _get_bucket(self, identity: ClientIdentity) -> _Bucket | None:
        """Return (and lazily create) the bucket for *identity*.

        Returns ``None`` when the client has ``rate_limit_per_min == 0``
        (unlimited).
        """
        limit = identity.rate_limit_per_min
        if limit == 0:
            return None
        with self._lock:
            if identity.client_id not in self._buckets:
                self._buckets[identity.client_id] = _Bucket(rate_per_min=limit)
            return self._buckets[identity.client_id]

    def check(self, identity: ClientIdentity, tokens: int = 1) -> None:
        """Assert that *identity* has not exceeded its rate limit.

        Args:
            identity: Resolved client identity with ``rate_limit_per_min``.
            tokens: Cost of this operation in tokens (default 1).

        Raises:
            RateLimitExceededError: If the bucket is empty.
        """
        if not self._config.enabled:
            return
        bucket = self._get_bucket(identity)
        if bucket is None:
            return  # Unlimited client.
        wait_s = bucket.consume(tokens)
        if wait_s > 0.0:
            logger.warning(
                f"RateLimiter: client={identity.client_id!r} throttled retry_after={wait_s:.1f}s"
            )
            raise RateLimitExceededError(identity.client_id, retry_after_s=wait_s)
        logger.debug(f"RateLimiter: client={identity.client_id!r} consumed {tokens} token(s)")

    def reset(self, client_id: str) -> None:
        """Reset the bucket for *client_id* (useful for testing / admin).

        Args:
            client_id: The client whose bucket to reset.
        """
        with self._lock:
            self._buckets.pop(client_id, None)
