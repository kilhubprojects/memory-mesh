"""Tiered memory manager for MemoryMesh.

Implements a three-tier memory hierarchy - hot / warm / cold - on top of the
:class:`~memorymesh.storage.metadata_store.MetadataStore` and a lightweight
in-process LRU cache.

Tier definitions
----------------
* **hot** - recently accessed (within ``hot_tier_days``) or manually pinned.
  Chunk IDs are cached in RAM for zero-latency tier lookups.  Maximum
  ``hot_max_chunks`` entries; LRU eviction to warm on overflow.
* **warm** - normal indexed content.  Default tier for all new chunks.
* **cold** - chunks not accessed for ``cold_tier_days``.  When the optional
  forgetting policy is enabled, their effective search scores are multiplied by
  an exponential decay factor.

Promotion / demotion
--------------------
Call :meth:`TieredMemoryManager.run_maintenance` (e.g. once per hour from the
daemon's scheduler) to batch-promote warm->hot and demote warm/hot->cold based
on the ``last_accessed`` timestamps recorded by
:meth:`~memorymesh.storage.metadata_store.MetadataStore.record_chunk_access`.

Score decay
-----------
:meth:`TieredMemoryManager.apply_decay` takes a list of ``(chunk_id, score)``
pairs and returns adjusted scores.  Cold chunks that have not been pinned are
multiplied by ``max(floor, 0.5 ** (age_days / half_life_days))``.

Privacy invariant: no document content or query text is stored or logged here.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from loguru import logger

from memorymesh.core.models import (
    ChunkTierRecord,
    ForgettingConfig,
    MemoryTier,
    MemoryTierConfig,
)

if TYPE_CHECKING:
    from memorymesh.storage.metadata_store import MetadataStore


class _LRUSet:
    """A fixed-capacity ordered set with LRU eviction.

    Used to track the hot-tier chunk IDs in RAM.  On overflow the least-recently
    used item is removed and returned so the caller can demote it to warm.

    Args:
        capacity: Maximum number of items.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, capacity)
        self._data: OrderedDict[str, None] = OrderedDict()

    def touch(self, key: str) -> str | None:
        """Add *key* (or move to most-recent position).

        Returns:
            The evicted key if capacity was exceeded, otherwise ``None``.
        """
        if key in self._data:
            self._data.move_to_end(key)
            return None
        self._data[key] = None
        if len(self._data) > self._capacity:
            evicted, _ = self._data.popitem(last=False)
            return evicted
        return None

    def discard(self, key: str) -> None:
        """Remove *key* if present."""
        self._data.pop(key, None)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def keys(self) -> list[str]:
        """Return all current keys (most-recent last)."""
        return list(self._data.keys())


class TieredMemoryManager:
    """Manages hot/warm/cold tier assignments and score decay for chunks.

    This class is a thin orchestration layer; all persistent state lives in
    the :class:`~memorymesh.storage.metadata_store.MetadataStore`.  The
    in-process LRU cache (``_hot_cache``) is a performance optimisation and is
    rebuilt from SQLite on startup if needed.

    Args:
        store: The metadata store used to persist tier records.
        tier_config: Thresholds for promotion/demotion.
        forgetting_config: Settings for optional score decay.
    """

    def __init__(
        self,
        store: MetadataStore,
        tier_config: MemoryTierConfig | None = None,
        forgetting_config: ForgettingConfig | None = None,
    ) -> None:
        self._store = store
        self._tier_cfg = tier_config or MemoryTierConfig()
        self._forget_cfg = forgetting_config or ForgettingConfig()
        self._hot_cache = _LRUSet(self._tier_cfg.hot_max_chunks)

    def record_access(self, chunk_id: str) -> None:
        """Record that *chunk_id* was retrieved.

        Updates ``last_accessed`` / ``access_count`` in SQLite and promotes the
        chunk to hot in the in-process cache.  Any LRU eviction is written back
        to SQLite as a demotion to warm.

        Args:
            chunk_id: ``<path>:<chunk_index>`` stable identifier.
        """
        self._store.record_chunk_access(chunk_id)
        evicted = self._hot_cache.touch(chunk_id)
        if evicted:
            self._demote_to_warm(evicted)

    def pin(self, chunk_id: str) -> None:
        """Manually pin *chunk_id* to the hot tier.

        Pinned chunks are never demoted by maintenance runs.  They can only be
        un-pinned via :meth:`unpin`.

        Args:
            chunk_id: ``<path>:<chunk_index>`` stable identifier.
        """
        existing = self._store.get_chunk_tier(chunk_id)
        record = ChunkTierRecord(
            chunk_id=chunk_id,
            tier=MemoryTier.hot,
            last_accessed=existing.last_accessed if existing else time.time(),
            access_count=existing.access_count if existing else 0,
            pinned=True,
        )
        self._store.set_chunk_tier(record)
        self._hot_cache.touch(chunk_id)
        logger.info(f"TieredMemory: pinned chunk {chunk_id!r}")

    def unpin(self, chunk_id: str) -> None:
        """Remove the manual pin from *chunk_id*, allowing normal demotion.

        The chunk remains in hot tier until the next maintenance run.

        Args:
            chunk_id: ``<path>:<chunk_index>`` stable identifier.
        """
        existing = self._store.get_chunk_tier(chunk_id)
        if existing is None:
            return
        record = ChunkTierRecord(
            chunk_id=chunk_id,
            tier=existing.tier,
            last_accessed=existing.last_accessed,
            access_count=existing.access_count,
            pinned=False,
        )
        self._store.set_chunk_tier(record)
        logger.info(f"TieredMemory: unpinned chunk {chunk_id!r}")

    def forget(self, chunk_id: str) -> None:
        """Force *chunk_id* to cold tier immediately.

        This does not delete the chunk - it sets the tier to cold so that
        :meth:`apply_decay` will maximally discount its score on future searches.

        Args:
            chunk_id: ``<path>:<chunk_index>`` stable identifier.
        """
        existing = self._store.get_chunk_tier(chunk_id)
        record = ChunkTierRecord(
            chunk_id=chunk_id,
            tier=MemoryTier.cold,
            last_accessed=existing.last_accessed if existing else time.time(),
            access_count=existing.access_count if existing else 0,
            pinned=False,
        )
        self._store.set_chunk_tier(record)
        self._hot_cache.discard(chunk_id)
        logger.info(f"TieredMemory: forcibly forgot chunk {chunk_id!r}")

    def apply_decay(
        self,
        scored_chunks: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Apply exponential score decay to cold-tier chunks.

        Hot and warm chunks are returned with their original scores.  Cold
        chunks (that are not pinned) have their scores multiplied by
        ``max(min_score_floor, 0.5 ** (age_days / half_life_days))``.

        If the forgetting policy is disabled, returns *scored_chunks* unchanged.

        Args:
            scored_chunks: List of ``(chunk_id, score)`` pairs.

        Returns:
            List of ``(chunk_id, adjusted_score)`` in the same order.
        """
        if not self._forget_cfg.enabled:
            return scored_chunks

        now = time.time()
        half_life_s = self._forget_cfg.decay_half_life_days * 86_400.0
        floor = self._forget_cfg.min_score_floor

        result: list[tuple[str, float]] = []
        for chunk_id, score in scored_chunks:
            record = self._store.get_chunk_tier(chunk_id)
            if record is None or record.tier != MemoryTier.cold or record.pinned:
                result.append((chunk_id, score))
            else:
                age_s = max(0.0, now - record.last_accessed)
                factor = max(floor, math.pow(0.5, age_s / half_life_s))
                result.append((chunk_id, score * factor))
        return result

    def run_maintenance(self) -> dict[str, int]:
        """Batch-promote and demote chunks based on access timestamps.

        Should be called periodically (e.g. once per hour).  Updates SQLite
        tier records in bulk; does **not** affect pinned chunks.

        Returns:
            Dict with ``promoted`` (warm->hot) and ``demoted`` (warm/hot->cold)
            counts.
        """
        now = time.time()
        hot_cutoff = now - self._tier_cfg.hot_tier_days * 86_400.0
        cold_cutoff = now - self._tier_cfg.cold_tier_days * 86_400.0

        promoted = self._promote_recent_to_hot(hot_cutoff)
        demoted = self._demote_stale_to_cold(cold_cutoff)

        logger.info(
            f"TieredMemory maintenance: promoted={promoted} warm->hot, demoted={demoted} ->cold"
        )
        return {"promoted": promoted, "demoted": demoted}

    def get_tier(self, chunk_id: str) -> MemoryTier:
        """Return the current tier for *chunk_id*.

        Checks the in-process cache first (O(1)), then SQLite.  Returns
        ``MemoryTier.warm`` if the chunk has never been tier-tracked.

        Args:
            chunk_id: ``<path>:<chunk_index>`` stable identifier.
        """
        if chunk_id in self._hot_cache:
            return MemoryTier.hot
        record = self._store.get_chunk_tier(chunk_id)
        return record.tier if record else MemoryTier.warm

    def _demote_to_warm(self, chunk_id: str) -> None:
        """Write a warm tier record for *chunk_id* (LRU cache overflow path)."""
        record = self._store.get_chunk_tier(chunk_id)
        if record and record.pinned:
            # Don't demote pinned chunks even on LRU overflow.
            return
        new_record = ChunkTierRecord(
            chunk_id=chunk_id,
            tier=MemoryTier.warm,
            last_accessed=record.last_accessed if record else time.time(),
            access_count=record.access_count if record else 0,
            pinned=False,
        )
        self._store.set_chunk_tier(new_record)

    def _promote_recent_to_hot(self, cutoff_ts: float) -> int:
        """Promote warm chunks accessed after *cutoff_ts* to hot.

        Args:
            cutoff_ts: Unix timestamp; chunks with ``last_accessed >= cutoff_ts``
                are candidates for hot promotion.

        Returns:
            Number of chunks promoted.
        """
        rows = (
            self._store._connection()
            .execute(
                "SELECT chunk_id FROM chunk_tiers"
                " WHERE tier = 'warm' AND last_accessed >= ? AND pinned = 0"
                " ORDER BY last_accessed DESC LIMIT ?",
                (cutoff_ts, self._tier_cfg.hot_max_chunks),
            )
            .fetchall()
        )
        ids = [r["chunk_id"] for r in rows]
        if not ids:
            return 0
        count = self._store.promote_chunks_to_tier(ids, MemoryTier.hot)
        for cid in ids:
            self._hot_cache.touch(cid)
        return count

    def _demote_stale_to_cold(self, cutoff_ts: float) -> int:
        """Demote hot/warm chunks not accessed since *cutoff_ts* to cold.

        Args:
            cutoff_ts: Unix timestamp; chunks with ``last_accessed < cutoff_ts``
                are candidates for cold demotion.

        Returns:
            Number of chunks demoted.
        """
        rows = (
            self._store._connection()
            .execute(
                "SELECT chunk_id FROM chunk_tiers"
                " WHERE tier != 'cold' AND last_accessed < ? AND pinned = 0",
                (cutoff_ts,),
            )
            .fetchall()
        )
        ids = [r["chunk_id"] for r in rows]
        if not ids:
            return 0
        count = self._store.promote_chunks_to_tier(ids, MemoryTier.cold)
        for cid in ids:
            self._hot_cache.discard(cid)
        return count
