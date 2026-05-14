"""Unit tests for Wave 3 tiered memory - MetadataStore extensions and TieredMemoryManager."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from memorymesh.core.models import (
    ChunkTierRecord,
    Entity,
    EpisodicEvent,
    ForgettingConfig,
    MemoryTier,
    MemoryTierConfig,
)
from memorymesh.storage.metadata_store import MetadataStore
from memorymesh.storage.tiered import TieredMemoryManager


@pytest.fixture()
def store(tmp_path: Path) -> MetadataStore:
    s = MetadataStore(tmp_path / "test.sqlite3")
    s.init_schema()
    return s


@pytest.fixture()
def manager(store: MetadataStore) -> TieredMemoryManager:
    return TieredMemoryManager(
        store=store,
        tier_config=MemoryTierConfig(hot_tier_days=7, cold_tier_days=90, hot_max_chunks=5),
        forgetting_config=ForgettingConfig(
            enabled=True, decay_half_life_days=90.0, min_score_floor=0.1
        ),
    )


class TestChunkTiersSchema:
    def test_new_tables_exist(self, store: MetadataStore) -> None:
        rows = (
            store._connection()
            .execute("SELECT name FROM sqlite_master WHERE type='table'")
            .fetchall()
        )
        names = {r["name"] for r in rows}
        assert "chunk_tiers" in names
        assert "episodic_events" in names
        assert "entities" in names
        assert "entity_mentions" in names


class TestSetAndGetChunkTier:
    def test_roundtrip(self, store: MetadataStore) -> None:
        rec = ChunkTierRecord(
            chunk_id="doc.txt:0",
            tier=MemoryTier.hot,
            last_accessed=1_000_000.0,
            access_count=5,
            pinned=True,
        )
        store.set_chunk_tier(rec)
        result = store.get_chunk_tier("doc.txt:0")
        assert result is not None
        assert result.tier == MemoryTier.hot
        assert result.access_count == 5
        assert result.pinned is True

    def test_missing_returns_none(self, store: MetadataStore) -> None:
        assert store.get_chunk_tier("nonexistent:0") is None

    def test_upsert_updates(self, store: MetadataStore) -> None:
        rec = ChunkTierRecord(chunk_id="f.txt:0", tier=MemoryTier.warm)
        store.set_chunk_tier(rec)
        rec2 = ChunkTierRecord(chunk_id="f.txt:0", tier=MemoryTier.cold, access_count=99)
        store.set_chunk_tier(rec2)
        result = store.get_chunk_tier("f.txt:0")
        assert result is not None
        assert result.tier == MemoryTier.cold
        assert result.access_count == 99


class TestRecordChunkAccess:
    def test_creates_row_on_first_access(self, store: MetadataStore) -> None:
        store.record_chunk_access("newfile.txt:3")
        rec = store.get_chunk_tier("newfile.txt:3")
        assert rec is not None
        assert rec.access_count == 1
        assert rec.tier == MemoryTier.warm

    def test_increments_count(self, store: MetadataStore) -> None:
        for _ in range(5):
            store.record_chunk_access("doc.txt:1")
        rec = store.get_chunk_tier("doc.txt:1")
        assert rec is not None
        assert rec.access_count == 5

    def test_updates_last_accessed(self, store: MetadataStore) -> None:
        before = time.time()
        store.record_chunk_access("doc.txt:2")
        after = time.time()
        rec = store.get_chunk_tier("doc.txt:2")
        assert rec is not None
        assert before <= rec.last_accessed <= after


class TestListChunksByTier:
    def test_filters_by_tier(self, store: MetadataStore) -> None:
        store.set_chunk_tier(ChunkTierRecord(chunk_id="a:0", tier=MemoryTier.hot))
        store.set_chunk_tier(ChunkTierRecord(chunk_id="b:0", tier=MemoryTier.warm))
        store.set_chunk_tier(ChunkTierRecord(chunk_id="c:0", tier=MemoryTier.cold))

        hot = store.list_chunks_by_tier(MemoryTier.hot)
        assert len(hot) == 1
        assert hot[0].chunk_id == "a:0"

    def test_limit_respected(self, store: MetadataStore) -> None:
        for i in range(10):
            store.set_chunk_tier(ChunkTierRecord(chunk_id=f"x:{i}", tier=MemoryTier.warm))
        result = store.list_chunks_by_tier(MemoryTier.warm, limit=3)
        assert len(result) == 3


class TestPromoteChunksToTier:
    def test_bulk_promotion(self, store: MetadataStore) -> None:
        ids = ["a:0", "b:0", "c:0"]
        for cid in ids:
            store.set_chunk_tier(ChunkTierRecord(chunk_id=cid, tier=MemoryTier.warm))
        n = store.promote_chunks_to_tier(ids, MemoryTier.cold)
        assert n == 3
        for cid in ids:
            assert store.get_chunk_tier(cid).tier == MemoryTier.cold  # type: ignore[union-attr]

    def test_skips_pinned(self, store: MetadataStore) -> None:
        store.set_chunk_tier(ChunkTierRecord(chunk_id="pinned:0", tier=MemoryTier.hot, pinned=True))
        n = store.promote_chunks_to_tier(["pinned:0"], MemoryTier.cold)
        assert n == 0
        assert store.get_chunk_tier("pinned:0").tier == MemoryTier.hot  # type: ignore[union-attr]


class TestTieredMemoryManagerAccess:
    def test_record_access_promotes_to_hot_cache(self, manager: TieredMemoryManager) -> None:
        manager.record_access("doc.txt:0")
        assert manager.get_tier("doc.txt:0") == MemoryTier.hot

    def test_lru_evicts_oldest(self, manager: TieredMemoryManager) -> None:
        # Fill the LRU cache (capacity=5 from fixture).
        for i in range(5):
            manager.record_access(f"file{i}.txt:0")
        # Adding a 6th should evict the oldest (file0.txt:0 -> warm).
        manager.record_access("file5.txt:0")
        # The evicted chunk should no longer be in the hot in-process cache.
        assert "file0.txt:0" not in manager._hot_cache


class TestPinUnpin:
    def test_pin_sets_pinned(self, manager: TieredMemoryManager, store: MetadataStore) -> None:
        manager.pin("important.txt:0")
        rec = store.get_chunk_tier("important.txt:0")
        assert rec is not None
        assert rec.pinned is True
        assert rec.tier == MemoryTier.hot

    def test_unpin_clears_pinned(self, manager: TieredMemoryManager, store: MetadataStore) -> None:
        manager.pin("x.txt:0")
        manager.unpin("x.txt:0")
        rec = store.get_chunk_tier("x.txt:0")
        assert rec is not None
        assert rec.pinned is False

    def test_unpin_nonexistent_is_noop(self, manager: TieredMemoryManager) -> None:
        # Should not raise.
        manager.unpin("does_not_exist.txt:999")


class TestForget:
    def test_forget_sets_cold(self, manager: TieredMemoryManager, store: MetadataStore) -> None:
        manager.record_access("doc.txt:1")
        manager.forget("doc.txt:1")
        rec = store.get_chunk_tier("doc.txt:1")
        assert rec is not None
        assert rec.tier == MemoryTier.cold
        assert rec.pinned is False

    def test_forget_removes_from_hot_cache(self, manager: TieredMemoryManager) -> None:
        manager.record_access("doc.txt:2")
        manager.forget("doc.txt:2")
        assert "doc.txt:2" not in manager._hot_cache


class TestApplyDecay:
    def test_warm_chunk_not_decayed(
        self, manager: TieredMemoryManager, store: MetadataStore
    ) -> None:
        store.set_chunk_tier(ChunkTierRecord(chunk_id="w.txt:0", tier=MemoryTier.warm))
        result = manager.apply_decay([("w.txt:0", 1.0)])
        assert result[0][1] == pytest.approx(1.0)

    def test_cold_chunk_decayed(self, manager: TieredMemoryManager, store: MetadataStore) -> None:
        # Simulate a chunk last accessed 180 days ago -> 2 half-lives -> factor ~= 0.25.
        old_ts = time.time() - 180 * 86_400
        store.set_chunk_tier(
            ChunkTierRecord(
                chunk_id="old.txt:0",
                tier=MemoryTier.cold,
                last_accessed=old_ts,
                pinned=False,
            )
        )
        result = manager.apply_decay([("old.txt:0", 1.0)])
        assert result[0][1] < 0.5

    def test_pinned_cold_not_decayed(
        self, manager: TieredMemoryManager, store: MetadataStore
    ) -> None:
        old_ts = time.time() - 180 * 86_400
        store.set_chunk_tier(
            ChunkTierRecord(
                chunk_id="pinned_old.txt:0",
                tier=MemoryTier.cold,
                last_accessed=old_ts,
                pinned=True,
            )
        )
        result = manager.apply_decay([("pinned_old.txt:0", 1.0)])
        assert result[0][1] == pytest.approx(1.0)

    def test_floor_respected(self, manager: TieredMemoryManager, store: MetadataStore) -> None:
        # Extremely old chunk -> should hit the 0.1 floor.
        ancient_ts = time.time() - 99_999 * 86_400
        store.set_chunk_tier(
            ChunkTierRecord(
                chunk_id="ancient.txt:0",
                tier=MemoryTier.cold,
                last_accessed=ancient_ts,
            )
        )
        result = manager.apply_decay([("ancient.txt:0", 1.0)])
        assert result[0][1] >= 0.1

    def test_decay_disabled(self, store: MetadataStore) -> None:
        mgr = TieredMemoryManager(
            store=store,
            forgetting_config=ForgettingConfig(enabled=False),
        )
        old_ts = time.time() - 180 * 86_400
        store.set_chunk_tier(
            ChunkTierRecord(chunk_id="x.txt:0", tier=MemoryTier.cold, last_accessed=old_ts)
        )
        result = mgr.apply_decay([("x.txt:0", 1.0)])
        assert result[0][1] == pytest.approx(1.0)


class TestEpisodicEvents:
    def _event(self, **kwargs: object) -> EpisodicEvent:
        defaults: dict[str, object] = {
            "event_id": "evt-001",
            "timestamp": time.time(),
            "event_type": "retrieval",
            "source": "/data/notes",
            "chunk_ids": ["a.txt:0", "b.txt:1"],
            "client_id": "agent-alpha",
            "metadata": {"query_hash": "abc"},
        }
        defaults.update(kwargs)
        return EpisodicEvent(**defaults)  # type: ignore[arg-type]

    def test_upsert_and_list(self, store: MetadataStore) -> None:
        store.upsert_episodic_event(self._event())
        events = store.list_episodic_events()
        assert len(events) == 1
        assert events[0].event_id == "evt-001"
        assert events[0].chunk_ids == ["a.txt:0", "b.txt:1"]

    def test_filter_by_event_type(self, store: MetadataStore) -> None:
        store.upsert_episodic_event(self._event(event_id="e1", event_type="retrieval"))
        store.upsert_episodic_event(self._event(event_id="e2", event_type="index"))
        result = store.list_episodic_events(event_type="index")
        assert len(result) == 1
        assert result[0].event_id == "e2"

    def test_filter_by_since(self, store: MetadataStore) -> None:
        now = time.time()
        store.upsert_episodic_event(self._event(event_id="old", timestamp=now - 1000))
        store.upsert_episodic_event(self._event(event_id="new", timestamp=now))
        result = store.list_episodic_events(since=now - 1)
        ids = [e.event_id for e in result]
        assert "new" in ids
        assert "old" not in ids

    def test_limit_respected(self, store: MetadataStore) -> None:
        for i in range(10):
            store.upsert_episodic_event(self._event(event_id=f"e{i}"))
        result = store.list_episodic_events(limit=3)
        assert len(result) == 3

    def test_metadata_roundtrip(self, store: MetadataStore) -> None:
        meta = {"key": "value", "number": 42}
        store.upsert_episodic_event(self._event(metadata=meta))
        result = store.list_episodic_events()
        assert result[0].metadata == meta


class TestEntities:
    def test_upsert_and_list(self, store: MetadataStore) -> None:
        entity = Entity(
            name="alice",
            entity_type="person",
            mention_count=3,
            first_seen=1_000_000.0,
            last_seen=2_000_000.0,
        )
        store.upsert_entity(entity)
        results = store.list_entities()
        assert len(results) == 1
        assert results[0].name == "alice"
        assert results[0].mention_count == 3

    def test_upsert_increments_mention_count(self, store: MetadataStore) -> None:
        e = Entity(name="memorymesh", entity_type="project", mention_count=2)
        store.upsert_entity(e)
        store.upsert_entity(e)
        results = store.list_entities()
        assert results[0].mention_count == 4

    def test_filter_by_type(self, store: MetadataStore) -> None:
        store.upsert_entity(Entity(name="alice", entity_type="person"))
        store.upsert_entity(Entity(name="vectordb", entity_type="concept"))
        persons = store.list_entities(entity_type="person")
        assert all(e.entity_type == "person" for e in persons)
        assert len(persons) == 1

    def test_min_mentions_filter(self, store: MetadataStore) -> None:
        store.upsert_entity(Entity(name="rare", entity_type="concept", mention_count=1))
        store.upsert_entity(Entity(name="common", entity_type="concept", mention_count=10))
        result = store.list_entities(min_mentions=5)
        names = [e.name for e in result]
        assert "common" in names
        assert "rare" not in names

    def test_entity_mentions_roundtrip(self, store: MetadataStore) -> None:
        store.upsert_entity(Entity(name="alice", entity_type="person"))
        store.add_entity_mention("alice", "person", "notes.txt:0")
        store.add_entity_mention("alice", "person", "notes.txt:1")
        chunks = store.get_entity_chunks("alice", "person")
        assert set(chunks) == {"notes.txt:0", "notes.txt:1"}

    def test_duplicate_mention_ignored(self, store: MetadataStore) -> None:
        store.upsert_entity(Entity(name="bob", entity_type="person"))
        store.add_entity_mention("bob", "person", "doc.txt:0")
        store.add_entity_mention("bob", "person", "doc.txt:0")  # duplicate
        chunks = store.get_entity_chunks("bob", "person")
        assert chunks.count("doc.txt:0") == 1
