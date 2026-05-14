"""Unit tests for EncryptionManager and encrypted AuditLogger.

Requires the ``cryptography`` package (already a main dep).
"""

from __future__ import annotations

from pathlib import Path


class TestEncryptionManager:
    def test_generate_key_creates_file(self, tmp_path: Path) -> None:
        from memorymesh.storage.encryption import EncryptionManager

        key_file = tmp_path / "secret.key"
        EncryptionManager.generate_key(key_file)
        assert key_file.exists()
        assert len(key_file.read_bytes()) > 0

    def test_round_trip_encrypt_decrypt(self, tmp_path: Path) -> None:
        from memorymesh.storage.encryption import EncryptionManager

        key_file = tmp_path / "secret.key"
        EncryptionManager.generate_key(key_file)
        mgr = EncryptionManager(key_file)

        plaintext = b"hello world"
        assert mgr.decrypt(mgr.encrypt(plaintext)) == plaintext

    def test_different_keys_cannot_decrypt(self, tmp_path: Path) -> None:
        from cryptography.fernet import InvalidToken

        from memorymesh.storage.encryption import EncryptionManager

        key_a = tmp_path / "key_a.key"
        key_b = tmp_path / "key_b.key"
        EncryptionManager.generate_key(key_a)
        EncryptionManager.generate_key(key_b)

        mgr_a = EncryptionManager(key_a)
        mgr_b = EncryptionManager(key_b)
        ciphertext = mgr_a.encrypt(b"secret")

        try:
            mgr_b.decrypt(ciphertext)
            assert False, "Expected InvalidToken"  # noqa: B011
        except InvalidToken:
            pass

    def test_from_config_returns_none_when_disabled(self, tmp_path: Path) -> None:
        from memorymesh.core.models import EncryptionConfig
        from memorymesh.storage.encryption import EncryptionManager

        cfg = EncryptionConfig(enabled=False, key_file=tmp_path / "key.key")
        assert EncryptionManager.from_config(cfg) is None

    def test_from_config_returns_manager_when_enabled(self, tmp_path: Path) -> None:
        from memorymesh.core.models import EncryptionConfig
        from memorymesh.storage.encryption import EncryptionManager

        key_file = tmp_path / "key.key"
        EncryptionManager.generate_key(key_file)
        cfg = EncryptionConfig(enabled=True, key_file=key_file)
        mgr = EncryptionManager.from_config(cfg)
        assert mgr is not None

    def test_from_config_returns_none_on_missing_key_file(self, tmp_path: Path) -> None:
        from memorymesh.core.models import EncryptionConfig
        from memorymesh.storage.encryption import EncryptionManager

        cfg = EncryptionConfig(enabled=True, key_file=tmp_path / "missing.key")
        assert EncryptionManager.from_config(cfg) is None

    def test_generate_key_overwrites_existing(self, tmp_path: Path) -> None:
        from memorymesh.storage.encryption import EncryptionManager

        key_file = tmp_path / "key.key"
        EncryptionManager.generate_key(key_file)
        old_key = key_file.read_bytes()
        EncryptionManager.generate_key(key_file)
        new_key = key_file.read_bytes()
        assert old_key != new_key


class TestAuditLoggerEncryption:
    def test_encrypted_line_is_not_plaintext_json(self, tmp_path: Path) -> None:
        from memorymesh.observability.audit import AuditLogger
        from memorymesh.storage.encryption import EncryptionManager

        key_file = tmp_path / "key.key"
        EncryptionManager.generate_key(key_file)
        mgr = EncryptionManager(key_file)
        log = tmp_path / "audit.jsonl"

        logger = AuditLogger(log, encryption=mgr)
        logger.log_query("search_memory", "test", n_results=3, latency_ms=10.0)

        raw = log.read_bytes()
        assert b"search_memory" not in raw

    def test_encrypted_line_decrypts_to_valid_json(self, tmp_path: Path) -> None:
        import json

        from memorymesh.observability.audit import AuditLogger
        from memorymesh.storage.encryption import EncryptionManager

        key_file = tmp_path / "key.key"
        EncryptionManager.generate_key(key_file)
        mgr = EncryptionManager(key_file)
        log = tmp_path / "audit.jsonl"

        logger = AuditLogger(log, encryption=mgr)
        logger.log_query("graph_memory", "q", n_results=5, latency_ms=2.5)

        token = log.read_bytes().strip()
        record = json.loads(mgr.decrypt(token))
        assert record["tool"] == "graph_memory"
        assert record["n_results"] == 5

    def test_unencrypted_log_is_valid_jsonl(self, tmp_path: Path) -> None:
        import json

        from memorymesh.observability.audit import AuditLogger

        log = tmp_path / "audit.jsonl"
        logger = AuditLogger(log)
        logger.log_query("list_sources", "x", n_results=0, latency_ms=1.0)

        record = json.loads(log.read_text())
        assert record["tool"] == "list_sources"


class TestMetadataStoreExportEncrypted:
    def test_export_creates_file(self, tmp_path: Path) -> None:
        from memorymesh.storage.encryption import EncryptionManager
        from memorymesh.storage.metadata_store import MetadataStore

        key_file = tmp_path / "key.key"
        EncryptionManager.generate_key(key_file)
        mgr = EncryptionManager(key_file)

        db_path = tmp_path / "meta.db"
        store = MetadataStore(db_path)

        out = tmp_path / "backup.enc"
        result = store.export_encrypted(out, mgr)

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_export_decrypts_to_sql_dump(self, tmp_path: Path) -> None:
        from memorymesh.storage.encryption import EncryptionManager
        from memorymesh.storage.metadata_store import MetadataStore

        key_file = tmp_path / "key.key"
        EncryptionManager.generate_key(key_file)
        mgr = EncryptionManager(key_file)

        db_path = tmp_path / "meta.db"
        store = MetadataStore(db_path)
        store.init_schema()  # creates the tables so the dump is non-trivial

        out = tmp_path / "backup.enc"
        store.export_encrypted(out, mgr)

        decrypted = mgr.decrypt(out.read_bytes())
        assert b"CREATE TABLE" in decrypted
