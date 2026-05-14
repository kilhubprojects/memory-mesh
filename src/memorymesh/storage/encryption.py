"""Fernet-based encryption manager for MemoryMesh.

Provides transparent encrypt/decrypt for the audit log and SQLite backup.
The key is stored in a user-owned file (chmod 600 on POSIX).  If the key
file does not exist yet, :meth:`EncryptionManager.generate_key` creates it.

Usage::

    mgr = EncryptionManager.from_config(config.encryption)
    ciphertext = mgr.encrypt(b"plaintext data")
    plaintext  = mgr.decrypt(ciphertext)

Requires ``cryptography>=42.0`` (``pip install memorymesh[encryption]``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from memorymesh.core.models import EncryptionConfig

_INSTALL_MSG = (
    "MemoryMesh encryption requires 'cryptography'.  "
    "Install with: pip install memorymesh[encryption]"
)


class EncryptionManager:
    """Symmetric encryption/decryption using Fernet (AES-128-CBC + HMAC-SHA256).

    Args:
        key_file: Path to the Fernet key file.  Must contain a
            URL-safe base64-encoded 32-byte key (as written by
            :meth:`generate_key`).

    Raises:
        ImportError: If the ``cryptography`` package is not installed.
        FileNotFoundError: If *key_file* does not exist.
        ValueError: If the key file contains an invalid key.
    """

    def __init__(self, key_file: Path) -> None:
        try:
            from cryptography.fernet import Fernet  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(_INSTALL_MSG) from exc

        self._key_file = key_file
        raw = key_file.read_bytes().strip()
        self._fernet = Fernet(raw)

    def encrypt(self, data: bytes) -> bytes:
        """Return Fernet-encrypted ciphertext for *data*.

        Args:
            data: Plaintext bytes to encrypt.

        Returns:
            Fernet token (URL-safe base64 encoded ciphertext + HMAC).
        """
        return self._fernet.encrypt(data)

    def decrypt(self, token: bytes) -> bytes:
        """Decrypt a Fernet *token* and return the original plaintext.

        Args:
            token: Fernet token as returned by :meth:`encrypt`.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            cryptography.fernet.InvalidToken: If the token is invalid or
                the key does not match.
        """
        return self._fernet.decrypt(token)

    @property
    def key_file(self) -> Path:
        """Path to the Fernet key file."""
        return self._key_file

    @classmethod
    def from_config(cls, config: EncryptionConfig) -> EncryptionManager | None:
        """Create an :class:`EncryptionManager` from an :class:`EncryptionConfig`.

        Args:
            config: Encryption configuration section from
                :class:`~memorymesh.core.models.MemoryMeshConfig`.

        Returns:
            A ready :class:`EncryptionManager`, or ``None`` if encryption is
            disabled or the ``cryptography`` package is not installed.
        """
        if not config.enabled:
            return None
        try:
            return cls(config.key_file)
        except ImportError:
            logger.warning(_INSTALL_MSG)
            return None
        except FileNotFoundError:
            logger.warning(
                f"EncryptionManager: key file not found: {config.key_file}. "
                "Run `memorymesh keygen` to generate one."
            )
            return None
        except Exception as exc:
            logger.warning(f"EncryptionManager: failed to load key: {exc}")
            return None

    @staticmethod
    def generate_key(key_file: Path) -> Path:
        """Generate a new Fernet key and write it to *key_file*.

        The parent directory is created if necessary.  On POSIX systems the
        key file and its directory are set to ``600`` / ``700`` permissions.

        Args:
            key_file: Destination path for the key file.

        Returns:
            The resolved *key_file* path.

        Raises:
            ImportError: If the ``cryptography`` package is not installed.
        """
        try:
            from cryptography.fernet import Fernet  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(_INSTALL_MSG) from exc

        key = Fernet.generate_key()
        key_file = key_file.expanduser().resolve()
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_bytes(key)

        if sys.platform != "win32":
            try:
                key_file.chmod(0o600)
                key_file.parent.chmod(0o700)
            except OSError as exc:
                logger.warning(f"EncryptionManager: could not set key file permissions: {exc}")

        logger.info(f"EncryptionManager: generated new key at {key_file}")
        return key_file
