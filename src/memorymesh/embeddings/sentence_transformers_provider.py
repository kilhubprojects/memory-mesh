"""SentenceTransformers embedding provider.

The model is loaded lazily on first use so the daemon starts quickly even when
``sentence-transformers`` is installed but not yet needed.

Device selection follows the ``device`` field in ``EmbeddingsConfig``:
- ``"auto"`` -> CUDA if available, then MPS, then CPU.
- ``"cpu"`` / ``"cuda"`` / ``"mps"`` -> explicit override.

Batching is handled internally by SentenceTransformers; the ``batch_size``
config controls how many texts are encoded per forward pass.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from memorymesh.core.exceptions import EmbeddingError
from memorymesh.embeddings.base import EmbeddingProvider

if TYPE_CHECKING:
    from memorymesh.core.models import EmbeddingsConfig


class SentenceTransformersProvider(EmbeddingProvider):
    """EmbeddingProvider backed by the ``sentence-transformers`` library.

    Args:
        config: Embedding settings from ``config.yaml``.  Uses defaults when
            ``None``.
    """

    def __init__(self, config: EmbeddingsConfig | None = None) -> None:
        from memorymesh.core.models import EmbeddingsConfig as _Cfg

        self._config: EmbeddingsConfig = config or _Cfg()
        self._model: Any = None  # lazy-loaded SentenceTransformer

    @property
    def model_id(self) -> str:
        """The sentence-transformers model identifier from config."""
        return self._config.model

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors produced by this model."""
        return len(self.embed_query("dim probe"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* in batches.

        Args:
            texts: List of document strings to embed.

        Returns:
            List of float vectors.

        Raises:
            :class:`~memorymesh.core.exceptions.EmbeddingError`: On model failure.
        """
        if not texts:
            return []
        model = self._get_model()
        try:
            vectors = model.encode(
                texts,
                batch_size=self._config.batch_size,
                normalize_embeddings=self._config.normalize,
                show_progress_bar=False,
            )
            logger.debug(f"SentenceTransformers: embedded {len(texts)} document(s)")
            return [cast(list[float], v.tolist()) for v in vectors]
        except Exception as exc:
            raise EmbeddingError(f"embed_documents failed: {exc}") from exc

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: Query string.

        Returns:
            Float vector.

        Raises:
            :class:`~memorymesh.core.exceptions.EmbeddingError`: On model failure.
        """
        model = self._get_model()
        try:
            vector = model.encode(
                [text],
                batch_size=1,
                normalize_embeddings=self._config.normalize,
                show_progress_bar=False,
            )
            return cast(list[float], vector[0].tolist())
        except Exception as exc:
            raise EmbeddingError(f"embed_query failed: {exc}") from exc

    def _get_model(self) -> Any:
        """Lazy-load and cache the SentenceTransformer model."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed; run 'uv add sentence-transformers'"
            ) from exc

        device = self._resolve_device()
        logger.info(
            f"SentenceTransformers: loading model {self._config.model!r} on device={device!r}"
        )
        try:
            self._model = SentenceTransformer(self._config.model, device=device)
        except Exception as exc:
            raise EmbeddingError(f"Failed to load model {self._config.model!r}: {exc}") from exc

        logger.info(f"SentenceTransformers: model {self._config.model!r} ready")
        return self._model

    def _resolve_device(self) -> str:
        """Resolve ``"auto"`` to an actual device string."""
        device = self._config.device
        if device != "auto":
            return device

        try:
            import torch  # type: ignore[import-untyped]

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            logger.debug("Torch is not installed; defaulting sentence-transformers to CPU")
        return "cpu"
