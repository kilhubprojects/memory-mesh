"""CLIP image-embedding provider for MemoryMesh.

Real implementation using ``open-clip-torch`` (optional dependency).
When the library is not installed, all methods return ``None`` and a one-time
warning is logged.  Install with::

    pip install memorymesh[multimodal]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

_INSTALL_MSG = (
    "CLIP image embeddings require 'open-clip-torch' and 'Pillow'.  "
    "Install with: pip install memorymesh[multimodal]"
)

_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"})


class CLIPConfig(BaseModel):
    """Configuration for the CLIP embedding provider.

    Args:
        model_name: Open-CLIP model architecture (e.g. ``"ViT-B-32"``).
        pretrained: Pre-trained weights tag (e.g. ``"openai"``).
        device: Compute device.  ``"auto"`` selects CUDA when available, else CPU.
    """

    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str = "auto"


class CLIPProvider:
    """CLIP-based embedding provider for images and cross-modal text queries.

    All public methods return ``None`` gracefully when ``open-clip-torch`` or
    ``Pillow`` are not installed - no exception is raised.

    Args:
        config: Provider configuration.
    """

    def __init__(self, config: CLIPConfig | None = None) -> None:
        self.config = config or CLIPConfig()
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._device: str = self.config.device
        self._warned: bool = False

    @property
    def model_id(self) -> str:
        """Unique identifier for this provider and model."""
        return f"clip-{self.config.model_name}"

    @property
    def dimension(self) -> int:
        """Output vector dimensionality (512 for ViT-B-32)."""
        return 512

    def embed_image(self, image_path: Path) -> list[float] | None:
        """Return the CLIP visual embedding for *image_path*.

        Args:
            image_path: Absolute path to an image file.

        Returns:
            L2-normalised embedding vector, or ``None`` if dependencies are
            missing or embedding fails.
        """
        if not self._load():
            return None
        try:
            import torch
            from PIL import Image  # type: ignore[import]

            img = Image.open(image_path).convert("RGB")
            tensor = self._preprocess(img).unsqueeze(0).to(self._device)
            with torch.no_grad():
                features = self._model.encode_image(tensor)
                features /= features.norm(dim=-1, keepdim=True)
            return features[0].cpu().tolist()
        except Exception as exc:
            logger.warning(f"CLIPProvider: embed_image failed for {image_path}: {exc}")
            return None

    def embed_text(self, text: str) -> list[float] | None:
        """Return the CLIP text embedding for cross-modal search.

        Args:
            text: Query string to embed.

        Returns:
            L2-normalised embedding vector, or ``None`` if dependencies are missing.
        """
        if not self._load():
            return None
        try:
            import torch

            tokens = self._tokenizer([text]).to(self._device)
            with torch.no_grad():
                features = self._model.encode_text(tokens)
                features /= features.norm(dim=-1, keepdim=True)
            return features[0].cpu().tolist()
        except Exception as exc:
            logger.warning(f"CLIPProvider: embed_text failed: {exc}")
            return None

    def get_image_dimensions(self, image_path: Path) -> tuple[int, int] | None:
        """Return ``(width, height)`` for *image_path* using Pillow, or ``None``."""
        try:
            from PIL import Image  # type: ignore[import]

            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return None

    def _load(self) -> bool:
        """Lazily load the CLIP model.  Returns ``True`` on success."""
        if self._model is not None:
            return True
        try:
            import open_clip  # type: ignore[import]
            import torch
        except ImportError:
            if not self._warned:
                logger.warning(_INSTALL_MSG)
                self._warned = True
            return False

        device = self.config.device
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
        self._device = device
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.config.model_name,
            pretrained=self.config.pretrained,
            device=device,
        )
        self._tokenizer = open_clip.get_tokenizer(self.config.model_name)
        logger.info(
            f"CLIPProvider loaded model={self.config.model_name!r} "
            f"pretrained={self.config.pretrained!r} device={device!r}"
        )
        return True
