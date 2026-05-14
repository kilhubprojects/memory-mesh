"""Whisper audio transcription provider for MemoryMesh.

Real implementation using ``faster-whisper`` (optional dependency).
When the library is not installed, ``transcribe()`` returns ``None`` and a
one-time warning is logged.  Install with::

    pip install memorymesh[multimodal]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel

_INSTALL_MSG = (
    "Whisper audio transcription requires 'faster-whisper'.  "
    "Install with: pip install memorymesh[multimodal]"
)

_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".mkv", ".webm"}
)


class WhisperConfig(BaseModel):
    """Configuration for the Whisper transcription provider.

    Args:
        model_size: Faster-whisper model size.
        device: Compute device.  ``"auto"`` selects CUDA when available, else CPU.
        language: ISO-639-1 language code (e.g. ``"en"``).  ``None`` = auto-detect.
    """

    model_size: Literal["tiny", "base", "small", "medium", "large-v3"] = "base"
    device: str = "auto"
    language: str | None = None


class WhisperProvider:
    """Transcribes audio/video files to text using faster-whisper.

    Args:
        config: Provider configuration.
    """

    def __init__(self, config: WhisperConfig | None = None) -> None:
        self.config = config or WhisperConfig()
        self._model: Any = None
        self._warned: bool = False

    @property
    def supported_extensions(self) -> frozenset[str]:
        """File extensions this provider can transcribe."""
        return _SUPPORTED_EXTENSIONS

    def transcribe(self, audio_path: Path) -> str | None:
        """Transcribe *audio_path* to plain text.

        Args:
            audio_path: Absolute path to the audio/video file.

        Returns:
            Transcribed text, or ``None`` if faster-whisper is not installed or
            transcription fails.
        """
        if not self._load():
            return None
        try:
            segments, _ = self._model.transcribe(
                str(audio_path),
                language=self.config.language,
                beam_size=5,
            )
            text = "".join(seg.text for seg in segments).strip()
            logger.info(f"WhisperProvider transcribed {audio_path.name!r}: {len(text)} chars")
            return text or None
        except Exception as exc:
            logger.warning(f"WhisperProvider: transcription failed for {audio_path}: {exc}")
            return None

    def _load(self) -> bool:
        """Lazily load the Whisper model.  Returns ``True`` on success."""
        if self._model is not None:
            return True
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]
        except ImportError:
            if not self._warned:
                logger.warning(_INSTALL_MSG)
                self._warned = True
            return False

        device = self.config.device
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        compute_type = "float16" if device == "cuda" else "int8"
        self._model = WhisperModel(
            self.config.model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info(f"WhisperProvider loaded model={self.config.model_size!r} device={device!r}")
        return True
