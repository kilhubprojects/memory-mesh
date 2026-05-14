"""Multi-modal embedding providers for MemoryMesh (Wave 5 skeleton).

This package defines the provider stubs for image (CLIP) and audio (Whisper)
modalities.  The concrete implementations are gated behind optional dependency
groups and raise :class:`RuntimeError` when the relevant packages are not
installed.

Roadmap
-------
* ``CLIPProvider``   — image embeddings via ``open-clip-torch`` (planned v0.6).
* ``WhisperProvider`` — audio transcription + embedding via ``openai-whisper``
  (planned v0.6).
* ``MultiModalFusion`` — fuses text, image, and audio embeddings into a shared
  embedding space using late-fusion RRF (planned v0.7).
"""

from memorymesh.embeddings.multimodal.clip_provider import CLIPProvider
from memorymesh.embeddings.multimodal.whisper_provider import WhisperProvider

__all__ = ["CLIPProvider", "WhisperProvider"]
