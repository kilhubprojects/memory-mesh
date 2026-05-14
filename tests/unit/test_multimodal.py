"""Unit tests for multimodal providers (CLIP + Whisper) and routing.

All ML model dependencies are mocked - no actual GPU or model weights needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from memorymesh.embeddings.multimodal.clip_provider import CLIPConfig, CLIPProvider
from memorymesh.embeddings.multimodal.whisper_provider import WhisperConfig, WhisperProvider


class TestCLIPProvider:
    def test_embed_image_returns_none_when_open_clip_missing(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        provider = CLIPProvider(CLIPConfig())
        with patch.dict(sys.modules, {"open_clip": None, "torch": None}):
            provider._model = None
            provider._warned = False
            result = provider.embed_image(img)
        assert result is None

    def test_embed_text_returns_none_when_open_clip_missing(self) -> None:
        provider = CLIPProvider(CLIPConfig())
        with patch.dict(sys.modules, {"open_clip": None, "torch": None}):
            provider._model = None
            provider._warned = False
            result = provider.embed_text("a cat on a mat")
        assert result is None

    def test_warn_flag_prevents_duplicate_logs(self) -> None:
        provider = CLIPProvider(CLIPConfig())
        with patch.dict(sys.modules, {"open_clip": None, "torch": None}):
            provider._model = None
            provider._warned = False
            provider._load()
            assert provider._warned is True
            provider._load()
            assert provider._warned is True  # still True, no second message

    def test_model_id_reflects_config(self) -> None:
        cfg = CLIPConfig(model_name="ViT-L-14")
        assert CLIPProvider(cfg).model_id == "clip-ViT-L-14"

    def test_dimension_is_512(self) -> None:
        assert CLIPProvider().dimension == 512

    def test_embed_image_returns_none_on_exception(self, tmp_path: Path) -> None:
        img = tmp_path / "bad.jpg"
        img.write_bytes(b"not an image")

        provider = CLIPProvider(CLIPConfig())
        fake_model = MagicMock()
        fake_model.encode_image.side_effect = RuntimeError("corrupt image")
        provider._model = fake_model
        provider._preprocess = MagicMock()
        provider._device = "cpu"

        result = provider.embed_image(img)
        assert result is None


class TestWhisperProvider:
    def test_transcribe_returns_none_when_faster_whisper_missing(self, tmp_path: Path) -> None:
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\xff\xfb")

        provider = WhisperProvider(WhisperConfig())
        with patch.dict(sys.modules, {"faster_whisper": None}):
            provider._model = None
            provider._warned = False
            result = provider.transcribe(audio)
        assert result is None

    def test_warn_flag_set_after_first_load_attempt(self, tmp_path: Path) -> None:
        provider = WhisperProvider(WhisperConfig())
        with patch.dict(sys.modules, {"faster_whisper": None}):
            provider._model = None
            provider._warned = False
            provider._load()
            assert provider._warned is True

    def test_supported_extensions(self) -> None:
        provider = WhisperProvider()
        assert ".mp3" in provider.supported_extensions
        assert ".mkv" in provider.supported_extensions
        assert ".txt" not in provider.supported_extensions

    def test_transcribe_joins_segments(self, tmp_path: Path) -> None:
        audio = tmp_path / "speech.wav"
        audio.write_bytes(b"RIFF")

        provider = WhisperProvider(WhisperConfig())
        seg1, seg2 = MagicMock(), MagicMock()
        seg1.text = "Hello"
        seg2.text = " world"
        fake_model = MagicMock()
        fake_model.transcribe.return_value = ([seg1, seg2], MagicMock())
        provider._model = fake_model

        result = provider.transcribe(audio)
        assert result == "Hello world"

    def test_transcribe_returns_none_on_empty_transcript(self, tmp_path: Path) -> None:
        audio = tmp_path / "silent.mp3"
        audio.write_bytes(b"\xff\xfb")

        provider = WhisperProvider(WhisperConfig())
        seg = MagicMock()
        seg.text = "  "
        fake_model = MagicMock()
        fake_model.transcribe.return_value = ([seg], MagicMock())
        provider._model = fake_model

        result = provider.transcribe(audio)
        assert result is None


class TestFileIndexerMultimodalSkip:
    def _make_indexer(self, tmp_path: Path) -> object:
        from memorymesh.core.models import MemoryMeshConfig
        from memorymesh.indexer.file_indexer import FileIndexer
        from memorymesh.storage.bm25_index import BM25Index
        from memorymesh.storage.metadata_store import MetadataStore
        from memorymesh.storage.vector_store import ChromaVectorStore

        cfg = MemoryMeshConfig()
        vs = ChromaVectorStore(db_path=tmp_path / "chroma", embedding_model_id="test")
        ms = MetadataStore(tmp_path / "meta.db")
        bm25 = BM25Index(tmp_path / "bm25.pkl")
        provider = MagicMock()
        provider.model_id = "test-model"
        return FileIndexer(
            config=cfg,
            vector_store=vs,
            metadata_store=ms,
            embedding_provider=provider,
            bm25_index=bm25,
            clip_provider=None,
            whisper_provider=None,
        )

    def test_image_returns_unsupported_when_clip_is_none(self, tmp_path: Path) -> None:
        indexer = self._make_indexer(tmp_path)
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0")

        result = indexer.index_file(img, source_name="test")
        assert result.status == "unsupported"

    def test_audio_returns_unsupported_when_whisper_is_none(self, tmp_path: Path) -> None:
        indexer = self._make_indexer(tmp_path)
        audio = tmp_path / "talk.mp3"
        audio.write_bytes(b"\xff\xfb")

        result = indexer.index_file(audio, source_name="test")
        assert result.status == "unsupported"


class TestSearchEngineModality:
    def _engine_with_hits(self, hits: list) -> object:
        from memorymesh.core.models import SearchConfig, SearchHybridConfig
        from memorymesh.search.engine import SearchEngine

        vs = MagicMock()
        vs.search.return_value = hits
        bm25 = MagicMock()
        bm25.search.return_value = []
        provider = MagicMock()
        provider.embed_query.return_value = [0.1] * 8
        cfg = SearchConfig(hybrid=SearchHybridConfig(enabled=False))
        return SearchEngine(vs, bm25, provider, config=cfg)

    def _hit(self, path: str, file_type: str = ".txt") -> object:
        from memorymesh.core.models import SearchHit

        return SearchHit(
            path=path,
            chunk_index=0,
            score=0.9,
            preview="p",
            file_type=file_type,
            mtime=1000.0,
            source_root="docs",
        )

    def test_modality_all_returns_everything(self) -> None:
        hits = [self._hit("/a.txt"), self._hit("/b.jpg", ".jpg")]
        engine = self._engine_with_hits(hits)
        resp = engine.search("q", mode="dense", modality="all")
        assert len(resp.hits) == 2

    def test_modality_image_keeps_only_images(self) -> None:
        hits = [
            self._hit("/a.txt"),
            self._hit("/b.jpg", ".jpg"),
            self._hit("/c.png", ".png"),
        ]
        engine = self._engine_with_hits(hits)
        resp = engine.search("q", mode="dense", modality="image")
        assert all(h.file_type in {".jpg", ".png"} for h in resp.hits)
        assert len(resp.hits) == 2

    def test_modality_text_excludes_images(self) -> None:
        hits = [self._hit("/a.txt"), self._hit("/b.jpg", ".jpg")]
        engine = self._engine_with_hits(hits)
        resp = engine.search("q", mode="dense", modality="text")
        assert all(h.file_type == ".txt" for h in resp.hits)

    def test_default_modality_is_all(self) -> None:
        hits = [self._hit("/a.txt"), self._hit("/b.jpg", ".jpg")]
        engine = self._engine_with_hits(hits)
        resp = engine.search("q", mode="dense")
        assert len(resp.hits) == 2
