"""Unit tests for CrossEncoderReranker.

Uses a mock CrossEncoder so no ML model is downloaded during CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from memorymesh.core.models import SearchHit, SearchRerankerConfig
from memorymesh.search.reranker import CrossEncoderReranker


def _hit(path: str, score: float = 0.5, preview: str = "preview text") -> SearchHit:
    return SearchHit(
        path=path,
        chunk_index=0,
        score=score,
        preview=preview,
        file_type=".py",
        mtime=0.0,
        source_root="test",
    )


def _config(model: str = "cross-encoder/test", top_k: int = 20) -> SearchRerankerConfig:
    return SearchRerankerConfig(enabled=True, model=model, top_k_before_rerank=top_k)


class TestCrossEncoderReranker:
    def test_rerank_reorders_by_score(self) -> None:
        hits = [_hit("low.py", score=0.9), _hit("high.py", score=0.1)]
        mock_ce = MagicMock()
        # high.py gets score 0.95, low.py gets 0.1
        mock_ce.predict.return_value = np.array([0.1, 0.95])

        reranker = CrossEncoderReranker(_config())
        reranker._model = mock_ce

        result = reranker.rerank("query", hits, top_k=2)

        assert len(result) == 2
        assert result[0].path == "high.py"
        assert result[0].score == pytest.approx(0.95, abs=1e-5)
        assert result[1].path == "low.py"

    def test_rerank_respects_top_k(self) -> None:
        hits = [_hit(f"doc{i}.py") for i in range(5)]
        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

        reranker = CrossEncoderReranker(_config())
        reranker._model = mock_ce

        result = reranker.rerank("query", hits, top_k=3)

        assert len(result) == 3

    def test_rerank_empty_hits_returns_empty(self) -> None:
        reranker = CrossEncoderReranker(_config())
        result = reranker.rerank("query", [], top_k=5)
        assert result == []

    def test_rerank_preserves_hit_fields(self) -> None:
        hit = SearchHit(
            path="src/foo.py",
            chunk_index=3,
            score=0.1,
            preview="some text",
            file_type=".py",
            mtime=12345.0,
            source_root="myrepo",
            start_char=10,
            end_char=50,
            extended_preview="wider context",
        )
        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.8])

        reranker = CrossEncoderReranker(_config())
        reranker._model = mock_ce

        result = reranker.rerank("query", [hit], top_k=1)

        r = result[0]
        assert r.path == "src/foo.py"
        assert r.chunk_index == 3
        assert r.mtime == 12345.0
        assert r.source_root == "myrepo"
        assert r.start_char == 10
        assert r.end_char == 50
        assert r.extended_preview == "wider context"
        assert r.score == pytest.approx(0.8, abs=1e-5)

    def test_rerank_fallback_when_model_unavailable(self) -> None:
        reranker = CrossEncoderReranker(_config())
        reranker._load_failed = True  # simulate failed load

        hits = [_hit("a.py", score=0.9), _hit("b.py", score=0.1)]
        result = reranker.rerank("query", hits, top_k=2)

        # Should return original order (no reranking)
        assert [r.path for r in result] == ["a.py", "b.py"]

    def test_rerank_fallback_on_predict_exception(self) -> None:
        mock_ce = MagicMock()
        mock_ce.predict.side_effect = RuntimeError("CUDA OOM")

        reranker = CrossEncoderReranker(_config())
        reranker._model = mock_ce

        hits = [_hit("a.py"), _hit("b.py")]
        result = reranker.rerank("query", hits, top_k=2)

        # Should return original order, not crash
        assert len(result) == 2

    def test_lazy_load_skipped_when_model_none(self) -> None:
        reranker = CrossEncoderReranker(_config(model="nonexistent/model"))
        reranker._load_failed = True

        # Should not attempt to load again
        result = reranker._get_model()
        assert result is None

    def test_lazy_load_called_on_first_rerank(self) -> None:
        reranker = CrossEncoderReranker(_config())

        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.5])

        with patch(
            "memorymesh.search.reranker.CrossEncoderReranker._get_model",
            return_value=mock_ce,
        ):
            hits = [_hit("doc.py")]
            result = reranker.rerank("query", hits, top_k=1)

        assert len(result) == 1
