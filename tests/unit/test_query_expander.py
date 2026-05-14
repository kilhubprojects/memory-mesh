"""Unit tests for QueryExpander.

All tests use Mode A (lexical expansion) only — no Ollama required.
HyDE mode is tested with a mocked urllib call.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from memorymesh.core.models import OllamaConfig, QueryExpansionConfig
from memorymesh.search.query_expander import QueryExpander, _stem, merge_hit_lists


def _config(n_variants: int = 2, use_hyde: bool = False) -> QueryExpansionConfig:
    return QueryExpansionConfig(
        enabled=True,
        use_hyde=use_hyde,
        n_lexical_variants=n_variants,
    )


def _provider() -> MagicMock:
    p = MagicMock()
    p.embed_query.return_value = [0.1, 0.2, 0.3]
    return p


class TestStemmer:
    def test_strips_ing(self) -> None:
        assert _stem("indexing") == "index"

    def test_strips_ed(self) -> None:
        assert _stem("indexed") == "index"

    def test_strips_er(self) -> None:
        assert _stem("indexer") == "indexer"[:-2]  # "index"

    def test_short_word_unchanged(self) -> None:
        # Word too short to strip suffix safely
        assert _stem("is") == "is"

    def test_unknown_word_unchanged(self) -> None:
        # "frobnicator" has no matching stem rule → returned unchanged
        assert _stem("frobnicator") == "frobnicator"


class TestLexicalExpansion:
    def test_returns_original_as_first(self) -> None:
        expander = QueryExpander(_config(), _provider())
        result = expander.expand_queries("index files")
        assert result[0] == "index files"

    def test_generates_variants(self) -> None:
        expander = QueryExpander(_config(n_variants=2), _provider())
        result = expander.expand_queries("search query")
        assert len(result) >= 2

    def test_variants_are_unique(self) -> None:
        expander = QueryExpander(_config(n_variants=3), _provider())
        result = expander.expand_queries("indexing flow")
        assert len(result) == len(set(result))

    def test_unknown_query_returns_only_original(self) -> None:
        expander = QueryExpander(_config(), _provider())
        result = expander.expand_queries("xyzzy frobnicator")
        assert result == ["xyzzy frobnicator"]

    def test_n_variants_zero_returns_only_original(self) -> None:
        expander = QueryExpander(_config(n_variants=0), _provider())
        result = expander.expand_queries("index files")
        assert result == ["index files"]

    def test_known_synonym_appears_in_variant(self) -> None:
        expander = QueryExpander(_config(n_variants=3), _provider())
        result = expander.expand_queries("search files")
        # "search" should produce a variant with "query", "retrieve" etc.
        all_text = " ".join(result)
        assert any(kw in all_text for kw in ("query", "retrieve", "lookup", "find"))

    def test_indexing_produces_index_variant(self) -> None:
        expander = QueryExpander(_config(n_variants=2), _provider())
        result = expander.expand_queries("indexing flow")
        # At least one variant should substitute "indexing" → "index" or synonym
        assert len(result) >= 2
        assert any("index" in v for v in result[1:])


class TestExpandVectors:
    def test_no_hyde_returns_empty(self) -> None:
        expander = QueryExpander(_config(use_hyde=False), _provider())
        assert expander.expand_vectors("query") == []

    def test_hyde_disabled_ollama_returns_empty(self) -> None:
        ollama = OllamaConfig(enabled=False)
        expander = QueryExpander(_config(use_hyde=True), _provider(), ollama_config=ollama)
        assert expander.expand_vectors("query") == []

    def test_hyde_with_ollama_returns_vector(self) -> None:
        ollama = OllamaConfig(enabled=True, base_url="http://localhost:11434", model="llama3")
        provider = _provider()
        expander = QueryExpander(_config(use_hyde=True), provider, ollama_config=ollama)

        fake_response = b'{"response": "A hypothetical passage about indexing."}'
        with patch("urllib.request.urlopen") as mock_open:
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_ctx.read.return_value = fake_response
            mock_open.return_value = mock_ctx

            result = expander.expand_vectors("how does indexing work?")

        assert len(result) == 1
        assert result[0] == [0.1, 0.2, 0.3]
        provider.embed_query.assert_called_once()

    def test_hyde_ollama_unreachable_returns_empty(self) -> None:
        import urllib.error

        ollama = OllamaConfig(enabled=True)
        expander = QueryExpander(_config(use_hyde=True), _provider(), ollama_config=ollama)

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = expander.expand_vectors("query")

        assert result == []


class TestMergeHitLists:
    def test_merge_deduplicates(self) -> None:
        from memorymesh.core.models import SearchHit

        hit_a = SearchHit(
            path="a.py",
            chunk_index=0,
            score=0.9,
            preview="",
            file_type=".py",
            mtime=0.0,
            source_root="",
        )
        hit_b = SearchHit(
            path="b.py",
            chunk_index=0,
            score=0.5,
            preview="",
            file_type=".py",
            mtime=0.0,
            source_root="",
        )
        # hit_a appears in both lists
        result = merge_hit_lists([[hit_a, hit_b], [hit_a]])
        paths = [h.path for h in result]
        assert paths.count("a.py") == 1

    def test_merge_promotes_shared_hits(self) -> None:
        from memorymesh.core.models import SearchHit

        def _h(path: str) -> SearchHit:
            return SearchHit(
                path=path,
                chunk_index=0,
                score=0.1,
                preview="",
                file_type=".py",
                mtime=0.0,
                source_root="",
            )

        shared = _h("shared.py")
        unique_a = _h("unique_a.py")
        unique_b = _h("unique_b.py")

        result = merge_hit_lists([[shared, unique_a], [shared, unique_b]])
        # shared should rank first (appears in both lists → higher RRF score)
        assert result[0].path == "shared.py"
