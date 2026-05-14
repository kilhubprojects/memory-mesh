"""Unit tests for OllamaClient.

All tests mock urllib.request to avoid requiring a running Ollama server.
"""

from __future__ import annotations

import json
import unittest.mock as mock
import urllib.error

from memorymesh.core.models import OllamaConfig
from memorymesh.llm.ollama_client import OllamaClient


def _make_response(body: dict, status: int = 200) -> mock.MagicMock:
    """Create a fake urllib response context manager."""
    m = mock.MagicMock()
    m.__enter__ = mock.Mock(return_value=m)
    m.__exit__ = mock.Mock(return_value=False)
    m.read.return_value = json.dumps(body).encode()
    m.status = status
    return m


def _enabled_config(**kwargs: object) -> OllamaConfig:
    defaults: dict[str, object] = {
        "enabled": True,
        "base_url": "http://localhost:11434",
        "model": "llama3",
        "timeout_s": 5,
    }
    defaults.update(kwargs)  # type: ignore[arg-type]
    return OllamaConfig(**defaults)


def _disabled_config() -> OllamaConfig:
    return OllamaConfig(
        enabled=False, base_url="http://localhost:11434", model="llama3", timeout_s=5
    )


# is_available


class TestIsAvailable:
    def test_returns_true_when_server_up(self) -> None:
        client = OllamaClient(_enabled_config())
        resp = _make_response({"models": []})
        with mock.patch("urllib.request.urlopen", return_value=resp):
            assert client.is_available() is True

    def test_returns_false_when_server_down(self) -> None:
        client = OllamaClient(_enabled_config())
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            assert client.is_available() is False

    def test_returns_false_when_disabled(self) -> None:
        client = OllamaClient(_disabled_config())
        with mock.patch("urllib.request.urlopen") as m:
            assert client.is_available() is False
            m.assert_not_called()


# generate


class TestGenerate:
    def test_returns_response_text(self) -> None:
        client = OllamaClient(_enabled_config())
        resp = _make_response({"response": "Generated text here."})
        with mock.patch("urllib.request.urlopen", return_value=resp):
            result = client.generate("Hello")
        assert result == "Generated text here."

    def test_returns_empty_string_when_disabled(self) -> None:
        client = OllamaClient(_disabled_config())
        with mock.patch("urllib.request.urlopen") as m:
            result = client.generate("Hello")
        assert result == ""
        m.assert_not_called()

    def test_returns_empty_string_on_timeout(self) -> None:
        client = OllamaClient(_enabled_config())
        with mock.patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            result = client.generate("Hello")
        assert result == ""

    def test_returns_empty_string_on_url_error(self) -> None:
        client = OllamaClient(_enabled_config())
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = client.generate("Hello")
        assert result == ""

    def test_model_override_used(self) -> None:
        client = OllamaClient(_enabled_config(model="mistral"))
        resp = _make_response({"response": "ok"})
        captured: list[bytes] = []

        def fake_urlopen(req: object, timeout: int = 0) -> object:
            captured.append(req.data)  # type: ignore[union-attr]
            return resp

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client.generate("test prompt", model="llama3")

        payload = json.loads(captured[0].decode())
        assert payload["model"] == "llama3"

    def test_warning_flag_set_on_first_call(self) -> None:
        """The internal _warned_generate flag must be False before and True after."""
        client = OllamaClient(_enabled_config())
        assert client._warned_generate is False
        resp = _make_response({"response": "ok"})
        with mock.patch("urllib.request.urlopen", return_value=resp):
            client.generate("Hello")
        assert client._warned_generate is True

    def test_warning_only_fires_once(self) -> None:
        """The _warned_generate flag ensures the warning is emitted only once."""
        client = OllamaClient(_enabled_config())
        resp = _make_response({"response": "ok"})

        warnings_emitted: list[str] = []

        def _patched_warning(msg: str) -> None:
            warnings_emitted.append(msg)

        with (
            mock.patch("urllib.request.urlopen", return_value=resp),
            mock.patch("memorymesh.llm.ollama_client.logger") as mock_logger,
        ):
            mock_logger.warning.side_effect = _patched_warning
            client.generate("Hello")
            client.generate("Hello again")

        ollama_warns = [m for m in warnings_emitted if "Ollama call" in m]
        assert len(ollama_warns) == 1


# list_models


class TestListModels:
    def test_returns_model_names(self) -> None:
        client = OllamaClient(_enabled_config())
        resp = _make_response({"models": [{"name": "llama3"}, {"name": "mistral"}]})
        with mock.patch("urllib.request.urlopen", return_value=resp):
            models = client.list_models()
        assert "llama3" in models
        assert "mistral" in models

    def test_returns_empty_list_when_disabled(self) -> None:
        client = OllamaClient(_disabled_config())
        with mock.patch("urllib.request.urlopen") as m:
            models = client.list_models()
        assert models == []
        m.assert_not_called()

    def test_returns_empty_list_on_error(self) -> None:
        client = OllamaClient(_enabled_config())
        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            models = client.list_models()
        assert models == []
