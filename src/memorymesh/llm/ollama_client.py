"""Ollama HTTP client.

Wraps the Ollama local LLM server's ``/api/generate`` and ``/api/tags``
endpoints using only the stdlib's ``urllib.request``.

Privacy invariant: the first time :meth:`OllamaClient.generate` is called in a
session it logs a WARNING informing the user that text is being sent to the
local Ollama process (not the internet, but still off-process).

Design rules:
* **Never called** when ``ollama.enabled: false``.
* Every failure returns a safe default (``""``, ``False``, ``[]``) and logs at
  WARNING level - the daemon must not crash due to LLM unavailability.
* Timeout is configured per :class:`~memorymesh.core.models.OllamaConfig`.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from loguru import logger

from memorymesh.core.models import OllamaConfig


class OllamaClient:
    """Thin client for the Ollama local LLM API.

    Args:
        config: Ollama connection and model settings.
    """

    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._warned_generate: bool = False

    def generate(self, prompt: str, model: str | None = None) -> str:
        """Generate a text completion using a local Ollama model.

        The first call in each session logs a WARNING reminding the user that
        text is being sent to the local Ollama process.

        Args:
            prompt: The input prompt text.
            model: Model name override.  Falls back to ``config.model``.

        Returns:
            Generated text string, or ``""`` on any error.
        """
        if not self._config.enabled:
            return ""

        if not self._warned_generate:
            logger.warning(f"Ollama call: sending data to local LLM at {self._config.base_url}")
            self._warned_generate = True

        effective_model = model or self._config.model
        payload = json.dumps(
            {
                "model": effective_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            }
        ).encode()

        url = f"{self._config.base_url.rstrip('/')}/api/generate"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._config.timeout_s) as resp:
                body = json.loads(resp.read())
                text = body.get("response", "").strip()
                if not text:
                    logger.warning("OllamaClient.generate: empty response from Ollama")
                return text
        except urllib.error.URLError as exc:
            logger.warning(f"OllamaClient.generate: Ollama unreachable: {exc}")
            return ""
        except TimeoutError:
            logger.warning(
                f"OllamaClient.generate: request timed out after {self._config.timeout_s}s"
            )
            return ""
        except Exception as exc:
            logger.warning(f"OllamaClient.generate: unexpected error: {exc}")
            return ""

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable.

        Returns:
            ``True`` when the ``/api/tags`` endpoint responds within timeout,
            ``False`` otherwise.
        """
        if not self._config.enabled:
            return False

        url = f"{self._config.base_url.rstrip('/')}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=self._config.timeout_s) as resp:
                return resp.status == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return the names of models available in the local Ollama installation.

        Returns:
            List of model name strings, or ``[]`` if Ollama is unreachable.
        """
        if not self._config.enabled:
            return []

        url = f"{self._config.base_url.rstrip('/')}/api/tags"
        try:
            with urllib.request.urlopen(url, timeout=self._config.timeout_s) as resp:
                body = json.loads(resp.read())
                models = body.get("models", [])
                return [m.get("name", "") for m in models if isinstance(m, dict)]
        except Exception as exc:
            logger.warning(f"OllamaClient.list_models: {exc}")
            return []
