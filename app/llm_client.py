"""
Thin LLM client. Supports three providers, selected by LLM_PROVIDER env var:

  - "groq"   : Groq's free-tier, OpenAI-compatible chat completions API.
               Needs GROQ_API_KEY. Fast, generous free quota, good for a
               48h assessment — no local GPU/download needed.
  - "ollama" : A locally running Ollama server (fully offline, zero cost,
               zero signup). Needs `ollama pull <model>` done beforehand.
  - "none"   : No LLM configured. query_engine.py falls back to a rule-based
               parser so the system still runs end-to-end for grading even
               without any API key set up.

Kept provider-agnostic behind one `complete()` method so swapping providers
never touches business logic in query_engine.py / anomaly.py.
"""

from __future__ import annotations

import json
import logging
import os

import requests

logger = logging.getLogger(__name__)


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "none").lower()
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1")
        self.timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))

    @property
    def is_configured(self) -> bool:
        if self.provider == "groq":
            return bool(self.groq_api_key)
        if self.provider == "ollama":
            return True  # assumed reachable; checked at call time
        return False

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Returns the raw text completion. Raises LLMError on failure."""
        if self.provider == "groq":
            return self._complete_groq(system_prompt, user_prompt)
        if self.provider == "ollama":
            return self._complete_ollama(system_prompt, user_prompt)
        raise LLMError("No LLM provider configured (LLM_PROVIDER=none)")

    def _complete_groq(self, system_prompt: str, user_prompt: str) -> str:
        if not self.groq_api_key:
            raise LLMError("GROQ_API_KEY is not set")
        try:
            payload = {
                "model": self.groq_model,
                "temperature": 0,
                "max_completion_tokens": 1024,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            # gpt-oss-20b/120b are reasoning models: at the default
            # "medium" reasoning_effort they can burn the entire token
            # budget on hidden reasoning and return empty content with
            # finish_reason="length" (intermittently — depends on how
            # long the model "thinks"). Forcing "low" keeps reasoning
            # short so there's always room left for the actual JSON
            # answer. Harmless to send for non-gpt-oss models; Groq
            # ignores it if the model doesn't support it.
            if "gpt-oss" in self.groq_model:
                payload["reasoning_effort"] = "low"

            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.groq_api_key}"},
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]

            if not content or not content.strip():
                finish_reason = choice.get("finish_reason", "unknown")
                raise LLMError(
                    f"Groq returned empty content (finish_reason={finish_reason}). "
                    "This usually means the model spent its whole token budget on "
                    "hidden reasoning — try raising max_completion_tokens or lowering "
                    "reasoning_effort further."
                )
            return content
        except requests.RequestException as e:
            logger.exception("Groq call failed")
            raise LLMError(f"Groq request failed: {e}") from e
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.exception("Unexpected Groq response shape")
            raise LLMError(f"Unexpected Groq response: {e}") from e

    def _complete_ollama(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.ollama_model,
                    "stream": False,
                    "options": {"temperature": 0},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["message"]["content"]
            if not content or not content.strip():
                raise LLMError("Ollama returned empty content")
            return content
        except requests.RequestException as e:
            logger.exception("Ollama call failed")
            raise LLMError(f"Ollama request failed: {e}") from e
        except (KeyError, json.JSONDecodeError) as e:
            logger.exception("Unexpected Ollama response shape")
            raise LLMError(f"Unexpected Ollama response: {e}") from e