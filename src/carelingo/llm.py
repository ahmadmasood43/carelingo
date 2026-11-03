"""Chat-completion clients with token, cost, and latency accounting."""

import os
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

# USD per million tokens (input, output), from the model cards at
# https://docs.mistral.ai/models, read 2026-11-03. Prices change; re-check before
# quoting costs.
PRICES_USD_PER_M = {
    "mistral-large-4": (0.70, 2.10),
    "mistral-small-2603": (0.15, 0.60),
    "ministral-14b-2512": (0.20, 0.20),
    "ministral-8b-2512": (0.15, 0.15),
    # Local models served by Ollama: no per-token price. Hardware and power are not
    # counted here and must be costed separately.
    "qwen3:8b": (0.0, 0.0),
}


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    # Hidden reasoning tokens; already included in output_tokens and billed as output.
    reasoning_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        input_price, output_price = PRICES_USD_PER_M[self.model]
        return (self.input_tokens * input_price + self.output_tokens * output_price) / 1e6


class ChatModel(Protocol):
    def complete(
        self,
        model: str,
        system: str,
        user: str,
        json_schema: dict | None = None,
        max_tokens: int | None = None,
    ) -> Completion: ...


class ApiError(RuntimeError):
    pass


class MistralClient:
    base_url = "https://api.mistral.ai/v1"
    retryable_status = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str | None = None,
        timeout_s: float = 180,
        max_retries: int = 6,
        reasoning_effort: str | None = None,
    ):
        api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set")
        self._http = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_s,
        )
        self._max_retries = max_retries
        # For reasoning models (mistral-large-4): "low", "medium" or "high".
        self._reasoning_effort = reasoning_effort

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        json_schema: dict | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Greedy decoding; reasoning models reject temperature 0 without top_p 1.
            "temperature": 0.0,
            "top_p": 1.0,
            "random_seed": 0,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self._reasoning_effort is not None:
            payload["reasoning_effort"] = self._reasoning_effort
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": json_schema},
            }
        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._http.post("/chat/completions", json=payload)
            except httpx.TransportError:
                if attempt == self._max_retries:
                    raise
                time.sleep(2**attempt)
                continue
            latency = time.perf_counter() - started
            if response.status_code not in self.retryable_status or attempt == self._max_retries:
                break
            # Token-per-minute limits on larger models reset within a minute.
            time.sleep(min(60, 4 * 2**attempt))
        if response.is_error:
            raise ApiError(f"{model}: HTTP {response.status_code}: {response.text[:500]}")
        body = response.json()
        usage = body["usage"]
        # Latency covers only the successful attempt, so retries after rate limiting
        # do not inflate the per-request figure.
        return Completion(
            text=_answer_text(body["choices"][0]["message"]["content"]),
            model=model,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
            latency_s=latency,
            reasoning_tokens=(usage.get("completion_tokens_details") or {}).get(
                "reasoning_tokens", 0
            ),
        )


def _answer_text(content: str | list[dict]) -> str:
    # Reasoning models return [{"type": "thinking", ...}, {"type": "text", ...}].
    if isinstance(content, str):
        return content.strip()
    return "".join(chunk["text"] for chunk in content if chunk["type"] == "text").strip()


class OllamaClient:
    """Local models served by Ollama, with the same interface as MistralClient."""

    def __init__(
        self, base_url: str = "http://127.0.0.1:11434", timeout_s: float = 900, think: bool = False
    ):
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s)
        self._think = think

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        json_schema: dict | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # Qwen3 reasons by default; off unless asked for, since it multiplies latency.
            "think": self._think,
            "options": {"temperature": 0.0, "seed": 0, "num_ctx": 4096},
        }
        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens
        if json_schema is not None:
            payload["format"] = json_schema
        started = time.perf_counter()
        response = self._http.post("/api/chat", json=payload)
        latency = time.perf_counter() - started
        if response.is_error:
            raise ApiError(f"{model}: HTTP {response.status_code}: {response.text[:500]}")
        body = response.json()
        return Completion(
            text=body["message"]["content"].strip(),
            model=model,
            input_tokens=body.get("prompt_eval_count", 0),
            output_tokens=body.get("eval_count", 0),
            latency_s=latency,
        )
