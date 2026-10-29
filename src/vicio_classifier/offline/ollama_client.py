"""Cliente mínimo do Ollama, usado apenas nas etapas offline."""
from __future__ import annotations

import json
import os
import time
import urllib.request

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _post(path: str, payload: dict, timeout: float = 300.0, retries: int = 3) -> dict:
    data = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                f"{BASE_URL}{path}", data=data, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - rede/servidor: tenta de novo
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Falha ao chamar Ollama {path}: {last_error}")


def generate_json(model: str, prompt: str) -> dict:
    out = _post(
        "/api/generate",
        {"model": model, "prompt": prompt, "stream": False, "format": "json",
         "options": {"temperature": 0, "seed": 7}},
    )
    return json.loads(out["response"])


def embed(model: str, texts: list[str]) -> list[list[float]]:
    out = _post("/api/embed", {"model": model, "input": texts})
    return out["embeddings"]


def model_digest(model: str) -> str:
    out = _post("/api/show", {"model": model})
    return out.get("details", {}).get("parameter_size", "") + " " + str(out.get("modified_at", ""))
