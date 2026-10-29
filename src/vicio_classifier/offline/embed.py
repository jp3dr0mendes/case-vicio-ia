"""Gera embeddings (offline) das questões e dos temas com um modelo local via Ollama.

Os vetores ficam em cache versionado (artifacts/embeddings/*.npz), com o hash SHA-256 do
texto de cada item. O comando `classify` só lê esse cache; se o texto de uma questão mudar
(ou ela não estiver no cache), a questão vai para revisão em vez de chamar o modelo.

Uso: python -m vicio_classifier.offline.embed --what questoes|temas
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from vicio_classifier.data import load_questions, load_temas
from vicio_classifier.offline.ollama_client import embed
from vicio_classifier.profiles import question_embedding_text, tema_profiles
from vicio_classifier.taxonomy import Taxonomy

MAX_CHARS = 3000


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(ids: list[str], texts: list[str], model: str, out: Path, batch: int = 16) -> None:
    vecs: list[list[float]] = []
    for i in range(0, len(texts), batch):
        vecs.extend(embed(model, [t[:MAX_CHARS] for t in texts[i:i + batch]]))
        print(f"{min(i + batch, len(texts))}/{len(texts)}", flush=True)
    arr = np.asarray(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, ids=np.array(ids), vectors=arr.astype(np.float16),
                        hashes=np.array([text_hash(t) for t in texts]), model=np.array(model))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["questoes", "temas"], required=True)
    ap.add_argument("--data-dir", default="data/anexos")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--model", default="bge-m3")
    args = ap.parse_args()
    data, art = Path(args.data_dir), Path(args.artifacts_dir)
    if args.what == "questoes":
        qs = load_questions(data / "questoes.jsonl")
        run([q.id for q in qs], [question_embedding_text(q) for q in qs], args.model,
            art / "embeddings" / "questoes.npz")
    else:
        tax = Taxonomy(load_temas(data / "temas.csv"))
        prof = tema_profiles(tax, art)
        ids = sorted(prof)
        run([str(i) for i in ids], [prof[i] for i in ids], args.model, art / "embeddings" / "temas.npz")


if __name__ == "__main__":
    main()
