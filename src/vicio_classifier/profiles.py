"""Textos usados para representar questões e temas (TF-IDF e embeddings)."""
from __future__ import annotations

import json
from pathlib import Path

from vicio_classifier.data import Question
from vicio_classifier.taxonomy import Taxonomy


def question_embedding_text(q: Question) -> str:
    # A origem não entra no texto: é um sinal separado (e ruidoso), tratado como feature.
    return q.text


def load_enrichment(artifacts_dir: Path) -> dict[int, dict]:
    path = artifacts_dir / "temas_enriquecidos.jsonl"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if not rec.get("erro"):
                out[int(rec["tema_id"])] = rec
    return out


def tema_profiles(tax: Taxonomy, artifacts_dir: Path) -> dict[int, str]:
    """Perfil textual de cada tema selecionável: caminho + subtemas + descrição/termos enriquecidos."""
    enr = load_enrichment(artifacts_dir)
    out = {}
    for tid in sorted(tax.selectable_ids):
        node = tax.nodes[tid]
        parts = [node.caminho.replace(" > ", ". ")]
        subs = [tax.nodes[c].titulo for c in tax.children.get(tid, []) if c in tax.selectable_ids]
        if subs:
            parts.append("Subtemas: " + "; ".join(subs) + ".")
        e = enr.get(tid)
        if e:
            parts.append(e.get("descricao_llm", ""))
            if e.get("termos_llm"):
                parts.append("Termos: " + "; ".join(e["termos_llm"]) + ".")
        out[tid] = " ".join(p for p in parts if p)
    return out
