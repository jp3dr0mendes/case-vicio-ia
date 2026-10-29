"""Enriquece a descrição de cada tema com um modelo de linguagem local (offline).

O modelo só vê o caminho do tema na árvore e os vizinhos; nunca vê questões.
A saída é gravada como dado versionado em artifacts/temas_enriquecidos.jsonl e é
retomável (temas já presentes não são refeitos).

Uso: python -m vicio_classifier.offline.enrich_temas --model qwen2.5:14b
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from vicio_classifier.data import load_temas
from vicio_classifier.offline.ollama_client import generate_json
from vicio_classifier.taxonomy import Taxonomy

PROMPT_VERSION = "enriquecer_tema_v1"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/anexos")
    ap.add_argument("--out", default="artifacts/temas_enriquecidos.jsonl")
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--prompts-dir", default="prompts")
    args = ap.parse_args()

    tax = Taxonomy(load_temas(Path(args.data_dir) / "temas.csv"))
    template = (Path(args.prompts_dir) / f"{PROMPT_VERSION}.txt").read_text(encoding="utf-8")
    out = Path(args.out)
    done: set[int] = set()
    if out.exists():
        done = {json.loads(line)["tema_id"] for line in out.open(encoding="utf-8") if line.strip()}

    todo = [t for t in tax.selectable_ids if t not in done]
    print(f"{len(todo)} temas a enriquecer ({len(done)} já prontos)", flush=True)
    with out.open("a", encoding="utf-8") as fh:
        for i, tid in enumerate(sorted(todo), 1):
            node = tax.nodes[tid]
            irmaos = [tax.nodes[s].titulo for s in tax.children.get(node.parent_id, []) if s != tid]
            filhos = [tax.nodes[c].titulo for c in tax.children.get(tid, [])]
            prompt = template.format(
                caminho=node.caminho,
                irmaos="; ".join(irmaos[:25]) or "(nenhum)",
                filhos="; ".join(filhos) or "(nenhum)",
            )
            t0 = time.time()
            try:
                resp = generate_json(args.model, prompt)
                descricao = str(resp.get("descricao", "")).strip()
                termos = [str(x).strip() for x in resp.get("termos", []) if str(x).strip()]
                erro = None
            except Exception as exc:  # noqa: BLE001
                descricao, termos, erro = "", [], str(exc)
            rec = {"tema_id": tid, "caminho": node.caminho, "descricao_llm": descricao,
                   "termos_llm": termos, "modelo": args.model, "prompt": PROMPT_VERSION,
                   "erro": erro, "segundos": round(time.time() - t0, 1)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{i}/{len(todo)}] {tid} {node.caminho} ({rec['segundos']}s)", flush=True)


if __name__ == "__main__":
    main()
