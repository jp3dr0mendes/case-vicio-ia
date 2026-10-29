"""Linha de comando.

  audit     audita os anexos            -> artifacts/data_audit.json
  classify  regenera a publicação        -> publicacao.csv + artifacts/ (sem modelo de linguagem)
  validate  confere um publicacao.csv contra o contrato do LEIA-ME
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from vicio_classifier.audit import audit
from vicio_classifier.contract import ContractError, read_publication, validate_publication
from vicio_classifier.data import load_questions, load_temas
from vicio_classifier.pipeline import Config, run
from vicio_classifier.taxonomy import Taxonomy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vicio_classifier")
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data/anexos"))
    ap.add_argument("--artifacts-dir", default=os.environ.get("ARTIFACTS_DIR", "artifacts"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    c = sub.add_parser("classify")
    c.add_argument("--out", default=os.environ.get("OUTPUT_CSV", "publicacao.csv"))
    c.add_argument("--threshold", type=float, default=None,
                   help="limiar fixo de confiança; padrão: escolhido pela validação cruzada")
    c.add_argument("--cv-repeats", type=int, default=10)
    c.add_argument("--db", action="store_true", help="grava a execução no PostgreSQL (DATABASE_URL)")
    v = sub.add_parser("validate")
    v.add_argument("csv", nargs="?", default="publicacao.csv")
    args = ap.parse_args(argv)
    data, art = Path(args.data_dir), Path(args.artifacts_dir)

    if args.cmd == "audit":
        rep = audit(data)
        art.mkdir(parents=True, exist_ok=True)
        (art / "data_audit.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({k: rep[k] for k in ("questoes", "folha_verdade")}, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "classify":
        out = Path(args.out)
        meta = run(data, art, out, Config(threshold=args.threshold, cv_repeats=args.cv_repeats))
        shutil.copyfile(out, art / "publicacao.csv")
        shutil.copyfile(art / "previsao.txt", out.parent / "previsao.txt")
        if args.db:
            from vicio_classifier.db import save_run
            run_id = save_run(os.environ["DATABASE_URL"], meta, art / "decision_audit.jsonl")
            print(f"execução gravada no PostgreSQL: run_id={run_id}")
        print(json.dumps({k: meta[k] for k in (
            "limiar_publicacao", "decisoes_todas", "decisoes_fora_da_folha", "cobertura_fora_da_folha",
            "precisao_esperada_pelo_calibrador", "segundos")}, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "validate":
        tax = Taxonomy(load_temas(data / "temas.csv"))
        ids = {q.id for q in load_questions(data / "questoes.jsonl")}
        try:
            validate_publication(read_publication(Path(args.csv)), ids, tax)
        except ContractError as exc:
            print(f"INVÁLIDO: {exc}", file=sys.stderr)
            return 1
        print(f"OK: {args.csv} cumpre o contrato ({len(ids)} questões)")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
