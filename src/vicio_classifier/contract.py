"""Contrato do publicacao.csv (LEIA-ME.md, seção "Formatos de entrega")."""
from __future__ import annotations

import csv
from pathlib import Path

from vicio_classifier.taxonomy import Taxonomy

COLUMNS = ["id", "decisao", "tema_central_id", "extras_ids", "confianca", "motivo"]
DECISIONS = {"publicar", "revisar", "fora"}


class ContractError(ValueError):
    pass


def validate_publication(rows: list[dict], question_ids: set[str], tax: Taxonomy) -> None:
    ids = [str(r.get("id", "")) for r in rows]
    if len(ids) != len(set(ids)):
        raise ContractError("IDs duplicados na publicação")
    if set(ids) != question_ids:
        raise ContractError(f"IDs divergentes: faltam {sorted(question_ids - set(ids))[:5]}, "
                            f"sobram {sorted(set(ids) - question_ids)[:5]}")
    for r in rows:
        qid, dec = r.get("id"), r.get("decisao")
        if set(r) != set(COLUMNS):
            raise ContractError(f"{qid}: colunas {sorted(r)} diferentes do contrato")
        if dec not in DECISIONS:
            raise ContractError(f"{qid}: decisão inválida {dec!r}")
        central = str(r.get("tema_central_id", "")).strip()
        if dec == "fora":
            if central:
                raise ContractError(f"{qid}: 'fora' deve ter tema_central_id vazio")
        else:
            if not central.isdigit() or int(central) not in tax.selectable_ids:
                raise ContractError(f"{qid}: tema_central_id {central!r} inexistente, inativo ou raiz")
        extras = [x for x in str(r.get("extras_ids", "")).split(";") if x.strip()]
        if len(extras) != len(set(extras)):
            raise ContractError(f"{qid}: extras repetidos")
        for x in extras:
            if not x.strip().isdigit() or int(x) not in tax.selectable_ids:
                raise ContractError(f"{qid}: extra {x!r} inexistente, inativo ou raiz")
            if x.strip() == central:
                raise ContractError(f"{qid}: extra igual ao tema central")
        try:
            c = float(r.get("confianca"))
        except (TypeError, ValueError) as exc:
            raise ContractError(f"{qid}: confiança não numérica") from exc
        if not 0.0 <= c <= 1.0:
            raise ContractError(f"{qid}: confiança fora de [0, 1]")
        if dec in {"revisar", "fora"} and not str(r.get("motivo", "")).strip():
            raise ContractError(f"{qid}: motivo obrigatório em {dec}")


def write_publication(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "confianca": f"{float(r['confianca']):.4f}"})


def read_publication(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != COLUMNS:
            raise ContractError(f"cabeçalho {reader.fieldnames} diferente de {COLUMNS}")
        return list(reader)
