"""Auditoria dos anexos: esquema, árvore, campos ausentes e ruído da origem."""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from vicio_classifier.data import load_folha, load_questions, load_temas, normalize, sha256
from vicio_classifier.taxonomy import Taxonomy

_PARTE = re.compile(r"\s*\(parte \d+\)|\s*-\s*crimes$|\s*\(parte 0?\d+\)\s*-\s*especies")


def audit(data_dir: Path) -> dict:
    qs = load_questions(data_dir / "questoes.jsonl")
    temas = load_temas(data_dir / "temas.csv")
    tax = Taxonomy(temas)
    folha = load_folha(data_dir / "folha_verdade.csv")
    qids = {q.id for q in qs}

    titulo_norm: dict[str, list[int]] = {}
    for t in temas:
        if t.nivel >= 1:
            titulo_norm.setdefault(_PARTE.sub("", normalize(t.titulo)).strip(), []).append(t.tema_id)
    duplicados = {k: v for k, v in titulo_norm.items() if len(v) > 1}

    gold_ids = [g.tema_central_id for g in folha if g.tema_central_id]
    folha_problemas = [g.id for g in folha if g.id not in qids]
    folha_problemas += [g.id for g in folha for t in ([g.tema_central_id] if g.tema_central_id else []) + g.extras_ids
                        if t not in tax.selectable_ids]
    disc = Counter(q.disciplina_origem or "(vazio)" for q in qs)
    html_residual = sum(bool(re.search(r"<[a-z/][^>]*>", q.text)) for q in qs)
    return {
        "arquivos_sha256": {p.name: sha256(p) for p in sorted(data_dir.iterdir()) if p.is_file()},
        "questoes": {
            "total": len(qs), "ids_unicos": len(qids),
            "faixa_ids": [min(qids), max(qids)],
            "banco_Q0001_Q0359": sum(q.id <= "Q0359" for q in qs),
            "importacao_nova_Q0360_Q0659": sum(q.id >= "Q0360" for q in qs),
            "sem_comando": sum(not q.comando for q in qs),
            "sem_assunto_origem": sum(not q.assunto_origem for q in qs),
            "sem_disciplina_origem": sum(not q.disciplina_origem for q in qs),
            "sem_alternativas": sum(not q.alternativas for q in qs),
            "html_residual_apos_limpeza": html_residual,
            "assuntos_origem_distintos": len({q.assunto_origem for q in qs}),
            "disciplina_origem": dict(disc.most_common()),
            "disciplina_origem_fora_das_4_areas": sum(
                v for k, v in disc.items()
                if normalize(k) not in {"penal", "direito penal", "processo penal", "criminologia",
                                        "direito penal - legislacao especial", "direito processual penal"}),
        },
        "temas": {
            "total": len(temas),
            "por_nivel": dict(Counter(t.nivel for t in temas)),
            "por_status": dict(Counter(t.status for t in temas)),
            "inativos": [{"tema_id": t.tema_id, "caminho": t.caminho} for t in temas if t.status != "active"],
            "selecionaveis": len(tax.selectable_ids),
            "selecionaveis_nivel_1": len(tax.level1_selectable()),
            "sem_descricao_clones": [t.tema_id for t in temas if t.nivel >= 1 and not t.descricao],
            "titulos_duplicados_ou_partes": duplicados,
            "arvore_sem_ciclos_e_pais_validos": True,
        },
        "folha_verdade": {
            "total": len(folha), "fora_da_taxonomia": sum(g.fora for g in folha),
            "com_extras": sum(bool(g.extras_ids) for g in folha),
            "central_nivel": dict(Counter(tax.nodes[t].nivel for t in gold_ids)),
            "central_por_disciplina": dict(Counter(tax.nodes[tax.root_of(t)].titulo for t in gold_ids)),
            "temas_distintos_central": len(set(gold_ids)),
            "temas_nivel1_distintos": len({tax.level1_of(t) for t in gold_ids}),
            "ids_ou_temas_invalidos": folha_problemas,
            "usa_clones": sorted({t for g in folha for t in [g.tema_central_id, *g.extras_ids]
                                  if t and not tax.nodes[t].descricao}),
        },
    }
