"""Ablações e escolha de hiperparâmetros na folha-verdade.

Para cada variante: validação cruzada repetida (10 x 10), calibrador e limiar como na produção
(alvo de 97%), e a avaliação aninhada da regra inteira, que é a métrica honesta.

Uso: PYTHONPATH=src python scripts/experimentos.py 1 3 > artifacts/experimentos.json
(os argumentos são os valores de regularização L2 testados; a produção usa 1).
Não chama modelo de linguagem.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from vicio_classifier.features import FEATURES
from vicio_classifier.model import Calibrator
from vicio_classifier.pipeline import Config, choose_threshold, cross_validate, load_context, nested_evaluation

DATA = Path("data/anexos")
ART = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
ctx_full = load_context(DATA, ART)
ctx_plain = load_context(DATA, ART, use_enrichment=False)

VARIANTS = {
    "completo": (ctx_full, None),
    "sem_enriquecimento_dos_temas": (ctx_plain, None),
    "so_tfidf": (ctx_full, ["tfidf", "tfidf_gap"]),
    "so_embedding": (ctx_full, ["emb", "emb_gap"]),
    "so_origem": (ctx_full, ["origem", "origem_gap", "disciplina"]),
    "sem_origem": (ctx_full, [f for f in FEATURES if f not in ("origem", "origem_gap", "disciplina", "origem_rotulada")]),
    "sem_knn_prior": (ctx_full, [f for f in FEATURES if f not in ("knn", "knn_gap", "prior", "origem_rotulada")]),
    "sem_origem_rotulada": (ctx_full, [f for f in FEATURES if f != "origem_rotulada"]),
    "sem_clone_volume": (ctx_full, [f for f in FEATURES if f not in ("clone", "volume", "volume_desconhecido")]),
    "sem_leis_artigos": (ctx_full, [f for f in FEATURES if f not in ("lei", "artigo")]),
    "sem_embedding": (ctx_full, [f for f in FEATURES if f not in ("emb", "emb_gap", "knn", "knn_gap")]),
}
L2 = [float(x) for x in sys.argv[1:]] or [1.0]
res = {}
for l2 in L2:
    for name, (ctx, mask) in VARIANTS.items():
        cfg = Config(l2=l2, feature_mask=mask)
        oof = cross_validate(ctx, cfg)
        z = np.array([r["z"] for r in oof]); y = np.array([r["acerto"] for r in oof])
        cal = Calibrator.fit(z, y)
        conf = np.clip(cal(z), 1e-6, 1 - 1e-6)
        thr = choose_threshold(conf, y, cfg)
        pub = conf >= thr
        nested = nested_evaluation(oof, cfg)
        res[f"{name}|l2={l2}"] = {
            "top1": round(float(y.mean()), 3),
            "logloss_calibrado": round(float(-np.mean(y * np.log(conf) + (1 - y) * np.log(1 - conf))), 3),
            "limiar": thr,
            "cobertura_oof": round(float(pub.mean()), 3),
            "precisao_oof": round(float(y[pub].mean()), 3) if pub.any() else None,
            "cobertura_aninhada": round(nested["cobertura"], 3),
            "precisao_aninhada": round(nested["precisao"], 3) if nested["precisao"] is not None else None,
        }
        print(name, l2, res[f"{name}|l2={l2}"], file=sys.stderr, flush=True)
print(json.dumps(res, ensure_ascii=False, indent=2))
