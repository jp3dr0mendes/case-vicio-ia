"""Ablações e escolha de hiperparâmetros na folha-verdade (validação cruzada repetida).

Uso: PYTHONPATH=src python scripts/experimentos.py > artifacts/experimentos.json
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
from vicio_classifier.pipeline import Config, choose_threshold, cross_validate, load_context

ctx = load_context(Path("data/anexos"), Path(os.environ.get("ARTIFACTS_DIR", "artifacts")))
VARIANTS = {
    "completo": None,
    "so_tfidf": ["tfidf", "tfidf_gap"],
    "so_embedding": ["emb", "emb_gap"],
    "so_origem": ["origem", "origem_gap", "disciplina"],
    "sem_origem": [f for f in FEATURES if f not in ("origem", "origem_gap", "disciplina", "origem_rotulada")],
    "sem_knn_prior": [f for f in FEATURES if f not in ("knn", "knn_gap", "prior", "origem_rotulada")],
    "sem_origem_rotulada": [f for f in FEATURES if f != "origem_rotulada"],
    "sem_clone_volume": [f for f in FEATURES if f not in ("clone", "volume", "volume_desconhecido")],
    "sem_leis_artigos": [f for f in FEATURES if f not in ("lei", "artigo")],
    "sem_embedding": [f for f in FEATURES if f not in ("emb", "emb_gap", "knn", "knn_gap")],
}
L2 = [float(x) for x in sys.argv[1:]] or [1.0]
res = {}
for l2 in L2:
    for name, mask in VARIANTS.items():
        cfg = Config(l2=l2, feature_mask=mask, cv_repeats=5)
        oof = cross_validate(ctx, cfg)
        z = np.array([r["z"] for r in oof]); y = np.array([r["acerto"] for r in oof])
        cal = Calibrator.fit(z, y)
        conf = cal(z)
        thr = choose_threshold(conf, y, cfg)
        pub = conf >= thr
        ll = float(-np.mean(y * np.log(conf) + (1 - y) * np.log(1 - conf)))
        res[f"{name}|l2={l2}"] = {
            "top1": round(float(y.mean()), 3), "logloss_cal": round(ll, 3),
            "limiar": thr, "cobertura": round(float(pub.mean()), 3),
            "precisao": round(float(y[pub].mean()), 3) if pub.any() else None,
            "cobertura_a_090": round(float((conf >= 0.90).mean()), 3),
            "precisao_a_090": round(float(y[conf >= 0.90].mean()), 3) if (conf >= 0.90).any() else None,
        }
        print(name, l2, res[f"{name}|l2={l2}"], file=sys.stderr, flush=True)
print(json.dumps(res, ensure_ascii=False, indent=2))
