"""Teste de estresse para a mudança de distribuição da importação nova (Q0360-Q0659).

A folha-verdade só cobre o banco antigo. Na importação nova o assunto de origem tem outro formato.
Aqui a validação cruzada é refeita zerando, nas questões de teste, os sinais que dependem da
origem (assunto e disciplina), e mede quanto a precisão no limiar escolhido se degrada.

Uso: PYTHONPATH=src python scripts/estresse_origem.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from vicio_classifier.features import FEATURES
from vicio_classifier.model import Calibrator, confidence_inputs
from vicio_classifier.pipeline import SEED, Config, _fit_predict, choose_threshold, cross_validate, load_context

ctx = load_context(Path("data/anexos"), Path(os.environ.get("ARTIFACTS_DIR", "artifacts")))
cfg = Config()
oof = cross_validate(ctx, cfg)
z = np.array([r["z"] for r in oof]); y = np.array([r["acerto"] for r in oof])
cal = Calibrator.fit(z, y)
thr = choose_threshold(cal(z), y, cfg)

ORIGEM = [FEATURES.index(f) for f in ("origem", "origem_gap", "disciplina")]
rng = np.random.default_rng(SEED)
idx = ctx.index
pub = hit = 0
for rep in range(cfg.cv_repeats):
    order = rng.permutation(len(ctx.folha))
    for test in np.array_split(order, cfg.cv_folds):
        ts = set(test.tolist())
        train = [g for i, g in enumerate(ctx.folha) if i not in ts]
        test_rows = [idx[ctx.folha[i].id] for i in test]
        saved = ctx.sig.X[test_rows][:, :, ORIGEM].copy()
        ctx.sig.X[np.ix_(test_rows, range(ctx.sig.X.shape[1]), ORIGEM)] = 0.0
        P, _, X = _fit_predict(ctx, train, cfg)
        ctx.sig.X[np.ix_(test_rows, range(ctx.sig.X.shape[1]), ORIGEM)] = saved
        top, Z = confidence_inputs(P, X, FEATURES)
        conf = cal(Z)
        for i in test:
            g = ctx.folha[i]; qi = idx[g.id]
            if conf[qi] >= thr:
                pub += 1
                hit += ctx.tax.is_correct(ctx.sig.candidates[top[qi]], g.tema_central_id, g.extras_ids)
base = cal(z) >= thr
print(json.dumps({
    "limiar": thr,
    "normal": {"cobertura": float(base.mean()), "precisao": float(y[base].mean()) if base.any() else None},
    "sem_origem_no_teste": {"cobertura": pub / len(oof), "precisao": hit / pub if pub else None},
}, indent=2))
