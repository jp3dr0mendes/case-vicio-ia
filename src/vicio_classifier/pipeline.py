"""Orquestração: carrega anexos e artefatos, avalia na folha-verdade e gera a publicação.

Nenhuma função deste módulo chama modelo de linguagem. Embeddings são lidos do cache
versionado em artifacts/embeddings.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vicio_classifier import __version__
from vicio_classifier.contract import COLUMNS, validate_publication, write_publication
from vicio_classifier.data import Gabarito, Question, load_folha, load_questions, load_temas, sha256
from vicio_classifier.forecast import forecast, render
from vicio_classifier.features import FEATURES, StaticSignals, build_static, label_features
from vicio_classifier.legal import LegalRules
from vicio_classifier.model import CONF_FEATURES, Calibrator, Ranker, confidence_inputs
from vicio_classifier.profiles import tema_profiles
from vicio_classifier.taxonomy import Taxonomy

POLICY_VERSION = "politica_v2.0"
SEED = 20261102


@dataclass
class Config:
    l2: float = 1.0
    threshold: float | None = None   # None = escolher pela validação cruzada
    # Alvo de 97% na validação cruzada para garantir 95% na avaliação: margem para a amostra
    # pequena (~25 publicações por rodada) e para a importação nova, que a folha não cobre.
    target_precision: float = 0.97
    min_threshold: float = 0.90
    cv_folds: int = 10
    cv_repeats: int = 10
    feature_mask: list[str] | None = None  # ablação: subconjunto de FEATURES


@dataclass
class Context:
    data_dir: Path
    artifacts_dir: Path
    questions: list[Question]
    tax: Taxonomy
    folha: list[Gabarito]
    sig: StaticSignals
    embedding_model: str
    hashes: dict[str, str] = field(default_factory=dict)

    @property
    def index(self) -> dict[str, int]:
        return {qid: i for i, qid in enumerate(self.sig.ids)}


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, str], str]:
    if not path.exists():
        return {}, {}, ""
    z = np.load(path, allow_pickle=False)
    ids = [str(x) for x in z["ids"]]
    vecs = z["vectors"].astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return dict(zip(ids, vecs)), dict(zip(ids, [str(h) for h in z["hashes"]])), str(z["model"])


def load_context(data_dir: Path, artifacts_dir: Path) -> Context:
    questions = load_questions(data_dir / "questoes.jsonl")
    tax = Taxonomy(load_temas(data_dir / "temas.csv"))
    folha = load_folha(data_dir / "folha_verdade.csv")
    rules = LegalRules(artifacts_dir / "regras")
    profiles = tema_profiles(tax, artifacts_dir)

    q_emb, q_hash, model = _load_npz(artifacts_dir / "embeddings" / "questoes.npz")
    # Só usa o vetor em cache se o texto da questão não mudou desde que foi gerado.
    for q in questions:
        h = hashlib.sha256(q.text.encode("utf-8")).hexdigest()
        if q.id in q_hash and q_hash[q.id] != h:
            q_emb.pop(q.id, None)
    t_raw, t_hash, _ = _load_npz(artifacts_dir / "embeddings" / "temas.npz")
    t_emb = {}
    for k, v in t_raw.items():
        tid = int(k)
        if tid in profiles and hashlib.sha256(profiles[tid].encode("utf-8")).hexdigest() == t_hash[k]:
            t_emb[tid] = v
    sig = build_static(questions, tax, profiles, rules, q_emb, t_emb)
    hashes = {p.name: sha256(p) for p in sorted(data_dir.iterdir()) if p.is_file()}
    for p in sorted((artifacts_dir / "regras").glob("*.json")):
        hashes[f"regras/{p.name}"] = sha256(p)
    for p in [artifacts_dir / "temas_enriquecidos.jsonl", *sorted((artifacts_dir / "embeddings").glob("*.npz"))]:
        if p.exists():
            hashes[str(p.relative_to(artifacts_dir))] = sha256(p)
    return Context(data_dir, artifacts_dir, questions, tax, folha, sig, model, hashes)


def targets_of(ctx: Context, g: Gabarito) -> set[int]:
    ids = [g.tema_central_id, *g.extras_ids] if g.tema_central_id else []
    return {ctx.tax.level1_of(t) for t in ids} & set(ctx.sig.candidates)


def _features(ctx: Context, train_idx: np.ndarray, train_targets: list[set[int]], cfg: Config) -> np.ndarray:
    X = np.concatenate([ctx.sig.X, label_features(ctx.sig, train_idx, train_targets)], axis=2)
    if cfg.feature_mask:
        X = X[:, :, [FEATURES.index(f) for f in cfg.feature_mask]]
    return X


def _fit_predict(ctx: Context, train: list[Gabarito], cfg: Config) -> np.ndarray:
    """Treina com os itens de `train` (fora da taxonomia excluídos) e devolve p para todas as questões."""
    idx = ctx.index
    labeled = [g for g in train if targets_of(ctx, g)]
    tr_idx = np.array([idx[g.id] for g in labeled])
    tr_targets = [targets_of(ctx, g) for g in labeled]
    X = _features(ctx, tr_idx, tr_targets, cfg)
    pos = {c: j for j, c in enumerate(ctx.sig.candidates)}
    T = np.zeros((len(labeled), len(pos)), dtype=bool)
    for r, ts in enumerate(tr_targets):
        for t in ts:
            T[r, pos[t]] = True
    ranker = Ranker.fit(X[tr_idx], T, l2=cfg.l2)
    return ranker.proba(X), ranker, X


def _names(cfg: Config) -> list[str]:
    return cfg.feature_mask or FEATURES


def cross_validate(ctx: Context, cfg: Config) -> list[dict]:
    """Validação cruzada repetida na folha-verdade; devolve predições fora da amostra."""
    rng = np.random.default_rng(SEED)
    folha = ctx.folha
    idx = ctx.index
    out = []
    for rep in range(cfg.cv_repeats):
        order = rng.permutation(len(folha))
        folds = np.array_split(order, cfg.cv_folds)
        for f, test in enumerate(folds):
            test_set = set(test.tolist())
            train = [g for i, g in enumerate(folha) if i not in test_set]
            P, _, X = _fit_predict(ctx, train, cfg)
            top, Z = confidence_inputs(P, X, _names(cfg))
            for i in test:
                g = folha[i]
                qi = idx[g.id]
                j = int(top[qi])
                pred = ctx.sig.candidates[j]
                out.append({
                    "repeticao": rep, "id": g.id, "p_top": float(P[qi, j]), "pred": pred,
                    "z": [float(v) for v in Z[qi]],
                    "acerto": bool(ctx.tax.is_correct(pred, g.tema_central_id, g.extras_ids)),
                    "fora": g.fora, "tem_embedding": bool(ctx.sig.has_embedding[idx[g.id]]),
                })
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - r) / d, (c + r) / d)


def choose_threshold(conf: np.ndarray, correct: np.ndarray, cfg: Config) -> float:
    """Menor limiar (>= min_threshold) em que a precisão OOF acumulada fica >= alvo
    para todo limiar maior (curva monotônica), evitando escolher um ponto isolado de sorte."""
    grid = np.round(np.arange(cfg.min_threshold, 0.995, 0.005), 3)
    ok = []
    for t in grid:
        m = conf >= t
        prec = correct[m].mean() if m.any() else 1.0
        ok.append(prec >= cfg.target_precision)
    best = grid[-1]
    for t, good in zip(grid[::-1], ok[::-1]):
        if not good:
            break
        best = t
    return float(best)


def summarize_cv(oof: list[dict], cal: Calibrator, threshold: float) -> dict:
    y = np.array([r["acerto"] for r in oof])
    conf = cal(np.array([r["z"] for r in oof]))
    pub = conf >= threshold
    reps = sorted({r["repeticao"] for r in oof})
    per_rep = []
    for rep in reps:
        m = np.array([r["repeticao"] == rep for r in oof])
        k = int((pub & m & y).sum())
        n = int((pub & m).sum())
        per_rep.append({"publicadas": n, "acertos": k, "cobertura": n / m.sum(),
                        "precisao": (k / n) if n else None, "top1": float(y[m].mean())})
    k_all, n_all = int((pub & y).sum()), int(pub.sum())
    return {
        "top1_acerto": float(y.mean()),
        "publicadas_oof_total": n_all, "acertos_oof_total": k_all,
        "precisao_oof": (k_all / n_all) if n_all else None,
        "precisao_oof_wilson95_por_item": wilson(round(k_all / len(reps)), round(n_all / len(reps))),
        "cobertura_oof": float(pub.mean()),
        "brier_calibrado": float(((conf - y) ** 2).mean()),
        "confianca_media_publicadas": float(conf[pub].mean()) if pub.any() else None,
        "por_repeticao": per_rep,
    }


def run(data_dir: Path, artifacts_dir: Path, out_csv: Path, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    t0 = time.time()
    ctx = load_context(data_dir, artifacts_dir)
    tax = ctx.tax

    # 1) Validação cruzada na folha-verdade -> calibração e limiar.
    oof = cross_validate(ctx, cfg)
    z_oof = np.array([r["z"] for r in oof])
    y_oof = np.array([r["acerto"] for r in oof])
    cal = Calibrator.fit(z_oof, y_oof)
    threshold = cfg.threshold if cfg.threshold is not None else choose_threshold(cal(z_oof), y_oof, cfg)
    cv = summarize_cv(oof, cal, threshold)

    # 2) Modelo final com os 100 rótulos.
    P, ranker, X = _fit_predict(ctx, ctx.folha, cfg)
    _, Z = confidence_inputs(P, X, _names(cfg))
    conf_all = cal(Z)
    folha_by_id = {g.id: g for g in ctx.folha}
    cands = ctx.sig.candidates
    rows, audit = [], []
    for i, q in enumerate(ctx.questions):
        p = P[i]
        order = np.argsort(-p)
        top = cands[int(order[0])]
        p_top = float(p[order[0]])
        conf = float(conf_all[i])
        sugestoes = [{"tema_id": cands[int(j)], "caminho": tax.nodes[cands[int(j)]].caminho,
                      "p": round(float(p[j]), 4)} for j in order[:5]]
        g = folha_by_id.get(q.id)
        if g is not None:
            # Rótulo da equipe: usado como está (não entra na avaliação, LEIA-ME).
            if g.fora:
                row = dict(id=q.id, decisao="fora", tema_central_id="", extras_ids="", confianca=1.0,
                           motivo="folha-verdade: fora da taxonomia segundo a equipe")
            else:
                row = dict(id=q.id, decisao="publicar", tema_central_id=g.tema_central_id,
                           extras_ids=";".join(map(str, g.extras_ids)), confianca=1.0,
                           motivo="folha-verdade: tema atribuído pela equipe")
            origem = "folha_verdade"
        elif not ctx.sig.has_embedding[i]:
            row = dict(id=q.id, decisao="revisar", tema_central_id=top, extras_ids="", confianca=round(conf, 4),
                       motivo="sem embedding em cache para o texto atual; rode o passo offline de embeddings")
            origem = "sistema"
        elif conf >= threshold:
            row = dict(id=q.id, decisao="publicar", tema_central_id=top, extras_ids="",
                       confianca=round(conf, 4), motivo="")
            origem = "sistema"
        else:
            alt = sugestoes[1]
            row = dict(id=q.id, decisao="revisar", tema_central_id=top, extras_ids="", confianca=round(conf, 4),
                       motivo=f"confiança {conf:.2f} abaixo de {threshold:.2f}; 2ª opção: {alt['tema_id']} "
                              f"({tax.nodes[alt['tema_id']].titulo})")
            origem = "sistema"
        rows.append(row)
        audit.append({"id": q.id, "origem_decisao": origem, **{k: row[k] for k in COLUMNS},
                      "p_top": round(p_top, 4), "concordancia_sinais": int(Z[i, 2]),
                      "candidatos": sugestoes,
                      "disciplina_origem": q.disciplina_origem, "assunto_origem": q.assunto_origem})

    validate_publication(rows, {q.id for q in ctx.questions}, tax)
    write_publication(rows, out_csv)

    eval_rows = [r for r in rows if r["id"] not in folha_by_id]
    pub_eval = [r for r in eval_rows if r["decisao"] == "publicar"]
    expected_prec = float(np.mean([r["confianca"] for r in pub_eval])) if pub_eval else None
    counts = {d: sum(r["decisao"] == d for r in rows) for d in ("publicar", "revisar", "fora")}
    counts_eval = {d: sum(r["decisao"] == d for r in eval_rows) for d in ("publicar", "revisar", "fora")}
    meta = {
        "versao_codigo": __version__, "versao_politica": POLICY_VERSION,
        "chama_modelo_de_linguagem": False,
        "modelo_embeddings_em_cache": ctx.embedding_model,
        "hashes_sha256": ctx.hashes,
        "config": {k: v for k, v in cfg.__dict__.items()},
        "features": cfg.feature_mask or FEATURES,
        "pesos_ranqueador": dict(zip(cfg.feature_mask or FEATURES, map(float, ranker.w))),
        "calibrador": {"entradas": CONF_FEATURES, "coef": [float(c) for c in cal.coef],
                       "intercepto": cal.intercept},
        "limiar_publicacao": threshold,
        "questoes": len(rows), "sem_embedding": int((~ctx.sig.has_embedding).sum()),
        "decisoes_todas": counts, "decisoes_fora_da_folha": counts_eval,
        "cobertura_fora_da_folha": counts_eval["publicar"] / len(eval_rows),
        "precisao_esperada_pelo_calibrador": expected_prec,
        "validacao_cruzada": cv,
        "publicadas_por_lote": {
            "banco_Q0001_Q0359": sum(r["id"] <= "Q0359" for r in pub_eval),
            "importacao_nova": sum(r["id"] > "Q0359" for r in pub_eval),
        },
    }
    conf_oof = cal(z_oof)
    meta["previsao"] = forecast(oof, conf_oof, threshold, counts_eval["publicar"], len(eval_rows), expected_prec)
    meta["segundos"] = round(time.time() - t0, 1)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "run_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (artifacts_dir / "previsao.txt").write_text(render(meta["previsao"], meta), encoding="utf-8")
    with (artifacts_dir / "decision_audit.jsonl").open("w", encoding="utf-8") as fh:
        for a in audit:
            fh.write(json.dumps(a, ensure_ascii=False) + "\n")
    with (artifacts_dir / "validacao_cruzada_oof.jsonl").open("w", encoding="utf-8") as fh:
        for r in oof:
            fh.write(json.dumps({**r, "confianca": round(float(cal(np.array(r["z"]))[0]), 4)},
                                ensure_ascii=False) + "\n")
    return meta
