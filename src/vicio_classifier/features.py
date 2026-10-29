"""Sinais por par (questão, tema candidato de nível 1).

Os candidatos são os temas de nível 1 selecionáveis. Pela régua, publicar no tema de nível 1
acerta tanto quando a equipe usou o próprio tema quanto quando usou um subtema dele, enquanto
escolher um subtema arrisca errar para um irmão. Cada candidato agrega o próprio tema e os
subtemas (máximo dos sinais).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from vicio_classifier.data import Question, normalize
from vicio_classifier.legal import LegalRules
from vicio_classifier.taxonomy import Taxonomy

ROOT_BY_ORIGEM = {
    "penal": "Penal", "direito penal": "Penal", "direito penal militar": "Penal",
    "processo penal": "Processo Penal", "direito processual penal": "Processo Penal",
    "direito processual penal militar": "Processo Penal",
    "direito penal - legislacao especial": "Direito Penal - Legislação Especial",
    "criminologia": "Criminologia",
}

STATIC_FEATURES = [
    "emb", "emb_gap", "tfidf", "tfidf_gap", "lei", "artigo", "origem", "origem_gap", "disciplina",
    "clone", "volume", "volume_desconhecido",
]
_VOLUME = re.compile(r"(\d+) questões no source")
LABEL_FEATURES = ["knn", "knn_gap", "origem_rotulada", "prior"]
FEATURES = STATIC_FEATURES + LABEL_FEATURES


def _gap(m: np.ndarray) -> np.ndarray:
    return m - m.max(axis=1, keepdims=True)


@dataclass
class StaticSignals:
    ids: list[str]
    candidates: list[int]
    X: np.ndarray            # (n_questoes, n_candidatos, n_static)
    has_embedding: np.ndarray  # (n_questoes,) bool
    q_emb: np.ndarray        # (n_questoes, d) normalizado (zeros se ausente)
    origem_key: list[str]    # assunto de origem normalizado ("" se ausente)


def build_static(
    questions: list[Question], tax: Taxonomy, profiles: dict[int, str], rules: LegalRules,
    q_emb: dict[str, np.ndarray], t_emb: dict[int, np.ndarray],
) -> StaticSignals:
    cands = tax.level1_selectable()
    members = {c: [c] + [d for d in tax.descendants(c) if d in tax.selectable_ids] for c in cands}
    n, k = len(questions), len(cands)

    # Embeddings: cosseno com o perfil do tema e dos subtemas (máximo).
    dim = len(next(iter(t_emb.values()))) if t_emb else 1
    Q = np.zeros((n, dim), dtype=np.float32)
    has = np.zeros(n, dtype=bool)
    for i, q in enumerate(questions):
        if q.id in q_emb:
            Q[i], has[i] = q_emb[q.id], True
    emb = np.zeros((n, k), dtype=np.float32)
    for j, c in enumerate(cands):
        vs = [t_emb[m] for m in members[c] if m in t_emb]
        if vs:  # tema sem embedding em cache: sinal fica 0 (o ranqueador usa os demais)
            emb[:, j] = (Q @ np.stack(vs).T).max(axis=1)

    # TF-IDF (palavras 1-2, sem acento) entre a questão e o perfil textual.
    q_txt = [normalize(q.text) for q in questions]
    member_ids = sorted({m for c in cands for m in members[c]})
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, strip_accents=None)
    vec.fit(q_txt + [normalize(profiles[m]) for m in member_ids])
    Vq = vec.transform(q_txt)
    Vt = vec.transform([normalize(profiles[m]) for m in member_ids])
    sim = (Vq @ Vt.T).toarray()
    col = {m: i for i, m in enumerate(member_ids)}
    tfidf = np.stack([sim[:, [col[m] for m in members[c]]].max(axis=1) for c in cands], axis=1)

    # Assunto de origem (ruidoso) vs títulos do tema e subtemas.
    vec_o = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    titles = [normalize(tax.nodes[m].titulo) for m in member_ids]
    origem_txt = [normalize(q.assunto_origem) for q in questions]
    vec_o.fit(titles + origem_txt)
    so = (vec_o.transform(origem_txt) @ vec_o.transform(titles).T).toarray()
    origem = np.stack([so[:, [col[m] for m in members[c]]].max(axis=1) for c in cands], axis=1)
    empty_origem = np.array([not q.assunto_origem for q in questions])
    origem[empty_origem] = 0.0

    lei = np.zeros((n, k), dtype=np.float32)
    art = np.zeros((n, k), dtype=np.float32)
    pos = {c: j for j, c in enumerate(cands)}
    for i, q in enumerate(questions):
        hits = rules.extract(q.text)
        for t in hits.temas_lei:
            if t in pos:
                lei[i, pos[t]] = 1.0
        for t in hits.temas_artigo:
            if t in pos:
                art[i, pos[t]] = 1.0

    disc = np.zeros((n, k), dtype=np.float32)
    root_title = [tax.nodes[tax.root_of(c)].titulo for c in cands]
    for i, q in enumerate(questions):
        r = ROOT_BY_ORIGEM.get(normalize(q.disciplina_origem))
        if r:
            disc[i] = [1.0 if rt == r else 0.0 for rt in root_title]

    # Sinais do próprio tema, iguais para toda questão:
    # - clone: nó sem descrição na plataforma (duplicatas como "Penal > Tráfico de Drogas" e
    #   "Teoria da Pena (Parte 1)"); a folha-verdade quase nunca os usa.
    # - volume: log do nº de questões que o tema tem na plataforma (campo descricao).
    clone = np.array([0.0 if tax.nodes[c].descricao else 1.0 for c in cands], dtype=np.float32)
    vol = np.full(k, np.nan, dtype=np.float32)
    for j, c in enumerate(cands):
        counts = [int(m.group(1)) for x in members[c] if (m := _VOLUME.search(tax.nodes[x].descricao))]
        if counts:
            vol[j] = np.log1p(max(counts[0] if _VOLUME.search(tax.nodes[c].descricao) else 0, sum(counts)))
    unknown = np.isnan(vol).astype(np.float32)
    known = vol[~np.isnan(vol)]
    vol = np.where(np.isnan(vol), np.median(known) if known.size else 0.0, vol)
    clone_m, vol_m, unk_m = (np.broadcast_to(v, (n, k)) for v in (clone, vol, unknown))

    X = np.stack([emb, _gap(emb), tfidf, _gap(tfidf), lei, art, origem, _gap(origem), disc,
                  clone_m, vol_m, unk_m], axis=2)
    X[~has, :, 0:2] = 0.0
    keys = [normalize(q.assunto_origem).strip() for q in questions]
    return StaticSignals([q.id for q in questions], cands, X.astype(np.float32), has, Q, keys)


def label_features(
    sig: StaticSignals, train_idx: np.ndarray, train_targets: list[set[int]],
) -> np.ndarray:
    """kNN sobre os exemplos rotulados de treino e prior de frequência dos temas.

    A própria questão nunca conta como vizinha dela mesma (evita vazamento no treino).
    """
    n, k = sig.X.shape[:2]
    pos = {c: j for j, c in enumerate(sig.candidates)}
    Y = np.zeros((len(train_idx), k), dtype=np.float32)
    for r, targets in enumerate(train_targets):
        for t in targets:
            Y[r, pos[t]] = 1.0
    S = sig.q_emb @ sig.q_emb[train_idx].T  # (n, n_train)
    for r, i in enumerate(train_idx):
        S[i, r] = -1.0
    knn = np.zeros((n, k), dtype=np.float32)
    for j in range(k):
        cols = Y[:, j] > 0
        if cols.any():
            knn[:, j] = np.clip(S[:, cols].max(axis=1), 0, None)
    knn[~sig.has_embedding] = 0.0
    # Convenção da equipe por assunto de origem: entre os exemplos de treino com o mesmo
    # assunto_origem, que fração foi para cada tema (suavizado: +1 no denominador).
    orig = np.zeros((n, k), dtype=np.float32)
    by_key: dict[str, list[int]] = {}
    for r, i in enumerate(train_idx):
        if sig.origem_key[i]:
            by_key.setdefault(sig.origem_key[i], []).append(r)
    for i in range(n):
        rs = [r for r in by_key.get(sig.origem_key[i], []) if train_idx[r] != i]
        if rs:
            orig[i] = Y[rs].sum(axis=0) / (len(rs) + 1.0)
    counts = Y.sum(axis=0)
    prior = np.log((counts + 0.5) / (counts.sum() + 0.5 * k))
    prior = np.broadcast_to(prior, (n, k))
    return np.stack([knn, _gap(knn), orig, prior], axis=2).astype(np.float32)
