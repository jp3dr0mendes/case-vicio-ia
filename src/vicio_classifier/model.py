"""Ranqueador (logit condicional) + calibração.

p(tema | questão) = softmax(w · x) sobre os candidatos de nível 1. O treino maximiza a
probabilidade do conjunto de temas corretos (tema central e extras, levados ao nível 1).
A calibração mapeia p_top -> P(acerto) com regressão logística em logit(p_top), ajustada
em predições fora da amostra (validação cruzada na folha-verdade).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Standardizer":
        flat = X.reshape(-1, X.shape[-1])
        std = flat.std(axis=0)
        return cls(flat.mean(axis=0), np.where(std < 1e-6, 1.0, std))

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


@dataclass
class Ranker:
    w: np.ndarray
    scaler: Standardizer

    @classmethod
    def fit(cls, X: np.ndarray, targets: np.ndarray, l2: float = 1.0) -> "Ranker":
        """X: (n, k, f); targets: (n, k) bool com os candidatos corretos (>=1 por linha)."""
        scaler = Standardizer.fit(X)
        Z = scaler(X)
        n, _, f = Z.shape

        def loss(w: np.ndarray) -> tuple[float, np.ndarray]:
            s = Z @ w                                   # (n, k)
            lse_all = logsumexp(s, axis=1)
            s_pos = np.where(targets, s, -np.inf)
            lse_pos = logsumexp(s_pos, axis=1)
            p_all = np.exp(s - lse_all[:, None])
            p_pos = np.where(targets, np.exp(s - lse_pos[:, None]), 0.0)
            nll = float((lse_all - lse_pos).sum()) + l2 * float(w @ w)
            grad = np.einsum("nk,nkf->f", p_all - p_pos, Z) + 2 * l2 * w
            return nll, grad

        res = minimize(loss, np.zeros(f), jac=True, method="L-BFGS-B")
        return cls(res.x, scaler)

    def proba(self, X: np.ndarray) -> np.ndarray:
        s = self.scaler(X) @ self.w
        return np.exp(s - logsumexp(s, axis=1, keepdims=True))


AGREEMENT_SIGNALS = ["emb", "tfidf", "knn", "origem"]
CONF_FEATURES = ["logit_p_top", "logit_margem", "concordancia"]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def confidence_inputs(P: np.ndarray, X: np.ndarray, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Para cada questão: (índice do candidato do topo, entradas do calibrador).

    concordancia = quantos sinais independentes (embedding, TF-IDF, kNN, assunto de origem)
    apontam, sozinhos, para o mesmo tema que o ranqueador escolheu (0 a 4).
    """
    order = np.argsort(-P, axis=1)
    top, second = order[:, 0], order[:, 1]
    rows = np.arange(P.shape[0])
    p_top, p_2 = P[rows, top], P[rows, second]
    agree = np.zeros(P.shape[0])
    for name in AGREEMENT_SIGNALS:
        if name in feature_names:
            sig = X[:, :, feature_names.index(name)]
            has_signal = sig.max(axis=1) > 0
            agree += ((sig.argmax(axis=1) == top) & has_signal).astype(float)
    Z = np.stack([_logit(p_top), _logit(np.clip(p_top - p_2, 1e-4, 1)), agree], axis=1)
    return top, Z


@dataclass
class Calibrator:
    """Regressão logística P(acerto | entradas de confiança), ajustada fora da amostra."""
    coef: np.ndarray
    intercept: float

    @classmethod
    def fit(cls, Z: np.ndarray, correct: np.ndarray, l2: float = 1.0) -> "Calibrator":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1.0 / l2, max_iter=1000).fit(Z, correct.astype(int))
        return cls(lr.coef_[0].astype(float), float(lr.intercept_[0]))

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        z = np.atleast_2d(Z) @ self.coef + self.intercept
        return 1 / (1 + np.exp(-z))
