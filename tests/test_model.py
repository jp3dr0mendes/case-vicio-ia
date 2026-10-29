import numpy as np

from vicio_classifier.model import Calibrator, Ranker, confidence_inputs


def test_ranker_aprende_sinal_util_e_ignora_ruido():
    rng = np.random.default_rng(0)
    n, k = 200, 8
    X = rng.normal(size=(n, k, 2))
    gold = rng.integers(0, k, size=n)
    X[np.arange(n), gold, 0] += 3.0          # feature 0 indica o tema certo
    T = np.zeros((n, k), dtype=bool)
    T[np.arange(n), gold] = True
    r = Ranker.fit(X, T, l2=0.1)
    P = r.proba(X)
    assert np.allclose(P.sum(axis=1), 1)
    assert (P.argmax(axis=1) == gold).mean() > 0.8
    assert abs(r.w[0]) > 5 * abs(r.w[1])


def test_calibrador_monotonico_e_concordancia():
    P = np.array([[0.9, 0.1], [0.6, 0.4]])
    X = np.zeros((2, 2, 1)); X[0, 0, 0] = 1.0
    top, Z = confidence_inputs(P, X, ["emb"])
    assert list(top) == [0, 0] and Z[0, 2] == 1 and Z[1, 2] == 0
    z = np.array([[-2, -2, 0], [0, 0, 1], [2, 2, 2], [3, 3, 3]] * 10, dtype=float)
    y = np.array([0, 0, 1, 1] * 10)
    cal = Calibrator.fit(z, y)
    c = cal(z[:4])
    assert np.all(np.diff(c) > 0) and np.all((c >= 0) & (c <= 1))
