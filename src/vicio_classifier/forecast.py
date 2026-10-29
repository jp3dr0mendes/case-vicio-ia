"""previsao.txt: previsão de precisão e cobertura nas questões fora da folha-verdade.

Precisão: a única evidência rotulada é a folha-verdade, usada fora da amostra (validação
cruzada repetida). O intervalo combina duas incertezas por simulação:
  1) quanto a precisão verdadeira pode variar, dado que só ~100 questões rotuladas existem
     (bootstrap das questões da folha, mantendo o limiar fixo);
  2) o sorteio binomial das n questões que de fato publicamos no conjunto avaliado.
Cobertura: é uma contagem das nossas próprias decisões, então é conhecida exatamente.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

SEED = 7


def forecast(oof: list[dict], conf_oof: np.ndarray, threshold: float, n_pub_eval: int, n_eval: int,
             expected_by_calibrator: float | None, n_sim: int = 20000) -> dict:
    by_item: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for r, c in zip(oof, conf_oof):
        by_item[r["id"]].append((bool(c >= threshold), bool(r["acerto"])))
    items = list(by_item)
    pub = np.array([sum(p for p, _ in by_item[i]) for i in items], dtype=float)
    hit = np.array([sum(p and a for p, a in by_item[i]) for i in items], dtype=float)
    point = hit.sum() / pub.sum() if pub.sum() else float("nan")

    rng = np.random.default_rng(SEED)
    sims = []
    for _ in range(n_sim):
        s = rng.integers(0, len(items), len(items))
        if pub[s].sum() == 0:
            continue
        p_true = hit[s].sum() / pub[s].sum()
        sims.append(rng.binomial(n_pub_eval, p_true) / n_pub_eval if n_pub_eval else p_true)
    lo, hi = (np.percentile(sims, [10, 90]) if sims else (float("nan"),) * 2)
    cov = n_pub_eval / n_eval
    return {
        "precisao_oof_folha": float(point),
        "publicacoes_oof_por_repeticao": float(pub.sum() / max(1, len({r['repeticao'] for r in oof}))),
        "precisao_esperada_calibrador": expected_by_calibrator,
        "precisao_prevista": float(point if expected_by_calibrator is None
                                   else min(point, expected_by_calibrator)),
        "precisao_intervalo_80": [float(lo), float(min(1.0, hi))],
        "cobertura_prevista": cov,
        "cobertura_intervalo": [cov, cov],
        "n_publicadas_avaliadas": n_pub_eval, "n_avaliadas": n_eval,
    }


def render(f: dict, meta: dict) -> str:
    cv = meta["validacao_cruzada"]
    lo, hi = f["precisao_intervalo_80"]
    return f"""precisao_prevista: {f['precisao_prevista']:.2f}
precisao_intervalo: {lo:.2f}-{hi:.2f}
cobertura_prevista: {f['cobertura_prevista']:.2f}
cobertura_intervalo: {f['cobertura_intervalo'][0]:.2f}-{f['cobertura_intervalo'][1]:.2f}

Como cheguei nesses números (gerado por `python -m vicio_classifier classify`; detalhes em artifacts/run_metadata.json)

Cobertura. É uma contagem das minhas próprias decisões: publiquei {f['n_publicadas_avaliadas']} das {f['n_avaliadas']} questões fora da
folha-verdade ({f['cobertura_prevista']:.1%}). Não há incerteza sobre ela; por isso o intervalo é um ponto.

Precisão. A única evidência rotulada são os 100 exemplos da folha-verdade. Rodei validação cruzada repetida
({meta['config']['cv_folds']} partes x {meta['config']['cv_repeats']} repetições): em cada rodada o sistema é treinado sem as questões testadas,
e as questões testadas recebem tema e confiança como se fossem novas. O calibrador (regressão logística sobre
probabilidade do topo, margem para o 2º tema e concordância entre sinais) e o limiar de publicação
({meta['limiar_publicacao']:.3f}) foram escolhidos nessas predições fora da amostra.
- Nessas predições, o sistema publicaria {f['publicacoes_oof_por_repeticao']:.1f} das 100 questões por rodada, com precisão de
  {f['precisao_oof_folha']:.1%} (inclui as questões "fora" da folha, que contam como erro se publicadas).
- A confiança média que o calibrador atribui às {f['n_publicadas_avaliadas']} questões publicadas é {f['precisao_esperada_calibrador']:.1%}.
A previsão pontual é o menor dos dois números. O intervalo (80%) simula duas fontes de incerteza: bootstrap das
questões da folha (só ~{f['publicacoes_oof_por_repeticao']:.0f} publicações por rodada sustentam a estimativa) e o sorteio binomial das
{f['n_publicadas_avaliadas']} publicações reais.

Limitações que pesam contra a previsão:
1. A folha-verdade só tem questões Q0001-Q0359 (banco antigo). As Q0360-Q0659 (importação nova) vêm sem comando,
   com assunto de origem em outro formato e sem a disciplina "Legislação Especial". Elas recebem
   {meta['publicadas_por_lote']['importacao_nova']} das publicações; a calibração não foi testada nesse lote.
2. 8% da folha é "fora da taxonomia" e o sistema não sabe reconhecer esses casos com 8 exemplos; quando um deles
   parece com um tema, ele pode ser publicado e conta como erro. Esse é o principal tipo de erro com confiança alta.
3. Com ~{f['publicacoes_oof_por_repeticao']:.0f} publicações por rodada, um único erro a mais muda a precisão estimada em vários pontos.
Isto é uma previsão, não uma medição: a precisão real só é conhecida com o gabarito da equipe.
"""
