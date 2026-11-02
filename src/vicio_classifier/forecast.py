"""previsao.txt: previsão de precisão e cobertura nas questões fora da folha-verdade.

Precisão: a única evidência rotulada é a folha-verdade. A estimativa vem da avaliação
aninhada (pipeline.nested_evaluation): a regra inteira (calibrador + limiar) é ajustada sem as
questões em que é medida. O intervalo combina duas incertezas por simulação:
  1) quanto a precisão verdadeira pode variar com só 100 questões rotuladas
     (bootstrap das questões da folha sobre os resultados aninhados);
  2) o sorteio binomial das n questões que de fato publicamos no conjunto avaliado.
Cobertura: é uma contagem das nossas próprias decisões, então é conhecida exatamente.
"""
from __future__ import annotations

import numpy as np

SEED = 7


def forecast(nested: dict, n_pub_eval: int, n_eval: int, expected_by_calibrator: float | None,
             n_sim: int = 20000) -> dict:
    items = sorted(nested["por_questao"])
    pub = np.array([nested["por_questao"][i][0] for i in items], dtype=float)
    hit = np.array([nested["por_questao"][i][1] for i in items], dtype=float)
    seen = np.array([nested["por_questao"][i][2] for i in items], dtype=float)
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
    pred = point if expected_by_calibrator is None else min(point, expected_by_calibrator)
    return {
        "precisao_aninhada_folha": float(point),
        "publicacoes_aninhadas_por_100": float(100 * pub.sum() / seen.sum()),
        "erros_aninhados_por_questao": nested["erros_por_questao"],
        "precisao_esperada_calibrador": expected_by_calibrator,
        "precisao_prevista": float(pred),
        "precisao_intervalo_80": [float(lo), float(min(1.0, hi))],
        "cobertura_prevista": cov,
        "cobertura_intervalo": [cov, cov],
        "n_publicadas_avaliadas": n_pub_eval, "n_avaliadas": n_eval,
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%".replace(".", ",")


def render(f: dict, meta: dict) -> str:
    lo, hi = f["precisao_intervalo_80"]
    erros = f["erros_aninhados_por_questao"]
    erros_txt = ", ".join(sorted(erros)) if erros else "nenhuma questão"
    lote = meta["publicadas_por_lote"]
    cv = meta["validacao_cruzada"]
    return f"""precisao_prevista: {f['precisao_prevista']:.2f}
precisao_intervalo: {lo:.2f}-{hi:.2f}
cobertura_prevista: {f['cobertura_prevista']:.2f}
cobertura_intervalo: {f['cobertura_intervalo'][0]:.2f}-{f['cobertura_intervalo'][1]:.2f}

Como cheguei nesses números (o arquivo é gerado pelo comando `classify`; os detalhes estão em
artifacts/run_metadata.json)

Cobertura. É uma contagem das minhas próprias decisões: publiquei {f['n_publicadas_avaliadas']} das {f['n_avaliadas']} questões que estão fora
da folha-verdade ({_pct(f['cobertura_prevista'])}). Como não depende do gabarito, o intervalo é um ponto só.

Precisão. A única evidência rotulada são as 100 questões da folha-verdade, e todas foram usadas fora da amostra:
1. Validação cruzada repetida ({meta['config']['cv_folds']} partes x {meta['config']['cv_repeats']} repetições): cada questão recebe tema e confiança
   de um modelo treinado sem ela.
2. Avaliação aninhada da regra de decisão: o calibrador e o limiar são ajustados em 90% das questões e
   testados nos 10% que eles não viram (10 partes x 20 sorteios). Esse passo importa: se eu medisse a
   precisão nas mesmas predições usadas para escolher o limiar, ela sairia otimista ({_pct(cv['precisao_oof'])} em vez
   de {_pct(f['precisao_aninhada_folha'])}), porque o limiar seria escolhido justamente para deixar de fora os erros conhecidos.
Na avaliação aninhada, a regra publica {f['publicacoes_aninhadas_por_100']:.0f} de cada 100 questões, com precisão de {_pct(f['precisao_aninhada_folha'])}.
O único erro recorrente é {erros_txt}, uma fronteira de convenção (Provas em Espécie x Teoria Geral das Provas).
A confiança média que o calibrador dá às {f['n_publicadas_avaliadas']} questões que publiquei é {_pct(f['precisao_esperada_calibrador'])}.
A previsão é o menor desses dois números. O intervalo de 80% soma duas incertezas: ter só 100 questões
rotuladas (bootstrap da folha) e o sorteio das {f['n_publicadas_avaliadas']} publicações reais. O limite de cima chega a 1,00
porque, em boa parte dos sorteios, nenhum erro cai entre as publicações; não é uma promessa de 100%.

O que pode deixar a precisão real abaixo da prevista:
1. A folha-verdade só tem questões do banco antigo (Q0001-Q0359). A importação nova (Q0360-Q0659) vem sem
   comando, com o assunto de origem em outro formato e sem a disciplina "Legislação Especial". Ela recebe
   {lote['importacao_nova']} das {f['n_publicadas_avaliadas']} publicações, e nenhum rótulo confirma a calibração nesse lote. Simulei a perda do
   assunto de origem na folha (scripts/estresse_origem.py): a cobertura cai e a precisão fica perto de 95%.
2. Questões que a equipe considera "fora da taxonomia", mas que parecem ter tema, contam como erro se
   publicadas. A regra só publica quando pelo menos 3 de 4 sinais independentes concordam, o que barrou os
   casos da folha (Q0043 e Q0039 ficam em 0,92), mas não há como garantir que barre todos. Escolhi essa
   forma de calibrar depois de olhar a folha, o que a avaliação aninhada não consegue descontar.
3. Cada erro pesa muito: na folha, cerca de {f['publicacoes_aninhadas_por_100']:.0f} publicações por rodada sustentam a estimativa.
Isto é uma previsão, não uma medição: a precisão real só se conhece com o gabarito da equipe.
"""
