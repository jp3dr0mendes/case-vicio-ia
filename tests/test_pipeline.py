"""Ponta a ponta nos dados reais, sem rede: prova que a classificação não chama modelo."""
import csv
import socket

import pytest

from vicio_classifier.contract import read_publication, validate_publication
from vicio_classifier.data import load_folha, load_questions, load_temas
from vicio_classifier.pipeline import Config, run
from vicio_classifier.taxonomy import Taxonomy

from conftest import ARTIFACTS, DATA, ROOT


@pytest.fixture
def sem_rede(monkeypatch):
    def bloqueado(*a, **k):
        raise RuntimeError("rede bloqueada no teste: a classificação não pode chamar serviços")
    monkeypatch.setattr(socket, "socket", bloqueado)
    monkeypatch.setattr(socket, "create_connection", bloqueado)


def test_classify_roda_sem_rede_e_cumpre_contrato(tmp_path, sem_rede):
    art = tmp_path / "artifacts"
    art.mkdir()
    for sub in ("regras", "embeddings"):
        (art / sub).symlink_to(ARTIFACTS / sub)
    enr = ARTIFACTS / "temas_enriquecidos.jsonl"
    if enr.exists():
        (art / enr.name).symlink_to(enr)
    out = tmp_path / "publicacao.csv"
    meta = run(DATA, art, out, Config(cv_repeats=1))
    rows = read_publication(out)
    tax = Taxonomy(load_temas(DATA / "temas.csv"))
    validate_publication(rows, {q.id for q in load_questions(DATA / "questoes.jsonl")}, tax)
    assert meta["chama_modelo_de_linguagem"] is False
    assert meta["sem_embedding"] == 0
    # Política: nada publicado abaixo do limiar; fora só vem da folha-verdade.
    folha = {g.id: g for g in load_folha(DATA / "folha_verdade.csv")}
    for r in rows:
        if r["id"] in folha:
            continue
        assert r["decisao"] in {"publicar", "revisar"}
        if r["decisao"] == "publicar":
            assert float(r["confianca"]) >= meta["limiar_publicacao"]
            assert tax.nodes[int(r["tema_central_id"])].nivel == 1


def test_entrega_versionada_cumpre_contrato_e_respeita_folha():
    path = ROOT / "publicacao.csv"
    if not path.exists():
        pytest.skip("publicacao.csv ainda não gerado")
    rows = read_publication(path)
    tax = Taxonomy(load_temas(DATA / "temas.csv"))
    validate_publication(rows, {q.id for q in load_questions(DATA / "questoes.jsonl")}, tax)
    by_id = {r["id"]: r for r in rows}
    for g in load_folha(DATA / "folha_verdade.csv"):
        r = by_id[g.id]
        if g.fora:
            assert r["decisao"] == "fora"
        else:
            assert int(r["tema_central_id"]) == g.tema_central_id
    assert (ARTIFACTS / "publicacao.csv").read_bytes() == path.read_bytes()
    with path.open(encoding="utf-8") as fh:
        assert next(csv.reader(fh)) == ["id", "decisao", "tema_central_id", "extras_ids", "confianca", "motivo"]


def test_previsao_segue_o_formato_do_leia_me():
    import re

    path = ROOT / "previsao.txt"
    if not path.exists():
        pytest.skip("previsao.txt ainda não gerado")
    linhas = path.read_text(encoding="utf-8").splitlines()
    padroes = [r"precisao_prevista: \d\.\d\d", r"precisao_intervalo: \d\.\d\d-\d\.\d\d",
               r"cobertura_prevista: \d\.\d\d", r"cobertura_intervalo: \d\.\d\d-\d\.\d\d"]
    for linha, padrao in zip(linhas[:4], padroes):
        assert re.fullmatch(padrao, linha), linha
    lo, hi = map(float, linhas[1].split(": ")[1].split("-"))
    assert lo <= float(linhas[0].split(": ")[1]) <= hi
    assert len(linhas) > 6  # explicação em texto livre depois das 4 linhas


def test_avaliacao_aninhada_nao_ve_a_questao_testada():
    """Uma questão sempre errada e muito confiante só pode ser publicada quando está fora do ajuste."""
    from vicio_classifier.pipeline import nested_evaluation

    oof = []
    for rep in range(3):
        for i in range(60):
            certo = i != 0 and not (i >= 20 and i % 2 == 0)  # Q0000 confiante e errada; metade das fracas erra
            z = [0.99, 0.9, 4.0] if i < 20 else [0.3, 0.05, 1.0]
            oof.append({"repeticao": rep, "id": f"Q{i:04d}", "z": z, "acerto": certo})
    res = nested_evaluation(oof, Config(), seeds=2, folds=5)
    assert set(res["por_questao"]) == {f"Q{i:04d}" for i in range(60)}
    assert all(v[2] == 6 for v in res["por_questao"].values())  # cada questão testada 1x por sorteio x 3 reps
    assert res["precisao"] is not None and 0.0 <= res["precisao"] <= 1.0
