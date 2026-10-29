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
