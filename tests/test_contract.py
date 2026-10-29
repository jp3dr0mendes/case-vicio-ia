import pytest

from vicio_classifier.contract import ContractError, validate_publication

IDS = {"Q1", "Q2", "Q3"}


def rows():
    return [
        dict(id="Q1", decisao="publicar", tema_central_id=10, extras_ids="20", confianca=0.97, motivo=""),
        dict(id="Q2", decisao="revisar", tema_central_id=11, extras_ids="", confianca=0.4, motivo="dúvida"),
        dict(id="Q3", decisao="fora", tema_central_id="", extras_ids="", confianca=0.9, motivo="sem tema"),
    ]


def test_publicacao_valida(small_tax):
    validate_publication(rows(), IDS, small_tax)


@pytest.mark.parametrize("i, campo, valor", [
    (0, "decisao", "talvez"),
    (0, "tema_central_id", ""),          # publicar sem tema
    (0, "tema_central_id", 30),          # inativo
    (0, "tema_central_id", 31),          # filho de inativo
    (0, "tema_central_id", 1),           # raiz
    (0, "tema_central_id", 999),         # inexistente
    (1, "tema_central_id", ""),          # revisar sem sugestão
    (2, "tema_central_id", 10),          # fora com tema
    (0, "confianca", 1.2),
    (0, "confianca", "alta"),
    (1, "motivo", ""),                   # revisar sem motivo
    (2, "motivo", " "),                  # fora sem motivo
    (0, "extras_ids", "10"),             # extra igual ao central
    (0, "extras_ids", "20;20"),          # extra repetido
    (0, "extras_ids", "30"),             # extra inativo
])
def test_violacoes_do_contrato(small_tax, i, campo, valor):
    r = rows()
    r[i][campo] = valor
    with pytest.raises(ContractError):
        validate_publication(r, IDS, small_tax)


def test_ids_duplicados_ausentes_ou_extras(small_tax):
    with pytest.raises(ContractError):
        validate_publication(rows() + [rows()[0]], IDS, small_tax)
    with pytest.raises(ContractError):
        validate_publication(rows()[:2], IDS, small_tax)
    with pytest.raises(ContractError):
        validate_publication(rows(), IDS - {"Q3"}, small_tax)
