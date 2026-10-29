"""Leitura, limpeza e validação dos anexos (questoes.jsonl, temas.csv, folha_verdade.csv)."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

QUESTION_FIELDS = {"id", "enunciado_html", "comando", "alternativas", "disciplina_origem", "assunto_origem"}
TEMA_FIELDS = ["tema_id", "parent_id", "disciplina", "nivel", "titulo", "caminho", "descricao", "status"]
FOLHA_FIELDS = ["id", "tema_central_id", "tema_central_caminho", "extras_ids", "fora_da_taxonomia"]


class DataError(ValueError):
    """Dado de entrada inválido: interrompe a execução em vez de produzir saída errada."""


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_html(value: object) -> str:
    """Remove tags preservando o texto; converte <br>/<p> em espaço e decodifica entidades."""
    if value is None:
        return ""
    text = re.sub(r"(?i)<\s*(br|/p|/li|/div)[^>]*>", " ", str(value))
    text = html.unescape(_TAG.sub(" ", text))
    return _WS.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Minúsculas e sem acentos: base para regras léxicas e TF-IDF."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


@dataclass
class Question:
    id: str
    enunciado: str
    comando: str
    alternativas: dict[str, str]
    disciplina_origem: str
    assunto_origem: str
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def text(self) -> str:
        alts = " ".join(f"{k}) {v}" for k, v in self.alternativas.items())
        return " ".join(p for p in (self.enunciado, self.comando, alts) if p)


@dataclass
class Tema:
    tema_id: int
    parent_id: int | None
    disciplina: str
    nivel: int
    titulo: str
    caminho: str
    descricao: str
    status: str


@dataclass
class Gabarito:
    id: str
    tema_central_id: int | None
    extras_ids: list[int]
    fora: bool


def load_questions(path: Path) -> list[Question]:
    if not path.exists():
        raise DataError(f"Arquivo não encontrado: {path}")
    out: list[Question] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataError(f"{path.name} linha {n}: JSON inválido ({exc})") from exc
        missing = QUESTION_FIELDS - item.keys()
        if missing:
            raise DataError(f"{path.name} linha {n}: campos ausentes {sorted(missing)}")
        alts = item.get("alternativas") or {}
        if isinstance(alts, list):
            alts = {chr(65 + i): v for i, v in enumerate(alts)}
        out.append(Question(
            id=str(item["id"]).strip(),
            enunciado=clean_html(item.get("enunciado_html")),
            comando=clean_html(item.get("comando")),
            alternativas={str(k): clean_html(v) for k, v in alts.items()},
            disciplina_origem=(item.get("disciplina_origem") or "").strip(),
            assunto_origem=(item.get("assunto_origem") or "").strip(),
            raw=item,
        ))
    ids = [q.id for q in out]
    if any(not i for i in ids):
        raise DataError("Questão com id vazio")
    if len(ids) != len(set(ids)):
        raise DataError("IDs de questão duplicados")
    return out


def load_temas(path: Path) -> list[Tema]:
    if not path.exists():
        raise DataError(f"Arquivo não encontrado: {path}")
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != TEMA_FIELDS:
            raise DataError(f"temas.csv com colunas inesperadas: {reader.fieldnames}")
        rows = list(reader)
    temas = []
    for r in rows:
        try:
            temas.append(Tema(
                tema_id=int(r["tema_id"]),
                parent_id=int(r["parent_id"]) if r["parent_id"].strip() else None,
                disciplina=r["disciplina"].strip(),
                nivel=int(r["nivel"]),
                titulo=r["titulo"].strip(),
                caminho=r["caminho"].strip(),
                descricao=r["descricao"].strip(),
                status=r["status"].strip().lower(),
            ))
        except ValueError as exc:
            raise DataError(f"temas.csv: linha inválida {r}") from exc
    return temas


def load_folha(path: Path) -> list[Gabarito]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != FOLHA_FIELDS:
            raise DataError(f"folha_verdade.csv com colunas inesperadas: {reader.fieldnames}")
        rows = list(reader)
    out = []
    for r in rows:
        fora = r["fora_da_taxonomia"].strip() == "1"
        central = int(r["tema_central_id"]) if r["tema_central_id"].strip() else None
        if fora == (central is not None):
            raise DataError(f"folha_verdade: {r['id']} incoerente entre fora e tema central")
        extras = [int(x) for x in r["extras_ids"].split(";") if x.strip()]
        out.append(Gabarito(r["id"].strip(), central, extras, fora))
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
