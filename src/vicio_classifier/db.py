"""Persistência opcional da execução no PostgreSQL (rastreabilidade, não é necessária para gerar o CSV).

Cada execução vira uma linha em `runs` (metadados + hashes) e 659 linhas em `decisions`.
Reprocessar cria um novo run_id: o histórico não é sobrescrito.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        BIGSERIAL PRIMARY KEY,
    criado_em     TIMESTAMPTZ NOT NULL DEFAULT now(),
    versao_codigo TEXT NOT NULL,
    versao_politica TEXT NOT NULL,
    limiar        DOUBLE PRECISION NOT NULL,
    metadata      JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS decisions (
    run_id          BIGINT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    questao_id      TEXT NOT NULL,
    decisao         TEXT NOT NULL CHECK (decisao IN ('publicar', 'revisar', 'fora')),
    tema_central_id INTEGER,
    extras_ids      TEXT NOT NULL DEFAULT '',
    confianca       DOUBLE PRECISION NOT NULL CHECK (confianca BETWEEN 0 AND 1),
    motivo          TEXT NOT NULL DEFAULT '',
    origem_decisao  TEXT NOT NULL,
    auditoria       JSONB NOT NULL,
    PRIMARY KEY (run_id, questao_id)
);
"""


def save_run(database_url: str, meta: dict, audit_path: Path) -> int:
    import psycopg

    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    with psycopg.connect(database_url, connect_timeout=10) as conn, conn.cursor() as cur:
        cur.execute(SCHEMA)
        cur.execute(
            "INSERT INTO runs (versao_codigo, versao_politica, limiar, metadata) VALUES (%s, %s, %s, %s) "
            "RETURNING run_id",
            (meta["versao_codigo"], meta["versao_politica"], meta["limiar_publicacao"], json.dumps(meta)),
        )
        run_id = cur.fetchone()[0]
        cur.executemany(
            "INSERT INTO decisions (run_id, questao_id, decisao, tema_central_id, extras_ids, confianca, "
            "motivo, origem_decisao, auditoria) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [(run_id, r["id"], r["decisao"], int(r["tema_central_id"]) if str(r["tema_central_id"]) else None,
              str(r["extras_ids"]), float(r["confianca"]), r["motivo"], r["origem_decisao"], json.dumps(r))
             for r in rows],
        )
        cur.execute("SELECT count(*) FROM decisions WHERE run_id = %s", (run_id,))
        if cur.fetchone()[0] != len(rows):
            raise RuntimeError("contagem de decisões gravadas difere da execução")
    return run_id
