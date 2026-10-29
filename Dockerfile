FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

COPY pyproject.toml ./
COPY src/ src/
COPY tests/ tests/
COPY data/anexos/ data/anexos/
COPY artifacts/regras/ artifacts/regras/
COPY artifacts/embeddings/ artifacts/embeddings/
COPY artifacts/temas_enriquecidos.jsonl artifacts/
COPY prompts/ prompts/

RUN useradd --create-home --uid 10001 app && mkdir -p /app/saida && chown -R app:app /app
USER app

ENTRYPOINT ["python", "-m", "vicio_classifier"]
CMD ["--artifacts-dir", "/app/artifacts", "classify", "--out", "/app/saida/publicacao.csv"]
