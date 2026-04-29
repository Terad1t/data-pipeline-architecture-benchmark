FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md /app/
COPY src /app/src

RUN pip install --upgrade pip \
    && pip install -e .

CMD ["python", "-m", "pibic_sentiment.cli", "--help"]
