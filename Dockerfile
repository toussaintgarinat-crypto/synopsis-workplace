# ── Stage 1: Build (dépendances, couche cachée tant que les requirements ne changent pas) ──
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt requirements-full.txt ./
RUN pip install --no-cache-dir --prefix=/install -r requirements-full.txt

# ── Stage 2: Final (mode complet, avec ffmpeg) ──
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local
WORKDIR /app
COPY . .
EXPOSE 6090
ENV PYTHONPATH="/app"
ENV PORT=6090
ENV HOST=0.0.0.0
ENV SYNOPSIS_MODE=full
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:6090/sante')"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "6090", "--log-level", "info"]
