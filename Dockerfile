# ============================================================
# EasyML — Production Dockerfile
# ============================================================
# Multi-stage build:
#   Stage 1 (builder)  → install Python deps in an isolated layer
#   Stage 2 (runtime)  → slim image with only runtime libraries
# ============================================================

# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile native wheels (psycopg2, numpy, scipy…)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install massive packages separately to avoid WSL OOM hangs during wheel extraction
RUN pip install --no-cache-dir --prefix=/install tensorflow==2.18.0
RUN pip install --no-cache-dir --prefix=/install xgboost==2.1.4 catboost==1.2.7 lightgbm==4.5.0
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="EasyML Team"
LABEL description="EasyML Flask API served via Gunicorn"

# Runtime-only system libraries
#   libpq5        → PostgreSQL client lib (psycopg2)
#   libgomp1      → OpenMP (LightGBM, XGBoost, sklearn n_jobs)
#   curl          → container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built Python packages from builder
COPY --from=builder /install /usr/local

# Create non-root user for security
RUN groupadd -r easyml && useradd -r -g easyml -d /app -s /sbin/nologin easyml

WORKDIR /app

# Copy application code
COPY . .

# Create writable directories the app expects
RUN mkdir -p uploads outputs outputs/checkpoints \
    && chown -R easyml:easyml /app

USER easyml

# ── Environment defaults (overridden by docker-compose / .env) ──
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production \
    CELERY_BROKER_URL=redis://redis:6379/0 \
    CELERY_RESULT_BACKEND=redis://redis:6379/1 \
    DATABASE_URL=postgresql://easyml:easyml@db:5432/easyml

EXPOSE 5000

# ── Health check ──
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# ── Entrypoint: Gunicorn ──
# 4 sync workers, 120s timeout (ML training can be slow on /api/pipeline/run)
CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
