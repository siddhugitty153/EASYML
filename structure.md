# ════════════════════════════════════════════════════════════════
#  EasyML — Production Revamp: Complete Structure & Roadmap
# ════════════════════════════════════════════════════════════════
#  Version : 2.0 (Production-Grade Rewrite)
#  Date    : August 2026
#  Author  : EasyML Engineering
# ════════════════════════════════════════════════════════════════


## Table of Contents

1. [Current State Audit](#1--current-state-audit)
2. [Target Architecture](#2--target-architecture)
3. [Technology Stack](#3--technology-stack-full-reference)
4. [Directory Structure — Before & After](#4--directory-structure--before--after)
5. [Phase 1 — Foundation (Docker + Config)](#5--phase-1--foundation)
6. [Phase 2 — Database Migration (SQLite → PostgreSQL)](#6--phase-2--database-migration)
7. [Phase 3 — Celery Task Queue Integration](#7--phase-3--celery-task-queue)
8. [Phase 4 — API Hardening & Security](#8--phase-4--api-hardening--security)
9. [Phase 5 — Frontend Decoupling](#9--phase-5--frontend-decoupling)
10. [Phase 6 — Observability & Monitoring](#10--phase-6--observability--monitoring)
11. [Phase 7 — CI/CD Pipeline](#11--phase-7--cicd-pipeline)
12. [Phase 8 — Cloud Deployment](#12--phase-8--cloud-deployment)
13. [Phase 9 — Performance & Scalability](#13--phase-9--performance--scalability)
14. [Phase 10 — Testing Strategy](#14--phase-10--testing-strategy)
15. [Environment Variable Reference](#15--environment-variable-reference)
16. [Risk Register & Mitigations](#16--risk-register--mitigations)
17. [Timeline Estimate](#17--timeline-estimate)

---

## 1 — Current State Audit

### 1.1  What Exists Today

```
Easy ML-app/
├── app.py                  # 853-line monolithic Flask server
├── database.py             # SQLite experiment tracker
├── requirements.txt        # 20 dependencies (dev only)
├── static/
│   ├── index.html          # 32 KB single-page dashboard
│   ├── style.css           # 57 KB stylesheet
│   └── app.js              # 67 KB monolithic frontend JS
├── pipeline/
│   ├── __init__.py
│   ├── ml_pipeline.py      # 1074-line orchestrator
│   ├── data_cleaning.py    # 325 lines
│   ├── data_transformer.py # 242 lines
│   ├── data_profiler.py    # 10 KB
│   ├── data_splitter.py    # 4.9 KB
│   ├── feature_engine.py   # 345 lines
│   ├── dim_reduction.py    # 10 KB
│   ├── imbalance_handler.py# 4.5 KB
│   ├── model_trainer.py    # 245 lines
│   ├── model_evaluator.py  # 16 KB
│   ├── explainer.py        # 922 lines (largest module)
│   ├── visualization.py    # 8.7 KB
│   └── code_generator.py   # 16 KB
├── uploads/                # User-uploaded CSV files
└── outputs/                # Models, checkpoints, reports
```

### 1.2  Critical Production Gaps

| Gap | Severity | Details |
|-----|----------|---------|
| **No WSGI server** | 🔴 Critical | Runs on Flask dev server (`app.run(debug=False)`) — single-threaded, no graceful shutdown |
| **Global mutable state** | 🔴 Critical | `pipeline`, `training_status`, `training_thread` are module-level globals — breaks with multiple Gunicorn workers |
| **SQLite under concurrency** | 🔴 Critical | `database.py` uses SQLite with no WAL mode — fails under concurrent writes |
| **threading.Thread for async** | 🟠 High | Background training uses Python threads (GIL-bound) — no retry, no persistence, no horizontal scaling |
| **No authentication** | 🟠 High | All API endpoints are open — no auth, no rate limiting |
| **No input validation** | 🟠 High | Uploaded files not scanned; JSON payloads not schema-validated |
| **Hardcoded paths** | 🟡 Medium | `UPLOAD_DIR`, `OUTPUT_DIR`, `DB_PATH` use `os.path.dirname(__file__)` — breaks in containers |
| **No health checks** | 🟡 Medium | No `/health` or `/ready` endpoints for orchestrators |
| **No structured logging** | 🟡 Medium | Uses `print()` and bare `traceback.format_exc()` |
| **Monolithic frontend** | 🟡 Medium | 67 KB `app.js` + 57 KB `style.css` — no build step, no code splitting |
| **No tests in CI** | 🟡 Medium | Test files exist but no CI pipeline to run them |
| **Pickle security** | 🟡 Medium | Model export via `pickle.dump` with no signing |
| **No CORS config** | 🟢 Low | Works today (same-origin) but blocks future API consumers |

---

## 2 — Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LOAD BALANCER                            │
│                    (Nginx / Cloud ALB)                          │
└──────────────┬──────────────────────────────┬───────────────────┘
               │                              │
    ┌──────────▼──────────┐        ┌──────────▼──────────┐
    │   Flask + Gunicorn  │        │   Flask + Gunicorn  │
    │   (Web Worker ×N)   │        │   (Web Worker ×N)   │
    │                     │        │                     │
    │ • REST API routes   │        │ • REST API routes   │
    │ • File upload       │        │ • File upload       │
    │ • Task dispatch     │        │ • Task dispatch     │
    │ • Result polling    │        │ • Result polling    │
    └──────────┬──────────┘        └──────────┬──────────┘
               │ .delay()                     │ .delay()
    ┌──────────▼──────────────────────────────▼──────────┐
    │                   REDIS                            │
    │           (Broker + Result Backend)                 │
    │                                                    │
    │  • Task queue (celery)                             │
    │  • Result store (JSON/pickle)                      │
    │  • Pub/Sub for real-time status                    │
    └──────────┬─────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐    ┌─────────────────────────┐
    │  Celery Worker ×M   │    │      PostgreSQL          │
    │                     │    │                           │
    │ • Feature eng.      │    │ • Experiment tracking     │
    │ • Model training    │    │ • User sessions           │
    │ • SHAP computation  │    │ • Task result metadata    │
    │ • Auto-Explore      │    │                           │
    └──────────┬──────────┘    └───────────────────────────┘
               │
    ┌──────────▼──────────┐
    │   Shared Volume     │
    │   (S3 / NFS / EFS)  │
    │                     │
    │ • uploads/          │
    │ • outputs/          │
    │ • checkpoints/      │
    │ • models/           │
    └─────────────────────┘
```

---

## 3 — Technology Stack (Full Reference)

### 3.1  Backend

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Language** | Python | 3.11+ | Core runtime |
| **Web Framework** | Flask | 3.1.x | REST API serving |
| **WSGI Server** | Gunicorn | ≥22.0 | Production HTTP server (prefork workers) |
| **Task Queue** | Celery | ≥5.4 | Distributed async task execution |
| **Message Broker** | Redis | 7.x (Alpine) | Celery broker + result backend |
| **Database** | PostgreSQL | 16.x (Alpine) | Persistent experiment tracking, user data |
| **DB Adapter** | psycopg2-binary | ≥2.9.9 | Python ↔ PostgreSQL |
| **ORM (optional)** | SQLAlchemy | ≥2.0 | If schema complexity grows |
| **Migrations** | Alembic | ≥1.13 | Schema version control |

### 3.2  ML / Data Science

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | 2.2.x | DataFrame operations |
| numpy | 2.2.x | Numerical computing |
| scikit-learn | 1.6.x | Core ML algorithms, preprocessing |
| XGBoost | 2.1.x | Gradient boosted trees |
| LightGBM | 4.5.x | Fast gradient boosting |
| CatBoost | 1.2.x | Categorical-native boosting |
| TensorFlow | 2.18.x | Deep learning (MLP, future models) |
| Optuna | 4.2.x | Bayesian hyperparameter tuning |
| SHAP | ≥0.43 | Model explainability |
| imbalanced-learn | 0.12.x | SMOTE, class imbalance handling |
| UMAP | 0.5.x | Dimensionality reduction |
| joblib | 1.4.x | Model serialization |

### 3.3  AI / NLP

| Library | Purpose |
|---------|---------|
| google-generativeai ≥0.8.2 | Gemini API for chatbot + failure analysis |

### 3.4  Infrastructure

| Tool | Purpose |
|------|---------|
| **Docker** | Containerization |
| **Docker Compose** | Local multi-service orchestration |
| **Nginx** | Reverse proxy + static file serving (production) |
| **GitHub Actions** | CI/CD pipeline |
| **Prometheus** | Metrics collection |
| **Grafana** | Dashboards |
| **Sentry** | Error tracking |
| **Flower** | Celery task monitoring dashboard |

### 3.5  Frontend (Current → Future)

| Current | Future Candidate | Rationale |
|---------|-----------------|-----------|
| Vanilla HTML/CSS/JS | **React + Vite** (or keep vanilla) | Current monolith works; migrate only if team velocity demands it |
| No build step | Vite build pipeline | Tree-shaking, code splitting, HMR |
| 67 KB `app.js` | Component-based modules | Maintainability |

### 3.6  Testing

| Tool | Purpose |
|------|---------|
| pytest | Unit + integration tests |
| pytest-cov | Coverage reporting |
| httpx / requests | API endpoint testing |
| Playwright | End-to-end browser tests |
| locust | Load / stress testing |

---

## 4 — Directory Structure — Before & After

### After Revamp

```
Easy ML-app/
│
├── docker-compose.yml          # 4-service stack definition
├── docker-compose.prod.yml     # Production overrides (no bind mounts)
├── Dockerfile                  # Multi-stage Python 3.11-slim
├── .env.example                # Environment variable template
├── .env                        # Local secrets (git-ignored)
├── .dockerignore               # Exclude __pycache__, .git, uploads/
├── .gitignore
├── requirements.txt            # Pinned production deps
├── requirements-dev.txt        # pytest, flake8, mypy, locust, etc.
├── Makefile                    # Convenience commands
│
├── app.py                      # Flask app factory + route registration
├── config.py                   # [NEW] Centralized configuration (env-based)
├── celery_app.py               # [NEW] Celery factory
├── tasks.py                    # [NEW] Celery task definitions
├── extensions.py               # [NEW] Shared extensions (db, redis, etc.)
│
├── database.py                 # [REWRITE] SQLite → PostgreSQL via psycopg2
│
├── pipeline/
│   ├── __init__.py
│   ├── ml_pipeline.py          # [REFACTOR] Remove global state dependence
│   ├── data_cleaning.py        # [UNCHANGED]
│   ├── data_transformer.py     # [UNCHANGED]
│   ├── data_profiler.py        # [UNCHANGED]
│   ├── data_splitter.py        # [UNCHANGED]
│   ├── feature_engine.py       # [UNCHANGED]
│   ├── dim_reduction.py        # [UNCHANGED]
│   ├── imbalance_handler.py    # [UNCHANGED]
│   ├── model_trainer.py        # [UNCHANGED]
│   ├── model_evaluator.py      # [UNCHANGED]
│   ├── explainer.py            # [UNCHANGED]
│   ├── visualization.py        # [UNCHANGED]
│   └── code_generator.py       # [UNCHANGED]
│
├── middleware/                  # [NEW]
│   ├── __init__.py
│   ├── auth.py                 # API key / JWT authentication
│   ├── rate_limiter.py         # Token bucket rate limiting
│   ├── error_handler.py        # Centralized error handling
│   └── request_validator.py    # JSON schema validation
│
├── static/                     # Frontend assets
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── nginx/                      # [NEW] Reverse proxy config
│   └── nginx.conf
│
├── tests/                      # [NEW] Organized test suite
│   ├── conftest.py             # Shared fixtures
│   ├── test_api.py             # API endpoint tests
│   ├── test_pipeline.py        # Pipeline unit tests
│   ├── test_tasks.py           # Celery task tests
│   ├── test_database.py        # DB layer tests
│   └── test_e2e.py             # End-to-end Playwright tests
│
├── scripts/                    # [NEW] Operational scripts
│   ├── init_db.py              # Database initialization / migration
│   ├── seed_data.py            # Load sample datasets
│   └── health_check.sh         # Container health check script
│
├── monitoring/                 # [NEW] Observability configs
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboards/
│           └── easyml.json
│
├── uploads/                    # User-uploaded files (volume-mounted)
├── outputs/                    # Models, reports, checkpoints (volume-mounted)
│
└── docs/
    └── structure.md            # ← This document
```

---

## 5 — Phase 1 — Foundation

> **Goal**: Containerize the app so every developer runs the identical environment.

### Step 1.1 — Create `config.py`

Centralize all configuration into a single module that reads from environment variables:

```python
# config.py — Centralized Configuration (12-Factor App)

import os

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))

    # Paths — container-friendly (no __file__ gymnastics)
    BASE_DIR = os.environ.get('APP_BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL', f'sqlite:///{os.path.join(BASE_DIR, "outputs", "experiment_history.db")}')

    # Celery
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')

    # Gemini
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')

class DevelopmentConfig(Config):
    DEBUG = True
    FLASK_ENV = 'development'

class ProductionConfig(Config):
    DEBUG = False
    FLASK_ENV = 'production'

class TestingConfig(Config):
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'

config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
}
```

### Step 1.2 — Create `Dockerfile` (multi-stage)

✅ **Already created** — see `Dockerfile` in project root.

Key decisions:
- `python:3.11-slim` base (~150 MB vs ~1 GB for full image)
- Builder stage compiles `psycopg2`, `numpy`, `scipy` wheels
- Runtime stage has only `libpq5` + `libgomp1`
- Non-root `easyml` user
- `HEALTHCHECK` via `curl`

### Step 1.3 — Create `docker-compose.yml`

✅ **Already created** — 4 services: `web`, `celery_worker`, `redis`, `db`.

### Step 1.4 — Create `.env.example`

✅ **Already created**.

### Step 1.5 — Create `.dockerignore`

```
__pycache__
*.pyc
.git
.env
*.rar
uploads/*
outputs/*
reproduction_outputs/
strat_test_outputs/
*.log
*.txt
!requirements.txt
```

### Step 1.6 — Create `Makefile`

```makefile
.PHONY: up down build logs test shell db-init

up:
	docker compose up -d --build

down:
	docker compose down

build:
	docker compose build --no-cache

logs:
	docker compose logs -f web celery_worker

test:
	docker compose exec web pytest tests/ -v --tb=short

shell:
	docker compose exec web bash

db-init:
	docker compose exec web python scripts/init_db.py

worker-logs:
	docker compose logs -f celery_worker

flower:
	docker compose exec celery_worker celery -A celery_app.celery flower --port=5555
```

### Step 1.7 — Verify

```bash
cp .env.example .env
make up
# Verify: http://localhost:5000 should load the dashboard
make logs
```

---

## 6 — Phase 2 — Database Migration

> **Goal**: Replace SQLite with PostgreSQL for concurrent writes and durability.

### Step 2.1 — Rewrite `database.py`

Replace `sqlite3.connect()` with `psycopg2` connection pool:

```python
# database.py — PostgreSQL Experiment Tracker

import os
import json
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get(
    'DATABASE_URL',
    'postgresql://easyml:easyml@localhost:5432/easyml'
)

# Connection pool (min 2, max 10 connections)
_pool = None

def _get_pool():
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(2, 10, DATABASE_URL)
    return _pool

@contextmanager
def get_conn():
    """Context manager for pooled PostgreSQL connections."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)

def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    dataset_name TEXT,
                    task_type TEXT,
                    target_col TEXT,
                    best_model TEXT,
                    config_json JSONB,
                    metrics_json JSONB,
                    n_rows INTEGER,
                    n_features INTEGER,
                    n_models_trained INTEGER,
                    duration_seconds REAL,
                    auto_explore BOOLEAN DEFAULT FALSE
                )
            ''')

# ... (save_run, get_runs, delete_run, clear_history stay the same
#      but use get_conn() context manager + %s placeholders instead of ?)
```

### Step 2.2 — Key Differences from SQLite

| Feature | SQLite | PostgreSQL |
|---------|--------|-----------|
| Concurrent writes | ❌ Single writer | ✅ MVCC — unlimited writers |
| Connection pooling | N/A | `ThreadedConnectionPool` (2–10) |
| JSON support | TEXT + `json.loads()` | Native `JSONB` (indexable, queryable) |
| Placeholder syntax | `?` | `%s` |
| Timestamps | TEXT | `TIMESTAMPTZ` (timezone-aware) |
| Boolean | INTEGER (0/1) | Native `BOOLEAN` |

### Step 2.3 — Migration Script

```python
# scripts/init_db.py
"""Initialize or migrate the PostgreSQL database."""
import database
database.init_db()
print("✅ Database initialized successfully")
```

### Step 2.4 — Backwards Compatibility

Keep a SQLite fallback for local dev without Docker:

```python
# In config.py
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///...')

# In database.py
if DATABASE_URL.startswith('sqlite'):
    # use existing sqlite3 logic
else:
    # use psycopg2 pool
```

---

## 7 — Phase 3 — Celery Task Queue

> **Goal**: Replace `threading.Thread` with Celery for durable, retryable, scalable async tasks.

### Step 3.1 — Create `celery_app.py`

```python
import os
from celery import Celery

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')

celery = Celery(
    'easyml',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['tasks'],
)

celery.conf.update(
    task_serializer='pickle',
    result_serializer='pickle',
    accept_content=['pickle', 'json'],
    task_time_limit=3600,
    task_soft_time_limit=3300,
    result_expires=86400,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_max_tasks_per_child=10,
)
```

### Step 3.2 — Create `tasks.py`

Define three granularity levels:

| Task | Inputs | Duration | Use Case |
|------|--------|----------|----------|
| `run_pipeline_task` | DataFrame dict + config | 1–60 min | Full end-to-end pipeline |
| `feature_engineer_task` | DataFrame dict + config | 5–60 sec | Standalone feature engineering |
| `train_models_task` | X_train + y_train + config | 1–30 min | Standalone model training |

### Step 3.3 — Modify `app.py`

Key changes to `app.py`:

1. **Import Celery** and task definitions at the top
2. **Replace** `training_thread` / `training_status` globals with `active_task_id`
3. **Replace** `/api/pipeline/run-async` to call `run_pipeline_task.delay()`
4. **Replace** `/api/pipeline/status` to query `AsyncResult(task_id)`
5. **Add** `/api/tasks/<task_id>/revoke` endpoint for cancellation

### Step 3.4 — Progress Tracking

Inside Celery tasks, use `self.update_state()`:

```python
@celery.task(bind=True)
def run_pipeline_task(self, ...):
    self.update_state(state='PROGRESS', meta={'progress': 10, 'step': 'cleaning'})
    # ... do work ...
    self.update_state(state='PROGRESS', meta={'progress': 50, 'step': 'training'})
```

Frontend polls `/api/pipeline/status?task_id=xxx` which reads the meta from Redis.

### Step 3.5 — Flower Dashboard

Add Flower (Celery monitoring) to `docker-compose.yml`:

```yaml
flower:
  build: .
  container_name: easyml-flower
  command: celery -A celery_app.celery flower --port=5555
  ports:
    - "5555:5555"
  depends_on:
    - redis
```

Access at `http://localhost:5555` — shows active tasks, worker status, task history.

---

## 8 — Phase 4 — API Hardening & Security

> **Goal**: Make every endpoint production-safe.

### Step 4.1 — Input Validation

Use `jsonschema` or `marshmallow` to validate all POST payloads:

```python
# middleware/request_validator.py
from functools import wraps
from flask import request, jsonify

PIPELINE_RUN_SCHEMA = {
    'type': 'object',
    'required': ['target_col'],
    'properties': {
        'target_col': {'type': 'string', 'minLength': 1},
        'task': {'type': 'string', 'enum': ['classification', 'regression']},
        'models': {'type': 'array', 'items': {'type': 'string'}},
    },
    'additionalProperties': True,
}

def validate_json(schema):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            from jsonschema import validate, ValidationError
            try:
                validate(instance=request.json, schema=schema)
            except ValidationError as e:
                return jsonify({'error': f'Validation error: {e.message}'}), 422
            return f(*args, **kwargs)
        return wrapper
    return decorator
```

### Step 4.2 — Authentication

Add API key middleware:

```python
# middleware/auth.py
from functools import wraps
from flask import request, jsonify
import os

API_KEY = os.environ.get('EASYML_API_KEY')

def require_api_key(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not API_KEY:
            return f(*args, **kwargs)  # No key configured = open access
        key = request.headers.get('X-API-Key') or request.args.get('api_key')
        if key != API_KEY:
            return jsonify({'error': 'Invalid or missing API key'}), 401
        return f(*args, **kwargs)
    return wrapper
```

### Step 4.3 — Rate Limiting

```bash
pip install flask-limiter
```

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per hour"],
    storage_uri="redis://redis:6379/2",
)

@app.route('/api/pipeline/run', methods=['POST'])
@limiter.limit("10 per hour")
def run_pipeline():
    ...
```

### Step 4.4 — CORS

```python
from flask_cors import CORS
CORS(app, resources={r"/api/*": {"origins": os.environ.get('CORS_ORIGINS', '*')}})
```

### Step 4.5 — Centralized Error Handling

```python
# middleware/error_handler.py
import logging
import traceback
from flask import jsonify

logger = logging.getLogger(__name__)

def register_error_handlers(app):
    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({'error': 'Bad request', 'detail': str(e)}), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Not found'}), 404

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({'error': 'File too large', 'max_bytes': app.config['MAX_CONTENT_LENGTH']}), 413

    @app.errorhandler(429)
    def rate_limited(e):
        return jsonify({'error': 'Rate limit exceeded', 'retry_after': e.description}), 429

    @app.errorhandler(500)
    def internal_error(e):
        logger.error(f"Internal error: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500
```

### Step 4.6 — Health & Readiness Endpoints

```python
@app.route('/health')
def health():
    """Liveness probe — is the process alive?"""
    return jsonify({'status': 'ok'}), 200

@app.route('/ready')
def ready():
    """Readiness probe — are dependencies reachable?"""
    checks = {}

    # Redis check
    try:
        import redis
        r = redis.from_url(app.config['CELERY_BROKER_URL'])
        r.ping()
        checks['redis'] = 'ok'
    except Exception as e:
        checks['redis'] = str(e)

    # DB check
    try:
        from database import get_conn
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT 1')
        checks['database'] = 'ok'
    except Exception as e:
        checks['database'] = str(e)

    all_ok = all(v == 'ok' for v in checks.values())
    return jsonify({'status': 'ready' if all_ok else 'degraded', 'checks': checks}), 200 if all_ok else 503
```

### Step 4.7 — File Upload Security

```python
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
MAX_FILENAME_LENGTH = 255

def validate_upload(file):
    """Validate uploaded file before saving."""
    if not file or file.filename == '':
        raise ValueError('No file selected')

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f'Unsupported format: {ext}. Use CSV or Excel.')

    if len(file.filename) > MAX_FILENAME_LENGTH:
        raise ValueError('Filename too long')

    return ext
```

---

## 9 — Phase 5 — Frontend Decoupling

> **Goal**: Separate the static frontend from the Flask API to enable independent scaling.

### Option A — Keep Vanilla (Recommended for now)

1. Move static serving to **Nginx** (much faster than Flask's `send_from_directory`)
2. Add cache headers for `style.css` and `app.js`
3. Serve the Flask API on `/api/*` only

```nginx
# nginx/nginx.conf
upstream flask_api {
    server web:5000;
}

server {
    listen 80;

    # Static files — served by Nginx directly
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;
    }

    # API — proxied to Flask
    location /api/ {
        proxy_pass http://flask_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
        client_max_body_size 100M;
    }
}
```

### Option B — Migrate to React + Vite (Future)

If the frontend grows significantly, migrate to a component-based framework:

```bash
npx -y create-vite@latest frontend -- --template react
```

This is **not required** for production readiness — only for long-term maintainability if the dashboard grows.

---

## 10 — Phase 6 — Observability & Monitoring

> **Goal**: Know what's happening in production at all times.

### Step 6.1 — Structured Logging

Replace `print()` with Python `logging`:

```python
# In app.py
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('easyml')
```

### Step 6.2 — Prometheus Metrics

```bash
pip install prometheus-flask-instrumentator
```

```python
from prometheus_flask_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

Key metrics auto-exported:
- Request count by endpoint + status code
- Request latency histogram
- Active request gauge

### Step 6.3 — Custom ML Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

PIPELINE_RUNS = Counter('easyml_pipeline_runs_total', 'Total pipeline runs', ['task_type', 'status'])
PIPELINE_DURATION = Histogram('easyml_pipeline_duration_seconds', 'Pipeline run duration')
MODELS_TRAINED = Counter('easyml_models_trained_total', 'Models trained', ['model_type'])
ACTIVE_TASKS = Gauge('easyml_active_celery_tasks', 'Currently running Celery tasks')
```

### Step 6.4 — Sentry Error Tracking

```bash
pip install sentry-sdk[flask]
```

```python
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.celery import CeleryIntegration

sentry_sdk.init(
    dsn=os.environ.get('SENTRY_DSN'),
    integrations=[FlaskIntegration(), CeleryIntegration()],
    traces_sample_rate=0.1,
)
```

### Step 6.5 — Grafana Dashboard

Create a dashboard with panels for:
- Pipeline runs per hour
- Average training duration by model type
- Celery queue depth
- Error rate
- Redis memory usage
- PostgreSQL connection pool utilization

---

## 11 — Phase 7 — CI/CD Pipeline

> **Goal**: Every push is tested, linted, and deployable.

### `.github/workflows/ci.yml`

```yaml
name: EasyML CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install flake8 mypy
      - run: flake8 app.py pipeline/ --max-line-length=120
      - run: mypy pipeline/ --ignore-missing-imports

  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports: ['6379:6379']
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: easyml_test
        ports: ['5432:5432']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt -r requirements-dev.txt
      - run: pytest tests/ -v --tb=short --cov=pipeline --cov-report=xml
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/easyml_test
          CELERY_BROKER_URL: redis://localhost:6379/0

  docker-build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t easyml:test .
      - run: docker compose -f docker-compose.yml config  # Validate compose file

  deploy:
    needs: [lint, test, docker-build]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      # Add your deployment steps here (e.g., push to ECR, deploy to ECS/GKE)
```

---

## 12 — Phase 8 — Cloud Deployment

> **Goal**: Deploy to a cloud provider with auto-scaling.

### Option A — AWS (ECS + Fargate)

| Component | AWS Service |
|-----------|-------------|
| Web containers | ECS Fargate (Flask + Gunicorn) |
| Celery workers | ECS Fargate (separate task definition) |
| Redis | ElastiCache for Redis |
| PostgreSQL | RDS PostgreSQL |
| File storage | S3 (replace local uploads/outputs) |
| Load balancer | Application Load Balancer (ALB) |
| DNS | Route 53 |
| SSL | ACM (AWS Certificate Manager) |

### Option B — GCP (Cloud Run + GKE)

| Component | GCP Service |
|-----------|-------------|
| Web containers | Cloud Run |
| Celery workers | GKE (Kubernetes) |
| Redis | Memorystore for Redis |
| PostgreSQL | Cloud SQL |
| File storage | Cloud Storage (GCS) |
| Load balancer | Cloud Load Balancing |

### Option C — Self-Hosted (VPS)

Use Docker Compose on a single VPS (DigitalOcean, Hetzner, etc.) with:
- Nginx reverse proxy
- Let's Encrypt SSL (via Certbot)
- Docker Compose for orchestration
- Watchtower for automatic container updates

---

## 13 — Phase 9 — Performance & Scalability

> **Goal**: Handle concurrent users and large datasets.

### Step 9.1 — Gunicorn Tuning

```bash
# Production formula: workers = 2 × CPU_CORES + 1
gunicorn app:app \
    --workers $(( 2 * $(nproc) + 1 )) \
    --timeout 120 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 100
```

### Step 9.2 — Celery Worker Scaling

```bash
# CPU-bound ML tasks: 1 worker per core
celery -A celery_app.celery worker \
    --concurrency=1 \
    --pool=prefork \
    --max-tasks-per-child=10 \
    --autoscale=4,1  # Max 4 workers, min 1
```

### Step 9.3 — Upload Optimization

For large files (>50 MB):
1. Stream uploads to disk (don't buffer in memory)
2. Use `chunked_transfer_encoding` in Nginx
3. Consider presigned S3 URLs for direct client→S3 upload

### Step 9.4 — Result Caching

```python
from functools import lru_cache

@lru_cache(maxsize=32)
def get_data_profile(data_hash):
    """Cache data profiling results by data hash."""
    return profiler.profile(pipeline.raw_data)
```

### Step 9.5 — Database Indexing

```sql
-- Speed up experiment history queries
CREATE INDEX idx_runs_timestamp ON runs (timestamp DESC);
CREATE INDEX idx_runs_task_type ON runs (task_type);
CREATE INDEX idx_runs_best_model ON runs (best_model);
```

---

## 14 — Phase 10 — Testing Strategy

> **Goal**: Confidence in every deployment.

### Test Pyramid

```
          ╱╲
         ╱  ╲          E2E Tests (Playwright)
        ╱    ╲         • Full pipeline flow in browser
       ╱──────╲        • 5-10 tests
      ╱        ╲
     ╱          ╲      Integration Tests (pytest + httpx)
    ╱            ╲     • API endpoints with real DB/Redis
   ╱──────────────╲    • 20-30 tests
  ╱                ╲
 ╱                  ╲   Unit Tests (pytest)
╱                    ╲  • Pipeline modules in isolation
╱────────────────────╲  • 50-100 tests
```

### Unit Tests

```python
# tests/test_pipeline.py
import pytest
import pandas as pd
from pipeline.feature_engine import FeatureEngineer
from pipeline.model_trainer import ModelTrainer

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'target': [0, 1, 0, 1, 0],
    })

def test_feature_engineer_polynomial(sample_df):
    fe = FeatureEngineer()
    result = fe.create_polynomial_features(sample_df, degree=2)
    assert len(result.columns) > len(sample_df.columns)

def test_model_trainer_classification(sample_df):
    trainer = ModelTrainer()
    X = sample_df[['feature1', 'feature2']]
    y = sample_df['target']
    result = trainer.train('logistic_regression', X, y, task='classification')
    assert 'model' in result
    assert result['train_time'] > 0
```

### Integration Tests

```python
# tests/test_api.py
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health(client):
    resp = client.get('/health')
    assert resp.status_code == 200

def test_upload_csv(client, tmp_path):
    csv = tmp_path / "test.csv"
    csv.write_text("a,b,target\n1,2,0\n3,4,1\n")
    with open(csv, 'rb') as f:
        resp = client.post('/api/upload', data={'file': f})
    assert resp.status_code == 200
```

### Load Tests

```python
# tests/locustfile.py
from locust import HttpUser, task, between

class EasyMLUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def view_dashboard(self):
        self.client.get("/")

    @task(1)
    def check_status(self):
        self.client.get("/api/pipeline/status")
```

---

## 15 — Environment Variable Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FLASK_ENV` | No | `production` | `development` or `production` |
| `SECRET_KEY` | Yes (prod) | `dev-secret-change-me` | Flask session signing key |
| `DATABASE_URL` | Yes | SQLite fallback | PostgreSQL connection string |
| `POSTGRES_USER` | Yes | `easyml` | PostgreSQL username |
| `POSTGRES_PASSWORD` | Yes | `easyml` | PostgreSQL password |
| `POSTGRES_DB` | Yes | `easyml` | PostgreSQL database name |
| `POSTGRES_PORT` | No | `5432` | PostgreSQL port |
| `CELERY_BROKER_URL` | Yes | `redis://localhost:6379/0` | Redis broker URL |
| `CELERY_RESULT_BACKEND` | Yes | `redis://localhost:6379/1` | Redis result backend URL |
| `REDIS_PORT` | No | `6379` | Redis port |
| `GOOGLE_API_KEY` | No | *(empty)* | Gemini API key for chatbot |
| `EASYML_API_KEY` | No | *(empty)* | API authentication key |
| `SENTRY_DSN` | No | *(empty)* | Sentry error tracking DSN |
| `CORS_ORIGINS` | No | `*` | Allowed CORS origins |
| `GUNICORN_WORKERS` | No | `4` | Number of Gunicorn workers |
| `GUNICORN_TIMEOUT` | No | `120` | Gunicorn request timeout (seconds) |
| `WEB_PORT` | No | `5000` | Web server port |
| `MAX_CONTENT_LENGTH` | No | `104857600` | Max upload size in bytes (100 MB) |

---

## 16 — Risk Register & Mitigations

| # | Risk | Impact | Probability | Mitigation |
|---|------|--------|-------------|------------|
| 1 | **Pickle deserialization attack** via Celery | 🔴 Critical | Low | Restrict Redis to private network; consider JSON serialization for non-ML tasks |
| 2 | **Memory exhaustion** during large model training | 🟠 High | Medium | `worker_max_tasks_per_child=10`; set container memory limits in Docker |
| 3 | **Data loss** if Redis crashes with unacked tasks | 🟠 High | Low | `task_acks_late=True`; Redis AOF persistence; consider RabbitMQ for guaranteed delivery |
| 4 | **Stale model files** on shared volume | 🟡 Medium | Medium | Add TTL-based cleanup cron; timestamp model filenames |
| 5 | **PostgreSQL connection exhaustion** | 🟡 Medium | Low | `ThreadedConnectionPool(2, 10)`; monitor with `pg_stat_activity` |
| 6 | **Gemini API rate limits** blocking chatbot | 🟢 Low | Medium | Cache recent responses; implement exponential backoff |
| 7 | **Frontend breaking** during API changes | 🟡 Medium | Medium | API versioning (`/api/v1/`); OpenAPI spec |

---

## 17 — Timeline Estimate

| Phase | Effort | Dependencies | Priority |
|-------|--------|-------------|----------|
| **Phase 1**: Foundation (Docker + Config) | 2–3 days | None | 🔴 P0 |
| **Phase 2**: Database Migration | 2–3 days | Phase 1 | 🔴 P0 |
| **Phase 3**: Celery Integration | 3–5 days | Phases 1 & 2 | 🔴 P0 |
| **Phase 4**: API Hardening | 3–4 days | Phase 1 | 🟠 P1 |
| **Phase 5**: Frontend Decoupling | 1–2 days | Phase 1 | 🟡 P2 |
| **Phase 6**: Observability | 2–3 days | Phase 1 | 🟠 P1 |
| **Phase 7**: CI/CD | 1–2 days | Phase 10 | 🟠 P1 |
| **Phase 8**: Cloud Deployment | 3–5 days | All above | 🟡 P2 |
| **Phase 9**: Performance | 2–3 days | Phases 1–3 | 🟡 P2 |
| **Phase 10**: Testing | 3–5 days | Phases 1–3 | 🟠 P1 |
| | | | |
| **Total** | **~25–35 days** | | |

### Recommended Execution Order

```
Week 1:  Phase 1 (Foundation) → Phase 2 (Database)
Week 2:  Phase 3 (Celery) → Phase 4 (API Hardening)
Week 3:  Phase 10 (Testing) → Phase 6 (Observability)
Week 4:  Phase 7 (CI/CD) → Phase 5 (Frontend)
Week 5:  Phase 9 (Performance) → Phase 8 (Cloud Deployment)
```

---

> **This document is a living reference. Update it as decisions are made and phases are completed.**
