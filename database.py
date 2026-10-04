"""
Experiment Tracker — Dual-Backend Database Layer
---------------------------------------------------
Supports both SQLite (local dev) and PostgreSQL (production).
Auto-detects based on DATABASE_URL environment variable.

Local dev:  DATABASE_URL not set → SQLite fallback
Docker:     DATABASE_URL=postgresql://... → PostgreSQL with connection pool
"""

import json
import os
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager


# ── Detect backend from environment ──────────────────────────
DATABASE_URL = os.environ.get('DATABASE_URL', '')
_USE_POSTGRES = DATABASE_URL.startswith('postgresql')

# ── SQLite fallback path ─────────────────────────────────────
_SQLITE_PATH = os.path.join(
    os.environ.get('APP_BASE_DIR', os.path.dirname(os.path.abspath(__file__))),
    'outputs', 'experiment_history.db'
)

# ── PostgreSQL connection pool (lazy init) ───────────────────
_pg_pool = None


def _get_pg_pool():
    """Lazy-initialize the PostgreSQL threaded connection pool."""
    global _pg_pool
    if _pg_pool is None:
        import psycopg2
        from psycopg2.pool import ThreadedConnectionPool
        _pg_pool = ThreadedConnectionPool(2, 10, DATABASE_URL)
    return _pg_pool


@contextmanager
def _pg_conn():
    """Context manager for a pooled PostgreSQL connection."""
    pool = _get_pg_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def _sqlite_conn():
    """Get a SQLite connection with Row factory."""
    import sqlite3
    os.makedirs(os.path.dirname(_SQLITE_PATH), exist_ok=True)
    conn = sqlite3.connect(_SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ═════════════════════════════════════════════════════════════
#  Public API
# ═════════════════════════════════════════════════════════════

def init_db():
    """Create the runs table if it doesn't exist."""
    if _USE_POSTGRES:
        _init_db_pg()
    else:
        _init_db_sqlite()


def save_run(config: dict, metrics: dict, best_model_name: str,
             task_type: str, n_rows: int = 0, n_features: int = 0,
             n_models: int = 0, duration: float = 0.0,
             dataset_name: str = 'uploaded') -> str:
    """
    Save a completed pipeline run to the database.
    Returns the run_id.
    """
    init_db()
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(timezone.utc).isoformat()

    config_json = json.dumps(_safe_serialize(config))
    metrics_json = json.dumps(_safe_serialize(metrics))
    auto_explore = 1 if config.get('auto_explore') else 0

    if _USE_POSTGRES:
        _save_run_pg(
            run_id, timestamp, dataset_name, task_type,
            config.get('target_col', ''), best_model_name,
            config_json, metrics_json,
            n_rows, n_features, n_models,
            round(duration, 2), auto_explore
        )
    else:
        _save_run_sqlite(
            run_id, timestamp, dataset_name, task_type,
            config.get('target_col', ''), best_model_name,
            config_json, metrics_json,
            n_rows, n_features, n_models,
            round(duration, 2), auto_explore
        )
    return run_id


def get_runs(limit: int = 50) -> list:
    """Get all past runs, most recent first."""
    init_db()
    if _USE_POSTGRES:
        return _get_runs_pg(limit)
    return _get_runs_sqlite(limit)


def delete_run(run_id: str) -> bool:
    """Delete a run by ID. Returns True if found and deleted."""
    init_db()
    if _USE_POSTGRES:
        return _delete_run_pg(run_id)
    return _delete_run_sqlite(run_id)


def clear_history() -> int:
    """Delete all runs. Returns count of deleted rows."""
    init_db()
    if _USE_POSTGRES:
        return _clear_history_pg()
    return _clear_history_sqlite()


# ═════════════════════════════════════════════════════════════
#  PostgreSQL Implementation
# ═════════════════════════════════════════════════════════════

_CREATE_TABLE_PG = '''
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
'''


def _init_db_pg():
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLE_PG)


def _save_run_pg(run_id, timestamp, dataset_name, task_type,
                 target_col, best_model_name, config_json, metrics_json,
                 n_rows, n_features, n_models, duration, auto_explore):
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute('''
                INSERT INTO runs
                    (id, timestamp, dataset_name, task_type, target_col,
                     best_model, config_json, metrics_json,
                     n_rows, n_features, n_models_trained,
                     duration_seconds, auto_explore)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                run_id, timestamp, dataset_name, task_type, target_col,
                best_model_name, config_json, metrics_json,
                n_rows, n_features, n_models, duration, bool(auto_explore)
            ))


def _get_runs_pg(limit):
    from psycopg2.extras import RealDictCursor
    with _pg_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                'SELECT * FROM runs ORDER BY timestamp DESC LIMIT %s', (limit,)
            )
            rows = cur.fetchall()

    runs = []
    for row in rows:
        run = dict(row)
        # Parse JSONB fields
        config_raw = run.pop('config_json', '{}')
        metrics_raw = run.pop('metrics_json', '{}')
        run['config'] = config_raw if isinstance(config_raw, dict) else _safe_json_loads(config_raw)
        run['metrics'] = metrics_raw if isinstance(metrics_raw, dict) else _safe_json_loads(metrics_raw)
        # Convert timestamp to ISO string for JSON serialization
        if hasattr(run.get('timestamp'), 'isoformat'):
            run['timestamp'] = run['timestamp'].isoformat()
        runs.append(run)
    return runs


def _delete_run_pg(run_id):
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute('DELETE FROM runs WHERE id = %s', (run_id,))
            return cur.rowcount > 0


def _clear_history_pg():
    with _pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute('DELETE FROM runs')
            return cur.rowcount


# ═════════════════════════════════════════════════════════════
#  SQLite Implementation (local dev fallback)
# ═════════════════════════════════════════════════════════════

_CREATE_TABLE_SQLITE = '''
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        dataset_name TEXT,
        task_type TEXT,
        target_col TEXT,
        best_model TEXT,
        config_json TEXT,
        metrics_json TEXT,
        n_rows INTEGER,
        n_features INTEGER,
        n_models_trained INTEGER,
        duration_seconds REAL,
        auto_explore INTEGER DEFAULT 0
    )
'''


def _init_db_sqlite():
    conn = _sqlite_conn()
    conn.execute(_CREATE_TABLE_SQLITE)
    conn.commit()
    conn.close()


def _save_run_sqlite(run_id, timestamp, dataset_name, task_type,
                     target_col, best_model_name, config_json, metrics_json,
                     n_rows, n_features, n_models, duration, auto_explore):
    conn = _sqlite_conn()
    conn.execute('''
        INSERT INTO runs
            (id, timestamp, dataset_name, task_type, target_col,
             best_model, config_json, metrics_json,
             n_rows, n_features, n_models_trained,
             duration_seconds, auto_explore)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        run_id, timestamp, dataset_name, task_type, target_col,
        best_model_name, config_json, metrics_json,
        n_rows, n_features, n_models, duration, auto_explore
    ))
    conn.commit()
    conn.close()


def _get_runs_sqlite(limit):
    conn = _sqlite_conn()
    cursor = conn.execute(
        'SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?', (limit,)
    )
    runs = []
    for row in cursor.fetchall():
        run = dict(row)
        run['config'] = _safe_json_loads(run.pop('config_json', '{}'))
        run['metrics'] = _safe_json_loads(run.pop('metrics_json', '{}'))
        runs.append(run)
    conn.close()
    return runs


def _delete_run_sqlite(run_id):
    conn = _sqlite_conn()
    cursor = conn.execute('DELETE FROM runs WHERE id = ?', (run_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def _clear_history_sqlite():
    conn = _sqlite_conn()
    cursor = conn.execute('DELETE FROM runs')
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count


# ═════════════════════════════════════════════════════════════
#  Utilities
# ═════════════════════════════════════════════════════════════

def _safe_json_loads(val):
    """Safely parse a JSON string, returning {} on failure."""
    if isinstance(val, dict):
        return val
    try:
        return json.loads(val or '{}')
    except (json.JSONDecodeError, TypeError):
        return {}


def _safe_serialize(obj):
    """Make an object JSON-safe by converting numpy types etc."""
    import numpy as np
    import pandas as pd

    if isinstance(obj, dict):
        return {str(k): _safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, 'predict'):
        return f'<{type(obj).__name__}>'
    return obj
