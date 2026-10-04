# ============================================================
# EasyML — Centralized Configuration (12-Factor App)
# ============================================================
# All settings read from environment variables with safe defaults.
# Usage:
#   from config import get_config
#   cfg = get_config()
# ============================================================

import os
try:
    from dotenv import load_dotenv
    # Load .env file from the directory containing config.py
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    load_dotenv(_env_path)
except ImportError:
    pass

class Config:
    """Base configuration — shared across all environments."""

    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-change-me')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))  # 100 MB

    # ── Paths (container-friendly — no __file__ gymnastics) ────
    BASE_DIR = os.environ.get(
        'APP_BASE_DIR',
        os.path.dirname(os.path.abspath(__file__))
    )
    UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

    # ── Database ──────────────────────────────────────────────
    DATABASE_URL = os.environ.get(
        'DATABASE_URL',
        f'sqlite:///{os.path.join(BASE_DIR, "outputs", "experiment_history.db")}'
    )

    # ── Celery ────────────────────────────────────────────────
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')

    # ── Gemini AI ─────────────────────────────────────────────
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')

    # ── CORS ──────────────────────────────────────────────────
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')


class DevelopmentConfig(Config):
    """Local development — debug mode ON, SQLite fallback."""
    DEBUG = True
    FLASK_ENV = 'development'


class ProductionConfig(Config):
    """Production — debug OFF, expects PostgreSQL + Redis."""
    DEBUG = False
    FLASK_ENV = 'production'


class TestingConfig(Config):
    """Test suite — in-memory SQLite, no external services."""
    TESTING = True
    DEBUG = True
    DATABASE_URL = 'sqlite:///:memory:'
    CELERY_BROKER_URL = 'memory://'
    CELERY_RESULT_BACKEND = 'cache+memory://'


# ── Config map (keyed by FLASK_ENV) ──────────────────────────
_config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
}


def get_config():
    """Return the config class matching the current FLASK_ENV."""
    env = os.environ.get('FLASK_ENV', 'development')
    return _config_map.get(env, DevelopmentConfig)
