# ============================================================
# EasyML — Celery Application Factory
# ============================================================
# Creates and configures the Celery instance.
# Used by:
#   - tasks.py          (task definitions)
#   - docker-compose    (celery -A celery_app.celery worker ...)
#   - Flower            (celery -A celery_app.celery flower ...)
# ============================================================

import os

from celery import Celery

# ── Broker & Backend URLs ────────────────────────────────────
CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/1')

# ── Create Celery instance ───────────────────────────────────
celery = Celery(
    'easyml',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['tasks'],  # auto-discover task modules
)

# ── Celery configuration ────────────────────────────────────
celery.conf.update(
    # Serialization — pickle allows passing DataFrames / numpy arrays
    task_serializer='pickle',
    result_serializer='pickle',
    accept_content=['pickle', 'json'],

    # Timeouts
    task_time_limit=3600,           # hard kill after 60 min
    task_soft_time_limit=3300,      # SoftTimeLimitExceeded after 55 min

    # Result expiry
    result_expires=86400,           # results expire after 24 hours

    # Worker tuning
    worker_prefetch_multiplier=1,   # fetch one task at a time (fair scheduling)
    task_acks_late=True,            # ack AFTER execution (crash-safe)
    task_reject_on_worker_lost=True,  # re-queue if worker dies mid-task
    worker_max_tasks_per_child=10,  # restart worker process every 10 tasks (leak guard)
)
