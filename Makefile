# ============================================================
# EasyML — Makefile (Convenience Commands)
# ============================================================
# On Windows: install Make via `choco install make` or run
# the commands directly from each target.
# ============================================================

.PHONY: up down build logs test shell db-init worker-logs flower clean

## Start all services (build images if needed)
up:
	docker compose up -d --build

## Stop all services
down:
	docker compose down

## Rebuild images from scratch (no cache)
build:
	docker compose build --no-cache

## Tail logs from web + celery worker
logs:
	docker compose logs -f web celery_worker

## Run the test suite inside the web container
test:
	docker compose exec web pytest tests/ -v --tb=short

## Open a shell inside the web container
shell:
	docker compose exec web bash

## Initialize / migrate the database
db-init:
	docker compose exec web python scripts/init_db.py

## Tail celery worker logs only
worker-logs:
	docker compose logs -f celery_worker

## Launch Flower (Celery monitoring dashboard)
flower:
	docker compose exec celery_worker celery -A celery_app.celery flower --port=5555

## Remove volumes and all containers (DESTRUCTIVE)
clean:
	docker compose down -v --rmi local --remove-orphans
