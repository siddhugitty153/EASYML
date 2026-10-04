# ============================================================
# EasyML — Celery Task Definitions
# ============================================================
# Async task wrappers for the ML pipeline.
# Replaces the old `threading.Thread` approach with durable,
# retryable, horizontally-scalable Celery tasks.
# ============================================================

import traceback

from celery_app import celery


@celery.task(bind=True, name='easyml.run_pipeline')
def run_pipeline_task(self, dataframe_records, config):
    """
    Full end-to-end ML pipeline execution.

    Args:
        dataframe_records: DataFrame converted via df.to_dict('records')
        config: dict with keys like target_col, task, models, etc.

    Returns:
        dict with pipeline results (metrics, model paths, etc.)
    """
    import pandas as pd
    from pipeline.ml_pipeline import EasyMLPipeline
    import database as experiment_db

    output_dir = config.pop('_output_dir', 'outputs')

    try:
        # ── Step 1: Reconstruct DataFrame ────────────────────
        self.update_state(state='PROGRESS', meta={
            'progress': 5,
            'step': 'Initializing pipeline...',
        })

        df = pd.DataFrame(dataframe_records)

        # ── Step 2: Initialize pipeline ──────────────────────
        self.update_state(state='PROGRESS', meta={
            'progress': 10,
            'step': 'Loading data and configuring pipeline...',
        })

        pipeline = EasyMLPipeline(output_dir=output_dir)
        pipeline.load_data(dataframe=df)

        target_col = config.get('target_col')
        task_type = config.get('task')

        # ── Step 3: Run pipeline ─────────────────────────────
        self.update_state(state='PROGRESS', meta={
            'progress': 20,
            'step': 'Running pipeline...',
        })

        if config.get('auto_explore'):
            report = pipeline.run_auto_explore(
                target_col=target_col, task=task_type, config=config
            )
        else:
            report = pipeline.run(
                target_col=target_col, task=task_type, config=config
            )

        # ── Step 4: Save experiment ──────────────────────────
        self.update_state(state='PROGRESS', meta={
            'progress': 95,
            'step': 'Saving experiment results...',
        })

        try:
            metrics = {}
            if pipeline.evaluation_results and pipeline.best_model_name:
                metrics = pipeline.evaluation_results.get(
                    pipeline.best_model_name, {}
                )
            experiment_db.save_run(
                config=config,
                metrics=metrics,
                best_model_name=pipeline.best_model_name or '',
                task_type=pipeline.pipeline_report.get('task', task_type or ''),
                n_rows=len(df),
                n_features=(
                    len(pipeline.X_train.columns)
                    if pipeline.X_train is not None else 0
                ),
                n_models=(
                    len(pipeline.evaluation_results)
                    if pipeline.evaluation_results else 0
                ),
            )
        except Exception as db_err:
            # Don't fail the whole pipeline if DB logging fails
            print(f"Warning: Failed to save experiment to DB: {db_err}")

        # ── Done ─────────────────────────────────────────────
        self.update_state(state='PROGRESS', meta={
            'progress': 100,
            'step': 'Complete!',
        })

        # Serialize the report for Celery result backend
        from app import make_serializable
        return {
            'status': 'success',
            'report': make_serializable(report),
            'best_model': pipeline.best_model_name,
            'task_id': self.request.id,
        }

    except Exception as e:
        error_info = {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'task_id': self.request.id,
        }
        # Re-raise so Celery marks the task as FAILURE
        raise
