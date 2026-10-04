"""
EasyML Web App — Flask API Server
------------------------------------
Serves the web dashboard and API endpoints for the no-code EasyML pipeline.
Includes: chatbot, multi-CSV upload, merge, explain, and auto-explore.
"""

import os
import sys
import json
import re
import time
import logging
import traceback
import threading

from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import get_config
from pipeline.ml_pipeline import EasyMLPipeline
from pipeline.code_generator import generate_pipeline_code
from pipeline.data_profiler import DataProfiler
from pipeline.explainer import PipelineExplainer, HAS_GEMINI
from pipeline.ai_planner import PipelinePlanner
from pipeline.agent import AgentLoop
from middleware.error_handler import register_error_handlers
import database as experiment_db

# ── Celery (optional — falls back to threading if Redis unavailable) ──
_CELERY_AVAILABLE = False
try:
    import redis as _redis_lib
    _r = _redis_lib.from_url(os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'))
    _r.ping()
    from tasks import run_pipeline_task
    from celery.result import AsyncResult
    from celery_app import celery as celery_app_instance
    _CELERY_AVAILABLE = True
except Exception:
    pass  # Redis not running — will use threading fallback

# ── Structured Logging ──
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('easyml')

# ── App Configuration ──
cfg = get_config()

app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = cfg.MAX_CONTENT_LENGTH

# Register centralized error handlers (JSON responses for all HTTP errors)
register_error_handlers(app)

# ── Global State ──
UPLOAD_DIR = cfg.UPLOAD_DIR
OUTPUT_DIR = cfg.OUTPUT_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

pipeline = EasyMLPipeline(output_dir=OUTPUT_DIR)
profiler = DataProfiler()
explainer = PipelineExplainer()
ai_planner = PipelinePlanner()
training_thread = None
training_status = {'running': False, 'progress': 0, 'message': 'idle'}
active_task_id = None  # Celery task ID when using async dispatch
last_pipeline_config = {}  # Store config for code generation

# Agent state — replaced on every new agent run
_agent_instance = None
_agent_thread = None

# Initialize experiment tracking DB
experiment_db.init_db()
logger.info(f"EasyML initialized | uploads={UPLOAD_DIR} | outputs={OUTPUT_DIR} | celery={'ON' if _CELERY_AVAILABLE else 'OFF (thread fallback)'}")


def make_serializable(obj):
    """Recursively make objects JSON-serializable."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, 'predict'):  # sklearn model
        return f'<{type(obj).__name__}>'
    return obj


# ── Routes ──

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# ═══════════════════════════════════════════════════
# Health & Readiness Probes
# ═══════════════════════════════════════════════════

@app.route('/health')
def health():
    """Liveness probe — is the process alive?"""
    return jsonify({'status': 'ok', 'timestamp': time.time()}), 200


@app.route('/ready')
def ready():
    """Readiness probe — are dependencies reachable?"""
    checks = {}

    # Database check
    try:
        from database import get_runs
        get_runs(limit=1)
        checks['database'] = 'ok'
    except Exception as e:
        checks['database'] = str(e)

    # Redis check (only if configured for production)
    redis_url = os.environ.get('CELERY_BROKER_URL', '')
    if redis_url.startswith('redis://'):
        try:
            import redis
            r = redis.from_url(redis_url)
            r.ping()
            checks['redis'] = 'ok'
        except Exception as e:
            checks['redis'] = str(e)

    all_ok = all(v == 'ok' for v in checks.values())
    return jsonify({
        'status': 'ready' if all_ok else 'degraded',
        'checks': checks,
    }), 200 if all_ok else 503


# ═══════════════════════════════════════════════════
# Data Upload & Merge
# ═══════════════════════════════════════════════════

@app.route('/api/upload', methods=['POST'])
def upload_data():
    """Upload one or more CSV/Excel files."""
    global pipeline

    files = request.files.getlist('file')
    if not files or (len(files) == 1 and files[0].filename == ''):
        # Fallback: check single file key
        if 'file' in request.files:
            files = [request.files['file']]

    if not files or files[0].filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Reset pipeline
    pipeline = EasyMLPipeline(output_dir=OUTPUT_DIR)

    saved_paths = []
    for i, file in enumerate(files):
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ('.csv', '.xlsx', '.xls'):
            return jsonify({'error': f'Unsupported format: {file.filename}. Use CSV or Excel.'}), 400
        safe_name = f'uploaded_{i}{ext}'
        filepath = os.path.join(UPLOAD_DIR, safe_name)
        file.save(filepath)
        saved_paths.append(filepath)

    try:
        if len(saved_paths) == 1:
            preview = pipeline.load_data(saved_paths[0])
            return jsonify(make_serializable({**preview, 'multi': False}))
        else:
            # Multiple files — load all but don't merge yet
            file_previews = pipeline.load_multiple_files(saved_paths)
            return jsonify(make_serializable({
                'multi': True,
                'file_count': len(saved_paths),
                'files': file_previews,
            }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/merge', methods=['POST'])
def merge_data():
    """Merge loaded multi-CSV files."""
    config = request.json or {}
    method = config.get('method', 'concat')
    on = config.get('on')
    how = config.get('how', 'inner')

    try:
        preview = pipeline.merge_data(method=method, on=on, how=how)
        return jsonify(make_serializable(preview))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/configure', methods=['POST'])
def configure_data():
    """Configure multi-file data with role assignments and column exclusions."""
    config = request.json or {}
    file_roles = config.get('files', [])
    merge_config = config.get('merge_config', {})
    exclude_columns = config.get('exclude_columns', [])

    if not file_roles:
        return jsonify({'error': 'No file roles provided'}), 400

    try:
        preview = pipeline.configure_data(
            file_roles=file_roles,
            merge_config=merge_config,
            exclude_columns=exclude_columns
        )
        return jsonify(make_serializable(preview))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data/import_url', methods=['POST'])
def import_url():
    """Import dataset from a public URL."""
    global pipeline
    url = request.json.get('url') if request.json else None
    if not url:
        return jsonify({'error': 'URL is required'}), 400
        
    try:
        import requests
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        
        ext = '.csv'
        if '.xlsx' in url:
            ext = '.xlsx'
            
        filename = f"imported_url{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        with open(filepath, 'wb') as f:
            f.write(resp.content)
            
        pipeline = EasyMLPipeline(output_dir=OUTPUT_DIR)
        preview = pipeline.load_data(filepath)
        return jsonify(make_serializable({**preview, 'multi': False}))
    except Exception as e:
        return jsonify({'error': f'Failed to fetch URL: {str(e)}'}), 500


@app.route('/api/load-sample', methods=['POST'])
def load_sample_data():
    """Load a built-in sample dataset for testing."""
    global pipeline

    dataset_name = request.json.get('dataset', 'iris') if request.json else 'iris'
    pipeline = EasyMLPipeline(output_dir=OUTPUT_DIR)

    try:
        if dataset_name == 'iris':
            from sklearn.datasets import load_iris
            data = load_iris(as_frame=True)
            df = data.frame
            df.columns = ['sepal_length', 'sepal_width', 'petal_length',
                          'petal_width', 'species']
        elif dataset_name == 'wine':
            from sklearn.datasets import load_wine
            data = load_wine(as_frame=True)
            df = data.frame
        elif dataset_name == 'diabetes':
            from sklearn.datasets import load_diabetes
            data = load_diabetes(as_frame=True)
            df = data.frame
        elif dataset_name == 'breast_cancer':
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer(as_frame=True)
            df = data.frame
        else:
            return jsonify({'error': f'Unknown dataset: {dataset_name}'}), 400

        preview = pipeline.load_data(dataframe=df)
        return jsonify(make_serializable(preview))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════
# Data Inspection
# ═══════════════════════════════════════════════════

@app.route('/api/data/preview')
def data_preview():
    """Get preview of loaded data."""
    if pipeline.raw_data is None:
        return jsonify({'error': 'No data loaded'}), 404
    preview = pipeline.get_data_preview()
    return jsonify(make_serializable(preview))


@app.route('/api/data/columns')
def get_columns():
    """Get column names and types."""
    if pipeline.raw_data is None:
        return jsonify({'error': 'No data loaded'}), 404

    columns = []
    for col in pipeline.raw_data.columns:
        columns.append({
            'name': col,
            'dtype': str(pipeline.raw_data[col].dtype),
            'unique': int(pipeline.raw_data[col].nunique()),
            'is_numeric': pipeline.raw_data[col].dtype in ['int64', 'float64',
                                                              'int32', 'float32']
        })
    return jsonify({'columns': columns})


@app.route('/api/data/clean', methods=['POST'])
def clean_data():
    """Run data cleaning step."""
    if pipeline.raw_data is None:
        return jsonify({'error': 'No data loaded'}), 404

    config = request.json or {}
    try:
        pipeline.cleaned_data = pipeline.cleaner.clean(pipeline.raw_data, config)
        report = pipeline.cleaner.get_report()
        preview = {
            'rows_before': len(pipeline.raw_data),
            'rows_after': len(pipeline.cleaned_data),
            'report': report
        }
        return jsonify(make_serializable(preview))
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════
# Data Profiling (EDA)
# ═══════════════════════════════════════════════════

@app.route('/api/profile')
def data_profile():
    """Generate an EDA profile report for the loaded data."""
    if pipeline.raw_data is None:
        return jsonify({'error': 'No data loaded'}), 404

    try:
        report = profiler.profile(pipeline.raw_data)
        return jsonify(make_serializable(report))
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════
# Pipeline Execution
# ═══════════════════════════════════════════════════

@app.route('/api/pipeline/ai-plan', methods=['POST'])
def ai_plan_pipeline():
    """Generate an optimal pipeline configuration using AI."""
    if pipeline.raw_data is None:
        return jsonify({'error': 'No data loaded'}), 404

    config = request.json or {}
    target_col = config.get('target_col')
    task = config.get('task', 'classification')

    if not target_col:
        return jsonify({'error': 'target_col is required'}), 400

    try:
        profile = profiler.profile(pipeline.raw_data)
        plan_result = ai_planner.plan_pipeline(profile, target_col, task)
        
        if 'error' in plan_result and plan_result['error']:
            return jsonify({'error': plan_result['error']}), 500
            
        return jsonify(make_serializable(plan_result))
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/pipeline/run', methods=['POST'])
def run_pipeline():
    """Run the full pipeline (blocking)."""
    global training_status

    if pipeline.raw_data is None:
        return jsonify({'error': 'No data loaded'}), 404

    config = request.json or {}
    target_col = config.get('target_col')
    task = config.get('task')

    if not target_col:
        return jsonify({'error': 'target_col is required'}), 400

    if target_col not in pipeline.raw_data.columns:
        return jsonify({'error': f'Column "{target_col}" not found'}), 400

    training_status = {'running': True, 'progress': 0, 'message': 'Starting pipeline...'}

    try:
        last_pipeline_config.clear()
        last_pipeline_config.update(config)

        # Auto-explore mode
        if config.get('auto_explore'):
            report = pipeline.run_auto_explore(target_col=target_col, task=task, config=config)
        else:
            report = pipeline.run(target_col=target_col, task=task, config=config)
        training_status = {'running': False, 'progress': 100, 'message': 'Complete'}

        # Save experiment run
        try:
            metrics = {}
            if pipeline.evaluation_results and pipeline.best_model_name:
                metrics = pipeline.evaluation_results.get(pipeline.best_model_name, {})
            experiment_db.save_run(
                config=config,
                metrics=metrics,
                best_model_name=pipeline.best_model_name or '',
                task_type=pipeline.pipeline_report.get('task', task or ''),
                n_rows=len(pipeline.raw_data) if pipeline.raw_data is not None else 0,
                n_features=len(pipeline.X_train.columns) if pipeline.X_train is not None else 0,
                n_models=len(pipeline.evaluation_results) if pipeline.evaluation_results else 0,
            )
        except Exception:
            pass  # Don't fail the pipeline if tracking fails

        return jsonify(make_serializable(report))
    except Exception as e:
        error_traceback = traceback.format_exc()
        
        # Generate AI explanation for the failure
        data_preview_str = ""
        if pipeline.raw_data is not None:
            # Capture a small string representation of the data to give the AI context
            data_preview_str = str(pipeline.raw_data.head(3).to_dict(orient='records'))
            
        ai_explanation = ""
        if HAS_GEMINI and explainer:
            ai_explanation = explainer.explain_failure(
                error_traceback=error_traceback,
                config=config,
                data_preview=data_preview_str
            )

        training_status = {
            'running': False, 
            'progress': 0, 
            'message': f'Error: {str(e)}',
            'ai_explanation': ai_explanation
        }
        return jsonify({
            'error': str(e), 
            'traceback': error_traceback,
            'ai_explanation': ai_explanation
        }), 500


@app.route('/api/pipeline/run-async', methods=['POST'])
def run_pipeline_async():
    """Run the full pipeline asynchronously.
    
    Uses Celery when Redis is available (production/Docker),
    falls back to threading.Thread for local dev.
    """
    global training_thread, training_status, active_task_id

    if pipeline.raw_data is None:
        return jsonify({'error': 'No data loaded'}), 404

    config = request.json or {}
    target_col = config.get('target_col')
    task = config.get('task')

    if not target_col:
        return jsonify({'error': 'target_col is required'}), 400

    if training_status.get('running'):
        return jsonify({'error': 'Pipeline already running'}), 409

    last_pipeline_config.clear()
    last_pipeline_config.update(config)

    # ── Path A: Celery (production / Docker) ─────────────────
    if _CELERY_AVAILABLE:
        logger.info(f"Dispatching pipeline via Celery | target={target_col}")
        try:
            # Serialize DataFrame as records for Celery
            df_records = pipeline.raw_data.to_dict('records')
            task_config = {**config, '_output_dir': OUTPUT_DIR}

            result = run_pipeline_task.delay(df_records, task_config)
            active_task_id = result.id
            training_status = {
                'running': True, 'progress': 0,
                'message': 'Pipeline dispatched to Celery worker...',
                'task_id': result.id,
                'backend': 'celery',
            }
            return jsonify({
                'status': 'started',
                'message': 'Pipeline dispatched to Celery worker',
                'task_id': result.id,
                'backend': 'celery',
            })
        except Exception as e:
            logger.warning(f"Celery dispatch failed, falling back to thread: {e}")
            # Fall through to threading fallback

    # ── Path B: Threading fallback (local dev) ───────────────
    logger.info(f"Running pipeline via thread | target={target_col}")
    active_task_id = None

    def run_in_background():
        global training_status
        training_status = {'running': True, 'progress': 10, 'message': 'Cleaning data...', 'backend': 'thread'}
        try:
            if config.get('auto_explore'):
                pipeline.run_auto_explore(target_col=target_col, task=task, config=config)
            else:
                pipeline.run(target_col=target_col, task=task, config=config)
            training_status = {'running': False, 'progress': 100, 'message': 'Complete', 'backend': 'thread'}

            # Save experiment run
            try:
                metrics = {}
                if pipeline.evaluation_results and pipeline.best_model_name:
                    metrics = pipeline.evaluation_results.get(pipeline.best_model_name, {})
                experiment_db.save_run(
                    config=config,
                    metrics=metrics,
                    best_model_name=pipeline.best_model_name or '',
                    task_type=pipeline.pipeline_report.get('task', task or ''),
                    n_rows=len(pipeline.raw_data) if pipeline.raw_data is not None else 0,
                    n_features=len(pipeline.X_train.columns) if pipeline.X_train is not None else 0,
                    n_models=len(pipeline.evaluation_results) if pipeline.evaluation_results else 0,
                )
            except Exception:
                pass

        except Exception as e:
            error_traceback = traceback.format_exc()
            
            # Generate AI explanation for the failure
            data_preview_str = ""
            if pipeline.raw_data is not None:
                data_preview_str = str(pipeline.raw_data.head(3).to_dict(orient='records'))
                
            ai_explanation = ""
            if HAS_GEMINI and explainer:
                ai_explanation = explainer.explain_failure(
                    error_traceback=error_traceback,
                    config=config,
                    data_preview=data_preview_str
                )

            training_status = {
                'running': False, 
                'progress': 0, 
                'message': f'Error: {str(e)}',
                'ai_explanation': ai_explanation,
                'backend': 'thread',
            }

    training_thread = threading.Thread(target=run_in_background, daemon=True)
    training_thread.start()

    return jsonify({'status': 'started', 'message': 'Pipeline running in background', 'backend': 'thread'})


@app.route('/api/pipeline/status')
def pipeline_status():
    """Get current pipeline execution status.
    
    If a Celery task is active, queries AsyncResult for live progress.
    Otherwise returns the thread-based training_status.
    """
    global training_status, active_task_id

    # ── Celery task active? Query Redis for live status ───────
    if active_task_id and _CELERY_AVAILABLE:
        result = AsyncResult(active_task_id, app=celery_app_instance)
        state = result.state

        if state == 'PROGRESS':
            meta = result.info or {}
            status = {
                'running': True,
                'progress': meta.get('progress', 0),
                'message': meta.get('step', 'Processing...'),
                'task_id': active_task_id,
                'backend': 'celery',
            }
        elif state == 'SUCCESS':
            task_result = result.result or {}
            training_status = {
                'running': False, 'progress': 100,
                'message': 'Complete',
                'task_id': active_task_id,
                'backend': 'celery',
                'result': task_result,
            }
            active_task_id = None
            status = {**training_status}
        elif state == 'FAILURE':
            error_msg = str(result.info) if result.info else 'Unknown error'
            training_status = {
                'running': False, 'progress': 0,
                'message': f'Error: {error_msg}',
                'task_id': active_task_id,
                'backend': 'celery',
            }
            active_task_id = None
            status = {**training_status}
        elif state == 'REVOKED':
            training_status = {
                'running': False, 'progress': 0,
                'message': 'Task cancelled',
                'task_id': active_task_id,
                'backend': 'celery',
            }
            active_task_id = None
            status = {**training_status}
        else:
            # PENDING or other states
            status = {
                'running': True,
                'progress': 0,
                'message': f'Queued ({state})',
                'task_id': active_task_id,
                'backend': 'celery',
            }
    else:
        # ── Thread-based status ───────────────────────────────
        status = {**training_status}

    status['pipeline_report'] = make_serializable(pipeline.pipeline_report)
    return jsonify(status)


@app.route('/api/tasks/<task_id>/revoke', methods=['POST'])
def revoke_task(task_id):
    """Cancel a running Celery task."""
    global active_task_id, training_status

    if not _CELERY_AVAILABLE:
        return jsonify({'error': 'Celery not available'}), 503

    try:
        celery_app_instance.control.revoke(task_id, terminate=True, signal='SIGTERM')
        if active_task_id == task_id:
            active_task_id = None
            training_status = {'running': False, 'progress': 0, 'message': 'Task cancelled'}
        logger.info(f"Task {task_id} revoked")
        return jsonify({'status': 'revoked', 'task_id': task_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════
# Model Results & Export
# ═══════════════════════════════════════════════════

@app.route('/api/models/results')
def model_results():
    """Get model evaluation results."""
    if not pipeline.evaluation_results:
        return jsonify({'error': 'No models trained yet'}), 404

    return jsonify(make_serializable({
        'results': pipeline.evaluation_results,
        'best_model': pipeline.best_model_name,
        'report': pipeline.get_full_report()
    }))


@app.route('/api/models/compare')
def compare_models():
    """Get model comparison data for charts."""
    if not pipeline.evaluation_results:
        return jsonify({'error': 'No models trained yet'}), 404

    comparison = []
    for name, metrics in pipeline.evaluation_results.items():
        if 'error' in metrics:
            continue
        entry = {
            'model': name,
            'display_name': metrics.get('display_name', name),
            'rank': metrics.get('rank', 99),
            'train_time': metrics.get('train_time', 0)
        }
        for key in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                     'mse', 'rmse', 'mae', 'r2'):
            if key in metrics:
                entry[key] = metrics[key]
        comparison.append(entry)

    comparison.sort(key=lambda x: x.get('rank', 99))
    return jsonify(make_serializable({'models': comparison}))


@app.route('/api/export/model')
def export_model():
    """Download the trained model."""
    model_path = os.path.join(OUTPUT_DIR, 'best_model.joblib')
    if not os.path.exists(model_path):
        return jsonify({'error': 'No model saved yet'}), 404
    return send_from_directory(OUTPUT_DIR, 'best_model.joblib', as_attachment=True)


@app.route('/api/export/report')
def export_report():
    """Download the pipeline report as JSON."""
    try:
        report_path = pipeline.save_report()
        return send_from_directory(OUTPUT_DIR, 'pipeline_report.json', as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/visualizations/report')
def export_visualizations_report():
    """Download the generated HTML visualization report."""
    vis_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    report_path = os.path.join(vis_dir, 'report.html')
    if not os.path.exists(report_path):
        return jsonify({'error': 'Visualization report not generated yet.'}), 404
    return send_from_directory(vis_dir, 'report.html', as_attachment=True, download_name='easyml_visualization_report.html')


@app.route('/api/export/model/<fmt>')
def export_model_format(fmt):
    """Export model in different formats: joblib, pkl, or json."""
    import pickle

    if pipeline.best_model is None:
        return jsonify({'error': 'No model saved yet'}), 404

    try:
        if fmt == 'pkl':
            pkl_path = os.path.join(OUTPUT_DIR, 'best_model.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(pipeline.best_model, f)
            return send_from_directory(OUTPUT_DIR, 'best_model.pkl', as_attachment=True)
        elif fmt == 'joblib':
            return send_from_directory(OUTPUT_DIR, 'best_model.joblib', as_attachment=True)
        else:
            return jsonify({'error': f'Unsupported format: {fmt}. Use joblib or pkl.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/code')
def export_code():
    """Generate and download a Python script reproducing the pipeline."""
    try:
        report = pipeline.get_full_report()
        code = generate_pipeline_code(last_pipeline_config, report)
        # Save to file
        code_path = os.path.join(OUTPUT_DIR, 'pipeline_script.py')
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code)
        return send_from_directory(OUTPUT_DIR, 'pipeline_script.py',
                                   as_attachment=True, mimetype='text/x-python')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════
# What-If Predictor & SHAP Local
# ═══════════════════════════════════════════════════

@app.route('/api/predict/whatif', methods=['POST'])
def predict_whatif():
    """Interactive What-If prediction with optional SHAP local explanation."""
    if pipeline.best_model is None:
        return jsonify({'error': 'No model trained yet'}), 404

    data = request.json or {}
    feature_values = data.get('features', {})

    try:
        result = pipeline.predict_whatif(feature_values)
        return jsonify(make_serializable(result))
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


@app.route('/api/predict/whatif-info')
def whatif_info():
    """Get feature info for the What-If form (names, types, ranges)."""
    if pipeline.X_train is None:
        return jsonify({'error': 'No model trained yet'}), 404

    features = []
    for col in pipeline.X_train.columns:
        info = {
            'name': col,
            'min': round(float(pipeline.X_train[col].min()), 4),
            'max': round(float(pipeline.X_train[col].max()), 4),
            'mean': round(float(pipeline.X_train[col].mean()), 4),
            'median': round(float(pipeline.X_train[col].median()), 4),
        }
        features.append(info)

    return jsonify(make_serializable({
        'features': features,
        'task': pipeline.pipeline_report.get('task', 'classification'),
        'best_model': pipeline.best_model_name,
    }))


@app.route('/api/explain/pdp', methods=['POST'])
def get_pdp():
    """Get Partial Dependence Plot data for a feature."""
    if pipeline.best_model is None:
        return jsonify({'error': 'No model trained yet'}), 404

    data = request.json or {}
    feature_name = data.get('feature')
    if not feature_name:
        return jsonify({'error': 'feature name required'}), 400

    try:
        pdp_data = pipeline.get_pdp_data(feature_name)
        return jsonify(make_serializable(pdp_data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════
# Explainability
# ═══════════════════════════════════════════════════

@app.route('/api/explain')
def get_explanations():
    """Get AI-generated explanations for the pipeline run."""
    if not pipeline.explanations:
        return jsonify({'error': 'No explanations available. Run the pipeline first.'}), 404
    return jsonify(make_serializable(pipeline.explanations))


# ═══════════════════════════════════════════════════
# Chatbot
# ═══════════════════════════════════════════════════

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Rule-based chatbot that parses natural-language commands
    and returns both a reply and an optional action for the frontend.
    """
    data = request.json or {}
    message = data.get('message', '').strip().lower()

    if not message:
        return jsonify({'reply': "I didn't catch that. Try asking me something!"})

    reply, action, params = _parse_chat(message)
    response = {'reply': reply}
    if action:
        response['action'] = action
    if params:
        response['params'] = params
    return jsonify(make_serializable(response))


def _parse_chat(msg: str):
    """Parse a chat message into (reply, action, params) using Gemini AI."""
    
    # ── Fallback if Gemini is not available ──
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not HAS_GEMINI or not api_key:
        return _parse_chat_regex(msg)
        
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Use gemini-2.5-flash for fast reasoning
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Get current state context
        data_loaded = pipeline.raw_data is not None
        target_set = bool(pipeline.pipeline_report.get('target_col'))
        models_trained = bool(pipeline.best_model_name)
        
        system_prompt = f"""
You are the AI assistant for EasyML, a no-code machine learning platform.
The user is talking to you. You must map their intent to one of the following precise JSON structures.
Only output valid JSON. Do not include markdown formatting like ```json.

Current State:
- Data Loaded: {data_loaded}
- Target Set: {target_set}
- Models Trained: {models_trained}

Available actions:
1. Load sample data (iris, wine, diabetes, breast_cancer):
   {{"reply": "Friendly message loading the dataset", "action": "load_sample", "params": {{"dataset": "<name>"}}}}
2. Set target column:
   {{"reply": "Friendly message setting target", "action": "set_target", "params": {{"column": "<col_name>"}}}}
3. Run Auto-Explore:
   {{"reply": "Friendly message starting auto explore", "action": "run_auto_explore", "params": {{}}}}
4. Run standard pipeline:
   {{"reply": "Friendly message starting training", "action": "run_pipeline", "params": {{}}}}
5. Show results/best model:
   {{"reply": "Friendly message showing best model", "action": "show_results", "params": {{}}}}
6. Show explanations/insights:
   {{"reply": "Friendly message showing insights", "action": "show_explain", "params": {{}}}}
7. General conversational reply / Data questions / Help / Fallback (No action):
   {{"reply": "Your helpful response here", "action": null, "params": null}}
   
If the user asks about the data (rows, missing values, etc.), answer them directly in the reply if you can infer it, or just say you can't see the data.
"""
        response = model.generate_content([system_prompt, f"User message: {msg}"])
        result = json.loads(response.text.strip().removeprefix('```json').removesuffix('```').strip())
        
        reply = result.get('reply', "I'm not sure what you mean.")
        action = result.get('action')
        params = result.get('params', {})
        
        return reply, action, params
        
    except Exception as e:
        logger.error(f"Gemini chat parse failed: {e}")
        # Fallback to regex
        return _parse_chat_regex(msg)


def _parse_chat_regex(msg: str):
    """Fallback regex parser for chat."""
    # ── Load sample datasets ──
    sample_patterns = {
        'iris': r'\biris\b',
        'wine': r'\bwine\b',
        'diabetes': r'\bdiabetes\b',
        'breast_cancer': r'\bbreast.?cancer\b',
    }
    for dataset, pattern in sample_patterns.items():
        if re.search(pattern, msg) and any(w in msg for w in ('load', 'use', 'try', 'sample', 'open', 'dataset')):
            return (
                f"Loading the **{dataset}** dataset for you!",
                'load_sample',
                {'dataset': dataset}
            )

    # ── Set target column ──
    target_match = re.search(r'(?:set|change|use)\s+(?:the\s+)?target\s+(?:to\s+|column\s+)?["\']?(\w+)["\']?', msg)
    if target_match:
        col = target_match.group(1)
        return (
            f"Setting target column to `{col}`.",
            'set_target',
            {'column': col}
        )
    if 'target' in msg and ('what' in msg or 'which' in msg or 'current' in msg):
        target = pipeline.pipeline_report.get('target_col')
        if target:
            return (f"Current target column is `{target}`.", None, None)
        return ("No target column set yet. Upload data and configure the pipeline.", None, None)

    # ── Train / Run ──
    if any(w in msg for w in ('train', 'run', 'start', 'execute', 'go')):
        if 'auto' in msg or 'explore' in msg or 'best' in msg:
            return (
                "Starting **Auto-Explore** mode! I'll try multiple preprocessing "
                "strategies with cross-validation and pick the best one.",
                'run_auto_explore',
                {}
            )
        return (
            "Starting model training! This may take a moment...",
            'run_pipeline',
            {}
        )

    # ── Data info ──
    if any(w in msg for w in ('rows', 'shape', 'size', 'count', 'how many')):
        if pipeline.raw_data is not None:
            rows, cols = pipeline.raw_data.shape
            return (f"Your dataset has **{rows:,}** rows and **{cols}** columns.", None, None)
        return ("No data loaded yet. Upload a CSV or try a sample dataset.", None, None)

    if any(w in msg for w in ('columns', 'features', 'fields', 'variables')):
        if pipeline.raw_data is not None:
            cols = list(pipeline.raw_data.columns)
            cols_str = ', '.join(f'`{c}`' for c in cols[:20])
            extra = f' ...and {len(cols) - 20} more' if len(cols) > 20 else ''
            return (f"Columns: {cols_str}{extra}", None, None)
        return ("No data loaded yet.", None, None)

    if any(w in msg for w in ('missing', 'null', 'nan', 'empty')):
        if pipeline.raw_data is not None:
            total = int(pipeline.raw_data.isnull().sum().sum())
            if total == 0:
                return ("No missing values found!", None, None)
            per_col = pipeline.raw_data.isnull().sum()
            bad = per_col[per_col > 0]
            detail = ', '.join(f'`{c}`: {int(v)}' for c, v in bad.items())
            return (f"**{total}** missing values total. By column: {detail}", None, None)
        return ("No data loaded yet.", None, None)

    # ── Results / best model ──
    if any(w in msg for w in ('best', 'winner', 'result', 'model', 'score')):
        if pipeline.best_model_name:
            best = pipeline.evaluation_results.get(pipeline.best_model_name, {})
            display = best.get('display_name', pipeline.best_model_name)
            f1 = best.get('f1', best.get('r2', '?'))
            acc = best.get('accuracy', '?')
            return (
                f"**{display}** is the best model (F1/R²: {f1}, Accuracy: {acc})",
                'show_results',
                {}
            )
        return ("No models trained yet. Configure and run the pipeline first.", None, None)

    # ── Why questions ──
    if 'why' in msg:
        if pipeline.explanations:
            # Return best model explanation
            best_exp = pipeline.explanations.get('best_model', [])
            overfit = pipeline.explanations.get('overfitting_check', [])
            reply_parts = best_exp + overfit
            return ('\n\n'.join(reply_parts) if reply_parts else "No explanation available yet.", 'show_results', {})
        return ("Run the pipeline first, and I'll explain every decision!", None, None)

    # ── Explain ──
    if any(w in msg for w in ('explain', 'insight', 'analysis')):
        if pipeline.explanations:
            return ("Here are the AI insights from your pipeline run!", 'show_explain', {})
        return ("No insights yet. Run the pipeline first!", None, None)

    # ── Suggestions ──
    if any(w in msg for w in ('suggest', 'improve', 'better', 'tip', 'advice', 'recommend')):
        if pipeline.explanations:
            sugg = pipeline.explanations.get('suggestions', [])
            return ('\n\n'.join(sugg) if sugg else "No suggestions available.", None, None)
        return ("Run the pipeline first, and I'll suggest improvements!", None, None)

    # ── Help ──
    if any(w in msg for w in ('help', 'command', 'what can', 'how to', 'guide')):
        return (
            "Here's what I can do:\n\n"
            "• **\"Load iris\"** — Load a sample dataset\n"
            "• **\"Set target to species\"** — Set the target column\n"
            "• **\"Train models\"** — Run the pipeline\n"
            "• **\"Auto explore\"** — Try multiple strategies automatically\n"
            "• **\"Show columns\"** — List all columns\n"
            "• **\"How many rows?\"** — Get data shape\n"
            "• **\"Missing values?\"** — Check for nulls\n"
            "• **\"Best model?\"** — See the winning model\n"
            "• **\"Why did it win?\"** — Get explainability insights\n"
            "• **\"Suggestions\"** — Get improvement recommendations",
            None, None
        )

    # ── Greetings ──
    if any(w in msg for w in ('hello', 'hi', 'hey', 'sup', 'yo', 'greet')):
        return ("Hey there! I'm your EasyML assistant. Type **help** to see what I can do.", None, None)

    # ── Fallback ──
    return (
        "I'm not sure what you mean. Try **help** to see what I can do, "
        "or ask about your data, models, or pipeline!",
        None, None
    )


# ═══════════════════════════════════════════════════
# Experiment Tracking
# ═══════════════════════════════════════════════════

@app.route('/api/runs/history')
def run_history():
    """Get all past experiment runs."""
    try:
        runs = experiment_db.get_runs(limit=50)
        return jsonify(make_serializable({'runs': runs}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/runs/<run_id>', methods=['DELETE'])
def delete_run(run_id):
    """Delete a specific experiment run."""
    try:
        deleted = experiment_db.delete_run(run_id)
        if deleted:
            return jsonify({'status': 'deleted', 'id': run_id})
        return jsonify({'error': 'Run not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/runs/clear', methods=['DELETE'])
def clear_runs():
    """Clear all experiment history."""
    try:
        count = experiment_db.clear_history()
        return jsonify({'status': 'cleared', 'deleted_count': count})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════
# Autonomous Agent
# ═══════════════════════════════════════════════

@app.route('/api/agent/run', methods=['POST'])
def run_agent():
    """Launch the autonomous agent loop in a background thread.

    Expects JSON body:
      target_col  (str, required) -- column to predict
      task        (str, optional) -- 'classification', 'regression', or 'auto'
      goal        (dict, optional) -- {'metric': 'f1', 'target': 0.85}
      max_iterations (int, optional, default 4) -- max agent loops
    """
    global _agent_instance, _agent_thread, training_status

    if pipeline.raw_data is None:
        return jsonify({'error': 'No data loaded. Upload a dataset first.'}), 404

    # Prevent double-launch
    if _agent_instance and _agent_instance.state.status == 'running':
        return jsonify({'error': 'Agent already running. Wait for it to finish.'}), 409

    body = request.json or {}
    target_col = body.get('target_col')
    task = body.get('task', 'auto')
    goal = body.get('goal')  # e.g. {'metric': 'f1', 'target': 0.85}
    max_iterations = int(body.get('max_iterations', 4))

    if not target_col:
        return jsonify({'error': 'target_col is required'}), 400
    if target_col not in pipeline.raw_data.columns:
        return jsonify({'error': f'Column "{target_col}" not found'}), 400

    # Create a fresh agent bound to the current pipeline + profiler
    _agent_instance = AgentLoop(pipeline=pipeline, profiler=profiler)

    def _run_agent_bg():
        global training_status
        try:
            _agent_instance.run(
                target_col=target_col,
                task=task,
                goal=goal,
                max_iterations=max_iterations,
            )
        except Exception as e:
            _agent_instance.state.status = 'failed'
            _agent_instance.state.error = str(e)
            _agent_instance.state.push_log(f'FATAL: {e}')

    _agent_thread = threading.Thread(target=_run_agent_bg, daemon=True)
    _agent_thread.start()

    logger.info(f'Agent launched | target={target_col} | task={task} | max_iters={max_iterations}')
    return jsonify({
        'status': 'started',
        'target_col': target_col,
        'task': task,
        'max_iterations': max_iterations,
    })


@app.route('/api/agent/status')
def agent_status():
    """Poll the live state of the running agent."""
    global _agent_instance
    if _agent_instance is None:
        return jsonify({'status': 'idle', 'log': [], 'history': []})
    return jsonify(make_serializable(_agent_instance.state.to_dict()))


@app.route('/api/agent/stop', methods=['POST'])
def stop_agent():
    """Request a graceful stop (marks status; the running iteration will complete)."""
    global _agent_instance
    if _agent_instance is None or _agent_instance.state.status not in ('running',):
        return jsonify({'error': 'No running agent to stop'}), 400
    _agent_instance.state.status = 'stopped'
    _agent_instance.state.push_log('Stop requested by user.')
    return jsonify({'status': 'stop_requested'})


# ══ Main ══

if __name__ == '__main__':
    is_dev = os.environ.get('FLASK_ENV', 'development') == 'development'
    logger.info(f"Starting EasyML | env={'development' if is_dev else 'production'}")
    print("\n*  EasyML App running at http://localhost:5000\n")
    app.run(debug=is_dev, port=5000, host='0.0.0.0')
