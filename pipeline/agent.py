"""
EasyML Autonomous Agent Loop
------------------------------
A fully agentic pipeline that autonomously:
  1. PERCEIVE  -- profiles the dataset to understand its structure
  2. PLAN      -- uses Gemini to design an optimal pipeline config
  3. ACT       -- runs the full ML pipeline
  4. OBSERVE   -- evaluates results (primary metric, overfitting, etc.)
  5. REFLECT   -- uses Gemini to reason about failures and suggest fixes
  6. ITERATE   -- re-plans and re-runs if the goal metric is not met

The agent runs until either:
  - The target metric is achieved (goal reached)
  - Max iterations are exhausted
  - A non-recoverable error occurs
"""

import os
import json
import time
import copy
import logging
import traceback

logger = logging.getLogger(__name__)

HAS_GEMINI = False
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    pass


# -- Default goals by task -------------------------------------------------
DEFAULT_GOALS = {
    'classification': {'metric': 'f1', 'target': 0.80},
    'regression':     {'metric': 'r2', 'target': 0.75},
}

# -- Metric display names --------------------------------------------------
METRIC_LABELS = {
    'f1': 'F1 Score', 'accuracy': 'Accuracy', 'roc_auc': 'ROC-AUC',
    'r2': 'R2',       'rmse': 'RMSE',          'mae': 'MAE',
}


class AgentState:
    """
    Mutable state bag shared across all agent iterations.
    Serialisable to JSON for polling.
    """

    def __init__(self):
        self.status = 'idle'           # idle | running | success | finished | failed
        self.iteration = 0
        self.max_iterations = 5
        self.goal_metric = 'f1'
        self.goal_target = 0.80
        self.goal_direction = 'maximize'   # maximize | minimize (RMSE, MAE)

        self.current_phase = ''        # perceive | plan | act | observe | reflect
        self.best_score = None
        self.best_config = {}
        self.best_run_report = {}

        # Full trace of every iteration
        self.history = []

        # Human-readable log shown in the UI
        self.log = []

        # Final summary narrative (Gemini-generated)
        self.final_summary = ''

        self.error = ''

    # -- helpers -----------------------------------------------------------

    def push_log(self, msg):
        ts = time.strftime('%H:%M:%S')
        entry = '[{}] {}'.format(ts, msg)
        logger.info('[Agent] %s', msg)
        self.log.append(entry)

    def to_dict(self):
        return {
            'status':          self.status,
            'iteration':       self.iteration,
            'max_iterations':  self.max_iterations,
            'goal_metric':     self.goal_metric,
            'goal_target':     self.goal_target,
            'goal_direction':  self.goal_direction,
            'current_phase':   self.current_phase,
            'best_score':      self.best_score,
            'best_config':     self.best_config,
            'best_run_report': self.best_run_report,
            'history':         self.history,
            'log':             self.log,
            'final_summary':   self.final_summary,
            'error':           self.error,
        }


# -- Main Agent Class ------------------------------------------------------

class AgentLoop:
    """
    Autonomous ML agent.

    Usage::

        agent = AgentLoop(pipeline, profiler)
        agent.run(
            target_col='survived',
            task='classification',
            goal={'metric': 'f1', 'target': 0.85},
            max_iterations=4,
        )
    """

    def __init__(self, pipeline, profiler):
        self.pipeline = pipeline   # EasyMLPipeline instance
        self.profiler = profiler   # DataProfiler instance
        self.state = AgentState()

        self._gemini_json_model = None
        self._gemini_text_model = None
        api_key = os.environ.get('GOOGLE_API_KEY')
        if HAS_GEMINI and api_key:
            genai.configure(api_key=api_key)
            self._gemini_json_model = genai.GenerativeModel(
                'gemini-2.5-flash',
                generation_config={'response_mime_type': 'application/json'}
            )
            self._gemini_text_model = genai.GenerativeModel('gemini-2.5-flash')

    # =====================================================================
    # Public entry-point
    # =====================================================================

    def run(self, target_col, task=None, goal=None, max_iterations=4):
        """
        Execute the autonomous agent loop and return the final state dict.
        Designed to be run inside a background thread.
        """
        s = self.state
        s.status = 'running'
        s.max_iterations = max_iterations

        # Auto-detect task
        if task is None or task == 'auto':
            task = self.pipeline.detect_task_type(target_col)

        # Set goal
        default_goal = DEFAULT_GOALS.get(task, {'metric': 'f1', 'target': 0.80})
        goal = goal or default_goal
        s.goal_metric = goal.get('metric', default_goal['metric'])
        s.goal_target = float(goal.get('target', default_goal['target']))
        s.goal_direction = (
            'minimize' if s.goal_metric in ('rmse', 'mae', 'mse') else 'maximize'
        )

        s.push_log(
            'Agent started | task={} | goal={}>={}'.format(task, s.goal_metric, s.goal_target)
        )

        # PHASE 1: PERCEIVE
        s.current_phase = 'perceive'
        s.push_log('PERCEIVE -- profiling dataset...')
        try:
            profile = self.profiler.profile(self.pipeline.raw_data)
        except Exception as e:
            s.status = 'failed'
            s.error = 'Profiling failed: {}'.format(e)
            s.push_log('ERROR during profiling: {}'.format(e))
            return s.to_dict()

        # Agentic loop
        previous_feedback = ''

        for iteration in range(1, max_iterations + 1):
            s.iteration = iteration
            s.push_log('=== Iteration {}/{} ==='.format(iteration, max_iterations))

            # -- PLAN --------------------------------------------------
            s.current_phase = 'plan'
            s.push_log('PLAN -- asking Gemini to design pipeline...')
            config = self._plan(profile, target_col, task, previous_feedback)
            s.push_log('  -> models: {}'.format(config.get('models', [])))
            s.push_log('  -> cleaning: {}'.format(config.get('cleaning', {})))

            # -- ACT ---------------------------------------------------
            s.current_phase = 'act'
            s.push_log('ACT -- running pipeline...')

            # Reset pipeline modules so each iteration is a fresh run
            try:
                self.pipeline._reset_modules()
            except Exception:
                pass
            self.pipeline.pipeline_report = {
                'status': 'idle', 'start_time': None, 'end_time': None,
                'total_time': None, 'steps_completed': [], 'current_step': None,
                'error': None, 'checkpoints': [],
            }

            try:
                run_report = self.pipeline.run(
                    target_col=target_col,
                    task=task,
                    config=config,
                )
            except Exception as e:
                err_tb = traceback.format_exc()
                s.push_log('  ! Pipeline crashed: {}'.format(e))
                s.history.append({
                    'iteration': iteration,
                    'config':    config,
                    'score':     None,
                    'status':    'crashed',
                    'error':     str(e),
                })
                previous_feedback = self._build_crash_feedback(str(e), err_tb)
                continue

            if run_report.get('status') == 'error':
                err = run_report.get('error', 'Unknown error')
                err_tb = run_report.get('traceback', '')
                s.push_log('  ! Pipeline error: {}'.format(err))
                s.history.append({
                    'iteration': iteration,
                    'config':    config,
                    'score':     None,
                    'status':    'error',
                    'error':     err,
                })
                previous_feedback = self._build_crash_feedback(err, err_tb)
                continue

            # -- OBSERVE -----------------------------------------------
            s.current_phase = 'observe'
            s.push_log('OBSERVE -- evaluating results...')
            score = self._extract_score(run_report, s.goal_metric)
            metrics_summary = self._extract_metrics_summary(run_report)
            overfitting = self._check_overfitting(run_report)
            best_model = run_report.get('best_model', '?')

            s.push_log(
                '  -> best model: {} | {}={}'.format(best_model, s.goal_metric, score)
            )
            if overfitting:
                s.push_log('  WARNING: Possible overfitting detected')

            # Track personal best
            improved = self._is_better(score, s.best_score, s.goal_direction)
            if improved:
                s.best_score = score
                s.best_config = copy.deepcopy(config)
                s.best_run_report = copy.deepcopy(run_report)
                s.push_log('  * New best score: {}'.format(score))

            s.history.append({
                'iteration':       iteration,
                'config':          config,
                'score':           score,
                'best_model':      best_model,
                'metrics_summary': metrics_summary,
                'overfitting':     overfitting,
                'status':          'ok',
                'is_best':         improved,
            })

            # Goal check
            if self._goal_met(score, s.goal_target, s.goal_direction):
                s.push_log(
                    'GOAL ACHIEVED! {}={} >= {}'.format(s.goal_metric, score, s.goal_target)
                )
                break

            # -- REFLECT (skip on last iteration) ----------------------
            if iteration < max_iterations:
                s.current_phase = 'reflect'
                s.push_log('REFLECT -- generating improvement strategy...')
                previous_feedback = self._reflect(
                    profile, target_col, task,
                    config, run_report, score,
                    s.goal_metric, s.goal_target,
                    overfitting
                )
                s.push_log('  -> {}'.format((previous_feedback or '')[:120]))

        # Final status
        if s.best_score is not None and self._goal_met(
                s.best_score, s.goal_target, s.goal_direction):
            s.status = 'success'
        else:
            s.status = 'finished'

        # FINAL SUMMARY
        s.current_phase = 'summarize'
        s.push_log('Generating final summary...')
        s.final_summary = self._generate_final_summary(task, s)
        s.push_log('Agent finished | status={} | best_score={}'.format(s.status, s.best_score))
        return s.to_dict()

    # =====================================================================
    # PLAN -- Gemini designs a pipeline config
    # =====================================================================

    def _plan(self, profile, target_col, task, feedback):
        """Ask Gemini to produce a pipeline config JSON.
        Falls back to a sensible default if Gemini is unavailable."""
        if not self._gemini_json_model:
            return self._default_config(task)

        overview = profile.get('overview', {})
        missing = profile.get('missing', {})
        columns_summary = [
            {
                'name': c.get('name'), 'type': c.get('type'),
                'unique': c.get('unique'), 'missing': c.get('missing')
            }
            for c in profile.get('columns', [])
        ]

        prompt_data = {
            'task': task,
            'target_column': target_col,
            'dataset_overview': overview,
            'missing_values_summary': {
                'total': missing.get('total'),
                'columns_with_missing': missing.get('columns_with_missing'),
            },
            'columns': columns_summary,
            'data_quality_alerts': [a.get('message') for a in profile.get('alerts', [])],
        }

        feedback_section = ''
        if feedback:
            feedback_section = (
                '\nPrevious iteration feedback (use this to improve your plan):\n' + feedback
            )

        primary_metric = 'F1 Score' if task == 'classification' else 'R2 Score'

        prompt = (
            'You are an expert ML engineer running an autonomous optimization loop.\n'
            'Your job is to design the BEST possible pipeline configuration to '
            'maximize {} on this dataset.\n\n'.format(primary_metric)
            + 'dataset_profile = {}\n'.format(json.dumps(prompt_data, indent=2))
            + feedback_section
            + '\nReturn a JSON object with a single key "config" containing:\n'
            '{\n'
            '  "config": {\n'
            '    "cleaning": {\n'
            '      "missing_values": "auto",\n'
            '      "outliers": "iqr"\n'
            '    },\n'
            '    "transformation": {\n'
            '      "encoding": "auto",\n'
            '      "scaling": "standard"\n'
            '    },\n'
            '    "feature_engineering": {\n'
            '      "polynomial": false,\n'
            '      "interactions": false,\n'
            '      "selection": "none"\n'
            '    },\n'
            '    "imbalance": { "method": "none" },\n'
            '    "models": ["random_forest", "xgboost"],\n'
            '    "tuning": { "enable": false }\n'
            '  }\n'
            '}\n\n'
            'missing_values options: auto, mean, median, mode, knn, drop, none\n'
            'outliers options: iqr, zscore, isolation_forest, none\n'
            'encoding options: auto, onehot, label, none\n'
            'scaling options: standard, minmax, robust, none\n'
            'selection options: mutual_info, tree, correlation, rfe, lasso, none\n'
            'imbalance method: smote, adasyn, none (classification only)\n'
            'Classification models: random_forest, xgboost, lightgbm, catboost, '
            'logistic_regression, decision_tree, svm, mlp\n'
            'Regression models: random_forest, xgboost, lightgbm, catboost, '
            'linear_regression, decision_tree, svr, mlp\n'
            'Keep models to 2-4 best choices. Return ONLY the JSON.\n'
        )

        try:
            response = self._gemini_json_model.generate_content(prompt)
            result = json.loads(response.text)
            return result.get('config', self._default_config(task))
        except Exception as e:
            logger.warning('[Agent] Gemini plan failed: %s -- using default config', e)
            return self._default_config(task)

    # =====================================================================
    # REFLECT -- Gemini reasons about what went wrong and how to improve
    # =====================================================================

    def _reflect(self, profile, target_col, task, config, run_report,
                 score, goal_metric, goal_target, overfitting):
        """Ask Gemini to analyse this iteration and generate actionable improvement
        suggestions for the next PLAN call. Returns plain-text feedback."""
        if not self._gemini_text_model:
            return self._default_reflect_feedback(score, goal_metric, goal_target, overfitting)

        eval_results = run_report.get('evaluation_results', {})
        feature_importance = run_report.get('feature_importance', {})
        top_features = list(feature_importance.items())[:5] if feature_importance else []

        prompt = (
            'You are an expert ML engineer reviewing iteration results.\n\n'
            'Task: {} | Target: {}\n'.format(task, target_col)
            + 'Goal: Maximize {} to >= {}\n'.format(goal_metric, goal_target)
            + 'This iteration best {}: {}\n\n'.format(goal_metric, score)
            + 'Config used:\n{}\n\n'.format(json.dumps(config, indent=2))
            + 'Evaluation results (summary):\n'
            + '{}\n\n'.format(
                json.dumps({k: {m: v.get(m) for m in ('f1', 'accuracy', 'r2', 'rmse')
                                if v.get(m) is not None}
                            for k, v in eval_results.items()}, indent=2)
            )
            + 'Overfitting detected: {}\n'.format(overfitting)
            + 'Top features: {}\n\n'.format(json.dumps(top_features, indent=2))
            + 'Provide 3-5 specific, actionable bullet points for the NEXT iteration '
            'to improve the {} score. Name exact models or methods.\n'.format(goal_metric)
        )

        try:
            resp = self._gemini_text_model.generate_content(prompt)
            return resp.text.strip()
        except Exception as e:
            logger.warning('[Agent] Gemini reflect failed: %s', e)
            return self._default_reflect_feedback(score, goal_metric, goal_target, overfitting)

    # =====================================================================
    # FINAL SUMMARY -- Gemini narrates the whole journey
    # =====================================================================

    def _generate_final_summary(self, task, s):
        if not self._gemini_text_model:
            return self._default_final_summary(s)

        history_summary = [
            {
                'iteration': h['iteration'],
                'score': h.get('score'),
                'models': h.get('config', {}).get('models', []),
                'status': h.get('status'),
                'is_best': h.get('is_best', False),
            }
            for h in s.history
        ]

        prompt = (
            'Summarize this autonomous ML optimization run for the user in '
            'friendly markdown (3-5 sentences, under 200 words).\n\n'
            'Task: {}\n'.format(task)
            + 'Goal: {} >= {}\n'.format(s.goal_metric, s.goal_target)
            + 'Iterations run: {}\n'.format(s.iteration)
            + 'Best score: {}\n'.format(s.best_score)
            + 'Status: {}\n'.format(s.status)
            + 'Best models: {}\n\n'.format(s.best_config.get('models', []))
            + 'History:\n{}\n\n'.format(json.dumps(history_summary, indent=2))
            + 'Mention what worked, what was tried, and how the agent improved. '
            'Write for a non-expert user.\n'
        )

        try:
            resp = self._gemini_text_model.generate_content(prompt)
            return resp.text.strip()
        except Exception:
            return self._default_final_summary(s)

    # =====================================================================
    # Helper utilities
    # =====================================================================

    def _extract_score(self, report, metric):
        """Pull the primary metric value from the best model's eval results."""
        try:
            best = report.get('best_model')
            evals = report.get('evaluation_results', {})
            if best and best in evals:
                val = evals[best].get(metric)
                if val is not None:
                    return round(float(val), 4)
            # Fallback: best score across all models
            scores = [
                float(v.get(metric))
                for v in evals.values()
                if v.get(metric) is not None
            ]
            if scores:
                return round(max(scores), 4)
        except Exception:
            pass
        return None

    def _extract_metrics_summary(self, report):
        """Compact metrics dict for the history log."""
        try:
            best = report.get('best_model')
            evals = report.get('evaluation_results', {})
            if best and best in evals:
                r = evals[best]
                return {
                    k: r.get(k) for k in
                    ('accuracy', 'f1', 'roc_auc', 'r2', 'rmse', 'mae')
                    if r.get(k) is not None
                }
        except Exception:
            pass
        return {}

    def _check_overfitting(self, report):
        """Heuristic: flag if test metric is much lower than CV score."""
        try:
            evals = report.get('evaluation_results', {})
            for m in evals.values():
                cv = m.get('cv_scores', [])
                if cv:
                    cv_mean = sum(cv) / len(cv)
                    for metric in ('f1', 'accuracy', 'r2'):
                        val = m.get(metric)
                        if val is not None and (val - cv_mean) > 0.08:
                            return True
        except Exception:
            pass
        return False

    def _is_better(self, score, best, direction):
        if score is None:
            return False
        if best is None:
            return True
        if direction == 'maximize':
            return score > best
        return score < best  # minimize

    def _goal_met(self, score, target, direction):
        if score is None:
            return False
        if direction == 'maximize':
            return score >= target
        return score <= target

    def _build_crash_feedback(self, error, tb):
        tb_short = tb[-800:] if len(tb) > 800 else tb
        return (
            'The previous run crashed: {}\n'
            'Traceback (last part): {}\n'
            'For the next iteration: simplify the config -- fewer models, '
            'disable polynomial/interaction features, use "auto" imputation, '
            'avoid "knn" imputer on large datasets.'
        ).format(error, tb_short)

    def _default_reflect_feedback(self, score, metric, target, overfitting):
        lines = ['{} = {} (target: {})'.format(metric, score, target)]
        if overfitting:
            lines.append('- Overfitting detected -- try simpler models or more regularization')
        if score is None or (isinstance(score, float) and score < 0.7):
            lines.append('- Try ensemble models: xgboost, lightgbm, catboost')
            lines.append('- Try feature selection: mutual_info or tree')
        return '\n'.join(lines)

    def _default_final_summary(self, s):
        return (
            'The agent ran {} iteration(s) and achieved a best {} of {}. '
            'Status: {}.'
        ).format(s.iteration, s.goal_metric, s.best_score, s.status)

    def _default_config(self, task):
        models = (
            ['random_forest', 'xgboost', 'lightgbm']
            if task == 'classification'
            else ['random_forest', 'xgboost', 'lightgbm']
        )
        return {
            'cleaning': {'missing_values': 'auto', 'outliers': 'iqr'},
            'transformation': {'encoding': 'auto', 'scaling': 'standard'},
            'feature_engineering': {
                'polynomial': False, 'interactions': False, 'selection': 'none'
            },
            'imbalance': {'method': 'none'},
            'models': models,
            'tuning': {'enable': False},
        }
