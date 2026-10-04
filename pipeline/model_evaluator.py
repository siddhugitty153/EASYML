"""
Model Evaluation & Tuning Module
-----------------------------------
Evaluate model performance with metrics, cross-validation,
and hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, TimeSeriesSplit
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)


class ModelEvaluator:
    """Evaluate and tune ML models."""

    def __init__(self):
        self.report = {
            'evaluations': {},
            'cross_validations': {},
            'tuning': {},
            'actions': []
        }

    def get_report(self):
        return self.report

    # ── Metrics ──

    def evaluate_classification(self, model, X_test, y_test,
                                model_name: str = 'model') -> dict:
        """Compute classification metrics."""
        y_pred = model.predict(X_test)

        # Ensure both y_test and y_pred are discrete for classification metrics
        try:
            y_test = np.asarray(y_test)
            y_pred = np.asarray(y_pred)
            # If predictions are float (e.g., from a regressor used for classification),
            # round them to nearest int so sklearn treats them as class labels
            if y_pred.dtype.kind == 'f':
                y_pred = np.round(y_pred).astype(int)
            if y_test.dtype.kind == 'f':
                y_test = np.round(y_test).astype(int)
        except Exception:
            pass

        metrics = {
            'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
            'precision': round(float(precision_score(y_test, y_pred,
                                                      average='weighted', zero_division=0)), 4),
            'recall': round(float(recall_score(y_test, y_pred,
                                               average='weighted', zero_division=0)), 4),
            'f1': round(float(f1_score(y_test, y_pred,
                                       average='weighted', zero_division=0)), 4),
        }

        # ROC-AUC (if probability predictions available)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:
                    metrics['roc_auc'] = round(float(roc_auc_score(y_test, y_proba[:, 1])), 4)
                else:
                    metrics['roc_auc'] = round(float(roc_auc_score(
                        y_test, y_proba, multi_class='ovr', average='weighted'
                    )), 4)
        except Exception:
            metrics['roc_auc'] = None

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Classification report
        try:
            cr = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            # Clean up for JSON serialization
            clean_cr = {}
            for k, v in cr.items():
                if isinstance(v, dict):
                    clean_cr[str(k)] = {kk: round(float(vv), 4) for kk, vv in v.items()}
                else:
                    clean_cr[str(k)] = round(float(v), 4)
            metrics['classification_report'] = clean_cr
        except Exception:
            pass

        self.report['evaluations'][model_name] = metrics
        self.report['actions'].append(
            f'Evaluated {model_name}: accuracy={metrics["accuracy"]}, f1={metrics["f1"]}'
        )
        return metrics

    def evaluate_regression(self, model, X_test, y_test,
                            model_name: str = 'model') -> dict:
        """Compute regression metrics."""
        y_pred = model.predict(X_test)
        
        # Robustness: ensure numeric types
        y_test = np.asarray(y_test, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        
        metrics = {
            'mse': round(float(mean_squared_error(y_test, y_pred)), 4),
            'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            'mae': round(float(mean_absolute_error(y_test, y_pred)), 4),
            'r2': round(float(r2_score(y_test, y_pred)), 4)
        }

        self.report['evaluations'][model_name] = metrics
        self.report['actions'].append(
            f'Evaluated {model_name}: R²={metrics["r2"]}, RMSE={metrics["rmse"]}'
        )
        return metrics

    def evaluate(self, model, X_test, y_test,
                 task: str = 'classification',
                 model_name: str = 'model') -> dict:
        """Auto-evaluate based on task type."""
        if task == 'classification':
            return self.evaluate_classification(model, X_test, y_test, model_name)
        else:
            return self.evaluate_regression(model, X_test, y_test, model_name)

    def evaluate_multiple(self, trained_models: dict, X_test, y_test,
                          task: str = 'classification') -> dict:
        """Evaluate multiple trained models and rank them."""
        results = {}
        for name, info in trained_models.items():
            if 'error' in info:
                results[name] = {'error': info['error']}
                continue

            try:
                model = info['model']
                metrics = self.evaluate(model, X_test, y_test, task, name)
                metrics['train_time'] = info.get('train_time', 0)
                metrics['display_name'] = info.get('display_name', name)
                results[name] = metrics
            except Exception as e:
                results[name] = {
                    'error': f'Evaluation failed: {str(e)}',
                    'display_name': info.get('display_name', name)
                }

        # Rank by primary metric
        primary_metric = 'f1' if task == 'classification' else 'r2'
        ranked = sorted(
            [(k, v) for k, v in results.items() if 'error' not in v],
            key=lambda x: x[1].get(primary_metric, -999999),
            reverse=True
        )
        for rank, (name, _) in enumerate(ranked, 1):
            results[name]['rank'] = rank

        return results

    # ── Cross-Validation ──

    def cross_validate(self, model, X, y, task: str = 'classification',
                       method: str = 'kfold', n_splits: int = 5,
                       model_name: str = 'model') -> dict:
        """
        Perform cross-validation.

        Methods: 'kfold', 'stratified', 'timeseries'
        """
        if method == 'stratified':
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        elif method == 'timeseries':
            cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        scoring = 'f1_weighted' if task == 'classification' else 'r2'

        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            result = {
                'method': method,
                'n_splits': n_splits,
                'scoring': scoring,
                'scores': [round(float(s), 4) for s in scores],
                'mean': round(float(scores.mean()), 4),
                'std': round(float(scores.std()), 4)
            }
        except Exception as e:
            result = {
                'method': method,
                'error': str(e)
            }

        self.report['cross_validations'][model_name] = result
        if 'mean' in result:
            self.report['actions'].append(
                f'CV {model_name}: {result["mean"]:.4f} ± {result["std"]:.4f}'
            )
        return result

    # ── Hyperparameter Tuning ──

    def tune_grid_search(self, model, X, y, param_grid: dict,
                         task: str = 'classification',
                         n_splits: int = 3,
                         model_name: str = 'model') -> dict:
        """Grid search hyperparameter tuning."""
        from sklearn.model_selection import GridSearchCV

        scoring = 'f1_weighted' if task == 'classification' else 'r2'
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) \
            if task == 'classification' else KFold(n_splits=n_splits, shuffle=True, random_state=42)

        start = time.time()
        search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring,
                              n_jobs=1, refit=True)
        search.fit(X, y)
        tune_time = round(time.time() - start, 3)

        result = {
            'method': 'grid_search',
            'best_params': search.best_params_,
            'best_score': round(float(search.best_score_), 4),
            'best_model': search.best_estimator_,
            'tune_time': tune_time,
            'n_candidates': len(search.cv_results_['mean_test_score'])
        }

        self.report['tuning'][model_name] = {
            k: v for k, v in result.items() if k != 'best_model'
        }
        self.report['actions'].append(
            f'Grid tuned {model_name}: best={result["best_score"]:.4f} in {tune_time}s'
        )
        return result

    def tune_random_search(self, model, X, y, param_distributions: dict,
                           n_iter: int = 50, task: str = 'classification',
                           n_splits: int = 3,
                           model_name: str = 'model') -> dict:
        """Random search hyperparameter tuning."""
        from sklearn.model_selection import RandomizedSearchCV

        scoring = 'f1_weighted' if task == 'classification' else 'r2'
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) \
            if task == 'classification' else KFold(n_splits=n_splits, shuffle=True, random_state=42)

        start = time.time()
        search = RandomizedSearchCV(model, param_distributions, n_iter=n_iter,
                                     cv=cv, scoring=scoring, n_jobs=1,
                                     random_state=42, refit=True)
        search.fit(X, y)
        tune_time = round(time.time() - start, 3)

        result = {
            'method': 'random_search',
            'best_params': search.best_params_,
            'best_score': round(float(search.best_score_), 4),
            'best_model': search.best_estimator_,
            'tune_time': tune_time,
            'n_iter': n_iter
        }

        self.report['tuning'][model_name] = {
            k: v for k, v in result.items() if k != 'best_model'
        }
        self.report['actions'].append(
            f'Random tuned {model_name}: best={result["best_score"]:.4f} in {tune_time}s'
        )
        return result

    def tune_bayesian(self, model_name: str, X, y,
                      task: str = 'classification',
                      n_trials: int = 50,
                      n_splits: int = 3) -> dict:
        """
        Bayesian hyperparameter optimization using Optuna.
        Auto-defines search spaces for known model types.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            return {'error': 'Optuna not installed'}

        scoring = 'f1_weighted' if task == 'classification' else 'r2'

        def objective(trial):
            params = self._get_optuna_params(trial, model_name, task)
            if params is None:
                return 0

            from pipeline.model_trainer import ModelTrainer
            trainer = ModelTrainer()
            result = trainer.train(model_name, X, y, task, params)
            if 'error' in result:
                return 0

            model = result['model']
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) \
                if task == 'classification' else KFold(n_splits=n_splits, shuffle=True, random_state=42)
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
                return scores.mean()
            except Exception:
                return 0

        start = time.time()
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False, timeout=60)
        except Exception as e:
            return {'error': f'Optuna tuning failed or timed out completely: {str(e)}'}
            
        tune_time = round(time.time() - start, 3)

        # Retrain best model
        from pipeline.model_trainer import ModelTrainer
        trainer = ModelTrainer()
        best_result = trainer.train(model_name, X, y, task, study.best_params)

        result = {
            'method': 'bayesian',
            'best_params': study.best_params,
            'best_score': round(float(study.best_value), 4),
            'best_model': best_result.get('model'),
            'tune_time': tune_time,
            'n_trials': n_trials
        }

        self.report['tuning'][model_name] = {
            k: v for k, v in result.items() if k != 'best_model'
        }
        self.report['actions'].append(
            f'Bayesian tuned {model_name}: best={result["best_score"]:.4f} in {tune_time}s'
        )
        return result

    def _get_optuna_params(self, trial, model_name, task):
        """Define Optuna search spaces for each model type."""
        if model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
        elif model_name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
        elif model_name == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            }
        elif model_name == 'logistic_regression':
            return {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
            }
        elif model_name == 'svm':
            return {
                'C': trial.suggest_float('C', 0.01, 100, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly']),
            }
        elif model_name == 'decision_tree':
            return {
                'max_depth': trial.suggest_int('max_depth', 2, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            }
        return None

    def tune(self, model, model_name: str, X, y,
             method: str = 'bayesian', task: str = 'classification',
             param_grid: dict = None, n_trials: int = 50) -> dict:
        """
        Tune hyperparameters with chosen method.
        Methods: 'grid', 'random', 'bayesian'
        """
        if method == 'grid' and param_grid:
            return self.tune_grid_search(model, X, y, param_grid,
                                          task, model_name=model_name)
        elif method == 'random' and param_grid:
            return self.tune_random_search(model, X, y, param_grid,
                                            task=task, model_name=model_name)
        elif method == 'bayesian':
            return self.tune_bayesian(model_name, X, y, task, n_trials=n_trials)
        return {'error': f'Invalid tuning method: {method}'}
