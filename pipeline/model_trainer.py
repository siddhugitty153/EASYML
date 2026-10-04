"""
Model Training Module
-----------------------
Unified interface for training supervised, unsupervised, and deep learning models.
"""

import pandas as pd
import numpy as np
import time


class ModelTrainer:
    """Train ML models with a unified API."""

    CLASSIFICATION_MODELS = {
        'logistic_regression': 'Logistic Regression',
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'catboost': 'CatBoost',
        'svm': 'SVM',
        'mlp': 'MLP Neural Network'
    }

    REGRESSION_MODELS = {
        'linear_regression': 'Linear Regression',
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'catboost': 'CatBoost',
        'svm': 'SVM',
        'mlp': 'MLP Neural Network'
    }

    UNSUPERVISED_MODELS = {
        'kmeans': 'K-Means',
        'dbscan': 'DBSCAN',
        'hierarchical': 'Hierarchical Clustering',
        'gmm': 'Gaussian Mixture Model'
    }

    def __init__(self):
        self.trained_models = {}
        self.report = {'models': {}, 'actions': []}

    def get_report(self):
        return self.report

    def get_available_models(self, task: str = 'classification') -> dict:
        """List available models for a given task type."""
        if task == 'classification':
            return self.CLASSIFICATION_MODELS.copy()
        elif task == 'regression':
            return self.REGRESSION_MODELS.copy()
        elif task == 'clustering':
            return self.UNSUPERVISED_MODELS.copy()
        return {**self.CLASSIFICATION_MODELS, **self.REGRESSION_MODELS, **self.UNSUPERVISED_MODELS}

    def train(self, model_name: str, X_train, y_train=None,
              task: str = 'classification', params: dict = None) -> dict:
        """
        Train a single model.

        Args:
            model_name: key from available models
            X_train: training features
            y_train: training labels (None for unsupervised)
            task: 'classification', 'regression', or 'clustering'
            params: optional hyperparameters dict

        Returns: { model, train_time, model_name, task }
        """
        params = params or {}
        start = time.time()

        # Quick check: is the model valid for the task?
        available = self.get_available_models(task)
        if model_name not in available:
            return {'error': f"Model '{model_name}' is not suitable for task '{task}'"}

        model = self._create_model(model_name, task, params)
        if model is None:
            return {'error': f'Failed to create model: {model_name}'}

        try:
            if y_train is not None:
                model.fit(X_train, y_train)
            else:
                model.fit(X_train)
        except Exception as e:
            return {'error': f'Training failed for {model_name}: {str(e)}'}

        train_time = round(time.time() - start, 3)

        result = {
            'model': model,
            'model_name': model_name,
            'display_name': available.get(model_name, model_name),
            'task': task,
            'train_time': train_time,
            'n_features': X_train.shape[1] if hasattr(X_train, 'shape') else 0,
            'n_samples': len(X_train)
        }

        self.trained_models[model_name] = result
        self.report['models'][model_name] = {
            'display_name': result['display_name'],
            'train_time': train_time,
            'n_samples': result['n_samples'],
            'n_features': result['n_features']
        }
        self.report['actions'].append(f'Trained {result["display_name"]} in {train_time}s')
        return result

    def train_multiple(self, model_names: list, X_train, y_train=None,
                       task: str = 'classification',
                       params: dict = None) -> dict:
        """Train multiple models and return results."""
        params = params or {}
        results = {}
        for name in model_names:
            model_params = params.get(name, {})
            results[name] = self.train(name, X_train, y_train, task, model_params)
        return results

    def train_all(self, X_train, y_train=None,
                  task: str = 'classification',
                  params: dict = None) -> dict:
        """Train all applicable models for the given task."""
        available = self.get_available_models(task)
        models = list(available.keys())
        return self.train_multiple(models, X_train, y_train, task, params)

    def _create_model(self, name: str, task: str, params: dict):
        """Instantiate a model by name."""

        if name == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=1000, random_state=42, **params)

        elif name == 'linear_regression':
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**params)

        elif name == 'decision_tree':
            if task == 'classification':
                from sklearn.tree import DecisionTreeClassifier
                return DecisionTreeClassifier(random_state=42, **params)
            else:
                from sklearn.tree import DecisionTreeRegressor
                return DecisionTreeRegressor(random_state=42, **params)

        elif name == 'random_forest':
            kwargs = {'n_estimators': 100, 'random_state': 42, 'n_jobs': 1}
            kwargs.update(params)
            if task == 'classification':
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(**kwargs)
            else:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(**kwargs)

        elif name == 'xgboost':
            try:
                kwargs = {'n_estimators': 100, 'random_state': 42, 'n_jobs': 1}
                kwargs.update(params)
                if task == 'classification':
                    from xgboost import XGBClassifier
                    kwargs['eval_metric'] = kwargs.get('eval_metric', 'logloss')
                    # use_label_encoder is deprecated in new xgboost, safe to omit or keep if old
                    return XGBClassifier(**kwargs)
                else:
                    from xgboost import XGBRegressor
                    return XGBRegressor(**kwargs)
            except ImportError:
                return None

        elif name == 'lightgbm':
            try:
                kwargs = {'n_estimators': 100, 'random_state': 42, 'verbose': -1, 'n_jobs': 1}
                kwargs.update(params)
                if task == 'classification':
                    from lightgbm import LGBMClassifier
                    return LGBMClassifier(**kwargs)
                else:
                    from lightgbm import LGBMRegressor
                    return LGBMRegressor(**kwargs)
            except ImportError:
                return None

        elif name == 'catboost':
            try:
                kwargs = {'iterations': 100, 'random_state': 42, 'verbose': 0}
                kwargs.update(params)
                if task == 'classification':
                    from catboost import CatBoostClassifier
                    return CatBoostClassifier(**kwargs)
                else:
                    from catboost import CatBoostRegressor
                    return CatBoostRegressor(**kwargs)
            except ImportError:
                return None

        elif name == 'svm':
            if task == 'classification':
                from sklearn.svm import SVC
                return SVC(probability=True, random_state=42, **params)
            else:
                from sklearn.svm import SVR
                return SVR(**params)

        elif name == 'mlp':
            kwargs = {'hidden_layer_sizes': (100, 50), 'max_iter': 500, 'random_state': 42}
            kwargs.update(params)
            if task == 'classification':
                from sklearn.neural_network import MLPClassifier
                return MLPClassifier(**kwargs)
            else:
                from sklearn.neural_network import MLPRegressor
                return MLPRegressor(**kwargs)

        # ── Unsupervised Models ──

        elif name == 'kmeans':
            from sklearn.cluster import KMeans
            return KMeans(n_clusters=params.get('n_clusters', 3),
                          random_state=42, n_init=10)

        elif name == 'dbscan':
            from sklearn.cluster import DBSCAN
            return DBSCAN(eps=params.get('eps', 0.5),
                          min_samples=params.get('min_samples', 5))

        elif name == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            return AgglomerativeClustering(
                n_clusters=params.get('n_clusters', 3))

        elif name == 'gmm':
            from sklearn.mixture import GaussianMixture
            return GaussianMixture(n_components=params.get('n_components', 3),
                                   random_state=42)

        return None
