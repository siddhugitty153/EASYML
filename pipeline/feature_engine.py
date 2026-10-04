"""
Feature Engineering & Selection Module
-----------------------------------------
Creates new features and selects the most informative subset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import (
    SelectKBest, chi2, mutual_info_classif, mutual_info_regression,
    RFE
)
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class FeatureEngineer:
    """Create and select features for ML models."""

    def __init__(self):
        self.report = {
            'created': {},
            'selected': {},
            'actions': []
        }

    def get_report(self):
        return self.report

    # ── Feature Creation ──

    def create_polynomial_features(self, df: pd.DataFrame, columns: list = None,
                                   degree: int = 2, interaction_only: bool = False,
                                   max_features: int = 50) -> pd.DataFrame:
        """Create polynomial and interaction features."""
        result = df.copy()
        cols = columns or result.select_dtypes(include=[np.number]).columns.tolist()

        if len(cols) < 2:
            return result

        # Limit input columns to avoid explosion
        cols = cols[:min(len(cols), 10)]

        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only,
                                  include_bias=False)
        poly_data = poly.fit_transform(result[cols])
        poly_names = poly.get_feature_names_out(cols)

        # Only add new features (not the original ones)
        new_features = []
        for i, name in enumerate(poly_names):
            if name not in cols and len(new_features) < max_features:
                clean_name = name.replace(' ', '_')
                result[clean_name] = poly_data[:, i]
                new_features.append(clean_name)

        self.report['created']['polynomial'] = {
            'input_columns': cols,
            'degree': degree,
            'new_features': new_features,
            'count': len(new_features)
        }
        if new_features:
            self.report['actions'].append(f'Created {len(new_features)} polynomial features')
        return result

    def create_interaction_features(self, df: pd.DataFrame,
                                    columns: list = None) -> pd.DataFrame:
        """Create pairwise interaction (multiplication) features."""
        result = df.copy()
        cols = columns or result.select_dtypes(include=[np.number]).columns.tolist()
        cols = cols[:min(len(cols), 8)]  # Limit to avoid explosion

        new_features = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                name = f'{cols[i]}_x_{cols[j]}'
                result[name] = result[cols[i]] * result[cols[j]]
                new_features.append(name)

        self.report['created']['interactions'] = {
            'count': len(new_features),
            'features': new_features
        }
        if new_features:
            self.report['actions'].append(f'Created {len(new_features)} interaction features')
        return result

    def create_time_features(self, df: pd.DataFrame,
                             datetime_col: str = None) -> pd.DataFrame:
        """Extract time-based features from a datetime column."""
        result = df.copy()

        # Auto-detect datetime columns
        if datetime_col is None:
            dt_cols = result.select_dtypes(include=['datetime64']).columns.tolist()
            if not dt_cols:
                # Try to parse object columns
                for col in result.select_dtypes(include=['object']).columns:
                    try:
                        pd.to_datetime(result[col].head(5))
                        dt_cols.append(col)
                        break
                    except Exception:
                        pass
            if dt_cols:
                datetime_col = dt_cols[0]
            else:
                return result

        dt = pd.to_datetime(result[datetime_col], errors='coerce')
        new_features = []

        result[f'{datetime_col}_hour'] = dt.dt.hour
        result[f'{datetime_col}_dayofweek'] = dt.dt.dayofweek
        result[f'{datetime_col}_month'] = dt.dt.month
        result[f'{datetime_col}_day'] = dt.dt.day
        result[f'{datetime_col}_quarter'] = dt.dt.quarter
        result[f'{datetime_col}_is_weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        new_features = [f'{datetime_col}_{x}' for x in
                        ['hour', 'dayofweek', 'month', 'day', 'quarter', 'is_weekend']]

        self.report['created']['time'] = {
            'source_column': datetime_col,
            'features': new_features
        }
        self.report['actions'].append(f'Created {len(new_features)} time features from {datetime_col}')
        return result

    def create_aggregation_features(self, df: pd.DataFrame, columns: list = None,
                                    windows: list = None) -> pd.DataFrame:
        """Create rolling aggregation features (mean, std, min, max)."""
        result = df.copy()
        cols = columns or result.select_dtypes(include=[np.number]).columns.tolist()
        windows = windows or [3, 5, 10]
        cols = cols[:min(len(cols), 5)]  # Limit columns

        new_features = []
        for col in cols:
            for w in windows:
                if len(result) >= w:
                    result[f'{col}_rolling_mean_{w}'] = result[col].rolling(w, min_periods=1).mean()
                    result[f'{col}_rolling_std_{w}'] = result[col].rolling(w, min_periods=1).std()
                    new_features.extend([f'{col}_rolling_mean_{w}', f'{col}_rolling_std_{w}'])

        self.report['created']['aggregations'] = {
            'columns': cols,
            'windows': windows,
            'count': len(new_features)
        }
        if new_features:
            self.report['actions'].append(f'Created {len(new_features)} aggregation features')
        return result

    # ── Feature Selection ──

    def select_by_correlation(self, df: pd.DataFrame, target_col: str,
                              threshold: float = 0.05) -> list:
        """Select features with correlation above threshold to target."""
        numeric = df.select_dtypes(include=[np.number])
        if target_col not in numeric.columns:
            return numeric.columns.tolist()

        corr = numeric.corr()[target_col].abs().sort_values(ascending=False)
        selected = corr[corr >= threshold].index.tolist()
        if target_col in selected:
            selected.remove(target_col)

        self.report['selected']['correlation'] = {
            'threshold': threshold,
            'selected': selected,
            'count': len(selected)
        }
        return selected

    def select_by_mutual_info(self, df: pd.DataFrame, target_col: str,
                              k: int = 20, task: str = 'classification') -> list:
        """Select top-k features by mutual information."""
        numeric = df.select_dtypes(include=[np.number]).dropna()
        if target_col not in numeric.columns:
            return numeric.columns.tolist()

        X = numeric.drop(columns=[target_col])
        y = numeric[target_col]

        if len(X.columns) == 0:
            return []

        k = min(k, len(X.columns))
        scorer = mutual_info_classif if task == 'classification' else mutual_info_regression

        selector = SelectKBest(score_func=scorer, k=k)
        selector.fit(X, y)
        mask = selector.get_support()
        selected = X.columns[mask].tolist()

        scores = dict(zip(X.columns, selector.scores_))
        self.report['selected']['mutual_info'] = {
            'task': task,
            'k': k,
            'selected': selected,
            'scores': {k: round(float(v), 4) for k, v in
                       sorted(scores.items(), key=lambda x: x[1], reverse=True)}
        }
        self.report['actions'].append(f'Selected {len(selected)} features by mutual info')
        return selected

    def select_by_rfe(self, df: pd.DataFrame, target_col: str,
                      n_features: int = 10, task: str = 'classification') -> list:
        """Select features using Recursive Feature Elimination."""
        numeric = df.select_dtypes(include=[np.number]).dropna()
        if target_col not in numeric.columns:
            return numeric.columns.tolist()

        X = numeric.drop(columns=[target_col])
        y = numeric[target_col]

        if len(X.columns) == 0:
            return []

        n_features = min(n_features, len(X.columns))

        if task == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

        rfe = RFE(estimator=estimator, n_features_to_select=n_features, step=1)
        rfe.fit(X, y)
        selected = X.columns[rfe.support_].tolist()

        self.report['selected']['rfe'] = {
            'n_features': n_features,
            'selected': selected,
            'ranking': dict(zip(X.columns, rfe.ranking_.tolist()))
        }
        self.report['actions'].append(f'Selected {len(selected)} features by RFE')
        return selected

    def select_by_importance(self, df: pd.DataFrame, target_col: str,
                             method: str = 'lasso', threshold: float = 0.01,
                             task: str = 'classification') -> list:
        """
        Select features using embedded methods.
        Methods: 'lasso', 'ridge', 'tree'
        """
        numeric = df.select_dtypes(include=[np.number]).dropna()
        if target_col not in numeric.columns:
            return numeric.columns.tolist()

        X = numeric.drop(columns=[target_col])
        y = numeric[target_col]

        if len(X.columns) == 0:
            return []

        if method == 'lasso':
            model = Lasso(alpha=0.01, random_state=42)
            model.fit(X, y)
            importances = np.abs(model.coef_)
        elif method == 'ridge':
            model = Ridge(alpha=1.0)
            model.fit(X, y)
            importances = np.abs(model.coef_)
        elif method == 'tree':
            if task == 'classification':
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X, y)
            importances = model.feature_importances_

        imp_dict = dict(zip(X.columns, importances))
        selected = [col for col, imp in imp_dict.items() if imp >= threshold]

        self.report['selected'][method] = {
            'threshold': threshold,
            'selected': selected,
            'importances': {k: round(float(v), 4) for k, v in
                            sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)}
        }
        self.report['actions'].append(f'Selected {len(selected)} features by {method}')
        return selected

    def engineer(self, df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
        """
        Run full feature engineering pipeline.

        Config:
        {
            'polynomial': False,
            'polynomial_degree': 2,
            'interactions': False,
            'time_features': True,
            'aggregations': False,
            'selection_method': None,    # 'correlation', 'mutual_info', 'rfe', 'lasso', 'tree'
            'target_col': None,
            'task': 'classification',
            'n_select': 20
        }
        """
        config = config or {}
        result = df.copy()

        # Creation
        if config.get('time_features', True):
            result = self.create_time_features(result)
        if config.get('polynomial', False):
            degree = config.get('polynomial_degree', 2)
            result = self.create_polynomial_features(result, degree=degree)
        if config.get('interactions', False):
            result = self.create_interaction_features(result)
        if config.get('aggregations', False):
            result = self.create_aggregation_features(result)

        # Selection
        target_col = config.get('target_col')
        selection = config.get('selection_method')
        task = config.get('task', 'classification')

        if selection and target_col and target_col in result.columns:
            if selection == 'correlation':
                selected = self.select_by_correlation(result, target_col)
            elif selection == 'mutual_info':
                k = config.get('n_select', 20)
                selected = self.select_by_mutual_info(result, target_col, k=k, task=task)
            elif selection == 'rfe':
                n = config.get('n_select', 10)
                selected = self.select_by_rfe(result, target_col, n_features=n, task=task)
            elif selection in ('lasso', 'ridge', 'tree'):
                selected = self.select_by_importance(result, target_col,
                                                     method=selection, task=task)
            else:
                selected = None

            if selected is not None:
                keep = selected + [target_col]
                non_numeric = result.select_dtypes(exclude=[np.number]).columns.tolist()
                keep = list(set(keep + non_numeric))
                result = result[[c for c in keep if c in result.columns]]

        return result
