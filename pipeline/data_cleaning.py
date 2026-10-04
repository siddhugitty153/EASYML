"""
Data Cleaning Module
---------------------
Handles missing values, duplicates, and outliers in uploaded datasets.
Designed for non-technical users — returns human-readable reports.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy import stats


class DataCleaner:
    """Clean raw data: imputation, deduplication, outlier handling."""

    def __init__(self):
        self.report = {
            'missing_values': {},
            'duplicates': {},
            'outliers': {},
            'actions': []
        }

    def get_report(self):
        return self.report

    # ── Missing Value Analysis ──

    def analyze_missing(self, df: pd.DataFrame) -> dict:
        """Return per-column missing value summary."""
        total = len(df)
        missing = df.isnull().sum()
        summary = {}
        for col in df.columns:
            count = int(missing[col])
            summary[col] = {
                'count': count,
                'percentage': round(count / total * 100, 2) if total > 0 else 0
            }
        return summary

    def impute_missing(self, df: pd.DataFrame, strategy: str = 'auto',
                       columns: list = None) -> pd.DataFrame:
        """
        Fill missing values.

        Strategies:
            'auto'   — mean for numeric, mode for categorical
            'mean'   — mean (numeric only)
            'median' — median (numeric only)
            'mode'   — most frequent value
            'knn'    — KNN imputation (numeric only, k=5)
            'drop'   — drop rows with any missing values
        """
        result = df.copy()
        cols = columns or result.columns.tolist()
        filled = {}

        if strategy == 'drop':
            before = len(result)
            result = result.dropna(subset=cols).reset_index(drop=True)
            after = len(result)
            self.report['missing_values'] = {'strategy': 'drop', 'rows_dropped': before - after}
            self.report['actions'].append(f'Dropped {before - after} rows with missing values')
            return result

        numeric_cols = result[cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = [c for c in cols if c not in numeric_cols]

        if strategy == 'knn':
            if numeric_cols:
                imputer = KNNImputer(n_neighbors=5)
                before_na = result[numeric_cols].isnull().sum().to_dict()
                result[numeric_cols] = imputer.fit_transform(result[numeric_cols])
                filled.update({c: {'strategy': 'knn', 'count': int(before_na.get(c, 0))}
                               for c in numeric_cols if before_na.get(c, 0) > 0})
            # KNN can't handle categorical — fall back to mode
            for col in categorical_cols:
                if result[col].isnull().any():
                    mode_val = result[col].mode()
                    if len(mode_val) > 0:
                        count = int(result[col].isnull().sum())
                        result[col] = result[col].fillna(mode_val[0])
                        filled[col] = {'strategy': 'mode', 'count': count, 'fill_value': str(mode_val[0])}
        else:
            for col in cols:
                if not result[col].isnull().any():
                    continue

                count = int(result[col].isnull().sum())
                strat = strategy

                if strat == 'auto':
                    strat = 'mean' if col in numeric_cols else 'mode'

                if strat == 'mean' and col in numeric_cols:
                    fill_val = result[col].mean()
                    result[col] = result[col].fillna(fill_val)
                    filled[col] = {'strategy': 'mean', 'count': count, 'fill_value': round(float(fill_val), 4)}
                elif strat == 'median' and col in numeric_cols:
                    fill_val = result[col].median()
                    result[col] = result[col].fillna(fill_val)
                    filled[col] = {'strategy': 'median', 'count': count, 'fill_value': round(float(fill_val), 4)}
                elif strat == 'mode' or col in categorical_cols:
                    mode_val = result[col].mode()
                    if len(mode_val) > 0:
                        result[col] = result[col].fillna(mode_val[0])
                        filled[col] = {'strategy': 'mode', 'count': count, 'fill_value': str(mode_val[0])}
                elif strat in ('mean', 'median') and col in categorical_cols:
                    mode_val = result[col].mode()
                    if len(mode_val) > 0:
                        result[col] = result[col].fillna(mode_val[0])
                        filled[col] = {'strategy': 'mode (fallback)', 'count': count, 'fill_value': str(mode_val[0])}

        self.report['missing_values'] = {
            'strategy': strategy,
            'columns_filled': filled
        }
        total_filled = sum(v['count'] for v in filled.values())
        self.report['actions'].append(f'Imputed {total_filled} missing values using {strategy}')
        return result

    # ── Duplicate Removal ──

    def remove_duplicates(self, df: pd.DataFrame, subset: list = None,
                          keep: str = 'first') -> pd.DataFrame:
        """Remove duplicate rows."""
        result = df.copy()
        before = len(result)
        result = result.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        after = len(result)
        removed = before - after

        self.report['duplicates'] = {
            'total_before': before,
            'duplicates_removed': removed,
            'total_after': after,
            'subset': subset,
            'keep': keep
        }
        if removed > 0:
            self.report['actions'].append(f'Removed {removed} duplicate rows')
        return result

    # ── Outlier Detection ──

    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr',
                        columns: list = None, threshold: float = None) -> dict:
        """
        Detect outliers in numeric columns.

        Methods:
            'zscore' — |z| > threshold (default 3.0)
            'iqr'    — outside Q1 - 1.5*IQR to Q3 + 1.5*IQR
            'isolation_forest' — Isolation Forest anomaly detection
            'one_class_svm'    — One-Class SVM
        """
        numeric_cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_info = {}

        if method == 'zscore':
            z_thresh = threshold or 3.0
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                mask = z_scores > z_thresh
                indices = df[col].dropna().index[mask].tolist()
                outlier_info[col] = {
                    'count': len(indices),
                    'indices': indices[:50],  # Cap for readability
                    'threshold': z_thresh
                }

        elif method == 'iqr':
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = (df[col] < lower) | (df[col] > upper)
                indices = df.index[mask].tolist()
                outlier_info[col] = {
                    'count': int(mask.sum()),
                    'indices': indices[:50],
                    'lower_bound': round(float(lower), 4),
                    'upper_bound': round(float(upper), 4)
                }

        elif method == 'isolation_forest':
            contamination = threshold or 0.05
            data = df[numeric_cols].dropna()
            if len(data) > 0:
                iso = IsolationForest(contamination=contamination, random_state=42)
                preds = iso.fit_predict(data)
                outlier_indices = data.index[preds == -1].tolist()
                outlier_info['_all_columns'] = {
                    'count': len(outlier_indices),
                    'indices': outlier_indices[:50],
                    'contamination': contamination
                }

        elif method == 'one_class_svm':
            data = df[numeric_cols].dropna()
            if len(data) > 0 and len(data) < 10000:  # SVM is slow on large data
                svm = OneClassSVM(nu=threshold or 0.05, kernel='rbf', gamma='scale')
                preds = svm.fit_predict(data)
                outlier_indices = data.index[preds == -1].tolist()
                outlier_info['_all_columns'] = {
                    'count': len(outlier_indices),
                    'indices': outlier_indices[:50]
                }

        return outlier_info

    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr',
                        action: str = 'clip', columns: list = None,
                        threshold: float = None) -> pd.DataFrame:
        """
        Detect and handle outliers.

        Actions:
            'clip'   — clip values to bounds
            'remove' — remove outlier rows
            'nan'    — replace outliers with NaN (for later imputation)
        """
        result = df.copy()
        outlier_info = self.detect_outliers(result, method=method, columns=columns, threshold=threshold)
        numeric_cols = columns or result.select_dtypes(include=[np.number]).columns.tolist()

        total_handled = 0

        if method in ('zscore', 'iqr'):
            for col in numeric_cols:
                if col not in outlier_info or outlier_info[col]['count'] == 0:
                    continue

                if method == 'iqr':
                    lower = outlier_info[col]['lower_bound']
                    upper = outlier_info[col]['upper_bound']
                elif method == 'zscore':
                    mean = result[col].mean()
                    std = result[col].std()
                    z_thresh = threshold or 3.0
                    lower = mean - z_thresh * std
                    upper = mean + z_thresh * std

                if action == 'clip':
                    result[col] = result[col].clip(lower=lower, upper=upper)
                elif action == 'nan':
                    mask = (result[col] < lower) | (result[col] > upper)
                    result.loc[mask, col] = np.nan
                elif action == 'remove':
                    mask = (result[col] >= lower) & (result[col] <= upper)
                    result = result[mask]

                total_handled += outlier_info[col]['count']

        elif method in ('isolation_forest', 'one_class_svm'):
            info = outlier_info.get('_all_columns', {})
            indices = info.get('indices', [])
            if action == 'remove':
                # Re-detect to get ALL indices (not capped at 50)
                data = result[numeric_cols].dropna()
                if method == 'isolation_forest':
                    model = IsolationForest(contamination=threshold or 0.05, random_state=42)
                else:
                    model = OneClassSVM(nu=threshold or 0.05, kernel='rbf', gamma='scale')
                preds = model.fit_predict(data)
                all_outlier_indices = data.index[preds == -1].tolist()
                result = result.drop(all_outlier_indices).reset_index(drop=True)
                total_handled = len(all_outlier_indices)
            elif action == 'nan':
                for idx in indices:
                    if idx in result.index:
                        result.loc[idx, numeric_cols] = np.nan
                total_handled = len(indices)

        self.report['outliers'] = {
            'method': method,
            'action': action,
            'details': outlier_info,
            'total_handled': total_handled
        }
        if total_handled > 0:
            self.report['actions'].append(
                f'Handled {total_handled} outliers using {method}/{action}'
            )

        return result

    def clean(self, df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
        """
        Run full cleaning pipeline with a config dict.

        Default config:
        {
            'imputation': 'auto',
            'remove_duplicates': True,
            'outlier_method': 'iqr',
            'outlier_action': 'clip'
        }
        """
        config = config or {}
        result = df.copy()

        # Step 1: Remove duplicates
        if config.get('remove_duplicates', True):
            result = self.remove_duplicates(result)

        # Step 2: Handle outliers (before imputation so outliers don't skew means)
        outlier_method = config.get('outlier_method', 'iqr')
        outlier_action = config.get('outlier_action', 'clip')
        if outlier_method and outlier_method != 'none':
            result = self.handle_outliers(result, method=outlier_method, action=outlier_action)

        # Step 3: Impute missing values
        imputation = config.get('imputation', 'auto')
        if imputation and imputation != 'none':
            result = self.impute_missing(result, strategy=imputation)

        return result
