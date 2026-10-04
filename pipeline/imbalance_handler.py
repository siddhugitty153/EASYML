"""
Imbalanced Data Handler Module
---------------------------------
Adjusts class distribution for classification tasks.
Supports undersampling, oversampling, and hybrid methods.
"""

import pandas as pd
import numpy as np


class ImbalanceHandler:
    """Handle class imbalance in classification datasets."""

    def __init__(self):
        self.report = {
            'method': None,
            'before': {},
            'after': {},
            'actions': []
        }

    def get_report(self):
        return self.report

    def get_class_distribution(self, y) -> dict:
        """Get class distribution as counts and percentages."""
        counts = pd.Series(y).value_counts().to_dict()
        total = sum(counts.values())
        return {
            str(k): {
                'count': int(v),
                'percentage': round(v / total * 100, 2)
            }
            for k, v in counts.items()
        }

    def apply_smote(self, X: pd.DataFrame, y, random_state: int = 42):
        """Apply SMOTE oversampling."""
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            self.report['actions'].append('SMOTE skipped: imbalanced-learn not installed')
            return X, y

        sampler = SMOTE(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), y_res

    def apply_adasyn(self, X: pd.DataFrame, y, random_state: int = 42):
        """Apply ADASYN oversampling."""
        try:
            from imblearn.over_sampling import ADASYN
        except ImportError:
            self.report['actions'].append('ADASYN skipped: imbalanced-learn not installed')
            return X, y

        sampler = ADASYN(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), y_res

    def apply_tomek(self, X: pd.DataFrame, y):
        """Apply Tomek Links undersampling."""
        try:
            from imblearn.under_sampling import TomekLinks
        except ImportError:
            self.report['actions'].append('Tomek Links skipped: imbalanced-learn not installed')
            return X, y

        sampler = TomekLinks()
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), y_res

    def apply_enn(self, X: pd.DataFrame, y):
        """Apply Edited Nearest Neighbors undersampling."""
        try:
            from imblearn.under_sampling import EditedNearestNeighbours
        except ImportError:
            self.report['actions'].append('ENN skipped: imbalanced-learn not installed')
            return X, y

        sampler = EditedNearestNeighbours()
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), y_res

    def apply_smote_enn(self, X: pd.DataFrame, y, random_state: int = 42):
        """Apply SMOTE-ENN hybrid method."""
        try:
            from imblearn.combine import SMOTEENN
        except ImportError:
            self.report['actions'].append('SMOTE-ENN skipped: imbalanced-learn not installed')
            return X, y

        sampler = SMOTEENN(random_state=random_state)
        X_res, y_res = sampler.fit_resample(X, y)
        return pd.DataFrame(X_res, columns=X.columns), y_res

    def handle(self, X: pd.DataFrame, y, method: str = 'smote') -> tuple:
        """
        Apply imbalance handling.

        Methods: 'smote', 'adasyn', 'tomek', 'enn', 'smote_enn'
        Returns: (X_resampled, y_resampled)
        """
        self.report['before'] = self.get_class_distribution(y)

        method_map = {
            'smote': self.apply_smote,
            'adasyn': self.apply_adasyn,
            'tomek': self.apply_tomek,
            'enn': self.apply_enn,
            'smote_enn': self.apply_smote_enn
        }

        handler = method_map.get(method)
        if handler is None:
            self.report['actions'].append(f'Unknown method: {method}')
            return X, y

        X_res, y_res = handler(X, y)

        self.report['method'] = method
        self.report['after'] = self.get_class_distribution(y_res)
        self.report['samples_before'] = len(X)
        self.report['samples_after'] = len(X_res)
        self.report['actions'].append(
            f'{method}: {len(X)} → {len(X_res)} samples'
        )

        return X_res, y_res
