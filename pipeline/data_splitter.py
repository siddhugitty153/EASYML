"""
Data Splitting Module
-----------------------
Partition data for training, validation, and evaluation.
Supports random, stratified, and time-series splits.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit


class DataSplitter:
    """Split data into train / validation / test sets."""

    def __init__(self):
        self.report = {
            'method': None,
            'train_size': 0,
            'val_size': 0,
            'test_size': 0,
            'actions': []
        }

    def get_report(self):
        return self.report

    def split(self, X: pd.DataFrame, y=None,
              method: str = 'random',
              test_size: float = 0.2,
              val_size: float = 0.1,
              random_state: int = 42,
              n_splits: int = 5) -> dict:
        """
        Split data into train/val/test.

        Methods:
            'random'     — random shuffle split
            'stratified' — preserves class distribution (requires y)
            'timeseries' — respects temporal ordering

        Returns dict:
            {
                'X_train', 'X_val', 'X_test',
                'y_train', 'y_val', 'y_test',
                'cv_splits' (for timeseries only)
            }
        """
        result = {}

        if method == 'timeseries':
            return self._timeseries_split(X, y, n_splits=n_splits,
                                          test_size=test_size)

        stratify_col = None
        if method == 'stratified' and y is not None:
            # Safety check: stratified split requires at least 2 members per class
            counts = pd.Series(y).value_counts()
            if counts.min() < 2:
                self.report['actions'].append(
                    f'Stratification fallback: least populated class has {counts.min()} members (min 2 required). '
                    'Using random split instead.'
                )
                stratify_col = None
            else:
                stratify_col = y

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=stratify_col
        )

        # Second split: train vs val
        if val_size > 0:
            val_fraction = val_size / (1 - test_size)
            stratify_inner = y_temp if method == 'stratified' and y_temp is not None else None
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_fraction,
                random_state=random_state, stratify=stratify_inner
            )
        else:
            X_train, X_val = X_temp, pd.DataFrame()
            y_train, y_val = y_temp, pd.Series(dtype=float) if y is not None else None

        result = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }

        self.report['method'] = method
        self.report['train_size'] = len(X_train)
        self.report['val_size'] = len(X_val) if X_val is not None else 0
        self.report['test_size'] = len(X_test)
        self.report['actions'].append(
            f'{method.capitalize()} split: train={len(X_train)}, '
            f'val={len(X_val) if X_val is not None else 0}, test={len(X_test)}'
        )
        return result

    def _timeseries_split(self, X, y, n_splits: int = 5,
                          test_size: float = 0.2) -> dict:
        """Time-series aware split. Returns the last fold as test."""
        n_test = int(len(X) * test_size)
        n_train = len(X) - n_test

        X_train = X.iloc[:n_train]
        X_test = X.iloc[n_train:]
        y_train = y.iloc[:n_train] if y is not None else None
        y_test = y.iloc[n_train:] if y is not None else None

        # Generate CV splits for the training set
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_splits = []
        for train_idx, val_idx in tscv.split(X_train):
            cv_splits.append({
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist()
            })

        result = {
            'X_train': X_train, 'X_val': pd.DataFrame(), 'X_test': X_test,
            'y_train': y_train, 'y_val': None, 'y_test': y_test,
            'cv_splits': cv_splits
        }

        self.report['method'] = 'timeseries'
        self.report['train_size'] = len(X_train)
        self.report['test_size'] = len(X_test)
        self.report['n_cv_splits'] = n_splits
        self.report['actions'].append(
            f'Time-series split: train={len(X_train)}, test={len(X_test)}, '
            f'{n_splits} CV folds'
        )
        return result
