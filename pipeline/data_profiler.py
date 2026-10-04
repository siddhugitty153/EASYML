"""
Data Profiler — Lightweight EDA Report Generator
---------------------------------------------------
Generates a comprehensive data profile using only pandas/numpy.
Includes: overview, column stats, missing values, correlations,
and data quality alerts.
"""

import numpy as np
import pandas as pd


class DataProfiler:
    """Generate an EDA profile report from a DataFrame."""

    def profile(self, df: pd.DataFrame) -> dict:
        """
        Run full profiling on the DataFrame.

        Returns dict with sections:
            overview, columns, missing, correlations, alerts
        """
        report = {}
        report['overview'] = self._overview(df)
        report['columns'] = self._column_stats(df)
        report['missing'] = self._missing_values(df)
        report['correlations'] = self._correlations(df)
        report['alerts'] = self._quality_alerts(df)
        return report

    # ──────────────────────────────────────────────
    # Overview
    # ──────────────────────────────────────────────

    def _overview(self, df: pd.DataFrame) -> dict:
        """High-level dataset summary."""
        mem_bytes = int(df.memory_usage(deep=True).sum())
        dtype_counts = df.dtypes.value_counts()

        return {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_mb': round(mem_bytes / (1024 * 1024), 2),
            'duplicates': int(df.duplicated().sum()),
            'total_missing': int(df.isnull().sum().sum()),
            'missing_pct': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2) if len(df) > 0 else 0,
            'dtypes': {str(k): int(v) for k, v in dtype_counts.items()},
            'numeric_cols': int(df.select_dtypes(include='number').shape[1]),
            'categorical_cols': int(df.select_dtypes(include=['object', 'category']).shape[1]),
        }

    # ──────────────────────────────────────────────
    # Per-Column Statistics
    # ──────────────────────────────────────────────

    def _column_stats(self, df: pd.DataFrame) -> list:
        """Detailed stats for each column."""
        stats = []
        for col in df.columns:
            s = df[col]
            info = {
                'name': col,
                'dtype': str(s.dtype),
                'missing': int(s.isnull().sum()),
                'missing_pct': round(s.isnull().mean() * 100, 2),
                'unique': int(s.nunique()),
                'unique_pct': round(s.nunique() / len(df) * 100, 2) if len(df) > 0 else 0,
            }

            if pd.api.types.is_numeric_dtype(s):
                info['type'] = 'numeric'
                clean = s.dropna()
                if len(clean) > 0:
                    info['min'] = round(float(clean.min()), 4)
                    info['max'] = round(float(clean.max()), 4)
                    info['mean'] = round(float(clean.mean()), 4)
                    info['median'] = round(float(clean.median()), 4)
                    info['std'] = round(float(clean.std()), 4)
                    info['skewness'] = round(float(clean.skew()), 4)
                    info['kurtosis'] = round(float(clean.kurtosis()), 4)
                    info['zeros'] = int((clean == 0).sum())
                    info['negatives'] = int((clean < 0).sum())

                    # Outlier count (IQR method)
                    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                    iqr = q3 - q1
                    outliers = ((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum()
                    info['outliers'] = int(outliers)

                    # Simple histogram (10 bins)
                    try:
                        counts, edges = np.histogram(clean, bins=10)
                        info['histogram'] = {
                            'counts': counts.tolist(),
                            'edges': [round(float(e), 4) for e in edges],
                        }
                    except Exception:
                        pass
            else:
                info['type'] = 'categorical'
                clean = s.dropna()
                if len(clean) > 0:
                    vc = clean.value_counts()
                    top_n = min(10, len(vc))
                    info['top_values'] = [
                        {'value': str(v), 'count': int(c), 'pct': round(c / len(clean) * 100, 2)}
                        for v, c in vc.head(top_n).items()
                    ]
                    info['mode'] = str(vc.index[0])
                    info['mode_count'] = int(vc.iloc[0])

            stats.append(info)
        return stats

    # ──────────────────────────────────────────────
    # Missing Values
    # ──────────────────────────────────────────────

    def _missing_values(self, df: pd.DataFrame) -> dict:
        """Missing value analysis."""
        missing = df.isnull().sum()
        missing_cols = missing[missing > 0].sort_values(ascending=False)

        return {
            'total': int(missing.sum()),
            'columns_with_missing': len(missing_cols),
            'details': [
                {
                    'column': col,
                    'count': int(count),
                    'pct': round(count / len(df) * 100, 2),
                }
                for col, count in missing_cols.items()
            ],
        }

    # ──────────────────────────────────────────────
    # Correlations
    # ──────────────────────────────────────────────

    def _correlations(self, df: pd.DataFrame) -> dict:
        """Correlation analysis for numeric columns."""
        num_df = df.select_dtypes(include='number')
        if num_df.shape[1] < 2:
            return {'matrix': [], 'columns': [], 'top_pairs': []}

        corr = num_df.corr()

        # Build top correlated pairs (excluding self-correlations)
        pairs = []
        cols = corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if not np.isnan(val):
                    pairs.append({
                        'col1': cols[i],
                        'col2': cols[j],
                        'correlation': round(float(val), 4),
                        'abs': round(abs(float(val)), 4),
                    })

        pairs.sort(key=lambda x: x['abs'], reverse=True)

        # Build matrix for heatmap
        matrix = []
        for row in corr.values:
            matrix.append([round(float(v), 4) if not np.isnan(v) else 0 for v in row])

        return {
            'columns': cols,
            'matrix': matrix,
            'top_pairs': pairs[:15],  # Top 15 most correlated pairs
        }

    # ──────────────────────────────────────────────
    # Data Quality Alerts
    # ──────────────────────────────────────────────

    def _quality_alerts(self, df: pd.DataFrame) -> list:
        """Generate data quality warnings and alerts."""
        alerts = []

        # Duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            pct = round(dup_count / len(df) * 100, 1)
            alerts.append({
                'level': 'warning',
                'category': 'duplicates',
                'message': f'{dup_count} duplicate rows ({pct}% of data)',
            })

        for col in df.columns:
            s = df[col]
            miss_pct = s.isnull().mean() * 100

            # High missing
            if miss_pct > 30:
                alerts.append({
                    'level': 'danger' if miss_pct > 70 else 'warning',
                    'category': 'missing',
                    'message': f'Column "{col}" has {miss_pct:.1f}% missing values',
                })

            # Constant column
            if s.nunique() <= 1:
                alerts.append({
                    'level': 'danger',
                    'category': 'constant',
                    'message': f'Column "{col}" is constant (only 1 unique value)',
                })

            # High cardinality categorical
            if s.dtype == 'object' and s.nunique() > 50:
                alerts.append({
                    'level': 'info',
                    'category': 'cardinality',
                    'message': f'Column "{col}" has high cardinality ({s.nunique()} unique values)',
                })

            # High skewness
            if pd.api.types.is_numeric_dtype(s):
                try:
                    skew = abs(float(s.dropna().skew()))
                    if skew > 3:
                        alerts.append({
                            'level': 'info',
                            'category': 'skewness',
                            'message': f'Column "{col}" is highly skewed (skewness = {skew:.2f})',
                        })
                except Exception:
                    pass

        if not alerts:
            alerts.append({
                'level': 'success',
                'category': 'quality',
                'message': 'No data quality issues detected! Your data looks clean.',
            })

        return alerts
