"""
Data Transformation & Encoding Module
---------------------------------------
Converts raw fields into numeric features.
Handles categorical encoding, numerical scaling, and non-linear transforms.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, MinMaxScaler,
    StandardScaler, RobustScaler
)
from scipy import stats


class DataTransformer:
    """Transform and encode data for ML consumption."""

    def __init__(self):
        self.encoders = {}   # column -> fitted encoder
        self.scalers = {}    # column -> fitted scaler
        self.report = {
            'encoding': {},
            'scaling': {},
            'transforms': {},
            'actions': []
        }

    def get_report(self):
        return self.report

    # ── Categorical Encoding ──

    def label_encode(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """Apply Label Encoding to categorical columns."""
        result = df.copy()
        cols = columns or result.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in cols:
            le = LabelEncoder()
            mask = result[col].notna()
            result.loc[mask, col] = le.fit_transform(result.loc[mask, col].astype(str))
            result[col] = pd.to_numeric(result[col], errors='coerce')
            self.encoders[col] = {'type': 'label', 'encoder': le}
            self.report['encoding'][col] = {
                'method': 'label',
                'classes': le.classes_.tolist()
            }

        self.report['actions'].append(f'Label encoded {len(cols)} columns: {cols}')
        return result

    def onehot_encode(self, df: pd.DataFrame, columns: list = None,
                      max_categories: int = 20) -> pd.DataFrame:
        """Apply One-Hot Encoding to categorical columns."""
        result = df.copy()
        cols = columns or result.select_dtypes(include=['object', 'category']).columns.tolist()

        encoded_cols = []
        for col in cols:
            nunique = result[col].nunique()
            if nunique > max_categories:
                # Too many categories — skip or use label encoding
                self.report['encoding'][col] = {
                    'method': 'skipped',
                    'reason': f'Too many categories ({nunique} > {max_categories})'
                }
                continue

            dummies = pd.get_dummies(result[col], prefix=col, drop_first=True, dtype=int)
            result = pd.concat([result, dummies], axis=1)
            result = result.drop(columns=[col])
            encoded_cols.append(col)
            self.report['encoding'][col] = {
                'method': 'onehot',
                'new_columns': dummies.columns.tolist(),
                'categories': int(nunique)
            }

        if encoded_cols:
            self.report['actions'].append(f'One-hot encoded {len(encoded_cols)} columns')
        return result

    def target_encode(self, df: pd.DataFrame, target_col: str,
                      columns: list = None) -> pd.DataFrame:
        """Apply Target/Mean Encoding to categorical columns."""
        result = df.copy()
        cols = columns or result.select_dtypes(include=['object', 'category']).columns.tolist()

        if target_col in cols:
            cols.remove(target_col)

        for col in cols:
            means = result.groupby(col)[target_col].mean()
            result[col] = result[col].map(means)
            self.encoders[col] = {'type': 'target', 'mapping': means.to_dict()}
            self.report['encoding'][col] = {
                'method': 'target',
                'mapping': {str(k): round(float(v), 4) for k, v in means.items()}
            }

        self.report['actions'].append(f'Target encoded {len(cols)} columns using {target_col}')
        return result

    def encode(self, df: pd.DataFrame, method: str = 'auto',
               target_col: str = None) -> pd.DataFrame:
        """
        Apply encoding with chosen method.
        'auto' uses one-hot for low cardinality, label for high.
        """
        if method == 'onehot':
            return self.onehot_encode(df)
        elif method == 'label':
            return self.label_encode(df)
        elif method == 'target' and target_col:
            return self.target_encode(df, target_col)
        elif method == 'auto':
            result = df.copy()
            cat_cols = result.select_dtypes(include=['object', 'category']).columns.tolist()
            low_card = [c for c in cat_cols if result[c].nunique() <= 10]
            high_card = [c for c in cat_cols if result[c].nunique() > 10]

            if low_card:
                result = self.onehot_encode(result, columns=low_card)
            if high_card:
                result = self.label_encode(result, columns=high_card)
            return result
        return df

    # ── Numerical Scaling ──

    def scale(self, df: pd.DataFrame, method: str = 'standard',
              columns: list = None, exclude: list = None) -> pd.DataFrame:
        """
        Scale numeric columns.

        Methods: 'minmax', 'standard', 'robust'
        """
        result = df.copy()
        cols = columns or result.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude target and specified columns
        if exclude:
            cols = [c for c in cols if c not in exclude]

        if not cols:
            return result

        scaler_map = {
            'minmax': MinMaxScaler,
            'standard': StandardScaler,
            'robust': RobustScaler
        }

        scaler_class = scaler_map.get(method, StandardScaler)
        scaler = scaler_class()
        result[cols] = scaler.fit_transform(result[cols])

        self.scalers['_numeric'] = {'type': method, 'scaler': scaler, 'columns': cols}
        self.report['scaling'] = {
            'method': method,
            'columns': cols
        }
        self.report['actions'].append(f'Scaled {len(cols)} numeric columns using {method}')
        return result

    # ── Non-linear Transforms ──

    def log_transform(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """Apply log(1 + x) transform to numeric columns."""
        result = df.copy()
        cols = columns or result.select_dtypes(include=[np.number]).columns.tolist()

        transformed = []
        for col in cols:
            if (result[col] >= 0).all():
                result[col] = np.log1p(result[col])
                transformed.append(col)

        self.report['transforms']['log'] = {'columns': transformed}
        if transformed:
            self.report['actions'].append(f'Log transformed {len(transformed)} columns')
        return result

    def boxcox_transform(self, df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
        """Apply Box-Cox transform to positive numeric columns."""
        result = df.copy()
        cols = columns or result.select_dtypes(include=[np.number]).columns.tolist()

        transformed = {}
        for col in cols:
            col_data = result[col].dropna()
            if (col_data > 0).all() and len(col_data) > 3:
                try:
                    transformed_data, lam = stats.boxcox(col_data)
                    result.loc[col_data.index, col] = transformed_data
                    transformed[col] = {'lambda': round(float(lam), 4)}
                except Exception:
                    pass  # Skip columns that can't be Box-Cox transformed

        self.report['transforms']['boxcox'] = transformed
        if transformed:
            self.report['actions'].append(f'Box-Cox transformed {len(transformed)} columns')
        return result

    def transform(self, df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
        """
        Run full transformation pipeline.

        Config:
        {
            'encoding': 'auto',       # 'auto', 'onehot', 'label', 'target'
            'target_col': None,       # required for 'target' encoding
            'scaling': 'standard',    # 'minmax', 'standard', 'robust', 'none'
            'nonlinear': None         # 'log', 'boxcox', None
        }
        """
        config = config or {}
        result = df.copy()

        # Step 1: Encode categoricals
        encoding = config.get('encoding', 'auto')
        target_col = config.get('target_col')
        if encoding and encoding != 'none':
            result = self.encode(result, method=encoding, target_col=target_col)

        # Step 2: Non-linear transforms (before scaling)
        nonlinear = config.get('nonlinear')
        if nonlinear == 'log':
            result = self.log_transform(result)
        elif nonlinear == 'boxcox':
            result = self.boxcox_transform(result)

        # Step 3: Scale
        scaling = config.get('scaling', 'standard')
        if scaling and scaling != 'none':
            exclude_from_scaling = [target_col] if target_col else []
            result = self.scale(result, method=scaling, exclude=exclude_from_scaling)

        return result
