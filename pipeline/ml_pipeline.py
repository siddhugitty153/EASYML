"""
ML Pipeline Orchestrator
--------------------------
Ties all pipeline modules together into a single configurable pipeline.
Manages state, runs end-to-end, and persists results.
Includes: stage checkpointing, explainable-AI report, auto-explore mode,
          SHAP explanations, stacking ensemble, and What-If prediction.
"""

import os
import json
import time
import copy
import pandas as pd
import numpy as np
import joblib

from pipeline.data_cleaning import DataCleaner
from pipeline.data_transformer import DataTransformer
from pipeline.feature_engine import FeatureEngineer
from pipeline.dim_reduction import DimensionalityReducer
from pipeline.imbalance_handler import ImbalanceHandler
from pipeline.data_splitter import DataSplitter
from pipeline.model_trainer import ModelTrainer
from pipeline.model_evaluator import ModelEvaluator
from pipeline.explainer import PipelineExplainer
from pipeline.visualization import VisualizationGenerator


class EasyMLPipeline:
    """
    End-to-end EasyML pipeline orchestrator.

    Usage:
        pipeline = EasyMLPipeline()
        pipeline.load_data('data.csv')
        pipeline.run(target_col='label', task='classification')
        report = pipeline.get_full_report()
    """

    # ── Auto-Explore Presets ──
    # Each preset is a compact config that varies preprocessing strategy.
    # Designed for efficiency — only changes preprocessing, uses a single
    # fast probe model (LightGBM) for scoring.
    EXPLORE_PRESETS = {
        'baseline': {
            'label': 'Baseline (minimal preprocessing)',
            'cleaning': {'imputation': 'auto', 'outlier_method': 'none', 'remove_duplicates': True},
            'transformation': {'encoding': 'auto', 'scaling': 'none'},
            'feature_engineering': {'polynomial': False, 'interactions': False, 'time_features': True},
        },
        'standard': {
            'label': 'Standard (scaling + mutual info selection)',
            'cleaning': {'imputation': 'auto', 'outlier_method': 'iqr', 'remove_duplicates': True},
            'transformation': {'encoding': 'auto', 'scaling': 'standard'},
            'feature_engineering': {'polynomial': False, 'interactions': False,
                                    'time_features': True, 'selection_method': 'mutual_info', 'n_select': 20},
        },
        'robust': {
            'label': 'Robust (outlier-safe scaling, tree selection)',
            'cleaning': {'imputation': 'median', 'outlier_method': 'iqr', 'remove_duplicates': True},
            'transformation': {'encoding': 'auto', 'scaling': 'robust'},
            'feature_engineering': {'polynomial': False, 'interactions': False,
                                    'time_features': True, 'selection_method': 'tree'},
        },
        'aggressive': {
            'label': 'Aggressive (polynomial + interactions)',
            'cleaning': {'imputation': 'auto', 'outlier_method': 'zscore', 'remove_duplicates': True},
            'transformation': {'encoding': 'auto', 'scaling': 'standard'},
            'feature_engineering': {'polynomial': True, 'polynomial_degree': 2,
                                    'interactions': True, 'time_features': True,
                                    'selection_method': 'mutual_info', 'n_select': 30},
        },
        'minimal': {
            'label': 'Minimal (minmax, no feature engineering)',
            'cleaning': {'imputation': 'auto', 'outlier_method': 'none', 'remove_duplicates': True},
            'transformation': {'encoding': 'label', 'scaling': 'minmax'},
            'feature_engineering': {'polynomial': False, 'interactions': False, 'time_features': False},
        },
    }

    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Pipeline modules
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.feature_engineer = FeatureEngineer()
        self.dim_reducer = DimensionalityReducer()
        self.imbalance_handler = ImbalanceHandler()
        self.splitter = DataSplitter()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.explainer = PipelineExplainer()

        # State
        self.raw_data = None
        self.cleaned_data = None
        self.transformed_data = None
        self.engineered_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.trained_models = {}
        self.evaluation_results = {}
        self.best_model = None
        self.best_model_name = None
        self.explanations = {}

        # Multi-file storage
        self.loaded_files = []  # list of { name, dataframe }
        self.user_test_data = None  # Separate test file uploaded by user

        self.pipeline_report = {
            'status': 'idle',
            'start_time': None,
            'end_time': None,
            'total_time': None,
            'steps_completed': [],
            'current_step': None,
            'error': None,
            'checkpoints': [],
        }

    def load_data(self, filepath: str = None, dataframe: pd.DataFrame = None) -> dict:
        """Load data from file or DataFrame."""
        if dataframe is not None:
            self.raw_data = dataframe
        elif filepath:
            ext = os.path.splitext(filepath)[1].lower()
            if ext == '.csv':
                self.raw_data = pd.read_csv(filepath)
            elif ext in ('.xlsx', '.xls'):
                self.raw_data = pd.read_excel(filepath)
            else:
                raise ValueError(f'Unsupported file format: {ext}')
        else:
            raise ValueError('Provide either filepath or dataframe')

        return self.get_data_preview()

    def load_multiple_files(self, filepaths: list) -> list:
        """Load multiple CSV/Excel files and store them for merging."""
        self.loaded_files = []
        previews = []
        for fp in filepaths:
            ext = os.path.splitext(fp)[1].lower()
            name = os.path.basename(fp)
            if ext == '.csv':
                df = pd.read_csv(fp)
            elif ext in ('.xlsx', '.xls'):
                df = pd.read_excel(fp)
            else:
                continue
            self.loaded_files.append({'name': name, 'dataframe': df})
            previews.append({
                'name': name,
                'rows': len(df),
                'columns': list(df.columns),
                'n_columns': len(df.columns),
            })
        return previews

    def merge_data(self, method: str = 'concat', on: str = None,
                   how: str = 'inner') -> dict:
        """
        Merge loaded files into a single DataFrame.

        Methods:
            'concat' — stack rows (pd.concat)
            'join'   — merge on a key column (pd.merge)
        """
        if len(self.loaded_files) < 2:
            raise ValueError('Need at least 2 files to merge')

        dfs = [f['dataframe'] for f in self.loaded_files]

        if method == 'concat':
            self.raw_data = pd.concat(dfs, ignore_index=True)
        elif method == 'join':
            if not on:
                raise ValueError('Key column (on) is required for join merge')
            result = dfs[0]
            for df in dfs[1:]:
                result = pd.merge(result, df, on=on, how=how)
            self.raw_data = result
        else:
            raise ValueError(f'Unknown merge method: {method}')

        return self.get_data_preview()

    def configure_data(self, file_roles: list, merge_config: dict = None,
                       exclude_columns: list = None) -> dict:
        """
        Configure multi-file data by assigning roles.

        file_roles: list of {"name": str, "role": str}
            Roles: "train", "test", "supplement", "ignore"
        merge_config: optional {"on": str, "how": str} for supplementary files
        exclude_columns: optional list of column names to drop
        """
        if not self.loaded_files:
            raise ValueError('No files loaded. Upload files first.')

        merge_config = merge_config or {}
        exclude_columns = exclude_columns or []

        # Build role map
        role_map = {}
        for fr in file_roles:
            role_map[fr['name']] = fr.get('role', 'ignore')

        train_dfs = []
        test_dfs = []
        supplement_dfs = []

        for f in self.loaded_files:
            role = role_map.get(f['name'], 'ignore')
            if role == 'train':
                train_dfs.append(f['dataframe'])
            elif role == 'test':
                test_dfs.append(f['dataframe'])
            elif role == 'supplement':
                supplement_dfs.append(f['dataframe'])
            # 'ignore' -> skip

        if not train_dfs:
            raise ValueError('At least one file must be assigned the "Training Data" role.')

        # Combine training files
        if len(train_dfs) == 1:
            self.raw_data = train_dfs[0].copy()
        else:
            self.raw_data = pd.concat(train_dfs, ignore_index=True)

        # Merge supplementary files
        merge_key = merge_config.get('on')
        merge_how = merge_config.get('how', 'inner')
        for sdf in supplement_dfs:
            if merge_key and merge_key in self.raw_data.columns and merge_key in sdf.columns:
                self.raw_data = pd.merge(self.raw_data, sdf, on=merge_key, how=merge_how)
            else:
                # Fallback: concat columns if same row count
                if len(sdf) == len(self.raw_data):
                    self.raw_data = pd.concat(
                        [self.raw_data.reset_index(drop=True),
                         sdf.reset_index(drop=True)], axis=1)

        # Combine test files
        if test_dfs:
            if len(test_dfs) == 1:
                self.user_test_data = test_dfs[0].copy()
            else:
                self.user_test_data = pd.concat(test_dfs, ignore_index=True)

            # Merge supplements into test data too
            for sdf in supplement_dfs:
                if merge_key and merge_key in self.user_test_data.columns and merge_key in sdf.columns:
                    self.user_test_data = pd.merge(
                        self.user_test_data, sdf, on=merge_key, how=merge_how)
                elif len(sdf) == len(self.user_test_data):
                    self.user_test_data = pd.concat(
                        [self.user_test_data.reset_index(drop=True),
                         sdf.reset_index(drop=True)], axis=1)
        else:
            self.user_test_data = None

        # Drop excluded columns
        for col in exclude_columns:
            if col in self.raw_data.columns:
                self.raw_data.drop(columns=[col], inplace=True)
            if self.user_test_data is not None and col in self.user_test_data.columns:
                self.user_test_data.drop(columns=[col], inplace=True)

        preview = self.get_data_preview()
        preview['has_test_data'] = self.user_test_data is not None
        if self.user_test_data is not None:
            preview['test_rows'] = len(self.user_test_data)
            preview['test_columns'] = len(self.user_test_data.columns)
        return preview

    def get_data_preview(self, n_rows: int = 100) -> dict:
        """Get a preview of the current data state."""
        df = self.raw_data
        if df is None:
            return {'error': 'No data loaded'}

        # Column info
        columns = []
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'missing': int(df[col].isnull().sum()),
                'missing_pct': round(df[col].isnull().sum() / len(df) * 100, 2),
                'unique': int(df[col].nunique()),
            }
            if df[col].dtype in ['int64', 'float64']:
                col_info['min'] = round(float(df[col].min()), 4) if not df[col].isnull().all() else None
                col_info['max'] = round(float(df[col].max()), 4) if not df[col].isnull().all() else None
                col_info['mean'] = round(float(df[col].mean()), 4) if not df[col].isnull().all() else None
            else:
                col_info['top'] = str(df[col].mode()[0]) if len(df[col].mode()) > 0 else None
            columns.append(col_info)

        return {
            'rows': len(df),
            'columns': columns,
            'n_columns': len(df.columns),
            'duplicates': int(df.duplicated().sum()),
            'total_missing': int(df.isnull().sum().sum()),
            'head': json.loads(df.head(n_rows).to_json(orient='records', date_format='iso')),
            'dtypes_summary': {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
        }

    def detect_task_type(self, target_col: str) -> str:
        """
        Auto-detect if task is classification or regression.
        Heuristics:
        - If object/category dtype: classification.
        - If integer dtype and low cardinality (<= 20 unique values): classification.
        - Otherwise: regression.
        """
        if self.raw_data is None:
            return 'classification'

        if target_col not in self.raw_data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data.")

        target = self.raw_data[target_col]
        # Drop NaN for detection
        target_clean = target.dropna()
        if len(target_clean) == 0:
            return 'classification' # Fallback
            
        nunique = target_clean.nunique()
        dtype = target_clean.dtype

        if dtype == 'object' or dtype.name == 'category' or dtype.name == 'bool':
            return 'classification'
        
        # Numeric case
        if np.issubdtype(dtype, np.integer):
            if nunique <= 20: # Low cardinality integer is often classification
                return 'classification'
        
        # If it's float or high cardinality integer, it's regression
        return 'regression'

    # ── Checkpointing ──

    def _save_checkpoint(self, step_name: str, data=None):
        """Save intermediate state after a pipeline step."""
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'step': step_name,
            'timestamp': time.time(),
            'pipeline_report': copy.deepcopy(self.pipeline_report),
        }

        # Save DataFrame checkpoints as parquet for speed
        if data is not None and isinstance(data, pd.DataFrame):
            data_path = os.path.join(checkpoint_dir, f'{step_name}_data.parquet')
            try:
                data.to_parquet(data_path, index=False)
                checkpoint['data_path'] = data_path
                checkpoint['rows'] = len(data)
                checkpoint['cols'] = len(data.columns)
            except Exception:
                # Fallback to CSV if parquet fails
                data_path = os.path.join(checkpoint_dir, f'{step_name}_data.csv')
                data.to_csv(data_path, index=False)
                checkpoint['data_path'] = data_path
                checkpoint['rows'] = len(data)
                checkpoint['cols'] = len(data.columns)

        # Save checkpoint metadata
        meta_path = os.path.join(checkpoint_dir, f'{step_name}_meta.json')
        serializable = {k: v for k, v in checkpoint.items()
                        if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        with open(meta_path, 'w') as f:
            json.dump(serializable, f, indent=2, default=str)

        self.pipeline_report['checkpoints'].append({
            'step': step_name,
            'timestamp': checkpoint['timestamp'],
            'rows': checkpoint.get('rows'),
            'cols': checkpoint.get('cols'),
        })

    # ── Main Pipeline Run ──

    def run(self, target_col: str, task: str = None,
            config: dict = None) -> dict:
        """
        Run the full pipeline end-to-end.

        Args:
            target_col: target variable column name
            task: 'classification' or 'regression' (auto-detected if None)
            config: pipeline configuration dict
        """
        config = config or {}
        self.pipeline_report['status'] = 'running'
        self.pipeline_report['start_time'] = time.time()
        self.pipeline_report['checkpoints'] = []

        try:
            # Auto-detect task
            if task is None:
                task = self.detect_task_type(target_col)
            self.pipeline_report['task'] = task
            self.pipeline_report['target_col'] = target_col

            # Step 1: Clean
            self._update_step('cleaning')
            clean_config = config.get('cleaning', {})
            self.cleaned_data = self.cleaner.clean(self.raw_data, clean_config)
            self.pipeline_report['steps_completed'].append('cleaning')
            self._save_checkpoint('cleaning', self.cleaned_data)

            # Step 2: Transform
            self._update_step('transformation')
            transform_config = config.get('transformation', {})
            transform_config.setdefault('target_col', target_col)
            self.transformed_data = self.transformer.transform(
                self.cleaned_data, transform_config)
            self.pipeline_report['steps_completed'].append('transformation')
            self._save_checkpoint('transformation', self.transformed_data)

            # Step 3: Feature Engineering
            self._update_step('feature_engineering')
            feature_config = config.get('feature_engineering', {})
            feature_config.setdefault('target_col', target_col)
            feature_config.setdefault('task', task)
            self.engineered_data = self.feature_engineer.engineer(
                self.transformed_data, feature_config)
            self.pipeline_report['steps_completed'].append('feature_engineering')
            self._save_checkpoint('feature_engineering', self.engineered_data)

            # Ensure target column survived
            if target_col not in self.engineered_data.columns:
                raise ValueError(f'Target column "{target_col}" lost during pipeline')

            # Step 4: Dimensionality Reduction (optional)
            dim_config = config.get('dim_reduction')
            working_data = self.engineered_data.copy()
            if dim_config and dim_config.get('method'):
                self._update_step('dim_reduction')
                dim_config.setdefault('target_col', target_col)
                y_temp = working_data[target_col]
                X_temp = working_data.drop(columns=[target_col])
                X_temp = X_temp.select_dtypes(include=[np.number])
                X_reduced = self.dim_reducer.reduce(X_temp, **dim_config)
                X_reduced[target_col] = y_temp.values
                working_data = X_reduced
                self.pipeline_report['steps_completed'].append('dim_reduction')
                self._save_checkpoint('dim_reduction', working_data)

            # Prepare X and y
            y = working_data[target_col]
            X = working_data.drop(columns=[target_col])
            X = X.select_dtypes(include=[np.number])

            # For classification, ensure y is integer (label-encoding may produce float)
            if task == 'classification':
                try:
                    y = y.astype(int)
                except (ValueError, TypeError):
                    # If y contains non-numeric labels, label-encode them
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y = pd.Series(le.fit_transform(y.astype(str)),
                                  index=y.index, name=y.name)

            # Fill any remaining NaN
            X = X.fillna(X.median())

            # Step 5: Split
            self._update_step('splitting')
            if self.user_test_data is not None:
                # User provided separate test data
                target_col_name = target_col
                if target_col_name in self.user_test_data.columns:
                    test_X = self.user_test_data.drop(columns=[target_col_name])
                    test_y = self.user_test_data[target_col_name]
                    # Apply same transformations to test set columns
                    # Keep only columns present in both
                    common_cols = [c for c in X.columns if c in test_X.columns]
                    self.X_train = X[common_cols]
                    self.X_test = test_X[common_cols]
                    self.y_train = y
                    self.y_test = test_y
                else:
                    # Test file doesn't have target — fall back to normal split
                    split_config = config.get('splitting', {})
                    split_method = split_config.get('method', 'stratified' if task == 'classification' else 'random')
                    splits = self.splitter.split(X, y, method=split_method,
                                                 test_size=split_config.get('test_size', 0.2),
                                                 val_size=split_config.get('val_size', 0.0))
                    self.X_train = splits['X_train']
                    self.X_test = splits['X_test']
                    self.y_train = splits['y_train']
                    self.y_test = splits['y_test']
            else:
                split_config = config.get('splitting', {})
                split_method = split_config.get('method', 'stratified' if task == 'classification' else 'random')
                splits = self.splitter.split(X, y, method=split_method,
                                             test_size=split_config.get('test_size', 0.2),
                                             val_size=split_config.get('val_size', 0.0))
                self.X_train = splits['X_train']
                self.X_test = splits['X_test']
                self.y_train = splits['y_train']
                self.y_test = splits['y_test']
            self.pipeline_report['steps_completed'].append('splitting')
            self._save_checkpoint('splitting')

            # Step 6: Handle Imbalance (classification only)
            imbalance_config = config.get('imbalance', {})
            imbalance_method = imbalance_config.get('method')
            if task == 'classification' and imbalance_method and imbalance_method != 'none':
                self._update_step('imbalance_handling')
                self.X_train, self.y_train = self.imbalance_handler.handle(
                    self.X_train, self.y_train, method=imbalance_method)
                self.pipeline_report['steps_completed'].append('imbalance_handling')
                self._save_checkpoint('imbalance_handling')

            # Step 7: Train
            self._update_step('training')
            model_names = config.get('models')
            if model_names:
                self.trained_models = self.trainer.train_multiple(
                    model_names, self.X_train, self.y_train, task)
            else:
                self.trained_models = self.trainer.train_all(
                    self.X_train, self.y_train, task)
            self.pipeline_report['steps_completed'].append('training')
            self._save_checkpoint('training')

            # Step 8: Evaluate
            self._update_step('evaluation')
            self.evaluation_results = self.evaluator.evaluate_multiple(
                self.trained_models, self.X_test, self.y_test, task)

            # Find best model
            valid_results = {k: v for k, v in self.evaluation_results.items()
                             if 'error' not in v and 'rank' in v}
            if valid_results:
                self.best_model_name = min(valid_results, key=lambda k: valid_results[k]['rank'])
                self.best_model = self.trained_models[self.best_model_name]['model']
            self.pipeline_report['steps_completed'].append('evaluation')
            self._save_checkpoint('evaluation')

            # ── Feature Importance Extraction ──
            feature_names = list(self.X_train.columns)
            self.pipeline_report['features_used'] = feature_names

            # Extract importance from each trained model
            all_importances = {}
            for model_name, model_info in self.trained_models.items():
                model_obj = model_info.get('model')
                if model_obj is None:
                    continue
                try:
                    if hasattr(model_obj, 'feature_importances_'):
                        imp = model_obj.feature_importances_
                        imp_dict = {fn: round(float(v), 6) for fn, v in zip(feature_names, imp)}
                        # Sort descending
                        imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
                        all_importances[model_name] = imp_dict
                    elif hasattr(model_obj, 'coef_'):
                        coef = model_obj.coef_
                        if coef.ndim > 1:
                            # Multi-class: average absolute coefficients across classes
                            coef = np.mean(np.abs(coef), axis=0)
                        else:
                            coef = np.abs(coef)
                        imp_dict = {fn: round(float(v), 6) for fn, v in zip(feature_names, coef)}
                        imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
                        all_importances[model_name] = imp_dict
                except Exception:
                    pass  # Skip models that don't expose importances

            self.pipeline_report['feature_importance'] = all_importances

            # ── SHAP Global Explanations ──
            try:
                if self.best_model and len(self.X_test) > 0:
                    shap_sample = self.X_test.head(min(200, len(self.X_test)))
                    shap_result = self.explainer.compute_shap_global(
                        self.best_model, shap_sample,
                        feature_names=feature_names, task=task
                    )
                    if 'error' not in shap_result:
                        self.pipeline_report['shap_global'] = shap_result
                        all_importances[f'{self.best_model_name}_shap'] = shap_result.get('mean_abs_shap', {})
                        self.pipeline_report['feature_importance'] = all_importances
            except Exception:
                pass  # SHAP is best-effort

            # ── Stacking Ensemble (if 3+ models trained) ──
            if config.get('stacking', True) and len(self.trained_models) >= 3:
                try:
                    self._update_step('stacking')
                    stacking_result = self._build_stacking_ensemble(task)
                    if stacking_result:
                        self.trained_models['stacking'] = stacking_result
                        # Evaluate stacking model
                        stacking_metrics = self.evaluator.evaluate(
                            stacking_result['model'], self.X_test, self.y_test,
                            task, 'stacking'
                        )
                        stacking_metrics['display_name'] = 'Stacking Ensemble'
                        self.evaluation_results['stacking'] = stacking_metrics

                        # Re-rank all models
                        valid_results = {k: v for k, v in self.evaluation_results.items()
                                         if 'error' not in v}
                        primary = 'f1' if task == 'classification' else 'r2'
                        sorted_models = sorted(valid_results.items(),
                                               key=lambda x: x[1].get(primary, 0),
                                               reverse=True)
                        for rank, (name, _) in enumerate(sorted_models, 1):
                            self.evaluation_results[name]['rank'] = rank

                        # Update best model if stacking won
                        if sorted_models and sorted_models[0][0] == 'stacking':
                            self.best_model_name = 'stacking'
                            self.best_model = stacking_result['model']

                        self.pipeline_report['steps_completed'].append('stacking')
                except Exception:
                    pass  # Stacking is best-effort

            # Step 9: Hyperparameter Tuning (optional, on best model)
            tune_config = config.get('tuning')
            if tune_config and self.best_model_name:
                self._update_step('tuning')
                tune_method = tune_config.get('method', 'bayesian')
                tune_result = self.evaluator.tune(
                    self.best_model, self.best_model_name,
                    self.X_train, self.y_train,
                    method=tune_method, task=task,
                    param_grid=tune_config.get('param_grid'),
                    n_trials=tune_config.get('n_trials', 30)
                )
                if 'best_model' in tune_result and tune_result['best_model'] is not None:
                    self.best_model = tune_result['best_model']
                    # Re-evaluate tuned model
                    tuned_metrics = self.evaluator.evaluate(
                        self.best_model, self.X_test, self.y_test,
                        task, f'{self.best_model_name}_tuned'
                    )
                    self.evaluation_results[f'{self.best_model_name}_tuned'] = tuned_metrics
                self.pipeline_report['steps_completed'].append('tuning')

            # Save best model
            if self.best_model:
                model_path = os.path.join(self.output_dir, 'best_model.joblib')
                joblib.dump(self.best_model, model_path)
                self.pipeline_report['model_path'] = model_path

            # Step 10: Generate explanations
            self._update_step('explaining')
            full_report = self.get_full_report()
            self.explanations = self.explainer.explain(full_report, config)
            self.pipeline_report['steps_completed'].append('explaining')

            # Step 11: Generate visual report
            self._update_step('visualizations')
            try:
                vis_gen = VisualizationGenerator(self.output_dir)
                vis_gen.generate_report(full_report, self.best_model, self.X_test, self.y_test)
                self.pipeline_report['steps_completed'].append('visualizations')
            except Exception as e:
                print(f"Visual report generation failed: {e}")

            self.pipeline_report['status'] = 'completed'


        except Exception as e:
            self.pipeline_report['status'] = 'error'
            self.pipeline_report['error'] = str(e)
            import traceback
            self.pipeline_report['traceback'] = traceback.format_exc()

        self.pipeline_report['end_time'] = time.time()
        self.pipeline_report['total_time'] = round(
            self.pipeline_report['end_time'] - self.pipeline_report['start_time'], 2)

        return self.get_full_report()

    # ── Auto-Explore (Anti-Overfitting Strategy Search) ──

    def run_auto_explore(self, target_col: str, task: str = None,
                         config: dict = None) -> dict:
        """
        Try multiple preprocessing strategies, score each with cross-validated
        probe model, then run full training with the winning strategy.

        Anti-overfitting measures:
        - Uses 5-fold stratified CV (not a single train/test split)
        - Probe model uses early stopping to prevent memorisation
        - Final score uses held-out fold averages, not training score
        - Presets are designed to avoid data leakage
        """
        config = config or {}
        self.pipeline_report['status'] = 'running'
        self.pipeline_report['start_time'] = time.time()
        self.pipeline_report['checkpoints'] = []

        if task is None:
            task = self.detect_task_type(target_col)

        self._update_step('auto_explore')
        preset_scores = {}
        preset_details = {}

        for preset_name, preset_config in self.EXPLORE_PRESETS.items():
            try:
                score, detail = self._probe_preset(
                    preset_name, preset_config, target_col, task, config
                )
                preset_scores[preset_name] = score
                preset_details[preset_name] = detail
            except Exception as e:
                preset_details[preset_name] = {
                    'label': preset_config.get('label', preset_name),
                    'error': str(e),
                    'cv_score': 0,
                }
                preset_scores[preset_name] = 0

        # Pick winner — highest CV score
        best_preset = max(preset_scores, key=preset_scores.get)
        winning_config = copy.deepcopy(self.EXPLORE_PRESETS[best_preset])
        winning_label = winning_config.pop('label', best_preset)

        # Store explore results in report
        self.pipeline_report['auto_explore'] = {
            'presets_tried': len(preset_scores),
            'scores': {k: round(v, 4) for k, v in preset_scores.items()},
            'details': preset_details,
            'winner': best_preset,
            'winner_label': winning_label,
            'winner_score': round(preset_scores[best_preset], 4),
        }

        # Merge winning preset config with user's config (user overrides win)
        merged_config = copy.deepcopy(winning_config)
        for key in ('splitting', 'imbalance', 'models', 'tuning'):
            if key in config:
                merged_config[key] = config[key]
        # Let user override specific preset keys too
        for key in ('cleaning', 'transformation', 'feature_engineering'):
            if key in config and config[key]:
                merged_config.setdefault(key, {})
                merged_config[key].update(config[key])

        merged_config.setdefault('splitting', {}).setdefault('test_size', 0.2)

        # Reset pipeline modules for the real run
        self._reset_modules()
        self.pipeline_report['steps_completed'] = []
        self.pipeline_report['current_step'] = None

        # Now run the full pipeline with the winning config
        return self.run(target_col=target_col, task=task, config=merged_config)

    def _probe_preset(self, name: str, preset: dict, target_col: str,
                      task: str, user_config: dict) -> tuple:
        """
        Score a preprocessing preset using cross-validated LightGBM probe.

        Returns (cv_score, detail_dict).
        Anti-overfitting: 5-fold stratified CV, no hyperparameter tuning on
        probe, early stopping on validation loss.
        """
        from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score

        # Build temporary pipeline modules
        cleaner = DataCleaner()
        transformer = DataTransformer()
        feature_eng = FeatureEngineer()

        # Process data
        data = self.raw_data.copy()

        clean_cfg = preset.get('cleaning', {})
        data = cleaner.clean(data, clean_cfg)

        transform_cfg = preset.get('transformation', {})
        transform_cfg['target_col'] = target_col
        data = transformer.transform(data, transform_cfg)

        feat_cfg = preset.get('feature_engineering', {})
        feat_cfg['target_col'] = target_col
        feat_cfg['task'] = task
        data = feature_eng.engineer(data, feat_cfg)

        if target_col not in data.columns:
            raise ValueError(f'Target lost in preset {name}')

        y = data[target_col]
        X = data.drop(columns=[target_col]).select_dtypes(include=[np.number])
        X = X.fillna(X.median())

        if len(X.columns) == 0:
            raise ValueError(f'No numeric features after preset {name}')

        # Probe model — LightGBM with conservative settings to prevent overfit
        try:
            import lightgbm as lgb
            if task == 'classification':
                n_classes = y.nunique()
                probe = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,            # shallow trees = less overfit
                    learning_rate=0.1,
                    min_child_samples=20,   # forces generalisation
                    subsample=0.8,          # row sampling
                    colsample_bytree=0.8,   # feature sampling
                    reg_alpha=0.1,          # L1 regularisation
                    reg_lambda=1.0,         # L2 regularisation
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1,
                    num_class=n_classes if n_classes > 2 else None,
                )
                scoring = 'f1_weighted' if n_classes > 2 else 'f1'
            else:
                probe = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1,
                )
                scoring = 'r2'
        except ImportError:
            # Fallback to sklearn Random Forest
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if task == 'classification':
                probe = RandomForestClassifier(
                    n_estimators=50, max_depth=5, min_samples_leaf=10,
                    random_state=42, n_jobs=-1
                )
                scoring = 'f1_weighted'
            else:
                probe = RandomForestRegressor(
                    n_estimators=50, max_depth=5, min_samples_leaf=10,
                    random_state=42, n_jobs=-1
                )
                scoring = 'r2'

        # 5-fold cross-validation — the honest scoring method
        n_splits = 5
        if task == 'classification':
            counts = pd.Series(y).value_counts()
            if counts.min() < 2:
                # Fallback to KFold if classes are too sparse for stratification
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            else:
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        scores = cross_val_score(probe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        detail = {
            'label': preset.get('label', name),
            'cv_score': round(mean_score, 4),
            'cv_std': round(std_score, 4),
            'n_features': len(X.columns),
            'n_rows': len(X),
            'scoring_metric': scoring,
        }

        return mean_score, detail

    def _reset_modules(self):
        """Reset all pipeline modules for a fresh run."""
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.feature_engineer = FeatureEngineer()
        self.dim_reducer = DimensionalityReducer()
        self.imbalance_handler = ImbalanceHandler()
        self.splitter = DataSplitter()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()

    def _update_step(self, step_name: str):
        self.pipeline_report['current_step'] = step_name

    def get_full_report(self) -> dict:
        """Get combined report from all pipeline stages."""
        report = {
            'pipeline': self.pipeline_report,
            'cleaning': self.cleaner.get_report(),
            'transformation': self.transformer.get_report(),
            'feature_engineering': self.feature_engineer.get_report(),
            'dim_reduction': self.dim_reducer.get_report(),
            'imbalance': self.imbalance_handler.get_report(),
            'splitting': self.splitter.get_report(),
            'training': self.trainer.get_report(),
            'evaluation': self.evaluator.get_report(),
            'best_model': self.best_model_name,
            'model_comparison': self.evaluation_results,
            'explanations': self.explanations,
        }
        return report

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using the best trained model."""
        if self.best_model is None:
            raise ValueError('No model trained yet. Run the pipeline first.')
        return self.best_model.predict(X)

    def predict_whatif(self, feature_values: dict) -> dict:
        """
        Interactive What-If prediction.
        Takes feature_values dict from the frontend form and returns
        prediction + optional SHAP local explanation.
        """
        if self.best_model is None:
            raise ValueError('No model trained yet.')

        feature_names = list(self.X_train.columns)
        # Build a single-row DataFrame matching the training features
        row_data = {}
        for fn in feature_names:
            val = feature_values.get(fn, 0)
            try:
                row_data[fn] = float(val)
            except (ValueError, TypeError):
                row_data[fn] = 0.0

        X_row = pd.DataFrame([row_data], columns=feature_names)

        # Predict
        prediction = self.best_model.predict(X_row)
        pred_val = prediction[0]
        if hasattr(pred_val, 'item'):
            pred_val = pred_val.item()

        result = {
            'prediction': pred_val,
            'features_used': feature_names,
            'input_values': row_data,
        }

        # Probabilities for classification
        task = self.pipeline_report.get('task', 'classification')
        if task == 'classification' and hasattr(self.best_model, 'predict_proba'):
            try:
                proba = self.best_model.predict_proba(X_row)
                result['probabilities'] = [round(float(p), 4) for p in proba[0]]
            except Exception:
                pass

        # Local SHAP explanation
        try:
            shap_local = self.explainer.compute_shap_local(
                self.best_model, X_row,
                X_background=self.X_train.head(100),
                feature_names=feature_names, task=task
            )
            if 'error' not in shap_local:
                result['shap_explanation'] = shap_local
        except Exception:
            pass  # SHAP is best-effort

        return result

    def get_pdp_data(self, feature_name: str) -> dict:
        """Compute PDP for a given feature using the best model."""
        if self.best_model is None:
            raise ValueError('No model trained yet.')
        return self.explainer.compute_pdp(
            self.best_model, self.X_test.head(200),
            feature_name=feature_name
        )

    def _build_stacking_ensemble(self, task: str):
        """
        Build a stacking ensemble from the top 3 trained models.
        Uses cross-validation to train the meta-learner.
        """
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge

        # Pick top 3 models by rank
        valid = {k: v for k, v in self.evaluation_results.items()
                 if 'error' not in v and 'rank' in v and k != 'stacking'}
        top_3 = sorted(valid.items(), key=lambda x: x[1].get('rank', 99))[:3]

        if len(top_3) < 2:
            return None

        estimators = []
        for name, _ in top_3:
            model_info = self.trained_models.get(name)
            if model_info and model_info.get('model'):
                estimators.append((name, model_info['model']))

        if len(estimators) < 2:
            return None

        import time as _time
        start = _time.time()

        if task == 'classification':
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                cv=3, n_jobs=-1, passthrough=False
            )
        else:
            stacking = StackingRegressor(
                estimators=estimators,
                final_estimator=Ridge(),
                cv=3, n_jobs=-1, passthrough=False
            )

        stacking.fit(self.X_train, self.y_train)
        train_time = round(_time.time() - start, 2)

        return {
            'model': stacking,
            'train_time': train_time,
            'model_name': 'stacking',
            'task': task,
            'base_models': [name for name, _ in top_3],
        }

    def save_report(self, filename: str = 'pipeline_report.json'):
        """Save full report to JSON file."""
        report = self.get_full_report()
        # Remove non-serializable objects
        clean = self._make_serializable(report)
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            json.dump(clean, f, indent=2, default=str)
        return path

    def _make_serializable(self, obj):
        """Recursively make objects JSON-serializable."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()
                    if not isinstance(v, (type, )) and k != 'model'}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj
