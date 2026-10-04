"""
Pipeline Explainer — Explainable AI Report Generator
-------------------------------------------------------
Reads the full pipeline report and generates human-readable explanations
of every decision the pipeline made and why.

Enhanced with:
  • SHAP-based global & local feature explanations
  • Partial Dependence Plots (PDP)
  • Natural-language pipeline design reasoning
"""

import numpy as np
import warnings
import json
import logging
import os
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

logger = logging.getLogger(__name__)

class PipelineExplainer:
    """Generate human-readable explanations for pipeline decisions."""

    def __init__(self):
        self.explanations = []
        self._shap_values = None
        self._shap_expected = None
        self._shap_feature_names = None
        
        # Initialize Gemini for Failure Analysis if API key is present
        self.model = None
        api_key = os.environ.get("GOOGLE_API_KEY")
        if HAS_GEMINI and api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')

    def explain(self, full_report: dict, config: dict = None) -> dict:
        """
        Analyse the full pipeline report and produce explanations.

        Returns dict with sections: data, preprocessing, models, best_model,
        feature_importance, pipeline_reasoning, suggestions, overfitting_check.
        """
        config = config or {}
        self.explanations = []

        sections = {
            'data': self._explain_data(full_report),
            'preprocessing': self._explain_preprocessing(full_report, config),
            'models': self._explain_models(full_report),
            'best_model': self._explain_best_model(full_report),
            'feature_importance': self._explain_features(full_report),
            'pipeline_reasoning': self._explain_pipeline_reasoning(full_report, config),
            'suggestions': self._generate_suggestions(full_report, config),
            'overfitting_check': self._check_overfitting(full_report),
        }
        return sections

    # ══════════════════════════════════════════════════════
    # SHAP-Based Explanations
    # ══════════════════════════════════════════════════════

    def compute_shap_global(self, model, X_sample, feature_names=None, task='classification'):
        """
        Compute SHAP values for global feature importance.
        Returns dict with 'mean_abs_shap' per feature and raw 'shap_values'.
        """
        try:
            import shap
        except ImportError:
            return {'error': 'shap library not installed. pip install shap'}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                # Try TreeExplainer first (fast, for tree models)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            except Exception:
                try:
                    # Fallback to LinearExplainer
                    explainer = shap.LinearExplainer(model, X_sample)
                    shap_values = explainer.shap_values(X_sample)
                except Exception:
                    try:
                        # Last resort: KernelExplainer (slow but universal)
                        # Use a small background sample for speed
                        bg = shap.sample(X_sample, min(50, len(X_sample)))
                        explainer = shap.KernelExplainer(model.predict, bg)
                        shap_values = explainer.shap_values(X_sample.iloc[:50])
                    except Exception as e:
                        return {'error': f'SHAP computation failed: {str(e)}'}

        # Handle multi-class: shap_values is a list of arrays
        if isinstance(shap_values, list):
            # Average absolute SHAP across classes
            shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_abs = np.abs(shap_values)

        mean_shap = np.mean(shap_abs, axis=0)

        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(mean_shap))]

        # Build sorted importance dict
        importance = {}
        for name, val in sorted(zip(feature_names, mean_shap),
                                key=lambda x: x[1], reverse=True):
            importance[name] = round(float(val), 6)

        # Cache for local explanations
        self._shap_values = shap_values
        self._shap_expected = explainer.expected_value
        self._shap_feature_names = list(feature_names)

        return {
            'mean_abs_shap': importance,
            'top_features': list(importance.keys())[:10],
            'method': type(explainer).__name__,
        }

    def compute_shap_local(self, model, X_row, X_background=None,
                           feature_names=None, task='classification'):
        """
        Compute SHAP values for a single prediction (local explanation).
        Returns per-feature contribution dict for the "What-If" predictor.
        """
        try:
            import shap
        except ImportError:
            return {'error': 'shap library not installed'}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_row)
                expected = explainer.expected_value
            except Exception:
                try:
                    explainer = shap.LinearExplainer(model, X_background)
                    shap_vals = explainer.shap_values(X_row)
                    expected = explainer.expected_value
                except Exception:
                    try:
                        bg = shap.sample(X_background, min(50, len(X_background)))
                        explainer = shap.KernelExplainer(model.predict, bg)
                        shap_vals = explainer.shap_values(X_row)
                        expected = explainer.expected_value
                    except Exception as e:
                        return {'error': f'Local SHAP failed: {str(e)}'}

        # Handle multi-class
        if isinstance(shap_vals, list):
            # Pick class with highest probability for the explanation
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_row)
                pred_class = int(np.argmax(proba[0]))
            else:
                pred_class = 0
            sv = shap_vals[pred_class][0]
            base = expected[pred_class] if isinstance(expected, (list, np.ndarray)) else expected
        else:
            sv = shap_vals[0] if shap_vals.ndim > 1 else shap_vals
            base = expected

        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(sv))]

        # Build waterfall data
        contributions = []
        for name, val in zip(feature_names, sv):
            contributions.append({
                'feature': name,
                'contribution': round(float(val), 6),
                'direction': 'positive' if val > 0 else 'negative',
            })

        # Sort by |contribution| descending
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)

        prediction = model.predict(X_row)
        pred_val = prediction[0]
        if hasattr(pred_val, 'item'):
            pred_val = pred_val.item()

        return {
            'base_value': round(float(base), 6),
            'prediction': pred_val,
            'contributions': contributions[:15],  # top 15
        }

    def compute_pdp(self, model, X, feature_name, feature_idx=None, grid_size=50):
        """
        Compute Partial Dependence Plot data for a single feature.
        Returns { values: [...], predictions: [...] }
        """
        import pandas as pd

        if feature_idx is None:
            if isinstance(X, pd.DataFrame):
                feature_idx = list(X.columns).index(feature_name)
            else:
                feature_idx = 0

        if isinstance(X, pd.DataFrame):
            col_data = X.iloc[:, feature_idx]
        else:
            col_data = X[:, feature_idx]

        grid = np.linspace(float(col_data.min()), float(col_data.max()), grid_size)
        predictions = []

        for val in grid:
            X_temp = X.copy()
            if isinstance(X_temp, pd.DataFrame):
                X_temp.iloc[:, feature_idx] = val
            else:
                X_temp[:, feature_idx] = val

            preds = model.predict(X_temp)
            predictions.append(float(np.mean(preds)))

        return {
            'feature': feature_name,
            'values': [round(float(v), 4) for v in grid],
            'predictions': [round(float(p), 4) for p in predictions],
        }

    # ══════════════════════════════════════════════════════
    # Pipeline Design Reasoning (WHY the AI did what it did)
    # ══════════════════════════════════════════════════════

    def _explain_pipeline_reasoning(self, report: dict, config: dict) -> list:
        """
        Generate natural-language explanations for WHY the pipeline
        was configured the way it was. Analyses data characteristics
        and maps them to the preprocessing/model choices.
        """
        insights = []
        pipeline = report.get('pipeline', {})
        cleaning = report.get('cleaning', {})
        transformation = report.get('transformation', {})
        feat_eng = report.get('feature_engineering', {})
        task = pipeline.get('task', 'classification')
        target = pipeline.get('target_col', '?')

        # ── Why this task type? ──
        insights.append(
            f"**Task Detection:** The pipeline identified this as a "
            f"**{task}** problem. "
            + (
                f"Since `{target}` is a categorical/low-cardinality column, "
                f"predicting discrete class labels (classification) is appropriate."
                if task == 'classification' else
                f"Since `{target}` contains continuous numeric values, "
                f"predicting a number (regression) is the right approach."
            )
        )

        # ── Why was this imputation chosen? ──
        mv = cleaning.get('missing_values', {})
        strategy = mv.get('strategy', 'none')
        cols_filled = mv.get('columns_filled', {})
        if cols_filled:
            total_missing = sum(v.get('count', 0) for v in cols_filled.values())
            pct = round(total_missing / max(pipeline.get('total_rows', 1), 1) * 100, 1)
            if strategy == 'auto':
                insights.append(
                    f"**Imputation Choice:** Auto-mode was used because the dataset had "
                    f"**{total_missing}** missing values (~{pct}%). The pipeline used "
                    f"*mean* for numeric columns and *mode* for categorical columns. "
                    f"This is safer than dropping rows when missingness is below 30%."
                )
            elif strategy == 'median':
                insights.append(
                    f"**Imputation Choice:** Median imputation was chosen because it is "
                    f"*robust to outliers*. With {total_missing} missing values, "
                    f"mean imputation could be skewed by extreme values."
                )
            elif strategy == 'knn':
                insights.append(
                    f"**Imputation Choice:** KNN imputation was used to leverage "
                    f"*relationships between features* to fill missing values. "
                    f"This produces more realistic imputations than simple statistics."
                )

        # ── Why this outlier method? ──
        outliers = cleaning.get('outliers', {})
        method = outliers.get('method', 'none')
        total_handled = outliers.get('total_handled', 0)
        if method != 'none' and total_handled > 0:
            method_reasons = {
                'iqr': (
                    f"**Outlier Handling:** IQR-based clipping was used because it "
                    f"detected **{total_handled}** outlier values. IQR is a *non-parametric* "
                    f"method that doesn't assume a normal distribution, making it "
                    f"reliable across different data shapes."
                ),
                'zscore': (
                    f"**Outlier Handling:** Z-score clipping was used to detect "
                    f"**{total_handled}** values that deviated significantly from the "
                    f"mean. This works well when data is approximately normally distributed."
                ),
                'isolation_forest': (
                    f"**Outlier Handling:** Isolation Forest (ML-based) was chosen because "
                    f"it can detect multi-dimensional outliers that simple statistical "
                    f"methods would miss."
                ),
            }
            insights.append(method_reasons.get(method,
                f"**Outlier Handling:** Used *{method}* to handle {total_handled} outliers."))

        # ── Why this scaling? ──
        scaling = transformation.get('scaling', {})
        scale_method = scaling.get('method', 'none')
        if scale_method and scale_method != 'none':
            scaling_reasons = {
                'standard': (
                    "**Scaling Decision:** Standard scaling (z-score) was applied to "
                    "normalize features to zero mean and unit variance. This is critical for "
                    "distance-based models (SVM, KNN) and regularised models (Logistic "
                    "Regression) where feature magnitudes affect the outcome."
                ),
                'minmax': (
                    "**Scaling Decision:** Min-Max scaling (0–1) was applied. This bounded "
                    "transformation preserves the original distribution shape while ensuring "
                    "all features contribute equally — ideal for neural networks."
                ),
                'robust': (
                    "**Scaling Decision:** Robust scaling was chosen specifically because "
                    "your data contains outliers. Unlike Standard scaling (which uses "
                    "mean/std), Robust scaling uses *median* and *IQR*, preventing "
                    "outliers from dominating the normalisation."
                ),
            }
            insights.append(scaling_reasons.get(scale_method,
                f"**Scaling Decision:** Applied *{scale_method}* scaling."))

        # ── Why these features? ──
        selected = feat_eng.get('selected', {})
        for method_name, info in selected.items():
            n_selected = info.get('count', len(info.get('selected', [])))
            method_reasons = {
                'mutual_info': (
                    f"**Feature Selection:** Mutual Information kept **{n_selected}** "
                    f"features. MI measures the *information gain* each feature provides "
                    f"about the target — features with zero MI are pure noise."
                ),
                'tree': (
                    f"**Feature Selection:** Tree-based importance kept **{n_selected}** "
                    f"features. This method uses a Random Forest to determine which "
                    f"features contribute most to *splitting decisions* — a direct "
                    f"measure of predictive power."
                ),
                'correlation': (
                    f"**Feature Selection:** Correlation-based selection kept "
                    f"**{n_selected}** features, removing those with low correlation "
                    f"to the target and high inter-correlation with each other."
                ),
                'rfe': (
                    f"**Feature Selection:** Recursive Feature Elimination (RFE) kept "
                    f"**{n_selected}** features by iteratively training a model and "
                    f"removing the least important feature each round."
                ),
                'lasso': (
                    f"**Feature Selection:** Lasso (L1) regularisation was used to "
                    f"shrink unimportant feature coefficients to exactly zero, "
                    f"keeping **{n_selected}** non-zero features."
                ),
            }
            insights.append(method_reasons.get(method_name,
                f"**Feature Selection:** {method_name} kept {n_selected} features."))

        # ── Why did the best model win? (detailed reasoning) ──
        best_name = report.get('best_model')
        comparison = report.get('model_comparison', {})
        if best_name and len(comparison) > 1:
            best_metrics = comparison.get(best_name, {})
            other_models = {k: v for k, v in comparison.items()
                           if k != best_name and 'error' not in v}

            if other_models:
                primary = 'f1' if task == 'classification' else 'r2'
                best_score = best_metrics.get(primary, 0)
                avg_others = np.mean([m.get(primary, 0) for m in other_models.values()])

                if isinstance(best_score, (int, float)) and isinstance(avg_others, (int, float)):
                    lead = round(best_score - avg_others, 4)
                    display = best_metrics.get('display_name', best_name)
                    insights.append(
                        f"**Model Selection Reasoning:** **{display}** won because it scored "
                        f"**{round(best_score, 4)}** on {primary.upper()}, which is "
                        f"**{lead}** higher than the average of all other models "
                        f"({round(avg_others, 4)}). "
                        f"The best model was selected using a composite ranking that "
                        f"considers multiple metrics, not just a single score."
                    )

        # ── Auto-Explore reasoning ──
        explore = pipeline.get('auto_explore', {})
        if explore:
            winner = explore.get('winner', '?')
            winner_label = explore.get('winner_label', winner)
            winner_score = explore.get('winner_score', 0)
            scores = explore.get('scores', {})

            if scores:
                all_scores = list(scores.values())
                worst_score = min(all_scores)
                improvement = round(winner_score - worst_score, 4)

                insights.append(
                    f"**Auto-Explore Result:** The pipeline tried **{len(scores)}** different "
                    f"preprocessing strategies using 5-fold cross-validation. "
                    f"**{winner_label}** won with a CV score of **{winner_score}**, "
                    f"which was **{improvement}** higher than the worst strategy. "
                    f"This demonstrates that preprocessing choices can significantly "
                    f"impact model performance."
                )

        if not insights:
            insights.append(
                "The pipeline used default settings with no special preprocessing. "
                "Consider enabling Auto-Explore to find the optimal strategy."
            )

        return insights

    # ── Data Quality ──

    def _explain_data(self, report: dict) -> list:
        """Explain data quality observations."""
        insights = []
        pipeline = report.get('pipeline', {})
        cleaning = report.get('cleaning', {})

        # Row / column info
        target = pipeline.get('target_col', '?')
        task = pipeline.get('task', '?')
        insights.append(
            f"Task detected as **{task}** for target column `{target}`."
        )

        # Missing values
        mv = cleaning.get('missing_values', {})
        strategy = mv.get('strategy', 'none')
        cols_filled = mv.get('columns_filled', {})
        if cols_filled:
            total = sum(v.get('count', 0) for v in cols_filled.values())
            insights.append(
                f"Found **{total}** missing values across "
                f"**{len(cols_filled)}** columns. "
                f"Imputed using *{strategy}* strategy."
            )
            # Flag heavy missingness
            for col, info in cols_filled.items():
                if info.get('count', 0) > 50:
                    insights.append(
                        f"⚠️ Column `{col}` had {info['count']} missing values "
                        f"— consider whether this column is reliable."
                    )
        elif strategy == 'drop':
            dropped = mv.get('rows_dropped', 0)
            if dropped:
                insights.append(
                    f"Dropped **{dropped}** rows with missing values."
                )
        else:
            insights.append("No missing values detected — data is clean. ✅")

        # Duplicates
        dups = cleaning.get('duplicates', {})
        removed = dups.get('duplicates_removed', 0)
        if removed > 0:
            pct = round(removed / max(dups.get('total_before', 1), 1) * 100, 1)
            insights.append(
                f"Removed **{removed}** duplicate rows ({pct}% of data)."
            )
        else:
            insights.append("No duplicates found. ✅")

        # Outliers
        outliers = cleaning.get('outliers', {})
        total_handled = outliers.get('total_handled', 0)
        if total_handled > 0:
            method = outliers.get('method', '?')
            action = outliers.get('action', '?')
            insights.append(
                f"Handled **{total_handled}** outlier values using "
                f"*{method}* detection with *{action}* action."
            )

        return insights

    # ── Preprocessing Decisions ──

    def _explain_preprocessing(self, report: dict, config: dict) -> list:
        insights = []
        transformation = report.get('transformation', {})
        feat_eng = report.get('feature_engineering', {})

        # Encoding
        encoding = transformation.get('encoding', {})
        if encoding:
            methods_used = set()
            for col, info in encoding.items():
                m = info.get('method', 'unknown')
                methods_used.add(m)
                if m == 'skipped':
                    insights.append(
                        f"Column `{col}` skipped for encoding: "
                        f"{info.get('reason', 'too many categories')}."
                    )
            if 'onehot' in methods_used:
                insights.append(
                    "Used **one-hot encoding** for low-cardinality categorical "
                    "columns (≤ 10 unique values) — this preserves category "
                    "independence without imposing ordinal relationships."
                )
            if 'label' in methods_used:
                insights.append(
                    "Used **label encoding** for high-cardinality categorical "
                    "columns (> 10 unique values) — one-hot would create too "
                    "many sparse features."
                )

        # Scaling
        scaling = transformation.get('scaling', {})
        method = scaling.get('method', 'none')
        if method and method != 'none':
            reasons = {
                'standard': (
                    "**Standard scaling** (z-score) was applied — this centres "
                    "features around 0 with unit variance. Important for "
                    "distance-based models (SVM, KNN) and regularised models "
                    "(Logistic Regression, Lasso)."
                ),
                'minmax': (
                    "**Min-Max scaling** (0–1) was applied — keeps values "
                    "bounded. Good for neural networks and algorithms sensitive "
                    "to feature magnitude."
                ),
                'robust': (
                    "**Robust scaling** was applied — uses median/IQR instead "
                    "of mean/std. Chosen because it's resistant to outliers."
                ),
            }
            insights.append(reasons.get(method, f"Scaling: {method}"))

        # Feature engineering
        created = feat_eng.get('created', {})
        if created.get('polynomial', {}).get('count', 0) > 0:
            n = created['polynomial']['count']
            insights.append(
                f"Created **{n}** polynomial features — these capture "
                f"non-linear relationships between variables."
            )
        if created.get('interactions', {}).get('count', 0) > 0:
            n = created['interactions']['count']
            insights.append(
                f"Created **{n}** interaction features — these capture "
                f"combined effects of feature pairs."
            )

        # Feature selection
        selected = feat_eng.get('selected', {})
        for method_name, info in selected.items():
            n = info.get('count', len(info.get('selected', [])))
            insights.append(
                f"Feature selection via **{method_name}** kept "
                f"**{n}** most informative features."
            )

        if not insights:
            insights.append("Default preprocessing was applied with no special "
                            "feature engineering.")

        return insights

    # ── Model Explanations ──

    def _explain_models(self, report: dict) -> list:
        insights = []
        comparison = report.get('model_comparison', {})

        if not comparison:
            return ["No models were evaluated."]

        # Sort by rank
        ranked = sorted(
            [(name, m) for name, m in comparison.items() if 'rank' in m],
            key=lambda x: x[1].get('rank', 99)
        )

        task = report.get('pipeline', {}).get('task', 'classification')
        primary_metric = 'f1' if task == 'classification' else 'r2'

        for name, metrics in ranked:
            display = metrics.get('display_name', name)
            rank = metrics.get('rank', '?')
            score = metrics.get(primary_metric, metrics.get('accuracy', '?'))
            train_time = metrics.get('train_time', 0)

            if isinstance(score, (int, float)):
                score = round(score, 4)
            if isinstance(train_time, (int, float)):
                train_time = round(train_time, 2)

            insights.append(
                f"**#{rank} {display}**: {primary_metric}={score}, "
                f"trained in {train_time}s"
            )

        return insights

    def _explain_best_model(self, report: dict) -> list:
        insights = []
        best_name = report.get('best_model')
        comparison = report.get('model_comparison', {})
        task = report.get('pipeline', {}).get('task', 'classification')

        if not best_name or not comparison:
            return ["No best model determined."]

        best = comparison.get(best_name, {})
        primary_metric = 'f1' if task == 'classification' else 'r2'
        best_score = best.get(primary_metric, best.get('accuracy', 0))

        display = best.get('display_name', best_name)
        insights.append(f"🏆 **{display}** was selected as the best model.")

        # Why it won — compare against runner-up
        ranked = sorted(
            [(n, m) for n, m in comparison.items()
             if 'rank' in m and 'error' not in m],
            key=lambda x: x[1].get('rank', 99)
        )

        if len(ranked) >= 2:
            runner_name, runner = ranked[1]
            runner_score = runner.get(primary_metric, runner.get('accuracy', 0))
            runner_display = runner.get('display_name', runner_name)

            if isinstance(best_score, (int, float)) and isinstance(runner_score, (int, float)):
                margin = round(best_score - runner_score, 4)
                pct_margin = round(margin / max(abs(runner_score), 0.0001) * 100, 1)
                insights.append(
                    f"It beat **{runner_display}** by "
                    f"**{margin}** ({pct_margin}%) on {primary_metric}."
                )

        # Model-specific reasoning
        model_reasons = {
            'random_forest': (
                "Random Forest works well on structured/tabular data by "
                "averaging many decision trees, reducing variance and "
                "overfitting risk."
            ),
            'xgboost': (
                "XGBoost excels at tabular data through gradient boosting "
                "with built-in regularisation, handling missing values, "
                "and feature interactions."
            ),
            'lightgbm': (
                "LightGBM is a fast gradient boosting framework that uses "
                "leaf-wise growth for better accuracy on complex patterns."
            ),
            'catboost': (
                "CatBoost handles categorical features natively and uses "
                "ordered boosting to reduce prediction shift (overfitting)."
            ),
            'logistic_regression': (
                "Logistic Regression is a simple linear model — winning "
                "suggests the decision boundary is approximately linear."
            ),
            'decision_tree': (
                "A single Decision Tree winning suggests the data has clear, "
                "axis-aligned decision boundaries. Watch for overfitting. ⚠️"
            ),
            'svm': (
                "SVM found a good separating hyperplane — works best when "
                "the number of features is large relative to samples."
            ),
            'mlp': (
                "MLP Neural Network winning suggests complex non-linear "
                "patterns in the data that simpler models couldn't capture."
            ),
            'stacking': (
                "The Stacking Ensemble won by combining the strengths of "
                "multiple base models through a meta-learner, typically "
                "achieving higher accuracy than any single model alone."
            ),
        }

        reason = model_reasons.get(best_name, '')
        if reason:
            insights.append(reason)

        # Training time commentary
        best_time = best.get('train_time', 0)
        if isinstance(best_time, (int, float)) and best_time > 30:
            insights.append(
                f"⏱️ Training took {round(best_time, 1)}s — for production, "
                f"consider a faster alternative if accuracy is similar."
            )

        return insights

    # ── Feature Importance ──

    def _explain_features(self, report: dict) -> list:
        insights = []
        feat_report = report.get('feature_engineering', {})
        selected = feat_report.get('selected', {})

        # Look for importance scores
        for method, info in selected.items():
            scores = info.get('scores') or info.get('importances', {})
            if scores:
                top_5 = list(scores.items())[:5]
                features_str = ', '.join(
                    f"`{name}` ({round(score, 3)})"
                    for name, score in top_5
                )
                insights.append(
                    f"Top features by **{method}**: {features_str}"
                )

        # Also extract from pipeline report
        pipeline = report.get('pipeline', {})
        all_imp = pipeline.get('feature_importance', {})
        best = report.get('best_model')
        if best and best in all_imp:
            imp = all_imp[best]
            top_5 = list(imp.items())[:5]
            if top_5:
                features_str = ', '.join(
                    f"`{name}` ({round(score, 4)})"
                    for name, score in top_5
                )
                insights.append(
                    f"Top features for best model (**{best}**): {features_str}"
                )

        if not insights:
            insights.append(
                "Feature importance was not explicitly computed. Consider "
                "enabling feature selection (mutual info, tree importance) "
                "to understand which features drive predictions."
            )

        return insights

    # ── Overfitting Check ──

    def _check_overfitting(self, report: dict) -> list:
        """Check for signs of overfitting in the results."""
        insights = []
        comparison = report.get('model_comparison', {})
        task = report.get('pipeline', {}).get('task', 'classification')

        for name, metrics in comparison.items():
            if 'error' in metrics:
                continue

            if task == 'classification':
                acc = metrics.get('accuracy', 0)
                f1 = metrics.get('f1', 0)

                if isinstance(acc, (int, float)) and acc >= 0.99:
                    insights.append(
                        f"⚠️ **{name}** has {acc:.1%} accuracy — suspiciously "
                        f"perfect. This may indicate data leakage or "
                        f"overfitting. Verify with cross-validation."
                    )
                if isinstance(f1, (int, float)) and isinstance(acc, (int, float)):
                    if abs(acc - f1) > 0.15 and acc > 0:
                        insights.append(
                            f"⚠️ **{name}** has a large gap between accuracy "
                            f"({acc:.3f}) and F1 ({f1:.3f}) — this suggests "
                            f"class imbalance may be skewing results."
                        )
            else:
                r2 = metrics.get('r2', 0)
                if isinstance(r2, (int, float)) and r2 >= 0.99:
                    insights.append(
                        f"⚠️ **{name}** has R² = {r2:.4f} — near-perfect fit. "
                        f"Check for target leakage or overfitting."
                    )
                if isinstance(r2, (int, float)) and r2 < 0:
                    insights.append(
                        f"❌ **{name}** has negative R² ({r2:.4f}) — the model "
                        f"performs worse than predicting the mean. "
                        f"This model is not suitable for this data."
                    )

        if not insights:
            insights.append(
                "No obvious signs of overfitting detected. ✅ "
                "Metrics look reasonable for all models."
            )

        return insights

    # ── Suggestions ──

    def _generate_suggestions(self, report: dict, config: dict) -> list:
        suggestions = []
        comparison = report.get('model_comparison', {})
        pipeline = report.get('pipeline', {})
        task = pipeline.get('task', 'classification')
        steps = pipeline.get('steps_completed', [])

        # Check if tuning was done
        if 'tuning' not in steps:
            suggestions.append(
                "💡 **Hyperparameter tuning** was not used. Enable Bayesian "
                "tuning (Optuna) to potentially improve the best model's "
                "score by 2-8%."
            )

        # Check if imbalance handling was done for classification
        imbalance = report.get('imbalance', {})
        if task == 'classification' and not imbalance.get('actions'):
            suggestions.append(
                "💡 Consider enabling **imbalance handling** (SMOTE or "
                "SMOTE-ENN) if your classes are unequal — this can "
                "significantly improve recall on minority classes."
            )

        # Check for feature engineering
        feat = report.get('feature_engineering', {})
        if not feat.get('created'):
            suggestions.append(
                "💡 No feature engineering was applied. Try enabling "
                "**polynomial** or **interaction** features for potential "
                "accuracy gains."
            )

        # Model diversity
        models_trained = [n for n, m in comparison.items() if 'error' not in m]
        if len(models_trained) <= 2:
            suggestions.append(
                "💡 Only {n} model(s) were trained. Training more diverse "
                "models (ensemble + linear + tree) gives better "
                "comparison.".format(n=len(models_trained))
            )

        # Stacking suggestion
        if len(models_trained) >= 3 and 'stacking' not in models_trained:
            suggestions.append(
                "💡 You trained 3+ models. Consider enabling **Stacking Ensemble** "
                "to combine the top models into a meta-learner for potentially "
                "higher accuracy."
            )

        # Cross-validation suggestion
        eval_report = report.get('evaluation', {})
        if not eval_report.get('cross_validations'):
            suggestions.append(
                "💡 **Cross-validation** was not performed. Use 5-fold CV "
                "for more reliable performance estimates."
            )

        # Check training time
        total_time = pipeline.get('total_time', 0)
        if isinstance(total_time, (int, float)) and total_time > 120:
            suggestions.append(
                f"⏳ Pipeline took {round(total_time, 1)}s. For faster "
                f"iteration, try training fewer models or reducing data "
                f"size during exploration."
            )

        if not suggestions:
            suggestions.append(
                "✅ Pipeline looks well-configured! No major improvements "
                "to suggest."
            )

        return suggestions

    def explain_failure(self, error_traceback: str, config: dict, data_preview: str = "") -> str:
        """
        Explain why the pipeline failed and provide actionable steps to fix it.
        """
        prompt = f"""You are an expert Data Scientist and AI Assistant debugging an EasyML pipeline failure.
The pipeline crashed during execution. Analyze the error and the user's configuration to explain what went wrong.

--- User's Pipeline Configuration ---
{json.dumps(config, indent=2)}

--- Data Preview (if available) ---
{data_preview}

--- Python Error Traceback ---
{error_traceback}

--- Task ---
Provide a clear, user-friendly explanation of why the pipeline failed.
Do NOT just regurgitate the Python traceback. Explain it in plain English.
Most importantly, provide 1 to 3 bulleted, actionable steps the user can take to fix the issue (e.g., "Select a different target column", "Change the task type from Regression to Classification", "Ensure the target column does not contain text sentences").
Format your response in plain Markdown. Use bolding for emphasis. Do not use markdown headers (like # or ##) to keep it compact.
"""
        if not self.model:
            return "**Analysis Failed:** Gemini API key not configured or generativeai package missing."

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to explain failure: {e}")
            return f"**Analysis Failed:** The pipeline encountered a critical error, and the AI explainer also failed to generate a report. Raw error: `{error_traceback.splitlines()[-1] if error_traceback else str(e)}`"
