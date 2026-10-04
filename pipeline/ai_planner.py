"""
AI Pipeline Planner
-------------------
Uses Gemini to analyse a data profile and automatically recommend 
an optimal machine learning pipeline configuration.
"""

import os
import json
import logging
import traceback

logger = logging.getLogger(__name__)

HAS_GEMINI = False
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    pass


class PipelinePlanner:
    """Uses LLM to recommend pipeline configuration."""

    def __init__(self):
        self.model = None
        api_key = os.environ.get("GOOGLE_API_KEY")
        if HAS_GEMINI and api_key:
            genai.configure(api_key=api_key)
            # Use gemini-2.5-flash with JSON output mode
            self.model = genai.GenerativeModel(
                'gemini-2.5-flash',
                generation_config={"response_mime_type": "application/json"}
            )

    def plan_pipeline(self, data_profile: dict, target_col: str, task: str) -> dict:
        """
        Analyse the data profile and return a recommended pipeline configuration.
        """
        if not self.model:
            return {
                "error": "Gemini API key not configured or generativeai package missing.",
                "config": None,
                "reasoning": None
            }

        prompt = self._build_prompt(data_profile, target_col, task)
        
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text)
            return {
                "config": result.get("config", {}),
                "reasoning": result.get("reasoning", "No reasoning provided.")
            }
        except Exception as e:
            logger.error(f"Failed to plan pipeline: {e}\n{traceback.format_exc()}")
            return {
                "error": f"Failed to generate pipeline plan: {str(e)}",
                "config": None,
                "reasoning": None
            }

    def _build_prompt(self, profile: dict, target_col: str, task: str) -> str:
        """Construct the prompt for Gemini."""
        
        # We only need a summary of the profile to save tokens and focus the model
        overview = profile.get('overview', {})
        missing = profile.get('missing', {})
        alerts = profile.get('alerts', [])
        
        columns_summary = []
        for col in profile.get('columns', []):
            columns_summary.append({
                "name": col.get("name"),
                "type": col.get("type"),
                "unique": col.get("unique"),
                "missing": col.get("missing")
            })

        prompt_data = {
            "task": task,
            "target_column": target_col,
            "dataset_overview": overview,
            "missing_values_summary": {
                "total": missing.get("total"),
                "columns_with_missing": missing.get("columns_with_missing")
            },
            "columns": columns_summary,
            "data_quality_alerts": [a.get("message") for a in alerts]
        }
        
        prompt = f"""You are an expert Data Scientist. Your task is to design an optimal Machine Learning pipeline for the provided dataset profile.
        
dataset_profile = {json.dumps(prompt_data, indent=2)}

Based on this profile, generate a JSON object with two keys: "config" and "reasoning".

1. "config": A dictionary representing the optimal pipeline configuration. Use the following schema:
{{
    "target_col": "{target_col}",
    "task": "{task}",
    "cleaning": {{
        "missing_values": "mean", // Options: "auto", "mean", "median", "mode", "knn", "drop", "none"
        "outliers": "iqr" // Options: "iqr", "zscore", "isolation_forest", "none"
    }},
    "transformation": {{
        "encoding": "auto", // Options: "auto", "onehot", "label", "none"
        "scaling": "standard" // Options: "standard", "minmax", "robust", "none"
    }},
    "feature_engineering": {{
        "polynomial": false, // boolean
        "interactions": false, // boolean
        "selection": "none" // Options: "mutual_info", "tree", "correlation", "rfe", "lasso", "none"
    }},
    "imbalance": {{
        "method": "none" // Options: "smote", "adasyn", "none". ONLY use if task is classification and data might be imbalanced.
    }},
    "models": ["random_forest", "xgboost"], // Array of strings. Classification options: "random_forest", "xgboost", "lightgbm", "catboost", "logistic_regression", "decision_tree", "svm", "mlp". Regression options: "random_forest", "xgboost", "lightgbm", "catboost", "linear_regression", "decision_tree", "svr", "mlp".
    "tuning": {{
        "enable": false // boolean. Set to true if hyperparameter tuning is necessary.
    }}
}}

Choose the best options based on the dataset size, missing values, column types, and data quality alerts. 
For example, if there are outliers, you might choose "robust" scaling and "iqr" for outliers. 
If it's a classification task, select classification models.
Keep the list of models to the top 2-4 best choices for this dataset.

2. "reasoning": A brief, user-friendly markdown string (2-4 sentences) explaining why you chose this specific configuration.
"""
        return prompt
