import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

class VisualizationGenerator:
    """Generates visual plots and combines them into an HTML report."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.vis_dir = os.path.join(self.output_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
    
    def _plot_to_base64(self) -> str:
        """Helper to convert current matplotlib figure to base64 string."""
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close('all')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    def generate_feature_importance(self, report: dict) -> str:
        """Generate feature importance bar plot."""
        try:
            best_model = report.get('best_model')
            if not best_model:
                return ""
            
            importances = report.get('pipeline', {}).get('feature_importance', {}).get(best_model, {})
            if not importances:
                return ""
            
            # Take top 15
            features = list(importances.keys())[:15]
            values = list(importances.values())[:15]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=values, y=features, palette='viridis')
            plt.title('Top Feature Importances')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            
            return self._plot_to_base64()
        except Exception as e:
            print(f"Error generating feature importance plot: {e}")
            return ""

    def generate_confusion_matrix(self, report: dict) -> str:
        """Generate confusion matrix heatmap for classification."""
        try:
            task = report.get('pipeline', {}).get('task')
            if task != 'classification':
                return ""
            
            best_model = report.get('best_model')
            metrics = report.get('evaluation', {}).get('evaluations', {}).get(best_model, {})
            cm = metrics.get('confusion_matrix')
            if not cm:
                return ""
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            
            return self._plot_to_base64()
        except Exception as e:
            print(f"Error generating confusion matrix plot: {e}")
            return ""

    def generate_actual_vs_predicted(self, model, X_test, y_test) -> str:
        """Generate actual vs predicted scatter plot for regression."""
        try:
            if model is None or X_test is None or y_test is None:
                return ""
                
            y_pred = model.predict(X_test)
            
            plt.figure(figsize=(8, 8))
            plt.scatter(y_test, y_pred, alpha=0.5, color='royalblue')
            
            # Identity line
            min_val = min(np.min(y_test), np.min(y_pred))
            max_val = max(np.max(y_test), np.max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            plt.title('Actual vs Predicted Values')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.grid(True, linestyle=':', alpha=0.6)
            
            return self._plot_to_base64()
        except Exception as e:
            print(f"Error generating actual vs predicted plot: {e}")
            return ""

    def generate_report(self, full_report: dict, best_model_obj=None, X_test=None, y_test=None) -> str:
        """Generate the full HTML report with embedded base64 images."""
        task = full_report.get('pipeline', {}).get('task', 'Unknown Task')
        best_model_name = full_report.get('best_model', 'None')
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>EasyML Visualization Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px auto; max-width: 900px; line-height: 1.6; color: #333; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #2980b9; margin-top: 30px; }}
                .summary-card {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #3498db; margin-bottom: 30px; }}
                .plot-container {{ text-align: center; margin-bottom: 40px; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }}
                img {{ max-width: 100%; height: auto; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <h1>EasyML Visualization Report</h1>
            
            <div class="summary-card">
                <p><strong>Task Type:</strong> {task.capitalize()}</p>
                <p><strong>Best Model:</strong> {best_model_name.replace('_', ' ').title()}</p>
            </div>
            
            <h2>Model Comparison</h2>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 30px;">
                <tr style="background-color: #3498db; color: white;">
                    <th style="padding: 10px; border: 1px solid #ddd;">Model</th>
        """
        
        # Add headers dynamically based on the best model's metrics
        model_comparison = full_report.get('model_comparison', {})
        metrics_keys = []
        if best_model_name in model_comparison and not isinstance(model_comparison[best_model_name], str):
            metrics_keys = [k for k in model_comparison[best_model_name].keys() if k not in ['confusion_matrix', 'error']]
            for m in metrics_keys:
                html_content += f'<th style="padding: 10px; border: 1px solid #ddd;">{m.capitalize()}</th>\n'
                
        html_content += "</tr>\n"
        
        for model, metrics in model_comparison.items():
            if 'error' in metrics or isinstance(metrics, str):
                continue
            is_best = "background-color: #e8f4f8; font-weight: bold;" if model == best_model_name else ""
            html_content += f'<tr style="{is_best}">\n<td style="padding: 10px; border: 1px solid #ddd;">{model.replace("_", " ").title()}</td>\n'
            for m in metrics_keys:
                val = metrics.get(m, '')
                if isinstance(val, (int, float)):
                    val = f"{val:.4f}"
                html_content += f'<td style="padding: 10px; border: 1px solid #ddd; text-align: center;">{val}</td>\n'
            html_content += "</tr>\n"
            
        html_content += """
            </table>
        """

        # Feature Importance
        fi_b64 = self.generate_feature_importance(full_report)
        if fi_b64:
            html_content += f"""
            <h2>Feature Importance</h2>
            <div class="plot-container">
                <img src="data:image/png;base64,{fi_b64}" alt="Feature Importance">
            </div>
            """
            
        # Task Specific Plots
        if task == 'classification':
            cm_b64 = self.generate_confusion_matrix(full_report)
            if cm_b64:
                html_content += f"""
                <h2>Confusion Matrix</h2>
                <div class="plot-container">
                    <img src="data:image/png;base64,{cm_b64}" alt="Confusion Matrix">
                </div>
                """
        elif task == 'regression':
            avp_b64 = self.generate_actual_vs_predicted(best_model_obj, X_test, y_test)
            if avp_b64:
                html_content += f"""
                <h2>Actual vs Predicted</h2>
                <div class="plot-container">
                    <img src="data:image/png;base64,{avp_b64}" alt="Actual vs Predicted">
                </div>
                """
                
        html_content += """
        </body>
        </html>
        """
        
        report_path = os.path.join(self.vis_dir, 'report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return report_path
