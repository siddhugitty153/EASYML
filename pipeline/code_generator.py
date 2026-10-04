"""
Code Generator — Export Pipeline as Python Script
----------------------------------------------------
Generates a standalone .py script that reproduces the pipeline
configuration, preprocessing, and model training.
"""

import json


def generate_pipeline_code(config: dict, report: dict) -> str:
    """
    Generate a standalone Python script from pipeline config and report.

    Args:
        config: pipeline configuration dict (sent by the frontend)
        report: the full pipeline report dict

    Returns:
        A string containing valid, runnable Python code.
    """
    pipeline = report.get('pipeline', {})
    target_col = pipeline.get('target_col', 'target')
    task = pipeline.get('task', 'classification')
    best_model = report.get('best_model', 'random_forest')
    features_used = pipeline.get('features_used', [])

    cleaning = config.get('cleaning', {})
    transformation = config.get('transformation', {})
    feature_eng = config.get('feature_engineering', {})
    splitting = config.get('splitting', {})
    imbalance = config.get('imbalance', {})
    tuning = config.get('tuning', {})

    # Build imports
    imports = [
        "import pandas as pd",
        "import numpy as np",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.metrics import classification_report, accuracy_score" if task == 'classification'
        else "from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error",
        "from sklearn.preprocessing import LabelEncoder",
        "import joblib",
        "import warnings",
        "warnings.filterwarnings('ignore')",
    ]

    # Model import
    model_imports = _get_model_import(best_model, task)
    imports.append(model_imports)

    # Scaling import
    scaling = transformation.get('scaling', 'none')
    if scaling and scaling != 'none':
        scaler_map = {
            'standard': 'from sklearn.preprocessing import StandardScaler',
            'minmax': 'from sklearn.preprocessing import MinMaxScaler',
            'robust': 'from sklearn.preprocessing import RobustScaler',
        }
        if scaling in scaler_map:
            imports.append(scaler_map[scaling])

    # Imbalance import
    imbalance_method = imbalance.get('method', 'none')
    if imbalance_method and imbalance_method != 'none':
        imb_map = {
            'smote': 'from imblearn.over_sampling import SMOTE',
            'adasyn': 'from imblearn.over_sampling import ADASYN',
            'tomek': 'from imblearn.under_sampling import TomekLinks',
            'enn': 'from imblearn.under_sampling import EditedNearestNeighbours',
            'smote_enn': 'from imblearn.combine import SMOTEENN',
        }
        if imbalance_method in imb_map:
            imports.append(imb_map[imbalance_method])

    # Build code sections
    code_parts = []

    # ── Header ──
    code_parts.append(f'''"""
EasyML — Generated Pipeline Script
============================================
This script reproduces the pipeline that was configured and run
in EasyML. You can modify and extend it as needed.

Target Column: {target_col}
Task Type:     {task}
Best Model:    {best_model}
"""

{chr(10).join(imports)}
''')

    # ── Data Loading ──
    code_parts.append('''
# ══════════════════════════════════════════════════
# 1. Load Data
# ══════════════════════════════════════════════════
# Replace 'your_data.csv' with the path to your dataset
df = pd.read_csv('your_data.csv')
print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
''')

    # ── Data Cleaning ──
    imputation = cleaning.get('imputation', 'auto')
    outlier_method = cleaning.get('outlier_method', 'none')
    dedup = cleaning.get('remove_duplicates', True)

    clean_code = '''
# ══════════════════════════════════════════════════
# 2. Data Cleaning
# ══════════════════════════════════════════════════
'''
    if dedup:
        clean_code += '''
# Remove duplicates
before = len(df)
df = df.drop_duplicates()
print(f"Removed {before - len(df)} duplicate rows")
'''

    if imputation == 'auto' or imputation == 'mean':
        clean_code += '''
# Fill missing values: mean for numeric, mode for categorical
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mean(), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
'''
    elif imputation == 'median':
        clean_code += '''
# Fill missing numerics with median, categoricals with mode
for col in df.select_dtypes(include=[np.number]).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
'''
    elif imputation == 'drop':
        clean_code += '''
# Drop rows with missing values
df = df.dropna()
'''

    if outlier_method == 'iqr':
        clean_code += '''
# Clip outliers using IQR method
for col in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower, upper)
'''
    elif outlier_method == 'zscore':
        clean_code += '''
# Clip outliers using Z-score method
from scipy import stats
for col in df.select_dtypes(include=[np.number]).columns:
    z = np.abs(stats.zscore(df[col].dropna()))
    threshold = 3
    median = df[col].median()
    df.loc[z > threshold, col] = median
'''

    code_parts.append(clean_code)

    # ── Encoding & Scaling ──
    encoding = transformation.get('encoding', 'auto')
    transform_code = f'''
# ══════════════════════════════════════════════════
# 3. Feature Transformation
# ══════════════════════════════════════════════════
target_col = '{target_col}'
y = df[target_col]
X = df.drop(columns=[target_col])

# Encode categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns
label_encoders = {{}}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
'''
    if task == 'classification':
        transform_code += '''
# Encode target for classification
if y.dtype == 'object' or y.dtype.name == 'category':
    target_le = LabelEncoder()
    y = pd.Series(target_le.fit_transform(y.astype(str)), name=target_col)
else:
    y = y.astype(int)
'''

    if scaling and scaling != 'none':
        scaler_class = {
            'standard': 'StandardScaler',
            'minmax': 'MinMaxScaler',
            'robust': 'RobustScaler',
        }.get(scaling, 'StandardScaler')
        transform_code += f'''
# Scale numeric features
num_cols = X.select_dtypes(include=[np.number]).columns
scaler = {scaler_class}()
X[num_cols] = scaler.fit_transform(X[num_cols])
'''
    code_parts.append(transform_code)

    # ── Splitting ──
    test_size = splitting.get('test_size', 0.2)
    split_method = splitting.get('method', 'stratified' if task == 'classification' else 'random')
    stratify_arg = 'y' if split_method == 'stratified' else 'None'

    code_parts.append(f'''
# ══════════════════════════════════════════════════
# 4. Train/Test Split
# ══════════════════════════════════════════════════
X = X.select_dtypes(include=[np.number])  # Keep only numeric
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size={test_size}, random_state=42,
    stratify={stratify_arg}
)
print(f"Train: {{X_train.shape}}, Test: {{X_test.shape}}")
''')

    # ── Imbalance ──
    if imbalance_method and imbalance_method != 'none':
        imb_class = {
            'smote': 'SMOTE',
            'adasyn': 'ADASYN',
            'tomek': 'TomekLinks',
            'enn': 'EditedNearestNeighbours',
            'smote_enn': 'SMOTEENN',
        }.get(imbalance_method, 'SMOTE')
        code_parts.append(f'''
# ══════════════════════════════════════════════════
# 5. Handle Class Imbalance
# ══════════════════════════════════════════════════
resampler = {imb_class}(random_state=42)
X_train, y_train = resampler.fit_resample(X_train, y_train)
print(f"After resampling: {{X_train.shape}}")
''')

    # ── Model Training ──
    model_code = _get_model_instantiation(best_model, task)
    code_parts.append(f'''
# ══════════════════════════════════════════════════
# 6. Train Model
# ══════════════════════════════════════════════════
{model_code}

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
''')

    # ── Evaluation ──
    if task == 'classification':
        code_parts.append('''
# ══════════════════════════════════════════════════
# 7. Evaluate
# ══════════════════════════════════════════════════
print("\\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))
''')
    else:
        code_parts.append('''
# ══════════════════════════════════════════════════
# 7. Evaluate
# ══════════════════════════════════════════════════
print("\\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE:     {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"MAE:      {mean_absolute_error(y_test, y_pred):.4f}")
''')

    # ── Save Model ──
    code_parts.append('''
# ══════════════════════════════════════════════════
# 8. Save Model
# ══════════════════════════════════════════════════
joblib.dump(model, 'trained_model.joblib')
print("\\nModel saved to trained_model.joblib")

# To load later:
# model = joblib.load('trained_model.joblib')
# predictions = model.predict(new_data)
''')

    return '\n'.join(code_parts)


def _get_model_import(model_name: str, task: str) -> str:
    """Get the import statement for a given model."""
    imports = {
        'logistic_regression': 'from sklearn.linear_model import LogisticRegression',
        'linear_regression': 'from sklearn.linear_model import LinearRegression',
        'decision_tree': 'from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor',
        'random_forest': 'from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor',
        'xgboost': 'from xgboost import XGBClassifier, XGBRegressor',
        'lightgbm': 'from lightgbm import LGBMClassifier, LGBMRegressor',
        'catboost': 'from catboost import CatBoostClassifier, CatBoostRegressor',
        'svm': 'from sklearn.svm import SVC, SVR',
        'mlp': 'from sklearn.neural_network import MLPClassifier, MLPRegressor',
        'stacking': (
            'from sklearn.ensemble import StackingClassifier, StackingRegressor, '
            'RandomForestClassifier, RandomForestRegressor\n'
            'from sklearn.linear_model import LogisticRegression, Ridge\n'
            'from xgboost import XGBClassifier, XGBRegressor\n'
            'from lightgbm import LGBMClassifier, LGBMRegressor'
        ),
    }
    return imports.get(model_name, '# Unknown model — add import manually')


def _get_model_instantiation(model_name: str, task: str) -> str:
    """Get model instantiation code."""
    is_clf = task == 'classification'

    models = {
        'logistic_regression': 'model = LogisticRegression(max_iter=1000, random_state=42)',
        'linear_regression': 'model = LinearRegression()',
        'decision_tree': (
            'model = DecisionTreeClassifier(random_state=42)' if is_clf
            else 'model = DecisionTreeRegressor(random_state=42)'
        ),
        'random_forest': (
            'model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)' if is_clf
            else 'model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)'
        ),
        'xgboost': (
            'model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42)' if is_clf
            else 'model = XGBRegressor(n_estimators=100, random_state=42)'
        ),
        'lightgbm': (
            'model = LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)' if is_clf
            else 'model = LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)'
        ),
        'catboost': (
            'model = CatBoostClassifier(iterations=100, verbose=0, random_state=42)' if is_clf
            else 'model = CatBoostRegressor(iterations=100, verbose=0, random_state=42)'
        ),
        'svm': (
            'model = SVC(kernel="rbf", random_state=42)' if is_clf
            else 'model = SVR(kernel="rbf")'
        ),
        'mlp': (
            'model = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)' if is_clf
            else 'model = MLPRegressor(hidden_layer_sizes=(100,50), max_iter=500, random_state=42)'
        ),
        'stacking': (
            '''# Stacking Ensemble — combines multiple models with a meta-learner
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42)),
    ('lgb', LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)),
]
model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5, n_jobs=-1
)''' if is_clf else
            '''# Stacking Ensemble — combines multiple models with a meta-learner
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=100, random_state=42)),
    ('lgb', LGBMRegressor(n_estimators=100, verbose=-1, random_state=42)),
]
model = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(),
    cv=5, n_jobs=-1
)'''
        ),
    }
    return models.get(model_name, f'# model = ???  (unknown model: {model_name})')
