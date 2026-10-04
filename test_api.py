"""Verify pipeline output."""
import requests
import json

BASE = 'http://localhost:5000'

# Load sample
r = requests.post(f'{BASE}/api/load-sample', json={'dataset': 'iris'})
print(f"LOAD: {r.status_code}")

# Run pipeline
config = {
    'target_col': 'species',
    'task': 'classification',
    'cleaning': {'imputation': 'auto', 'outlier_method': 'none', 'remove_duplicates': True},
    'transformation': {'encoding': 'auto', 'scaling': 'standard', 'target_col': 'species'},
    'feature_engineering': {'target_col': 'species', 'task': 'classification'},
    'splitting': {'method': 'stratified', 'test_size': 0.2},
    'imbalance': {'method': 'none'},
    'models': ['logistic_regression', 'random_forest', 'decision_tree']
}

r2 = requests.post(f'{BASE}/api/pipeline/run', json=config)
result = r2.json()

status = result.get('pipeline', {}).get('status')
print(f"STATUS: {status}")
print(f"BEST: {result.get('best_model')}")
print(f"TIME: {result.get('pipeline', {}).get('total_time')}s")

if status == 'error':
    print(f"ERROR: {result.get('pipeline', {}).get('error')}")

for name, m in result.get('model_comparison', {}).items():
    if 'error' in m:
        print(f"  {name}: ERROR")
    else:
        print(f"  {name}: f1={m.get('f1')} acc={m.get('accuracy')} rank={m.get('rank')}")

print("DONE")
