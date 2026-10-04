"""
End-to-End API Test — verifies all 5 features work.
Run with: python test_all_features.py
"""
import requests
import json
import sys

BASE = 'http://localhost:5000'
PASS = 0
FAIL = 0

def check(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        print(f'  [PASS] {name}')
        PASS += 1
    else:
        print(f'  [FAIL] {name}: {detail}')
        FAIL += 1

def test_chat():
    print('\n[1] Chat Endpoint')
    # Help
    r = requests.post(f'{BASE}/api/chat', json={'message': 'help'})
    d = r.json()
    check('Chat returns reply', 'reply' in d, d)
    check('Help message has content', len(d.get('reply', '')) > 50, d.get('reply', '')[:80])

    # Load iris via chat
    r = requests.post(f'{BASE}/api/chat', json={'message': 'load iris dataset'})
    d = r.json()
    check('Chat load iris action', d.get('action') == 'load_sample', d)

    # Greeting
    r = requests.post(f'{BASE}/api/chat', json={'message': 'hello'})
    d = r.json()
    check('Chat greeting works', 'reply' in d and len(d.get('reply', '')) > 5, d.get('reply', ''))

    # Why question (before pipeline)
    r = requests.post(f'{BASE}/api/chat', json={'message': 'why did it win?'})
    d = r.json()
    check('Chat why (no model yet)', 'reply' in d, d)

def test_sample_load():
    print('\n[2] Sample Dataset Load')
    r = requests.post(f'{BASE}/api/load-sample', json={'dataset': 'iris'})
    d = r.json()
    check('Load iris success', 'rows' in d, d.get('error', ''))
    check('Iris has 150 rows', d.get('rows') == 150, f'rows={d.get("rows")}')
    check('Iris columns present', d.get('n_columns') == 5, f'n_columns={d.get("n_columns")}')
    check('Head data present', len(d.get('head', [])) > 0)

def test_data_preview():
    print('\n[3] Data Preview')
    r = requests.get(f'{BASE}/api/data/preview')
    d = r.json()
    check('Preview has rows', 'rows' in d, d.get('error', ''))
    check('Preview has columns', 'columns' in d)

def test_columns():
    print('\n[4] Column Info')
    r = requests.get(f'{BASE}/api/data/columns')
    d = r.json()
    check('Columns returned', 'columns' in d, d.get('error', ''))
    check('5 columns', len(d.get('columns', [])) == 5, f'{len(d.get("columns", []))}')

def test_pipeline_run():
    print('\n[5] Pipeline Run (standard)')
    config = {
        'target_col': 'species',
        'task': 'classification',
        'auto_explore': False,
        'cleaning': {'imputation': 'auto', 'outlier_method': 'iqr', 'remove_duplicates': True},
        'transformation': {'encoding': 'auto', 'scaling': 'standard'},
        'feature_engineering': {'time_features': False, 'polynomial': False},
        'splitting': {'method': 'stratified', 'test_size': 0.2},
        'models': ['random_forest', 'logistic_regression'],
    }
    r = requests.post(f'{BASE}/api/pipeline/run', json=config, timeout=120)
    d = r.json()
    check('Pipeline completed', d.get('pipeline', {}).get('status') == 'completed',
          d.get('pipeline', {}).get('error', d.get('error', '')))
    check('Best model exists', d.get('best_model') is not None, f'best={d.get("best_model")}')
    check('Model comparison exists', len(d.get('model_comparison', {})) > 0)

    # Checkpoints
    checks = d.get('pipeline', {}).get('checkpoints', [])
    check('Checkpoints saved', len(checks) >= 3, f'{len(checks)} checkpoints')

    # Explanations
    exp = d.get('explanations', {})
    check('Explanations generated', len(exp) > 0)
    check('Data insights present', len(exp.get('data', [])) > 0)
    check('Preprocessing insights', len(exp.get('preprocessing', [])) > 0)
    check('Best model insights', len(exp.get('best_model', [])) > 0)
    check('Overfitting check', len(exp.get('overfitting_check', [])) > 0)
    check('Suggestions present', len(exp.get('suggestions', [])) > 0)

    return d

def test_explain(report):
    print('\n[6] Explain Endpoint')
    r = requests.get(f'{BASE}/api/explain')
    d = r.json()
    check('Explain returns data', 'data' in d, d.get('error', ''))
    check('Explain has suggestions', 'suggestions' in d)

def test_model_compare():
    print('\n[7] Model Comparison')
    r = requests.get(f'{BASE}/api/models/compare')
    d = r.json()
    check('Compare returns models', 'models' in d, d.get('error', ''))
    check('Multiple models', len(d.get('models', [])) >= 2, f'{len(d.get("models", []))}')

def test_chat_with_results():
    print('\n[8] Chat with Results')
    # Best model
    r = requests.post(f'{BASE}/api/chat', json={'message': 'what is the best model?'})
    d = r.json()
    check('Chat knows best model', 'reply' in d and len(d.get('reply', '')) > 5, d.get('reply', '')[:80])

    # Suggestions
    r = requests.post(f'{BASE}/api/chat', json={'message': 'any suggestions?'})
    d = r.json()
    check('Chat gives suggestions', len(d.get('reply', '')) > 20, d.get('reply', '')[:80])

    # Why
    r = requests.post(f'{BASE}/api/chat', json={'message': 'why did the best model win?'})
    d = r.json()
    check('Chat explains why', len(d.get('reply', '')) > 20, d.get('reply', '')[:80])

def test_auto_explore():
    print('\n[9] Auto-Explore Mode')
    # Re-load iris fresh
    requests.post(f'{BASE}/api/load-sample', json={'dataset': 'iris'})

    config = {
        'target_col': 'species',
        'task': 'classification',
        'auto_explore': True,
        'models': ['random_forest', 'logistic_regression'],
    }
    r = requests.post(f'{BASE}/api/pipeline/run', json=config, timeout=300)
    d = r.json()
    check('Auto-explore completed', d.get('pipeline', {}).get('status') == 'completed',
          d.get('pipeline', {}).get('error', d.get('error', '')))

    explore = d.get('pipeline', {}).get('auto_explore', {})
    check('Explore tried presets', explore.get('presets_tried', 0) >= 3,
          f'tried={explore.get("presets_tried")}')
    check('Explore has winner', explore.get('winner') is not None,
          f'winner={explore.get("winner")}')
    check('Explore scores exist', len(explore.get('scores', {})) > 0)
    check('Winner score > 0', (explore.get('winner_score', 0) or 0) > 0,
          f'score={explore.get("winner_score")}')

    # Check that the full pipeline ran after
    check('Best model after explore', d.get('best_model') is not None)
    check('Explanations after explore', len(d.get('explanations', {})) > 0)

if __name__ == '__main__':
    print('=' * 50)
    print('EasyML \u2014 Feature Verification Test')
    print('=' * 50)

    test_chat()
    test_sample_load()
    test_data_preview()
    test_columns()
    report = test_pipeline_run()
    test_explain(report)
    test_model_compare()
    test_chat_with_results()
    test_auto_explore()

    print(f'\n{"=" * 50}')
    print(f'Results: {PASS} passed, {FAIL} failed')
    print(f'{"=" * 50}')

    sys.exit(1 if FAIL > 0 else 0)
