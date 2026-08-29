import json
import time
import shutil
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, ParameterGrid, cross_val_score
from xgboost import XGBClassifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
KEY = 'isolet'
TIME_BUDGET_HOURS = 11.5
total_start = time.time()

def budget_left():
    return (time.time() - total_start) < TIME_BUDGET_HOURS * 3600

_hits = sorted(Path('/kaggle/input').rglob(f'{KEY}_splits.npz'))
assert _hits, "attach the s10 output (new_prepared_data)"
DATA_DIR = _hits[0].parent

OUTPUT_DIR = Path('/kaggle/working/new_tuning_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
out_json = OUTPUT_DIR / f'{KEY}_best_hyperparameters.json'

prior_json = sorted(Path('/kaggle/input').rglob(f'{KEY}_best_hyperparameters.json'))
if prior_json:
    shutil.copy(prior_json[0], out_json)
    src = sorted(Path('/kaggle/input').rglob(f'{KEY}_xgb_grid_search_results.csv'))
    if src:
        shutil.copy(src[0], OUTPUT_DIR / f'{KEY}_xgb_grid_search_results.csv')
    raise SystemExit("Already complete, restored from attached input.")

try:
    tm = XGBClassifier(tree_method='hist', device='cuda', n_estimators=2)
    tm.fit(np.random.rand(100, 10), np.random.randint(0, 3, 100))
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

data = np.load(DATA_DIR / f'{KEY}_splits.npz')
X_tune, y_tune = data['X_train'], data['y_train']
with open(DATA_DIR / f'{KEY}_metadata.json') as f:
    n_classes = int(json.load(f)['n_classes'])
tune_size = len(y_tune)

RF_EMBEDDED = {'best_params': {'max_depth': 25, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}, 'best_cv_score': 0.9347, 'tuning_time_minutes': 13.7, 'note': 'completed in 13a run of 2026-08-13, values from its log'}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

xgb_param_grid = {'n_estimators': [200, 400, 600], 'max_depth': [6, 8, 10], 'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0]}
xgb_base_params = {'random_state': RANDOM_SEED, 'tree_method': 'hist', 'eval_metric': 'mlogloss', 'n_jobs': -1, 'verbosity': 0}
if GPU_AVAILABLE:
    xgb_base_params['device'] = 'cuda'

ckpt_csv = OUTPUT_DIR / f'{KEY}_xgb_grid_checkpoint.csv'
prior_ckpt = sorted(Path('/kaggle/input').rglob(f'{KEY}_xgb_grid_checkpoint.csv'))
if prior_ckpt:
    shutil.copy(prior_ckpt[0], ckpt_csv)
done_rows = pd.read_csv(ckpt_csv).to_dict('records') if ckpt_csv.exists() else []
done_keys = set(r['params'] for r in done_rows)

configs = list(ParameterGrid(xgb_param_grid))
xgb_grid_start = time.time()
for params in configs:
    pkey = json.dumps(params, sort_keys=True)
    if pkey in done_keys:
        continue
    if not budget_left():
        raise SystemExit("TIME BUDGET REACHED during XGB grid. Rerun with this version's output attached to resume.")
    t0 = time.time()
    model = XGBClassifier(**{**xgb_base_params, **params})
    scores = cross_val_score(model, X_tune, y_tune, cv=cv, scoring='f1_macro', n_jobs=1)
    dt = time.time() - t0
    done_rows.append({'params': pkey, 'mean_test_score': float(scores.mean()), 'std_test_score': float(scores.std()), 'mean_fit_time': float(dt / 3)})
    done_keys.add(pkey)
    pd.DataFrame(done_rows).to_csv(ckpt_csv, index=False)
    gc.collect()

xgb_elapsed = time.time() - xgb_grid_start
xgb_df = pd.DataFrame(done_rows).sort_values('mean_test_score', ascending=False).reset_index(drop=True)
xgb_df['rank_test_score'] = range(1, len(xgb_df) + 1)
xgb_df.to_csv(OUTPUT_DIR / f'{KEY}_xgb_grid_search_results.csv', index=False)
best_xgb_params = json.loads(xgb_df.iloc[0]['params'])
best_xgb_score = float(xgb_df.iloc[0]['mean_test_score'])

best_rf_params = RF_EMBEDDED['best_params']
et_params = {'n_estimators': best_rf_params['n_estimators'], 'max_depth': best_rf_params['max_depth'], 'min_samples_split': best_rf_params['min_samples_split'], 'min_samples_leaf': best_rf_params['min_samples_leaf'], 'max_features': 'sqrt', 'n_jobs': -1, 'random_state': RANDOM_SEED, 'class_weight': 'balanced_subsample'}
meta_learner_params = {'C': 1.0, 'max_iter': 1000, 'multi_class': 'multinomial', 'solver': 'lbfgs', 'random_state': RANDOM_SEED}
xgb_for_stack = best_xgb_params.copy()
xgb_for_stack.update(xgb_base_params)
xgb_for_stack.pop('device', None)

stage_json = OUTPUT_DIR / f'{KEY}_stage_checkpoint.json'
prior_stage = sorted(Path('/kaggle/input').rglob(f'{KEY}_stage_checkpoint.json'))
if prior_stage and not stage_json.exists():
    shutil.copy(prior_stage[0], stage_json)
stage = json.load(open(stage_json)) if stage_json.exists() else {}

if 'stacking' not in stage:
    assert budget_left(), "budget reached before stacking; rerun to resume"
    t0 = time.time()
    stacking_clf = StackingClassifier(estimators=[('rf', RandomForestClassifier(**best_rf_params, n_jobs=-1, random_state=RANDOM_SEED, class_weight='balanced_subsample')), ('xgb', XGBClassifier(**xgb_for_stack)), ('et', ExtraTreesClassifier(**et_params))], final_estimator=LogisticRegression(**meta_learner_params), cv=5, n_jobs=1, passthrough=False)
    s = cross_val_score(stacking_clf, X_tune, y_tune, cv=cv, scoring='f1_macro', n_jobs=1)
    stage['stacking'] = {'mean': float(s.mean()), 'std': float(s.std()), 'minutes': float((time.time() - t0) / 60)}
    json.dump(stage, open(stage_json, 'w'))

if 'voting' not in stage:
    assert budget_left(), "budget reached before voting; rerun to resume"
    t0 = time.time()
    voting_clf = VotingClassifier(estimators=[('rf', RandomForestClassifier(**best_rf_params, n_jobs=-1, random_state=RANDOM_SEED, class_weight='balanced_subsample')), ('xgb', XGBClassifier(**xgb_for_stack)), ('et', ExtraTreesClassifier(**et_params))], voting='soft', n_jobs=1)
    v = cross_val_score(voting_clf, X_tune, y_tune, cv=cv, scoring='f1_macro', n_jobs=1)
    stage['voting'] = {'mean': float(v.mean()), 'std': float(v.std()), 'minutes': float((time.time() - t0) / 60)}
    json.dump(stage, open(stage_json, 'w'))

best_hyperparameters = {'metadata': {'dataset': KEY, 'tuning_sample_size': int(tune_size), 'n_classes': n_classes, 'cv_folds': 3, 'scoring': 'f1_macro', 'random_seed': RANDOM_SEED, 'gpu_used_for_xgb': GPU_AVAILABLE, 'noise_level_for_tuning': 0.0}, 'random_forest': RF_EMBEDDED, 'xgboost': {'best_params': best_xgb_params, 'best_cv_score': best_xgb_score, 'tuning_time_minutes': float(xgb_elapsed / 60), 'base_params': xgb_base_params}, 'extra_trees': {'params': et_params, 'derived_from': 'random_forest_best_params + sqrt features'}, 'stacking': {'base_learners': ['rf', 'xgb', 'et'], 'meta_learner_params': meta_learner_params, 'cv_folds_internal': 5, 'cv_f1_macro_mean': stage['stacking']['mean'], 'cv_f1_macro_std': stage['stacking']['std'], 'validation_time_minutes': stage['stacking']['minutes']}, 'voting': {'base_learners': ['rf', 'xgb', 'et'], 'voting_type': 'soft', 'weights': None, 'cv_f1_macro_mean': stage['voting']['mean'], 'cv_f1_macro_std': stage['voting']['std'], 'validation_time_minutes': stage['voting']['minutes']}}
with open(out_json, 'w') as f:
    json.dump(best_hyperparameters, f, indent=2, default=str)
