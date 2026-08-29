import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
from xgboost import XGBClassifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = Path('/kaggle/input/datasets/rabianaz22/1-ensemble-noise-prepared-data-3rd-paper/prepared_data')
OUTPUT_DIR = Path('/kaggle/working/tuning_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
assert DATA_DIR.exists()

try:
    test_model = XGBClassifier(tree_method='hist', device='cuda', n_estimators=2)
    test_model.fit(np.random.rand(100, 10), np.random.randint(0, 3, 100))
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

data = np.load(DATA_DIR / 'covtype_splits.npz')
X_train = data['X_train']
y_train = data['y_train']
n_classes = 7

TUNE_SIZE = 80000
X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=TUNE_SIZE, stratify=y_train, random_state=RANDOM_SEED)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [15, 25, None], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2], 'max_features': ['sqrt']}
rf_base = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_SEED, class_weight='balanced_subsample')
t_start = time.time()
rf_grid = GridSearchCV(estimator=rf_base, param_grid=rf_param_grid, cv=cv, scoring='f1_macro', n_jobs=1, verbose=0, return_train_score=False)
rf_grid.fit(X_tune, y_tune)
rf_elapsed = time.time() - t_start

xgb_param_grid = {'n_estimators': [200, 400, 600], 'max_depth': [6, 8, 10], 'learning_rate': [0.05, 0.1], 'subsample': [0.8, 1.0], 'colsample_bytree': [0.8, 1.0]}
xgb_base_params = {'random_state': RANDOM_SEED, 'tree_method': 'hist', 'eval_metric': 'mlogloss', 'n_jobs': -1, 'verbosity': 0}
if GPU_AVAILABLE:
    xgb_base_params['device'] = 'cuda'
xgb_base = XGBClassifier(**xgb_base_params)
t_start = time.time()
xgb_grid = GridSearchCV(estimator=xgb_base, param_grid=xgb_param_grid, cv=cv, scoring='f1_macro', n_jobs=1, verbose=0, return_train_score=False)
xgb_grid.fit(X_tune, y_tune)
xgb_elapsed = time.time() - t_start

best_rf_params = rf_grid.best_params_.copy()
best_xgb_params = xgb_grid.best_params_.copy()
et_params = {'n_estimators': best_rf_params['n_estimators'], 'max_depth': best_rf_params['max_depth'], 'min_samples_split': best_rf_params['min_samples_split'], 'min_samples_leaf': best_rf_params['min_samples_leaf'], 'max_features': 'sqrt', 'n_jobs': -1, 'random_state': RANDOM_SEED, 'class_weight': 'balanced_subsample'}
meta_learner_params = {'C': 1.0, 'max_iter': 1000, 'multi_class': 'multinomial', 'solver': 'lbfgs', 'random_state': RANDOM_SEED}

xgb_for_stack = best_xgb_params.copy()
xgb_for_stack.update(xgb_base_params)

stacking_estimators = [('rf', RandomForestClassifier(**best_rf_params, n_jobs=-1, random_state=RANDOM_SEED, class_weight='balanced_subsample')), ('xgb', XGBClassifier(**xgb_for_stack)), ('et', ExtraTreesClassifier(**et_params))]
stacking_clf = StackingClassifier(estimators=stacking_estimators, final_estimator=LogisticRegression(**meta_learner_params), cv=5, n_jobs=1, passthrough=False)
t_start = time.time()
stack_scores = cross_val_score(stacking_clf, X_tune, y_tune, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED), scoring='f1_macro', n_jobs=1)
stack_elapsed = time.time() - t_start

voting_estimators = [('rf', RandomForestClassifier(**best_rf_params, n_jobs=-1, random_state=RANDOM_SEED, class_weight='balanced_subsample')), ('xgb', XGBClassifier(**xgb_for_stack)), ('et', ExtraTreesClassifier(**et_params))]
voting_clf = VotingClassifier(estimators=voting_estimators, voting='soft', n_jobs=1)
t_start = time.time()
voting_scores = cross_val_score(voting_clf, X_tune, y_tune, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED), scoring='f1_macro', n_jobs=1)
voting_elapsed = time.time() - t_start

best_hyperparameters = {'metadata': {'tuning_sample_size': int(TUNE_SIZE), 'cv_folds': 3, 'scoring': 'f1_macro', 'random_seed': RANDOM_SEED, 'gpu_used_for_xgb': GPU_AVAILABLE, 'noise_level_for_tuning': 0.0}, 'random_forest': {'best_params': best_rf_params, 'best_cv_score': float(rf_grid.best_score_), 'tuning_time_minutes': float(rf_elapsed / 60)}, 'xgboost': {'best_params': best_xgb_params, 'best_cv_score': float(xgb_grid.best_score_), 'tuning_time_minutes': float(xgb_elapsed / 60), 'base_params': xgb_base_params}, 'extra_trees': {'params': et_params, 'derived_from': 'random_forest_best_params + sqrt features'}, 'stacking': {'base_learners': ['rf', 'xgb', 'et'], 'meta_learner_params': meta_learner_params, 'cv_folds_internal': 5, 'cv_f1_macro_mean': float(stack_scores.mean()), 'cv_f1_macro_std': float(stack_scores.std()), 'validation_time_minutes': float(stack_elapsed / 60)}, 'voting': {'base_learners': ['rf', 'xgb', 'et'], 'voting_type': 'soft', 'weights': None, 'cv_f1_macro_mean': float(voting_scores.mean()), 'cv_f1_macro_std': float(voting_scores.std()), 'validation_time_minutes': float(voting_elapsed / 60)}}
with open(OUTPUT_DIR / 'best_hyperparameters.json', 'w') as f:
    json.dump(best_hyperparameters, f, indent=2, default=str)

rf_results_df = pd.DataFrame(rf_grid.cv_results_)[['params', 'mean_test_score', 'std_test_score', 'rank_test_score', 'mean_fit_time']].sort_values('rank_test_score').reset_index(drop=True)
rf_results_df.to_csv(OUTPUT_DIR / 'rf_grid_search_results.csv', index=False)

xgb_results_df = pd.DataFrame(xgb_grid.cv_results_)[['params', 'mean_test_score', 'std_test_score', 'rank_test_score', 'mean_fit_time']].sort_values('rank_test_score').reset_index(drop=True)
xgb_results_df.to_csv(OUTPUT_DIR / 'xgb_grid_search_results.csv', index=False)
