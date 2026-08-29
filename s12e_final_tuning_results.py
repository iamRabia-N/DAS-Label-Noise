import json
import shutil
from pathlib import Path

RANDOM_SEED = 42
OUTPUT_DIR = Path('/kaggle/working/new_tuning_final')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

XGB_BASE = {'random_state': RANDOM_SEED, 'tree_method': 'hist', 'eval_metric': 'mlogloss', 'n_jobs': -1, 'verbosity': 0, 'device': 'cuda'}
META_LR = {'C': 1.0, 'max_iter': 1000, 'multi_class': 'multinomial', 'solver': 'lbfgs', 'random_state': RANDOM_SEED}

RECOVERED = {
    'drybean': dict(tune=8166, ncls=7, rf={'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 300}, rf_f1=0.932529, rf_min=5.3, xgb={'colsample_bytree': 1.0, 'learning_rate': 0.05, 'max_depth': 10, 'n_estimators': 200, 'subsample': 0.8}, xgb_f1=0.939621, xgb_min=15.7, st=(0.9369, 0.0006, 2.6), vo=(0.9354, 0.0007, 0.5)),
    'pendigits': dict(tune=6594, ncls=10, rf={'max_depth': 25, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}, rf_f1=0.989346, rf_min=3.0, xgb={'colsample_bytree': 0.8, 'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 600, 'subsample': 0.8}, xgb_f1=0.989029, xgb_min=9.9, st=(0.9917, 0.0022, 2.1), vo=(0.9911, 0.0025, 0.4)),
    'shuttle': dict(tune=30000, ncls=4, rf={'max_depth': 25, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}, rf_f1=0.911524, rf_min=4.3, xgb={'colsample_bytree': 1.0, 'learning_rate': 0.05, 'max_depth': 8, 'n_estimators': 200, 'subsample': 0.8}, xgb_f1=0.944577, xgb_min=7.3, st=(0.9420, 0.0112, 1.8), vo=(0.9091, 0.0098, 0.4)),
    'satimage': dict(tune=3861, ncls=6, rf={'max_depth': 25, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}, rf_f1=0.888192, rf_min=2.6, xgb={'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 200, 'subsample': 0.8}, xgb_f1=0.899042, xgb_min=8.4, st=(0.8972, 0.0137, 1.4), vo=(0.8955, 0.0174, 0.2)),
    'har': dict(tune=6179, ncls=6, rf={'max_depth': 25, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}, rf_f1=0.967833, rf_min=14.9, xgb={'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 600, 'subsample': 0.8}, xgb_f1=0.986125, xgb_min=205.7, st=(0.9859, 0.0006, 19.6), vo=(0.9842, 0.0014, 3.7)),
    'fmnist': dict(tune=36000, ncls=10, rf={'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}, rf_f1=0.859510, rf_min=108.8, xgb={'colsample_bytree': 0.8, 'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 600, 'subsample': 0.8}, xgb_f1=0.893153, xgb_min=261.5, st=(0.8899, 0.0032, 128.9), vo=(0.8879, 0.0039, 25.4)),
}

for key, d in RECOVERED.items():
    et = {'n_estimators': d['rf']['n_estimators'], 'max_depth': d['rf']['max_depth'], 'min_samples_split': d['rf']['min_samples_split'], 'min_samples_leaf': d['rf']['min_samples_leaf'], 'max_features': 'sqrt', 'n_jobs': -1, 'random_state': RANDOM_SEED, 'class_weight': 'balanced_subsample'}
    h = {'metadata': {'dataset': key, 'tuning_sample_size': d['tune'], 'n_classes': d['ncls'], 'cv_folds': 3, 'scoring': 'f1_macro', 'random_seed': RANDOM_SEED, 'gpu_used_for_xgb': True, 'noise_level_for_tuning': 0.0, 'source': 'recovered from 13a/13b run logs of 2026-08-13 (sessions hit 12h limit; all values below were fully computed and printed before the cut)'}, 'random_forest': {'best_params': d['rf'], 'best_cv_score': d['rf_f1'], 'tuning_time_minutes': d['rf_min']}, 'xgboost': {'best_params': d['xgb'], 'best_cv_score': d['xgb_f1'], 'tuning_time_minutes': d['xgb_min'], 'base_params': XGB_BASE}, 'extra_trees': {'params': et, 'derived_from': 'random_forest_best_params + sqrt features'}, 'stacking': {'base_learners': ['rf', 'xgb', 'et'], 'meta_learner_params': META_LR, 'cv_folds_internal': 5, 'cv_f1_macro_mean': d['st'][0], 'cv_f1_macro_std': d['st'][1], 'validation_time_minutes': d['st'][2]}, 'voting': {'base_learners': ['rf', 'xgb', 'et'], 'voting_type': 'soft', 'weights': None, 'cv_f1_macro_mean': d['vo'][0], 'cv_f1_macro_std': d['vo'][1], 'validation_time_minutes': d['vo'][2]}}
    with open(OUTPUT_DIR / f'{key}_best_hyperparameters.json', 'w') as f:
        json.dump(h, f, indent=2, default=str)

for key in ['cifar10', 'isolet']:
    src = sorted(Path('/kaggle/input').rglob(f'{key}_best_hyperparameters.json'))
    if src:
        shutil.copy(src[0], OUTPUT_DIR / f'{key}_best_hyperparameters.json')
