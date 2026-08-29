import json
import time
import gc
import numpy as np
import pandas as pd
import psutil
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from xgboost import XGBClassifier
from s02_noise_injection import inject_symmetric_noise, inject_asymmetric_noise, NOISE_LEVELS

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = Path('/kaggle/input/datasets/rabianaz22/1-ensemble-noise-prepared-data-3rd-paper/prepared_data')
TUNING_DIR = Path('/kaggle/input/datasets/rabianaz22/3a-tabular-tuning-3rd-paper/tuning_results')
OUTPUT_DIR = Path('/kaggle/working/stacking_results')
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

assert DATA_DIR.exists()
assert TUNING_DIR.exists()

with open(TUNING_DIR / 'best_hyperparameters.json', 'r') as f:
    tuned = json.load(f)

best_rf_params = tuned['random_forest']['best_params'].copy()
if best_rf_params.get('max_depth') == 'None' or best_rf_params.get('max_depth') is None:
    best_rf_params['max_depth'] = None

best_xgb_params = tuned['xgboost']['best_params'].copy()
xgb_base_params = tuned['xgboost']['base_params'].copy()

et_params = tuned['extra_trees']['params'].copy()
if et_params.get('max_depth') == 'None' or et_params.get('max_depth') is None:
    et_params['max_depth'] = None

meta_learner_params = tuned['stacking']['meta_learner_params']

STACKING_INTERNAL_CV = 3
SUBSAMPLE_SIZE = 100000

data = np.load(DATA_DIR / 'covtype_splits.npz')
X_train_full = data['X_train']
y_train_full = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
n_classes = 7

X_train, _, y_train, _ = train_test_split(
    X_train_full, y_train_full,
    train_size=SUBSAMPLE_SIZE,
    stratify=y_train_full,
    random_state=RANDOM_SEED
)

del X_train_full, y_train_full
gc.collect()


def get_ram():
    return psutil.Process().memory_info().rss / 1e9


def compute_metrics(y_true, y_pred, n_classes):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    metrics = {'accuracy': float(acc), 'f1_macro': float(f1_macro), 'f1_weighted': float(f1_weighted)}
    for c in range(n_classes):
        metrics[f'precision_class_{c}'] = float(precision[c])
        metrics[f'recall_class_{c}'] = float(recall[c])
        metrics[f'f1_class_{c}'] = float(f1_per_class[c])
        metrics[f'support_class_{c}'] = int(support[c])
    return metrics


def get_fold_data(X, y, fold_idx, cv):
    splits = list(cv.split(X, y))
    train_idx, val_idx = splits[fold_idx]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def inject_noise(y_train_fold, noise_type, noise_rate, n_classes, random_state):
    if noise_rate == 0.0:
        return y_train_fold.copy(), np.zeros(len(y_train_fold), dtype=bool)
    if noise_type == 'symmetric':
        return inject_symmetric_noise(y_train_fold, noise_rate, n_classes, random_state)
    elif noise_type == 'asymmetric':
        return inject_asymmetric_noise(y_train_fold, noise_rate, n_classes, random_state=random_state)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


def build_stacking_classifier(rf_params, xgb_params, xgb_base, et_params, meta_params,
                              random_state, internal_cv):
    rf = RandomForestClassifier(
        **rf_params, n_jobs=-1, random_state=random_state, class_weight='balanced_subsample'
    )
    xgb_cpu_params = {**xgb_base, **xgb_params, 'random_state': random_state}
    xgb_cpu_params.pop('device', None)
    xgb = XGBClassifier(**xgb_cpu_params)
    et = ExtraTreesClassifier(**et_params)
    et.random_state = random_state
    meta = LogisticRegression(**meta_params)
    stacking = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('et', et)],
        final_estimator=meta,
        cv=internal_cv,
        n_jobs=1,
        passthrough=False
    )
    return stacking


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
NOISE_TYPES = ['symmetric', 'asymmetric']

checkpoint_file = CHECKPOINT_DIR / 'stacking_progress.json'
results_file = CHECKPOINT_DIR / 'stacking_results_intermediate.csv'

if checkpoint_file.exists() and results_file.exists():
    with open(checkpoint_file, 'r') as f:
        progress = json.load(f)
    completed_runs = set(tuple(r) for r in progress['completed'])
    all_results = pd.read_csv(results_file).to_dict('records')
else:
    completed_runs = set()
    all_results = []

all_runs = []
for noise_type in NOISE_TYPES:
    for noise_level in NOISE_LEVELS:
        for fold_idx in range(5):
            all_runs.append((noise_type, noise_level, fold_idx))

remaining_runs = [r for r in all_runs if r not in completed_runs]

n_completed_this_session = 0
for noise_type, noise_level, fold_idx in remaining_runs:
    run_start = time.time()
    stacking_clf = None

    try:
        X_train_fold, X_val_fold, y_train_fold_clean, y_val_fold = get_fold_data(
            X_train, y_train, fold_idx, cv
        )

        noise_seed = RANDOM_SEED + fold_idx * 100 + int(noise_level * 1000)
        y_train_fold_noisy, noise_mask = inject_noise(
            y_train_fold_clean, noise_type, noise_level, n_classes, noise_seed
        )

        stacking_clf = build_stacking_classifier(
            best_rf_params, best_xgb_params, xgb_base_params, et_params, meta_learner_params,
            random_state=RANDOM_SEED + fold_idx,
            internal_cv=STACKING_INTERNAL_CV
        )
        stacking_clf.fit(X_train_fold, y_train_fold_noisy)

        y_pred = stacking_clf.predict(X_val_fold)
        metrics = compute_metrics(y_val_fold, y_pred, n_classes)

        run_time = time.time() - run_start
        run_result = {
            'model': 'Stacking',
            'noise_type': noise_type,
            'noise_level': noise_level,
            'fold': fold_idx,
            'train_size': int(len(y_train_fold_noisy)),
            'val_size': int(len(y_val_fold)),
            'actual_noise_pct': float(noise_mask.mean() * 100),
            'runtime_seconds': float(run_time),
            'ram_gb': float(get_ram()),
            **metrics
        }
        all_results.append(run_result)
        completed_runs.add((noise_type, noise_level, fold_idx))

    except Exception:
        pass

    try:
        if stacking_clf is not None:
            del stacking_clf
    except Exception:
        pass
    try:
        del X_train_fold, X_val_fold, y_train_fold_clean, y_train_fold_noisy, y_val_fold, y_pred
    except Exception:
        pass
    for _ in range(3):
        gc.collect()

    n_completed_this_session += 1
    if n_completed_this_session % 3 == 0:
        pd.DataFrame(all_results).to_csv(results_file, index=False)
        with open(checkpoint_file, 'w') as f:
            json.dump({'completed': [list(r) for r in completed_runs]}, f)

pd.DataFrame(all_results).to_csv(results_file, index=False)
with open(checkpoint_file, 'w') as f:
    json.dump({'completed': [list(r) for r in completed_runs]}, f)

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR / 'stacking_results_full.csv', index=False)

agg_metrics = ['accuracy', 'f1_macro', 'f1_weighted']
summary = results_df.groupby(['noise_type', 'noise_level'])[agg_metrics].agg(['mean', 'std']).reset_index()
summary.columns = ['_'.join(col).rstrip('_') for col in summary.columns.values]
summary = summary.sort_values(['noise_type', 'noise_level']).reset_index(drop=True)
summary.to_csv(OUTPUT_DIR / 'stacking_summary.csv', index=False)

per_class_cols = [f'f1_class_{c}' for c in range(n_classes)]
per_class_summary = results_df.groupby(['noise_type', 'noise_level'])[per_class_cols].mean().reset_index()
per_class_summary.to_csv(OUTPUT_DIR / 'stacking_per_class_metrics.csv', index=False)

nsi_results = []
for noise_type in NOISE_TYPES:
    subset = summary[summary['noise_type'] == noise_type].sort_values('noise_level')
    f1_clean = subset[subset['noise_level'] == 0.0]['f1_macro_mean'].values[0]
    noise_levels_array = subset['noise_level'].values
    f1_values = subset['f1_macro_mean'].values
    slope = np.polyfit(noise_levels_array, f1_values, 1)[0]
    f1_at_max = f1_values[-1]
    total_drop = f1_clean - f1_at_max
    nsi_results.append({
        'model': 'Stacking',
        'noise_type': noise_type,
        'f1_clean': float(f1_clean),
        'f1_at_40pct_noise': float(f1_at_max),
        'total_drop': float(total_drop),
        'nsi_slope': float(slope),
        'relative_drop_pct': float(total_drop / f1_clean * 100)
    })

nsi_df = pd.DataFrame(nsi_results)
nsi_df.to_csv(OUTPUT_DIR / 'stacking_nsi.csv', index=False)

stacking_final = build_stacking_classifier(
    best_rf_params, best_xgb_params, xgb_base_params, et_params, meta_learner_params,
    random_state=RANDOM_SEED,
    internal_cv=STACKING_INTERNAL_CV
)
stacking_final.fit(X_train, y_train)
y_test_pred = stacking_final.predict(X_test)
test_metrics = compute_metrics(y_test, y_test_pred, n_classes)

pd.DataFrame([{
    'model': 'Stacking',
    'condition': 'clean_subsample_train_full_test',
    'train_size': SUBSAMPLE_SIZE,
    **test_metrics
}]).to_csv(OUTPUT_DIR / 'stacking_test_results.csv', index=False)
