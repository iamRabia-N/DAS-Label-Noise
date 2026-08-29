import json
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from s02_noise_injection import inject_symmetric_noise, inject_asymmetric_noise, NOISE_LEVELS

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURES_DIR = Path('/kaggle/input/datasets/rabianaz22/4a-image-features-3rd-paper/image_features')
TUNING_DIR = Path('/kaggle/input/datasets/rabianaz22/4b-image-tuning-3rd-paper/image_tuning_results')
OUTPUT_DIR = Path('/kaggle/working/rf_image_results')
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

assert FEATURES_DIR.exists()
assert TUNING_DIR.exists()

with open(TUNING_DIR / 'best_hyperparameters_image.json', 'r') as f:
    tuned = json.load(f)

best_rf_params = tuned['random_forest']['best_params']
if best_rf_params.get('max_depth') == 'None' or best_rf_params.get('max_depth') is None:
    best_rf_params['max_depth'] = None

data = np.load(FEATURES_DIR / 'eurosat_features_train.npz')
X_train = data['X']
y_train = data['y']
test_data = np.load(FEATURES_DIR / 'eurosat_features_test.npz')
X_test = test_data['X']
y_test = test_data['y']
n_classes = 10


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


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
NOISE_TYPES = ['symmetric', 'asymmetric']

checkpoint_file = CHECKPOINT_DIR / 'rf_image_progress.json'
results_file = CHECKPOINT_DIR / 'rf_image_results_intermediate.csv'

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

    X_train_fold, X_val_fold, y_train_fold_clean, y_val_fold = get_fold_data(
        X_train, y_train, fold_idx, cv
    )

    noise_seed = RANDOM_SEED + fold_idx * 100 + int(noise_level * 1000)
    y_train_fold_noisy, noise_mask = inject_noise(
        y_train_fold_clean, noise_type, noise_level, n_classes, noise_seed
    )

    rf = RandomForestClassifier(
        **best_rf_params,
        n_jobs=-1,
        random_state=RANDOM_SEED + fold_idx,
        class_weight='balanced_subsample'
    )
    rf.fit(X_train_fold, y_train_fold_noisy)

    y_pred = rf.predict(X_val_fold)
    metrics = compute_metrics(y_val_fold, y_pred, n_classes)

    run_time = time.time() - run_start
    run_result = {
        'model': 'RandomForest_Image',
        'noise_type': noise_type,
        'noise_level': noise_level,
        'fold': fold_idx,
        'train_size': int(len(y_train_fold_noisy)),
        'val_size': int(len(y_val_fold)),
        'actual_noise_pct': float(noise_mask.mean() * 100),
        'runtime_seconds': float(run_time),
        **metrics
    }
    all_results.append(run_result)
    completed_runs.add((noise_type, noise_level, fold_idx))

    n_completed_this_session += 1
    if n_completed_this_session % 5 == 0:
        pd.DataFrame(all_results).to_csv(results_file, index=False)
        with open(checkpoint_file, 'w') as f:
            json.dump({'completed': [list(r) for r in completed_runs]}, f)

    del rf, X_train_fold, X_val_fold, y_train_fold_clean, y_train_fold_noisy, y_val_fold, y_pred
    gc.collect()

pd.DataFrame(all_results).to_csv(results_file, index=False)
with open(checkpoint_file, 'w') as f:
    json.dump({'completed': [list(r) for r in completed_runs]}, f)

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR / 'rf_image_results_full.csv', index=False)

agg_metrics = ['accuracy', 'f1_macro', 'f1_weighted']
summary = results_df.groupby(['noise_type', 'noise_level'])[agg_metrics].agg(['mean', 'std']).reset_index()
summary.columns = ['_'.join(col).rstrip('_') for col in summary.columns.values]
summary = summary.sort_values(['noise_type', 'noise_level']).reset_index(drop=True)
summary.to_csv(OUTPUT_DIR / 'rf_image_summary.csv', index=False)

per_class_cols = [f'f1_class_{c}' for c in range(n_classes)]
per_class_summary = results_df.groupby(['noise_type', 'noise_level'])[per_class_cols].mean().reset_index()
per_class_summary.to_csv(OUTPUT_DIR / 'rf_image_per_class_metrics.csv', index=False)

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
        'model': 'RandomForest_Image',
        'noise_type': noise_type,
        'f1_clean': float(f1_clean),
        'f1_at_40pct_noise': float(f1_at_max),
        'total_drop': float(total_drop),
        'nsi_slope': float(slope),
        'relative_drop_pct': float(total_drop / f1_clean * 100)
    })

nsi_df = pd.DataFrame(nsi_results)
nsi_df.to_csv(OUTPUT_DIR / 'rf_image_nsi.csv', index=False)

rf_final = RandomForestClassifier(
    **best_rf_params,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    class_weight='balanced_subsample'
)
rf_final.fit(X_train, y_train)
y_test_pred = rf_final.predict(X_test)
test_metrics = compute_metrics(y_test, y_test_pred, n_classes)

test_result_record = {
    'model': 'RandomForest_Image',
    'condition': 'clean_train_full_test',
    **test_metrics
}
pd.DataFrame([test_result_record]).to_csv(OUTPUT_DIR / 'rf_image_test_results.csv', index=False)
