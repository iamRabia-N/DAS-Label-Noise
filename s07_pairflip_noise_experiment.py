import json
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from xgboost import XGBClassifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = Path('/kaggle/input/datasets/rabianaz22/1-ensemble-noise-prepared-data-3rd-paper/prepared_data')
FEATURES_DIR = Path('/kaggle/input/datasets/rabianaz22/4a-image-features-3rd-paper/image_features')
TUNING_DIR_TAB = Path('/kaggle/input/datasets/rabianaz22/3a-tabular-tuning-3rd-paper/tuning_results')
TUNING_DIR_IMG = Path('/kaggle/input/datasets/rabianaz22/4b-image-tuning-3rd-paper/image_tuning_results')

OUTPUT_DIR = Path('/kaggle/working/pairflip_results')
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

assert DATA_DIR.exists()
assert FEATURES_DIR.exists()
assert TUNING_DIR_TAB.exists()
assert TUNING_DIR_IMG.exists()

TABULAR_PAIRFLIP_MAP = {1: 2, 2: 1, 5: 6, 6: 5}
IMAGE_PAIRFLIP_MAP = {0: 5, 5: 0, 3: 4, 4: 3, 8: 9, 9: 8}


def inject_pairflip_noise(y_true, noise_rate, transition_map, random_state):
    assert 0.0 <= noise_rate <= 1.0
    rng = np.random.RandomState(random_state)
    y_noisy = y_true.copy()
    flip_mask = rng.rand(len(y_true)) < noise_rate
    for idx in np.where(flip_mask)[0]:
        c = y_true[idx]
        if c in transition_map:
            y_noisy[idx] = transition_map[c]
    actual_flip_mask = (y_noisy != y_true)
    return y_noisy, actual_flip_mask


def compute_metrics(y_true, y_pred, n_classes):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    out = {'accuracy': float(acc), 'f1_macro': float(f1_macro), 'f1_weighted': float(f1_weighted)}
    for c in range(n_classes):
        out[f'precision_class_{c}'] = float(precision[c])
        out[f'recall_class_{c}'] = float(recall[c])
        out[f'f1_class_{c}'] = float(f1_per_class[c])
        out[f'support_class_{c}'] = int(support[c])
    return out


def get_fold(X, y, fold_idx, cv):
    splits = list(cv.split(X, y))
    train_idx, val_idx = splits[fold_idx]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


with open(TUNING_DIR_TAB / 'best_hyperparameters.json', 'r') as f:
    tab_tuned = json.load(f)
with open(TUNING_DIR_IMG / 'best_hyperparameters_image.json', 'r') as f:
    img_tuned = json.load(f)


def fix_max_depth(params):
    if params.get('max_depth') == 'None' or params.get('max_depth') is None:
        params['max_depth'] = None
    return params


tab_rf = fix_max_depth(tab_tuned['random_forest']['best_params'].copy())
tab_xgb = tab_tuned['xgboost']['best_params'].copy()
tab_xgb_base = tab_tuned['xgboost']['base_params'].copy()
tab_et = fix_max_depth(tab_tuned['extra_trees']['params'].copy())
tab_meta = tab_tuned['stacking']['meta_learner_params']

img_rf = fix_max_depth(img_tuned['random_forest']['best_params'].copy())
img_xgb = img_tuned['xgboost']['best_params'].copy()
img_xgb_base = img_tuned['xgboost']['base_params'].copy()
img_et = fix_max_depth(img_tuned['extra_trees']['params'].copy())
img_meta = img_tuned['stacking']['meta_learner_params']

try:
    test_model = XGBClassifier(tree_method='hist', device='cuda', n_estimators=2)
    test_model.fit(np.random.rand(100, 10), np.random.randint(0, 3, 100))
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


def build_rf(params, random_state):
    return RandomForestClassifier(
        **params, n_jobs=-1, random_state=random_state,
        class_weight='balanced_subsample'
    )


def build_xgb(params, base, random_state, use_gpu):
    full = {**base, **params, 'random_state': random_state}
    if use_gpu:
        full['device'] = 'cuda'
    else:
        full.pop('device', None)
    return XGBClassifier(**full)


def build_et(params, random_state):
    p = params.copy()
    p['random_state'] = random_state
    return ExtraTreesClassifier(**p)


def build_voting(rf_p, xgb_p, xgb_base, et_p, random_state, use_gpu):
    return VotingClassifier(
        estimators=[('rf', build_rf(rf_p, random_state)),
                    ('xgb', build_xgb(xgb_p, xgb_base, random_state, use_gpu)),
                    ('et', build_et(et_p, random_state))],
        voting='soft', n_jobs=1
    )


def build_stacking(rf_p, xgb_p, xgb_base, et_p, meta_p, random_state):
    return StackingClassifier(
        estimators=[('rf', build_rf(rf_p, random_state)),
                    ('xgb', build_xgb(xgb_p, xgb_base, random_state, use_gpu=False)),
                    ('et', build_et(et_p, random_state))],
        final_estimator=LogisticRegression(**meta_p),
        cv=3, n_jobs=1, passthrough=False
    )


tab_data = np.load(DATA_DIR / 'covtype_splits.npz')
X_train_tab_full = tab_data['X_train']
y_train_tab_full = tab_data['y_train']

img_train = np.load(FEATURES_DIR / 'eurosat_features_train.npz')
X_train_img = img_train['X']
y_train_img = img_train['y']

STACK_SUBSET_SIZE = 100000
X_train_tab_stack, _, y_train_tab_stack, _ = train_test_split(
    X_train_tab_full, y_train_tab_full,
    train_size=STACK_SUBSET_SIZE,
    stratify=y_train_tab_full, random_state=RANDOM_SEED
)

NOISE_RATE = 0.4
N_FOLDS = 5
N_CLASSES_TAB = 7
N_CLASSES_IMG = 10

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

runs = []
for domain in ['Tabular', 'Image']:
    for ensemble in ['RF', 'XGBoost', 'Voting', 'Stacking']:
        for fold in range(N_FOLDS):
            runs.append((domain, ensemble, fold))

checkpoint_file = CHECKPOINT_DIR / 'pairflip_progress.json'
results_file = CHECKPOINT_DIR / 'pairflip_intermediate.csv'

# Resume order: local checkpoint first, then completed results attached as an
# input dataset (rebuilds the completed set so no training is repeated).
_prior = sorted(Path('/kaggle/input').rglob('pairflip_results_full.csv'))
if checkpoint_file.exists() and results_file.exists():
    with open(checkpoint_file) as f:
        completed = set(tuple(r) for r in json.load(f)['completed'])
    all_results = pd.read_csv(results_file).to_dict('records')
elif _prior:
    all_results = pd.read_csv(_prior[0]).to_dict('records')
    completed = set((r['domain'], r['ensemble'], int(r['fold'])) for r in all_results)
else:
    completed = set()
    all_results = []

remaining = [r for r in runs if r not in completed]

session_done = 0
for domain, ensemble, fold in remaining:
    run_start = time.time()
    clf = None

    try:
        if domain == 'Tabular':
            n_classes = N_CLASSES_TAB
            pair_map = TABULAR_PAIRFLIP_MAP
            rf_p, xgb_p, xgb_base, et_p, meta_p = tab_rf, tab_xgb, tab_xgb_base, tab_et, tab_meta
            if ensemble == 'Stacking':
                X_pool, y_pool = X_train_tab_stack, y_train_tab_stack
            else:
                X_pool, y_pool = X_train_tab_full, y_train_tab_full
        else:
            n_classes = N_CLASSES_IMG
            pair_map = IMAGE_PAIRFLIP_MAP
            rf_p, xgb_p, xgb_base, et_p, meta_p = img_rf, img_xgb, img_xgb_base, img_et, img_meta
            X_pool, y_pool = X_train_img, y_train_img

        X_tr, X_val, y_tr_clean, y_val = get_fold(X_pool, y_pool, fold, cv)

        noise_seed = RANDOM_SEED + fold * 100 + int(NOISE_RATE * 1000)
        y_tr_noisy, noise_mask = inject_pairflip_noise(
            y_tr_clean, NOISE_RATE, pair_map, random_state=noise_seed
        )

        rs = RANDOM_SEED + fold
        if ensemble == 'RF':
            clf = build_rf(rf_p, rs)
        elif ensemble == 'XGBoost':
            clf = build_xgb(xgb_p, xgb_base, rs, use_gpu=GPU_AVAILABLE)
        elif ensemble == 'Voting':
            clf = build_voting(rf_p, xgb_p, xgb_base, et_p, rs, use_gpu=GPU_AVAILABLE)
        elif ensemble == 'Stacking':
            clf = build_stacking(rf_p, xgb_p, xgb_base, et_p, meta_p, rs)
        else:
            raise ValueError(f"Unknown ensemble: {ensemble}")

        clf.fit(X_tr, y_tr_noisy)
        y_pred = clf.predict(X_val)
        metrics = compute_metrics(y_val, y_pred, n_classes)

        run_time = time.time() - run_start
        all_results.append({
            'model': ensemble,
            'noise_type': 'pairflip',
            'noise_level': NOISE_RATE,
            'fold': fold,
            'domain': domain,
            'ensemble': ensemble,
            'train_size': int(len(y_tr_noisy)),
            'val_size': int(len(y_val)),
            'actual_noise_pct': float(noise_mask.mean() * 100),
            'runtime_seconds': float(run_time),
            **metrics,
        })
        completed.add((domain, ensemble, fold))

    except Exception:
        pass

    if clf is not None:
        del clf
    for _ in range(3):
        gc.collect()

    session_done += 1
    if session_done % 3 == 0:
        pd.DataFrame(all_results).to_csv(results_file, index=False)
        with open(checkpoint_file, 'w') as f:
            json.dump({'completed': [list(r) for r in completed]}, f)

pd.DataFrame(all_results).to_csv(results_file, index=False)
with open(checkpoint_file, 'w') as f:
    json.dump({'completed': [list(r) for r in completed]}, f)

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR / 'pairflip_results_full.csv', index=False)

agg = results_df.groupby(['domain', 'ensemble'])['f1_macro'].agg(['mean', 'std']).reset_index()
agg.columns = ['domain', 'ensemble', 'f1_pairflip_mean', 'f1_pairflip_std']
agg = agg.sort_values(['domain', 'ensemble']).reset_index(drop=True)
agg.to_csv(OUTPUT_DIR / 'pairflip_summary.csv', index=False)

# Comparison against cyclic asymmetric baseline: REQUIRES master CSV.
_master_candidates = [
    Path('/kaggle/input/datasets/rabianaz22/05-aggregation-output-3rd-paper/aggregated_results/master_all_results.csv'),
    Path('/kaggle/input/datasets/rabianaz22/5-aggregation-3rd-paper/aggregated_results/master_all_results.csv'),
    Path('/kaggle/working/aggregated_results/master_all_results.csv'),
]
master_csv = next((p for p in _master_candidates if p.exists()), None)
if master_csv is None:
    _hits = sorted(Path('/kaggle/input').rglob('master_all_results.csv'))
    master_csv = _hits[0] if _hits else None
assert master_csv is not None, (
    "master_all_results.csv not found. Attach the aggregation dataset (s05 output) "
    "before running the comparison and bootstrap sections."
)
master = pd.read_csv(master_csv)

cyclic_40 = master[
    (master['noise_type'] == 'asymmetric') & (master['noise_level'] == 0.4)
].groupby(['domain', 'ensemble'])['f1_macro'].agg(['mean', 'std']).reset_index()
cyclic_40.columns = ['domain', 'ensemble', 'f1_cyclic_mean', 'f1_cyclic_std']

clean = master[
    (master['noise_type'] == 'symmetric') & (master['noise_level'] == 0.0)
].groupby(['domain', 'ensemble'])['f1_macro'].agg(['mean']).reset_index()
clean.columns = ['domain', 'ensemble', 'f1_clean_mean']

cmp = agg.merge(cyclic_40, on=['domain', 'ensemble']).merge(clean, on=['domain', 'ensemble'])
cmp['rel_drop_cyclic_pct'] = (cmp['f1_clean_mean'] - cmp['f1_cyclic_mean']) / cmp['f1_clean_mean'] * 100
cmp['rel_drop_pairflip_pct'] = (cmp['f1_clean_mean'] - cmp['f1_pairflip_mean']) / cmp['f1_clean_mean'] * 100
cmp = cmp[['domain', 'ensemble', 'f1_clean_mean',
           'f1_cyclic_mean', 'rel_drop_cyclic_pct',
           'f1_pairflip_mean', 'rel_drop_pairflip_pct']].round(4)
cmp.to_csv(OUTPUT_DIR / 'cyclic_vs_pairflip_comparison.csv', index=False)

# Bootstrap CIs for pair-flip relative drops, using the true clean baselines.
N_BOOTSTRAP = 1000
clean_lookup = {(row['domain'], row['ensemble']): row['f1_clean_mean'] for _, row in clean.iterrows()}

boot_records = []
for (domain, ensemble), grp in results_df.groupby(['domain', 'ensemble']):
    fold_scores = grp.sort_values('fold')['f1_macro'].values
    if len(fold_scores) < 5:
        continue
    clean_f1 = clean_lookup[(domain, ensemble)]
    rng = np.random.RandomState(RANDOM_SEED)
    boot_drops = np.zeros(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        sample = rng.choice(fold_scores, size=5, replace=True)
        boot_drops[b] = (clean_f1 - sample.mean()) / clean_f1 * 100
    point = (clean_f1 - fold_scores.mean()) / clean_f1 * 100
    ci_lo, ci_hi = np.percentile(boot_drops, [2.5, 97.5])
    boot_records.append({
        'domain': domain,
        'ensemble': ensemble,
        'rel_drop_pct': round(point, 2),
        'ci_low': round(ci_lo, 2),
        'ci_high': round(ci_hi, 2),
    })

boot_df = pd.DataFrame(boot_records).sort_values(['domain', 'ensemble'])
boot_df.to_csv(OUTPUT_DIR / 'pairflip_bootstrap_ci.csv', index=False)
