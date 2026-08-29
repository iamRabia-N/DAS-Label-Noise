import json
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from xgboost import XGBClassifier
from s02_noise_injection import inject_symmetric_noise, inject_asymmetric_noise, NOISE_LEVELS

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = Path('/kaggle/input/datasets/rabianaz22/1-ensemble-noise-prepared-data-3rd-paper/prepared_data')
TUNING_DIR = Path('/kaggle/input/datasets/rabianaz22/3a-tabular-tuning-3rd-paper/tuning_results')
OUTPUT_DIR = Path('/kaggle/working/equal_n_results')
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

assert DATA_DIR.exists()
assert TUNING_DIR.exists()

with open(TUNING_DIR / 'best_hyperparameters.json') as f:
    tuned = json.load(f)


def fix_depth(p):
    if p.get('max_depth') in ('None', None):
        p['max_depth'] = None
    return p


best_rf_params = fix_depth(tuned['random_forest']['best_params'].copy())
best_xgb_params = tuned['xgboost']['best_params'].copy()
xgb_base_params = tuned['xgboost']['base_params'].copy()
et_params = fix_depth(tuned['extra_trees']['params'].copy())

try:
    m = XGBClassifier(tree_method='hist', device='cuda', n_estimators=2)
    m.fit(np.random.rand(100, 10), np.random.randint(0, 3, 100))
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

data = np.load(DATA_DIR / 'covtype_splits.npz')
X_train_full, y_train_full = data['X_train'], data['y_train']
n_classes = 7

SUBSAMPLE_SIZE = 100000
X_train, _, y_train, _ = train_test_split(
    X_train_full, y_train_full,
    train_size=SUBSAMPLE_SIZE,
    stratify=y_train_full,
    random_state=RANDOM_SEED
)
del X_train_full, y_train_full
gc.collect()

TABULAR_PAIRFLIP_MAP = {1: 2, 2: 1, 5: 6, 6: 5}


def inject_pairflip_noise(y_true, noise_rate, transition_map, random_state):
    rng = np.random.RandomState(random_state)
    y_noisy = y_true.copy()
    flip_mask = rng.rand(len(y_true)) < noise_rate
    for idx in np.where(flip_mask)[0]:
        c = y_true[idx]
        if c in transition_map:
            y_noisy[idx] = transition_map[c]
    return y_noisy, (y_noisy != y_true)


def compute_metrics(y_true, y_pred, n_classes):
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0)
    for c in range(n_classes):
        out[f'f1_class_{c}'] = float(f[c])
    return out


def get_fold(X, y, fold_idx, cv):
    tr, va = list(cv.split(X, y))[fold_idx]
    return X[tr], X[va], y[tr], y[va]


def inject(y, ntype, rate, seed):
    if rate == 0.0:
        return y.copy(), np.zeros(len(y), dtype=bool)
    if ntype == 'symmetric':
        return inject_symmetric_noise(y, rate, n_classes, seed)
    if ntype == 'asymmetric':
        return inject_asymmetric_noise(y, rate, n_classes, random_state=seed)
    if ntype == 'pairflip':
        return inject_pairflip_noise(y, rate, TABULAR_PAIRFLIP_MAP, seed)
    raise ValueError(ntype)


def build_model(name, rs):
    if name == 'RF':
        return RandomForestClassifier(**best_rf_params, n_jobs=-1,
                                      random_state=rs, class_weight='balanced_subsample')
    if name == 'XGBoost':
        p = {**xgb_base_params, **best_xgb_params, 'random_state': rs}
        if GPU_AVAILABLE:
            p['device'] = 'cuda'
        else:
            p.pop('device', None)
        return XGBClassifier(**p)
    if name == 'Voting':
        rf = RandomForestClassifier(**best_rf_params, n_jobs=-1,
                                    random_state=rs, class_weight='balanced_subsample')
        xp = {**xgb_base_params, **best_xgb_params, 'random_state': rs}
        if GPU_AVAILABLE:
            xp['device'] = 'cuda'
        else:
            xp.pop('device', None)
        xgb = XGBClassifier(**xp)
        et = ExtraTreesClassifier(**{**et_params, 'random_state': rs})
        return VotingClassifier(estimators=[('rf', rf), ('xgb', xgb), ('et', et)],
                                voting='soft', n_jobs=1)
    raise ValueError(name)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
METHODS = ['RF', 'XGBoost', 'Voting']

runs = []
for method in METHODS:
    for ntype in ['symmetric', 'asymmetric']:
        for level in NOISE_LEVELS:
            for fold in range(5):
                runs.append((method, ntype, level, fold))
    for fold in range(5):
        runs.append((method, 'pairflip', 0.4, fold))

checkpoint_file = CHECKPOINT_DIR / 'equal_n_progress.json'
results_file = CHECKPOINT_DIR / 'equal_n_intermediate.csv'

if checkpoint_file.exists() and results_file.exists():
    with open(checkpoint_file) as f:
        completed = set(tuple(r) for r in json.load(f)['completed'])
    all_results = pd.read_csv(results_file).to_dict('records')
else:
    completed, all_results = set(), []

remaining = [r for r in runs if r not in completed]

for i, (method, ntype, level, fold) in enumerate(remaining, 1):
    t0 = time.time()
    clf = None
    try:
        X_tr, X_va, y_tr_clean, y_va = get_fold(X_train, y_train, fold, cv)
        seed = RANDOM_SEED + fold * 100 + int(level * 1000)
        y_tr_noisy, mask = inject(y_tr_clean, ntype, level, seed)

        clf = build_model(method, RANDOM_SEED + fold)
        clf.fit(X_tr, y_tr_noisy)
        metrics = compute_metrics(y_va, clf.predict(X_va), n_classes)

        all_results.append({
            'model': f'{method}_100k', 'noise_type': ntype,
            'noise_level': level, 'fold': fold,
            'train_size': int(len(y_tr_noisy)),
            'actual_noise_pct': float(mask.mean() * 100),
            'runtime_seconds': float(time.time() - t0),
            **metrics
        })
        completed.add((method, ntype, level, fold))
    except Exception:
        pass

    if clf is not None:
        del clf
    gc.collect()

    if i % 5 == 0:
        pd.DataFrame(all_results).to_csv(results_file, index=False)
        with open(checkpoint_file, 'w') as f:
            json.dump({'completed': [list(r) for r in completed]}, f)

pd.DataFrame(all_results).to_csv(results_file, index=False)
with open(checkpoint_file, 'w') as f:
    json.dump({'completed': [list(r) for r in completed]}, f)

df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_DIR / 'equal_n_results_full.csv', index=False)

summary = df.groupby(['model', 'noise_type', 'noise_level'])['f1_macro'] \
            .agg(['mean', 'std']).reset_index()
summary.to_csv(OUTPUT_DIR / 'equal_n_summary.csv', index=False)

records = []
for (model, ntype), grp in summary[summary['noise_type'] != 'pairflip'] \
                            .groupby(['model', 'noise_type']):
    g = grp.sort_values('noise_level')
    f1_clean = g[g['noise_level'] == 0.0]['mean'].values[0]
    slope = np.polyfit(g['noise_level'].values, g['mean'].values, 1)[0]
    f1_40 = g[g['noise_level'] == 0.4]['mean'].values[0]
    records.append({
        'model': model, 'noise_type': ntype,
        'f1_clean': round(f1_clean, 4), 'f1_at_40': round(f1_40, 4),
        'nsi_slope': round(slope, 4),
        'rel_drop_pct': round((f1_clean - f1_40) / f1_clean * 100, 2)
    })
nsi_df = pd.DataFrame(records)
nsi_df.to_csv(OUTPUT_DIR / 'equal_n_nsi.csv', index=False)

N_BOOT = 1000
boot_records = []
for (model, ntype), grp in df[df['noise_type'] != 'pairflip'] \
                            .groupby(['model', 'noise_type']):
    pivot = grp.pivot_table(index='fold', columns='noise_level', values='f1_macro')
    levels = np.array(sorted(pivot.columns))
    rng = np.random.RandomState(RANDOM_SEED)
    slopes = np.zeros(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.choice(pivot.index, size=len(pivot), replace=True)
        mean_curve = pivot.loc[idx].mean(axis=0).values
        slopes[b] = np.polyfit(levels, mean_curve, 1)[0]
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    boot_records.append({'model': model, 'noise_type': ntype,
                         'slope_ci_low': round(lo, 4),
                         'slope_ci_high': round(hi, 4)})
boot_df = pd.DataFrame(boot_records)
boot_df.to_csv(OUTPUT_DIR / 'equal_n_slope_ci.csv', index=False)

pf = df[df['noise_type'] == 'pairflip'].groupby('model')['f1_macro'] \
       .agg(['mean', 'std']).reset_index()
pf.to_csv(OUTPUT_DIR / 'equal_n_pairflip.csv', index=False)
