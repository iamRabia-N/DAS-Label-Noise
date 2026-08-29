import json
import time
import sys
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier

try:
    from s02_noise_injection import inject_symmetric_noise, inject_asymmetric_noise
except ImportError:
    _nm = sorted(Path('/kaggle/input').rglob('noise_injection.py'))
    assert _nm, "noise_injection.py not found - attach the noise module dataset"
    sys.path.insert(0, str(_nm[0].parent))
    from noise_injection import inject_symmetric_noise, inject_asymmetric_noise

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TIME_BUDGET_HOURS = 11.5
total_start = time.time()

def budget_left():
    return (time.time() - total_start) < TIME_BUDGET_HOURS * 3600


def find_dir(fname):
    hits = sorted(Path('/kaggle/input').rglob(fname))
    assert hits, f"{fname} not found - attach the required input"
    return hits[0].parent

DATA_DIR = find_dir('covtype_splits.npz')
TUNING_TAB = find_dir('best_hyperparameters.json')
OUTPUT_DIR = Path('/kaggle/working/das_perfeature_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fix_depth(p):
    if p.get('max_depth') in ('None', None):
        p['max_depth'] = None
    return p


with open(TUNING_TAB / 'best_hyperparameters.json') as f:
    tab = json.load(f)

RF_PARAMS = fix_depth(tab['random_forest']['best_params'].copy())
XGB_PARAMS = tab['xgboost']['best_params'].copy()
XGB_BASE = tab['xgboost']['base_params'].copy()
ET_PARAMS = fix_depth(tab['extra_trees']['params'].copy())
META_PARAMS = tab['stacking']['meta_learner_params']

try:
    m = XGBClassifier(tree_method='hist', device='cuda', n_estimators=2)
    m.fit(np.random.rand(50, 5), np.random.randint(0, 2, 50))
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False
print(f"GPU available: {GPU_AVAILABLE}")

d = np.load(DATA_DIR / 'covtype_splits.npz')
X_tab, _, y_tab, _ = train_test_split(d['X_train'], d['y_train'], train_size=100000, stratify=d['y_train'], random_state=RANDOM_SEED)
del d
gc.collect()
N_CLASSES = 7

PAIRFLIP = {1: 2, 2: 1, 5: 6, 6: 5}


def inject_pairflip(y, rate, pmap, seed):
    rng = np.random.RandomState(seed)
    yn = y.copy()
    for idx in np.where(rng.rand(len(y)) < rate)[0]:
        if y[idx] in pmap:
            yn[idx] = pmap[y[idx]]
    return yn, (yn != y)


def inject(y, ntype, rate, seed):
    if rate == 0.0:
        return y.copy(), np.zeros(len(y), dtype=bool)
    if ntype == 'symmetric':
        return inject_symmetric_noise(y, rate, N_CLASSES, seed)
    if ntype == 'asymmetric':
        return inject_asymmetric_noise(y, rate, N_CLASSES, random_state=seed)
    if ntype == 'pairflip':
        return inject_pairflip(y, rate, PAIRFLIP, seed)


def build_bases(rs):
    xp = {**XGB_BASE, **XGB_PARAMS, 'random_state': rs}
    if GPU_AVAILABLE:
        xp['device'] = 'cuda'
    else:
        xp.pop('device', None)
    return [('RF', RandomForestClassifier(**RF_PARAMS, n_jobs=-1, random_state=rs, class_weight='balanced_subsample')), ('XGB', XGBClassifier(**xp)), ('ET', ExtraTreesClassifier(**{**ET_PARAMS, 'random_state': rs}))]


def disagreement_features(prob_list):
    P = np.stack(prob_list)
    mean_p = P.mean(axis=0)
    ent = -(mean_p * np.log(np.clip(mean_p, 1e-12, None))).sum(axis=1)
    preds = P.argmax(axis=2)
    agree = np.zeros(P.shape[1])
    tv = np.zeros(P.shape[1])
    n_pairs = 0
    for a, b in combinations(range(P.shape[0]), 2):
        agree += (preds[a] == preds[b]).astype(float)
        tv += 0.5 * np.abs(P[a] - P[b]).sum(axis=1)
        n_pairs += 1
    tv /= n_pairs
    sorted_p = np.sort(mean_p, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    return np.column_stack([ent, agree, margin, tv])


# Feature matrix column order: [z1 entropy, z2 agreement, z3 margin, z4 total variation]
CONFIDENCE_COLS = [0, 2]
DISAGREEMENT_COLS = [1, 3]

INTERNAL_CV = 3
ARMS = ['PlainStack', 'AllFour', 'Confidence', 'Disagreement']


def run_condition(X_tr, y_noisy, X_va, y_va, rs):
    inner = StratifiedKFold(n_splits=INTERNAL_CV, shuffle=True, random_state=rs)
    oof_probs, val_probs = [], []
    for name, base in build_bases(rs):
        oof = cross_val_predict(base, X_tr, y_noisy, cv=inner, method='predict_proba', n_jobs=1)
        base.fit(X_tr, y_noisy)
        oof_probs.append(oof)
        val_probs.append(base.predict_proba(X_va))
        del base
        gc.collect()

    Z_tr = np.hstack(oof_probs)
    Z_va = np.hstack(val_probs)
    D_tr = disagreement_features(oof_probs)
    D_va = disagreement_features(val_probs)
    scaler = StandardScaler().fit(D_tr)
    S_tr = scaler.transform(D_tr)
    S_va = scaler.transform(D_va)

    arm_inputs = [
        ('PlainStack', Z_tr, Z_va),
        ('AllFour', np.hstack([Z_tr, S_tr]), np.hstack([Z_va, S_va])),
        ('Confidence', np.hstack([Z_tr, S_tr[:, CONFIDENCE_COLS]]), np.hstack([Z_va, S_va[:, CONFIDENCE_COLS]])),
        ('Disagreement', np.hstack([Z_tr, S_tr[:, DISAGREEMENT_COLS]]), np.hstack([Z_va, S_va[:, DISAGREEMENT_COLS]])),
    ]
    out = {}
    for arm, Ztr, Zva in arm_inputs:
        meta = LogisticRegression(**META_PARAMS)
        meta.fit(Ztr, y_noisy)
        yp = meta.predict(Zva)
        out[arm] = {'f1_macro': float(f1_score(y_va, yp, average='macro', zero_division=0)), 'accuracy': float(accuracy_score(y_va, yp))}
    return out


FULL = [0.0, 0.1, 0.2, 0.3, 0.4]
CONDITIONS = [('symmetric', l) for l in FULL] + [('asymmetric', l) for l in FULL] + [('pairflip', 0.4)]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

runs = [(nt, lv, f) for (nt, lv) in CONDITIONS for f in range(5)]

inter = OUTPUT_DIR / 'das_perfeature_intermediate.csv'
prior = sorted(Path('/kaggle/input').rglob('das_perfeature_intermediate.csv'))
results = []
done = set()
if prior:
    for r in pd.read_csv(prior[0]).to_dict('records'):
        results.append(r)
        done.add((r['noise_type'], float(r['noise_level']), int(r['fold'])))
print(f"Restored: {len(done)} of {len(runs)} runs")

remaining = [r for r in runs if r not in done]

stopped = False
for ntype, level, fold in remaining:
    if not budget_left():
        print("TIME BUDGET REACHED. Rerun with this version's output attached to resume.")
        stopped = True
        break
    t0 = time.time()
    tr, va = list(cv.split(X_tab, y_tab))[fold]
    seed = RANDOM_SEED + fold * 100 + int(level * 1000)
    y_noisy, mask = inject(y_tab[tr], ntype, level, seed)

    out = run_condition(X_tab[tr], y_noisy, X_tab[va], y_tab[va], RANDOM_SEED + fold)
    for arm in ARMS:
        results.append({'domain': 'Tabular', 'arm': arm, 'noise_type': ntype, 'noise_level': level, 'fold': fold, 'actual_noise_pct': float(mask.mean() * 100), 'runtime_seconds': float(time.time() - t0), **out[arm]})
    done.add((ntype, level, fold))
    pd.DataFrame(results).to_csv(inter, index=False)
    print(f"[{ntype:10s}|{level:.0%}|f{fold}] " + " ".join(f"{a}={out[a]['f1_macro']:.4f}" for a in ARMS) + f" | {(time.time() - t0) / 60:.1f}m elapsed={(time.time() - total_start) / 60:.1f}m")
    gc.collect()

pd.DataFrame(results).to_csv(inter, index=False)
df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / 'das_perfeature_results_full.csv', index=False)

if not stopped:
    pivot = df.pivot_table(index=['noise_type', 'noise_level', 'fold'], columns='arm', values='f1_macro').reset_index()
    for arm in ['AllFour', 'Confidence', 'Disagreement']:
        pivot[f'delta_{arm}'] = pivot[arm] - pivot['PlainStack']
    delta_cols = [f'delta_{a}' for a in ['AllFour', 'Confidence', 'Disagreement']]
    cond = pivot.groupby(['noise_type', 'noise_level'])[delta_cols].agg(['mean', 'std']).round(4).reset_index()
    cond.columns = ['_'.join(c).rstrip('_') for c in cond.columns.values]
    cond.to_csv(OUTPUT_DIR / 'das_perfeature_delta_by_condition.csv', index=False)
    print("\nDelta vs PlainStack by condition:")
    print(cond.to_string(index=False))

    wil = []
    noisy = pivot[pivot['noise_level'] > 0]
    groups = [(nt, noisy[noisy['noise_type'] == nt]) for nt in ['symmetric', 'asymmetric', 'pairflip']]
    groups.append(('symmetric_40_only', noisy[(noisy['noise_type'] == 'symmetric') & (noisy['noise_level'] == 0.4)]))
    for gname, g in groups:
        for arm in ['AllFour', 'Confidence', 'Disagreement']:
            deltas = g[f'delta_{arm}'].values
            if len(deltas) == 0:
                continue
            try:
                stat, p = wilcoxon(deltas)
            except ValueError:
                stat, p = np.nan, np.nan
            wil.append({'group': gname, 'arm': arm, 'n': len(deltas), 'mean_delta': round(float(deltas.mean()), 4), 'wilcoxon_p': round(float(p), 4) if not np.isnan(p) else None})
    wildf = pd.DataFrame(wil)
    wildf.to_csv(OUTPUT_DIR / 'das_perfeature_wilcoxon.csv', index=False)
    print("\nWilcoxon vs PlainStack:")
    print(wildf.to_string(index=False))
