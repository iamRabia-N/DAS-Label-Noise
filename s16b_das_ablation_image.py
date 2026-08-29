import json
import time
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
from s02_noise_injection import inject_symmetric_noise, inject_asymmetric_noise

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DOMAINS = ['Image']

TIME_BUDGET_HOURS = 11.5
total_start = time.time()

def budget_left():
    return (time.time() - total_start) < TIME_BUDGET_HOURS * 3600


def find_dir(fname):
    hits = sorted(Path('/kaggle/input').rglob(fname))
    assert hits, f"{fname} not found - attach the required input"
    return hits[0].parent

DATA_DIR = find_dir('covtype_splits.npz')
FEATURES_DIR = find_dir('eurosat_features_train.npz')
TUNING_TAB = find_dir('best_hyperparameters.json')
TUNING_IMG = find_dir('best_hyperparameters_image.json')
OUTPUT_DIR = Path('/kaggle/working/das_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fix_depth(p):
    if p.get('max_depth') in ('None', None):
        p['max_depth'] = None
    return p


with open(TUNING_TAB / 'best_hyperparameters.json') as f:
    tab = json.load(f)
with open(TUNING_IMG / 'best_hyperparameters_image.json') as f:
    img = json.load(f)

PARAMS = {
    'Tabular': dict(rf=fix_depth(tab['random_forest']['best_params'].copy()), xgb=tab['xgboost']['best_params'].copy(), xgb_base=tab['xgboost']['base_params'].copy(), et=fix_depth(tab['extra_trees']['params'].copy()), meta=tab['stacking']['meta_learner_params']),
    'Image': dict(rf=fix_depth(img['random_forest']['best_params'].copy()), xgb=img['xgboost']['best_params'].copy(), xgb_base=img['xgboost']['base_params'].copy(), et=fix_depth(img['extra_trees']['params'].copy()), meta=img['stacking']['meta_learner_params']),
}

try:
    m = XGBClassifier(tree_method='hist', device='cuda', n_estimators=2)
    m.fit(np.random.rand(50, 5), np.random.randint(0, 2, 50))
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

d = np.load(DATA_DIR / 'covtype_splits.npz')
X_tab, _, y_tab, _ = train_test_split(d['X_train'], d['y_train'], train_size=100000, stratify=d['y_train'], random_state=RANDOM_SEED)
del d
gc.collect()

img_d = np.load(FEATURES_DIR / 'eurosat_features_train.npz')
DOMAIN_DATA = {'Tabular': (X_tab, y_tab, 7), 'Image': (img_d['X'], img_d['y'], 10)}

PAIRFLIP = {'Tabular': {1: 2, 2: 1, 5: 6, 6: 5}, 'Image': {0: 5, 5: 0, 3: 4, 4: 3, 8: 9, 9: 8}}


def inject_pairflip(y, rate, pmap, seed):
    rng = np.random.RandomState(seed)
    yn = y.copy()
    for idx in np.where(rng.rand(len(y)) < rate)[0]:
        if y[idx] in pmap:
            yn[idx] = pmap[y[idx]]
    return yn, (yn != y)


def inject(y, ntype, rate, K, seed, dom):
    if rate == 0.0:
        return y.copy(), np.zeros(len(y), dtype=bool)
    if ntype == 'symmetric':
        return inject_symmetric_noise(y, rate, K, seed)
    if ntype == 'asymmetric':
        return inject_asymmetric_noise(y, rate, K, random_state=seed)
    if ntype == 'pairflip':
        return inject_pairflip(y, rate, PAIRFLIP[dom], seed)


def build_bases(dom, rs):
    p = PARAMS[dom]
    xp = {**p['xgb_base'], **p['xgb'], 'random_state': rs}
    if GPU_AVAILABLE:
        xp['device'] = 'cuda'
    else:
        xp.pop('device', None)
    return [('RF', RandomForestClassifier(**p['rf'], n_jobs=-1, random_state=rs, class_weight='balanced_subsample')), ('XGB', XGBClassifier(**xp)), ('ET', ExtraTreesClassifier(**{**p['et'], 'random_state': rs}))]


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


INTERNAL_CV = 3


def run_condition(dom, X_tr, y_noisy, X_va, y_va, rs, meta_params):
    inner = StratifiedKFold(n_splits=INTERNAL_CV, shuffle=True, random_state=rs)
    oof_probs, val_probs = [], []
    for name, base in build_bases(dom, rs):
        oof = cross_val_predict(base, X_tr, y_noisy, cv=inner, method='predict_proba', n_jobs=1)
        base.fit(X_tr, y_noisy)
        oof_probs.append(oof)
        val_probs.append(base.predict_proba(X_va))
        del base
        gc.collect()

    Z_tr_plain = np.hstack(oof_probs)
    Z_va_plain = np.hstack(val_probs)
    D_tr = disagreement_features(oof_probs)
    D_va = disagreement_features(val_probs)
    scaler = StandardScaler().fit(D_tr)
    Z_tr_das = np.hstack([Z_tr_plain, scaler.transform(D_tr)])
    Z_va_das = np.hstack([Z_va_plain, scaler.transform(D_va)])

    out = {}
    for arm, Ztr, Zva in [('PlainStack', Z_tr_plain, Z_va_plain), ('DAS', Z_tr_das, Z_va_das)]:
        meta = LogisticRegression(**meta_params)
        meta.fit(Ztr, y_noisy)
        yp = meta.predict(Zva)
        out[arm] = {'f1_macro': float(f1_score(y_va, yp, average='macro', zero_division=0)), 'accuracy': float(accuracy_score(y_va, yp))}
    return out


FULL = [0.0, 0.1, 0.2, 0.3, 0.4]
REDUCED = [0.0, 0.2, 0.4]
CONDITIONS = {
    'Tabular': [('symmetric', l) for l in FULL] + [('asymmetric', l) for l in FULL] + [('pairflip', 0.4)],
    'Image': [('symmetric', l) for l in REDUCED] + [('asymmetric', l) for l in REDUCED] + [('pairflip', 0.4)],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

runs = [(dom, nt, lv, f) for dom in DOMAINS for (nt, lv) in CONDITIONS[dom] for f in range(5)]

inter = OUTPUT_DIR / 'das_intermediate.csv'
prior = sorted(Path('/kaggle/input').rglob('das_intermediate.csv'))
results = []
done = set()
if prior:
    for r in pd.read_csv(prior[0]).to_dict('records'):
        results.append(r)
        done.add((r['domain'], r['noise_type'], float(r['noise_level']), int(r['fold'])))

remaining = [r for r in runs if r not in done]

stopped = False
for dom, ntype, level, fold in remaining:
    if not budget_left():
        stopped = True
        break
    t0 = time.time()
    X, y, K = DOMAIN_DATA[dom]
    tr, va = list(cv.split(X, y))[fold]
    seed = RANDOM_SEED + fold * 100 + int(level * 1000)
    y_noisy, mask = inject(y[tr], ntype, level, K, seed, dom)

    out = run_condition(dom, X[tr], y_noisy, X[va], y[va], RANDOM_SEED + fold, PARAMS[dom]['meta'])
    for arm in ['PlainStack', 'DAS']:
        results.append({'domain': dom, 'arm': arm, 'noise_type': ntype, 'noise_level': level, 'fold': fold, 'actual_noise_pct': float(mask.mean() * 100), 'runtime_seconds': float(time.time() - t0), **out[arm]})
    done.add((dom, ntype, level, fold))
    pd.DataFrame(results).to_csv(inter, index=False)
    gc.collect()

pd.DataFrame(results).to_csv(inter, index=False)
df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / 'das_results_full.csv', index=False)

if not stopped:
    pivot = df.pivot_table(index=['domain', 'noise_type', 'noise_level', 'fold'], columns='arm', values='f1_macro').reset_index()
    pivot['delta'] = pivot['DAS'] - pivot['PlainStack']
    cond = pivot.groupby(['domain', 'noise_type', 'noise_level'])['delta'].agg(['mean', 'std']).round(4).reset_index()
    cond.to_csv(OUTPUT_DIR / 'das_delta_by_condition.csv', index=False)

    wil = []
    for (dm, nt), g in pivot[pivot['noise_level'] > 0].groupby(['domain', 'noise_type']):
        deltas = g['delta'].values
        try:
            stat, p = wilcoxon(deltas)
        except ValueError:
            stat, p = np.nan, np.nan
        wil.append({'domain': dm, 'noise_type': nt, 'mean_delta': round(float(deltas.mean()), 4), 'n': len(deltas), 'wilcoxon_p': round(float(p), 4) if not np.isnan(p) else None})
    wildf = pd.DataFrame(wil)
    wildf.to_csv(OUTPUT_DIR / 'das_wilcoxon.csv', index=False)
