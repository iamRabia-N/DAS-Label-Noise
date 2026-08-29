import json
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, cohen_kappa_score
from xgboost import XGBClassifier
from s02_noise_injection import inject_symmetric_noise, inject_asymmetric_noise, NOISE_LEVELS

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = Path('/kaggle/input/datasets/rabianaz22/1-ensemble-noise-prepared-data-3rd-paper/prepared_data')
FEATURES_DIR = Path('/kaggle/input/datasets/rabianaz22/4a-image-features-3rd-paper/image_features')
TUNING_TAB = Path('/kaggle/input/datasets/rabianaz22/3a-tabular-tuning-3rd-paper/tuning_results')
TUNING_IMG = Path('/kaggle/input/datasets/rabianaz22/4b-image-tuning-3rd-paper/image_tuning_results')
OUTPUT_DIR = Path('/kaggle/working/kappa_results')
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

for p in [DATA_DIR, FEATURES_DIR, TUNING_TAB, TUNING_IMG]:
    assert p.exists(), f"Missing: {p}"

# Reuse completed run results if attached as input: attach the dataset holding
# kappa_results_full.csv and the training loop is skipped.
_prior = [p for p in Path('/kaggle/input').rglob('kappa_results_full.csv')]
SKIP_TRAINING = len(_prior) > 0


def fix_depth(p):
    if p.get('max_depth') in ('None', None):
        p['max_depth'] = None
    return p


with open(TUNING_TAB / 'best_hyperparameters.json') as f:
    tab = json.load(f)
with open(TUNING_IMG / 'best_hyperparameters_image.json') as f:
    img = json.load(f)

PARAMS = {
    'Tabular': dict(rf=fix_depth(tab['random_forest']['best_params'].copy()),
                    xgb=tab['xgboost']['best_params'].copy(),
                    xgb_base=tab['xgboost']['base_params'].copy(),
                    et=fix_depth(tab['extra_trees']['params'].copy())),
    'Image':   dict(rf=fix_depth(img['random_forest']['best_params'].copy()),
                    xgb=img['xgboost']['best_params'].copy(),
                    xgb_base=img['xgboost']['base_params'].copy(),
                    et=fix_depth(img['extra_trees']['params'].copy())),
}

try:
    m = XGBClassifier(tree_method='hist', device='cuda', n_estimators=2)
    m.fit(np.random.rand(50, 5), np.random.randint(0, 2, 50))
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

d = np.load(DATA_DIR / 'covtype_splits.npz')
X_tab, _, y_tab, _ = train_test_split(
    d['X_train'], d['y_train'], train_size=100000,
    stratify=d['y_train'], random_state=RANDOM_SEED)
del d
gc.collect()

img_d = np.load(FEATURES_DIR / 'eurosat_features_train.npz')
X_img, y_img = img_d['X'], img_d['y']

DOMAIN_DATA = {'Tabular': (X_tab, y_tab, 7), 'Image': (X_img, y_img, 10)}

PAIRFLIP = {
    'Tabular': {1: 2, 2: 1, 5: 6, 6: 5},
    'Image':   {0: 5, 5: 0, 3: 4, 4: 3, 8: 9, 9: 8},
}


def inject_pairflip(y, rate, pmap, seed):
    rng = np.random.RandomState(seed)
    yn = y.copy()
    for idx in np.where(rng.rand(len(y)) < rate)[0]:
        if y[idx] in pmap:
            yn[idx] = pmap[y[idx]]
    return yn, (yn != y)


def inject(y, ntype, rate, n_classes, seed, domain):
    if rate == 0.0:
        return y.copy(), np.zeros(len(y), dtype=bool)
    if ntype == 'symmetric':
        return inject_symmetric_noise(y, rate, n_classes, seed)
    if ntype == 'asymmetric':
        return inject_asymmetric_noise(y, rate, n_classes, random_state=seed)
    if ntype == 'pairflip':
        return inject_pairflip(y, rate, PAIRFLIP[domain], seed)
    raise ValueError(ntype)


def build(name, domain, rs):
    p = PARAMS[domain]
    if name == 'RF':
        return RandomForestClassifier(**p['rf'], n_jobs=-1, random_state=rs,
                                      class_weight='balanced_subsample')
    if name == 'XGB':
        full = {**p['xgb_base'], **p['xgb'], 'random_state': rs}
        if GPU_AVAILABLE:
            full['device'] = 'cuda'
        else:
            full.pop('device', None)
        return XGBClassifier(**full)
    if name == 'ET':
        return ExtraTreesClassifier(**{**p['et'], 'random_state': rs})


CONDITIONS = ([('symmetric', l) for l in NOISE_LEVELS] +
              [('asymmetric', l) for l in NOISE_LEVELS] +
              [('pairflip', 0.4)])
BASES = ['RF', 'XGB', 'ET']
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

runs = [(dom, nt, lv, f) for dom in ['Tabular', 'Image']
        for (nt, lv) in CONDITIONS for f in range(5)]

ckpt = CHECKPOINT_DIR / 'kappa_progress.json'
inter = CHECKPOINT_DIR / 'kappa_intermediate.csv'

if SKIP_TRAINING:
    results = pd.read_csv(_prior[0]).to_dict('records')
    done = set()
    remaining = []
elif ckpt.exists() and inter.exists():
    with open(ckpt) as f:
        done = set(tuple(r) for r in json.load(f)['completed'])
    results = pd.read_csv(inter).to_dict('records')
    remaining = [r for r in runs if r not in done]
else:
    done, results = set(), []
    remaining = runs

for i, (dom, ntype, level, fold) in enumerate(remaining, 1):
    t0 = time.time()
    try:
        X, y, K = DOMAIN_DATA[dom]
        tr, va = list(cv.split(X, y))[fold]
        X_tr, X_va, y_tr, y_va = X[tr], X[va], y[tr], y[va]
        seed = RANDOM_SEED + fold * 100 + int(level * 1000)
        y_noisy, mask = inject(y_tr, ntype, level, K, seed, dom)

        preds = {}
        f1s = {}
        for b in BASES:
            clf = build(b, dom, RANDOM_SEED + fold)
            clf.fit(X_tr, y_noisy)
            preds[b] = clf.predict(X_va)
            f1s[b] = f1_score(y_va, preds[b], average='macro', zero_division=0)
            del clf
            gc.collect()

        row = {'domain': dom, 'noise_type': ntype, 'noise_level': level,
               'fold': fold, 'actual_noise_pct': float(mask.mean() * 100),
               'runtime_seconds': float(time.time() - t0)}
        kappas = []
        for a, b in combinations(BASES, 2):
            k = cohen_kappa_score(preds[a], preds[b])
            row[f'kappa_{a}_{b}'] = float(k)
            kappas.append(k)
        row['kappa_mean'] = float(np.mean(kappas))
        row['full_agreement_rate'] = float(np.mean(
            (preds['RF'] == preds['XGB']) & (preds['XGB'] == preds['ET'])))
        for b in BASES:
            row[f'f1_{b}'] = float(f1s[b])
        results.append(row)
        done.add((dom, ntype, level, fold))
    except Exception:
        pass

    if i % 5 == 0:
        pd.DataFrame(results).to_csv(inter, index=False)
        with open(ckpt, 'w') as f:
            json.dump({'completed': [list(r) for r in done]}, f)

if remaining:
    pd.DataFrame(results).to_csv(inter, index=False)
    with open(ckpt, 'w') as f:
        json.dump({'completed': [list(r) for r in done]}, f)

df = pd.DataFrame(results)
df.to_csv(OUTPUT_DIR / 'kappa_results_full.csv', index=False)

summary = df.groupby(['domain', 'noise_type', 'noise_level'])[
    ['kappa_mean', 'full_agreement_rate']].agg(['mean', 'std']).reset_index()
summary.columns = ['_'.join(c).rstrip('_') for c in summary.columns.values]
summary.to_csv(OUTPUT_DIR / 'kappa_summary.csv', index=False)


def find_csv(fname):
    hits = list(Path('/kaggle/input').rglob(fname))
    return pd.read_csv(hits[0]) if hits else None


sources = {
    ('Tabular', 'stacking'): 'stacking_results_full.csv',
    ('Tabular', 'voting'):   'equal_n_results_full.csv',
    ('Image', 'stacking_sym'):  'stacking_symmetric_results_full.csv',
    ('Image', 'stacking_asym'): 'stacking_asymmetric_results_full.csv',
    ('Image', 'voting'):     'voting_image_results_full.csv',
}
loaded = {k: find_csv(v) for k, v in sources.items()}
missing = [v for k, v in sources.items() if loaded[k] is None]
assert not missing, f"Correlation sources missing, attach datasets containing: {missing}"

pf = find_csv('pairflip_results_full.csv')
assert pf is not None, "pairflip_results_full.csv missing; required for the n=20 correlation."

gap_rows = []
st = loaded[('Tabular', 'stacking')]
vo = loaded[('Tabular', 'voting')]
vo = vo[vo['model'] == 'Voting_100k']
for (nt, lv), g in st.groupby(['noise_type', 'noise_level']):
    vg = vo[(vo['noise_type'] == nt) & (vo['noise_level'] == lv)]
    if len(vg):
        gap_rows.append({'domain': 'Tabular', 'noise_type': nt, 'noise_level': lv,
                         'gap': g['f1_macro'].mean() - vg['f1_macro'].mean()})

si = pd.concat([loaded[('Image', 'stacking_sym')], loaded[('Image', 'stacking_asym')]])
vi = loaded[('Image', 'voting')]
for (nt, lv), g in si.groupby(['noise_type', 'noise_level']):
    vg = vi[(vi['noise_type'] == nt) & (vi['noise_level'] == lv)]
    if len(vg):
        gap_rows.append({'domain': 'Image', 'noise_type': nt, 'noise_level': lv,
                         'gap': g['f1_macro'].mean() - vg['f1_macro'].mean()})

sub_i = pf[pf['domain'] == 'Image']
s_i = sub_i[sub_i['ensemble'] == 'Stacking']['f1_macro'].mean()
v_i = sub_i[sub_i['ensemble'] == 'Voting']['f1_macro'].mean()
gap_rows.append({'domain': 'Image', 'noise_type': 'pairflip', 'noise_level': 0.4,
                 'gap': s_i - v_i})
s_t = pf[(pf['domain'] == 'Tabular') & (pf['ensemble'] == 'Stacking')]['f1_macro'].mean()
v_t_rows = vo[(vo['noise_type'] == 'pairflip') & (vo['noise_level'] == 0.4)]
if len(v_t_rows):
    gap_rows.append({'domain': 'Tabular', 'noise_type': 'pairflip', 'noise_level': 0.4,
                     'gap': s_t - v_t_rows['f1_macro'].mean()})

gaps = pd.DataFrame(gap_rows)

ksum = df.groupby(['domain', 'noise_type', 'noise_level'])[
    ['kappa_mean', 'full_agreement_rate']].mean().reset_index()
merged = ksum.merge(gaps, on=['domain', 'noise_type', 'noise_level'])
merged.to_csv(OUTPUT_DIR / 'kappa_vs_gap.csv', index=False)

correlation_rows = []
for scope, sub in [('All', merged),
                   ('Tabular', merged[merged['domain'] == 'Tabular']),
                   ('Image', merged[merged['domain'] == 'Image'])]:
    rho, p = spearmanr(sub['kappa_mean'], sub['gap'])
    correlation_rows.append({'scope': scope, 'dedup': False,
                             'spearman_rho': round(rho, 4),
                             'p_value': round(p, 6), 'n': len(sub)})

# Clean condition counted once per domain: it is a single experimental
# condition at the zero-noise endpoint of both sweeps, so one of its two
# stored rows is dropped per domain. These are the values reported in the paper.
dedup = merged[~((merged['noise_level'] == 0) &
                 (merged['noise_type'] == 'symmetric'))].copy()
dedup.to_csv(OUTPUT_DIR / 'kappa_vs_gap_dedup.csv', index=False)
for scope, sub in [('All', dedup),
                   ('Tabular', dedup[dedup['domain'] == 'Tabular']),
                   ('Image', dedup[dedup['domain'] == 'Image'])]:
    rho, p = spearmanr(sub['kappa_mean'], sub['gap'])
    correlation_rows.append({'scope': scope, 'dedup': True,
                             'spearman_rho': round(rho, 4),
                             'p_value': round(p, 6), 'n': len(sub)})

pd.DataFrame(correlation_rows).to_csv(OUTPUT_DIR / 'kappa_gap_correlations.csv', index=False)
