import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import wilcoxon, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
TIME_BUDGET_HOURS = 11.5
total_start = time.time()

OUTPUT_DIR = Path('/kaggle/working/gate_results_image')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

META_LR = {'C': 1.0, 'max_iter': 1000, 'multi_class': 'multinomial', 'solver': 'lbfgs', 'random_state': RANDOM_SEED}

prob_files = sorted(set(Path('/kaggle/input').rglob('probs_*.npz')), key=lambda p: p.name)
seen = set()
files = []
for p in prob_files:
    if p.name not in seen:
        seen.add(p.name)
        files.append(p)
assert files, "No probs_*.npz found - attach the s13d and s13e outputs"

datasets = sorted(set(f.name.split('_')[1] for f in files))


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


results = []
for fp in files:
    assert (time.time() - total_start) < TIME_BUDGET_HOURS * 3600
    parts = fp.stem.split('_')
    key = parts[1]
    cond = '_'.join(parts[2:-1])
    fold = int(parts[-1][1:])
    z = np.load(fp)
    y_noisy = z['y_train_noisy'].astype(int)
    y_val = z['y_val'].astype(int)
    noise_pct = float(z['noise_mask'].mean() * 100)
    oof = [z['oof_rf'].astype(np.float64), z['oof_xgb'].astype(np.float64), z['oof_et'].astype(np.float64)]
    val = [z['val_rf'].astype(np.float64), z['val_xgb'].astype(np.float64), z['val_et'].astype(np.float64)]

    row = {'dataset': key, 'condition': cond, 'noise_type': cond.rsplit('_', 1)[0], 'fold': fold, 'actual_noise_pct': noise_pct}

    single_f1 = {}
    val_preds = []
    for name, v in zip(['RF', 'XGB', 'ET'], val):
        yp = v.argmax(axis=1)
        val_preds.append(yp)
        f1 = f1_score(y_val, yp, average='macro', zero_division=0)
        single_f1[name] = f1
        row[f'f1_{name}'] = float(f1)
        row[f'acc_{name}'] = float(accuracy_score(y_val, yp))

    yp_vote = np.mean(np.stack(val), axis=0).argmax(axis=1)
    row['f1_Voting'] = float(f1_score(y_val, yp_vote, average='macro', zero_division=0))
    row['acc_Voting'] = float(accuracy_score(y_val, yp_vote))

    Z_tr = np.hstack(oof)
    Z_va = np.hstack(val)
    meta = LogisticRegression(**META_LR)
    meta.fit(Z_tr, y_noisy)
    yp_stack = meta.predict(Z_va)
    row['f1_PlainStack'] = float(f1_score(y_val, yp_stack, average='macro', zero_division=0))
    row['acc_PlainStack'] = float(accuracy_score(y_val, yp_stack))

    D_tr = disagreement_features(oof)
    D_va = disagreement_features(val)
    scaler = StandardScaler().fit(D_tr)
    meta2 = LogisticRegression(**META_LR)
    meta2.fit(np.hstack([Z_tr, scaler.transform(D_tr)]), y_noisy)
    yp_das = meta2.predict(np.hstack([Z_va, scaler.transform(D_va)]))
    row['f1_DAS'] = float(f1_score(y_val, yp_das, average='macro', zero_division=0))
    row['acc_DAS'] = float(accuracy_score(y_val, yp_das))

    kappas = [cohen_kappa_score(val_preds[a], val_preds[b]) for a, b in combinations(range(3), 2)]
    row['mean_pairwise_kappa'] = float(np.mean(kappas))
    row['f1_best_single'] = float(max(single_f1.values()))
    row['stack_gap_vs_best'] = row['f1_PlainStack'] - row['f1_best_single']
    row['das_delta'] = row['f1_DAS'] - row['f1_PlainStack']
    results.append(row)
    gc.collect()

df = pd.DataFrame(results).sort_values(['dataset', 'condition', 'fold'])
df.to_csv(OUTPUT_DIR / 'gate_results_full_image.csv', index=False)

f1_cols = ['f1_RF', 'f1_XGB', 'f1_ET', 'f1_Voting', 'f1_PlainStack', 'f1_DAS']
summary = df.groupby(['dataset', 'condition'])[f1_cols + ['mean_pairwise_kappa']].mean().round(4).reset_index()
summary.to_csv(OUTPUT_DIR / 'gate_summary_by_condition_image.csv', index=False)

DROP_CONDS = ['symmetric_40', 'asymmetric_40', 'pairflip_40', 'c10n_aggre_0', 'c10n_random1_0', 'c10n_worse_0']
drop_rows = []
for key in datasets:
    sub = summary[summary['dataset'] == key].set_index('condition')
    if 'clean_0' not in sub.index:
        continue
    clean = sub.loc['clean_0']
    for cond in DROP_CONDS:
        if cond not in sub.index:
            continue
        for m in f1_cols:
            drop_rows.append({'dataset': key, 'condition': cond, 'method': m[3:], 'f1_clean': round(float(clean[m]), 4), 'f1_noisy': round(float(sub.loc[cond, m]), 4), 'rel_drop_pct': round(float((clean[m] - sub.loc[cond, m]) / clean[m] * 100), 2)})
drops = pd.DataFrame(drop_rows)
drops.to_csv(OUTPUT_DIR / 'gate_relative_drops_image.csv', index=False)

noisy = df[df['condition'] != 'clean_0']
NOISE_TYPES = ['symmetric', 'asymmetric', 'pairflip', 'c10n_aggre', 'c10n_random1', 'c10n_worse']
wil_rows = []
for key in datasets:
    for nt in NOISE_TYPES:
        deltas = noisy[(noisy['dataset'] == key) & (noisy['noise_type'] == nt)]['das_delta'].values
        if len(deltas) < 5:
            continue
        try:
            stat, p = wilcoxon(deltas)
        except ValueError:
            stat, p = np.nan, np.nan
        wil_rows.append({'dataset': key, 'noise_type': nt, 'n': len(deltas), 'mean_das_delta': round(float(deltas.mean()), 4), 'wilcoxon_p': round(float(p), 4)})
wil = pd.DataFrame(wil_rows)
wil.to_csv(OUTPUT_DIR / 'gate_das_wilcoxon_image.csv', index=False)

sp_rows = []
noisy = noisy.copy()
noisy['stack_gap_vs_voting'] = noisy['f1_PlainStack'] - noisy['f1_Voting']
noisy['kappa_z'] = noisy.groupby('dataset')['mean_pairwise_kappa'].transform(lambda s: (s - s.mean()) / s.std())
noisy['gap_z'] = noisy.groupby('dataset')['stack_gap_vs_voting'].transform(lambda s: (s - s.mean()) / s.std())
for key in datasets:
    sub = noisy[noisy['dataset'] == key]
    rho, sp = spearmanr(sub['mean_pairwise_kappa'], sub['stack_gap_vs_voting'])
    sp_rows.append({'dataset': key, 'n': len(sub), 'spearman_kappa_vs_gap': round(float(rho), 4), 'p_value': round(float(sp), 4)})
rho, sp = spearmanr(noisy['kappa_z'], noisy['gap_z'])
sp_rows.append({'dataset': 'ALL_POOLED_STANDARDIZED', 'n': len(noisy), 'spearman_kappa_vs_gap': round(float(rho), 4), 'p_value': round(float(sp), 4)})
spdf = pd.DataFrame(sp_rows)
spdf.to_csv(OUTPUT_DIR / 'gate_kappa_gap_spearman_image.csv', index=False)
