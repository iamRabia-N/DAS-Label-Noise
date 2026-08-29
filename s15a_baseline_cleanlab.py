import subprocess
import sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'cleanlab'], check=True)

import json
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
from cleanlab.filter import find_label_issues

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATASETS = ['drybean', 'pendigits', 'satimage', 'shuttle', 'har', 'isolet', 'fmnist', 'cifar10']

TIME_BUDGET_HOURS = 11.5
total_start = time.time()

def budget_left():
    return (time.time() - total_start) < TIME_BUDGET_HOURS * 3600

OUTPUT_DIR = Path('/kaggle/working/cleanlab_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_csv = OUTPUT_DIR / 'cleanlab_results_full.csv'
prior = sorted(Path('/kaggle/input').rglob('cleanlab_results_full.csv'))
results = pd.read_csv(prior[0]).to_dict('records') if prior else []
done = set((r['dataset'], r['condition'], r['fold']) for r in results)

prob_files = {}
for p in sorted(Path('/kaggle/input').rglob('probs_*.npz')):
    if p.name not in prob_files:
        prob_files[p.name] = p

stopped = False
for key in DATASETS:
    split_hits = sorted(Path('/kaggle/input').rglob(f'{key}_splits.npz'))
    tune_hits = sorted(Path('/kaggle/input').rglob(f'{key}_best_hyperparameters.json'))
    if not split_hits or not tune_hits:
        continue
    data = np.load(split_hits[0])
    X_train, y_train = data['X_train'], data['y_train']
    with open(tune_hits[0]) as f:
        tuned = json.load(f)
    xgb_params = tuned['xgboost']['best_params'].copy()
    xgb_base = tuned['xgboost']['base_params'].copy()
    xgb_base.pop('device', None)
    n_classes = int(tuned['metadata']['n_classes'])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    splits = list(cv.split(X_train, y_train))

    unit_files = sorted(n for n in prob_files if n.startswith(f'probs_{key}_'))

    for fname in unit_files:
        parts = fname[:-4].split('_')
        cond = '_'.join(parts[2:-1])
        fold = int(parts[-1][1:])
        unit = (key, cond, fold)
        if unit in done:
            continue
        if not budget_left():
            stopped = True
            break

        t0 = time.time()
        z = np.load(prob_files[fname])
        y_noisy = z['y_train_noisy'].astype(int)
        y_val = z['y_val'].astype(int)
        noise_mask = z['noise_mask'].astype(bool)
        pred_probs = np.stack([z['oof_rf'], z['oof_xgb'], z['oof_et']]).mean(axis=0).astype(np.float64)
        pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)

        tr_idx, va_idx = splits[fold]
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        assert len(X_tr) == len(y_noisy), f"{fname}: fold alignment broken!"

        issues = find_label_issues(labels=y_noisy, pred_probs=pred_probs, filter_by='prune_by_noise_rate')
        keep = ~issues
        for c in range(n_classes):
            cls_idx = np.where(y_noisy == c)[0]
            if not keep[cls_idx].any() and len(cls_idx):
                top = cls_idx[np.argsort(-pred_probs[cls_idx, c])[:5]]
                keep[top] = True

        rs = RANDOM_SEED + fold
        clf = XGBClassifier(**{**xgb_base, **xgb_params, 'random_state': rs})
        clf.fit(X_tr[keep], y_noisy[keep])
        yp = clf.predict(X_va)

        n_flag = int(issues.sum())
        tp = int((issues & noise_mask).sum())
        row = {'dataset': key, 'condition': cond, 'fold': fold}
        row['actual_noise_pct'] = float(noise_mask.mean() * 100)
        row['n_train'] = int(len(y_noisy))
        row['n_flagged'] = n_flag
        row['flag_rate_pct'] = round(n_flag / len(y_noisy) * 100, 2)
        row['flag_precision'] = round(tp / n_flag, 4) if n_flag else None
        row['flag_recall'] = round(tp / int(noise_mask.sum()), 4) if noise_mask.sum() else None
        row['f1_CL_XGB'] = float(f1_score(y_val, yp, average='macro', zero_division=0))
        row['acc_CL_XGB'] = float(accuracy_score(y_val, yp))
        row['seconds'] = round(time.time() - t0, 1)
        results.append(row)
        done.add(unit)
        pd.DataFrame(results).to_csv(results_csv, index=False)
        del clf
        gc.collect()
    if stopped:
        break

pd.DataFrame(results).to_csv(results_csv, index=False)
df = pd.DataFrame(results)
if len(df):
    summ = df.groupby(['dataset', 'condition'])[['f1_CL_XGB', 'flag_rate_pct', 'flag_precision', 'flag_recall']].mean().round(4).reset_index()
    summ.to_csv(OUTPUT_DIR / 'cleanlab_summary.csv', index=False)
