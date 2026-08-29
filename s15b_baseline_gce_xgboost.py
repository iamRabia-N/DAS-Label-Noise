import json
import time
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATASETS = ['drybean', 'pendigits', 'satimage', 'shuttle', 'har', 'isolet', 'fmnist', 'cifar10']
GCE_Q = 0.7

TIME_BUDGET_HOURS = 11.5
total_start = time.time()

def budget_left():
    return (time.time() - total_start) < TIME_BUDGET_HOURS * 3600

OUTPUT_DIR = Path('/kaggle/working/robustloss_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results_csv = OUTPUT_DIR / 'robustloss_results_full.csv'
prior = sorted(Path('/kaggle/input').rglob('robustloss_results_full.csv'))
results = pd.read_csv(prior[0]).to_dict('records') if prior else []
done = set((r['dataset'], r['condition'], r['fold']) for r in results)

prob_files = {}
for p in sorted(Path('/kaggle/input').rglob('probs_*.npz')):
    if p.name not in prob_files:
        prob_files[p.name] = p


def softmax_rows(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def make_gce_obj(n_classes, q):
    def obj(predt, dtrain):
        y = dtrain.get_label().astype(int)
        z = predt.reshape(len(y), n_classes)
        p = softmax_rows(z)
        py = np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)
        w = py ** q
        grad = p.copy()
        grad[np.arange(len(y)), y] -= 1.0
        grad = grad * w[:, None]
        hess = np.maximum(2.0 * p * (1.0 - p) * w[:, None], 1e-6)
        return grad.reshape(-1), hess.reshape(-1)
    return obj


def train_gce(X_tr, y_tr, X_va, n_classes, xgb_params, rs):
    params = {'objective': 'multi:softprob', 'tree_method': 'hist', 'num_class': n_classes, 'disable_default_eval_metric': 1, 'seed': rs}
    params['max_depth'] = int(xgb_params['max_depth'])
    params['eta'] = float(xgb_params['learning_rate'])
    params['subsample'] = float(xgb_params['subsample'])
    params['colsample_bytree'] = float(xgb_params['colsample_bytree'])
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    booster = xgb.train(params, dtrain, num_boost_round=int(xgb_params['n_estimators']), obj=make_gce_obj(n_classes, GCE_Q))
    preds = booster.predict(xgb.DMatrix(X_va))
    if preds.ndim == 1:
        assert preds.size == len(X_va) * n_classes, f"unexpected predict shape {preds.shape}"
        preds = preds.reshape(-1, n_classes)
    return preds.argmax(axis=1)


def load_dataset(key):
    split_hits = sorted(Path('/kaggle/input').rglob(f'{key}_splits.npz'))
    tune_hits = sorted(Path('/kaggle/input').rglob(f'{key}_best_hyperparameters.json'))
    if not split_hits or not tune_hits:
        return None
    data = np.load(split_hits[0])
    with open(tune_hits[0]) as f:
        tuned = json.load(f)
    return data['X_train'], data['y_train'], tuned


val_key = None
for key in DATASETS:
    if f'probs_{key}_clean_0_f0.npz' in prob_files and load_dataset(key) is not None:
        val_key = key
        break
assert val_key is not None, "no dataset with clean_0_f0 prob file and data attached"
X_train, y_train, tuned = load_dataset(val_key)
n_classes = int(tuned['metadata']['n_classes'])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
tr_idx, va_idx = list(cv.split(X_train, y_train))[0]
z = np.load(prob_files[f'probs_{val_key}_clean_0_f0.npz'])
yp = train_gce(X_train[tr_idx], z['y_train_noisy'].astype(int), X_train[va_idx], n_classes, tuned['xgboost']['best_params'], RANDOM_SEED)
val_f1 = f1_score(z['y_val'].astype(int), yp, average='macro', zero_division=0)
assert val_f1 > 0.70, f"GCE objective FAILED validation (F1={val_f1:.4f}) - do not proceed"

stopped = False
for key in DATASETS:
    loaded = load_dataset(key)
    if loaded is None:
        continue
    X_train, y_train, tuned = loaded
    n_classes = int(tuned['metadata']['n_classes'])
    xgb_params = tuned['xgboost']['best_params']
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
        tr_idx, va_idx = splits[fold]
        assert len(tr_idx) == len(y_noisy), f"{fname}: fold alignment broken!"

        rs = RANDOM_SEED + fold
        yp = train_gce(X_train[tr_idx], y_noisy, X_train[va_idx], n_classes, xgb_params, rs)

        row = {'dataset': key, 'condition': cond, 'fold': fold}
        row['actual_noise_pct'] = float(noise_mask.mean() * 100)
        row['gce_q'] = GCE_Q
        row['f1_GCE_XGB'] = float(f1_score(y_val, yp, average='macro', zero_division=0))
        row['acc_GCE_XGB'] = float(accuracy_score(y_val, yp))
        row['seconds'] = round(time.time() - t0, 1)
        results.append(row)
        done.add(unit)
        pd.DataFrame(results).to_csv(results_csv, index=False)
        gc.collect()
    if stopped:
        break

pd.DataFrame(results).to_csv(results_csv, index=False)
df = pd.DataFrame(results)
if len(df):
    summ = df.groupby(['dataset', 'condition'])[['f1_GCE_XGB', 'acc_GCE_XGB']].mean().round(4).reset_index()
    summ.to_csv(OUTPUT_DIR / 'robustloss_summary.csv', index=False)
