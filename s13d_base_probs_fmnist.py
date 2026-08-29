import json
import time
import shutil
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
from s02_noise_injection import inject_symmetric_noise, inject_asymmetric_noise

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATASETS = ['fmnist']

TIME_BUDGET_HOURS = 10.5
total_start = time.time()

def budget_left():
    return (time.time() - total_start) < TIME_BUDGET_HOURS * 3600

OUTPUT_DIR = Path('/kaggle/working/base_probs_image')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    tm = XGBClassifier(tree_method='hist', device='cuda', n_estimators=2)
    tm.fit(np.random.rand(100, 10), np.random.randint(0, 3, 100))
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

SYNTH_CONDITIONS = [('clean', 0.0)] + [('symmetric', l) for l in [0.1, 0.2, 0.3, 0.4]] + [('asymmetric', l) for l in [0.1, 0.2, 0.3, 0.4]] + [('pairflip', 0.4)]
C10N_CONDITIONS = [('c10n_aggre', 0.0), ('c10n_random1', 0.0), ('c10n_worse', 0.0)]
C10N_KEYMAP = {'c10n_aggre': 'aggre_label_train', 'c10n_random1': 'random_label1_train', 'c10n_worse': 'worse_label_train'}

manifest_csv = OUTPUT_DIR / 'probs_manifest.csv'
prior_manifest = sorted(Path('/kaggle/input').rglob('probs_manifest.csv'))
manifest = []
done_units = set()
for pm in prior_manifest:
    for m in pd.read_csv(pm).to_dict('records'):
        u = (m['dataset'], m['condition'], m['fold'])
        if m['dataset'] in DATASETS and u not in done_units:
            manifest.append(m)
            done_units.add(u)

stopped = False
for key in DATASETS:
    tune_hits = sorted(Path('/kaggle/input').rglob(f'{key}_best_hyperparameters.json'))
    if not tune_hits:
        continue
    with open(tune_hits[0]) as f:
        tuned = json.load(f)

    split_hits = sorted(Path('/kaggle/input').rglob(f'{key}_splits.npz'))
    assert split_hits, f"{key}_splits.npz not found - attach the s11 output"
    data = np.load(split_hits[0])
    X_train, y_train = data['X_train'], data['y_train']
    with open(split_hits[0].parent / f'{key}_metadata.json') as f:
        meta = json.load(f)
    n_classes = int(meta['n_classes'])
    pairflip_map = {int(k): int(v) for k, v in meta['pairflip']['transition_map'].items()}

    conditions = list(SYNTH_CONDITIONS)
    c10n = None
    if key == 'cifar10':
        c10n_hits = sorted(Path('/kaggle/input').rglob('cifar10n_labels.npz'))
        assert c10n_hits, "cifar10n_labels.npz not found"
        c10n = np.load(c10n_hits[0])
        conditions += C10N_CONDITIONS

    rf_params = tuned['random_forest']['best_params'].copy()
    if rf_params.get('max_depth') == 'None':
        rf_params['max_depth'] = None
    xgb_params = tuned['xgboost']['best_params'].copy()
    xgb_base = tuned['xgboost']['base_params'].copy()
    et_params = tuned['extra_trees']['params'].copy()
    if et_params.get('max_depth') == 'None':
        et_params['max_depth'] = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    splits = list(cv.split(X_train, y_train))

    for cond_name, level in conditions:
        cond_tag = f"{cond_name}_{int(level*100)}"
        for fold in range(5):
            unit = (key, cond_tag, fold)
            fname = f'probs_{key}_{cond_tag}_f{fold}.npz'
            out_path = OUTPUT_DIR / fname
            if unit in done_units and out_path.exists():
                continue
            if unit in done_units:
                src = sorted(Path('/kaggle/input').rglob(fname))
                if src:
                    shutil.copy(src[0], out_path)
                    continue
            if not budget_left():
                stopped = True
                break

            t0 = time.time()
            tr_idx, va_idx = splits[fold]
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr_clean, y_va = y_train[tr_idx], y_train[va_idx]

            noise_seed = RANDOM_SEED + fold * 100 + int(level * 1000)
            if cond_name == 'clean':
                y_tr_noisy = y_tr_clean.copy()
                noise_mask = np.zeros(len(y_tr_clean), dtype=bool)
            elif cond_name == 'symmetric':
                y_tr_noisy, noise_mask = inject_symmetric_noise(y_tr_clean, level, n_classes, noise_seed)
            elif cond_name == 'asymmetric':
                y_tr_noisy, noise_mask = inject_asymmetric_noise(y_tr_clean, level, n_classes, random_state=noise_seed)
            elif cond_name == 'pairflip':
                y_tr_noisy, noise_mask = inject_asymmetric_noise(y_tr_clean, level, n_classes, transition_map=pairflip_map, random_state=noise_seed)
            elif cond_name.startswith('c10n_'):
                y_tr_noisy = c10n[C10N_KEYMAP[cond_name]][tr_idx].astype(y_tr_clean.dtype)
                noise_mask = (y_tr_noisy != y_tr_clean)
            else:
                raise ValueError(cond_name)

            rs = RANDOM_SEED + fold

            def make_rf():
                return RandomForestClassifier(**rf_params, n_jobs=-1, random_state=rs, class_weight='balanced_subsample')

            def make_xgb():
                p = {**xgb_base, **xgb_params, 'random_state': rs}
                if GPU_AVAILABLE:
                    p['device'] = 'cuda'
                else:
                    p.pop('device', None)
                return XGBClassifier(**p)

            def make_et():
                p = et_params.copy()
                p['random_state'] = rs
                return ExtraTreesClassifier(**p)

            arrays = {}
            for name, maker in [('rf', make_rf), ('xgb', make_xgb), ('et', make_et)]:
                oof = cross_val_predict(maker(), X_tr, y_tr_noisy, cv=3, method='predict_proba', n_jobs=1)
                model = maker()
                model.fit(X_tr, y_tr_noisy)
                val = model.predict_proba(X_va)
                arrays[f'oof_{name}'] = oof.astype(np.float16)
                arrays[f'val_{name}'] = val.astype(np.float16)
                del model
                gc.collect()

            np.savez_compressed(out_path, y_train_noisy=y_tr_noisy.astype(np.int16), y_train_clean=y_tr_clean.astype(np.int16), y_val=y_va.astype(np.int16), noise_mask=noise_mask, **arrays)

            manifest.append({'dataset': key, 'condition': cond_tag, 'fold': fold, 'n_train': int(len(y_tr_noisy)), 'n_val': int(len(y_va)), 'n_classes': n_classes, 'actual_noise_pct': float(noise_mask.mean() * 100), 'seconds': round(time.time() - t0, 1)})
            done_units.add(unit)
            pd.DataFrame(manifest).to_csv(manifest_csv, index=False)
            gc.collect()
        if stopped:
            break
    if stopped:
        break

pd.DataFrame(manifest).to_csv(manifest_csv, index=False)
