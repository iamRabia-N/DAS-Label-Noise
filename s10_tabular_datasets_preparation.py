import subprocess
import sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ucimlrepo'],
               check=False)

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MIN_CLASS_SIZE = 100
SIZE_CAP = 50000

OUTPUT_DIR = Path('/kaggle/working/new_prepared_data')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_dataset(spec):
    for kwargs in [{'id': spec['uci_id']}, {'name': spec['uci_name']}]:
        try:
            repo = fetch_ucirepo(**kwargs)
            title = str(repo.metadata.name)
            X = repo.data.features
            y = repo.data.targets
            if X is None or y is None:
                raise ValueError("ucimlrepo returned no data payload")
            y = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y
            doi = getattr(repo.metadata, 'doi', None)
            src = {'source': 'UCI ML Repository', 'resolved_title': title,
                   'uci_id': int(repo.metadata.uci_id), 'doi': doi,
                   'fetch_method': f'ucimlrepo({kwargs})'}
            return X, y, src
        except Exception:
            pass

    bunch = fetch_openml(name=spec['openml_name'], version=1,
                         as_frame=True, parser='auto')
    title = str(bunch.details.get('name', spec['openml_name']))
    src = {'source': 'OpenML', 'resolved_title': title,
           'openml_id': bunch.details.get('id'), 'doi': None,
           'fetch_method': f"fetch_openml(name='{spec['openml_name']}')"}
    return bunch.data, bunch.target, src


DATASET_SPECS = [
    dict(key='drybean',   uci_id=602, uci_name='Dry Bean',
         openml_name='Dry-Bean-Dataset',
         expected=dict(min_rows=13000, min_features=16, n_classes=7)),
    dict(key='pendigits', uci_id=81,
         uci_name='Pen-Based Recognition of Handwritten Digits',
         openml_name='pendigits',
         expected=dict(min_rows=10000, min_features=16, n_classes=10)),
    dict(key='shuttle',   uci_id=148, uci_name='Statlog (Shuttle)',
         openml_name='shuttle',
         expected=dict(min_rows=50000, min_features=7, n_classes=7)),
    dict(key='satimage',  uci_id=146, uci_name='Statlog (Landsat Satellite)',
         openml_name='satimage',
         expected=dict(min_rows=6000, min_features=36, n_classes=6)),
    dict(key='har',       uci_id=240,
         uci_name='Human Activity Recognition Using Smartphones',
         openml_name='har',
         expected=dict(min_rows=10000, min_features=500, n_classes=6)),
    dict(key='isolet',    uci_id=54, uci_name='ISOLET',
         openml_name='isolet',
         expected=dict(min_rows=7000, min_features=600, n_classes=26)),
]

ISOLET_LETTERS = {str(i): chr(64 + i) for i in range(1, 27)}

PAIRFLIP_SPECS = {
    'drybean': {
        'mode': 'semantic',
        'pairs': [('DERMASON', 'SIRA'), ('CALI', 'BARBUNYA')],
        'label_names_official': None,
        'justification': ('Dermason and Sira are small, morphologically '
                          'similar varieties; Cali and Barbunya share size '
                          'and shape profiles (Koklu & Ozkan 2020).')
    },
    'pendigits': {
        'mode': 'semantic',
        'pairs': [('1', '7'), ('3', '8'), ('4', '9')],
        'label_names_official': None,
        'justification': 'Visually confusable handwritten digit pairs.'
    },
    'shuttle': {
        'mode': 'confusion',
        'pairs': None,
        'label_names_official': None,
        'justification': ('Class labels are opaque operational states; the '
                          'two most-confused class pairs of a clean-data '
                          'random forest define the map. Derivation stored '
                          'in metadata.')
    },
    'satimage': {
        'mode': 'semantic',
        'pairs': [('grey soil', 'damp grey soil'),
                  ('cotton crop', 'vegetation stubble')],
        'label_names_official': {'1': 'red soil', '2': 'cotton crop',
                                 '3': 'grey soil', '4': 'damp grey soil',
                                 '5': 'vegetation stubble',
                                 '7': 'very damp grey soil'},
        'justification': ('Grey soil vs damp grey soil differ only in '
                          'moisture; cotton crop and vegetation stubble are '
                          'both sparse-vegetation covers. Codes per official '
                          'Statlog documentation.')
    },
    'har': {
        'mode': 'semantic',
        'pairs': [('WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS'),
                  ('SITTING', 'STANDING')],
        'label_names_official': {'1': 'WALKING', '2': 'WALKING_UPSTAIRS',
                                 '3': 'WALKING_DOWNSTAIRS', '4': 'SITTING',
                                 '5': 'STANDING', '6': 'LAYING'},
        'justification': ('Stair directions share gait dynamics; sitting and '
                          'standing are both static postures. Codes per '
                          'official UCI HAR encoding.')
    },
    'isolet': {
        'mode': 'semantic',
        'pairs': [('B', 'D'), ('P', 'T'), ('M', 'N')],
        'label_names_official': ISOLET_LETTERS,
        'justification': ('Classic acoustically confusable spoken-letter '
                          'pairs (E-set and nasal consonants). Codes 1-26 '
                          'map to letters A-Z per official ISOLET docs.')
    },
}


def _norm(s):
    return str(s).strip().lower().replace('_', ' ').replace('-', ' ')


def resolve_class(wanted, class_names, official_map):
    w = _norm(wanted)
    hits = [c for c in class_names if _norm(c) == w]
    if not hits:
        hits = [c for c in class_names if w in _norm(c)]
    if not hits and official_map:
        codes = [k for k, v in official_map.items() if _norm(v) == w]
        if not codes:
            codes = [k for k, v in official_map.items() if w in _norm(v)]
        hits = [c for c in class_names if str(c) in codes]
    assert len(hits) == 1, (
        f"Pair name '{wanted}' matched {hits} among {list(class_names)} \u2014 "
        f"fix PAIRFLIP_SPECS before continuing.")
    return hits[0]


def derive_confusion_pairs(X_tr, y_tr, n_pairs=2, seed=RANDOM_SEED):
    Xa, Xb, ya, yb = train_test_split(X_tr, y_tr, test_size=0.3,
                                      stratify=y_tr, random_state=seed)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                random_state=seed,
                                class_weight='balanced_subsample')
    rf.fit(Xa, ya)
    cm = confusion_matrix(yb, rf.predict(Xb))
    sym = cm + cm.T
    np.fill_diagonal(sym, 0)
    pairs, used = [], set()
    order = np.dstack(np.unravel_index(np.argsort(-sym, axis=None),
                                       sym.shape))[0]
    for a, b in order:
        if a < b and a not in used and b not in used:
            pairs.append((int(a), int(b)))
            used.update([a, b])
        if len(pairs) == n_pairs:
            break
    return pairs, sym.tolist()


summary_rows = []

for spec in DATASET_SPECS:
    key = spec['key']

    X_raw, y_raw, src = fetch_dataset(spec)

    exp = spec['expected']
    n_cls_raw = int(pd.Series(y_raw).nunique())
    assert (len(X_raw) >= exp['min_rows']
            and X_raw.shape[1] >= exp['min_features']
            and n_cls_raw == exp['n_classes']), (
        f"{key}: fetched shape {X_raw.shape}, classes={n_cls_raw}; "
        f"expected >= ({exp['min_rows']}, {exp['min_features']}) with "
        f"{exp['n_classes']} classes \u2014 WRONG DATASET, stopping.")

    X = X_raw.apply(pd.to_numeric, errors='coerce')
    all_nan_cols = X.columns[X.isna().all()].tolist()
    if all_nan_cols:
        X = X.drop(columns=all_nan_cols)
    y = pd.Series(y_raw).astype(str).str.strip()
    try:
        _y_num = pd.to_numeric(y)
        if (_y_num == _y_num.round()).all():
            y = _y_num.round().astype(int).astype(str)
    except (ValueError, TypeError):
        pass
    keep = X.notna().all(axis=1) & y.notna()
    n_dropped = int((~keep).sum())
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)

    counts = y.value_counts()
    removed_classes = counts[counts < MIN_CLASS_SIZE].index.tolist()
    if removed_classes:
        keep_mask = ~y.isin(removed_classes)
        X, y = X[keep_mask].reset_index(drop=True), \
               y[keep_mask].reset_index(drop=True)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)
    n_classes = len(class_names)
    assert n_classes >= 4, f"{key}: only {n_classes} classes after filter"

    capped = False
    if len(y_enc) > SIZE_CAP:
        X, _, y_enc, _ = train_test_split(
            X, y_enc, train_size=SIZE_CAP,
            stratify=y_enc, random_state=RANDOM_SEED)
        X = X.reset_index(drop=True)
        capped = True

    X_np = X.values.astype(np.float32)
    X_tv, X_te, y_tv, y_te = train_test_split(
        X_np, y_enc, test_size=0.20, stratify=y_enc,
        random_state=RANDOM_SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tv, y_tv, test_size=0.25, stratify=y_tv,
        random_state=RANDOM_SEED)

    scaler = StandardScaler().fit(X_tr)
    X_tr, X_va, X_te = (scaler.transform(a).astype(np.float32)
                        for a in (X_tr, X_va, X_te))

    for nm, arr in [('train', X_tr), ('val', X_va), ('test', X_te)]:
        assert np.isfinite(arr).all(), f"{key}/{nm}: non-finite values!"
    for nm, lab in [('train', y_tr), ('val', y_va), ('test', y_te)]:
        assert set(np.unique(lab)) == set(range(n_classes)), \
            f"{key}/{nm}: missing classes in split!"

    pf = PAIRFLIP_SPECS[key]
    official = pf.get('label_names_official')
    if pf['mode'] == 'semantic':
        idx_pairs, cm_stored = [], None
        for a, b in pf['pairs']:
            ia = class_names.index(resolve_class(a, class_names, official))
            ib = class_names.index(resolve_class(b, class_names, official))
            idx_pairs.append((ia, ib))
    else:
        idx_pairs, cm_stored = derive_confusion_pairs(X_tr, y_tr)
    pair_map = {}
    for a, b in idx_pairs:
        pair_map[int(a)] = int(b)
        pair_map[int(b)] = int(a)

    name_lookup = official or {}
    named_pairs = [(name_lookup.get(str(class_names[a]), str(class_names[a])),
                    name_lookup.get(str(class_names[b]), str(class_names[b])))
                   for a, b in idx_pairs]

    np.savez_compressed(OUTPUT_DIR / f'{key}_splits.npz',
                        X_train=X_tr, y_train=y_tr,
                        X_val=X_va, y_val=y_va,
                        X_test=X_te, y_test=y_te)
    meta = {
        'dataset_key': key, 'source_info': src,
        'n_train': int(len(y_tr)), 'n_val': int(len(y_va)),
        'n_test': int(len(y_te)), 'n_features': int(X_tr.shape[1]),
        'n_classes': int(n_classes),
        'class_names': {int(i): c for i, c in enumerate(class_names)},
        'class_names_official': official,
        'preprocessing_rules': {
            'R1_rows_dropped_nonfinite': n_dropped,
            'R2_min_class_size': MIN_CLASS_SIZE,
            'R2_removed_classes': removed_classes,
            'R3_size_cap_applied': capped,
            'R4_split': '60/20/20 stratified',
            'R5_scaling': 'StandardScaler fit on train split only',
        },
        'pairflip': {'mode': pf['mode'], 'pairs_named': named_pairs,
                     'transition_map': pair_map,
                     'justification': pf['justification'],
                     'confusion_matrix_if_derived': cm_stored},
        'random_seed': RANDOM_SEED,
    }
    with open(OUTPUT_DIR / f'{key}_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    imb = counts.max() / counts[counts >= MIN_CLASS_SIZE].min()
    summary_rows.append({'dataset': key, 'source': src['source'],
                         'doi': src.get('doi'), 'rows': len(y_enc),
                         'features': X_tr.shape[1], 'classes': n_classes,
                         'imbalance': round(float(imb), 1),
                         'pairflip_mode': pf['mode']})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_DIR / 'new_datasets_summary.csv', index=False)
