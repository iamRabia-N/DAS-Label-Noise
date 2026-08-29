import json
import gc
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets

from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert DEVICE.type == 'cuda', "Enable GPU accelerator for this script."

OUTPUT_DIR = Path('/kaggle/working/new_image_features')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = Path('/kaggle/working/raw_downloads')
RAW_DIR.mkdir(parents=True, exist_ok=True)

resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
for p in feature_extractor.parameters():
    p.requires_grad = False
feature_extractor.eval().to(DEVICE)

with torch.no_grad():
    probe = feature_extractor(torch.randn(1, 3, 224, 224).to(DEVICE))
FEATURE_DIM = probe.view(1, -1).shape[1]
assert FEATURE_DIM == 512, f"Unexpected feature dim {FEATURE_DIM}"

preprocess = transforms.Compose([
    transforms.Lambda(lambda im: im.convert('RGB')),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class WrappedDataset(Dataset):

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return preprocess(img), int(label), idx


def extract_features(base_dataset, tag, batch_size=256, num_workers=2):
    loader = DataLoader(WrappedDataset(base_dataset), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=True)
    n = len(base_dataset)
    feats = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    seen = np.zeros(n, dtype=bool)
    with torch.no_grad():
        for imgs, labs, idxs in loader:
            out = feature_extractor(imgs.to(DEVICE, non_blocking=True))
            out = out.view(out.size(0), -1).cpu().numpy()
            idxs = idxs.numpy()
            feats[idxs] = out
            labels[idxs] = labs.numpy()
            seen[idxs] = True
    assert seen.all(), f"{tag}: some samples were never processed!"
    assert np.isfinite(feats).all(), f"{tag}: non-finite features!"
    return feats, labels


FMNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

IMAGE_SPECS = {
    'fmnist': {
        'class_names': FMNIST_CLASSES,
        'source_info': {'source': 'Fashion-MNIST (Zalando Research)',
                        'reference': 'Xiao et al., arXiv:1708.07747',
                        'license': 'MIT',
                        'fetch_method': 'torchvision.datasets.FashionMNIST'},
        'pairflip': {
            'mode': 'semantic',
            'idx_pairs': [(0, 6), (2, 4), (5, 7)],
            'justification': ('T-shirt/top vs Shirt, Pullover vs Coat, and '
                              'Sandal vs Sneaker are the canonical visually '
                              'confusable Fashion-MNIST pairs.')},
    },
    'cifar10': {
        'class_names': CIFAR_CLASSES,
        'source_info': {'source': 'CIFAR-10 (Krizhevsky 2009)',
                        'reference': 'Krizhevsky, Learning Multiple Layers '
                                     'of Features from Tiny Images, 2009',
                        'fetch_method': 'torchvision.datasets.CIFAR10'},
        'pairflip': {
            'mode': 'semantic',
            'idx_pairs': [(3, 5), (1, 9), (4, 7)],
            'justification': ('cat vs dog, automobile vs truck, and deer vs '
                              'horse are canonical visually/semantically '
                              'confusable CIFAR-10 pairs.')},
    },
}


def split_and_save(key, X_pool, y_pool, spec, extra_train_labels=None):
    idx = np.arange(len(y_pool))
    idx_tv, idx_te = train_test_split(idx, test_size=0.20, stratify=y_pool,
                                      random_state=RANDOM_SEED)
    idx_tr, idx_va = train_test_split(idx_tv, test_size=0.25,
                                      stratify=y_pool[idx_tv],
                                      random_state=RANDOM_SEED)

    X_tr, X_va, X_te = X_pool[idx_tr], X_pool[idx_va], X_pool[idx_te]
    y_tr, y_va, y_te = y_pool[idx_tr], y_pool[idx_va], y_pool[idx_te]

    n_classes = len(spec['class_names'])
    for nm, lab in [('train', y_tr), ('val', y_va), ('test', y_te)]:
        assert set(np.unique(lab)) == set(range(n_classes)), \
            f"{key}/{nm}: missing classes!"

    np.savez_compressed(OUTPUT_DIR / f'{key}_splits.npz',
                        X_train=X_tr, y_train=y_tr, X_val=X_va, y_val=y_va,
                        X_test=X_te, y_test=y_te)
    np.savez_compressed(OUTPUT_DIR / f'{key}_split_indices.npz',
                        idx_train=idx_tr, idx_val=idx_va, idx_test=idx_te)

    pf = spec['pairflip']
    pair_map = {}
    for a, b in pf['idx_pairs']:
        pair_map[int(a)] = int(b)
        pair_map[int(b)] = int(a)
    named_pairs = [(spec['class_names'][a], spec['class_names'][b])
                   for a, b in pf['idx_pairs']]

    meta = {
        'dataset_key': key, 'source_info': spec['source_info'],
        'feature_extractor': 'ResNet18 IMAGENET1K_V1, fc removed, frozen '
                             '(identical to s04a)',
        'feature_dim': int(FEATURE_DIM),
        'pool_definition': 'official train partition, split 60/20/20 '
                           'stratified, seed 42 (CIFAR-10N labels exist '
                           'only for the train partition)',
        'n_train': int(len(y_tr)), 'n_val': int(len(y_va)),
        'n_test': int(len(y_te)), 'n_classes': int(n_classes),
        'class_names': {int(i): c for i, c in
                        enumerate(spec['class_names'])},
        'pairflip': {'mode': 'semantic', 'pairs_named': named_pairs,
                     'transition_map': pair_map,
                     'justification': pf['justification']},
        'random_seed': RANDOM_SEED,
    }

    if extra_train_labels is not None:
        noisy_out = {}
        for name, arr in extra_train_labels.items():
            noisy_out[f'{name}_train'] = arr[idx_tr]
            rate = float((arr[idx_tr] != y_tr).mean() * 100)
            meta.setdefault('cifar10n_realized_train_noise_pct', {})[name] = \
                round(rate, 2)
        np.savez_compressed(OUTPUT_DIR / f'{key}n_labels.npz', **noisy_out)

    with open(OUTPUT_DIR / f'{key}_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    return len(y_tr), len(y_va), len(y_te)


summary_rows = []

# Fashion-MNIST
fmnist = datasets.FashionMNIST(root=str(RAW_DIR), train=True, download=True)
assert len(fmnist) == 60000, f"Fashion-MNIST train size {len(fmnist)} != 60000"
X_fm, y_fm = extract_features(fmnist, 'fmnist')
assert set(np.unique(y_fm)) == set(range(10))
n_tr, n_va, n_te = split_and_save('fmnist', X_fm, y_fm, IMAGE_SPECS['fmnist'])
summary_rows.append({'dataset': 'fmnist', 'pool': 60000, 'features': 512,
                     'classes': 10, 'train': n_tr, 'val': n_va, 'test': n_te,
                     'noise_labels': 'synthetic only'})
del X_fm, fmnist
gc.collect()

# CIFAR-10
cifar = datasets.CIFAR10(root=str(RAW_DIR), train=True, download=True)
assert len(cifar) == 50000, f"CIFAR-10 train size {len(cifar)} != 50000"
X_cf, y_cf = extract_features(cifar, 'cifar10')
assert set(np.unique(y_cf)) == set(range(10))

# CIFAR-10N official human-noise labels
C10N_URL = ('https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/'
            'CIFAR-10_human.pt')
c10n_path = RAW_DIR / 'CIFAR-10_human.pt'
if not c10n_path.exists():
    urllib.request.urlretrieve(C10N_URL, c10n_path)

try:
    c10n = torch.load(c10n_path, map_location='cpu', weights_only=False)
except TypeError:
    c10n = torch.load(c10n_path, map_location='cpu')


def to_np(a):
    return a.numpy() if torch.is_tensor(a) else np.asarray(a)


clean_n = to_np(c10n['clean_label']).astype(np.int64)
assert clean_n.shape[0] == 50000, f"clean_label shape {clean_n.shape}"
assert np.array_equal(clean_n, y_cf), (
    "CIFAR-10N clean labels DO NOT match torchvision CIFAR-10 order \u2014 "
    "alignment broken, stopping. Do not proceed.")

noise_sets = {}
for k in ['aggre_label', 'random_label1', 'worse_label']:
    arr = to_np(c10n[k]).astype(np.int64)
    assert arr.shape[0] == 50000 and arr.min() >= 0 and arr.max() <= 9
    noise_sets[k] = arr

n_tr, n_va, n_te = split_and_save('cifar10', X_cf, y_cf,
                                  IMAGE_SPECS['cifar10'],
                                  extra_train_labels=noise_sets)
summary_rows.append({'dataset': 'cifar10', 'pool': 50000, 'features': 512,
                     'classes': 10, 'train': n_tr, 'val': n_va, 'test': n_te,
                     'noise_labels': 'synthetic + CIFAR-10N human'})
del X_cf, cifar
gc.collect()

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_DIR / 'new_image_datasets_summary.csv', index=False)
