import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = Path('/kaggle/input/datasets/rabianaz22/1-ensemble-noise-prepared-data-3rd-paper/prepared_data')
EUROSAT_ROOT = Path('/kaggle/input/datasets/apollo2506/eurosat-dataset/EuroSAT')
OUTPUT_DIR = Path('/kaggle/working/image_features')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

assert DATA_DIR.exists()
assert EUROSAT_ROOT.exists()

train_df = pd.read_csv(DATA_DIR / 'eurosat_train.csv')
val_df = pd.read_csv(DATA_DIR / 'eurosat_val.csv')
test_df = pd.read_csv(DATA_DIR / 'eurosat_test.csv')

class_names = sorted(train_df['class_name'].unique())
n_classes = len(class_names)

assert Path(train_df.iloc[0]['filepath']).exists()

resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
for param in feature_extractor.parameters():
    param.requires_grad = False
feature_extractor.eval()
feature_extractor = feature_extractor.to(DEVICE)

test_input = torch.randn(1, 3, 224, 224).to(DEVICE)
with torch.no_grad():
    test_output = feature_extractor(test_input)
feature_dim = test_output.view(1, -1).shape[1]

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class EuroSATFeatureDataset(Dataset):

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, row['class_id']


def extract_features(df, feature_extractor, transform, device, batch_size=128, num_workers=2):
    dataset = EuroSATFeatureDataset(df, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    all_features = []
    all_labels = []
    feature_extractor.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            features = feature_extractor(images)
            features = features.view(features.size(0), -1)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    features_array = np.vstack(all_features).astype(np.float32)
    labels_array = np.concatenate(all_labels).astype(np.int64)
    return features_array, labels_array


BATCH_SIZE = 128
NUM_WORKERS = 2

t_start = time.time()
X_train_img, y_train_img = extract_features(
    train_df, feature_extractor, preprocess, DEVICE,
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)
train_time = time.time() - t_start

t_start = time.time()
X_val_img, y_val_img = extract_features(
    val_df, feature_extractor, preprocess, DEVICE,
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)
val_time = time.time() - t_start

t_start = time.time()
X_test_img, y_test_img = extract_features(
    test_df, feature_extractor, preprocess, DEVICE,
    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)
test_time = time.time() - t_start

total_time = train_time + val_time + test_time

assert not np.isnan(X_train_img).any()
assert not np.isinf(X_train_img).any()
assert y_train_img.min() >= 0 and y_train_img.max() < n_classes

np.savez_compressed(OUTPUT_DIR / 'eurosat_features_train.npz', X=X_train_img, y=y_train_img)
np.savez_compressed(OUTPUT_DIR / 'eurosat_features_val.npz', X=X_val_img, y=y_val_img)
np.savez_compressed(OUTPUT_DIR / 'eurosat_features_test.npz', X=X_test_img, y=y_test_img)

metadata = {
    'model': 'ResNet18',
    'pretrained_weights': 'IMAGENET1K_V1',
    'feature_dim': int(feature_dim),
    'preprocessing': {
        'resize': '64x64 -> 224x224',
        'normalization': 'ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])',
    },
    'batch_size': BATCH_SIZE,
    'n_train': int(len(y_train_img)),
    'n_val': int(len(y_val_img)),
    'n_test': int(len(y_test_img)),
    'n_classes': int(n_classes),
    'class_names': class_names,
    'extraction_time_minutes': {
        'train': float(train_time / 60),
        'val': float(val_time / 60),
        'test': float(test_time / 60),
        'total': float(total_time / 60)
    },
    'random_seed': RANDOM_SEED,
    'feature_stats': {
        'mean': float(X_train_img.mean()),
        'std': float(X_train_img.std()),
        'min': float(X_train_img.min()),
        'max': float(X_train_img.max())
    }
}

with open(OUTPUT_DIR / 'extraction_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
