import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

EUROSAT_ROOT = Path('/kaggle/input/datasets/apollo2506/eurosat-dataset/EuroSAT')
OUTPUT_DIR = Path('/kaggle/working/prepared_data')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

covtype = fetch_covtype(random_state=RANDOM_SEED, shuffle=True)
X_tab = covtype.data.astype(np.float32)
y_tab = covtype.target.astype(np.int64) - 1

assert not np.isnan(X_tab).any()
assert not np.isinf(X_tab).any()
assert y_tab.min() == 0 and y_tab.max() == 6

class_names_tab = {0: 'Spruce/Fir', 1: 'Lodgepole Pine', 2: 'Ponderosa Pine', 3: 'Cottonwood/Willow', 4: 'Aspen', 5: 'Douglas-fir', 6: 'Krummholz'}

X_trainval, X_test_tab, y_trainval, y_test_tab = train_test_split(X_tab, y_tab, test_size=0.20, stratify=y_tab, random_state=RANDOM_SEED)
X_train_tab, X_val_tab, y_train_tab, y_val_tab = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=RANDOM_SEED)

scaler = StandardScaler()
X_train_tab[:, :10] = scaler.fit_transform(X_train_tab[:, :10])
X_val_tab[:, :10] = scaler.transform(X_val_tab[:, :10])
X_test_tab[:, :10] = scaler.transform(X_test_tab[:, :10])

np.savez_compressed(OUTPUT_DIR / 'covtype_splits.npz', X_train=X_train_tab, y_train=y_train_tab, X_val=X_val_tab, y_val=y_val_tab, X_test=X_test_tab, y_test=y_test_tab)

tab_metadata = {'dataset_name': 'Forest CoverType', 'source': 'UCI ML Repository', 'doi': '10.24432/C50K5N', 'n_samples_total': int(len(y_tab)), 'n_train': int(len(y_train_tab)), 'n_val': int(len(y_val_tab)), 'n_test': int(len(y_test_tab)), 'n_features': int(X_tab.shape[1]), 'n_classes': 7, 'class_names': class_names_tab, 'random_seed': RANDOM_SEED, 'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist()}
with open(OUTPUT_DIR / 'covtype_metadata.json', 'w') as f:
    json.dump(tab_metadata, f, indent=2, default=str)

assert EUROSAT_ROOT.exists()
class_folders = sorted(d for d in EUROSAT_ROOT.iterdir() if d.is_dir())
assert len(class_folders) == 10

records = []
for class_idx, class_folder in enumerate(class_folders):
    for img_path in sorted(class_folder.glob('*.jpg')):
        records.append({'filepath': str(img_path), 'class_name': class_folder.name, 'class_id': class_idx})
df_img = pd.DataFrame(records)
assert len(df_img) == 27000
assert df_img['class_id'].nunique() == 10

class_names_img = {i: folder.name for i, folder in enumerate(class_folders)}

train_val_df, test_df = train_test_split(df_img, test_size=0.20, stratify=df_img['class_id'], random_state=RANDOM_SEED)
train_df, val_df = train_test_split(train_val_df, test_size=0.25, stratify=train_val_df['class_id'], random_state=RANDOM_SEED)

train_df.reset_index(drop=True).to_csv(OUTPUT_DIR / 'eurosat_train.csv', index=False)
val_df.reset_index(drop=True).to_csv(OUTPUT_DIR / 'eurosat_val.csv', index=False)
test_df.reset_index(drop=True).to_csv(OUTPUT_DIR / 'eurosat_test.csv', index=False)

img_metadata = {'dataset_name': 'EuroSAT', 'source': 'IEEE JSTARS', 'doi': '10.1109/JSTARS.2019.2918242', 'n_samples_total': int(len(df_img)), 'n_train': int(len(train_df)), 'n_val': int(len(val_df)), 'n_test': int(len(test_df)), 'n_classes': 10, 'class_names': class_names_img, 'image_size': '64x64', 'channels': 3, 'random_seed': RANDOM_SEED}
with open(OUTPUT_DIR / 'eurosat_metadata.json', 'w') as f:
    json.dump(img_metadata, f, indent=2, default=str)
