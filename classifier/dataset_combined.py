"""
Combined dataset: real LC-TEM particles (merged-ihdh → 3 classes) + synthetic Pt particles.

Train: real merged-3c (10902) + all synth train (15000) = 25,902 images
Val:   real merged-3c only (1214) — synth val EXCLUDED so we get an honest measurement

Class indexing is unified to [Dh=0, FCC=1, Ih=2 (incl. Ih→Dh)] — matches both sources.
"""
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import tifffile
from PIL import Image

from dataset import DATA_ROOT as REAL_ROOT
from dataset_synthetic import DATA_ROOT_SYNTH as SYNTH_ROOT, build_synthetic_items, SYNTH_FOLDER_TO_IDX


COMBINED_CLASSES = ['Dh', 'FCC', 'Ih']


def build_combined_items(real_splits_json, synth_split='train'):
    """Build combined items list.
    real_splits_json: path to splits_particle/splits.json (4-class real data)
    synth_split: 'train' (default for training set), 'val' to use synth val too
    Returns: list of (path, class_idx, source) where source in {'real','synth'}
    """
    items = []
    with open(real_splits_json) as f:
        d = json.load(f)
    # Real: 4-class → merge Ih→Dh (idx=3) into Ih (idx=2)
    for p, y in d['train']:
        y3 = 2 if y == 3 else y
        items.append((p, y3, 'real'))
    if synth_split:
        synth_items = build_synthetic_items(split=synth_split)
        for p, y in synth_items:
            items.append((p, y, 'synth'))
    return items


def build_combined_val_items(real_splits_json):
    """Validation set: real val only (merged to 3-class)."""
    items = []
    with open(real_splits_json) as f:
        d = json.load(f)
    for p, y in d['val']:
        y3 = 2 if y == 3 else y
        items.append((p, y3, 'real'))
    return items


class CombinedDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, y, src = self.items[i]
        if src == 'real':
            arr = tifffile.imread(os.path.join(REAL_ROOT, rel))
            if arr.dtype != np.uint8:
                mn, mx = arr.min(), arr.max()
                arr = np.clip((arr - mn) / max(mx - mn, 1e-6) * 255, 0, 255).astype(np.uint8)
        else:
            arr = cv2.imread(os.path.join(SYNTH_ROOT, rel), cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(arr).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, y


def combined_sample_weights(items, n_classes=3, power=1.0, real_boost=1.0):
    """Per-sample weights for WeightedRandomSampler.
    power=1.0 fully equalizes classes.
    real_boost > 1.0 increases the per-sample weight for real images (so the model sees
    real-world artifacts more during training).
    """
    counts = np.zeros(n_classes, dtype=np.int64)
    for _, y, _ in items:
        counts[y] += 1
    inv = 1.0 / (counts ** power)
    weights = []
    for _, y, src in items:
        w = inv[y] * (real_boost if src == 'real' else 1.0)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.double), counts


def combined_class_weights(items, n_classes=3):
    counts = np.zeros(n_classes, dtype=np.int64)
    for _, y, _ in items:
        counts[y] += 1
    w = 1.0 / counts
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32), counts
