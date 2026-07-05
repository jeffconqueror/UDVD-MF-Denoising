"""
LC-TEM nanoparticle phase classification dataset.

- 4 classes: Dh, FCC, Ih, Ih to Dh
- Each image is a single 512x512 uint8 TIFF centered on one particle.
- Each particle yields ~2 consecutive .tif frames (indices like 0019+0020 = 1 particle).
- Splits MUST group by particle — random per-image leaks particle identity.
"""
import os, glob, json, random, re
from collections import defaultdict
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

DATA_ROOT = '/shared/jingchl6/material/lc-research/Training Data'
CLASSES = ['Dh', 'FCC', 'Ih', 'Ih to Dh']
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}


def particle_id(fname):
    """Group .tif filenames into (series_name, pair_index) — pairs (1,2),(3,4),(19,20)... = same particle."""
    base = os.path.splitext(fname)[0]
    m = re.search(r'(.*?)([-_])(\d{3,5})$', base)
    if not m:
        return base
    series, sep, idx = m.group(1), m.group(2), int(m.group(3))
    return f'{series}{sep}p{(idx + 1) // 2}'


def build_splits(seed=44, val_frac=0.10, save_path=None, by_particle=True):
    """90/10 stratified split.
    by_particle=True (default): split by particle so all frames of one particle go to one side.
    by_particle=False: legacy per-image split — leaks identity, only use for v1 compat.
    """
    rng = random.Random(seed)
    train, val = [], []
    for cls in CLASSES:
        files = sorted(glob.glob(os.path.join(DATA_ROOT, cls, '*.tif')))
        if by_particle:
            groups = defaultdict(list)
            for f in files:
                groups[particle_id(os.path.basename(f))].append(f)
            pids = list(groups.keys())
            rng.shuffle(pids)
            n_val_p = max(1, int(round(len(pids) * val_frac)))
            for pid in pids[:n_val_p]:
                for f in groups[pid]:
                    val.append((os.path.relpath(f, DATA_ROOT), CLS2IDX[cls]))
            for pid in pids[n_val_p:]:
                for f in groups[pid]:
                    train.append((os.path.relpath(f, DATA_ROOT), CLS2IDX[cls]))
        else:
            rng.shuffle(files)
            n_val = max(1, int(round(len(files) * val_frac)))
            for f in files[:n_val]:
                val.append((os.path.relpath(f, DATA_ROOT), CLS2IDX[cls]))
            for f in files[n_val:]:
                train.append((os.path.relpath(f, DATA_ROOT), CLS2IDX[cls]))
    rng.shuffle(train)
    rng.shuffle(val)
    if save_path:
        with open(save_path, 'w') as f:
            json.dump({'classes': CLASSES, 'train': train, 'val': val,
                       'seed': seed, 'val_frac': val_frac, 'by_particle': by_particle}, f)
    return train, val


class TEMParticleDataset(Dataset):
    def __init__(self, items, root=DATA_ROOT, transform=None):
        self.items = items
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, y = self.items[i]
        arr = tifffile.imread(os.path.join(self.root, rel))
        if arr.dtype != np.uint8:
            mn, mx = arr.min(), arr.max()
            arr = np.clip((arr - mn) / max(mx - mn, 1e-6) * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert('RGB')  # replicate to 3 channels for ImageNet backbone
        if self.transform:
            img = self.transform(img)
        return img, y


def build_transforms(image_size=224, train=True, strength='normal'):
    """strength='normal' (default) or 'strong' (adds RandAugment + RandomErasing + heavier crop/jitter)."""
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        ops = [transforms.RandomResizedCrop(image_size,
                                            scale=(0.5, 1.0) if strength == 'strong' else (0.6, 1.0),
                                            ratio=(0.9, 1.1)),
               transforms.RandomHorizontalFlip(),
               transforms.RandomVerticalFlip(),
               transforms.RandomChoice([
                   transforms.RandomRotation((0, 0)),
                   transforms.RandomRotation((90, 90)),
                   transforms.RandomRotation((180, 180)),
                   transforms.RandomRotation((270, 270)),
               ])]
        if strength == 'strong':
            # RandAugment + heavier color jitter + erasing
            ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
            ops.append(transforms.ColorJitter(brightness=0.3, contrast=0.3))
        else:
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
        ops.extend([transforms.ToTensor(), norm])
        if strength == 'strong':
            ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)))
        return transforms.Compose(ops)
    else:
        return transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            norm,
        ])


def class_weights(items, n_classes=4):
    """Inverse-frequency weights for weighted CrossEntropyLoss."""
    counts = np.zeros(n_classes, dtype=np.int64)
    for _, y in items:
        counts[y] += 1
    w = 1.0 / counts
    w = w / w.sum() * n_classes  # normalize so mean weight ~1
    return torch.tensor(w, dtype=torch.float32), counts


def sample_weights(items, n_classes=4, power=1.0):
    """Per-sample weights for WeightedRandomSampler.
    power=1.0: full inverse freq → equalize classes (v1, aggressive)
    power=0.5: sqrt-inverse → gentle rebalance (recommended for class imbalance)
    """
    counts = np.zeros(n_classes, dtype=np.int64)
    for _, y in items:
        counts[y] += 1
    inv = 1.0 / (counts ** power)
    return torch.tensor([inv[y] for _, y in items], dtype=torch.double)
