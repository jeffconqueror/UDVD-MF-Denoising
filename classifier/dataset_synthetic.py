"""
Synthetic dataset for the 3 generated structure classes (ih / deca / fcc).

Reads PNG files from /shared/jingchl6/material/lc-research/generated/{class}/{train|val}/

Output indices are aligned with the segcrop classifier target schema:
  - 'deca' → Dh (0)
  - 'fcc'  → FCC (1)
  - 'ih'   → Ih (2)

So a model trained on this dataset can be applied directly to videos using the
same softmax output order our classify_video.py already expects.
"""
import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2

DATA_ROOT_SYNTH = '/shared/jingchl6/material/lc-research/generated'
# Folder name → target class index (matches the real-data 3-class output ordering)
SYNTH_FOLDER_TO_IDX = {'deca': 0, 'fcc': 1, 'ih': 2}
SYNTH_CLASSES = ['Dh', 'FCC', 'Ih']  # for human-readable reports


def build_synthetic_items(split='train'):
    items = []
    for folder, idx in SYNTH_FOLDER_TO_IDX.items():
        files = sorted(glob.glob(os.path.join(DATA_ROOT_SYNTH, folder, split, '*.png')))
        for f in files:
            items.append((os.path.relpath(f, DATA_ROOT_SYNTH), idx))
    return items


class SynthDataset(Dataset):
    def __init__(self, items, root=DATA_ROOT_SYNTH, transform=None):
        self.items = items
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, y = self.items[i]
        arr = cv2.imread(os.path.join(self.root, rel), cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(arr).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, y


def build_synth_transforms(image_size=224, train=True):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomChoice([
                transforms.RandomRotation((0, 0)),
                transforms.RandomRotation((90, 90)),
                transforms.RandomRotation((180, 180)),
                transforms.RandomRotation((270, 270)),
            ]),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            norm,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            norm,
        ])
