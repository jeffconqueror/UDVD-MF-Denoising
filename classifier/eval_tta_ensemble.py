"""
Test-time augmentation (TTA) + model ensemble evaluation on the real particle-level
val set (3-class merged: Dh / FCC / Ih+IhDh).

Models evaluated (all 3-class, same output order [Dh, FCC, Ih]):
  A  swin_3c_merged_warmstart  (best real-only, 95.88%)
  B  swin_tiny_combined        (real+synth, 94.4%)
  C  swin_base_combined        (real+synth, 94.8%)

Configs reported:
  each model plain (sanity) and with 8-fold dihedral TTA
  ensembles (mean softmax): A+B, A+C, B+C, A+B+C — plain and TTA
"""
import os, json, itertools
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import timm

from dataset import TEMParticleDataset, build_transforms

CLASSES = ['Dh', 'FCC', 'Ih+IhDh']
SPLITS = '/shared/jingchl6/material/lc-research/classifier_runs/splits_particle/splits.json'
RUNS = '/shared/jingchl6/material/lc-research/classifier_runs'
MODELS = {
    'A_merged_real': f'{RUNS}/swin_3c_merged_warmstart/best.pt',
    'B_tiny_combined': f'{RUNS}/swin_tiny_combined/best.pt',
    'C_base_combined': f'{RUNS}/swin_base_combined/best.pt',
}


def load_model(p, device):
    ck = torch.load(p, map_location=device, weights_only=False)
    m = timm.create_model(ck['arch'], pretrained=False, num_classes=3)
    m.load_state_dict(ck['state_dict'])
    return m.to(device).eval()


def dihedral_batch(x, k):
    """Apply k-th of 8 dihedral transforms to a batch (B,C,H,W)."""
    if k >= 4:
        x = torch.flip(x, dims=[3])
    return torch.rot90(x, k % 4, dims=[2, 3])


def main():
    device = torch.device('cuda')
    with open(SPLITS) as f:
        d = json.load(f)
    val_items = [(p, 2 if y == 3 else y) for p, y in d['val']]
    y_true = np.array([y for _, y in val_items])
    print(f'Val: {len(val_items)} images (real, particle-level, 3-class merged)')

    ds = TEMParticleDataset(val_items, transform=build_transforms(224, train=False))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # collect probs per model: plain and TTA
    probs_plain, probs_tta = {}, {}
    for name, path in MODELS.items():
        m = load_model(path, device)
        pp, pt = [], []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device, non_blocking=True)
                pp.append(F.softmax(m(x), 1).cpu())
                acc = torch.zeros(x.size(0), 3)
                for k in range(8):
                    acc += F.softmax(m(dihedral_batch(x, k)), 1).cpu()
                pt.append(acc / 8)
        probs_plain[name] = torch.cat(pp).numpy()
        probs_tta[name] = torch.cat(pt).numpy()
        del m; torch.cuda.empty_cache()
        a0 = (probs_plain[name].argmax(1) == y_true).mean()
        a1 = (probs_tta[name].argmax(1) == y_true).mean()
        print(f'{name}:  plain={a0*100:.2f}%   TTA={a1*100:.2f}%')

    print('\n=== Ensembles (mean softmax) ===')
    names = list(MODELS)
    best = (0, '')
    for r in (2, 3):
        for combo in itertools.combinations(names, r):
            for src, tag in ((probs_plain, 'plain'), (probs_tta, 'TTA')):
                p = np.mean([src[n] for n in combo], axis=0)
                acc = (p.argmax(1) == y_true).mean()
                label = '+'.join(c.split('_')[0] for c in combo) + f' ({tag})'
                print(f'  {label:22s}: {acc*100:.2f}%')
                if acc > best[0]:
                    best = (acc, label)
    print(f'\nBEST: {best[1]} = {best[0]*100:.2f}%   (single-model baseline 95.88%)')


if __name__ == '__main__':
    main()
