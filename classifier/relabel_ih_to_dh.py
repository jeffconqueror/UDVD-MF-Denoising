"""
Use the trained Swin-Tiny v3a to score every Ih to Dh image and propose
a relabeling: which ones the model thinks are clearly Ih, which clearly Dh,
which uncertain.

Outputs:
  - relabel.csv     — per-image predictions + softmax
  - relabel_hist.png — histogram of v3a confidence on Ih to Dh class
  - relabel_examples.png — grid of high/low confidence examples
  - proposal.json   — proposed new labels per (rel_path → new_class_idx)

Original folder is NOT modified — proposal can be applied by writing a new splits.json.
"""
import argparse, os, glob, json, csv
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from dataset import TEMParticleDataset, build_transforms, CLASSES, DATA_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint',
                    default='/shared/jingchl6/material/lc-research/classifier_runs/swin_tiny_v3a/best.pt')
    ap.add_argument('--target-class', default='Ih to Dh',
                    help='Which folder to re-score and propose relabeling for')
    ap.add_argument('--ih-prob-thresh', type=float, default=0.60,
                    help='If P(Ih) >= thresh → relabel as Ih')
    ap.add_argument('--dh-prob-thresh', type=float, default=0.60,
                    help='If P(Dh) >= thresh → relabel as Dh')
    ap.add_argument('--output-dir',
                    default='/shared/jingchl6/material/lc-research/classifier_runs/relabel_ih_to_dh')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load v3a checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arch = ckpt['arch']
    model = timm.create_model(arch, pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device).eval()
    print(f'Loaded {arch} (trained val_acc={ckpt.get("val_acc", "?")})')

    # Build dataset for target class
    files = sorted(glob.glob(os.path.join(DATA_ROOT, args.target_class, '*.tif')))
    print(f'{args.target_class}: {len(files)} images to score')
    items = [(os.path.relpath(f, DATA_ROOT), CLASSES.index(args.target_class)) for f in files]
    ds = TEMParticleDataset(items, transform=build_transforms(train=False))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    all_probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs)  # (N, 4) order: [Dh, FCC, Ih, Ih to Dh]

    # Propose relabeling
    cls = {n: i for i, n in enumerate(CLASSES)}
    p_dh, p_fcc, p_ih, p_itd = probs[:, cls['Dh']], probs[:, cls['FCC']], probs[:, cls['Ih']], probs[:, cls['Ih to Dh']]
    proposed = []
    for i in range(len(files)):
        if p_ih[i] >= args.ih_prob_thresh:
            proposed.append('Ih')
        elif p_dh[i] >= args.dh_prob_thresh:
            proposed.append('Dh')
        else:
            proposed.append('Ih to Dh')  # keep as-is
    proposed = np.array(proposed)

    n_total = len(files)
    n_to_ih = int((proposed == 'Ih').sum())
    n_to_dh = int((proposed == 'Dh').sum())
    n_keep  = int((proposed == 'Ih to Dh').sum())
    print(f'\nProposal:')
    print(f'  → Ih:       {n_to_ih} ({n_to_ih/n_total*100:.1f}%)')
    print(f'  → Dh:       {n_to_dh} ({n_to_dh/n_total*100:.1f}%)')
    print(f'  → keep Ih→Dh: {n_keep} ({n_keep/n_total*100:.1f}%)')

    # CSV
    csv_path = os.path.join(args.output_dir, 'relabel.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'p_Dh', 'p_FCC', 'p_Ih', 'p_Ih_to_Dh', 'proposed_label'])
        for i, fp in enumerate(files):
            w.writerow([os.path.relpath(fp, DATA_ROOT),
                        f'{p_dh[i]:.4f}', f'{p_fcc[i]:.4f}',
                        f'{p_ih[i]:.4f}', f'{p_itd[i]:.4f}', proposed[i]])
    print(f'Saved: {csv_path}')

    # Histogram
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, name in zip(axes, CLASSES):
        ax.hist(probs[:, cls[name]], bins=30, edgecolor='black')
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'v3a P({name})  on {args.target_class} images')
        ax.set_xlabel('softmax prob'); ax.set_ylabel('count'); ax.set_xlim(0, 1)
    plt.tight_layout()
    hist_path = os.path.join(args.output_dir, 'relabel_hist.png')
    plt.savefig(hist_path, dpi=120, bbox_inches='tight'); plt.close()
    print(f'Saved: {hist_path}')

    # Example grid: 4 highest-confidence Ih, 4 highest Dh, 4 uncertain (closest to 0.33 / 0.33 split)
    def topk_idx(arr, k=6, descending=True):
        idx = np.argsort(arr)
        return idx[::-1][:k] if descending else idx[:k]
    ih_top = topk_idx(p_ih, 6)
    dh_top = topk_idx(p_dh, 6)
    uncertain = np.argsort(np.abs(p_ih - p_dh) + np.abs(p_ih - p_itd))[:6]  # most balanced

    fig, axes = plt.subplots(3, 6, figsize=(20, 11))
    rows = [('high P(Ih) → relabel Ih', ih_top, p_ih),
            ('high P(Dh) → relabel Dh', dh_top, p_dh),
            ('uncertain → keep Ih→Dh', uncertain, p_itd)]
    for r, (title, idxs, p_arr) in enumerate(rows):
        for c, i in enumerate(idxs):
            arr = tifffile.imread(files[i])
            axes[r, c].imshow(arr, cmap='gray')
            axes[r, c].set_title(f'p_Ih={p_ih[i]:.2f}  p_Dh={p_dh[i]:.2f}\np_Ih→Dh={p_itd[i]:.2f}', fontsize=8)
            axes[r, c].axis('off')
        axes[r, 0].set_ylabel(title, fontsize=11)
    plt.suptitle(f'v3a auto-relabel proposal on {args.target_class}', fontsize=14)
    plt.tight_layout()
    ex_path = os.path.join(args.output_dir, 'relabel_examples.png')
    plt.savefig(ex_path, dpi=110, bbox_inches='tight'); plt.close()
    print(f'Saved: {ex_path}')

    # Proposal JSON
    proposal = {'thresholds': {'ih': args.ih_prob_thresh, 'dh': args.dh_prob_thresh},
                'counts': {'to_Ih': n_to_ih, 'to_Dh': n_to_dh, 'keep': n_keep, 'total': n_total},
                'classes': CLASSES,
                'items': [{'file': os.path.relpath(files[i], DATA_ROOT),
                           'p': {'Dh': float(p_dh[i]), 'FCC': float(p_fcc[i]),
                                 'Ih': float(p_ih[i]), 'Ih_to_Dh': float(p_itd[i])},
                           'proposed_label': proposed[i]}
                          for i in range(n_total)]}
    json_path = os.path.join(args.output_dir, 'proposal.json')
    with open(json_path, 'w') as f:
        json.dump(proposal, f, indent=2)
    print(f'Saved: {json_path}')


if __name__ == '__main__':
    main()
