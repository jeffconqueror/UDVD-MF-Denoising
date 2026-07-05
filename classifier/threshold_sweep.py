"""
Inference + confidence-threshold sweep for the drop-ihdh model.

Idea: train on 3 clean classes (Dh, FCC, Ih). At inference, if max softmax
prob < threshold, label the particle as 'Ih→Dh' (the "transition" basin).

Outputs a table of (threshold → 4-class accuracy) to find the best operating point.
"""
import argparse, os, json
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import TEMParticleDataset, build_transforms, CLASSES

# Trained-on (model output order) vs full (target order)
TRAINED = ['Dh', 'FCC', 'Ih']           # 3 outputs from drop-ihdh model
FULL    = ['Dh', 'FCC', 'Ih', 'Ih to Dh']  # 4-class ground-truth labels
IH_TO_DH_IDX = FULL.index('Ih to Dh')   # = 3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint',
                    default='/shared/jingchl6/material/lc-research/classifier_runs/swin_3c_drop_warmstart/best.pt')
    ap.add_argument('--splits',
                    default='/shared/jingchl6/material/lc-research/classifier_runs/splits_particle/splits.json')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--output-dir',
                    default='/shared/jingchl6/material/lc-research/classifier_runs/swin_3c_drop_warmstart')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = timm.create_model(ckpt['arch'], pretrained=False, num_classes=len(TRAINED))
    model.load_state_dict(ckpt['state_dict'])
    model.to(device).eval()
    print(f'Loaded {ckpt["arch"]}, train val_acc (3-class)={ckpt.get("val_acc","?")}')

    # FULL val set — keep the original 4-class labels (don't remap)
    with open(args.splits) as f:
        d = json.load(f)
    val_items = [(p, y) for p, y in d['val']]   # y in [0,1,2,3] = [Dh,FCC,Ih,Ih to Dh]
    print(f'Val items: {len(val_items)}')
    ds = TEMParticleDataset(val_items, transform=build_transforms(train=False))
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    all_y, all_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)                       # (B, 3) over [Dh, FCC, Ih]
            prob = F.softmax(logits, dim=1).cpu().numpy()
            all_y.append(y.numpy()); all_prob.append(prob)
    y = np.concatenate(all_y)                       # 4-class true label
    prob = np.concatenate(all_prob)                 # (N, 3) probabilities

    # Sweep thresholds
    thresholds = np.arange(0.30, 0.96, 0.02)
    results = []
    for t in thresholds:
        argm = prob.argmax(1)            # 0,1,2 in TRAINED space (Dh, FCC, Ih)
        maxp = prob.max(1)
        # Below threshold → predict Ih→Dh; else use argmax (which already aligns: Dh=0, FCC=1, Ih=2)
        pred = np.where(maxp >= t, argm, IH_TO_DH_IDX)
        acc = (pred == y).mean()
        # Per-class recall
        per_cls_rec = {}
        for ci, name in enumerate(FULL):
            mask = (y == ci)
            per_cls_rec[name] = float((pred[mask] == ci).mean()) if mask.any() else 0.0
        # Confusion matrix
        cm = np.zeros((4, 4), dtype=int)
        for yi, pi in zip(y, pred): cm[yi, pi] += 1
        results.append({'t': float(t), 'acc': float(acc), 'per_cls': per_cls_rec, 'cm': cm.tolist()})

    # Print
    print(f'\n{"threshold":>9}  {"4-class acc":>11}  ' + '  '.join(f'{c[:6]:>7}' for c in FULL))
    for r in results:
        per = '  '.join(f'{r["per_cls"][c]:>7.3f}' for c in FULL)
        print(f'  {r["t"]:>7.2f}  {r["acc"]:>11.4f}  {per}')

    # Find best
    best = max(results, key=lambda r: r['acc'])
    print(f'\nBest: threshold={best["t"]:.2f}  4-class acc={best["acc"]:.4f}')
    print(f'  per-class: ' + '  '.join(f'{c}:{best["per_cls"][c]:.3f}' for c in FULL))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ts = [r['t'] for r in results]
    ax.plot(ts, [r['acc'] for r in results], 'k-', linewidth=2, label='4-class acc')
    for ci, c in enumerate(FULL):
        ax.plot(ts, [r['per_cls'][c] for r in results], '--', label=f'{c} recall', alpha=0.7)
    ax.axhline(0.83, color='gray', linestyle=':', alpha=0.5, label='v3a 4-class (0.830)')
    ax.axhline(0.959, color='gray', linestyle=':', alpha=0.5, label='merged 3c (0.959)')
    ax.axvline(best['t'], color='red', linestyle=':', alpha=0.5)
    ax.set_xlabel('confidence threshold (below → label as Ih→Dh)')
    ax.set_ylabel('accuracy / recall')
    ax.set_title(f'Drop-ihdh + threshold for Ih→Dh — best acc={best["acc"]:.3f} at t={best["t"]:.2f}')
    ax.legend(loc='lower center', ncol=3, fontsize=9)
    ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
    plt.tight_layout()
    out = os.path.join(args.output_dir, 'threshold_sweep.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f'Saved: {out}')

    # Best confusion matrix
    cm = np.array(best['cm'])
    fig, ax = plt.subplots(figsize=(7, 6))
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(FULL, rotation=20)
    ax.set_yticks(range(4)); ax.set_yticklabels(FULL)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'drop-ihdh + threshold={best["t"]:.2f}  acc={best["acc"]:.3f}')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{cm[i,j]}\n{cmn[i,j]*100:.1f}%', ha='center', va='center',
                    color='white' if cmn[i,j]>0.5 else 'black', fontsize=10)
    plt.tight_layout()
    cm_out = os.path.join(args.output_dir, 'confusion_matrix_threshold.png')
    plt.savefig(cm_out, dpi=120, bbox_inches='tight')
    print(f'Saved: {cm_out}')

    with open(os.path.join(args.output_dir, 'threshold_sweep.json'), 'w') as f:
        json.dump({'results': results, 'best': best, 'classes': FULL}, f, indent=2)


if __name__ == '__main__':
    main()
