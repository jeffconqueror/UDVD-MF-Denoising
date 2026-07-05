"""
Evaluate a trained classifier: confusion matrix + per-class metrics + sample mispredictions.

Usage:
  python evaluate.py --checkpoint /path/to/best.pt --splits-json /path/to/splits.json
"""
import argparse, os, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset import TEMParticleDataset, build_transforms, CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--splits-json', required=True)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    out_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    arch = ckpt['arch']
    print(f'arch={arch}  trained val_acc={ckpt.get("val_acc", "?")}')

    model = timm.create_model(arch, pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device).eval()

    with open(args.splits_json) as f:
        d = json.load(f)
    val_items = [(p, y) for p, y in d['val']]
    print(f'Val items: {len(val_items)}')

    val_ds = TEMParticleDataset(val_items, transform=build_transforms(args.image_size, train=False))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    all_y, all_p, all_prob = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            prob = F.softmax(logits, dim=1).cpu().numpy()
            all_y.append(y.numpy())
            all_p.append(logits.argmax(1).cpu().numpy())
            all_prob.append(prob)
    y = np.concatenate(all_y); p = np.concatenate(all_p); prob = np.concatenate(all_prob)
    acc = (y == p).mean()

    # Confusion matrix (rows=true, cols=pred)
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for yi, pi in zip(y, p):
        cm[yi, pi] += 1

    # Per-class precision/recall/f1
    metrics = {}
    for ci, name in enumerate(CLASSES):
        tp = cm[ci, ci]
        fn = cm[ci, :].sum() - tp
        fp = cm[:, ci].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-9)
        metrics[name] = {'precision': float(prec), 'recall': float(rec), 'f1': float(f1),
                         'support': int(cm[ci, :].sum())}
    macro_f1 = float(np.mean([m['f1'] for m in metrics.values()]))

    print(f'\nOverall val_acc: {acc:.4f}   macro-F1: {macro_f1:.4f}')
    print(f'\nPer-class metrics:')
    print(f'  {"class":<12} {"prec":>6} {"rec":>6} {"f1":>6} {"n":>6}')
    for name, m in metrics.items():
        print(f'  {name:<12} {m["precision"]:>6.3f} {m["recall"]:>6.3f} {m["f1"]:>6.3f} {m["support"]:>6d}')
    print(f'\nConfusion matrix (rows=true, cols=pred):')
    print('  ' + '  '.join(f'{c:>9}' for c in CLASSES))
    for ci, name in enumerate(CLASSES):
        print(f'  {name:<12}' + '  '.join(f'{cm[ci, j]:>9d}' for j in range(len(CLASSES))))

    # Save plot
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=30)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{arch}  val_acc={acc:.3f}  macro-F1={macro_f1:.3f}')
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            txt = f'{cm[i, j]}\n{cm_norm[i, j]*100:.1f}%'
            ax.text(j, i, txt, ha='center', va='center',
                    color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.045)
    plt.tight_layout()
    cm_path = os.path.join(out_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=120, bbox_inches='tight')
    print(f'\nSaved: {cm_path}')

    # Save metrics JSON
    out = {'arch': arch, 'val_acc': float(acc), 'macro_f1': macro_f1,
           'per_class': metrics, 'confusion': cm.tolist(), 'classes': CLASSES}
    with open(os.path.join(out_dir, 'eval.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {os.path.join(out_dir, "eval.json")}')


if __name__ == '__main__':
    main()
