"""
DINOv2 (ViT-S/14) linear probe + light fine-tune for LC-TEM phase classification.

We use facebookresearch/dinov2 via torch.hub. By default we keep the backbone
frozen and train only a linear head — this is fast (~5 min) and often
surprisingly strong on small science datasets.

Pass --finetune to unfreeze the backbone with a 10x smaller LR.

Usage:
  python train_dinov2.py --epochs 20 --batch-size 64 --gpu 3
  python train_dinov2.py --finetune --epochs 10 --batch-size 32 --gpu 3
"""
import argparse, os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import (build_splits, TEMParticleDataset, build_transforms,
                     class_weights, sample_weights, CLASSES)


def evaluate(backbone, head, loader, device, frozen):
    backbone.eval(); head.eval()
    all_y, all_p = [], []
    tot_loss, tot_n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            feats = backbone(x) if frozen else backbone(x)
            logits = head(feats)
            loss = F.cross_entropy(logits, y, reduction='sum')
            tot_loss += loss.item(); tot_n += y.size(0)
            all_y.append(y.cpu().numpy())
            all_p.append(logits.argmax(1).cpu().numpy())
    y = np.concatenate(all_y); p = np.concatenate(all_p)
    acc = (y == p).mean()
    per_cls = {}
    for ci, name in enumerate(CLASSES):
        mask = y == ci
        per_cls[name] = float((p[mask] == ci).mean()) if mask.any() else 0.0
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for yi, pi in zip(y, p):
        cm[yi, pi] += 1
    return tot_loss / tot_n, acc, per_cls, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=44)
    ap.add_argument('--finetune', action='store_true', help='unfreeze backbone (lr/10)')
    ap.add_argument('--output', default='/shared/jingchl6/material/lc-research/classifier_runs/dinov2')
    ap.add_argument('--splits-json', default=None)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- splits ---
    splits_path = args.splits_json or os.path.join(args.output, 'splits.json')
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            d = json.load(f)
        train_items = [(p, y) for p, y in d['train']]
        val_items   = [(p, y) for p, y in d['val']]
    else:
        train_items, val_items = build_splits(seed=args.seed, save_path=splits_path)
    print(f'Train: {len(train_items)}  Val: {len(val_items)}')

    train_tf = build_transforms(args.image_size, train=True)
    val_tf   = build_transforms(args.image_size, train=False)
    train_ds = TEMParticleDataset(train_items, transform=train_tf)
    val_ds   = TEMParticleDataset(val_items,   transform=val_tf)

    sw = sample_weights(train_items, n_classes=len(CLASSES))
    sampler = WeightedRandomSampler(sw, num_samples=len(train_items), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # --- DINOv2 backbone via torch.hub ---
    print('Loading DINOv2 ViT-S/14...')
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', source='github')
    backbone = backbone.to(device)
    feat_dim = backbone.embed_dim  # 384 for vits14
    print(f'Backbone embed_dim={feat_dim}')

    if not args.finetune:
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        print('Backbone frozen (linear probe)')

    head = nn.Linear(feat_dim, len(CLASSES)).to(device)

    cw, counts = class_weights(train_items, n_classes=len(CLASSES))
    print(f'Train class counts: {dict(zip(CLASSES, counts.tolist()))}')
    crit = nn.CrossEntropyLoss(weight=cw.to(device), label_smoothing=0.05)

    if args.finetune:
        opt = torch.optim.AdamW([
            {'params': head.parameters(),     'lr': args.lr},
            {'params': backbone.parameters(), 'lr': args.lr * 0.1},
        ], weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc = 0.0; log = []
    for ep in range(args.epochs):
        t0 = time.time()
        if args.finetune:
            backbone.train()
        head.train()
        running_loss, running_correct, running_n = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f'Ep {ep:02d}', leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            if args.finetune:
                feats = backbone(x)
            else:
                with torch.no_grad():
                    feats = backbone(x)
            logits = head(feats)
            loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            running_n += y.size(0)
            pbar.set_postfix(loss=f'{running_loss/running_n:.3f}',
                             acc=f'{running_correct/running_n:.3f}')
        sched.step()
        tr_loss, tr_acc = running_loss/running_n, running_correct/running_n
        val_loss, val_acc, per_cls, cm = evaluate(backbone, head, val_loader, device,
                                                  frozen=not args.finetune)
        dt = time.time() - t0
        per_cls_str = '  '.join(f'{k}:{v:.3f}' for k, v in per_cls.items())
        print(f'Ep {ep:02d}  train_loss={tr_loss:.3f} train_acc={tr_acc:.3f}  '
              f'val_loss={val_loss:.3f} val_acc={val_acc:.3f}  ({dt:.1f}s)')
        print(f'   per-class: {per_cls_str}')

        log.append({'epoch': ep, 'train_loss': tr_loss, 'train_acc': tr_acc,
                    'val_loss': val_loss, 'val_acc': val_acc, 'per_class': per_cls,
                    'confusion': cm.tolist()})
        with open(os.path.join(args.output, 'log.json'), 'w') as f:
            json.dump({'args': vars(args), 'log': log, 'classes': CLASSES}, f, indent=2)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'head': head.state_dict(),
                        'backbone_finetune': args.finetune,
                        'arch': 'dinov2_vits14',
                        'classes': CLASSES, 'epoch': ep, 'val_acc': val_acc},
                       os.path.join(args.output, 'best.pt'))
            print(f'   ** new best val_acc={val_acc:.3f}')

    print(f'\nBest val_acc: {best_acc:.3f}')


if __name__ == '__main__':
    main()
