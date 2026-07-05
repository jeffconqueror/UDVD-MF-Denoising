"""
Train an ImageNet-pretrained backbone for LC-TEM nanoparticle phase classification.

Default: ResNet50 baseline at 224x224, 20 epochs, weighted sampler for class imbalance.

Usage:
  python train.py --arch resnet50 --epochs 20 --batch-size 64
  python train.py --arch swin_tiny_patch4_window7_224 --epochs 20 --batch-size 64
"""
import argparse, os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
from tqdm import tqdm

from dataset import (build_splits, TEMParticleDataset, build_transforms,
                     class_weights, sample_weights, CLASSES)
from dataset_synthetic import (build_synthetic_items, SynthDataset,
                               build_synth_transforms, SYNTH_CLASSES)
from dataset_combined import (build_combined_items, build_combined_val_items,
                               CombinedDataset, combined_sample_weights,
                               combined_class_weights, COMBINED_CLASSES)


def evaluate(model, loader, device, classes=None):
    classes = classes or CLASSES
    model.eval()
    all_y, all_p = [], []
    tot_loss, tot_n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction='sum')
            tot_loss += loss.item(); tot_n += y.size(0)
            all_y.append(y.cpu().numpy())
            all_p.append(logits.argmax(1).cpu().numpy())
    y = np.concatenate(all_y); p = np.concatenate(all_p)
    acc = (y == p).mean()
    # per-class
    per_cls = {}
    for ci, name in enumerate(classes):
        mask = y == ci
        per_cls[name] = float((p[mask] == ci).mean()) if mask.any() else 0.0
    # confusion matrix
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for yi, pi in zip(y, p):
        cm[yi, pi] += 1
    return tot_loss / tot_n, acc, per_cls, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='resnet50')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--image-size', type=int, default=224)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=44)
    ap.add_argument('--output', default='/shared/jingchl6/material/lc-research/classifier_runs/resnet50')
    ap.add_argument('--splits-json', default=None,
                    help='Reuse a previously written splits.json (else create one)')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--balance', default='sampler+loss',
                    choices=['sampler+loss', 'sampler-sqrt', 'loss-only', 'none'],
                    help='class imbalance handling')
    ap.add_argument('--mode', default='4class',
                    choices=['4class', 'merge-ihdh', 'drop-ihdh'],
                    help='4class (default); merge-ihdh: Ih+Ih→Dh become one class; '
                         'drop-ihdh: drop Ih→Dh entirely')
    ap.add_argument('--amp', action='store_true',
                    help='enable mixed-precision training (saves ~40% memory)')
    ap.add_argument('--init-weights', default=None,
                    help='path to a .pt checkpoint to warm-start the backbone from '
                         '(head is re-initialized for the active num_classes)')
    ap.add_argument('--aug-strength', default='normal', choices=['normal', 'strong'],
                    help='normal: existing crop+flip+rotate+jitter; strong: + RandAugment + RandomErasing')
    ap.add_argument('--epoch-mult', type=float, default=1.0,
                    help='multiplier on samples-per-epoch (2.0 = each class effectively seen 2x more often per epoch via augmented draws)')
    ap.add_argument('--data', default='real', choices=['real', 'synthetic', 'combined'],
                    help='real / synthetic / combined (real merged-3c + synthetic; val=real only)')
    ap.add_argument('--real-boost', type=float, default=1.0,
                    help='for --data combined: multiplier on per-sample weight for real images (>1 favors real)')
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- splits ---
    if args.data == 'combined':
        if not args.splits_json:
            args.splits_json = '/shared/jingchl6/material/lc-research/classifier_runs/splits_particle/splits.json'
        train_items = build_combined_items(args.splits_json, synth_split='train')
        val_items   = build_combined_val_items(args.splits_json)
        n_real_train = sum(1 for _, _, s in train_items if s == 'real')
        n_synth_train = sum(1 for _, _, s in train_items if s == 'synth')
        print(f'Combined dataset: train={len(train_items)} (real={n_real_train}, synth={n_synth_train}) '
              f'val={len(val_items)} (real only)  classes: {COMBINED_CLASSES}')
    elif args.data == 'synthetic':
        train_items = build_synthetic_items(split='train')
        val_items   = build_synthetic_items(split='val')
        print(f'Synthetic dataset: train={len(train_items)} val={len(val_items)}  (classes: {SYNTH_CLASSES})')
    else:
        splits_path = args.splits_json or os.path.join(args.output, 'splits.json')
        if os.path.exists(splits_path):
            print(f'Loading existing splits: {splits_path}')
            with open(splits_path) as f:
                d = json.load(f)
            train_items = [(p, y) for p, y in d['train']]
            val_items   = [(p, y) for p, y in d['val']]
        else:
            train_items, val_items = build_splits(seed=args.seed, save_path=splits_path)
            print(f'Wrote splits: {splits_path}')

    # Apply mode: remap or filter labels
    if args.data == 'combined':
        active_classes = COMBINED_CLASSES  # ['Dh', 'FCC', 'Ih'] — already merged
    elif args.data == 'synthetic':
        active_classes = SYNTH_CLASSES
    elif args.mode == 'merge-ihdh':
        # Original indices: Dh=0, FCC=1, Ih=2, Ih to Dh=3 → merge 3 into 2
        train_items = [(p, 2 if y == 3 else y) for p, y in train_items]
        val_items   = [(p, 2 if y == 3 else y) for p, y in val_items]
        active_classes = ['Dh', 'FCC', 'Ih+Ih_to_Dh']
    elif args.mode == 'drop-ihdh':
        train_items = [(p, y) for p, y in train_items if y != 3]
        val_items   = [(p, y) for p, y in val_items   if y != 3]
        active_classes = ['Dh', 'FCC', 'Ih']
    else:
        active_classes = CLASSES
    print(f'Mode: {args.mode}  Active classes ({len(active_classes)}): {active_classes}')
    print(f'Train: {len(train_items)}  Val: {len(val_items)}')

    # --- datasets / loaders ---
    train_tf = build_transforms(args.image_size, train=True, strength=args.aug_strength)
    val_tf   = build_transforms(args.image_size, train=False)
    print(f'Augmentation: {args.aug_strength}')
    if args.data == 'combined':
        train_ds = CombinedDataset(train_items, transform=train_tf)
        val_ds   = CombinedDataset(val_items,   transform=val_tf)
    elif args.data == 'synthetic':
        train_ds = SynthDataset(train_items, transform=train_tf)
        val_ds   = SynthDataset(val_items,   transform=val_tf)
    else:
        train_ds = TEMParticleDataset(train_items, transform=train_tf)
        val_ds   = TEMParticleDataset(val_items,   transform=val_tf)

    num_samples = int(len(train_items) * args.epoch_mult)

    def _sw(power):
        if args.data == 'combined':
            sw_t, _ = combined_sample_weights(train_items, n_classes=len(active_classes),
                                              power=power, real_boost=args.real_boost)
            return sw_t
        return sample_weights(train_items, n_classes=len(active_classes), power=power)

    if args.balance == 'sampler+loss':
        sw = _sw(1.0)
        sampler = WeightedRandomSampler(sw, num_samples=num_samples, replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
        print(f'Using weighted sampler power=1.0 (fully equalize classes per epoch), num_samples={num_samples}'
              + (f', real_boost={args.real_boost}' if args.data == 'combined' else ''))
    elif args.balance == 'sampler-sqrt':
        sw = _sw(0.5)
        sampler = WeightedRandomSampler(sw, num_samples=num_samples, replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
        print(f'Using weighted sampler power=0.5 (sqrt-inverse), num_samples={num_samples}')
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, drop_last=True)
        print(f'Using natural class distribution (balance={args.balance})')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    # --- model ---
    print(f'Building {args.arch} (pretrained)...')
    model = timm.create_model(args.arch, pretrained=True, num_classes=len(active_classes))
    model = model.to(device)

    if args.init_weights:
        ckpt_w = torch.load(args.init_weights, map_location=device, weights_only=False)
        sd = ckpt_w['state_dict'] if 'state_dict' in ckpt_w else ckpt_w
        # Drop classifier head keys (output dim may differ from init checkpoint)
        skip_prefixes = ('head.', 'fc.', 'classifier.')
        filtered = {k: v for k, v in sd.items() if not any(k.startswith(p) for p in skip_prefixes)}
        # Also drop any leftover head with mismatched shape
        msd = model.state_dict()
        filtered = {k: v for k, v in filtered.items() if k in msd and v.shape == msd[k].shape}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        n_loaded = len(filtered); n_total = len(msd)
        print(f'Warm-start: loaded {n_loaded}/{n_total} tensors from {args.init_weights}')
        print(f'  missing (head): {[m for m in missing if any(m.startswith(p) for p in skip_prefixes)][:5]}')
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Params: {n_params/1e6:.1f}M')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    if args.data == 'combined':
        cw, counts = combined_class_weights(train_items, n_classes=len(active_classes))
    else:
        cw, counts = class_weights(train_items, n_classes=len(active_classes))
    print(f'Train class counts: {dict(zip(active_classes, counts.tolist()))}')
    if args.balance == 'sampler+loss' or args.balance == 'loss-only':
        print(f'Class weights (inverse-freq): {cw.tolist()}')
        crit = nn.CrossEntropyLoss(weight=cw.to(device), label_smoothing=0.05)
    else:  # 'sampler-sqrt' or 'none' → no loss weights (rebalancing handled by sampler or nothing)
        print('Loss: plain CrossEntropy (no class weights)')
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    print(f'Mixed precision: {"on" if args.amp else "off"}')

    # --- training loop ---
    best_acc = 0.0
    log = []
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_n = 0
        pbar = tqdm(train_loader, desc=f'Ep {ep:02d}', leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if args.amp:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = crit(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(x)
                loss = crit(logits, y)
                loss.backward()
                opt.step()
            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()
            running_n += y.size(0)
            pbar.set_postfix(loss=f'{running_loss/running_n:.3f}',
                             acc=f'{running_correct/running_n:.3f}')
        sched.step()
        train_loss = running_loss / running_n
        train_acc  = running_correct / running_n

        val_loss, val_acc, per_cls, cm = evaluate(model, val_loader, device, classes=active_classes)
        dt = time.time() - t0
        per_cls_str = '  '.join(f'{k}:{v:.3f}' for k, v in per_cls.items())
        print(f'Ep {ep:02d}  train_loss={train_loss:.3f} train_acc={train_acc:.3f}  '
              f'val_loss={val_loss:.3f} val_acc={val_acc:.3f}  '
              f'lr={opt.param_groups[0]["lr"]:.1e}  ({dt:.1f}s)')
        print(f'   per-class: {per_cls_str}')

        log.append({'epoch': ep, 'train_loss': train_loss, 'train_acc': train_acc,
                    'val_loss': val_loss, 'val_acc': val_acc, 'per_class': per_cls,
                    'confusion': cm.tolist(), 'lr': opt.param_groups[0]['lr']})
        with open(os.path.join(args.output, 'log.json'), 'w') as f:
            json.dump({'args': vars(args), 'log': log, 'classes': active_classes}, f, indent=2)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'state_dict': model.state_dict(), 'arch': args.arch,
                        'classes': active_classes, 'epoch': ep, 'val_acc': val_acc},
                       os.path.join(args.output, 'best.pt'))
            print(f'   ** new best val_acc={val_acc:.3f} → saved best.pt')

    print(f'\nBest val_acc: {best_acc:.3f}')


if __name__ == '__main__':
    main()
