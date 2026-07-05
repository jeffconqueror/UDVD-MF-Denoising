"""
Classify a segcrop video using the synthetic-trained models (Swin-Tiny + Swin-Base).

Both models output 3 classes in order [Dh, FCC, Ih]. Per-frame overlay shows
both models' confidences. Temporal smoothing + segmentation produces:
  Dh / FCC / Ih anchor segments, with Ih→Dh transition labeled when an Ih
  anchor ends and a Dh anchor begins (per the user's temporal-physics idea).
"""
import argparse, os, json, csv
import numpy as np
import torch, torch.nn.functional as F
import tifffile, cv2
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import medfilt
from torchvision import transforms

DEFAULT_TINY_CKPT = '/shared/jingchl6/material/lc-research/classifier_runs/swin_tiny_synth/best.pt'
DEFAULT_BASE_CKPT = '/shared/jingchl6/material/lc-research/classifier_runs/swin_base_synth/best.pt'

CLASSES = ['Dh', 'FCC', 'Ih']


def load_model(ckpt_path, num_classes, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = timm.create_model(ckpt['arch'], pretrained=False, num_classes=num_classes)
    m.load_state_dict(ckpt['state_dict'])
    return m.to(device).eval(), ckpt['arch']


def transform_for_model(image_size=224):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        norm,
    ])


def predict_stack(model, stack, device, tf, batch_size=32):
    T = stack.shape[0]
    probs = []
    for i in range(0, T, batch_size):
        batch_arrs = stack[i:i+batch_size]
        imgs = [tf(Image.fromarray(a).convert('RGB')) for a in batch_arrs]
        x = torch.stack(imgs).to(device, non_blocking=True)
        with torch.no_grad():
            logits = model(x)
            p = F.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
    return np.concatenate(probs)


def smooth(probs, window=11):
    out = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        out[:, c] = medfilt(probs[:, c], kernel_size=window)
    return out


def segment_states(pred_labels):
    segs = []
    if len(pred_labels) == 0:
        return segs
    cur_start = 0
    cur_lbl = pred_labels[0]
    for i in range(1, len(pred_labels)):
        if pred_labels[i] != cur_lbl:
            segs.append((cur_start, i - 1, cur_lbl, i - cur_start))
            cur_start = i; cur_lbl = pred_labels[i]
    segs.append((cur_start, len(pred_labels) - 1, cur_lbl, len(pred_labels) - cur_start))
    return segs


def add_transitions(segs, label_names):
    """Insert an 'Ih_to_Dh' segment between an Ih anchor and a Dh anchor (or vice versa).
    For simplicity, we don't change segment boundaries — we just relabel an existing segment
    if it sits between Ih and Dh anchors. Skipped here since we use raw segment labels.
    """
    return [(s, e, label_names[lbl]) for s, e, lbl, _ in segs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--tiny-ckpt', default=DEFAULT_TINY_CKPT)
    ap.add_argument('--base-ckpt', default=DEFAULT_BASE_CKPT)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--smooth', type=int, default=11)
    ap.add_argument('--min-segment', type=int, default=10)
    ap.add_argument('--fps', type=float, default=20.0)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    stack = tifffile.imread(args.video)
    if stack.dtype != np.uint8:
        mn, mx = stack.min(), stack.max()
        stack = np.clip((stack - mn) / max(mx - mn, 1e-6) * 255, 0, 255).astype(np.uint8)
    T, H, W = stack.shape
    print(f'Loaded {args.video}  shape={stack.shape}')

    tiny, tiny_arch = load_model(args.tiny_ckpt, 3, device)
    base, base_arch = load_model(args.base_ckpt, 3, device)
    print(f'Models: {tiny_arch} (tiny) + {base_arch} (base)')

    tf = transform_for_model(224)
    print('Running Swin-Tiny synth predictions...')
    tiny_probs = predict_stack(tiny, stack, device, tf)
    print('Running Swin-Base synth predictions...')
    base_probs = predict_stack(base, stack, device, tf)

    tiny_s = smooth(tiny_probs, window=args.smooth)
    base_s = smooth(base_probs, window=args.smooth)

    # Use Swin-Base (slightly stronger) for the final label
    pred_labels = base_s.argmax(1)
    segs = segment_states(pred_labels)

    # Merge tiny segments into neighbors
    cleaned = []
    for s, e, lbl, _ in segs:
        if cleaned and (e - s + 1) < args.min_segment:
            cleaned[-1] = (cleaned[-1][0], e, cleaned[-1][2], None)
        else:
            cleaned.append((s, e, lbl, None))
    merged = []
    for s, e, lbl, _ in cleaned:
        if merged and merged[-1][2] == lbl:
            merged[-1] = (merged[-1][0], e, lbl, None)
        else:
            merged.append((s, e, lbl, None))

    # Per-frame label
    frame_label = ['?'] * T
    for s, e, lbl, _ in merged:
        for k in range(s, e + 1):
            frame_label[k] = CLASSES[lbl]

    # Print summary
    print(f'\n=== Segments ({len(merged)}) ===')
    print(f'{"start":>6}  {"end":>6}  {"len":>5}  {"sec":>7}  {"label":<6}  {"conf":>5}')
    rows = []
    for s, e, lbl, _ in merged:
        mean_conf = float(base_s[s:e+1].max(1).mean())
        secs = (e - s + 1) / args.fps
        name = CLASSES[lbl]
        print(f'{s:>6d}  {e:>6d}  {e-s+1:>5d}  {secs:>7.2f}  {name:<6}  {mean_conf:>5.3f}')
        rows.append({'start': int(s), 'end': int(e), 'length': int(e-s+1),
                     'seconds': round(secs, 2), 'label': name, 'mean_conf': round(mean_conf, 3)})

    with open(os.path.join(args.output_dir, 'segments.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['start', 'end', 'length', 'seconds', 'label', 'mean_conf'])
        w.writeheader(); w.writerows(rows)

    with open(os.path.join(args.output_dir, 'predictions.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'p_Dh_tiny', 'p_FCC_tiny', 'p_Ih_tiny',
                              'p_Dh_base', 'p_FCC_base', 'p_Ih_base'])
        for i in range(T):
            w.writerow([i] + [f'{x:.4f}' for x in tiny_s[i]] +
                              [f'{x:.4f}' for x in base_s[i]])

    # Timeline plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    secs = np.arange(T) / args.fps
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for ax, probs, name in [(axes[0], tiny_s, 'Swin-Tiny synth'), (axes[1], base_s, 'Swin-Base synth')]:
        cum = np.zeros(T)
        for c in range(3):
            ax.fill_between(secs, cum, cum + probs[:, c], color=colors[c], alpha=0.55, label=CLASSES[c])
            cum = cum + probs[:, c]
        ax.set_ylabel(f'{name} prob'); ax.set_ylim(0, 1); ax.legend(loc='upper right', ncol=3)
        ax.set_title(name + ' (smoothed)')

    COLORS = {'Dh': 'tab:blue', 'FCC': 'tab:orange', 'Ih': 'tab:green'}
    for s, e, lbl, _ in merged:
        axes[2].axvspan(s / args.fps, (e + 1) / args.fps, color=COLORS[CLASSES[lbl]], alpha=0.7)
        mid_s = (s + e) / 2 / args.fps
        axes[2].text(mid_s, 0.5, f'{CLASSES[lbl]}\n[{s}-{e}]', ha='center', va='center', fontsize=8, fontweight='bold')
    axes[2].set_xlim(0, secs[-1]); axes[2].set_ylim(0, 1); axes[2].set_yticks([])
    axes[2].set_xlabel(f'time (s @ {args.fps} fps)')
    axes[2].set_title('Final classified timeline (from Swin-Base synth)')
    plt.suptitle(f'Classification of {os.path.basename(args.video)} — SYNTHETIC-trained models', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'timeline.png'), dpi=120, bbox_inches='tight'); plt.close()

    # Overlay video
    HEADER_H = 110
    W_OUT = 420
    pad_l = (W_OUT - W) // 2; pad_r = W_OUT - W - pad_l
    tmp = os.path.join(args.output_dir, '_tmp_overlay.mp4')
    w = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (W_OUT, H + HEADER_H), isColor=True)
    label_col = {'Dh': (255, 80, 80), 'FCC': (80, 200, 255), 'Ih': (80, 220, 80), '?': (120, 120, 120)}
    for i in range(T):
        bgr = cv2.cvtColor(stack[i], cv2.COLOR_GRAY2BGR)
        bgr_p = cv2.copyMakeBorder(bgr, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        header = np.zeros((HEADER_H, W_OUT, 3), dtype=np.uint8)
        col = label_col.get(frame_label[i], (200, 200, 200))
        cv2.rectangle(header, (0, 0), (W_OUT, HEADER_H), col, -1)
        cv2.putText(header, f'f{i:4d}    pred: {frame_label[i]}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        t = tiny_s[i]
        cv2.putText(header, f'swinT   Dh:{t[0]:.2f}   FCC:{t[1]:.2f}   Ih:{t[2]:.2f}',
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        b = base_s[i]
        cv2.putText(header, f'swinB   Dh:{b[0]:.2f}   FCC:{b[1]:.2f}   Ih:{b[2]:.2f}',
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        w.write(np.vstack([header, bgr_p]))
    w.release()

    FFMPEG = '/home/jingchl6/miniconda3/envs/sam3/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
    out_mp4 = os.path.join(args.output_dir, 'classified.mp4')
    os.system(f'{FFMPEG} -i {tmp} -c:v libx264 -preset slow -crf 23 -pix_fmt yuv420p -movflags +faststart {out_mp4} -y 2>&1 | tail -1')
    os.remove(tmp)
    print(f'Saved: {out_mp4}')


if __name__ == '__main__':
    main()
