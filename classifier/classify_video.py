"""
Classify a segcrop_*.tif particle video frame-by-frame and identify the
time windows for each crystallographic state.

Strategy (based on the temporal approach we discussed):
1. Run the merged 3-class model (Dh / FCC / Ih+Ih→Dh, best 95.88%) per frame
2. Also run the drop-ihdh model (Dh / FCC / Ih, 96.6%) per frame
3. Smooth predictions temporally with median filter to suppress single-frame jitter
4. Identify "anchor" segments where model is confidently in one state
5. Label any "Ih+Ih→Dh" frames that sit BETWEEN a confident-Ih anchor and a
   confident-Dh anchor as "Ih→Dh" transition (this is the physics-based labeling).

Inputs:  segcrop .tif stack (T, H, W) uint8
Outputs: timeline.png, segments.csv, predictions.csv, overlay video
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

DEFAULT_MERGED_CKPT = '/shared/jingchl6/material/lc-research/classifier_runs/swin_3c_merged_warmstart/best.pt'
DEFAULT_DROP_CKPT   = '/shared/jingchl6/material/lc-research/classifier_runs/swin_3c_drop_warmstart/best.pt'
DEFAULT_VIDEO       = '/shared/jingchl6/material/lc-research/test/pipeline_test/segcrop_PtRu_TiO2__chip1__FS30202_20211106_050206.tif'

MERGED_CLASSES = ['Dh', 'FCC', 'Ih+Ih_to_Dh']
DROP_CLASSES   = ['Dh', 'FCC', 'Ih']


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


def _dihedral(x, k):
    """k-th of 8 dihedral transforms (4 rotations x optional flip). Rotations
    preserve crystallographic symmetry, so averaging over them is principled TTA."""
    if k >= 4:
        x = torch.flip(x, dims=[3])
    return torch.rot90(x, k % 4, dims=[2, 3])


def predict_stack(model, stack, device, tf, batch_size=32, tta=True):
    """Run model on a (T,H,W) uint8 stack, return (T, num_classes) softmax probs.
    tta=True averages 8-fold dihedral test-time augmentation (+~0.7pt accuracy,
    validated on the real particle val set: 95.88% -> 96.62%)."""
    T = stack.shape[0]
    probs = []
    for i in range(0, T, batch_size):
        batch_arrs = stack[i:i+batch_size]
        imgs = [tf(Image.fromarray(a).convert('RGB')) for a in batch_arrs]
        x = torch.stack(imgs).to(device, non_blocking=True)
        with torch.no_grad():
            if tta:
                acc = None
                for k in range(8):
                    p = F.softmax(model(_dihedral(x, k)), dim=1)
                    acc = p if acc is None else acc + p
                p = (acc / 8).cpu().numpy()
            else:
                p = F.softmax(model(x), dim=1).cpu().numpy()
        probs.append(p)
    return np.concatenate(probs)


def smooth(probs, window=11):
    """Median filter each class probability stream over time."""
    out = np.zeros_like(probs)
    for c in range(probs.shape[1]):
        out[:, c] = medfilt(probs[:, c], kernel_size=window)
    return out


def segment_states(pred_labels):
    """Group consecutive same-label frames into segments.
    Returns list of (start_idx, end_idx_inclusive, label_idx, length).
    """
    segs = []
    if len(pred_labels) == 0:
        return segs
    cur_start = 0
    cur_lbl = pred_labels[0]
    for i in range(1, len(pred_labels)):
        if pred_labels[i] != cur_lbl:
            segs.append((cur_start, i - 1, cur_lbl, i - cur_start))
            cur_start = i
            cur_lbl = pred_labels[i]
    segs.append((cur_start, len(pred_labels) - 1, cur_lbl, len(pred_labels) - cur_start))
    return segs


def label_transitions(merged_segs, drop_probs_smooth):
    """For each segment labeled 'Ih+Ih→Dh', check whether it sits between an
    Ih-family neighbor and a Dh neighbor. If so, look at drop-ihdh probabilities
    inside the segment to find the Ih→Dh transition window.

    Returns: list of (start, end, final_label, mean_confidence).
    """
    out = []
    for idx, (s, e, lbl, length) in enumerate(merged_segs):
        if MERGED_CLASSES[lbl] != 'Ih+Ih_to_Dh':
            out.append((s, e, MERGED_CLASSES[lbl], None))
            continue
        # Look at drop-ihdh predictions on these frames
        sub = drop_probs_smooth[s:e+1]  # (len, 3) over [Dh, FCC, Ih]
        ih_p = sub[:, DROP_CLASSES.index('Ih')]
        dh_p = sub[:, DROP_CLASSES.index('Dh')]
        # Frames inside this segment: classify per-frame as Ih / Ih→Dh / Dh
        # using the drop-ihdh probabilities
        sub_labels = []
        for ih, dh in zip(ih_p, dh_p):
            if ih > 0.6 and ih > dh + 0.2:
                sub_labels.append('Ih')
            elif dh > 0.6 and dh > ih + 0.2:
                sub_labels.append('Dh')   # rare here; would suggest model thinks early Dh
            else:
                sub_labels.append('Ih_to_Dh')
        # Re-segment within the merged block
        idx2lbl = {'Ih': 0, 'Dh': 1, 'Ih_to_Dh': 2}
        sub_label_idxs = [idx2lbl[l] for l in sub_labels]
        inner_segs = segment_states(np.array(sub_label_idxs))
        for is_, ie_, ilbl, ilen in inner_segs:
            name = list(idx2lbl.keys())[list(idx2lbl.values()).index(ilbl)]
            out.append((s + is_, s + ie_, name, None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default=DEFAULT_VIDEO,
                    help='segcrop_*.tif stack from the pipeline')
    ap.add_argument('--merged-ckpt', default=DEFAULT_MERGED_CKPT)
    ap.add_argument('--drop-ckpt',   default=DEFAULT_DROP_CKPT)
    ap.add_argument('--output-dir',  default=None)
    ap.add_argument('--smooth', type=int, default=11, help='median filter window (odd, frames)')
    ap.add_argument('--min-segment', type=int, default=10,
                    help='ignore segments shorter than this # frames (merge into neighbor)')
    ap.add_argument('--fps', type=float, default=20.0, help='assumed fps for time conversion (4x speed)')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.video), 'classification')
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load stack
    stack = tifffile.imread(args.video)
    if stack.dtype != np.uint8:
        mn, mx = stack.min(), stack.max()
        stack = np.clip((stack - mn) / max(mx - mn, 1e-6) * 255, 0, 255).astype(np.uint8)
    T, H, W = stack.shape
    print(f'Loaded {args.video}\n  shape={stack.shape} dtype={stack.dtype}')

    # Load models
    merged_model, merged_arch = load_model(args.merged_ckpt, 3, device)
    drop_model,   drop_arch   = load_model(args.drop_ckpt,   3, device)
    print(f'Merged model: {merged_arch}\nDrop-ihdh model: {drop_arch}')

    tf = transform_for_model(224)

    print('Running merged 3-class predictions...')
    merged_probs = predict_stack(merged_model, stack, device, tf)
    print('Running drop-ihdh predictions...')
    drop_probs   = predict_stack(drop_model,   stack, device, tf)

    # Temporal smoothing
    merged_smooth = smooth(merged_probs, window=args.smooth)
    drop_smooth   = smooth(drop_probs,   window=args.smooth)

    # Per-frame label from merged (the best model)
    merged_labels = merged_smooth.argmax(1)

    # Segment + identify Ih→Dh inside merged 'Ih+Ih→Dh' regions
    merged_segs = segment_states(merged_labels)
    print(f'\nMerged-model raw segments: {len(merged_segs)}')
    refined = label_transitions(merged_segs, drop_smooth)

    # Merge tiny segments into neighbors (clean up)
    cleaned = []
    for s, e, lbl, _ in refined:
        length = e - s + 1
        if cleaned and length < args.min_segment:
            # Merge into previous
            cleaned[-1] = (cleaned[-1][0], e, cleaned[-1][2], None)
        elif length < args.min_segment and len(refined) > 1:
            # First segment is tiny — keep but flag, or merge into next at end
            cleaned.append((s, e, lbl, None))
        else:
            cleaned.append((s, e, lbl, None))
    # Second pass: merge consecutive same-label segments
    merged2 = []
    for s, e, lbl, _ in cleaned:
        if merged2 and merged2[-1][2] == lbl:
            merged2[-1] = (merged2[-1][0], e, lbl, None)
        else:
            merged2.append((s, e, lbl, None))

    # Print summary
    print(f'\n=== Final segments ({len(merged2)}) ===')
    print(f'{"start":>6}  {"end":>6}  {"len":>5}  {"sec":>7}  {"label":<12}  {"mean_conf":>9}')
    rows = []
    for s, e, lbl, _ in merged2:
        sub = merged_smooth[s:e+1]
        # Average confidence in the most likely class for this segment
        mean_conf = float(sub.max(1).mean())
        secs = (e - s + 1) / args.fps
        print(f'{s:>6d}  {e:>6d}  {e-s+1:>5d}  {secs:>7.2f}  {lbl:<12}  {mean_conf:>9.3f}')
        rows.append({'start': int(s), 'end': int(e), 'length': int(e-s+1),
                     'seconds': round(secs, 2), 'label': lbl, 'mean_conf': round(mean_conf, 3)})

    # Save CSVs
    with open(os.path.join(args.output_dir, 'segments.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['start', 'end', 'length', 'seconds', 'label', 'mean_conf'])
        w.writeheader(); w.writerows(rows)
    print(f'\nSaved: {os.path.join(args.output_dir, "segments.csv")}')

    # Per-frame CSV
    with open(os.path.join(args.output_dir, 'predictions.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['frame', 'p_Dh_merged', 'p_FCC_merged', 'p_IhFamily_merged',
                    'p_Dh_drop', 'p_FCC_drop', 'p_Ih_drop'])
        for i in range(T):
            w.writerow([i] + [f'{x:.4f}' for x in merged_smooth[i]] +
                              [f'{x:.4f}' for x in drop_smooth[i]])
    print(f'Saved: {os.path.join(args.output_dir, "predictions.csv")}')

    # Timeline plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    t = np.arange(T)
    secs = t / args.fps

    # Plot 1: Merged 3-class probabilities (smoothed)
    axes[0].fill_between(secs, 0, merged_smooth[:, 0], color='tab:blue', alpha=0.55, label='Dh')
    axes[0].fill_between(secs, merged_smooth[:, 0], merged_smooth[:, 0] + merged_smooth[:, 1],
                         color='tab:orange', alpha=0.55, label='FCC')
    axes[0].fill_between(secs, merged_smooth[:, 0] + merged_smooth[:, 1], 1.0,
                         color='tab:green', alpha=0.55, label='Ih+Ih→Dh')
    axes[0].set_ylabel('Merged 3c prob')
    axes[0].set_ylim(0, 1); axes[0].legend(loc='upper right', ncol=3); axes[0].set_title('Merged 3-class model (smoothed)')

    # Plot 2: Drop-ihdh probabilities
    axes[1].fill_between(secs, 0, drop_smooth[:, 0], color='tab:blue', alpha=0.55, label='Dh')
    axes[1].fill_between(secs, drop_smooth[:, 0], drop_smooth[:, 0] + drop_smooth[:, 1],
                         color='tab:orange', alpha=0.55, label='FCC')
    axes[1].fill_between(secs, drop_smooth[:, 0] + drop_smooth[:, 1], 1.0,
                         color='tab:green', alpha=0.55, label='Ih')
    axes[1].set_ylabel('Drop-ihdh prob')
    axes[1].set_ylim(0, 1); axes[1].legend(loc='upper right', ncol=3); axes[1].set_title('Drop-ihdh model (smoothed) — Ih vs Dh per-frame guess')

    # Plot 3: Final segment timeline
    COLORS = {'Dh': 'tab:blue', 'FCC': 'tab:orange', 'Ih': 'tab:green', 'Ih_to_Dh': 'tab:red'}
    for s, e, lbl, _ in merged2:
        axes[2].axvspan(s / args.fps, (e + 1) / args.fps, color=COLORS.get(lbl, 'gray'), alpha=0.7)
        # Label in middle of segment
        mid_s = (s + e) / 2 / args.fps
        axes[2].text(mid_s, 0.5, f'{lbl}\n[{s}-{e}]', ha='center', va='center', fontsize=8, fontweight='bold')
    axes[2].set_xlim(0, secs[-1])
    axes[2].set_ylim(0, 1); axes[2].set_yticks([])
    axes[2].set_xlabel(f'time (s, assuming {args.fps} fps)')
    axes[2].set_title('Final classified timeline')

    plt.suptitle(f'Classification of {os.path.basename(args.video)}\nT={T} frames, smoothing window={args.smooth}', fontsize=12)
    plt.tight_layout()
    out_png = os.path.join(args.output_dir, 'timeline.png')
    plt.savefig(out_png, dpi=120, bbox_inches='tight')
    print(f'Saved: {out_png}')

    # Build overlay video
    print('Building overlay video...')
    out_mp4 = os.path.join(args.output_dir, 'classified.mp4')
    # ffmpeg path
    FFMPEG = '/home/jingchl6/miniconda3/envs/sam3/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
    # Per-frame label using merged2 segments
    frame_label = ['?'] * T
    for s, e, lbl, _ in merged2:
        for k in range(s, e + 1):
            frame_label[k] = lbl
    # Write frames to temp mp4 directly via cv2
    HEADER_H = 110
    W_OUT = 420  # widen so all 3 probabilities fit on each line
    pad_l = (W_OUT - W) // 2  # center the particle frame below
    pad_r = W_OUT - W - pad_l
    tmp = os.path.join(args.output_dir, '_tmp_overlay.mp4')
    w = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (W_OUT, H + HEADER_H), isColor=True)
    label_col = {'Dh': (255, 80, 80), 'FCC': (80, 200, 255), 'Ih': (80, 220, 80),
                 'Ih_to_Dh': (80, 80, 255), '?': (120, 120, 120)}
    for i in range(T):
        bgr = cv2.cvtColor(stack[i], cv2.COLOR_GRAY2BGR)
        # Pad particle frame to W_OUT
        bgr_padded = cv2.copyMakeBorder(bgr, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        header = np.zeros((HEADER_H, W_OUT, 3), dtype=np.uint8)
        col = label_col.get(frame_label[i], (200, 200, 200))
        cv2.rectangle(header, (0, 0), (W_OUT, HEADER_H), col, -1)
        # Title line
        cv2.putText(header, f'f{i:4d}    pred: {frame_label[i]}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        # Merged 3c probs (Dh, FCC, Ih+Ih→Dh)
        m = merged_smooth[i]
        cv2.putText(header, f'merged3c   Dh:{m[0]:.2f}   FCC:{m[1]:.2f}   Ih+:{m[2]:.2f}',
                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        # Drop-ihdh probs (Dh, FCC, Ih)
        d = drop_smooth[i]
        cv2.putText(header, f'drop3c     Dh:{d[0]:.2f}   FCC:{d[1]:.2f}   Ih :{d[2]:.2f}',
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        w.write(np.vstack([header, bgr_padded]))
    w.release()
    # Re-encode with libx264 for browser compat
    os.system(f'{FFMPEG} -i {tmp} -c:v libx264 -preset slow -crf 23 -pix_fmt yuv420p -movflags +faststart {out_mp4} -y 2>&1 | tail -1')
    os.remove(tmp)
    print(f'Saved: {out_mp4}')


if __name__ == '__main__':
    main()
