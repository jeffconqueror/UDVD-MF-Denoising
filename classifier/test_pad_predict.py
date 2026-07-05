"""
Quick experiment: take a few PtRu frames, pad them to add vacuum-like border
(making the particle ~60% of the frame instead of 100%), then predict.

If the synth model's predictions change significantly (now finding Ih), we
have confirmation that the scale/framing mismatch is the cause and we should
either:
  - regenerate synth with particle-filling-frame variants
  - or use this padding trick at inference time
"""
import argparse
import numpy as np
import torch, torch.nn.functional as F
import tifffile, cv2
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

CLASSES = ['Dh', 'FCC', 'Ih']
BASE_CKPT = '/shared/jingchl6/material/lc-research/classifier_runs/swin_base_synth/best.pt'


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = timm.create_model(ckpt['arch'], pretrained=False, num_classes=3)
    m.load_state_dict(ckpt['state_dict'])
    return m.to(device).eval()


def transform_for_model(image_size=224):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize(image_size + 32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        norm,
    ])


def pad_to_match_synth(frame_u8, fill_fraction=0.60, out_size=256):
    """Resize particle so it takes `fill_fraction` of the output, pad with mean."""
    inner = int(round(out_size * fill_fraction))
    small = cv2.resize(frame_u8, (inner, inner), interpolation=cv2.INTER_AREA)
    pad = (out_size - inner) // 2
    mean_val = int(round(np.median(frame_u8)))  # use median so noise doesn't bias
    canvas = np.full((out_size, out_size), mean_val, dtype=np.uint8)
    canvas[pad:pad+inner, pad:pad+inner] = small
    return canvas


def predict_one(model, arr_u8, device, tf):
    img = Image.fromarray(arr_u8).convert('RGB')
    x = tf(img).unsqueeze(0).to(device, non_blocking=True)
    with torch.no_grad():
        p = F.softmax(model(x), dim=1).cpu().numpy()[0]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', default='/shared/jingchl6/material/lc-research/test/pipeline_test/segcrop_PtRu_TiO2__chip1__FS30202_20211106_050206.tif')
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    model = load_model(BASE_CKPT, device)
    tf = transform_for_model(224)

    stack = tifffile.imread(args.video)
    print(f'Loaded {stack.shape}')

    # Pick 8 frames spread across the video
    T = len(stack)
    sample_frames = np.linspace(50, T - 50, 8, dtype=int)

    fill_fractions = [1.00, 0.80, 0.60, 0.40]  # 1.00 = original (no padding)

    fig, axes = plt.subplots(len(sample_frames), len(fill_fractions), figsize=(16, 26))

    print(f'\nPer-frame predictions across pad fractions:')
    print(f'{"frame":>6}  ' + '  '.join(f'{int(f*100)}%fill->[Dh FCC Ih]' for f in fill_fractions))
    for r, fnum in enumerate(sample_frames):
        line = f'{fnum:>6d}  '
        for c, ff in enumerate(fill_fractions):
            if ff >= 1.0:
                padded = stack[fnum]
            else:
                padded = pad_to_match_synth(stack[fnum], fill_fraction=ff, out_size=256)
            p = predict_one(model, padded, device, tf)
            pred_label = CLASSES[p.argmax()]
            line += f'  {p[0]:.2f} {p[1]:.2f} {p[2]:.2f} -> {pred_label}'
            axes[r, c].imshow(padded, cmap='gray')
            axes[r, c].set_title(f'f{fnum} fill={int(ff*100)}%\n'
                                  f'Dh:{p[0]:.2f} FCC:{p[1]:.2f} Ih:{p[2]:.2f}\n'
                                  f'pred: {pred_label}', fontsize=9)
            axes[r, c].axis('off')
        print(line)
    plt.suptitle('Swin-Base synth predictions vs pad fraction (PtRu 050206)', fontsize=14)
    plt.tight_layout()
    out = '/home/jingchl6/.local/UDVD-MF-Denoising/results/pad_test.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
