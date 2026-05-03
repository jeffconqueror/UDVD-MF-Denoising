"""
Inference-only UDVD-MF: load pretrained weights, run denoising on a new video.
No training — tests generalization of the trained model.
"""
import argparse, os, numpy as np, torch, torch.nn as nn, tifffile, cv2
from skimage import io
from torch.utils.data import Dataset, DataLoader
import data, utils, models
import warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class DataSet(Dataset):
    def __init__(self, filename, multiply=1):
        super().__init__()
        if filename.lower().endswith('.npy'):
            self.img = np.load(filename) * multiply
        else:
            self.img = io.imread(filename) * multiply

    def __len__(self):
        return self.img.shape[0]

    def __getitem__(self, index):
        if index < 2:
            out = np.concatenate((np.repeat(np.array([self.img[0]]), 2, axis=0), self.img[index:index+3]), axis=0)
        elif index > self.img.shape[0]-3:
            out = np.concatenate((self.img[index-2:index+1], np.repeat(np.array([self.img[-1]]), 2, axis=0)), axis=0)
        else:
            out = self.img[index-2:index+3]
        out = np.copy(out)
        return torch.Tensor(np.float32(out)).to(device)


def tiled_inference(model, frames_tensor, tile_size, device, overlap=32):
    C, H, W = frames_tensor.shape
    stride = tile_size - overlap
    out = torch.zeros(1, H, W, device='cpu')
    weight = torch.zeros(1, H, W, device='cpu')
    win_1d = torch.ones(tile_size)
    if overlap > 0:
        ramp = torch.linspace(0, 1, overlap)
        win_1d[:overlap] = ramp
        win_1d[-overlap:] = ramp.flip(0)
    win_2d = win_1d.unsqueeze(0) * win_1d.unsqueeze(1)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = min(y, H - tile_size); x1 = min(x, W - tile_size)
            y2, x2 = y1 + tile_size, x1 + tile_size
            tile_in = frames_tensor[:, y1:y2, x1:x2].unsqueeze(0).to(device)
            tile_out = model(tile_in).cpu()
            w = win_2d.clone()
            if y1 == 0: w[:overlap, :] = 1
            if x1 == 0: w[:, :overlap] = 1
            if y2 == H: w[-overlap:, :] = 1
            if x2 == W: w[:, -overlap:] = 1
            out[0, y1:y2, x1:x2] += tile_out[0, 0] * w
            weight[0, y1:y2, x1:x2] += w
    out /= weight.clamp(min=1e-8)
    return out


def main(args):
    print(f'Loading data: {args.data}')
    ds = DataSet(args.data, multiply=args.multiply)
    print(f'Frames: {len(ds)}, shape: {ds.img.shape}')

    model_args = argparse.Namespace(model='blind-video-net-5', channels=1, out_channels=1,
                                     bias=False, normal=False, blind_noise=False)
    model = models.build_model(model_args).to(device)

    # Load weights — handle DataParallel wrapping
    state_dict = torch.load(args.weights, map_location=device, weights_only=True)
    new_state_dict = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith('module.') else k
        new_state_dict[nk] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f'Loaded weights from {args.weights}')

    H, W = ds.img.shape[-2], ds.img.shape[-1]
    tile_size = 256
    use_tiled = (H > tile_size * 2 or W > tile_size * 2)
    if use_tiled:
        print(f'Using tiled inference (tile={tile_size})')

    denoised = np.zeros(ds.img.shape, dtype=np.float32)
    with torch.no_grad():
        for k in range(len(ds)):
            sample = ds[k]
            if use_tiled:
                o = tiled_inference(model, sample, tile_size, device)
            else:
                o = model(sample.unsqueeze(0))
            denoised[k] = o.cpu().numpy()
            if (k+1) % 100 == 0:
                print(f'  {k+1}/{len(ds)}')

    np.save(args.output, denoised)
    print(f'Saved: {args.output}  shape={denoised.shape}')

    # Save side-by-side comparison video
    if args.compare:
        raw = ds.img.astype(np.float32)
        T = len(raw)
        H, W = raw.shape[1], raw.shape[2]
        def norm_u8(x):
            mn, mx = x.min(), x.max()
            return np.clip(((x - mn) / (mx - mn + 1e-8) * 255), 0, 255).astype(np.uint8)
        cmp_path = args.output.replace('.npy', '_comparison.mp4')
        w = cv2.VideoWriter(cmp_path, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (W*2, H))
        for i in range(T):
            r = norm_u8(raw[i]); d = norm_u8(denoised[i])
            combined = np.concatenate([r, d], axis=1)
            bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
            cv2.putText(bgr, 'RAW', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            cv2.putText(bgr, 'DENOISED (transferred)', (W+20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
            cv2.putText(bgr, f'f{i}', (20, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
            w.write(bgr)
        w.release()
        print(f'Saved: {cmp_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--multiply', default=1, type=int)
    parser.add_argument('--compare', action='store_true', default=True)
    args = parser.parse_args()
    main(args)
