"""
Generate synthetic HRTEM training data for 3 nanoparticle structure classes:
  ih   — Icosahedron Au923
  deca — Marks Decahedron Au906
  fcc  — FCC truncated-octahedron Au976

For each structure: 5000 training + 1000 validation images, each is a random
HRTEM simulation with:
  - random tilt:  x, y, z ∈ [0, 90] degrees (uniform)
  - random defocus jitter (50-90 Å, centered at 70)
  - random dose    (1500-3500 e-/Å²)
  - sampling jitter ±5% (acts as a small physical-size variation)

Output: /shared/jingchl6/material/lc-research/generated/{class}/{train|val}/{idx:05d}.png
(uint8 256×256 grayscale, center-cropped from the native abtem output)

Uses multiprocessing — pass --workers to set parallelism.
"""
import argparse, os, time, traceback
import numpy as np
from multiprocessing import Pool, current_process
from scipy.ndimage import gaussian_filter

from ase.cluster import Icosahedron, Decahedron, Octahedron
import abtem
import cv2

abtem.config.set({"local_diagnostics.progress_bar": False})

OUT_ROOT = '/shared/jingchl6/material/lc-research/generated'
CLASSES = ['ih', 'deca', 'fcc']

# Image size we save (matches the segcrop input the classifier will see)
OUT_SIZE = 256


def _align(atoms, axis_vec, to=(0, 0, 1)):
    a = atoms.copy()
    a.rotate(axis_vec, to, center='COU')
    return a


# Use Pt to match the PtRu test particles more closely (Au→Pt: lattice 4.08→3.92 Å,
# both ~similar atomic number for scattering). PtRu lattice falls between Pt and Au.
ELEMENT = 'Pt'


def base_icosahedron():
    a = Icosahedron(ELEMENT, noshells=7)
    p = a.get_positions() - a.get_center_of_mass()
    five = p[np.argmax(np.linalg.norm(p, axis=1))]
    return _align(a, five)


def base_decahedron():
    a = Decahedron(ELEMENT, 5, 3, 2)
    p = a.get_positions() - a.get_center_of_mass()
    w, vec = np.linalg.eigh(np.cov(p.T))
    five = vec[:, np.argmin(w)]
    return _align(a, five)


def base_fcc():
    a = Octahedron(ELEMENT, length=12, cutoff=4)
    a = a.copy(); a.rotate(45, 'x', center='COU')  # → [110] zone
    return a


BASE_FNS = {'ih': base_icosahedron, 'deca': base_decahedron, 'fcc': base_fcc}


def simulate_one(structure, ax, ay, az, defocus, dose, sampling, vacuum, seed):
    base = BASE_FNS[structure]()
    a = base.copy()
    if ax: a.rotate(ax, 'x', center='COU')
    if ay: a.rotate(ay, 'y', center='COU')
    if az: a.rotate(az, 'z', center='COU')
    a.center(vacuum=vacuum)

    pot = abtem.Potential(a, sampling=sampling, slice_thickness=2.0,
                          projection='infinite', parametrization='kirkland')
    ctf = abtem.CTF(energy=300e3, semiangle_cutoff=30, defocus=defocus,
                    Cs=-13e-6 * 1e10, focal_spread=40)
    clean = np.asarray(abtem.PlaneWave(energy=300e3).multislice(pot).apply_ctf(ctf).intensity().compute().array, float)
    img = gaussian_filter(clean, 0.6)

    px2 = sampling ** 2
    rng = np.random.default_rng(seed)
    counts = np.clip(img * dose * px2, 0, None)
    noisy = rng.poisson(counts).astype(float) / (dose * px2)
    noisy += rng.normal(0, 0.02, noisy.shape)

    # Resize the FULL abtem output to OUT_SIZE x OUT_SIZE.
    # (Previously we center-cropped, which cut off the particle boundary —
    # the reference montage shows the whole particle including its silhouette.)
    resized = cv2.resize(noisy, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)

    # Normalize to uint8
    mn, mx = resized.min(), resized.max()
    u8 = np.clip((resized - mn) / max(mx - mn, 1e-6) * 255, 0, 255).astype(np.uint8)
    return u8


def worker_job(args):
    structure, split, idx, seed = args
    try:
        rng = np.random.default_rng(seed)
        ax = float(rng.uniform(0, 90))
        ay = float(rng.uniform(0, 90))
        az = float(rng.uniform(0, 90))
        defocus  = float(rng.uniform(50, 90))
        dose     = float(rng.uniform(1500, 3500))
        sampling = float(rng.uniform(0.095, 0.105))
        # Multi-scale: vacuum 0.5-25 Å → particle covers ~40% (small vacuum) to ~95% (large vacuum) of frame
        vacuum   = float(rng.uniform(0.5, 25.0))

        out_dir = os.path.join(OUT_ROOT, structure, split)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'{idx:05d}.png')
        if os.path.exists(out_path):
            return f'skip {structure}/{split}/{idx:05d}'

        u8 = simulate_one(structure, ax, ay, az, defocus, dose, sampling, vacuum, seed + 1)
        cv2.imwrite(out_path, u8)
        return f'ok {structure}/{split}/{idx:05d}  ax={ax:.1f} ay={ay:.1f} az={az:.1f} d={defocus:.0f} vac={vacuum:.1f}'
    except Exception as e:
        return f'ERR {structure}/{split}/{idx}: {e}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=5000)
    ap.add_argument('--n-val',   type=int, default=1000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed-base', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)

    # Build job list
    jobs = []
    for cls_idx, structure in enumerate(CLASSES):
        # Distinct seed ranges per structure / split for reproducibility
        for i in range(args.n_train):
            jobs.append((structure, 'train', i, args.seed_base + cls_idx * 1_000_000 + i))
        for i in range(args.n_val):
            jobs.append((structure, 'val', i, args.seed_base + cls_idx * 1_000_000 + 500_000 + i))
    print(f'Total jobs: {len(jobs)}  ({args.n_train} train + {args.n_val} val) x {len(CLASSES)} structures')
    print(f'Workers: {args.workers}')
    t0 = time.time()
    n_done = 0; n_err = 0
    with Pool(args.workers) as p:
        for res in p.imap_unordered(worker_job, jobs, chunksize=8):
            n_done += 1
            if res.startswith('ERR'):
                n_err += 1
                print(res)
            if n_done % 200 == 0 or n_done == len(jobs):
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1)
                eta = (len(jobs) - n_done) / max(rate, 0.001)
                print(f'  [{n_done:5d}/{len(jobs)}] '
                      f'elapsed={elapsed/60:.1f}m  rate={rate:.1f}/s  '
                      f'ETA={eta/60:.1f}m  errors={n_err}')
    print(f'\nDONE in {(time.time() - t0)/60:.1f} min, {n_err} errors')


if __name__ == '__main__':
    main()
