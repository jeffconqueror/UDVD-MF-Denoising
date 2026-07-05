"""
三种纳米颗粒结构 x/y 倾转系列 HRTEM 模拟 (用于结构分类训练集)
================================================================
结构 (原子数接近):
    Icosahedron      Au923   (noshells=7)              -- 沿 5 次轴基准
    Marks Decahedron Au906   (p=5,q=3,r=2)             -- 沿 5 次轴基准
    FCC Trunc.Octa.  Au976   (length=12,cutoff=4)      -- 沿 [110] 基准

每种结构: 绕 x 0-30度 / 绕 y 0-30度, 每 5 度一张 -> 7x7 = 49 张带噪声照片
共 3 x 49 = 147 张, 文件名 {struct}_x{xx}_y{yy}.png / .npy,
每种结构另存一张 7x7 总览图 montage。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from ase.cluster import Icosahedron, Decahedron, Octahedron
import abtem

abtem.config.set({"local_diagnostics.progress_bar": False})

# ============================ 参数 ============================
C = dict(
    energy_keV=300, sampling=0.10, slice_thickness=2.0,
    semiangle_cutoff=30, defocus=70, Cs=-13e-6 * 1e10, focal_spread=40,
    dose=2000, mtf_sigma=0.6, readout=0.02,
    vacuum=5.0, seed=0,
    x_angles=range(0, 31, 5),   # 0,5,...,30
    y_angles=range(0, 31, 5),
    z_angle=0,
)


# ---------- 基准取向 (x=y=z=0) ----------
def _align(atoms, axis_vec, to=(0, 0, 1)):
    a = atoms.copy()
    a.rotate(axis_vec, to, center="COU")
    return a

def base_icosahedron():
    a = Icosahedron("Au", noshells=7)                 # 923
    p = a.get_positions() - a.get_center_of_mass()
    five = p[np.argmax(np.linalg.norm(p, axis=1))]    # 5-fold thru vertex
    return _align(a, five)

def base_decahedron():
    a = Decahedron("Au", 5, 3, 2)                     # 906, Marks
    p = a.get_positions() - a.get_center_of_mass()
    w, vec = np.linalg.eigh(np.cov(p.T))
    five = vec[:, np.argmin(w)]                       # unique (5-fold) axis
    return _align(a, five)

def base_fcc():
    a = Octahedron("Au", length=12, cutoff=4)         # 976
    a = a.copy(); a.rotate(45, "x", center="COU")     # [100] -> [110] zone
    return a

STRUCTURES = {
    "ico":  ("Icosahedron Au923",        base_icosahedron),
    "deca": ("Marks Decahedron Au906",   base_decahedron),
    "fcc":  ("FCC TruncOcta Au976",      base_fcc),
}


# ---------- 倾转 + 成像 + 噪声 ----------
def tilt(base, ax, ay, az):
    a = base.copy()
    if ax: a.rotate(ax, "x", center="COU")
    if ay: a.rotate(ay, "y", center="COU")
    if az: a.rotate(az, "z", center="COU")
    return a

def simulate(atoms, c, rng):
    a = atoms.copy(); a.center(vacuum=c["vacuum"])
    pot = abtem.Potential(a, sampling=c["sampling"], slice_thickness=c["slice_thickness"],
                          projection="infinite", parametrization="kirkland")
    ctf = abtem.CTF(energy=c["energy_keV"] * 1e3, semiangle_cutoff=c["semiangle_cutoff"],
                    defocus=c["defocus"], Cs=c["Cs"], focal_spread=c["focal_spread"])
    clean = np.asarray(abtem.PlaneWave(energy=c["energy_keV"] * 1e3).multislice(pot)
                       .apply_ctf(ctf).intensity().compute().array, float)
    img = gaussian_filter(clean, c["mtf_sigma"]); px2 = c["sampling"] ** 2
    counts = np.clip(img * c["dose"] * px2, 0, None)
    noisy = rng.poisson(counts).astype(float) / (c["dose"] * px2)
    noisy += rng.normal(0, c["readout"], noisy.shape)
    return noisy


# ---------- 单结构 7x7 ----------
def run_structure(key, c=C):
    title, base_fn = STRUCTURES[key]
    rng = np.random.default_rng(c["seed"])
    base = base_fn()
    xs, ys, z = list(c["x_angles"]), list(c["y_angles"]), c["z_angle"]
    print(f"[{key}] {title}: {len(xs)}x{len(ys)} = {len(xs)*len(ys)} 张")

    fig, axes = plt.subplots(len(ys), len(xs), figsize=(2.1 * len(xs), 2.1 * len(ys)))
    for iy, ay in enumerate(ys):
        for ix, ax_ in enumerate(xs):
            noisy = simulate(tilt(base, ax_, ay, z), c, rng)
            tag = f"{key}_x{ax_:02d}_y{ay:02d}_z{z:02d}"
            np.save(f"{tag}.npy", noisy)
            plt.imsave(f"{tag}.png", noisy, cmap="gray")
            ax = axes[iy, ix]
            ax.imshow(noisy, cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.04, 0.96, f"x={ax_} y={ay} z={z}", transform=ax.transAxes,
                    fontsize=8, color="yellow", va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55, ec="none"))
        print(f"   y={ay:2d} row done")
    fig.suptitle(f"{title} — TEM tilt series (x/y deg)", y=1.004, fontsize=14)
    plt.tight_layout()
    fig.savefig(f"{key}_montage.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"   -> {key}_montage.png")


def main(c=C):
    for key in STRUCTURES:
        run_structure(key, c)
    print("ALL DONE: 3 structures x 49 = 147 images")


if __name__ == "__main__":
    main()
