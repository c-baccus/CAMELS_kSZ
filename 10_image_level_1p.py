"""
10_image_level_1p.py

Per-parameter stacked kSZ images for the 1P sweep.

For each parameter, makes a 3-panel figure:
  - Fiducial (combined stack across all 27 CV sims)
  - 1P_pX_n2 (low extreme)
  - 1P_pX_2  (high extreme)

All maps are velocity-weighted, beam-smoothed, and stacked across the
high-mass bin (log10 M_h >= 13.5). Shared symmetric colorbar across the
three panels in each figure so the visual comparison is fair.

Outputs PNGs to ./outputs_1p_images/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.cosmology import FlatLambdaCDM

# ---- Paths ----
CV_OUTPUT_BASE = "outputs"
ONEP_OUTPUT_BASE = "outputs_1p"
OUT_DIR = "outputs_1p_images"

# ---- Constants (match 08_compare_1p.py) ----
BEAM_FWHM_ARCMIN = 1.6
T_CMB_UK = 2.725e6
MASS_BIN_HIGH = (13.5, 14.5)

# ---- Parameters to plot ----
MAX_VARS = [3, 12, 26]
HIGH_VARS = [4, 5, 6, 13, 15, 16, 18, 25, 27]
ALL_PARAMS = MAX_VARS + HIGH_VARS


def load_and_stack(npz_path, mass_lo, mass_hi, beam_sigma_pix):
    """
    Load one sim's all_maps.npz, apply velocity weighting, beam-smooth each
    galaxy, then stack (mean) across galaxies in the [mass_lo, mass_hi) bin.

    Returns (stacked_map [n_pix, n_pix] in uK, N_galaxies) or (None, 0).
    """
    if not os.path.exists(npz_path):
        return None, 0

    d = np.load(npz_path)
    vel = d['vel']
    m_halo = d['m_halo_msun']
    log_m = np.log10(m_halo)
    mask = (log_m >= mass_lo) & (log_m < mass_hi)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return None, 0

    # Velocity-weighted projection-averaged map per galaxy
    # (same sign convention as 08_compare_1p.py: +np.sign)
    sign_xy = +np.sign(vel[:, 2])
    sign_yz = +np.sign(vel[:, 0])
    sign_zx = +np.sign(vel[:, 1])
    b_avg = (sign_xy[:, None, None] * d['b_xy'] +
             sign_yz[:, None, None] * d['b_yz'] +
             sign_zx[:, None, None] * d['b_zx']) / 3.0

    n_pix = b_avg.shape[1]
    stack = np.zeros((n_pix, n_pix), dtype=np.float64)
    for gi in idx:
        smoothed = gaussian_filter(b_avg[gi], beam_sigma_pix)
        stack += smoothed
    stack /= len(idx)         # per-galaxy mean
    stack *= T_CMB_UK         # to uK
    return stack, len(idx)


def compute_pixel_scale(sample_npz):
    """Compute the arcmin-per-pixel and beam-sigma in pixels from a sample sim."""
    d = np.load(sample_npz)
    n_pix = int(d['n_pix'][0])
    radius_ckpc_h = float(d['radius_ckpc_h'][0])
    redshift = float(d['redshift'][0])
    hubble_h = float(d['hubble_h'][0])
    scale_factor = float(d['scale_factor'][0])

    cosmo = FlatLambdaCDM(H0=hubble_h * 100, Om0=0.3)
    d_A = cosmo.angular_diameter_distance(redshift).value   # Mpc
    pixel_size_ckpc_h = (2 * radius_ckpc_h) / n_pix
    pixel_size_mpc = pixel_size_ckpc_h * scale_factor / hubble_h / 1000.0
    pixel_arcmin = (pixel_size_mpc / d_A) * (180 / np.pi) * 60
    beam_sigma_pix = (BEAM_FWHM_ARCMIN / 2.355) / pixel_arcmin
    half_extent_mpch = radius_ckpc_h / 1000.0    # comoving Mpc/h
    return n_pix, beam_sigma_pix, pixel_arcmin, half_extent_mpch


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    sample = os.path.join(CV_OUTPUT_BASE, "CV_0", "all_maps.npz")
    n_pix, beam_sigma_pix, pixel_arcmin, half_extent_mpch = compute_pixel_scale(sample)
    extent = [-half_extent_mpch, half_extent_mpch, -half_extent_mpch, half_extent_mpch]
    print(f"n_pix={n_pix}, pixel={pixel_arcmin:.4f} arcmin, "
          f"beam_sigma={beam_sigma_pix:.2f} pix, extent=+/-{half_extent_mpch:.1f} cMpc/h")

    # ---- Build fiducial stack from all 27 CV sims ----
    print("\nBuilding fiducial stack from 27 CV sims...")
    fid_stack = np.zeros((n_pix, n_pix), dtype=np.float64)
    fid_count = 0
    for i in range(27):
        path = os.path.join(CV_OUTPUT_BASE, f"CV_{i}", "all_maps.npz")
        s, n = load_and_stack(path, *MASS_BIN_HIGH, beam_sigma_pix)
        if s is None:
            continue
        fid_stack += s * n     # un-normalize, then renormalize across all sims
        fid_count += n
    if fid_count == 0:
        print("No fiducial data found in 'outputs/'. Aborting.")
        return
    fid_stack /= fid_count
    print(f"  Fiducial: N={fid_count} high-mass galaxies")

    # ---- One figure per parameter ----
    for param in ALL_PARAMS:
        print(f"\n=== Parameter {param} ===")

        low_path = os.path.join(ONEP_OUTPUT_BASE, f"1P_p{param}_n2", "all_maps.npz")
        high_path = os.path.join(ONEP_OUTPUT_BASE, f"1P_p{param}_2", "all_maps.npz")
        # Special case: param 15 has fiducial=0, so no _n2; CAMELS provides _3 instead.
        if param == 15:
            low_path = None
            high_path = os.path.join(ONEP_OUTPUT_BASE, f"1P_p{param}_3", "all_maps.npz")

        low_stack, low_n = (None, 0)
        if low_path is not None:
            low_stack, low_n = load_and_stack(low_path, *MASS_BIN_HIGH, beam_sigma_pix)
        high_stack, high_n = load_and_stack(high_path, *MASS_BIN_HIGH, beam_sigma_pix)
        print(f"  Low: N={low_n}   High: N={high_n}")

        if high_stack is None and low_stack is None:
            print("  No data for this parameter; skipping.")
            continue

        # Symmetric shared color range across all panels for fair comparison
        all_stacks = [fid_stack]
        if low_stack is not None:
            all_stacks.append(low_stack)
        if high_stack is not None:
            all_stacks.append(high_stack)
        vmax = max(float(np.max(np.abs(s))) for s in all_stacks)
        vmin = -vmax

        # Build panels list
        panels = [(fid_stack, f"Fiducial CV  (N={fid_count})", "black")]
        if low_stack is not None:
            panels.append((low_stack, f"1P_p{param}_n2  (N={low_n})", "tab:blue"))
        if high_stack is not None:
            label = f"1P_p{param}_3" if param == 15 else f"1P_p{param}_2"
            panels.append((high_stack, f"{label}  (N={high_n})", "tab:red"))

        ncols = len(panels)
        fig, axes = plt.subplots(
            1, ncols,
            figsize=(5 * ncols, 5),
            gridspec_kw={'wspace': 0.18}
        )
        if ncols == 1:
            axes = [axes]

        im = None
        for ax, (stack, title, color) in zip(axes, panels):
            im = ax.imshow(stack, cmap="RdBu_r", origin="lower",
                           vmin=vmin, vmax=vmax, extent=extent,
                           interpolation="nearest")
            ax.set_title(title, fontsize=11, color=color)
            ax.set_xlabel("cMpc/h")
            ax.set_ylabel("cMpc/h")

        cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label(r"$T_{\rm kSZ}$  [$\mu$K]")

        fig.suptitle(
            f"Parameter {param}: stacked kSZ maps  "
            f"(velocity-weighted, beam-smoothed, log10 M_h $\\geq$ {MASS_BIN_HIGH[0]:.1f})",
            fontsize=12,
            y=1.02,
        )

        out_path = os.path.join(OUT_DIR, f"param_{param}_image_stack.png")
        plt.savefig(out_path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out_path}")

    print(f"\nAll done. Images in {OUT_DIR}/")


if __name__ == "__main__":
    main()