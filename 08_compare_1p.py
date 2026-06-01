"""
08_compare_1p.py

Produce comparison plots for 1P parameter sweep.
For each parameter, plot CAP profile of:
  - Fiducial (averaged from 27 CV sims)
  - 1P_pX_n2 (low extreme)
  - 1P_pX_2 (high extreme)

One figure per parameter. Velocity-weighted stacking applied.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.cosmology import FlatLambdaCDM

CV_OUTPUT_BASE = "outputs"
ONEP_OUTPUT_BASE = "outputs_1p"
COMBINED_OUTPUT_BASE = "outputs_1p_plots"

THETA_D_ARCMIN = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])
BEAM_FWHM_ARCMIN = 1.6
T_CMB_UK = 2.725e6

MAX_VARS = [3, 12, 26]
HIGH_VARS = [4, 5, 6, 13, 15, 16, 18, 25, 27]
ALL_PARAMS = MAX_VARS + HIGH_VARS

MASS_BIN_HIGH = (13.5, 14.5)


def make_cap_filter(n_pix, pixel_arcmin, theta_d_arcmin):
    cx, cy = n_pix / 2.0, n_pix / 2.0
    y, x = np.indices((n_pix, n_pix), dtype=float)
    r_arcmin = np.sqrt((x - cx)**2 + (y - cy)**2) * pixel_arcmin
    W = np.zeros((n_pix, n_pix))
    W[r_arcmin <= theta_d_arcmin] = 1.0
    W[(r_arcmin > theta_d_arcmin) & (r_arcmin <= np.sqrt(2) * theta_d_arcmin)] = -1.0
    return W


def cap_scalar(map_2d, pixel_arcmin, theta_d):
    W = make_cap_filter(map_2d.shape[0], pixel_arcmin, theta_d)
    return np.sum(map_2d * W)


def load_and_velocity_weight(path):
    if not os.path.exists(path):
        return None
    d = np.load(path)
    vel = d['vel']
    sign_xy = +np.sign(vel[:, 2])
    sign_yz = +np.sign(vel[:, 0])
    sign_zx = +np.sign(vel[:, 1])
    b_avg = (sign_xy[:, None, None] * d['b_xy'] +
             sign_yz[:, None, None] * d['b_yz'] +
             sign_zx[:, None, None] * d['b_zx']) / 3.0
    return {
        'b_avg': b_avg,
        'm_halo': d['m_halo_msun'],
        'n_pix': int(d['n_pix'][0]),
        'radius_ckpc_h': float(d['radius_ckpc_h'][0]),
        'redshift': float(d['redshift'][0]),
        'hubble_h': float(d['hubble_h'][0]),
        'scale_factor': float(d['scale_factor'][0]),
    }


def compute_profile(data, beam_sigma_pix, pixel_arcmin, mass_lo, mass_hi):
    log_m = np.log10(data['m_halo'])
    mask = (log_m >= mass_lo) & (log_m < mass_hi)
    idx = np.where(mask)[0]
    n_in_bin = len(idx)
    if n_in_bin == 0:
        return None, None, 0
    cap_per_galaxy = np.zeros((n_in_bin, len(THETA_D_ARCMIN)))
    for j, gi in enumerate(idx):
        b_smooth = gaussian_filter(data['b_avg'][gi], beam_sigma_pix)
        b_uk = b_smooth * T_CMB_UK
        for a, theta_d in enumerate(THETA_D_ARCMIN):
            cap_per_galaxy[j, a] = cap_scalar(b_uk, pixel_arcmin, theta_d)
    mean_prof = cap_per_galaxy.mean(axis=0)
    se_prof = cap_per_galaxy.std(axis=0) / np.sqrt(n_in_bin)
    return mean_prof, se_prof, n_in_bin


def main():
    os.makedirs(COMBINED_OUTPUT_BASE, exist_ok=True)

    print("Loading fiducial (CV_0 to CV_26)...")
    fiducial_data = None
    for i in range(27):
        path = os.path.join(CV_OUTPUT_BASE, f"CV_{i}", "all_maps.npz")
        sim_data = load_and_velocity_weight(path)
        if sim_data is None:
            continue
        if fiducial_data is None:
            fiducial_data = sim_data
        else:
            fiducial_data['b_avg'] = np.concatenate([fiducial_data['b_avg'], sim_data['b_avg']])
            fiducial_data['m_halo'] = np.concatenate([fiducial_data['m_halo'], sim_data['m_halo']])

    print(f"  Fiducial: {len(fiducial_data['m_halo'])} galaxies")

    cosmo = FlatLambdaCDM(H0=fiducial_data['hubble_h'] * 100, Om0=0.3)
    d_A = cosmo.angular_diameter_distance(fiducial_data['redshift']).value
    pixel_size_ckpc_h = (2 * fiducial_data['radius_ckpc_h']) / fiducial_data['n_pix']
    pixel_size_mpc = pixel_size_ckpc_h * fiducial_data['scale_factor'] / fiducial_data['hubble_h'] / 1000
    pixel_arcmin = (pixel_size_mpc / d_A) * (180 / np.pi) * 60
    beam_sigma_pix = (BEAM_FWHM_ARCMIN / 2.355) / pixel_arcmin
    print(f"  Pixel scale: {pixel_arcmin:.4f} arcmin/pix, beam_sigma: {beam_sigma_pix:.2f} pix")

    fid_mean, fid_se, fid_n = compute_profile(fiducial_data, beam_sigma_pix, pixel_arcmin,
                                              MASS_BIN_HIGH[0], MASS_BIN_HIGH[1])
    print(f"  Fiducial high-mass profile: N={fid_n}")

    for param in ALL_PARAMS:
        print(f"\n=== Parameter {param} ===")

        low_path = os.path.join(ONEP_OUTPUT_BASE, f"1P_p{param}_n2", "all_maps.npz")
        high_path = os.path.join(ONEP_OUTPUT_BASE, f"1P_p{param}_2", "all_maps.npz")

        if param == 15:
            low_path = None
            high_path = os.path.join(ONEP_OUTPUT_BASE, f"1P_p{param}_3", "all_maps.npz")

        low_data = load_and_velocity_weight(low_path) if low_path else None
        high_data = load_and_velocity_weight(high_path)

        low_mean, low_se, low_n = (None, None, 0)
        high_mean, high_se, high_n = (None, None, 0)

        if low_data is not None:
            low_mean, low_se, low_n = compute_profile(low_data, beam_sigma_pix, pixel_arcmin,
                                                     MASS_BIN_HIGH[0], MASS_BIN_HIGH[1])
            print(f"  Low extreme: N={low_n}")

        if high_data is not None:
            high_mean, high_se, high_n = compute_profile(high_data, beam_sigma_pix, pixel_arcmin,
                                                        MASS_BIN_HIGH[0], MASS_BIN_HIGH[1])
            print(f"  High extreme: N={high_n}")

        if high_data is None and low_data is None:
            print(f"  No data for param {param}, skipping")
            continue

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.errorbar(THETA_D_ARCMIN, fid_mean, yerr=fid_se,
                    marker='o', label=f"Fiducial CV (N={fid_n})", color='black', lw=2, capsize=3)

        if low_data is not None:
            ax.errorbar(THETA_D_ARCMIN, low_mean, yerr=low_se,
                        marker='s', label=f"1P_p{param}_n2 low (N={low_n})", color='blue', capsize=3)

        if high_data is not None:
            label = f"1P_p{param}_3 high (N={high_n})" if param == 15 else f"1P_p{param}_2 high (N={high_n})"
            ax.errorbar(THETA_D_ARCMIN, high_mean, yerr=high_se,
                        marker='^', label=label, color='red', capsize=3)

        ax.axhline(0, color='gray', lw=0.5)
        ax.set_xlabel(r"Aperture radius $\theta_d$ [arcmin]")
        ax.set_ylabel(r"$T_{\rm kSZ}(\theta_d)$ [$\mu$K $\cdot$ pixel$^2$]")
        ax.set_title(f"Parameter {param}: high-mass bin (log10 M_h >= {MASS_BIN_HIGH[0]:.1f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(COMBINED_OUTPUT_BASE, f"param_{param}_comparison.png")
        plt.savefig(out_path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f"  Saved {out_path}")

    print(f"\nAll done. {len(ALL_PARAMS)} comparison plots in {COMBINED_OUTPUT_BASE}/")


if __name__ == "__main__":
    main()
