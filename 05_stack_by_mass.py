"""
05_stack_by_mass.py

Stack per-galaxy kSZ maps in halo-mass bins and produce CAP profiles.

Pipeline:
  1. Load all_maps.npz (100 galaxies x 6 maps each)
  2. Average over the 3 projections per galaxy (factor 3 noise reduction)
  3. Bin galaxies into 3 halo-mass bins
  4. Stack (mean) within each bin
  5. Convolve stacked maps with 1.6 arcmin Gaussian beam
  6. Compute CAP profiles at multiple aperture sizes
  7. Plot kSZ profile vs aperture, one curve per mass bin

Final deliverable: outputs/ksz_profile_vs_mass.png

Usage:
    python 05_stack_by_mass.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u


# ---- Configuration ----
INPUT_FILE = "outputs/all_maps.npz"
OUTPUT_DIR = "outputs"

# Mass bins in log10(M_halo / M_sun)
MASS_BIN_EDGES = [12.5, 13.0, 13.5, 14.5]
MASS_BIN_LABELS = [
    r"$12.5 \leq \log_{10} M_h < 13.0$",
    r"$13.0 \leq \log_{10} M_h < 13.5$",
    r"$13.5 \leq \log_{10} M_h < 14.5$",
]
MASS_BIN_COLORS = ['C0', 'C1', 'C2']

# Beam parameters (ACT-like)
BEAM_FWHM_ARCMIN = 1.6

# CAP aperture radii in arcmin (paper uses ~0.5 to ~6 arcmin)
THETA_D_ARCMIN = np.linspace(0.5, 6.0, 12)

# Cosmology (CAMELS fiducial)
OMEGA_M = 0.3
T_CMB_MUK = 2.725e6  # mu K


def average_projections(maps):
    """Average xy, yz, zx projections per galaxy. Returns array of shape (N_gal, n_pix, n_pix)."""
    return (maps['xy'] + maps['yz'] + maps['zx']) / 3.0


def bin_galaxies_by_mass(log_m_halo, bin_edges):
    """Return list of index arrays, one per bin."""
    bin_indices = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (log_m_halo >= lo) & (log_m_halo < hi)
        bin_indices.append(np.where(mask)[0])
    return bin_indices


def pixel_to_arcmin(pixel_size_ckpc_h, hubble_h, scale_factor, redshift, omega_m):
    """Compute pixel angular size in arcmin at given redshift."""
    cosmo = FlatLambdaCDM(H0=hubble_h * 100, Om0=omega_m)
    d_A_mpc = cosmo.angular_diameter_distance(redshift).to(u.Mpc).value
    pixel_proper_mpc = pixel_size_ckpc_h * scale_factor / hubble_h / 1000.0
    pixel_rad = pixel_proper_mpc / d_A_mpc
    return pixel_rad * (180.0 / np.pi) * 60.0  # arcmin


def make_cap_filter(n_pix, pixel_arcmin, theta_d_arcmin):
    """
    Build a CAP (Compensated Aperture Photometry) filter centered on map center.
    Returns 2D filter: +1 inside disk of radius theta_d, -1 in annulus to sqrt(2)*theta_d.
    """
    center = n_pix / 2.0
    y, x = np.indices((n_pix, n_pix))
    r_arcmin = np.sqrt((x - center) ** 2 + (y - center) ** 2) * pixel_arcmin

    W = np.zeros((n_pix, n_pix))
    W[r_arcmin < theta_d_arcmin] = 1.0
    W[(r_arcmin >= theta_d_arcmin) & (r_arcmin < np.sqrt(2) * theta_d_arcmin)] = -1.0
    return W


def cap_profile(stacked_map, pixel_arcmin, theta_d_arcmin_array):
    """Compute CAP value at each aperture radius. Returns array of CAP values."""
    n_pix = stacked_map.shape[0]
    profile = np.zeros(len(theta_d_arcmin_array))
    for i, theta_d in enumerate(theta_d_arcmin_array):
        W = make_cap_filter(n_pix, pixel_arcmin, theta_d)
        profile[i] = np.sum(stacked_map * W)
    return profile


def main():
    # 1. Load the maps
    print(f"=== Loading {INPUT_FILE} ===")
    data = np.load(INPUT_FILE)
    n_gal = len(data['m_halo_msun'])
    n_pix = int(data['n_pix'][0])
    radius_ckpc_h = float(data['radius_ckpc_h'][0])
    redshift = float(data['redshift'][0])
    hubble_h = float(data['hubble_h'][0])
    scale_factor = float(data['scale_factor'][0])
    print(f"  {n_gal} galaxies, {n_pix}x{n_pix} maps, z = {redshift:.3f}")

    # 2. Average over the 3 projections per galaxy
    print("\n=== Averaging over projections (xy, yz, zx) ===")
    tau_avg = average_projections({'xy': data['tau_xy'], 'yz': data['tau_yz'], 'zx': data['tau_zx']})
    b_avg = average_projections({'xy': data['b_xy'], 'yz': data['b_yz'], 'zx': data['b_zx']})
    print(f"  tau_avg shape: {tau_avg.shape}")
    print(f"  b_avg shape:   {b_avg.shape}")

    # 3. Bin galaxies by halo mass
    log_m_halo = np.log10(data['m_halo_msun'])
    bin_indices = bin_galaxies_by_mass(log_m_halo, MASS_BIN_EDGES)
    print("\n=== Mass bins ===")
    for i, idx in enumerate(bin_indices):
        if len(idx) > 0:
            print(f"  Bin {i} ({MASS_BIN_LABELS[i]}): N={len(idx)}, "
                  f"<log10 M_h> = {log_m_halo[idx].mean():.2f}")
        else:
            print(f"  Bin {i} ({MASS_BIN_LABELS[i]}): N=0 (empty!)")

    # 4. Compute pixel angular size
    pixel_size_ckpc_h = 2.0 * radius_ckpc_h / n_pix
    pixel_arcmin = pixel_to_arcmin(pixel_size_ckpc_h, hubble_h, scale_factor, redshift, OMEGA_M)
    print(f"\n=== Pixel angular size ===")
    print(f"  Pixel = {pixel_size_ckpc_h:.1f} ckpc/h = {pixel_arcmin:.3f} arcmin at z={redshift:.3f}")

    # 5. Beam smoothing sigma (in pixels)
    beam_sigma_arcmin = BEAM_FWHM_ARCMIN / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    beam_sigma_pix = beam_sigma_arcmin / pixel_arcmin
    print(f"  Beam FWHM = {BEAM_FWHM_ARCMIN} arcmin -> sigma = {beam_sigma_pix:.2f} pixels")

    # 6. Stack + smooth + CAP profile for each bin
    print("\n=== Stacking, smoothing, and CAP-filtering each bin ===")
    fig_stack, axes_stack = plt.subplots(2, 3, figsize=(15, 10))
    profiles = []

    for i, idx in enumerate(bin_indices):
        if len(idx) == 0:
            profiles.append(np.zeros(len(THETA_D_ARCMIN)))
            continue

        # Stack: mean over galaxies in this bin
        tau_stack = np.mean(tau_avg[idx], axis=0)
        b_stack = np.mean(b_avg[idx], axis=0)

        # Beam smooth
        tau_smooth = gaussian_filter(tau_stack, beam_sigma_pix)
        b_smooth = gaussian_filter(b_stack, beam_sigma_pix)

        # CAP profile (on the kSZ map)
        prof = cap_profile(b_smooth, pixel_arcmin, THETA_D_ARCMIN)
        profiles.append(prof)

        # Plot stacked maps for visual inspection
        extent_mpc_h = radius_ckpc_h / 1000.0
        extent = [-extent_mpc_h, extent_mpc_h, -extent_mpc_h, extent_mpc_h]

        im0 = axes_stack[0, i].imshow(np.log10(np.maximum(tau_smooth, 1e-12)),
                                       extent=extent, origin='lower', cmap='magma')
        axes_stack[0, i].set_title(f'log10(tau), {MASS_BIN_LABELS[i]} (N={len(idx)})')
        axes_stack[0, i].set_xlabel('Mpc/h')
        axes_stack[0, i].set_ylabel('Mpc/h')
        plt.colorbar(im0, ax=axes_stack[0, i])

        vmax = max(abs(b_smooth.min()), abs(b_smooth.max()))
        im1 = axes_stack[1, i].imshow(b_smooth, extent=extent, origin='lower',
                                       cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes_stack[1, i].set_title(f'b (kSZ), {MASS_BIN_LABELS[i]} (N={len(idx)})')
        axes_stack[1, i].set_xlabel('Mpc/h')
        axes_stack[1, i].set_ylabel('Mpc/h')
        plt.colorbar(im1, ax=axes_stack[1, i])

    plt.suptitle(f'Stacked maps by halo mass bin, CV_0, z={redshift:.2f}, beam smoothed')
    plt.tight_layout()
    stack_plot = os.path.join(OUTPUT_DIR, 'stacked_maps_by_mass.png')
    plt.savefig(stack_plot, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {stack_plot}")

    # 7. Plot CAP profile vs aperture
    print("\n=== Plotting CAP profile ===")
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (prof, label, color) in enumerate(zip(profiles, MASS_BIN_LABELS, MASS_BIN_COLORS)):
        if len(bin_indices[i]) == 0:
            continue
        # Convert dimensionless b CAP to muK by multiplying by T_CMB
        # (b is the kSZ momentum; T_kSZ = T_CMB * b; CAP integrates over the map)
        prof_muK = prof * T_CMB_MUK
        ax.plot(THETA_D_ARCMIN, prof_muK, 'o-', color=color, label=label)

    ax.axhline(0, color='gray', lw=0.5)
    ax.set_xlabel(r'Aperture radius $\theta_d$ [arcmin]')
    ax.set_ylabel(r'$T_\mathrm{kSZ}(\theta_d)$ [$\mu$K $\cdot$ pixel$^2$]')
    ax.set_title(f'kSZ CAP profile vs aperture, CV_0 z={redshift:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    profile_plot = os.path.join(OUTPUT_DIR, 'ksz_profile_vs_mass.png')
    plt.savefig(profile_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {profile_plot}")

    # 8. Save numerical profiles
    profile_data = os.path.join(OUTPUT_DIR, 'profiles.npz')
    np.savez(
        profile_data,
        theta_d_arcmin=THETA_D_ARCMIN,
        profile_bin0=profiles[0],
        profile_bin1=profiles[1],
        profile_bin2=profiles[2],
        mass_bin_edges=np.array(MASS_BIN_EDGES),
        n_per_bin=np.array([len(b) for b in bin_indices]),
        pixel_arcmin=np.array([pixel_arcmin]),
        beam_sigma_pix=np.array([beam_sigma_pix]),
    )
    print(f"  Saved {profile_data}")

    print("\n=== Done ♥︎♥︎♥︎ ===")


if __name__ == "__main__":
    main()