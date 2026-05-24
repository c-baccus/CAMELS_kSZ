"""
03_test_one_galaxy.py

Test: run the kSZ pipeline on a single LRG.
Loads all gas, builds the KD-tree, picks galaxy 0 from the LRG catalog,
makes its kSZ + tau maps, and saves a figure.

Usage:
    python 03_test_one_galaxy.py
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from io_camels import load_header, load_all_gas
from kdtree_maps import (
    compute_per_cell_quantities,
    build_periodic_kdtree,
    make_galaxy_maps,
)


# ---- Configuration ----
SNAP_DIR = "/scratch/cjb9346/camels/CV_0/snapdir_074"
CATALOG_PATH = "outputs/lrg_catalog.npz"
OUTPUT_DIR = "outputs"
SNAPSHOT_NUM = 74
GALAXY_INDEX = 0           # which LRG to test (0 = highest stellar mass)
RADIUS_CKPC_H = 5000.0     # 5 Mpc/h query radius
N_PIX = 250                # 250x250 pixel maps
BOX_SIZE = 50000.0         # ckpc/h


def main():
    # 1. Read header to get hubble_h and scale_factor
    print("=== Reading header ===")
    sample_chunk = os.path.join(SNAP_DIR, f"snap_{SNAPSHOT_NUM:03d}.0.hdf5")
    header = load_header(sample_chunk)
    hubble_h = float(header['HubbleParam'])
    scale_factor = float(header['Time'])
    redshift = float(header['Redshift'])
    print(f"  z = {redshift:.4f}, a = {scale_factor:.4f}, h = {hubble_h:.4f}")

    # 2. Load all gas particles
    print("\n=== Loading all gas particles ===")
    t0 = time.time()
    gas = load_all_gas(SNAP_DIR, snapshot_num=SNAPSHOT_NUM)
    print(f"  ({time.time() - t0:.1f} sec)")

    # 3. Compute per-cell physical quantities
    print("\n=== Computing per-cell quantities ===")
    cell_data = compute_per_cell_quantities(gas, hubble_h, scale_factor)
    # Free raw arrays we no longer need
    del gas

    # 4. Build periodic KD-tree
    print("\n=== Building periodic KD-tree ===")
    t0 = time.time()
    tree = build_periodic_kdtree(cell_data['pos_ckpc_h'], BOX_SIZE)
    print(f"  ({time.time() - t0:.1f} sec)")

    # 5. Load LRG catalog, pick one galaxy
    print(f"\n=== Loading LRG catalog and picking galaxy {GALAXY_INDEX} ===")
    cat = np.load(CATALOG_PATH)
    galaxy_pos = cat['pos'][GALAXY_INDEX]
    m_halo_msun = cat['m_halo_msun'][GALAXY_INDEX]
    m_star_msun = cat['m_star_msun'][GALAXY_INDEX]
    print(f"  Galaxy {GALAXY_INDEX}: pos = {galaxy_pos}")
    print(f"    log10(M_halo) = {np.log10(m_halo_msun):.2f}, log10(M_star) = {np.log10(m_star_msun):.2f}")

    # 6. Run the map-making
    print(f"\n=== Making maps (radius={RADIUS_CKPC_H} ckpc/h, n_pix={N_PIX}) ===")
    t0 = time.time()
    maps = make_galaxy_maps(
        tree=tree,
        cell_data=cell_data,
        galaxy_pos=galaxy_pos,
        radius_ckpc_h=RADIUS_CKPC_H,
        hubble_h=hubble_h,
        scale_factor=scale_factor,
        n_pix=N_PIX,
        box_size=BOX_SIZE,
    )
    print(f"  ({time.time() - t0:.1f} sec)")
    print(f"  n_cells in query: {maps['n_cells']}")

    # 7. Print map stats
    print("\n=== Map stats ===")
    for key in ['tau_xy', 'tau_yz', 'tau_zx', 'b_xy', 'b_yz', 'b_zx']:
        m = maps[key]
        print(f"  {key}: min={m.min():.2e}, max={m.max():.2e}, "
              f"sum={m.sum():.2e}, |.|>0 in {(m != 0).sum()} pixels")

    # 8. Make a 2x3 plot: rows = tau/b, cols = xy/yz/zx
    print("\n=== Plotting ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    extent_mpc_h = RADIUS_CKPC_H / 1000.0    # convert to Mpc/h for axis labels
    extent = [-extent_mpc_h, extent_mpc_h, -extent_mpc_h, extent_mpc_h]

    for j, proj in enumerate(['xy', 'yz', 'zx']):
        # tau (log scale because it spans many decades)
        tau = maps[f'tau_{proj}']
        tau_safe = np.where(tau > 0, tau, np.nan)
        im = axes[0, j].imshow(np.log10(tau_safe), extent=extent, origin='lower', cmap='magma')
        axes[0, j].set_title(f'log10(tau), {proj}')
        axes[0, j].set_xlabel('Mpc/h')
        axes[0, j].set_ylabel('Mpc/h')
        plt.colorbar(im, ax=axes[0, j])

        # b (linear, signed - kSZ can be positive or negative)
        b = maps[f'b_{proj}']
        vmax = max(abs(b.min()), abs(b.max()))
        im = axes[1, j].imshow(b, extent=extent, origin='lower', cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax)
        axes[1, j].set_title(f'b (kSZ), {proj}')
        axes[1, j].set_xlabel('Mpc/h')
        axes[1, j].set_ylabel('Mpc/h')
        plt.colorbar(im, ax=axes[1, j])

    plt.suptitle(f'Galaxy {GALAXY_INDEX}: log10(M_halo)={np.log10(m_halo_msun):.2f}, '
                 f'log10(M_star)={np.log10(m_star_msun):.2f}, z=0.47')
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f'galaxy_{GALAXY_INDEX}_maps.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")

    print("\n=== Yay It's Done!!!!!! ===")


if __name__ == "__main__":
    main()