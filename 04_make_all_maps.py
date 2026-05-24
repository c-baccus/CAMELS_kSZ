"""
04_make_all_maps.py

Run the per-galaxy kSZ map pipeline on all LRGs in the catalog.
Saves 100 sets of 6 maps (tau xyz × 3 projections) to outputs/all_maps.npz.

Usage:
    python 04_make_all_maps.py
"""

import os
import time
import numpy as np
from tqdm import tqdm

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
OUTPUT_FILE = "outputs/all_maps.npz"
SNAPSHOT_NUM = 74
RADIUS_CKPC_H = 5000.0    # 5 Mpc/h query radius
N_PIX = 250
BOX_SIZE = 50000.0


def main():
    # 1. Header
    print("=== Reading header ===")
    sample_chunk = os.path.join(SNAP_DIR, f"snap_{SNAPSHOT_NUM:03d}.0.hdf5")
    header = load_header(sample_chunk)
    hubble_h = float(header['HubbleParam'])
    scale_factor = float(header['Time'])
    redshift = float(header['Redshift'])
    print(f"  z = {redshift:.4f}, a = {scale_factor:.4f}, h = {hubble_h:.4f}")

    # 2. Load gas + compute physics + build tree (one-time, expensive)
    print("\n=== Loading all gas particles ===")
    t0 = time.time()
    gas = load_all_gas(SNAP_DIR, snapshot_num=SNAPSHOT_NUM)
    print(f"  ({time.time() - t0:.1f} sec)")

    print("\n=== Computing per-cell quantities ===")
    cell_data = compute_per_cell_quantities(gas, hubble_h, scale_factor)
    del gas

    print("\n=== Building periodic KD-tree ===")
    t0 = time.time()
    tree = build_periodic_kdtree(cell_data['pos_ckpc_h'], BOX_SIZE)
    print(f"  ({time.time() - t0:.1f} sec)")

    # 3. Load galaxy catalog
    print(f"\n=== Loading LRG catalog ===")
    cat = np.load(CATALOG_PATH)
    n_gal = len(cat['m_halo_msun'])
    print(f"  {n_gal} galaxies to process")

    # 4. Allocate output arrays
    map_shape = (n_gal, N_PIX, N_PIX)
    tau_xy = np.zeros(map_shape, dtype=np.float64)
    tau_yz = np.zeros(map_shape, dtype=np.float64)
    tau_zx = np.zeros(map_shape, dtype=np.float64)
    b_xy = np.zeros(map_shape, dtype=np.float64)
    b_yz = np.zeros(map_shape, dtype=np.float64)
    b_zx = np.zeros(map_shape, dtype=np.float64)
    n_cells_per_gal = np.zeros(n_gal, dtype=np.int64)

    # 5. Loop over galaxies
    print(f"\n=== Making maps for all {n_gal} galaxies ===")
    t0 = time.time()
    for i in tqdm(range(n_gal), desc="Galaxies"):
        galaxy_pos = cat['pos'][i]
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
        tau_xy[i] = maps['tau_xy']
        tau_yz[i] = maps['tau_yz']
        tau_zx[i] = maps['tau_zx']
        b_xy[i] = maps['b_xy']
        b_yz[i] = maps['b_yz']
        b_zx[i] = maps['b_zx']
        n_cells_per_gal[i] = maps['n_cells']

    print(f"  All galaxies done in {time.time() - t0:.1f} sec")
    print(f"  mean n_cells per galaxy: {n_cells_per_gal.mean():.0f}")

    # 6. Save everything
    print(f"\n=== Saving to {OUTPUT_FILE} ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(
        OUTPUT_FILE,
        # maps
        tau_xy=tau_xy, tau_yz=tau_yz, tau_zx=tau_zx,
        b_xy=b_xy, b_yz=b_yz, b_zx=b_zx,
        # per-galaxy metadata (copied from catalog for convenience)
        m_halo_msun=cat['m_halo_msun'],
        m_star_msun=cat['m_star_msun'],
        pos=cat['pos'],
        vel=cat['vel'],
        subhalo_idx=cat['subhalo_idx'],
        n_cells_per_gal=n_cells_per_gal,
        # config
        radius_ckpc_h=np.array([RADIUS_CKPC_H]),
        n_pix=np.array([N_PIX]),
        redshift=np.array([redshift]),
        hubble_h=np.array([hubble_h]),
        scale_factor=np.array([scale_factor]),
    )
    print(f"  Saved {OUTPUT_FILE}")
    print(f"  File size: {os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")

    print("\n=== ࣪˖⋆˚★₊⊹★ ࣪˖ ࣪₊ ࣪˖ Yay it's Done!!♥︎♥︎♥︎ ࣪˖⋆˚★₊⊹★ ࣪˖ ࣪₊ ࣪˖ ===")


if __name__ == "__main__":
    main()