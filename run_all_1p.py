"""
run_all_cv.py

Run the kSZ pipeline (LRG selection + per-galaxy maps) on all 27 CV simulations.
For each CV sim, saves:
  outputs/CV_X/lrg_catalog.npz
  outputs/CV_X/all_maps.npz

This is the multi-sim version of (02 + 04). Designed to be re-runnable —
skips sims that already have outputs unless --force is passed.

Usage:
    python run_all_cv.py           # process any sims not yet done
    python run_all_cv.py --force   # redo all sims even if outputs exist
    python run_all_cv.py --sims 1,2,5,10   # only specific sims
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from io_camels import load_header, load_all_gas, load_subhalo_catalog, load_group_catalog
from kdtree_maps import (
    compute_per_cell_quantities,
    build_periodic_kdtree,
    make_galaxy_maps,
)


# ---- Configuration ----
CAMELS_BASE = "/scratch/cjb9346/camels_1p"
OUTPUT_BASE = "outputs_1p"
SNAPSHOT_NUM = 74
N_GAL = 100
HUBBLE_H = 0.6711
RADIUS_CKPC_H = 5000.0
N_PIX = 250
BOX_SIZE = 50000.0


def select_lrgs(group_dir):
    """
    LRG selection logic from 02_select_lrgs.py, returns a dict ready to save.
    """
    subhalos = load_subhalo_catalog(group_dir, snapshot_num=SNAPSHOT_NUM)
    groups = load_group_catalog(group_dir, snapshot_num=SNAPSHOT_NUM)

    n_sub = len(subhalos['SubhaloMass'])

    # Identify centrals
    is_central = np.zeros(n_sub, dtype=bool)
    central_indices = groups['GroupFirstSub']
    valid_mask = central_indices >= 0
    is_central[central_indices[valid_mask]] = True

    # Sort centrals by stellar mass
    m_star_sim = subhalos['SubhaloMassType'][:, 4]
    m_star_centrals = np.where(is_central, m_star_sim, -np.inf)
    top_idx = np.argsort(m_star_centrals)[::-1][:N_GAL]

    # Get halo properties
    grnr = subhalos['SubhaloGrNr'][top_idx]
    m_halo_sim = groups['Group_M_Crit200'][grnr]
    r_halo_sim = groups['Group_R_Crit200'][grnr]

    return {
        'subhalo_idx': top_idx,
        'pos': subhalos['SubhaloPos'][top_idx],
        'vel': subhalos['SubhaloVel'][top_idx],
        'm_star_msun': m_star_sim[top_idx] * 1e10 / HUBBLE_H,
        'm_halo_msun': m_halo_sim * 1e10 / HUBBLE_H,
        'r_halo_ckpc_h': r_halo_sim,
        'grnr': grnr,
        'n_subhalos': n_sub,
        'n_groups': len(groups['Group_M_Crit200']),
    }


def make_all_maps(snap_dir, lrg_catalog, hubble_h, scale_factor, redshift):
    """
    Build tree + make maps for all galaxies in lrg_catalog. Returns map dict.
    """
    # Load gas
    print(f"  Loading gas...")
    t0 = time.time()
    gas = load_all_gas(snap_dir, snapshot_num=SNAPSHOT_NUM)
    print(f"    ({time.time() - t0:.1f} sec)")

    # Compute per-cell quantities
    print(f"  Computing per-cell quantities...")
    cell_data = compute_per_cell_quantities(gas, hubble_h, scale_factor)
    del gas  # free memory

    # Build KD-tree
    print(f"  Building KD-tree...")
    t0 = time.time()
    tree = build_periodic_kdtree(cell_data['pos_ckpc_h'], BOX_SIZE)
    print(f"    ({time.time() - t0:.1f} sec)")

    # Loop over galaxies
    n_gal = len(lrg_catalog['m_halo_msun'])
    map_shape = (n_gal, N_PIX, N_PIX)
    tau_xy = np.zeros(map_shape)
    tau_yz = np.zeros(map_shape)
    tau_zx = np.zeros(map_shape)
    b_xy = np.zeros(map_shape)
    b_yz = np.zeros(map_shape)
    b_zx = np.zeros(map_shape)
    n_cells_per_gal = np.zeros(n_gal, dtype=np.int64)

    print(f"  Making maps for {n_gal} galaxies...")
    t0 = time.time()
    for i in tqdm(range(n_gal), desc="    Galaxies"):
        maps = make_galaxy_maps(
            tree=tree,
            cell_data=cell_data,
            galaxy_pos=lrg_catalog['pos'][i],
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
    print(f"    ({time.time() - t0:.1f} sec)")

    return {
        'tau_xy': tau_xy, 'tau_yz': tau_yz, 'tau_zx': tau_zx,
        'b_xy': b_xy, 'b_yz': b_yz, 'b_zx': b_zx,
        'n_cells_per_gal': n_cells_per_gal,
        'm_halo_msun': lrg_catalog['m_halo_msun'],
        'm_star_msun': lrg_catalog['m_star_msun'],
        'pos': lrg_catalog['pos'],
        'vel': lrg_catalog['vel'],
        'subhalo_idx': lrg_catalog['subhalo_idx'],
        'r_halo_ckpc_h': lrg_catalog['r_halo_ckpc_h'],
        'radius_ckpc_h': np.array([RADIUS_CKPC_H]),
        'n_pix': np.array([N_PIX]),
        'redshift': np.array([redshift]),
        'hubble_h': np.array([hubble_h]),
        'scale_factor': np.array([scale_factor]),
    }


def process_one_sim(sim_name):
    """Run the full pipeline (select + make maps) on one sim."""
    snap_dir = os.path.join(CAMELS_BASE, f"{sim_name}/snapdir_074")
    group_dir = os.path.join(CAMELS_BASE, f"{sim_name}/groups_074")
    output_dir = os.path.join(OUTPUT_BASE, f"{sim_name}")
    os.makedirs(output_dir, exist_ok=True)

    cat_path = os.path.join(output_dir, "lrg_catalog.npz")
    maps_path = os.path.join(output_dir, "all_maps.npz")

    print(f"\n========= {sim_name} =========")

    # 1. LRG selection
    print(f"  Selecting LRGs...")
    lrg_cat = select_lrgs(group_dir)
    np.savez(cat_path, **lrg_cat)
    print(f"    Saved {cat_path}")
    print(f"    log10(M_halo) median: {np.median(np.log10(lrg_cat['m_halo_msun'])):.2f}")

    # 2. Header
    sample_chunk = os.path.join(snap_dir, f"snap_{SNAPSHOT_NUM:03d}.0.hdf5")
    header = load_header(sample_chunk)
    hubble_h = float(header['HubbleParam'])
    scale_factor = float(header['Time'])
    redshift = float(header['Redshift'])

    # 3. Make all maps
    maps = make_all_maps(snap_dir, lrg_cat, hubble_h, scale_factor, redshift)
    np.savez(maps_path, **maps)
    print(f"    Saved {maps_path} ({os.path.getsize(maps_path) / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='Reprocess sims even if outputs exist')
    parser.add_argument('--sims', type=str, default=None,
                        help='Comma-separated list of sim names (e.g., "1P_p3_2,1P_p3_n2"). Default: all in CAMELS_BASE.')
    args = parser.parse_args()

    if args.sims:
        sim_names = args.sims.split(',')
    else:
        sim_names = sorted([d for d in os.listdir(CAMELS_BASE)
                           if os.path.isdir(os.path.join(CAMELS_BASE, d))])

    print(f"Will process sims: {sim_names}")
    overall_t0 = time.time()

    for sim_name in sim_names:
        # Skip if outputs already exist (unless --force)
        maps_path = os.path.join(OUTPUT_BASE, f"{sim_name}", "all_maps.npz")
        if os.path.exists(maps_path) and not args.force:
            print(f"\n{sim_name}: outputs exist, skipping (use --force to redo)")
            continue

        # Verify the input data exists
        snap_dir = os.path.join(CAMELS_BASE, f"{sim_name}/snapdir_074")
        group_dir = os.path.join(CAMELS_BASE, f"{sim_name}/groups_074")
        if not os.path.isdir(snap_dir) or not os.path.isdir(group_dir):
            print(f"\n{sim_name}: data not found, skipping")
            continue

        try:
            process_one_sim(sim_name)
        except Exception as e:
            print(f"\n!!! Error processing {sim_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total = time.time() - overall_t0
    print(f"\n========= ALL DONE in {total/60:.1f} min =========")


if __name__ == "__main__":
    main()
    