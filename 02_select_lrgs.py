"""
Select LRG-like galaxies from a CAMELS TNG50 group catalog:
- take only central subhlos (the most massive one per FoF group)
-  rank by stellar mass 
- take the top N_GAL galaxies
- plot the M_halo distribution
- Save the LRG catalog to a .npz file for downstream use


NOTE: WE SELECT CENTRALS (THE MOST MASSIVE SUBHALO PER FOF GROUP)
DESI LRGs include ~5-15% satellites; we ignore that mixing for now.

Usage:
    python 02_select_lrgs.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg") #for computer nodes
import matplotlib.pyplot as plt

from io_camels import load_subhalo_catalog, load_group_catalog

#Config
GROUP_DIR = "/scratch/cjb9346/camels/CV_0/groups_074"
OUTPUT_DIR = "outputs"
SNAPSHOT_NUM = 74
N_GAL = 100  # number of top-stellar-mass centrals to select
HUBBLE_H = 0.6711          # CAMELS fiducial H0 value

def main():
    #1 load catalogs
    print("Loading subhalo catalog...")
    subhalos = load_subhalo_catalog(GROUP_DIR, snapshot_num=SNAPSHOT_NUM)
    n_sub = len(subhalos['SubhaloMass'])
    print(f"  {n_sub} subhalos loaded")

    print("Loading group catalog...")
    groups = load_group_catalog(GROUP_DIR, snapshot_num=SNAPSHOT_NUM)
    n_groups = len(groups['Group_M_Crit200'])
    print(f"  {n_groups} groups loaded")


    #2 identify centrals
    # vectorized because its better ig?
    is_central = np.zeros(n_sub, dtype=bool)
    # GroupFirstSub gives the subhalo index for the central of each group, -1 for empty groups
    central_indices = groups['GroupFirstSub']
    valid_mask = central_indices >= 0
    is_central[central_indices[valid_mask]] = True
    num_centrals = int(valid_mask.sum())
    print(f"  Identified {num_centrals} centrals (groups with non-empty central subhalo)")
    #3 compute stellar mass for each subhalo (sim units: 10^10 M_sun/h)
    m_star_sim = subhalos['SubhaloMassType'][:, 4]  # Index 4: stellar mass

    #4 restrict to centrals: sort by stellar mass (descending), then take top N_GAL
    m_star_centrals = np.where(is_central, m_star_sim, -np.inf)
    top_idx = np.argsort(m_star_centrals)[::-1][:N_GAL]  # subhalo indices of top N_GAL centrals

    #5 for each selected galaxy, look up the host halo's M_200c
    grnr = subhalos['SubhaloGrNr'][top_idx]  # FoF group number for selected galaxies
    m_halo_sim = groups['Group_M_Crit200'][grnr]  # host halo mass
    r_halo_sim = groups['Group_R_Crit200'][grnr]  # host halo radius (ckpc/h)

    #6 convert masses to physical units (M_sun, no h factor)
    m_star_msun = m_star_sim[top_idx] * 1e10 / HUBBLE_H
    m_halo_msun = m_halo_sim * 1e10 / HUBBLE_H
    # r_halo_sim stays in ckpc/h for later spatial lookups

    #7 print summary statistics
    print(f"N selected: {N_GAL}")
    log_m_halo = np.log10(m_halo_msun)
    log_m_star = np.log10(m_star_msun)
    print(f"log10(M_halo/M_sun): mean={log_m_halo.mean():.3f}, median={np.median(log_m_halo):.3f}, std={log_m_halo.std():.3f}, min={log_m_halo.min():.3f}, max={log_m_halo.max():.3f}")
    print(f"log10(M_star/M_sun): mean={log_m_star.mean():.3f}, median={np.median(log_m_star):.3f}, min={log_m_star.min():.3f}, max={log_m_star.max():.3f}")
    print(f"r_halo_sim (ckpc/h): min={r_halo_sim.min():.2f}, max={r_halo_sim.max():.2f}")

    #8 plot hist of log10(M_halo)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure()
    plt.hist(log_m_halo, bins=20, color='C0', alpha=0.85, edgecolor='k')
    plt.xlabel(r'log$_{10}$(M$_{\mathrm{halo}}$ / M$_\odot$)')
    plt.ylabel('N galaxies')
    plt.title(f'LRG-like galaxies (N={N_GAL}), CV_0 z=0.47')
    out_hist = os.path.join(OUTPUT_DIR, 'lrg_halo_mass_distribution.png')
    plt.savefig(out_hist, dpi=150, bbox_inches='tight')
    plt.close()

    #9 save the LRG catalog to a .npz file
    out_cat = os.path.join(OUTPUT_DIR, 'lrg_catalog.npz')
    np.savez(
        out_cat,
        subhalo_idx=top_idx,
        pos=subhalos['SubhaloPos'][top_idx],
        vel=subhalos['SubhaloVel'][top_idx],
        m_star_msun=m_star_msun,
        m_halo_msun=m_halo_msun,
        r_halo_ckpc_h=r_halo_sim,
        grnr=grnr
    )
    print(f"Saved LRG catalog to: {out_cat}")

    print("Yay Done!")

if __name__ == "__main__":
    main()
