"""
diagnose_centering.py

Investigate why the stacked kSZ blob isn't centered on (0,0).
For each LRG, compute:
  - distance from SubhaloPos to GroupPos (host halo gas center proxy)
  - distance from SubhaloPos to box edges (to flag periodic-edge candidates)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from io_camels import load_group_catalog, load_subhalo_catalog

GROUP_DIR = "/scratch/cjb9346/camels/CV_0/groups_074"
CATALOG_PATH = "outputs/lrg_catalog.npz"
OUTPUT_DIR = "outputs"
BOX_SIZE = 50000.0  # ckpc/h


def main():
    cat = np.load(CATALOG_PATH)
    
    # Need GroupPos for each LRG's host halo - load the full group catalog
    groups = load_group_catalog(GROUP_DIR)
    
    grnr = cat['grnr']           # FoF group number for each LRG
    sub_pos = cat['pos']          # SubhaloPos in ckpc/h
    group_pos = groups['GroupPos'][grnr]
    
    # Periodic offsets (galaxy from its halo center)
    offset = sub_pos - group_pos
    offset = (offset + BOX_SIZE / 2) % BOX_SIZE - BOX_SIZE / 2
    offset_mag = np.linalg.norm(offset, axis=1)
    
    # Distance from each LRG to nearest box edge (along any axis)
    dist_to_edge = np.minimum(sub_pos, BOX_SIZE - sub_pos).min(axis=1)
    
    log_m_halo = np.log10(cat['m_halo_msun'])
    
    print(f"=== Subhalo-Group center offset (per LRG) ===")
    print(f"  median: {np.median(offset_mag):.2f} ckpc/h")
    print(f"  mean:   {offset_mag.mean():.2f} ckpc/h")
    print(f"  max:    {offset_mag.max():.2f} ckpc/h")
    print(f"  >100 ckpc/h: {np.sum(offset_mag > 100)} galaxies")
    print(f"  >500 ckpc/h: {np.sum(offset_mag > 500)} galaxies (clearly off-center)")
    
    print(f"\n=== Distance to nearest box edge ===")
    print(f"  min:    {dist_to_edge.min():.1f} ckpc/h")
    print(f"  median: {np.median(dist_to_edge):.1f} ckpc/h")
    print(f"  <5000 ckpc/h (within query radius): {np.sum(dist_to_edge < 5000)} galaxies")
    
    high_mass_mask = log_m_halo >= 13.5
    print(f"\n=== High-mass bin (log10 M_h >= 13.5, N={high_mass_mask.sum()}) ===")
    print(f"  Subhalo-Group offsets: median={np.median(offset_mag[high_mass_mask]):.2f}, "
          f"max={offset_mag[high_mass_mask].max():.2f} ckpc/h")
    print(f"  Distance to edge:      min={dist_to_edge[high_mass_mask].min():.1f}, "
          f"median={np.median(dist_to_edge[high_mass_mask]):.1f} ckpc/h")
    
    # Plot
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(offset_mag, bins=20, alpha=0.5, label='All LRGs')
    axes[0].hist(offset_mag[high_mass_mask], bins=20, alpha=0.7, color='r', label='High-mass bin')
    axes[0].set_xlabel('|SubhaloPos - GroupPos| [ckpc/h]')
    axes[0].set_ylabel('N galaxies')
    axes[0].set_title('Galaxy offset from halo center')
    axes[0].legend()
    
    axes[1].scatter(log_m_halo, offset_mag, alpha=0.6, s=20)
    axes[1].set_xlabel('log10(M_halo / M_sun)')
    axes[1].set_ylabel('|SubhaloPos - GroupPos| [ckpc/h]')
    axes[1].set_title('Offset vs halo mass')
    
    axes[2].scatter(log_m_halo, dist_to_edge, alpha=0.6, s=20)
    axes[2].axhline(5000, color='r', ls='--', label='query radius (5 Mpc/h)')
    axes[2].set_xlabel('log10(M_halo / M_sun)')
    axes[2].set_ylabel('Dist to nearest box edge [ckpc/h]')
    axes[2].set_title('Periodic edge proximity')
    axes[2].legend()
    
    plt.suptitle('Centering diagnostic')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'centering_diagnostic.png'), dpi=130, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {OUTPUT_DIR}/centering_diagnostic.png")


if __name__ == "__main__":
    main()