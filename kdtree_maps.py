"""
kdtree_maps.py

Per-galaxy 2D kSZ map construction using a periodic KD-tree on gas particles.

Workflow:
  1. load_all_gas() (from io_camels) concatenates gas fields across the 16 chunks
  2. compute_per_cell_quantities() converts raw fields to physical units
  3. build_periodic_kdtree() builds a tree with periodic boundary conditions
  4. make_galaxy_maps() queries one galaxy's neighborhood and produces 2D maps
"""

import os
import numpy as np
from scipy.spatial import cKDTree

from io_camels import load_all_gas


# ---- Physical constants (cgs) ----
SIGMA_T = 6.6524587158e-25       # Thomson cross section, cm^2
M_P = 1.6726e-24                 # proton mass, g
C_CGS = 2.99792458e10            # speed of light, cm/s
X_H = 0.76                       # primordial hydrogen mass fraction
KPC_TO_CM = 3.0857e21            # 1 kpc in cm
SOLAR_MASS_G = 1.989e33          # 1 solar mass in grams
GAMMA = 5.0 / 3.0                # adiabatic index for monatomic ideal gas


def compute_per_cell_quantities(gas, hubble_h, scale_factor):
    """
    Convert raw HDF5 gas fields into physical quantities needed for kSZ maps.

    Computes:
      - n_e_V (cell electron count, dimensionless)
      - V_cm3 (cell volume in cm^3)
      - v_cgs (peculiar velocity in cm/s, shape (N, 3))

    Parameters
    ----------
    gas : dict
        Output of load_all_gas() with raw HDF5 fields.
    hubble_h : float
        Hubble parameter from the snapshot header.
    scale_factor : float
        a = 1/(1+z) from the snapshot header.

    Returns
    -------
    dict with keys 'n_e_V', 'V_cm3', 'v_cgs', 'pos_ckpc_h'.
    """
    # Compute cell mass in grams
    M_cell_g = gas['Masses'] * 1e10 * SOLAR_MASS_G / hubble_h  # grams

    # Compute electron count per cell (dimensionless, electron number)
    n_e_V = gas['ElectronAbundance'] * X_H * M_cell_g / M_P

    # Compute cell volume in (ckpc/h)^3, then convert to cm^3 (physical)
    V_sim = gas['Masses'] / gas['Density']                    # (ckpc/h)^3, sim units
    V_cm3 = V_sim * (scale_factor * KPC_TO_CM / hubble_h)**3  # cm^3

    # Compute peculiar velocity in cm/s (shape (N,3))
    v_cgs = gas['Velocities'] * np.sqrt(scale_factor) * 1e5   # cm/s

    # Particle positions (ckpc/h)
    pos_ckpc_h = gas['Coordinates']

    # Print basic sanity stats
    print(f"n_e_V [electron count]: min={n_e_V.min():.3e}, mean={n_e_V.mean():.3e}, max={n_e_V.max():.3e}")
    print(f"V_cm3 [cm^3]: min={V_cm3.min():.3e}, mean={V_cm3.mean():.3e}, max={V_cm3.max():.3e}")
    print(f"v_cgs mean |v| [cm/s]: mean={np.linalg.norm(v_cgs, axis=1).mean():.3e}")

    return {
        'n_e_V': n_e_V,
        'V_cm3': V_cm3,
        'v_cgs': v_cgs,
        'pos_ckpc_h': pos_ckpc_h
    }


def build_periodic_kdtree(positions, box_size):
    """
    Build a cKDTree with periodic boundary conditions.

    Parameters
    ----------
    positions : (N, 3) array
        Particle positions in ckpc/h.
    box_size : float
        Box length in ckpc/h.

    Returns
    -------
    cKDTree
        Tree object usable for query_ball_point with periodic wrap.
    """
    positions = np.mod(positions, box_size) 
    tree = cKDTree(positions, boxsize=box_size)
    print(f"Periodic cKDTree built: {positions.shape[0]} points, box size = {box_size}")
    return tree


def make_galaxy_maps(
    tree,
    cell_data,
    galaxy_pos,
    radius_ckpc_h,
    hubble_h,
    scale_factor,
    n_pix=250,
    box_size=50000.0,
):
    """
    For one galaxy, query nearby gas particles and produce 2D tau and kSZ maps.

    Three projections (xy, yz, zx) are produced for each of tau and b (kSZ momentum).

    Parameters
    ----------
    tree : cKDTree
        Periodic KD-tree from build_periodic_kdtree().
    cell_data : dict
        Output of compute_per_cell_quantities().
    galaxy_pos : (3,) array
        Galaxy position in ckpc/h.
    radius_ckpc_h : float
        Query radius around the galaxy in ckpc/h.
    hubble_h : float
        Hubble parameter from the snapshot header.
    scale_factor : float
        a = 1/(1+z) from the snapshot header.
    n_pix : int
        Pixels per side of the output maps.
    box_size : float
        Simulation box size in ckpc/h (for periodic offset wrapping).

    Returns
    -------
    dict with keys 'tau_xy', 'tau_yz', 'tau_zx', 'b_xy', 'b_yz', 'b_zx', 'n_cells'.
    """
    # Query particles within sphere of given radius (periodic-aware)
    indices = np.array(tree.query_ball_point(galaxy_pos, r=radius_ckpc_h))

    # If no particles found, return all-zero maps
    if len(indices) == 0:
        empty = {key: np.zeros((n_pix, n_pix)) for key in
                 ['tau_xy', 'tau_yz', 'tau_zx', 'b_xy', 'b_yz', 'b_zx']}
        empty['n_cells'] = 0
        return empty

    # Subset cell data arrays
    n_e_V_sub = cell_data['n_e_V'][indices]
    v_cgs_sub = cell_data['v_cgs'][indices]
    pos_sub = cell_data['pos_ckpc_h'][indices]

    # Compute periodic offsets (wrap into [-box_size/2, +box_size/2])
    offsets = pos_sub - galaxy_pos
    offsets = (offsets + box_size / 2) % box_size - box_size / 2

    # Bin edges centered on the galaxy
    bin_edges = np.linspace(-radius_ckpc_h, radius_ckpc_h, n_pix + 1)
    pixel_size_ckpc_h = bin_edges[1] - bin_edges[0]

    # Per-pixel physical area in cm^2
    pixel_size_cm = pixel_size_ckpc_h * scale_factor * KPC_TO_CM / hubble_h
    A_pix_cm2 = pixel_size_cm ** 2

    # Make tau and b maps for each of the three projections
    tau_maps = {}
    b_maps = {}
    projections = [
        ('xy', (0, 1, 2)),  # bin in (x, y); LOS = z
        ('yz', (1, 2, 0)),  # bin in (y, z); LOS = x
        ('zx', (2, 0, 1)),  # bin in (z, x); LOS = y
    ]

    for proj_name, (ax_i, ax_j, ax_los) in projections:
        weights_tau = n_e_V_sub
        weights_b = n_e_V_sub * v_cgs_sub[:, ax_los] / C_CGS

        H_tau, _, _ = np.histogram2d(
            offsets[:, ax_i], offsets[:, ax_j],
            bins=bin_edges, weights=weights_tau,
        )
        H_b, _, _ = np.histogram2d(
            offsets[:, ax_i], offsets[:, ax_j],
            bins=bin_edges, weights=weights_b,
        )

        tau_maps[proj_name] = SIGMA_T * H_tau / A_pix_cm2
        b_maps[proj_name] = SIGMA_T * H_b / A_pix_cm2

    return {
        'tau_xy': tau_maps['xy'],
        'tau_yz': tau_maps['yz'],
        'tau_zx': tau_maps['zx'],
        'b_xy': b_maps['xy'],
        'b_yz': b_maps['yz'],
        'b_zx': b_maps['zx'],
        'n_cells': len(indices),
    }