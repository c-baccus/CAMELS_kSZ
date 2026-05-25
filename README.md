# CAMELS_kSZ
## kSZ analysis pipeline for the CAMELS TNG50 simulations

**Status (May 25):** Pipeline working on one CAMELS TNG50 CV sim (CV_0, z=0.47). Selects 100 LRG-like centrals by stellar mass (`02_select_lrgs.py`, median log10 M_halo = 13.0), builds a periodic cKDTree on 130M gas particles and produces per-galaxy τ and kSZ maps in three projections — xy, yz, zx (`04_make_all_maps.py`, using `kdtree_maps.py`), stacks by halo mass in three bins, beam-smooths at 1.6 arcmin, and computes CAP profiles (`05_stack_by_mass.py`). Halo mass distribution and per-galaxy maps look physically sensible (clear massive cluster centers, filament structure, and ~10⁻³ peak τ). Stacked CAP profile shows the expected mass ordering (high-mass bin ~8× larger amplitude than low-mass), but profiles are noisy and turn negative at large apertures (perhaps because we're doing raw rather than velocity-weighted stacking like Boryana and only have one CV sim).

**Issues:**
- All three curves cross at small apertures (θ < 1') instead of separating cleanly — likely because the 1.6' beam smears out signal at scales below the beam, and tiny apertures contain mostly noise
- High-mass curve (green) is "bell-shaped" with peak at θ ≈ 3' instead of the monotonically rising profile in the paper — probably because raw (not velocity-weighted) stacking lets adjacent halos with opposite-sign kSZ partially cancel beyond R_200c
- Mid-mass curve (orange) plunges sharply negative at θ > 4.5' — too steep to be a real signal, looks like one or two unlucky galaxies dominating with strong negative kSZ contributions near the bin edge (only 32 galaxies in this bin)
- Low-mass curve (blue) is essentially flat near zero, consistent with small halos having weak kSZ — but also possibly indicating the bin is too noisy to recover signal at all with N=53 galaxies
- Profile shape will likely change substantially after (a) adding velocity-weighted stacking and (b) averaging across all 27 CV sims for ×5 better noise floor

## Goal

Reproduce the kSZ measurement of Hadzhiyska et. al (https://arxiv.org/pdf/2407.07152), (performed on TNG300 simulations), on CAMELS TNG50 simulations. Binned by halo mass, to study how AGN/SN feedback shapes the gas distribution around LRG-like galaxies.

## Approach

1. Load CAMELS TNG50 snapshot at z = 0.47 (snapshot 074)
2. Select LRG-like galaxies by ranking subhalos by stellar mass
3. Build a periodic KD-tree on gas particle positions
4. For each galaxy, query nearby gas particles and construct a 2D kSZ map
5. Stack maps in bins of halo mass 
6. Extract CAP profiles per mass bin

## Data

Snapshots downloaded via Globus from the CAMELS public collections
(`CAMELS-Sims-TNG-L50n512`, `CAMELS-FOFSubfind-TNG-L50n512`).
Data lives in `/scratch/cjb9346/camels/` (not in repo, files are 100s of GB).

## Running

Pipeline scripts are numbered `01_*.py` through `04_*.py` and run in order.
See individual scripts for usage.
