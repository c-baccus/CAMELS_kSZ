# CAMELS_kSZ

kSZ analysis pipeline for the CAMELS TNG50 simulations

## Goal

Reproduce the kSZ measurement of Hadzhiyska (https://arxiv.org/pdf/2407.07152), which were performed on TNG300 simulations, on CAMELS TNG50 simulations. I bin by halo mass, to study how AGN/SN feedback shapes the gas distribution around LRG-like galaxies.

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
