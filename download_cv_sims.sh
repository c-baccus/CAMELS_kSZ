#!/bin/bash
# Download CV_1 through CV_26 from CAMELS to /scratch/cjb9346/camels/
# Only snapdir_074 (z=0.47) and groups_074 are transferred.

SIMS_ENDPOINT="8bfa28e1-0de9-41ea-a197-8114998b8646"      # CAMELS-Sims-TNG-L50n512
GROUPS_ENDPOINT="8b8707aa-04db-4f2c-b60e-0f88bb51f735"    # CAMELS-FOFSubfind-TNG-L50n512
TORCH_ENDPOINT="31fa4572-cd84-489b-8008-0bf0e52bb4d4"     # Torch scratch directory

DEST_BASE="/scratch/cjb9346/camels"   # path within Torch scratch endpoint

for i in $(seq 1 26); do
    echo "=== Submitting CV_$i ==="

    # Create destination directory (ignore "already exists" errors)
    globus mkdir "${TORCH_ENDPOINT}:${DEST_BASE}/CV_${i}" 2>/dev/null || true

    # Snapshot transfer (the big one)
    globus transfer --recursive \
        --label "CV_${i}_snapdir_074" \
        "${SIMS_ENDPOINT}:/CV/CV_${i}/snapdir_074/" \
        "${TORCH_ENDPOINT}:${DEST_BASE}/CV_${i}/snapdir_074/"

    # Group catalog transfer (small)
    globus transfer --recursive \
        --label "CV_${i}_groups_074" \
        "${GROUPS_ENDPOINT}:/CV/CV_${i}/groups_074/" \
        "${TORCH_ENDPOINT}:${DEST_BASE}/CV_${i}/groups_074/"

    sleep 1   # avoid rate-limiting
done

echo "All 52 transfers submitted!"
echo "Monitor with: globus task list"
