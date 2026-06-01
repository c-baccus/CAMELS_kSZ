#!/bin/bash
# Download 1P sims at extreme values (_n2 and _2) for:
#   max-variance: 3, 12, 26
#   high-variance: 4, 5, 6, 13, 15, 16, 18, 25, 27
# Total: 12 parameters × 2 extremes = 24 sims

SIMS_ENDPOINT="8bfa28e1-0de9-41ea-a197-8114998b8646"
GROUPS_ENDPOINT="8b8707aa-04db-4f2c-b60e-0f88bb51f735"
TORCH_ENDPOINT="31fa4572-cd84-489b-8008-0bf0e52bb4d4"

DEST_BASE="/scratch/cjb9346/camels_1p"

# Max-variance parameters (top 3)
MAX_VARS=(3 12 26)
# High-variance parameters (next 9)
HIGH_VARS=(4 5 6 13 15 16 18 25 27)

# Combine into one list
ALL_PARAMS=("${MAX_VARS[@]}" "${HIGH_VARS[@]}")

echo "Will download 1P sims for parameters: ${ALL_PARAMS[@]}"
echo "Each at extreme values _n2 and _2 (24 sims total)"
echo ""

# Pre-create destination directories
mkdir -p $DEST_BASE
for param in "${ALL_PARAMS[@]}"; do
    for val in n2 2; do
        mkdir -p $DEST_BASE/1P_p${param}_${val}
    done
done

# Submit transfers
count=0
for param in "${ALL_PARAMS[@]}"; do
    for val in n2 2; do
        sim_name="1P_p${param}_${val}"
        count=$((count + 1))
        echo "=== [$count/24] Submitting $sim_name ==="
        
        globus transfer --recursive \
            --label "${sim_name}_snapdir_074" \
            "${SIMS_ENDPOINT}:/1P/${sim_name}/snapdir_074/" \
            "${TORCH_ENDPOINT}:${DEST_BASE}/${sim_name}/snapdir_074/"
        
        globus transfer --recursive \
            --label "${sim_name}_groups_074" \
            "${GROUPS_ENDPOINT}:/1P/${sim_name}/groups_074/" \
            "${TORCH_ENDPOINT}:${DEST_BASE}/${sim_name}/groups_074/"
        
        sleep 1
    done
done

echo ""
echo "All 48 transfers submitted (24 sims x 2 file types)!"
echo "Monitor with: globus task list | grep 1P_ | awk '{print \$3}' | sort | uniq -c"
