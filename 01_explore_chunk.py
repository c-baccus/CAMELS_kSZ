"""This is a test to explore one chunk of the CAMELS TNG50 simulations 
and print the header, particle types , and available fields. 
This just lets us confirm the layout of the data.

Usage:
    python 01_explore_chunk.py
"""

import h5py
import numpy as np

CHUNK_PATH = "/scratch/cjb9346/camels/CV_0/snapdir_074/snap_074.0.hdf5"

def main():
    print("=== Opening HDF5 file ===")
    with h5py.File(CHUNK_PATH, "r") as f:
        # Print Header attributes
        print("\n=== Header Attributes ===")
        if "Header" in f:
            hdr = f["Header"]
            for key, val in hdr.attrs.items():
                print(f"{key}: {val}")
        else:
            print("No Header group found.")

        # Print all top-level groups with member counts
        print("\n=== Top-Level Groups & Member Counts ===")
        for name in f:
            group = f[name]
            if isinstance(group, h5py.Group):
                n_members = len(group)
                print(f"{name}: {n_members} members")
            else:
                print(f"{name}: Not a group")

        # For PartType0, print dataset details
        if "PartType0" in f:
            print("\n=== PartType0 Dataset Details ===")
            pt0 = f["PartType0"]
            for dset_name in pt0:
                dset = pt0[dset_name]
                print(f"\nDataset: {dset_name}")
                print(f"  Shape: {dset.shape}")
                print(f"  Dtype: {dset.dtype}")
                try:
                    data = dset[()]
                    # Only do stats for numeric types
                    if np.issubdtype(data.dtype, np.number):
                        min_v = np.min(data)
                        mean_v = np.mean(data)
                        max_v = np.max(data)
                        print(f"  Min:   {min_v}")
                        print(f"  Mean:  {mean_v}")
                        print(f"  Max:   {max_v}")
                    else:
                        print("  (Not numeric; skipping min/mean/max)")
                except Exception as e:
                    print(f"  (Error reading dataset: {e})")
        else:
            print("No PartType0 group found.")

if __name__ == "__main__":
    main()