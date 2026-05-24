"""
io_camels.py: input/output functions for CAMELS TNG50 simulations

The snapshots and groups come in 16 chunks each.
This script concatenates the fields from across chunks into single numpy arrays.
"""
import os
import re
import numpy as np
import h5py


def load_header(path):
    """
    Read HDF5 header attributes from a single chunk into a dictionary.

    Parameters
    ----------
    path : str
        Path to one HDF5 chunk file.

    Returns
    -------
    dict
        Header attributes as {key: value} pairs.
    """
    with h5py.File(path, "r") as f:
        header_group = f["Header"]
        header_attrs = dict(header_group.attrs.items())
    return header_attrs


def load_subhalo_catalog(group_dir, snapshot_num=74, fields=None):
    """
    Load subhalo catalog from CAMELS TNG50 FoF/Subfind output.

    Reads all 16 chunks of fof_subhalo_tab_<snapshot_num>.<i>.hdf5 and
    concatenates the requested fields.

    Parameters
    ----------
    group_dir : str
        Directory containing the chunks (e.g. /scratch/.../groups_074).
    snapshot_num : int
        Snapshot number (default 74).
    fields : list of str, optional
        Subhalo fields to load. Defaults to fields needed for kSZ analysis:
        ['SubhaloMassType', 'SubhaloPos', 'SubhaloVel', 'SubhaloGrNr', 'SubhaloMass'].

    Returns
    -------
    dict
        {field_name: concatenated_numpy_array}
    """
    default_fields = ['SubhaloMassType', 'SubhaloPos', 'SubhaloVel', 'SubhaloGrNr', 'SubhaloMass']
    if fields is None:
        fields = default_fields

    snap_str = str(snapshot_num).zfill(3)
    pattern = re.compile(rf"fof_subhalo_tab_{snap_str}\.(\d+)\.hdf5$")

    # List and numerically sort all matching files
    all_files = [f for f in os.listdir(group_dir) if pattern.match(f)]
    file_with_indices = []
    for fname in all_files:
        m = pattern.match(fname)
        if m:
            idx = int(m.group(1))
            file_with_indices.append((idx, fname))
    file_with_indices.sort()
    sorted_files = [fname for idx, fname in file_with_indices]

    # For each field, collect arrays from each chunk
    results = {field: [] for field in fields}

    for fname in sorted_files:
        fpath = os.path.join(group_dir, fname)
        with h5py.File(fpath, 'r') as f:
            subhalo_group = f['Subhalo']
            for field in fields:
                if field not in subhalo_group:
                    raise KeyError(f"Field '{field}' not found in {fpath}")
                results[field].append(subhalo_group[field][...])

    # Concatenate along axis 0
    for field in fields:
        results[field] = np.concatenate(results[field], axis=0)

    return results


def load_group_catalog(group_dir, snapshot_num=74, fields=None):
    """
    Load FoF group catalog from CAMELS TNG50.

    Same chunking logic as load_subhalo_catalog, but reads from the
    Group group instead of the Subhalo group.

    Parameters
    ----------
    group_dir : str
        Directory containing the chunks.
    snapshot_num : int
        Snapshot number (default 74).
    fields : list of str, optional
        Group fields to load. Defaults to a kSZ-relevant set:
        ['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos', 'GroupFirstSub', 'GroupNsubs'].

    Returns
    -------
    dict
        {field_name: concatenated_numpy_array}
    """
    default_fields = ['Group_M_Crit200', 'Group_R_Crit200', 'GroupPos', 'GroupFirstSub', 'GroupNsubs']
    if fields is None:
        fields = default_fields

    snap_str = str(snapshot_num).zfill(3)
    pattern = re.compile(rf"fof_subhalo_tab_{snap_str}\.(\d+)\.hdf5$")

    # List and numerically sort all matching files
    all_files = [f for f in os.listdir(group_dir) if pattern.match(f)]
    file_with_indices = []
    for fname in all_files:
        m = pattern.match(fname)
        if m:
            idx = int(m.group(1))
            file_with_indices.append((idx, fname))
    file_with_indices.sort()
    sorted_files = [fname for idx, fname in file_with_indices]

    # For each field, collect arrays from each chunk
    results = {field: [] for field in fields}

    for fname in sorted_files:
        fpath = os.path.join(group_dir, fname)
        with h5py.File(fpath, 'r') as f:
            group_group = f['Group']
            for field in fields:
                if field not in group_group:
                    raise KeyError(f"Field '{field}' not found in {fpath}")
                results[field].append(group_group[field][...])

    # Concatenate along axis 0
    for field in fields:
        results[field] = np.concatenate(results[field], axis=0)

    return results


def load_all_gas(snap_dir, snapshot_num=74, fields=None):
    """
    Load and concatenate all gas particle (PartType0) fields from a CAMELS snapshot.

    Reads all 16 chunks of snap_<snapshot_num>.<i>.hdf5 and concatenates
    the requested fields. Chunks with no PartType0 group are skipped.

    Parameters
    ----------
    snap_dir : str
        Directory containing the snapshot files (e.g. /scratch/.../snapdir_074).
    snapshot_num : int
        Snapshot number (default 74). Will be zero-padded to 3 digits.
    fields : list of str, optional
        Gas fields to load. Defaults to fields needed for kSZ analysis:
        ['Coordinates', 'Velocities', 'Masses', 'Density', 'ElectronAbundance'].

    Returns
    -------
    dict
        {field_name: concatenated_numpy_array}
    """
    default_fields = ['Coordinates', 'Velocities', 'Masses', 'Density', 'ElectronAbundance']
    if fields is None:
        fields = default_fields

    snap_str = str(snapshot_num).zfill(3)
    pattern = re.compile(rf"^snap_{snap_str}\.(\d+)\.hdf5$")

    # List and numerically sort all matching files
    all_files = [f for f in os.listdir(snap_dir) if pattern.match(f)]
    file_with_indices = []
    for fname in all_files:
        m = pattern.match(fname)
        if m:
            idx = int(m.group(1))
            file_with_indices.append((idx, fname))
    file_with_indices.sort()
    sorted_files = [fname for idx, fname in file_with_indices]

    results = {field: [] for field in fields}
    n_chunks = 0

    for fname in sorted_files:
        fpath = os.path.join(snap_dir, fname)
        try:
            with h5py.File(fpath, "r") as f:
                if 'PartType0' not in f:
                    print(f"Warning: PartType0 not found in {fname}; skipping (no gas particles)")
                    continue
                p0 = f['PartType0']
                for field in fields:
                    if field not in p0:
                        raise KeyError(f"Field '{field}' not found in PartType0 of {fpath}")
                    results[field].append(p0[field][...])
                n_chunks += 1
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            continue

    # Concatenate across all chunks
    for field in fields:
        results[field] = np.concatenate(results[field], axis=0)

    n_final = results[fields[0]].shape[0]
    print(f"Loaded PartType0 from {n_chunks} chunks. Total particle count: {n_final}")

    return results