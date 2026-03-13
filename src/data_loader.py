"""Data loading utilities for OmniFold HDF5 output files.

Handles reading HDF5 files that contain Pandas DataFrames of
particle-level events with OmniFold weights.
"""

import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

from .schema import classify_columns


def load_hdf5(filepath: str, key: Optional[str] = None) -> pd.DataFrame:
    """Load a Pandas DataFrame from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    key : str, optional
        HDF5 group key. If None, reads the first available key.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If no valid DataFrame is found in the file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if key is not None:
        return pd.read_hdf(filepath, key=key)

    # Try to auto-detect the key
    with h5py.File(filepath, "r") as f:
        keys = list(f.keys())

    if not keys:
        raise ValueError(f"No datasets found in {filepath}")

    return pd.read_hdf(filepath, key=keys[0])


def inspect_hdf5(filepath: str) -> Dict:
    """Inspect the structure of an HDF5 file without loading full data.

    Returns a summary dict with keys, shapes, column names, and dtypes.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.

    Returns
    -------
    dict
        Summary of file contents.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    summary = {"filepath": filepath, "groups": {}}

    with h5py.File(filepath, "r") as f:
        for key in f.keys():
            group = f[key]
            group_info = {
                "type": type(group).__name__,
                "attrs": dict(group.attrs),
            }
            if hasattr(group, "keys"):
                group_info["children"] = list(group.keys())
            if hasattr(group, "shape"):
                group_info["shape"] = group.shape
                group_info["dtype"] = str(group.dtype)
            summary["groups"][key] = group_info

    return summary


def load_weights(filepath: str, key: Optional[str] = None) -> pd.DataFrame:
    """Load only the weight columns from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    key : str, optional
        HDF5 group key.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only weight columns.
    """
    df = load_hdf5(filepath, key=key)
    classified = classify_columns(df.columns.tolist())
    weight_cols = classified["weights"]

    if not weight_cols:
        raise ValueError(f"No weight columns found in {filepath}")

    return df[weight_cols]


def load_observables(filepath: str, key: Optional[str] = None) -> pd.DataFrame:
    """Load only the observable columns from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.
    key : str, optional
        HDF5 group key.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only observable columns.
    """
    df = load_hdf5(filepath, key=key)
    classified = classify_columns(df.columns.tolist())
    obs_cols = classified["observables"]

    if not obs_cols:
        raise ValueError(f"No observable columns found in {filepath}")

    return df[obs_cols]


def list_files(data_dir: str, pattern: str = ".h5") -> List[str]:
    """List all HDF5 files in a directory.

    Parameters
    ----------
    data_dir : str
        Directory to search.
    pattern : str
        File extension to match.

    Returns
    -------
    list of str
        Sorted list of matching file paths.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    files = [
        os.path.join(data_dir, f)
        for f in sorted(os.listdir(data_dir))
        if f.endswith(pattern)
    ]
    return files
