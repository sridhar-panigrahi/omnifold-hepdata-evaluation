"""Storage utilities for saving and loading OmniFold results.

Provides a uniform interface for persisting OmniFold outputs in
multiple formats (HDF5, Parquet, NumPy) alongside their metadata.
"""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml


def save_weights_hdf5(
    weights: pd.DataFrame,
    filepath: str,
    key: str = "weights",
    metadata: Optional[Dict] = None,
) -> None:
    """Save weight DataFrame to HDF5 with optional metadata attributes.

    Parameters
    ----------
    weights : pd.DataFrame
        Weight data to save.
    filepath : str
        Output file path.
    key : str
        HDF5 group key.
    metadata : dict, optional
        Key-value pairs stored as HDF5 attributes.
    """
    weights.to_hdf(filepath, key=key, mode="w")

    if metadata:
        import h5py
        with h5py.File(filepath, "a") as f:
            for k, v in metadata.items():
                f.attrs[k] = str(v)


def save_weights_parquet(weights: pd.DataFrame, filepath: str) -> None:
    """Save weight DataFrame to Parquet format.

    Parameters
    ----------
    weights : pd.DataFrame
        Weight data to save.
    filepath : str
        Output file path.
    """
    weights.to_parquet(filepath, index=False)


def save_weights_numpy(weights: np.ndarray, filepath: str) -> None:
    """Save weight array to NumPy .npz format.

    Parameters
    ----------
    weights : np.ndarray
        Weight array to save.
    filepath : str
        Output file path.
    """
    np.savez_compressed(filepath, weights=weights)


def save_metadata(metadata: Dict, filepath: str) -> None:
    """Save metadata dictionary to YAML.

    Parameters
    ----------
    metadata : dict
        Metadata to serialize.
    filepath : str
        Output YAML file path.
    """
    with open(filepath, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)


def load_metadata(filepath: str) -> Dict:
    """Load metadata from a YAML file.

    Parameters
    ----------
    filepath : str
        Path to the YAML metadata file.

    Returns
    -------
    dict
    """
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def export_dataset(
    data: pd.DataFrame,
    output_dir: str,
    name: str,
    fmt: str = "hdf5",
    metadata: Optional[Dict] = None,
) -> str:
    """Export a dataset with its metadata to the specified format.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to export.
    output_dir : str
        Directory to write files into.
    name : str
        Base name for the output files.
    fmt : str
        Output format: 'hdf5', 'parquet', or 'numpy'.
    metadata : dict, optional
        Metadata to save alongside the data.

    Returns
    -------
    str
        Path to the saved data file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if fmt == "hdf5":
        data_path = os.path.join(output_dir, f"{name}.h5")
        save_weights_hdf5(data, data_path, metadata=metadata)
    elif fmt == "parquet":
        data_path = os.path.join(output_dir, f"{name}.parquet")
        save_weights_parquet(data, data_path)
    elif fmt == "numpy":
        data_path = os.path.join(output_dir, f"{name}.npz")
        save_weights_numpy(data.values, data_path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    if metadata:
        meta_path = os.path.join(output_dir, f"{name}_metadata.yaml")
        save_metadata(metadata, meta_path)

    return data_path
