"""High-level API for interacting with OmniFold datasets.

Provides the OmniFoldDataset class — the primary entry point for
loading, inspecting, validating, and visualizing OmniFold results.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_loader import load_hdf5, inspect_hdf5, list_files
from .schema import DatasetMetadata, classify_columns
from .storage import export_dataset
from .validation import run_all_validations, ValidationResult


class OmniFoldDataset:
    """Unified interface for an OmniFold analysis result.

    Loads HDF5 files from a data directory and provides methods to
    access observables, weights, metadata, and validation results.

    Parameters
    ----------
    data_dir : str
        Directory containing HDF5 files and optional metadata YAML.
    metadata_path : str, optional
        Path to a YAML metadata file. If None, looks for
        'metadata.yaml' in data_dir.

    Examples
    --------
    >>> dataset = OmniFoldDataset("data/pseudodata/")
    >>> weights = dataset.load_weights("multifold")
    >>> obs = dataset.get_observable("multifold", "dimuon_pt")
    """

    def __init__(self, data_dir: str, metadata_path: Optional[str] = None):
        self.data_dir = data_dir
        self._dataframes: Dict[str, pd.DataFrame] = {}
        self._metadata: Optional[DatasetMetadata] = None

        if metadata_path and os.path.exists(metadata_path):
            self._metadata = DatasetMetadata.from_yaml(metadata_path)
        else:
            auto_path = os.path.join(data_dir, "metadata.yaml")
            if os.path.exists(auto_path):
                self._metadata = DatasetMetadata.from_yaml(auto_path)

    @property
    def files(self) -> List[str]:
        """List available HDF5 files in the data directory."""
        return list_files(self.data_dir)

    @property
    def file_names(self) -> List[str]:
        """List base names (without extension) of available files."""
        return [
            os.path.splitext(os.path.basename(f))[0] for f in self.files
        ]

    @property
    def metadata(self) -> Optional[DatasetMetadata]:
        """Access the dataset metadata, if available."""
        return self._metadata

    def load(self, name: str) -> pd.DataFrame:
        """Load a specific HDF5 file by its base name.

        Parameters
        ----------
        name : str
            Base name of the file (e.g. 'multifold' for 'multifold.h5').

        Returns
        -------
        pd.DataFrame
        """
        if name not in self._dataframes:
            filepath = os.path.join(self.data_dir, f"{name}.h5")
            self._dataframes[name] = load_hdf5(filepath)
        return self._dataframes[name]

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all HDF5 files in the data directory."""
        for name in self.file_names:
            self.load(name)
        return self._dataframes

    def load_weights(self, name: str) -> pd.DataFrame:
        """Load only weight columns from a specific file.

        Parameters
        ----------
        name : str
            Base name of the file.

        Returns
        -------
        pd.DataFrame containing weight columns only.
        """
        df = self.load(name)
        classified = classify_columns(df.columns.tolist())
        return df[classified["weights"]]

    def load_observables(self, name: str) -> pd.DataFrame:
        """Load only observable columns from a specific file.

        Parameters
        ----------
        name : str
            Base name of the file.

        Returns
        -------
        pd.DataFrame containing observable columns only.
        """
        df = self.load(name)
        classified = classify_columns(df.columns.tolist())
        return df[classified["observables"]]

    def get_observable(self, name: str, observable: str) -> np.ndarray:
        """Get a single observable as a NumPy array.

        Parameters
        ----------
        name : str
            Base name of the file.
        observable : str
            Column name of the observable.

        Returns
        -------
        np.ndarray
        """
        df = self.load(name)
        if observable not in df.columns:
            raise KeyError(
                f"Observable '{observable}' not found. "
                f"Available: {df.columns.tolist()}"
            )
        return df[observable].values

    def describe(self, name: str) -> Dict:
        """Get a summary of a loaded file's contents.

        Parameters
        ----------
        name : str
            Base name of the file.

        Returns
        -------
        dict with keys 'n_events', 'columns', 'weight_columns',
        'observable_columns', 'other_columns'
        """
        df = self.load(name)
        classified = classify_columns(df.columns.tolist())
        return {
            "name": name,
            "n_events": len(df),
            "n_columns": len(df.columns),
            "columns": classified,
            "weight_stats": df[classified["weights"]].describe().to_dict()
            if classified["weights"]
            else {},
        }

    def validate(self) -> List[ValidationResult]:
        """Run all validation checks on loaded data.

        Loads all files if not already loaded.

        Returns
        -------
        list of ValidationResult
        """
        if not self._dataframes:
            self.load_all()
        return run_all_validations(self._dataframes)

    def export(
        self,
        name: str,
        output_dir: str,
        fmt: str = "hdf5",
    ) -> str:
        """Export a dataset to a specified format.

        Parameters
        ----------
        name : str
            Base name of the file to export.
        output_dir : str
            Directory for the exported files.
        fmt : str
            Output format: 'hdf5', 'parquet', or 'numpy'.

        Returns
        -------
        str
            Path to the exported file.
        """
        df = self.load(name)
        meta = None
        if self._metadata:
            meta = {
                "analysis_name": self._metadata.analysis_name,
                "version": self._metadata.version,
            }
        return export_dataset(df, output_dir, name, fmt=fmt, metadata=meta)
