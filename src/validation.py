"""Validation utilities for OmniFold datasets.

Checks dataset integrity: weight normalization, column consistency,
observable ranges, and cross-file agreement.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .schema import classify_columns


class ValidationResult:
    """Container for a single validation check result."""

    def __init__(self, name: str, passed: bool, message: str):
        self.name = name
        self.passed = passed
        self.message = message

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


def validate_weights_positive(df: pd.DataFrame) -> ValidationResult:
    """Check that all weight columns contain non-negative values.

    OmniFold weights are likelihood ratios and should be >= 0.
    """
    classified = classify_columns(df.columns.tolist())
    weight_cols = classified["weights"]

    if not weight_cols:
        return ValidationResult(
            "weights_positive", False, "No weight columns found"
        )

    for col in weight_cols:
        if (df[col] < 0).any():
            n_neg = (df[col] < 0).sum()
            return ValidationResult(
                "weights_positive",
                False,
                f"Column '{col}' has {n_neg} negative values",
            )

    return ValidationResult(
        "weights_positive", True, f"All {len(weight_cols)} weight columns non-negative"
    )


def validate_weights_finite(df: pd.DataFrame) -> ValidationResult:
    """Check that weight columns contain no NaN or Inf values."""
    classified = classify_columns(df.columns.tolist())
    weight_cols = classified["weights"]

    if not weight_cols:
        return ValidationResult(
            "weights_finite", False, "No weight columns found"
        )

    for col in weight_cols:
        if not np.isfinite(df[col]).all():
            n_bad = (~np.isfinite(df[col])).sum()
            return ValidationResult(
                "weights_finite",
                False,
                f"Column '{col}' has {n_bad} non-finite values",
            )

    return ValidationResult(
        "weights_finite", True, "All weight columns are finite"
    )


def validate_event_count_consistency(
    dataframes: Dict[str, pd.DataFrame],
) -> ValidationResult:
    """Check that all files have the same number of events.

    For OmniFold systematic variations applied to the same MC sample,
    each file should contain the same events.
    """
    counts = {name: len(df) for name, df in dataframes.items()}
    unique_counts = set(counts.values())

    if len(unique_counts) == 1:
        n = unique_counts.pop()
        return ValidationResult(
            "event_count_consistency",
            True,
            f"All {len(dataframes)} files have {n} events",
        )

    details = ", ".join(f"{name}={count}" for name, count in counts.items())
    return ValidationResult(
        "event_count_consistency",
        False,
        f"Inconsistent event counts: {details}",
    )


def validate_column_consistency(
    dataframes: Dict[str, pd.DataFrame],
) -> ValidationResult:
    """Check that all files share the same column structure."""
    col_sets = {name: set(df.columns) for name, df in dataframes.items()}
    reference_name = list(col_sets.keys())[0]
    reference_cols = col_sets[reference_name]

    mismatches = []
    for name, cols in col_sets.items():
        if cols != reference_cols:
            extra = cols - reference_cols
            missing = reference_cols - cols
            mismatches.append(
                f"{name}: extra={extra or 'none'}, missing={missing or 'none'}"
            )

    if not mismatches:
        return ValidationResult(
            "column_consistency",
            True,
            f"All files share {len(reference_cols)} columns",
        )

    return ValidationResult(
        "column_consistency", False, "; ".join(mismatches)
    )


def validate_observable_ranges(
    df: pd.DataFrame,
    expected_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> ValidationResult:
    """Check that observables fall within physically sensible ranges.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to check.
    expected_ranges : dict, optional
        Mapping of column name -> (min, max). If None, checks for
        NaN/Inf only.
    """
    classified = classify_columns(df.columns.tolist())
    obs_cols = classified["observables"]

    for col in obs_cols:
        if not np.isfinite(df[col]).all():
            n_bad = (~np.isfinite(df[col])).sum()
            return ValidationResult(
                "observable_ranges",
                False,
                f"Observable '{col}' has {n_bad} non-finite values",
            )

    if expected_ranges:
        for col, (lo, hi) in expected_ranges.items():
            if col in df.columns:
                if (df[col] < lo).any() or (df[col] > hi).any():
                    return ValidationResult(
                        "observable_ranges",
                        False,
                        f"Observable '{col}' outside [{lo}, {hi}]",
                    )

    return ValidationResult(
        "observable_ranges",
        True,
        f"All {len(obs_cols)} observables have finite, valid values",
    )


def validate_weight_normalization(
    df: pd.DataFrame,
    weight_col: str,
    expected_sum: Optional[float] = None,
    tolerance: float = 0.05,
) -> ValidationResult:
    """Check that a weight column sums to an expected value.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing the weight column.
    weight_col : str
        Name of the weight column to check.
    expected_sum : float, optional
        Expected sum of weights. If None, just reports the actual sum.
    tolerance : float
        Fractional tolerance for the comparison.
    """
    if weight_col not in df.columns:
        return ValidationResult(
            "weight_normalization",
            False,
            f"Column '{weight_col}' not found",
        )

    actual_sum = df[weight_col].sum()

    if expected_sum is None:
        return ValidationResult(
            "weight_normalization",
            True,
            f"Sum of '{weight_col}' = {actual_sum:.4f} (no expected value given)",
        )

    rel_diff = abs(actual_sum - expected_sum) / expected_sum
    passed = rel_diff <= tolerance

    return ValidationResult(
        "weight_normalization",
        passed,
        f"Sum of '{weight_col}' = {actual_sum:.4f}, "
        f"expected {expected_sum:.4f} (rel. diff = {rel_diff:.4f})",
    )


def run_all_validations(
    dataframes: Dict[str, pd.DataFrame],
) -> List[ValidationResult]:
    """Run all validation checks on a set of DataFrames.

    Parameters
    ----------
    dataframes : dict
        Mapping of file name to DataFrame.

    Returns
    -------
    list of ValidationResult
    """
    results = []

    # Cross-file checks
    results.append(validate_event_count_consistency(dataframes))
    results.append(validate_column_consistency(dataframes))

    # Per-file checks
    for name, df in dataframes.items():
        results.append(validate_weights_positive(df))
        results.append(validate_weights_finite(df))
        results.append(validate_observable_ranges(df))

    return results
