"""Weighted histogram computation and visualization for OmniFold observables.

Part 3 of the GSoC 2026 evaluation task.

This module provides a single self-contained function that computes a weighted
histogram of an observable and optionally plots it. The design prioritizes:

- Correctness: proper handling of weighted statistics and edge cases
- Flexibility: sensible defaults with full user control over binning
- Reproducibility: deterministic output for the same inputs
"""

from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


def weighted_histogram(
    values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bins: Union[int, np.ndarray] = 50,
    range: Optional[Tuple[float, float]] = None,
    normalize: bool = False,
    density: bool = False,
    xlabel: str = "Observable",
    ylabel: Optional[str] = None,
    title: str = "",
    plot: bool = True,
    save_path: Optional[str] = None,
    comparison_weights: Optional[np.ndarray] = None,
    comparison_label: str = "Systematic variation",
    label: str = "Nominal",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute and optionally plot a weighted histogram of an observable.

    Computes bin counts (or densities) using event weights, calculates
    statistical uncertainties via sqrt(sum(w^2)) per bin, and produces
    a publication-style plot.

    Parameters
    ----------
    values : np.ndarray
        1-D array of observable values (one per event).
    weights : np.ndarray, optional
        1-D array of per-event weights. If None, uniform weights of 1.0
        are used.
    bins : int or np.ndarray
        Number of bins (int) or explicit bin edges (array).
    range : tuple of (float, float), optional
        Range for the histogram. Ignored if bins is an array.
    normalize : bool
        If True, normalize the histogram so the total sum of bin contents
        equals 1. Mutually exclusive with density.
    density : bool
        If True, normalize to form a probability density (area = 1).
        Mutually exclusive with normalize.
    xlabel : str
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis. Auto-generated if None.
    title : str
        Plot title.
    plot : bool
        If True, produce a matplotlib figure.
    save_path : str, optional
        If provided, save the figure to this path.
    comparison_weights : np.ndarray, optional
        A second set of weights for overlay comparison (e.g. systematic).
    comparison_label : str
        Legend label for the comparison histogram.
    label : str
        Legend label for the nominal histogram.

    Returns
    -------
    bin_contents : np.ndarray
        Weighted bin counts (or densities).
    bin_edges : np.ndarray
        Bin edge positions (length = len(bin_contents) + 1).
    bin_errors : np.ndarray
        Statistical uncertainty per bin: sqrt(sum(w_i^2)) in each bin.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths, contain no valid data, or
        if both normalize and density are True.

    Examples
    --------
    >>> import numpy as np
    >>> values = np.random.normal(91.2, 2.5, 10000)  # Z boson mass peak
    >>> weights = np.random.exponential(1.0, 10000)
    >>> contents, edges, errors = weighted_histogram(
    ...     values, weights, bins=40, xlabel="m_{ll} [GeV]"
    ... )
    """
    # --- Input validation ---
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {values.shape}")

    if len(values) == 0:
        raise ValueError("values array is empty")

    if weights is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    if values.shape != weights.shape:
        raise ValueError(
            f"Shape mismatch: values {values.shape} vs weights {weights.shape}"
        )

    if normalize and density:
        raise ValueError("Cannot set both normalize=True and density=True")

    # Filter out NaN/Inf values in both arrays (events where either is invalid)
    valid_mask = np.isfinite(values) & np.isfinite(weights)
    if not valid_mask.any():
        raise ValueError("No valid (finite) entries in values/weights")

    values_clean = values[valid_mask]
    weights_clean = weights[valid_mask]

    # --- Compute histogram ---
    bin_contents, bin_edges = np.histogram(
        values_clean, bins=bins, range=range, weights=weights_clean
    )

    # Uncertainty: sqrt(sum(w_i^2)) per bin — the standard error for
    # weighted counts, which reduces to sqrt(N) for uniform weights.
    bin_errors_sq, _ = np.histogram(
        values_clean, bins=bin_edges, weights=weights_clean ** 2
    )
    bin_errors = np.sqrt(bin_errors_sq)

    # --- Normalization ---
    if normalize:
        total = bin_contents.sum()
        if total > 0:
            bin_contents = bin_contents / total
            bin_errors = bin_errors / total

    if density:
        bin_widths = np.diff(bin_edges)
        total_area = (bin_contents * bin_widths).sum()
        if total_area > 0:
            bin_contents = bin_contents / (total_area * bin_widths / bin_widths)
            # Proper density normalization
            norm_factor = bin_contents.sum() * bin_widths
            bin_contents_density = bin_contents / bin_widths
            bin_errors_density = bin_errors / bin_widths
            # Re-normalize so integral = 1
            integral = (bin_contents_density * bin_widths).sum()
            if integral > 0:
                bin_contents = bin_contents_density / integral
                bin_errors = bin_errors_density / integral
            else:
                bin_contents = bin_contents_density
                bin_errors = bin_errors_density

    # --- Plotting ---
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = np.diff(bin_edges)

        # Nominal histogram as filled step plot with error bars
        ax.bar(
            bin_centers,
            bin_contents,
            width=bin_widths,
            alpha=0.35,
            color="steelblue",
            edgecolor="steelblue",
            linewidth=0.8,
            label=label,
        )
        ax.errorbar(
            bin_centers,
            bin_contents,
            yerr=bin_errors,
            fmt="none",
            ecolor="steelblue",
            elinewidth=1.2,
            capsize=2,
        )

        # Optional comparison overlay (e.g. systematic variation)
        if comparison_weights is not None:
            comp_weights = np.asarray(comparison_weights, dtype=np.float64)
            if comp_weights.shape != values.shape:
                raise ValueError(
                    f"comparison_weights shape {comp_weights.shape} "
                    f"does not match values shape {values.shape}"
                )
            comp_clean = comp_weights[valid_mask]
            comp_contents, _ = np.histogram(
                values_clean, bins=bin_edges, weights=comp_clean
            )
            comp_errsq, _ = np.histogram(
                values_clean, bins=bin_edges, weights=comp_clean ** 2
            )
            comp_errors = np.sqrt(comp_errsq)

            if normalize:
                comp_total = comp_contents.sum()
                if comp_total > 0:
                    comp_contents = comp_contents / comp_total
                    comp_errors = comp_errors / comp_total

            ax.step(
                bin_edges[:-1],
                comp_contents,
                where="post",
                color="darkorange",
                linewidth=1.5,
                label=comparison_label,
            )
            ax.errorbar(
                bin_centers,
                comp_contents,
                yerr=comp_errors,
                fmt="none",
                ecolor="darkorange",
                elinewidth=1.0,
                capsize=2,
                alpha=0.7,
            )

        ax.set_xlabel(xlabel, fontsize=13)
        if ylabel is None:
            if density:
                ylabel = "Probability density"
            elif normalize:
                ylabel = "Normalized events"
            else:
                ylabel = "Weighted events"
        ax.set_ylabel(ylabel, fontsize=13)

        if title:
            ax.set_title(title, fontsize=14)

        ax.legend(frameon=False, fontsize=11)
        ax.tick_params(axis="both", which="major", labelsize=11)

        # Clean up spines for a publication-style look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    return bin_contents, bin_edges, bin_errors
