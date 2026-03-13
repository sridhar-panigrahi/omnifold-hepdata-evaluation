"""Tests for the weighted_histogram function.

Edge cases chosen and rationale:
1. Uniform weights = 1 should reproduce np.histogram exactly.
   (Baseline correctness — if this fails, nothing else matters.)

2. Known analytic case: weights that double a half-sample should shift
   the mean. (Verifies that weights actually affect the result.)

3. Empty input must raise a clear error.
   (Protects downstream code from silent NaN propagation.)

4. NaN/Inf in values or weights should be filtered, not crash.
   (Real HDF5 data often has occasional bad entries.)

5. Mismatched array lengths must raise immediately.
   (Shape mismatches are the most common integration bug.)

6. Single-event input should still produce a valid histogram.
   (Boundary case that breaks many binning implementations.)

7. All-zero weights should produce an all-zero histogram, not NaN.
   (Degenerate but possible in reweighting; silent NaN is dangerous.)

8. normalize=True should make bin contents sum to 1.
   (Contract check for the normalization feature.)
"""

import numpy as np
import pytest

from weighted_histogram import weighted_histogram


# --- Helpers ---

def _no_plot(**kwargs):
    """Override defaults to suppress plotting during tests."""
    kwargs.setdefault("plot", False)
    return kwargs


# --- Test cases ---

class TestBasicCorrectness:
    """Verify that the function produces mathematically correct results."""

    def test_uniform_weights_match_numpy(self):
        """With weights=1, output must match np.histogram exactly."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 5000)

        contents, edges, errors = weighted_histogram(
            values, bins=30, range=(-4, 4), **_no_plot()
        )
        expected, _ = np.histogram(values, bins=30, range=(-4, 4))

        np.testing.assert_array_almost_equal(contents, expected)
        # For uniform weights, uncertainty = sqrt(N)
        np.testing.assert_array_almost_equal(errors, np.sqrt(expected))

    def test_weights_affect_result(self):
        """Doubling weights on the right half should shift the mean."""
        values = np.linspace(-5, 5, 1000)
        uniform_weights = np.ones(1000)
        biased_weights = np.where(values > 0, 2.0, 1.0)

        contents_uniform, edges, _ = weighted_histogram(
            values, uniform_weights, bins=20, range=(-5, 5), **_no_plot()
        )
        contents_biased, _, _ = weighted_histogram(
            values, biased_weights, bins=20, range=(-5, 5), **_no_plot()
        )

        # Weighted mean should be higher for biased weights
        centers = 0.5 * (edges[:-1] + edges[1:])
        mean_uniform = np.average(centers, weights=contents_uniform)
        mean_biased = np.average(centers, weights=contents_biased)
        assert mean_biased > mean_uniform

    def test_error_computation(self):
        """Errors should be sqrt(sum(w_i^2)) per bin."""
        # Place 4 events in one bin with weights [1, 2, 3, 4]
        values = np.array([0.5, 0.5, 0.5, 0.5])
        weights = np.array([1.0, 2.0, 3.0, 4.0])

        contents, edges, errors = weighted_histogram(
            values, weights, bins=1, range=(0, 1), **_no_plot()
        )

        assert contents[0] == pytest.approx(10.0)  # sum of weights
        expected_error = np.sqrt(1**2 + 2**2 + 3**2 + 4**2)
        assert errors[0] == pytest.approx(expected_error)


class TestEdgeCases:
    """Verify correct handling of degenerate and boundary inputs."""

    def test_empty_array_raises(self):
        """Empty input must raise ValueError, not return NaN."""
        with pytest.raises(ValueError, match="empty"):
            weighted_histogram(np.array([]), **_no_plot())

    def test_nan_values_filtered(self):
        """NaN entries in values should be silently excluded."""
        values = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        contents, _, _ = weighted_histogram(
            values, weights, bins=4, range=(0, 5), **_no_plot()
        )
        # Only 4 valid events contribute
        assert contents.sum() == pytest.approx(4.0)

    def test_inf_weights_filtered(self):
        """Inf in weights should be excluded, not corrupt the result."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.array([1.0, np.inf, 1.0])

        contents, _, _ = weighted_histogram(
            values, weights, bins=3, range=(0, 4), **_no_plot()
        )
        assert np.isfinite(contents).all()
        assert contents.sum() == pytest.approx(2.0)

    def test_mismatched_lengths_raises(self):
        """Different-length values and weights must raise immediately."""
        with pytest.raises(ValueError, match="mismatch"):
            weighted_histogram(
                np.array([1.0, 2.0]),
                np.array([1.0]),
                **_no_plot(),
            )

    def test_single_event(self):
        """A single event should produce a valid one-entry histogram."""
        contents, edges, errors = weighted_histogram(
            np.array([5.0]),
            np.array([3.0]),
            bins=1,
            range=(4, 6),
            **_no_plot(),
        )
        assert contents[0] == pytest.approx(3.0)
        assert errors[0] == pytest.approx(3.0)
        assert len(edges) == 2

    def test_all_zero_weights(self):
        """All-zero weights should give all-zero bins, not NaN."""
        values = np.array([1.0, 2.0, 3.0])
        weights = np.zeros(3)

        contents, _, errors = weighted_histogram(
            values, weights, bins=3, range=(0, 4), **_no_plot()
        )
        assert (contents == 0).all()
        assert (errors == 0).all()
        assert np.isfinite(contents).all()


class TestNormalization:
    """Verify normalization modes."""

    def test_normalize_sums_to_one(self):
        """normalize=True must make bin contents sum to 1."""
        rng = np.random.default_rng(99)
        values = rng.normal(0, 1, 2000)
        weights = rng.exponential(1.0, 2000)

        contents, _, _ = weighted_histogram(
            values, weights, bins=25, normalize=True, **_no_plot()
        )
        assert contents.sum() == pytest.approx(1.0, abs=1e-10)

    def test_normalize_and_density_exclusive(self):
        """Setting both normalize and density must raise."""
        with pytest.raises(ValueError, match="Cannot set both"):
            weighted_histogram(
                np.array([1.0, 2.0]),
                normalize=True,
                density=True,
                **_no_plot(),
            )


class TestInputTypes:
    """Verify that the function handles various input types gracefully."""

    def test_list_input(self):
        """Python lists should be accepted (converted to ndarray)."""
        contents, _, _ = weighted_histogram(
            [1.0, 2.0, 3.0],
            [1.0, 1.0, 1.0],
            bins=3,
            range=(0, 4),
            **_no_plot(),
        )
        assert contents.sum() == pytest.approx(3.0)

    def test_integer_input(self):
        """Integer arrays should work without type errors."""
        values = np.array([1, 2, 3, 4, 5])
        contents, _, _ = weighted_histogram(
            values, bins=5, range=(0, 6), **_no_plot()
        )
        assert contents.sum() == pytest.approx(5.0)

    def test_2d_input_raises(self):
        """2-D input must be rejected."""
        with pytest.raises(ValueError, match="1-D"):
            weighted_histogram(np.ones((3, 2)), **_no_plot())
