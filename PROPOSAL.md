# GSoC 2026 Proposal: Publication of OmniFold Weights

**Organization:** CERN-HSF / Stanford University
**Project:** Publication of OmniFold Weights
**Mentors:** Tanvi Wamorkar (tanvi.wamorkar@cern.ch), Benjamin Nachman (nachman@stanford.edu)
**Project Size:** 175 hours
**Difficulty:** Medium

---

## Contact Information

**Name:** Sridhar Panigrahi
**Email:** sridharpanigrahi2006@gmail.com
**GitHub:** [sridhar-panigrahi](https://github.com/sridhar-panigrahi)
**LinkedIn:** [shridhar-panigrahi](https://www.linkedin.com/in/shridhar-panigrahi-8bb176362)
**University:** Polaris School of Technology, Bangalore
**Program:** B.Tech CSE (Specialization in AI/ML), 1st Year
**Timezone:** IST (UTC+5:30)

---

## Synopsis

OmniFold produces per-event weights instead of fixed histograms, which makes it fundamentally different from traditional unfolding methods. But right now, there is no standard way to publish, preserve, or reuse those weights. The output is a raw numpy array that disappears when the notebook closes.

This project builds the tooling to change that. I will implement a metadata schema to describe what the weights represent, container formats (HDF5/Parquet) to store them, a Python API to publish and load results, validation procedures to verify correctness, and integration with HEPData so that OmniFold outputs can be archived alongside traditional measurements.

I have already started this work. I restructured the OmniFold codebase into a proper Python package and implemented the first layer of the publication framework (schema, weight container, publish/load API, 38 tests) as a [draft PR](https://github.com/hep-lbdl/OmniFold/pull/13). I also completed the mentor evaluation task, analyzing real ATLAS Z+jets HDF5 files and building a prototype framework around them ([omnifold-hepdata-evaluation](https://github.com/sridhar-panigrahi/omnifold-hepdata-evaluation)).

---

## Understanding the Problem

### What OmniFold does

OmniFold is an iterative reweighting algorithm for unfolding particle physics data. It trains binary classifiers to learn the mapping between detector-level observations and particle-level truth, producing per-event weights that correct for detector effects. The core loop alternates between two steps across multiple iterations:

```python
for i in range(iterations):
    # Step 1 (Pull): Train classifier on detector-level sim vs. data
    # Extracts weights that pull simulation toward data
    weights_pull = weights_push * reweight(theta0_S, model)

    # Step 2 (Push): Train classifier on gen-level vs. reweighted gen-level
    # Extracts weights that push generator toward corrected distribution
    weights_push = reweight(theta0_G, model)
```

The output is a `(iterations, 2, n_events)` numpy array where axis 0 is the iteration, axis 1 is the step (pull/push), and axis 2 is the event index. The final-iteration push weights are the main unfolding result.

### Why publication is hard

Traditional unfolding produces binned histograms that fit naturally into HEPData's YAML table format. OmniFold produces something different:

- **Per-event weights** for potentially millions of events, not binned tables.
- **Multiple iterations** where convergence matters and intermediate weights carry diagnostic value.
- **Statistical replicas and systematic variations** that multiply the storage requirement (the ATLAS Z+jets dataset I analyzed has 100 ensemble replicas, 50 bootstrap replicas, and 26 systematic variations per event, totaling 200 columns).
- **Trained models** whose architecture and preprocessing define how the weights were derived.
- **Observable definitions** that exist in analysis code but are not captured alongside the weights.

Without a standard format, every analysis team invents their own ad-hoc solution. Results cannot be compared, reproduced, or reinterpreted by others.

### What I learned from the ATLAS Z+jets data

As part of the evaluation task, I inspected three HDF5 files from an ATLAS OmniFold measurement on Zenodo. The gap analysis I performed ([gap_analysis.md](https://github.com/sridhar-panigrahi/omnifold-hepdata-evaluation/blob/main/gap_analysis.md)) found:

- **418,014 events across 200 columns** in the nominal file, with zero metadata about what any column represents.
- Weight columns follow naming conventions (`weight_ensemble_0`, `weight_bootstrap_0`, `weight_sys_JET_JER_up`) but there is no machine-readable schema documenting these conventions.
- Three HDF5 files (`multifold.h5`, `multifold_sherpa.h5`, `multifold_nonDY.h5`) have different column counts (200, 51, 26) with no documentation of their relationships.
- No fiducial region definitions, no jet algorithm specifications, no normalization information.

This is exactly the problem the project aims to solve.

---

## Prior Contributions to OmniFold

### PR #10: Pip Packaging and CI/Publish Pipelines

**Link:** [hep-lbdl/OmniFold#10](https://github.com/hep-lbdl/OmniFold/pull/10)
**Status:** Open (resolves [Issue #7](https://github.com/hep-lbdl/OmniFold/issues/7) created by mentor Tanvi Wamorkar)

Before any publication framework could be built, OmniFold needed to be an installable Python package. Issue #7 asked for a pip package with a CI pipeline. I implemented:

- **`pyproject.toml`** with PEP 621 metadata, dynamic versioning from `omnifold.__version__`, and dependency declarations (numpy, scikit-learn, tensorflow>=2.0).
- **CI workflow** (`ci.yml`) testing the build across Python 3.9-3.12.
- **Publish workflow** (`publish.yml`) using OIDC-based PyPI authentication triggered on GitHub releases.
- **`MANIFEST.in`** to exclude notebooks, benchmarks, and images from the distributed package.

This matters because a publication framework is useless if users cannot `pip install omnifold` and get it. The packaging work is the foundation that everything else builds on.

### PR #13: Weight Publication Framework (Proof of Concept)

**Link:** [hep-lbdl/OmniFold#13](https://github.com/hep-lbdl/OmniFold/pull/13)
**Status:** Draft (+1002 lines, 38 tests passing)

This is the core proof of concept for the GSoC project. I restructured the codebase from a single `omnifold.py` file into a proper Python package and built the first layer of the publication framework:

**Package restructure** (backward-compatible):
```
omnifold/
    __init__.py       # Re-exports, TF-optional imports
    _version.py       # Single source of truth for version
    core.py           # Original algorithm (untouched)
    schema.py         # Metadata schema
    weights.py        # Weight container with HDF5/Parquet I/O
    publication.py    # publish() and load() API
```

Old import patterns still work. `import omnifold; omnifold.omnifold(...)` behaves exactly as before. TensorFlow is only required if you run the algorithm; users who only load published weights do not need it.

**Metadata schema** (`schema.py`):
```python
@dataclass
class OmniFoldMetadata:
    schema_version: str = "1.0"
    omnifold_version: str = ""
    iterations: int = 0
    model: ModelInfo = field(default_factory=ModelInfo)
    dataset: DatasetInfo = field(default_factory=DatasetInfo)
    weight_format: str = "hdf5"
    normalization: str = "unit"
    created_at: str = ""
    description: str = ""
```

Built-in JSON serialization with round-trip fidelity, validation that catches bad configurations before writing files, and forward compatibility (unknown fields from newer schema versions are silently ignored on load).

**Weight container** (`weights.py`):
```python
class OmniFoldWeights:
    def nominal(self):
        """Final-iteration push weights."""
        return self._weights[-1, 1, :]

    def get_weights(self, iteration=-1, step="push"):
        step_idx = {"pull": 0, "push": 1}[step]
        return self._weights[iteration, step_idx, :]

    def to_hdf5(self, path): ...
    def to_parquet(self, path): ...

    @classmethod
    def from_hdf5(cls, path): ...
    @classmethod
    def from_parquet(cls, path): ...
```

HDF5 files store both structured per-iteration groups (for human inspection with `h5dump`) and a bulk array (for fast programmatic loading). Parquet files use one column per step/iteration with metadata embedded in the schema.

**Publish/Load API** (`publication.py`):
```python
# Publishing
of.publish(weights, "my_result/", metadata=meta, weight_format="hdf5")
# Creates: my_result/metadata.json + my_result/weights.h5

# Loading (auto-detects format)
result = of.load("my_result/")
w = result.nominal()
```

**Test suite:** 38 tests covering schema validation, JSON round-trips, HDF5/Parquet round-trips, auto-detection, error handling, and forward compatibility. All passing on Python 3.10 and 3.12.

---

## Related Projects I Built

### dataspec

**Link:** [sridhar-panigrahi/dataspec](https://github.com/sridhar-panigrahi/dataspec)

A Python framework for defining, validating, and documenting scientific datasets using YAML schemas. I built this because the OmniFold publication problem is fundamentally a dataset specification problem.

Key pieces that transfer directly:

- **YAML schema definitions** with column-level type/unit/range/constraint specifications, the same kind of per-observable metadata that the GSoC project requires.
- **Multi-format data loading** (CSV, HDF5, Parquet, NumPy) with auto-detection from file extension, directly relevant to loading OmniFold weight files.
- **Non-exception validation** returning structured `ValidationResult` objects so all issues are surfaced at once, the same pattern needed for closure tests and normalization checks.
- **Auto-generated dataset cards** from schemas, analogous to auto-generating HEPData submission READMEs.
- 35 tests across 4 test files.

```python
# Example: validating a particle physics dataset against its schema
schema = DatasetSchema.from_yaml("z_jets_schema.yaml")
result = SchemaValidator(schema).validate(dataframe)
if not result.is_valid:
    for issue in result.issues:
        print(f"{issue.column}: {issue.message}")
```

### hepdata-tools

**Link:** [sridhar-panigrahi/hepdata-tools](https://github.com/sridhar-panigrahi/hepdata-tools)

A Python toolkit for fetching, validating, and visualizing HEPData submissions. I built this to understand HEPData's data model from the inside, since the GSoC stretch goal involves mapping OmniFold outputs to HEPData records.

- **REST API client** with in-memory caching for the HEPData API.
- **Typed data models** (`Submission`, `Table`, `Variable`, `Uncertainty`) using dataclasses, supporting both symmetric and asymmetric errors with quadrature combination.
- **Submission validation** checking bin consistency, finite values, and uncertainty structure.
- **Publication-quality plotting** with error bars and overlays.
- **CSV/JSON export** with numpy type conversion.

Working through HEPData's JSON response format taught me exactly how their submission schema is organized, which tables map to which observables, and how uncertainties are represented. This is directly useful for designing the OmniFold-to-HEPData mapping.

### omnifold-hepdata-evaluation

**Link:** [sridhar-panigrahi/omnifold-hepdata-evaluation](https://github.com/sridhar-panigrahi/omnifold-hepdata-evaluation)

The mentor evaluation task. Three deliverables:

1. **Gap analysis** of real ATLAS Z+jets HDF5 files: classified all 200 columns, identified 24 particle-level observables, 176 weight columns (100 ensemble + 50 bootstrap + 26 systematics), and documented every missing piece of metadata.

2. **Metadata schema** (`metadata.yaml`): YAML schema covering observables (with units, ranges), weight semantics, systematic variation naming, normalization conventions, fiducial region cuts, jet definitions, and analysis provenance.

3. **Weighted histogram function** with uncertainty propagation (`sqrt(sum(w^2))`), NaN/Inf filtering, normalize/density modes, and comparison overlays. 14 pytest tests covering edge cases.

Plus a framework prototype with `OmniFoldDataset` API, HDF5 loading, column classification by naming convention, multi-format export, and dataset validation.

---

## Contributions to Other Open-Source Projects

### astropy (merged)

**PR [#19403](https://github.com/astropy/astropy/pull/19403):** Fixed a missing `f`-string prefix in `BaseAffineTransform._apply_transform` that caused error messages to show literal `{data.__class__}` instead of the actual class name. Backported to v7.2.x. Reviewed by core maintainers nstarman and pllim.

**PR [#19404](https://github.com/astropy/astropy/pull/19404):** Fixed silent data corruption in `FITS_rec` slice assignment with negative indices. `data[-2:] = new_rows` was writing to wrong rows because `max(0, key.start or 0)` clamps negative indices to 0. Replaced manual arithmetic with `slice.indices(len(self))`.

These show I can read large, mature codebases (astropy is 500k+ lines), identify root causes, and write fixes that pass extensive test suites (1029 FITS tests).

### Hyperledger Besu (merged)

**PR [#10020](https://github.com/besu-eth/besu/pull/10020):** Fixed a critical bug where `serializeAndDedupOperation()` in the transaction pool created a `failedFuture` with a `TimeoutException` but never returned it. The code silently fell through to `completedFuture(null)`, potentially causing all pending transactions to be lost during sync state transitions. One missing `return` keyword. Reviewed and approved by core maintainer fab-10.

**PR [#10042](https://github.com/besu-eth/besu/pull/10042):** Fixed `eth_estimateGas` using the parent block's hash instead of the current block's hash for balance lookup, causing false `TRANSACTION_UPFRONT_COST_EXCEEDS_BALANCE` errors.

### LFDT-Lockness generic-ec (Rust)

**PR [#59](https://github.com/LFDT-Lockness/generic-ec/pull/59):** Added secp384r1 (NIST P-384) curve support, the first curve with a non-32-byte scalar size. Implemented `FromUniformBytes` per RFC 9380, scalar trait implementations, and instantiated all tests (scalar ops, point ops, serde, Straus multiscalar multiplication, ZK proofs). 180 tests passing.

**PR [#57](https://github.com/LFDT-Lockness/generic-ec/pull/57):** Fixed hardcoded 32-byte scalar size in Straus NAF computation that caused panics for curves with different scalar sizes. Prerequisite for P-384 support.

---

## Proposed Implementation Plan

### Phase 1: Foundation (Weeks 1-3, ~40 hours)

Most of this is already done in [PR #13](https://github.com/hep-lbdl/OmniFold/pull/13). During the first weeks I will incorporate mentor feedback and finalize:

**1.1 Finalize package restructure and schema**
- Address any review comments on the package structure.
- Extend `OmniFoldMetadata` based on what I learned from the ATLAS Z+jets gap analysis: add fields for fiducial region, jet definitions, analysis references (arXiv, INSPIRE, HEPData IDs).
- Add YAML serialization alongside JSON (using `pyyaml` as optional dependency) since YAML is the standard in HEP tooling.

**1.2 Extend weight container for real-world use**
- Add support for statistical replicas and systematic variations as additional weight columns in the container:

```python
class OmniFoldWeights:
    def add_replica(self, name, weights_array):
        """Add a statistical replica or systematic variation."""

    def replicas(self):
        """Iterator over all replica weight sets."""

    def systematic(self, name):
        """Get weights for a specific systematic variation."""
```

- Handle the naming conventions I observed in the ATLAS data: `weight_ensemble_{i}`, `weight_bootstrap_{i}`, `weight_sys_{name}_{direction}`.

**1.3 Model and training detail capture**
- Store model architecture summary, optimizer config, epoch counts, and loss history alongside weights.
- Support both "minimal" publication (weights + metadata only) and "full" publication (weights + metadata + model checkpoint).

**Deliverable:** Finalized, reviewed, merged version of the schema, weight container, and publish/load API.

---

### Phase 2: Observable Definitions and Reinterpretation API (Weeks 4-6, ~45 hours)

**2.1 Observable definition format**

```python
@dataclass
class Observable:
    name: str                    # e.g., "jet_pt"
    expression: str              # Human-readable definition
    compute_fn: Callable = None  # Python function to evaluate
    units: str = ""              # e.g., "GeV"
    binning: list = None         # Bin edges
    phase_space: dict = None     # Selection cuts

    def histogram(self, events, weights=None):
        """Compute weighted histogram with uncertainties."""
        values = self.compute_fn(events)
        counts, edges = np.histogram(values, bins=self.binning, weights=weights)
        errors = np.sqrt(np.histogram(values, bins=self.binning,
                                       weights=weights**2)[0])
        return counts, errors, edges
```

- Serialize observable definitions to YAML/JSON so they travel with the published result.
- Support both string expressions (for documentation) and Python callables (for computation).

**2.2 Reinterpretation API**

```python
# Load a published result
result = omnifold.load("atlas_zjets_published/")

# Access the nominal unfolding weights
weights = result.nominal()

# Compute an observable that was not in the original analysis
jet_mass = Observable(
    name="jet_mass",
    compute_fn=lambda events: events["jet_m"],
    units="GeV",
    binning=[0, 20, 40, 60, 80, 100, 150, 200],
)
counts, errors, edges = jet_mass.histogram(events, weights=weights)
```

**2.3 Uncertainty propagation**
- Compute uncertainty bands from statistical replicas: run each replica's weights through the observable computation, take the standard deviation across replicas.
- Systematic uncertainties: compute up/down variations, report as asymmetric errors.

```python
def uncertainty_band(result, observable, events):
    nominal = observable.histogram(events, result.nominal())[0]
    replica_hists = [observable.histogram(events, r)[0]
                     for r in result.replicas()]
    stat_error = np.std(replica_hists, axis=0)
    return nominal, stat_error
```

**Deliverable:** Observable definition format, reinterpretation API, and uncertainty propagation. Tested against the Gaussian example and (if data access permits) the ATLAS Z+jets files.

---

### Phase 3: Validation Framework (Weeks 7-9, ~40 hours)

**3.1 Closure tests**

```python
def closure_test(theta0, model, iterations, observable):
    """Run OmniFold on MC-as-data and verify truth recovery.

    Uses one half of MC as 'data' and the other half as simulation.
    The unfolded result should recover the known particle-level truth.
    Returns chi2/ndf and per-bin pull distribution.
    """
```

**3.2 Normalization and convergence checks**

```python
def normalization_check(weights, tolerance=0.01):
    """Verify weighted event count matches expected normalization."""

def iteration_stability(weights, window=2):
    """Check that weights converge: compare last `window` iterations.
    Returns max relative change and per-event stability flag.
    """
```

**3.3 Validation report**

```python
def validate(result, events=None):
    """Run all standard checks and return a structured report.

    Checks: weights are finite, weights are positive,
    normalization is consistent, iterations have converged.
    If events are provided: observable distributions are physical.
    """
    return ValidationReport(checks=[...], passed=True/False)
```

**3.4 Automated tests**
- Expand the test suite to cover validation functions.
- Add a CI integration test that runs the full Gaussian example end-to-end: generate data, unfold, publish, load, validate.
- Target: >80% code coverage on the publication layer.

**Deliverable:** Validation framework with closure tests, convergence checks, and normalization verification. Validation integrated into the publish workflow (optional but recommended).

---

### Phase 4: HEPData Integration and Examples (Weeks 10-12, ~50 hours)

**4.1 HEPData submission generation**

```python
def to_hepdata(result, submission_dir, observables=None):
    """Generate HEPData-compatible YAML submission.

    Maps OmniFold outputs to HEPData conventions:
    - Observable histograms -> YAML data tables
    - Per-event weights -> auxiliary resource files (HDF5)
    - Metadata -> submission.yaml qualifiers
    """
```

- Use `hepdata_lib` for generating submission YAML files.
- Store per-event weights as auxiliary resource files (HEPData supports arbitrary file attachments alongside YAML tables).
- Generate submission templates that pass `hepdata-validator`.

**4.2 Reference examples**

Three complete Jupyter notebooks:

1. **Gaussian example end-to-end:** Generate data, run OmniFold, publish weights, load them back, compute observables, validate, and plot. This extends the existing `GaussianExample.ipynb`.

2. **Reinterpretation workflow:** Load a published result and compute observables that were not in the original analysis. Shows the value of per-event weight publication.

3. **HEPData submission:** Take a published OmniFold result and generate a complete HEPData submission directory. Walk through the generated files.

**4.3 Documentation**
- Docstrings on all public API functions and classes.
- Schema specification document describing all fields, their semantics, and when each is required vs. optional.
- Quick-start section in README showing the publish/load workflow.

**Deliverable:** HEPData integration (stretch goal), three reference notebooks, complete documentation.

---

## Timeline Summary

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1-3 | Foundation | Finalized schema, weight container with replicas/systematics, model capture |
| 4-6 | Observables & API | Observable definitions, reinterpretation API, uncertainty propagation |
| 7-9 | Validation | Closure tests, convergence checks, CI integration tests |
| 10-12 | HEPData & Examples | HEPData integration, 3 reference notebooks, documentation |

I am available 15-20 hours/week from June through October. I have no other commitments during this period.

---

## Relevant Skills

**Python and scientific computing:** All three of my standalone projects (dataspec, hepdata-tools, omnifold-hepdata-evaluation) are Python, using numpy, h5py, pandas, matplotlib, and pytest. I write dataclass-based schemas, multi-format I/O, and validation frameworks because that is exactly what this project requires.

**HDF5 and Parquet:** I have hands-on experience with both formats. In the evaluation task, I inspected real ATLAS HDF5 files with 200 columns and 418k events. In the PoC PR, I implemented HDF5 and Parquet round-trip serialization with metadata embedding.

**HEPData ecosystem:** I built hepdata-tools specifically to understand HEPData's data model, REST API, submission format, and uncertainty conventions. I know how `hepdata_lib` works and how submissions are structured.

**Testing and CI:** Every project I have worked on includes tests. The PoC PR has 38 tests. The evaluation task has 14 tests. dataspec has 35 tests. I set up GitHub Actions CI for OmniFold (PR #10) and understand multi-version matrix testing.

**Working in large codebases:** Merged PRs in astropy (500k+ lines, Python) and Besu (1M+ lines, Java). I can navigate unfamiliar code, identify root causes, write targeted fixes, and pass existing test suites.

**Machine learning:** B.Tech CSE with AI/ML specialization. I understand the binary classification approach that OmniFold uses, the iterative reweighting mechanism, and how the pull/push steps relate to the physics of detector unfolding.

---

## Why Me

I am not coming in cold. I have already:

1. **Read and understood the OmniFold algorithm** -- the core loop, the weighted binary crossentropy trick, the pull/push weight semantics, and how the `(iterations, 2, n_events)` output array is structured.

2. **Solved a real infrastructure need** -- PR #10 addresses Issue #7 (raised by mentor Tanvi) to make OmniFold pip-installable. This is prerequisite work that needed to happen regardless of the GSoC project.

3. **Built a working proof of concept** -- PR #13 is not a toy. It is 1002 lines of tested, backward-compatible code that implements the core deliverable of the project (standardized weight publication).

4. **Analyzed real OmniFold data** -- The evaluation task had me inspect actual ATLAS Z+jets HDF5 files. I know what the data looks like in practice, not just in theory. The gap analysis I wrote directly informs the schema design.

5. **Built adjacent tooling** -- dataspec and hepdata-tools are not filler projects. They are working implementations of the exact patterns (schema validation, HEPData integration, multi-format I/O) that this GSoC project needs.

I am a first-year student. I will not pretend to have years of physics experience. But the work speaks for itself: I have shipped code to this repository, I understand the problem from both the algorithm side and the data side, and I have built the tools to prove it.
