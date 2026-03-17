# OmniFold Publication Tools — Evaluation Task

**Author:** Shridhar Panigrahi
**Context:** GSoC 2026 evaluation — OmniFold Publication Tools (CERN / HEP Software Foundation)

**GSoC Proposal:** [PROPOSAL.md](PROPOSAL.md)

---

## Mentor Quick Walkthrough

This repository implements a prototype framework for organizing OmniFold outputs into a structured, reproducible dataset format compatible with publication workflows.

The solution is structured around the three evaluation tasks:

| Deliverable | File | Summary |
|-------------|------|---------|
| **Part 1 — Gap Analysis** | [`gap_analysis.md`](gap_analysis.md) | Inspection of all three HDF5 files, classification of 200 columns, identification of missing metadata |
| **Part 2 — Metadata Schema** | [`metadata.yaml`](metadata.yaml) + [`schema_design.md`](schema_design.md) | YAML schema documenting observables, weight semantics, systematics, normalization, and fiducial region |
| **Part 3 — Weighted Histogram** | [`weighted_histogram.py`](weighted_histogram.py) + [`tests/`](tests/) | Self-contained function with uncertainty propagation, 14 pytest tests covering edge cases |

To run the code:

```bash
pip install -r requirements.txt
python -m pytest tests/ -v                 # run all 14 tests
```

To try the demo notebook:

```bash
jupyter notebook examples/demo.ipynb
```

## Architecture

```
Raw OmniFold Outputs (.h5 files)
        │
        ▼
┌─────────────────┐
│   Data Loader    │  ← reads HDF5, auto-detects keys
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Schema Layer    │◄────│  metadata.yaml   │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐
│   Validation     │  ← integrity checks on weights & observables
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Analysis API   │  ← OmniFoldDataset: unified interface
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Visualization / Export / Storage    │
└─────────────────────────────────────┘
```

## Repository Structure

```
omnifold-hepdata-evaluation/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── gap_analysis.md              # Part 1: data exploration and gap identification
├── metadata.yaml                # Part 2: metadata schema for the nominal file
├── schema_design.md             # Part 2: justification of schema design choices
├── weighted_histogram.py        # Part 3: weighted histogram function
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # HDF5 loading and inspection utilities
│   ├── schema.py                # Metadata dataclasses and column classification
│   ├── storage.py               # Multi-format export (HDF5, Parquet, NumPy)
│   ├── validation.py            # Dataset integrity validation
│   └── api.py                   # OmniFoldDataset: high-level user API
├── tests/
│   ├── __init__.py
│   └── test_weighted_histogram.py  # 14 tests covering correctness and edge cases
├── examples/
│   └── demo.ipynb               # Interactive demonstration notebook
├── docs/
│   └── design.md                # Detailed architecture and design document
└── data/                        # (not committed — download from Zenodo)
    └── files/pseudodata/
        ├── multifold.h5
        ├── multifold_sherpa.h5
        └── multifold_nonDY.h5
```

## Data

The HDF5 files are not included in this repository due to their size (~770 MB for the three files).

Download from [Zenodo (record 11507450)](https://zenodo.org/records/11507450):

```bash
# Download and extract only the pseudodata files
curl -L -o data/files.zip "https://zenodo.org/api/records/11507450/files/files.zip/content"
unzip data/files.zip "files/pseudodata/multifold*.h5" -d data/
```

## Key Findings from Gap Analysis

The three HDF5 files contain 24 particle-level observables from an ATLAS Z+jets measurement, along with extensive per-event weights from OmniFold unfolding:

- **multifold.h5**: 418,014 events, 200 columns (24 observables + 176 weight columns including 100 ensemble replicas, 50 bootstrap replicas, and 26 detector/theory systematics)
- **multifold_sherpa.h5**: 326,430 events, 51 columns (alternative generator systematic)
- **multifold_nonDY.h5**: 433,397 events, 26 columns (sample composition systematic)

**Critical gaps identified:**
- No metadata about the experiment, generator, or analysis context
- No documentation of what each weight column represents
- No guidance on how to compute uncertainties from ensemble/bootstrap weights
- No link between files or explanation of their different event counts and column structures
- No fiducial region or normalization information

These gaps are addressed by the metadata schema in [`metadata.yaml`](metadata.yaml).

## Design Decisions

See [`docs/design.md`](docs/design.md) for the full design document. Key choices:

1. **Sidecar YAML metadata** — human-readable, diff-friendly, independent of data format
2. **Column classification by naming convention** — works with existing files without modification
3. **Validation returns results, not exceptions** — enables batch checking across files
4. **Self-contained histogram function** — computation + uncertainty + plotting in one call

## Installation

```bash
git clone https://github.com/sridhar-panigrahi/omnifold-hepdata-evaluation.git
cd omnifold-hepdata-evaluation
pip install -r requirements.txt
```

## Usage

### Weighted Histogram (Part 3)

```python
import numpy as np
from weighted_histogram import weighted_histogram

# Compute and plot a weighted histogram
values = np.random.normal(91.2, 2.5, 10000)
weights = np.random.exponential(1.0, 10000)
contents, edges, errors = weighted_histogram(
    values, weights, bins=40, xlabel="m_ll [GeV]", title="Z boson mass"
)
```

### Full API (Framework Prototype)

```python
from src.api import OmniFoldDataset

dataset = OmniFoldDataset("data/files/pseudodata/")
print(dataset.file_names)          # ['multifold', 'multifold_nonDY', 'multifold_sherpa']

# Load and inspect nominal file
summary = dataset.describe("multifold")
print(f"Events: {summary['n_events']}, Columns: {summary['n_columns']}")

# Access specific observable
pT_ll = dataset.get_observable("multifold", "pT_ll")

# Run validation suite
results = dataset.validate()
for r in results:
    print(r)  # [PASS] weights_positive: All 176 weight columns non-negative
```

## Tests

```bash
python -m pytest tests/ -v
```

14 tests covering:
- **Basic correctness**: uniform weights match `np.histogram`, weights shift the mean, error formula sqrt(sum(w^2))
- **Edge cases**: empty input, NaN/Inf filtering, shape mismatches, single event, all-zero weights
- **Normalization**: sum-to-one contract, mutually exclusive flags
- **Input types**: lists, integers, 2-D rejection

## Future Improvements

- **HEPData export**: Generate HEPData-compatible YAML submissions directly from weighted histograms + metadata, bridging unbinned OmniFold outputs with the binned publication format
- **Chunked I/O**: Support streaming large datasets via `pd.read_hdf(..., chunksize=...)` or Dask for memory-constrained environments
- **Schema validation tooling**: JSON Schema or Pydantic models for automated metadata validation in CI/CD pipelines
- **Cross-experiment portability**: Define a minimal common schema supporting ATLAS, CMS, and LHCb OmniFold results with experiment-specific extensions
- **Provenance tracking**: Capture full OmniFold training configuration (model architecture, hyperparameters, random seeds) to enable computational reproducibility
- **Integration with analysis frameworks**: Direct interoperability with ROOT, pyhf, and cabiern for downstream statistical analysis
