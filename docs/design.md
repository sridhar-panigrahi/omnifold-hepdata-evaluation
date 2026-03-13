# Design Document: OmniFold Publication Tools Prototype

**Author:** Shridhar Panigrahi
**Date:** March 2026
**Context:** GSoC 2026 evaluation — OmniFold Publication Tools (CERN / HEP Software Foundation)

---

## 1. Problem Statement

OmniFold produces per-event weights rather than traditional binned histograms.
This creates a gap between how results are generated and how they can be
published, shared, and reused by the broader physics community. Specifically:

- HDF5 files contain raw DataFrames but no embedded documentation of what
  each column represents, how the data was generated, or what selections
  were applied.
- There is no standard for distinguishing observable columns from weight
  columns from auxiliary metadata.
- Systematic variations live in separate files with no formal link to the
  nominal result.
- Normalization, fiducial region definitions, and generator information are
  not captured anywhere in the data files themselves.

This prototype addresses these gaps with a minimal but extensible framework.

## 2. Architecture

```
Raw OmniFold Outputs (.h5 files)
        │
        ▼
┌─────────────────┐
│   Data Loader    │  ← reads HDF5, auto-detects keys
│  (data_loader.py)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Schema Layer    │◄────│  metadata.yaml   │
│   (schema.py)    │     │  (user-authored) │
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐
│   Validation     │  ← integrity checks on weights & observables
│ (validation.py)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Analysis API   │  ← OmniFoldDataset: unified user-facing interface
│     (api.py)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Visualization / Export / Storage    │
│  (weighted_histogram.py, storage.py) │
└─────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Purpose | Key Design Choice |
|-------|---------|-------------------|
| **Data Loader** | Read HDF5 files into DataFrames | Auto-detect HDF5 keys; support lazy loading |
| **Schema** | Define and parse metadata | Dataclasses for type safety; YAML for human readability |
| **Validation** | Verify dataset integrity | Return structured results, not exceptions — so users can batch-check |
| **API** | Unified user interface | Single entry point (`OmniFoldDataset`) encapsulates all operations |
| **Storage** | Export to multiple formats | HDF5, Parquet, NumPy — users choose what fits their workflow |
| **Histogram** | Compute and plot weighted observables | Self-contained function with proper uncertainty propagation |

## 3. Key Design Decisions

### 3.1 Metadata as a Sidecar YAML File

**Decision:** Metadata lives in a separate YAML file alongside the HDF5 data,
rather than being embedded as HDF5 attributes.

**Rationale:**
- HDF5 attributes have limited type support (no nested structures).
- YAML is human-readable and diff-friendly for version control.
- Sidecar files allow metadata to evolve independently of the data format.
- This mirrors established patterns in scientific data (e.g., FITS headers,
  BIDSsidecar JSON in neuroimaging).

**Trade-off:** The metadata file can become disconnected from the data file.
Mitigation: the validation layer checks that metadata references match actual
file contents.

### 3.2 Column Classification by Convention

**Decision:** Columns are classified as observables, weights, or other based
on naming patterns, not embedded type annotations.

**Rationale:**
- The existing HDF5 files use a flat DataFrame with no column metadata.
- Convention-based classification works immediately with existing data.
- The schema module defines the classification rules in one place.

**Trade-off:** Relies on consistent column naming across analyses.
For the GSoC project, a formal column-type annotation standard could be
defined as part of the metadata schema.

### 3.3 Validation Returns Results, Not Exceptions

**Decision:** Validation functions return `ValidationResult` objects rather
than raising exceptions on failure.

**Rationale:**
- Users want to see *all* issues at once, not stop at the first failure.
- Result objects can be logged, serialized, and aggregated.
- Calling code can decide which failures are fatal vs. warnings.

### 3.4 Self-Contained Histogram Function

**Decision:** `weighted_histogram` is a single function that handles
computation, uncertainty propagation, and plotting.

**Rationale:**
- The evaluation task specifically asks for a self-contained function.
- Combining computation and visualization avoids state synchronization bugs.
- The `plot=False` flag allows pure-computation use in pipelines.
- Uncertainty is computed as sqrt(sum(w_i^2)), the standard formula for
  weighted histograms that reduces to sqrt(N) for unit weights.

## 4. Assumptions

- Input files are Pandas DataFrames stored in HDF5 via `pd.to_hdf()`.
- Systematic variations are stored in separate files with identical event
  ordering and observable columns.
- Weight columns follow OmniFold convention: per-event reweighting factors
  from each iteration.
- All files for a single analysis share the same event count (same underlying
  MC sample, different reweighting).

## 5. Limitations

- **No streaming/chunked reading:** All data is loaded into memory. For the
  24-observable ATLAS dataset (~3.6 GB), this requires sufficient RAM.
- **No HEPData export:** The prototype does not yet produce HEPData-compatible
  YAML submissions. This is a natural next step for the GSoC project.
- **No ML model artifact storage:** OmniFold model weights/checkpoints are
  not captured. The prototype focuses on the output weights only.
- **Single analysis scope:** The schema and API assume a single analysis
  result. Multi-analysis catalogs would require a higher-level index.

## 6. Scalability Considerations

For the full GSoC project, the following extensions would be needed:

1. **Chunked I/O:** Use `pd.read_hdf(..., chunksize=...)` or Dask for
   datasets that exceed available memory.
2. **Schema versioning:** Add a `schema_version` field so that older metadata
   files can be migrated forward as the format evolves.
3. **Cross-experiment portability:** Define a minimal common schema that
   works for ATLAS, CMS, and LHCb OmniFold results, with experiment-specific
   extensions.
4. **Automated validation pipelines:** Integrate validation into CI/CD so
   that published datasets are guaranteed to pass all checks.
5. **HEPData integration:** Generate HEPData submission YAML directly from
   the metadata + weighted histograms, bridging unbinned and binned worlds.
