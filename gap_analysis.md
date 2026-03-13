# Gap Analysis: OmniFold Pseudodata HDF5 Files

**Author:** Shridhar Panigrahi

---

## 1. File Summary

| File | Events | Columns | Description |
|------|--------|---------|-------------|
| `multifold.h5` | 418,014 | 200 | Nominal result (MG5 generator) |
| `multifold_sherpa.h5` | 326,430 | 51 | Alternative generator (Sherpa) |
| `multifold_nonDY.h5` | 433,397 | 26 | Alternative sample composition (includes EW Zjj/VBF + diboson) |

All three files are stored as Pandas DataFrames in HDF5 format under the key `df`, using PyTables format version 2.1.

## 2. Column Classification

### Observables (24 columns, shared across all files)

| Column | Physical meaning | Type | Range (nominal) |
|--------|-----------------|------|-----------------|
| `pT_ll` | Dilepton transverse momentum | float32 | [200.0, 3233.9] GeV |
| `pT_l1` | Leading muon pT | float32 | [100.4, 2384.7] GeV |
| `pT_l2` | Subleading muon pT | float32 | [25.0, 1252.6] GeV |
| `eta_l1` | Leading muon pseudorapidity | float32 | [-2.40, 2.40] |
| `eta_l2` | Subleading muon pseudorapidity | float32 | [-2.40, 2.40] |
| `phi_l1` | Leading muon azimuthal angle | float32 | [-3.14, 3.14] |
| `phi_l2` | Subleading muon azimuthal angle | float32 | [-3.14, 3.14] |
| `y_ll` | Dilepton rapidity | float32 | [-2.40, 2.40] |
| `pT_trackj1` | Leading charged-particle jet pT | float32 | [0.59, 3079.3] GeV |
| `y_trackj1` | Leading jet rapidity | float32 | [-2.50, 2.50] |
| `phi_trackj1` | Leading jet azimuthal angle | float32 | [-3.14, 3.14] |
| `m_trackj1` | Leading jet mass | float32 | [-0.13, 258.8] GeV |
| `tau1_trackj1` | Leading jet 1-subjettiness | float32 | [0.0, 0.92] |
| `tau2_trackj1` | Leading jet 2-subjettiness | float32 | [0.0, 0.53] |
| `tau3_trackj1` | Leading jet 3-subjettiness | float32 | [0.0, 0.34] |
| `pT_trackj2` | Subleading jet pT | float32 | [0.50, 2586.7] GeV |
| `y_trackj2` | Subleading jet rapidity | float32 | [-2.50, 2.50] |
| `phi_trackj2` | Subleading jet azimuthal angle | float32 | [-3.14, 3.14] |
| `m_trackj2` | Subleading jet mass | float32 | [-0.09, 286.9] GeV |
| `tau1_trackj2` | Subleading jet 1-subjettiness | float32 | [0.0, 0.96] |
| `tau2_trackj2` | Subleading jet 2-subjettiness | float32 | [0.0, 0.57] |
| `tau3_trackj2` | Subleading jet 3-subjettiness | float32 | [0.0, 0.37] |
| `Ntracks_trackj1` | Leading jet track multiplicity | int32 | [1, 71] |
| `Ntracks_trackj2` | Subleading jet track multiplicity | int32 | [1, 69] |

### Weight Columns

**multifold.h5** has the richest weight structure (176 weight columns):

| Category | Columns | Purpose |
|----------|---------|---------|
| `weight_mc` | 1 | Original MC event weight |
| `weights_nominal` | 1 | Nominal OmniFold result |
| `weights_ensemble_0..99` | 100 | Ensemble replicas for statistical uncertainty estimation |
| `weights_dd` | 1 | Data-driven weight |
| `target_dd` | 1 | Data-driven target (auxiliary) |
| `weights_bootstrap_mc_0..24` | 25 | MC bootstrap replicas |
| `weights_bootstrap_data_0..24` | 25 | Data bootstrap replicas |
| `weights_pileup` | 1 | Pileup systematic |
| `weights_muEff*` | 4 | Muon efficiency systematics (Reco, Iso, Track, Trig) |
| `weights_muCal*` | 4 | Muon calibration systematics (ID, MS, ResBias, Scale) |
| `weights_trackEff*`, `weights_trackFake`, `weights_trackPtScale` | 4 | Tracking systematics |
| `weights_theory*` | 7 | Theory systematics (PS, MPI, AlphaS, QCD, PDF) |
| `weights_topBackground` | 1 | Top-quark background systematic |
| `weights_lumi` | 1 | Luminosity systematic |

**multifold_sherpa.h5** has 27 weight columns: `weight_mc`, `weights_nominal`, and 25 `weights_bootstrap_mc_*`.

**multifold_nonDY.h5** has only 2 weight columns: `weight_mc` and `weights_nominal`.

## 3. Missing Information for Reuse

A physicist attempting to reuse these weights would need the following information that is **not present** in the files:

### 3.1 Analysis Context
- **Experiment and dataset**: Nothing in the files identifies this as an ATLAS 13 TeV Z+jets measurement.
- **Integrated luminosity**: The 139 fb^{-1} value is not recorded.
- **Reference publication**: No link to the paper (arXiv:2405.20041) or analysis identifier (CERN-EP-2024-132).

### 3.2 Generator and Simulation Details
- **Nominal generator**: The user must know externally that `multifold.h5` uses MadGraph5 (MG5).
- **Alternative generators**: Nothing in `multifold_sherpa.h5` identifies it as Sherpa or explains its role as a systematic variation.
- **Non-DY composition**: `multifold_nonDY.h5` gives no indication of what "nonDY" means (i.e., includes EW Zjj/VBF and diboson contributions).
- **Detector simulation**: No mention of ATLAS Geant4 simulation or the pseudo-data reweighting procedure.

### 3.3 Event Selection and Fiducial Region
- **Fiducial cuts**: The pT_ll > 200 GeV cut is visible in the data (minimum value is 200.0001) but is not documented. Other cuts on muon pT, eta, and jet definitions are not recorded.
- **Jet definition**: The "trackj" prefix suggests charged-particle jets, but the jet algorithm, radius parameter, and clustering scheme are not specified.

### 3.4 Weight Semantics
- **Column role descriptions**: There is no documentation of what each weight category represents. A user would not know, for example, that `weights_ensemble_*` are for statistical uncertainty quantification, or that `weights_muCalID` represents the muon ID calibration systematic.
- **How to compute uncertainties**: The procedure for deriving an uncertainty band from the ensemble/bootstrap weights is not described. Should the user take the standard deviation of histograms across replicas? Envelope? Specific percentiles?
- **Relationship between weight_mc and weights_nominal**: Is `weights_nominal` already inclusive of `weight_mc`, or must they be multiplied?

### 3.5 Normalization
- **Cross-section normalization**: Whether weights are already normalized to a cross section (and in what units) is not stated. The mean weight (~0.004) suggests some normalization has been applied, but the target integral is unknown.
- **Units**: Observable units (GeV, radians, dimensionless) are not documented in the files.

### 3.6 File Relationships
- **No manifest or index**: Nothing links the three files together as parts of the same analysis. The naming convention is the only guide.
- **Different event counts**: `multifold.h5` has 418,014 events, `multifold_sherpa.h5` has 326,430, and `multifold_nonDY.h5` has 433,397. The reason for these differences is not documented.
- **Different column counts**: The files have 200, 51, and 26 columns respectively. It is unclear whether the missing systematic weights in the Sherpa and nonDY files are intentional or an omission.

## 4. Standardization Challenges

### 4.1 Heterogeneous Weight Structures
The three files in this single analysis already have different numbers of weight columns (200 vs. 51 vs. 26). Across experiments (ATLAS, CMS, LHCb), the weight categories, naming conventions, and uncertainty decomposition strategies will differ substantially. A flexible schema must accommodate both richly decomposed systematics (as in `multifold.h5`) and minimal structures (as in `multifold_nonDY.h5`).

### 4.2 Event Correspondence Across Files
The nominal and nonDY files share many identical observable values (same MC events, different composition), while the Sherpa file has entirely different events (different generator). Documenting which files share events and which do not is critical for correct statistical treatment but has no standard mechanism.

### 4.3 Unbinned vs. Binned Tension
Traditional HEPData publishes binned histograms with well-defined formats. OmniFold outputs are fundamentally unbinned — the power of the method lies in deferred binning. Any publication standard must support both paradigms without forcing premature binning, while still allowing conversion to binned formats when needed.

### 4.4 Scale
The nominal file alone is 498 MB; the full dataset (with `target.h5`) is ~3.6 GB. Large-scale OmniFold results could easily exceed this. Standard metadata schemas and data repositories (e.g., HEPData) were designed for much smaller binned datasets. Infrastructure for storing, versioning, and serving large unbinned datasets needs to be considered.

### 4.5 Provenance and Reproducibility
Ideally, the metadata would capture the full provenance chain: generator settings, detector simulation version, OmniFold training hyperparameters, number of iterations, and the specific model architecture. Without this, a downstream user cannot assess whether the weights are applicable to their use case or attempt to reproduce the result.
