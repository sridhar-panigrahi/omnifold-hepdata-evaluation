# Schema Design Justification

**Author:** Shridhar Panigrahi

---

## Structure Overview

The `metadata.yaml` file is organized into seven clearly separated sections:

1. **Analysis identification** — what this dataset is and where it comes from
2. **Provenance** — how the data was generated
3. **Fiducial region** — what event selection was applied
4. **Observable definitions** — column-by-column documentation of physics quantities
5. **Weight column documentation** — what each weight category means and how to use it
6. **Systematic variation files** — how the companion files relate to the nominal result
7. **Normalization** — how the weights are scaled and what their sum represents

## Why This Structure

### Addressing the gaps, not duplicating the data

The gap analysis revealed that the HDF5 files are self-describing in the narrow sense (column names, types, shapes) but provide zero scientific context. The metadata schema focuses exclusively on filling those gaps rather than re-encoding information already present in the files.

I deliberately avoided embedding the metadata inside the HDF5 files for three reasons:

1. **Human readability**: YAML files are immediately readable in any text editor and diff cleanly in version control. HDF5 attributes require specialized tools to inspect.
2. **Separation of concerns**: The data files contain measurements. The metadata file contains the documentation needed to *interpret* those measurements. Keeping them separate means either can be updated independently (e.g., correcting a description without touching the data).
3. **Portability**: YAML is natively supported by most programming languages and data validation frameworks. This makes it straightforward to build tooling (loaders, validators, converters) without requiring HDF5-specific libraries just to read documentation.

### Hierarchical organization reflects user mental model

A physicist approaching this dataset typically asks questions in a natural order:

1. "What analysis is this?" → `analysis` section
2. "How was it produced?" → `provenance` section
3. "What phase space does it cover?" → `fiducial_region` section
4. "What do these columns mean?" → `observables` and `weight_columns` sections
5. "How do the files relate to each other?" → `systematic_files` section
6. "How are the weights normalized?" → `normalization` section

The schema mirrors this progression so that a user can read it top-to-bottom and build up understanding incrementally.

### Weight documentation is the most critical section

The gap analysis showed that the weight columns are by far the most confusing aspect of the files. The nominal file alone has 176 weight columns across 8 distinct categories, yet none are documented in the HDF5 file itself.

I structured the weight documentation by *category* (nominal, ensemble, bootstrap, detector systematics, theory systematics) rather than listing all 176 columns individually. This provides:

- **Grouped understanding**: A user learns the purpose of "ensemble weights" once, then understands all 100 columns.
- **Explicit usage instructions**: The `uncertainty_procedure` field tells users how to actually compute uncertainties from the ensemble replicas, which is not obvious from the column names alone.
- **Relationship clarity**: The note that `weights_nominal` already incorporates `weight_mc` prevents the most likely misuse (double-counting the MC weight).

### What I chose to include vs. leave out

**Included:**
- Everything needed for a physicist to correctly use the weights in a new analysis without consulting the original paper.
- Observable units and physical ranges (critical for plotting and sanity checks).
- Explicit `shares_events_with_nominal` flag for systematic files, since this affects how uncertainties are correlated.
- Schema version field for forward compatibility.

**Deliberately excluded:**
- Full OmniFold training hyperparameters (learning rate, architecture, epochs). These are important for *reproducibility of the training* but not for *reusing the output weights*. They belong in a separate provenance record or in the paper's supplementary material.
- Per-event truth-level labels or reco-level quantities (these live in `target.h5` which is a separate concern).
- Exhaustive listing of every column — the YAML groups columns by category with count and pattern, keeping the file readable rather than generating a 200-line flat list.

### How a new user would interact with this file

A physicist who has never seen this analysis before would:

1. **Read the `analysis` block** to confirm this is the dataset they're looking for.
2. **Check `fiducial_region`** to verify the phase space matches their needs.
3. **Look up a specific observable** in the `observables` list to find its units and meaning.
4. **Read `weight_columns.nominal`** to know which column to use for histogramming.
5. **Follow `weight_columns.ensemble.uncertainty_procedure`** to compute systematic uncertainty bands.
6. **Check `systematic_files`** to understand what `multifold_sherpa.h5` represents and whether it shares events with the nominal file.
7. **Use `normalization`** to verify that their histograms integrate to the expected cross section.

The metadata file is designed so that each of these lookups is a single scroll to the relevant section, without needing to parse the entire file or cross-reference multiple documents.

## Scalability Considerations

For a production version of this schema, I would add:

- **JSON Schema validation**: A formal JSON Schema (or equivalent YAML schema) that tools can use to programmatically validate metadata files, catching missing fields or type mismatches before publication.
- **Schema versioning with migration**: The `schema_version` field enables automated migration when the format evolves. Older metadata files remain valid via version-aware loaders.
- **Experiment-agnostic core + extensions**: A minimal common schema (analysis name, observables, weights, normalization) shared across ATLAS/CMS/LHCb, with experiment-specific extension blocks for detector-specific systematics or naming conventions.
- **Provenance chain**: Links to the exact software versions, random seeds, and training configurations used to produce the weights, enabling full computational reproducibility beyond what the paper provides.
