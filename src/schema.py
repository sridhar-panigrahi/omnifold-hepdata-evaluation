"""Schema definitions for OmniFold dataset metadata.

Defines the expected structure of metadata YAML files and provides
validation logic to ensure metadata completeness and consistency.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml


# Column classification based on actual ATLAS OmniFold Z+jets file contents.
# Observable names follow the conventions in the released HDF5 files.
OBSERVABLE_COLUMNS = [
    "pT_ll", "pT_l1", "pT_l2",
    "eta_l1", "eta_l2",
    "phi_l1", "phi_l2",
    "y_ll",
    "pT_trackj1", "y_trackj1", "phi_trackj1", "m_trackj1",
    "tau1_trackj1", "tau2_trackj1", "tau3_trackj1",
    "pT_trackj2", "y_trackj2", "phi_trackj2", "m_trackj2",
    "tau1_trackj2", "tau2_trackj2", "tau3_trackj2",
    "Ntracks_trackj1", "Ntracks_trackj2",
]

# Columns whose names contain "weight" or match known auxiliary weight columns
WEIGHT_PATTERN = "weight"
AUXILIARY_WEIGHT_COLUMNS = ["target_dd"]


@dataclass
class ObservableInfo:
    """Description of a single physics observable."""
    name: str
    description: str
    unit: str = ""
    range_min: Optional[float] = None
    range_max: Optional[float] = None


@dataclass
class SystematicVariation:
    """Description of a systematic uncertainty source."""
    name: str
    description: str
    file: str
    variation_type: str  # "generator", "composition", "detector", etc.


@dataclass
class DatasetMetadata:
    """Top-level metadata for an OmniFold analysis result."""
    analysis_name: str
    description: str
    experiment: str
    collision_energy: str
    luminosity: str
    generator_nominal: str
    detector_simulation: str
    omnifold_iterations: int
    fiducial_region: Dict[str, str]
    observables: List[ObservableInfo]
    systematic_variations: List[SystematicVariation]
    normalization: Dict[str, str]
    files: Dict[str, str]
    contact: str = ""
    reference: str = ""
    version: str = "1.0"

    @classmethod
    def from_yaml(cls, path: str) -> "DatasetMetadata":
        """Load metadata from a YAML file."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        observables = [
            ObservableInfo(**obs) for obs in raw.get("observables", [])
        ]
        systematics = [
            SystematicVariation(**sys)
            for sys in raw.get("systematic_variations", [])
        ]

        return cls(
            analysis_name=raw["analysis_name"],
            description=raw.get("description", ""),
            experiment=raw.get("experiment", ""),
            collision_energy=raw.get("collision_energy", ""),
            luminosity=raw.get("luminosity", ""),
            generator_nominal=raw.get("generator_nominal", ""),
            detector_simulation=raw.get("detector_simulation", ""),
            omnifold_iterations=raw.get("omnifold_iterations", 0),
            fiducial_region=raw.get("fiducial_region", {}),
            observables=observables,
            systematic_variations=systematics,
            normalization=raw.get("normalization", {}),
            files=raw.get("files", {}),
            contact=raw.get("contact", ""),
            reference=raw.get("reference", ""),
            version=raw.get("version", "1.0"),
        )

    def to_yaml(self, path: str) -> None:
        """Serialize metadata to a YAML file."""
        data = {
            "version": self.version,
            "analysis_name": self.analysis_name,
            "description": self.description,
            "experiment": self.experiment,
            "collision_energy": self.collision_energy,
            "luminosity": self.luminosity,
            "generator_nominal": self.generator_nominal,
            "detector_simulation": self.detector_simulation,
            "omnifold_iterations": self.omnifold_iterations,
            "fiducial_region": self.fiducial_region,
            "observables": [
                {
                    "name": o.name,
                    "description": o.description,
                    "unit": o.unit,
                    "range_min": o.range_min,
                    "range_max": o.range_max,
                }
                for o in self.observables
            ],
            "systematic_variations": [
                {
                    "name": s.name,
                    "description": s.description,
                    "file": s.file,
                    "variation_type": s.variation_type,
                }
                for s in self.systematic_variations
            ],
            "normalization": self.normalization,
            "files": self.files,
            "contact": self.contact,
            "reference": self.reference,
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def classify_columns(columns: List[str]) -> Dict[str, List[str]]:
    """Classify DataFrame columns into observables, weights, and other.

    Parameters
    ----------
    columns : list of str
        Column names from an HDF5 DataFrame.

    Returns
    -------
    dict with keys 'observables', 'weights', 'other'
    """
    result = {"observables": [], "weights": [], "other": []}
    for col in columns:
        if col in OBSERVABLE_COLUMNS:
            result["observables"].append(col)
        elif WEIGHT_PATTERN in col.lower() or col in AUXILIARY_WEIGHT_COLUMNS:
            result["weights"].append(col)
        else:
            result["other"].append(col)
    return result
