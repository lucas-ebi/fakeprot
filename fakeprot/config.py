from dataclasses import dataclass

import numpy as np

VALID_MSA_FORMATS = frozenset({"fasta", "clustal", "nexus", "phylip", "stockholm"})
VALID_TREE_FORMATS = frozenset({"newick", "nexus", "nexml", "phyloxml", "cdao"})


@dataclass
class SimulationConfig:
    """All parameters that govern a simulation run."""

    size: int
    length: int
    p_gap: float | None = None
    n_orthologs: int = 1
    gamma_shape: float = 0.75
    seed: int | None = None
    out: str = "fakeprot_out"
    msa_format: str = "fasta"
    tree_format: str = "newick"

    def __post_init__(self) -> None:
        if self.p_gap is None:
            self.p_gap = 1.0 / self.size
        if self.msa_format not in VALID_MSA_FORMATS:
            raise ValueError(
                f"msa_format must be one of {sorted(VALID_MSA_FORMATS)}, got {self.msa_format!r}"
            )
        if self.tree_format not in VALID_TREE_FORMATS:
            raise ValueError(
                f"tree_format must be one of {sorted(VALID_TREE_FORMATS)}, got {self.tree_format!r}"
            )
        np.random.seed(self.seed)
