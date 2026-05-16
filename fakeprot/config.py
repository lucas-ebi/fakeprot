from dataclasses import dataclass

import numpy as np
from scipy.stats import gamma as _gamma

VALID_MSA_FORMATS = frozenset({"fasta", "clustal", "nexus", "phylip", "stockholm"})
VALID_TREE_FORMATS = frozenset({"newick", "nexus", "nexml", "phyloxml", "cdao"})


@dataclass
class SimulationConfig:
    """All parameters that govern a simulation run."""

    size: int
    length: int
    p_del: float | None = None
    p_ins: float | None = None
    n_orthologs: int = 1
    gamma_shape: float = 0.75
    seed: int | None = None
    out: str = "fakeprot_out"
    msa_format: str = "fasta"
    tree_format: str = "newick"

    def __post_init__(self) -> None:
        scale = 1.0 / self.gamma_shape
        lo = _gamma.ppf(0.01, self.gamma_shape, scale=scale)
        hi = _gamma.ppf(0.99, self.gamma_shape, scale=scale)
        mean_r = (lo + hi) / (2.0 * hi)   # mean of linspace-normalised rates
        if self.p_del is None:
            # Target 3% gap content from deletions; f_del ≈ p_del × mean_r × 2 ln(n)
            self.p_del = 0.03 / (mean_r * 2.0 * np.log(self.size))
        if self.p_ins is None:
            # Target 2% gap content from insertions; f_ins ≈ 4n p_ins mean_r / (1 + 4n p_ins mean_r)
            self.p_ins = 0.02 / ((1.0 - 0.02) * 4.0 * self.size * mean_r)
        if self.msa_format not in VALID_MSA_FORMATS:
            raise ValueError(
                f"msa_format must be one of {sorted(VALID_MSA_FORMATS)}, got {self.msa_format!r}"
            )
        if self.tree_format not in VALID_TREE_FORMATS:
            raise ValueError(
                f"tree_format must be one of {sorted(VALID_TREE_FORMATS)}, got {self.tree_format!r}"
            )
        np.random.seed(self.seed)
