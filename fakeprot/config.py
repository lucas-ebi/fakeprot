from dataclasses import dataclass

import numpy as np

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
    branch_length: float = 0.05
    dup_boost_prob: float = 0.5
    dup_boost_factor: float = 2.0
    dup_boost_decay: float = 3.0
    seed: int | None = None
    out: str = "fakeprot_out"
    msa_format: str = "fasta"
    tree_format: str = "newick"

    def __post_init__(self) -> None:
        if self.p_del is None:
            # Deletion rate = 5 % of branch_length (5 % of substitutions result in deletions).
            self.p_del = 0.05 * self.branch_length
        if self.p_ins is None:
            # Per-lineage insertion rate ≈ 0.1 % of substitution rate.
            # Each insertion is amplified ~n-fold in the alignment (it creates a gap in
            # every other sequence), so this small coefficient already yields ~1 % insertion
            # gap content at n = 100, growing toward ~10 % at n = 1000.
            self.p_ins = 0.001 * self.branch_length
        if self.msa_format not in VALID_MSA_FORMATS:
            raise ValueError(
                f"msa_format must be one of {sorted(VALID_MSA_FORMATS)}, got {self.msa_format!r}"
            )
        if self.tree_format not in VALID_TREE_FORMATS:
            raise ValueError(
                f"tree_format must be one of {sorted(VALID_TREE_FORMATS)}, got {self.tree_format!r}"
            )
        if self.seed is None:
            self.seed = int(np.random.randint(0, 2**31))
        np.random.seed(self.seed)
