from dataclasses import dataclass

import numpy as np

VALID_MSA_FORMATS = frozenset({"fasta", "clustal", "nexus", "phylip", "stockholm"})
VALID_TREE_FORMATS = frozenset({"newick", "nexus", "nexml", "phyloxml", "cdao"})


@dataclass
class SimulationConfig:
    """All parameters that govern a simulation run."""

    size: int
    length: int
    del_factor: float = 0.05
    ins_factor: float = 0.001
    n_orthologs: int = 1
    gamma_shape: float = 0.75
    branch_length: float = 0.05
    dup_boost_factor: float = 2.0
    dup_boost_decay: float = 3.0
    seed: int | None = None
    out: str = "fakeprot_out"
    msa_format: str = "fasta"
    tree_format: str = "newick"

    def __post_init__(self) -> None:
        self.p_del = self.del_factor * self.branch_length
        self.p_ins = self.ins_factor * self.branch_length
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
