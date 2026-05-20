"""
Command-line entry point. Parses arguments, builds a SimulationConfig, and runs.
"""

import argparse

from fakeprot.config import SimulationConfig
from fakeprot.simulation import run


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "PFgenerator: generate artificial protein families by simulating protein evolution."
        )
    )
    parser.add_argument(
        "size",
        type=int,
        help=(
            "Minimum number of extant (leaf) sequences. "
            "Sequences at internal nodes are also produced during the simulation."
        ),
    )
    parser.add_argument(
        "length",
        type=int,
        help="Length of the root sequence (may grow if insertions occur).",
    )
    parser.add_argument(
        "-o", "--out",
        type=str,
        default="fakeprot_out",
        help="Prefix for all output filenames (default: fakeprot_out).",
    )
    parser.add_argument(
        "-d", "--del-factor",
        dest="del_factor",
        type=float,
        default=0.05,
        help="Deletion rate as a fraction of the branch-length parameter (default: 0.05).",
    )
    parser.add_argument(
        "-i", "--ins-factor",
        dest="ins_factor",
        type=float,
        default=0.001,
        help="Insertion rate as a fraction of the branch-length parameter (default: 0.001).",
    )
    parser.add_argument(
        "-n", "--n-orthologs",
        dest="n_orthologs",
        type=int,
        default=1,
        help="Target number of ortholog groups (default: 1).",
    )
    parser.add_argument(
        "-a", "--shape",
        dest="gamma_shape",
        type=float,
        default=0.75,
        help="Shape parameter of the gamma distribution for site-rate variation (default: 0.75).",
    )
    parser.add_argument(
        "-b", "--branch-length",
        dest="branch_length",
        type=float,
        default=0.05,
        help=(
            "Average branch length: expected substitutions per site (default: 0.05). "
            "Controls overall sequence divergence; also scales the default --del-factor and --ins-factor."
        ),
    )
    parser.add_argument(
        "--dup-boost-factor",
        dest="dup_boost_factor",
        type=float,
        default=2.0,
        help="Branch-length multiplier applied to a boosted duplicate edge (default: 2.0).",
    )
    parser.add_argument(
        "--dup-boost-decay",
        dest="dup_boost_decay",
        type=float,
        default=3.0,
        help=(
            "E-folding distance (in speciation steps) for the post-duplication rate boost "
            "(default: 3.0). Set to 0 to restrict the boost to the duplicate's creation "
            "edge only (v0.4.0 behaviour)."
        ),
    )
    parser.add_argument(
        "--branch-cv",
        dest="branch_length_cv",
        type=float,
        default=0.4,
        help=(
            "Coefficient of variation for per-branch length noise (default: 0.4, "
            "consistent with empirical relaxed-clock estimates; Drummond et al. 2006). "
            "Set to 0 for deterministic (equal-length) branches."
        ),
    )
    parser.add_argument(
        "-r", "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )
    parser.add_argument(
        "-f", "--msa-format",
        dest="msa_format",
        type=str,
        default="fasta",
        help="Output format for sequence alignments: fasta, clustal, nexus, phylip, stockholm (default: fasta).",
    )
    parser.add_argument(
        "-t", "--tree-format",
        dest="tree_format",
        type=str,
        default="newick",
        help="Output format for phylogeny trees: newick, nexus, nexml, phyloxml, cdao (default: newick).",
    )

    args = parser.parse_args()
    config = SimulationConfig(
        size=args.size,
        length=args.length,
        del_factor=args.del_factor,
        ins_factor=args.ins_factor,
        n_orthologs=args.n_orthologs,
        gamma_shape=args.gamma_shape,
        branch_length=args.branch_length,
        dup_boost_factor=args.dup_boost_factor,
        dup_boost_decay=args.dup_boost_decay,
        branch_length_cv=args.branch_length_cv,
        seed=args.seed,
        out=args.out,
        msa_format=args.msa_format,
        tree_format=args.tree_format,
    )
    run(config)


if __name__ == "__main__":
    main()
