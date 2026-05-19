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
        "-d", "--p-del",
        dest="p_del",
        type=float,
        default=None,
        help="Per-branch per-site deletion probability (default: 5%% of mean site substitution rate).",
    )
    parser.add_argument(
        "-i", "--p-ins",
        dest="p_ins",
        type=float,
        default=None,
        help="Per-branch per-site insertion probability (default: 0.1%% of mean site substitution rate).",
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
            "Expected substitutions per site on a branch of average rate (default: 0.05). "
            "Controls overall sequence divergence; also scales the default --p-del and --p-ins."
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
        p_del=args.p_del,
        p_ins=args.p_ins,
        n_orthologs=args.n_orthologs,
        gamma_shape=args.gamma_shape,
        branch_length=args.branch_length,
        seed=args.seed,
        out=args.out,
        msa_format=args.msa_format,
        tree_format=args.tree_format,
    )
    run(config)


if __name__ == "__main__":
    main()
