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
        "-g", "--p-gap",
        dest="p_gap",
        type=float,
        default=None,
        help="Prior probability of an indel event (default: 1/size).",
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
        default=2.0,
        help="Shape parameter of the gamma distribution for site-rate variation (default: 2.0).",
    )
    parser.add_argument(
        "-b", "--scale",
        dest="gamma_scale",
        type=float,
        default=1.0,
        help="Scale parameter of the gamma distribution for site-rate variation (default: 1.0).",
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
        p_gap=args.p_gap,
        n_orthologs=args.n_orthologs,
        gamma_shape=args.gamma_shape,
        gamma_scale=args.gamma_scale,
        seed=args.seed,
        out=args.out,
        msa_format=args.msa_format,
        tree_format=args.tree_format,
    )
    run(config)


if __name__ == "__main__":
    main()
