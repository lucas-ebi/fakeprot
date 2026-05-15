"""
Write all output files for a completed simulation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import networkx as nx
import numpy as np
from Bio import AlignIO, Phylo
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.BaseTree import Tree
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import seq3
from pandas import DataFrame

from fakeprot.config import SimulationConfig
from fakeprot.evolution.tree import (
    build_gene_tree,
    build_msa,
    build_species_tree,
    find_ortholog_groups,
    og_label,
)
from fakeprot.models.sequence import Sequence
from fakeprot.models.species import Species

# fakeprot version — keep in sync with pyproject.toml
_VERSION = "0.2.0"


def write_outputs(
    config: SimulationConfig,
    collection: list[Sequence],
    sequence_tree: nx.DiGraph,
    species_tree: nx.DiGraph,
    root_sequence: Sequence,
    root_species: Species,
    orthologs: list[Sequence],
    sequence_length: int,
) -> None:
    """Write all output files produced by a simulation run."""
    leaves = [n for n in sequence_tree.nodes() if sequence_tree.out_degree(n) == 0]
    ortholog_groups = find_ortholog_groups(sequence_tree, orthologs)

    _write_all_sequences(config, collection)
    _write_current_sequences(config, sequence_tree, root_sequence)
    _write_gene_tree(config, sequence_tree, root_sequence, sequence_length)
    _write_species_cladogram(config, species_tree, root_species)
    _write_ortholog_groups_csv(config, ortholog_groups, orthologs)
    if config.n_orthologs > 1:
        _write_ortholog_alignments(config, ortholog_groups)
    _write_pc_groups_csv(config, ortholog_groups, orthologs, sequence_length)
    _write_run_info(config)


def _write_all_sequences(config: SimulationConfig, collection: list[Sequence]) -> None:
    alignment = MultipleSeqAlignment(
        [SeqRecord(Seq(seq.sequence), id=str(seq), description="") for seq in collection]
    )
    path = f"{config.out}_all_sequences.{config.msa_format}"
    with open(path, "w") as fh:
        AlignIO.write(alignment, fh, config.msa_format)


def _write_current_sequences(
    config: SimulationConfig,
    sequence_tree: nx.DiGraph,
    root_sequence: Sequence,
) -> None:
    records = build_msa(sequence_tree, root_sequence)
    alignment = MultipleSeqAlignment(records)
    path = f"{config.out}_current_sequences.{config.msa_format}"
    with open(path, "w") as fh:
        AlignIO.write(alignment, fh, config.msa_format)


def _write_gene_tree(
    config: SimulationConfig,
    sequence_tree: nx.DiGraph,
    root_sequence: Sequence,
    sequence_length: int,
) -> None:
    root_clade = build_gene_tree(sequence_tree, root_sequence, sequence_length)
    tree = Tree(root=root_clade, rooted=True)
    Phylo.write(tree, f"{config.out}_gene_tree.{config.tree_format}", config.tree_format)


def _write_species_cladogram(
    config: SimulationConfig,
    species_tree: nx.DiGraph,
    root_species: Species,
) -> None:
    root_clade = build_species_tree(species_tree, root_species)
    cladogram = Tree(root=root_clade, rooted=True)
    Phylo.write(
        cladogram,
        f"{config.out}_species_cladogram.{config.tree_format}",
        config.tree_format,
    )


def _write_ortholog_groups_csv(
    config: SimulationConfig,
    ortholog_groups: dict[int, list[Sequence]],
    orthologs: list[Sequence],
) -> None:
    rows: dict[str, list] = {"Sequence": [], "OG": []}
    for i in sorted(ortholog_groups):
        for seq in ortholog_groups[i]:
            rows["Sequence"].append(seq)
            rows["OG"].append(og_label(i))
    DataFrame(rows).to_csv(f"{config.out}_ortholog_groups.csv", index=False)


def _write_ortholog_alignments(
    config: SimulationConfig,
    ortholog_groups: dict[int, list[Sequence]],
) -> None:
    for i, members in ortholog_groups.items():
        alignment = MultipleSeqAlignment(
            [SeqRecord(Seq(seq.sequence), id=str(seq), description="") for seq in members]
        )
        path = f"{config.out}_OG_{og_label(i)}.{config.msa_format}"
        with open(path, "w") as fh:
            AlignIO.write(alignment, fh, config.msa_format)


def _write_pc_groups_csv(
    config: SimulationConfig,
    ortholog_groups: dict[int, list[Sequence]],
    orthologs: list[Sequence],
    sequence_length: int,
) -> None:
    columns: dict[str, list] = {"MSA Column": list(range(1, sequence_length + 1))}
    for i in range(len(orthologs)):
        columns[f"OG {og_label(i)}"] = []

    for col in range(sequence_length):
        for j, ancestor in enumerate(orthologs):
            msa = np.array([list(seq.sequence) for seq in ortholog_groups[j]])
            column_residues = msa[:, col]
            freq: dict[str, float] = {}
            for aa in column_residues:
                if aa != "-":
                    freq[aa] = freq.get(aa, 0) + 1
            if freq:
                total = len(msa)
                freq = {k: v / total for k, v in freq.items()}
                freq_str = " ".join(
                    f"{seq3(aa)}:{pct * 100:.2f}%"
                    for aa, pct in sorted(freq.items(), key=lambda x: x[1], reverse=True)
                )
                entry = f"{ancestor.pc_groups[col]} ({freq_str})"
            else:
                entry = ancestor.pc_groups[col]
            columns[f"OG {og_label(j)}"].append(entry)

    DataFrame(columns).to_csv(f"{config.out}_physicochemical_groups.csv", index=False)


def _write_run_info(config: SimulationConfig) -> None:
    info = {
        "version": _VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "size": config.size,
            "length": config.length,
            "p_gap": config.p_gap,
            "n_orthologs": config.n_orthologs,
            "gamma_shape": config.gamma_shape,
            "gamma_scale": config.gamma_scale,
            "seed": config.seed,
            "out": config.out,
            "msa_format": config.msa_format,
            "tree_format": config.tree_format,
        },
    }
    with open(f"{config.out}_run_info.json", "w") as fh:
        json.dump(info, fh, indent=2)
