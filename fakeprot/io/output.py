"""
Write all output files for a completed simulation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import networkx as nx
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
    collect_leaf_sequences,
    find_ortholog_groups,
    og_label,
)
from fakeprot.models.msa_store import MsaStore, decode_chars
from fakeprot.models.sequence import Sequence
from fakeprot.models.species import Species
from fakeprot.substitution import AMINO_ACIDS, CHAR_GAP, PHYSICOCHEMICAL_GROUPS

# fakeprot version — keep in sync with pyproject.toml
_VERSION = "0.2.0"


def write_outputs(
    config: SimulationConfig,
    store: MsaStore,
    collection: list[Sequence],
    sequence_tree: nx.DiGraph,
    species_tree: nx.DiGraph,
    root_sequence: Sequence,
    root_species: Species,
    orthologs: list[Sequence],
    sequence_length: int,
) -> None:
    """Write all output files produced by a simulation run."""
    ortholog_groups = find_ortholog_groups(sequence_tree, orthologs)

    _write_all_sequences(config, store, collection)
    _write_current_sequences(config, store, sequence_tree, root_sequence)
    _write_gene_tree(config, store, sequence_tree, root_sequence, sequence_length)
    _write_species_cladogram(config, species_tree, root_species)
    _write_ortholog_groups_csv(config, ortholog_groups, orthologs)
    if config.n_orthologs > 1:
        _write_ortholog_alignments(config, store, ortholog_groups)
    _write_pc_groups_csv(config, store, ortholog_groups, orthologs, sequence_length)
    _write_run_info(config)


def _write_all_sequences(
    config: SimulationConfig, store: MsaStore, collection: list[Sequence]
) -> None:
    path = f"{config.out}_all_sequences.{config.msa_format}"
    if config.msa_format == "fasta":
        _write_fasta(path, store, collection)
        return
    alignment = MultipleSeqAlignment(
        [SeqRecord(Seq(decode_chars(store.chars[seq.row])), id=str(seq), description="")
         for seq in collection]
    )
    with open(path, "w") as fh:
        AlignIO.write(alignment, fh, config.msa_format)


def _write_current_sequences(
    config: SimulationConfig,
    store: MsaStore,
    sequence_tree: nx.DiGraph,
    root_sequence: Sequence,
) -> None:
    path = f"{config.out}_current_sequences.{config.msa_format}"
    if config.msa_format == "fasta":
        _write_fasta(path, store, collect_leaf_sequences(sequence_tree, root_sequence))
        return
    records = build_msa(sequence_tree, root_sequence, store)
    alignment = MultipleSeqAlignment(records)
    with open(path, "w") as fh:
        AlignIO.write(alignment, fh, config.msa_format)


def _write_gene_tree(
    config: SimulationConfig,
    store: MsaStore,
    sequence_tree: nx.DiGraph,
    root_sequence: Sequence,
    sequence_length: int,
) -> None:
    root_clade = build_gene_tree(sequence_tree, root_sequence, sequence_length, store)
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
    store: MsaStore,
    ortholog_groups: dict[int, list[Sequence]],
) -> None:
    for i, members in ortholog_groups.items():
        path = f"{config.out}_OG_{og_label(i)}.{config.msa_format}"
        if config.msa_format == "fasta":
            _write_fasta(path, store, members)
            continue
        alignment = MultipleSeqAlignment(
            [SeqRecord(Seq(decode_chars(store.chars[seq.row])), id=str(seq), description="")
             for seq in members]
        )
        with open(path, "w") as fh:
            AlignIO.write(alignment, fh, config.msa_format)


def _write_fasta(path: str, store: MsaStore, sequences: list[Sequence]) -> None:
    with open(path, "w") as fh:
        for seq in sequences:
            fh.write(f">{seq}\n")
            fh.write(decode_chars(store.chars[seq.row]))
            fh.write("\n")


def _write_pc_groups_csv(
    config: SimulationConfig,
    store: MsaStore,
    ortholog_groups: dict[int, list[Sequence]],
    orthologs: list[Sequence],
    sequence_length: int,
) -> None:
    columns: dict[str, list] = {"MSA Column": list(range(1, sequence_length + 1))}
    for i in range(len(orthologs)):
        columns[f"OG {og_label(i)}"] = []

    og_row_indices = [
        [seq.row for seq in ortholog_groups[j]] for j in range(len(orthologs))
    ]

    for col in range(sequence_length):
        for j, ancestor in enumerate(orthologs):
            col_chars = store.chars[og_row_indices[j], col]
            freq: dict[str, float] = {}
            for c in col_chars:
                if c < CHAR_GAP:
                    aa = AMINO_ACIDS[int(c)]
                    freq[aa] = freq.get(aa, 0) + 1
            pc_val = store.pc[ancestor.row, col]
            pc_name = PHYSICOCHEMICAL_GROUPS[int(pc_val)] if pc_val >= 0 else None
            if freq:
                total = len(og_row_indices[j])
                freq = {k: v / total for k, v in freq.items()}
                freq_str = " ".join(
                    f"{seq3(aa)}:{pct * 100:.2f}%"
                    for aa, pct in sorted(freq.items(), key=lambda x: x[1], reverse=True)
                )
                entry = f"{pc_name} ({freq_str})"
            else:
                entry = pc_name
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
