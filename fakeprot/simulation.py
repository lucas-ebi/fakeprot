"""
Core simulation loop: builds the species and sequence trees by alternating
gene duplication and speciation events until the target leaf count is reached.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from fakeprot.config import SimulationConfig
from fakeprot.evolution.mutation import (
    ROOT_AMINO_ACID,
    ROOT_STEREOCHEMISTRY,
    apply_gaps,
    make_mutant,
    mutation_rate_distribution,
)
from fakeprot.io.output import write_outputs
from fakeprot.models.sequence import Sequence
from fakeprot.models.species import Species
from fakeprot.substitution import (
    AA_FREQUENCIES,
    AMINO_ACIDS,
    SC_FREQUENCIES,
    SC_MASKS,
    STEREOCHEMICAL_GROUPS,
    STEREOCHEMICAL_SETS,
)


def run(config: SimulationConfig) -> None:
    """Execute a full simulation and write all output files."""
    root_species = Species(paralogs=[], label="sp0")
    species_tree: nx.DiGraph = nx.DiGraph()
    sequence_tree: nx.DiGraph = nx.DiGraph()

    mutation_rates = mutation_rate_distribution(config.length, config)
    root_seq_str, root_stereo = _generate_root_sequence(config.length, mutation_rates)

    root_sequence = Sequence(root_seq_str, mutation_rates, root_stereo, root_species, 0)
    collection: list[Sequence] = [root_sequence]
    root_species.paralogs.append(root_sequence)
    species_tree.add_node(root_species)

    orthologs: list[Sequence] = [root_sequence]
    sequence_length = config.length

    current_species = list(species_tree.nodes())
    leaves = [n for n in sequence_tree.nodes() if sequence_tree.out_degree(n) == 0]

    while len(leaves) < config.size:
        if len(current_species) == 1:
            chosen_idx = 0
        else:
            chosen_idx = np.random.randint(len(current_species))
        sp = current_species[chosen_idx]

        sequence_length = _do_gene_duplication(
            sp, collection, sequence_tree, orthologs, current_species, config, sequence_length
        )
        sequence_length = _do_speciation(
            sp, collection, sequence_tree, species_tree, config, sequence_length
        )

        current_species = [n for n in species_tree.nodes() if species_tree.out_degree(n) == 0]
        leaves = [n for n in sequence_tree.nodes() if sequence_tree.out_degree(n) == 0]

    if len(orthologs) < config.n_orthologs:
        print(
            f"Warning: too few speciation events to produce {config.n_orthologs} ortholog groups."
        )

    write_outputs(
        config,
        collection,
        sequence_tree,
        species_tree,
        root_sequence,
        root_species,
        orthologs,
        sequence_length,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _generate_root_sequence(
    length: int,
    mutation_rates: list[float],
) -> tuple[str, list[str | None]]:
    """
    Build the ancestral root sequence.

    Position 0 is always Met (all natural proteins start with Met).
    Each subsequent site either acquires a stereochemical class (if its mutation
    rate is low) or is drawn freely from background frequencies.
    """
    sequence = ROOT_AMINO_ACID
    stereochemistry: list[str | None] = [ROOT_STEREOCHEMISTRY]

    for i in range(1, length):
        rate = mutation_rates[i]
        if np.random.choice((True, False), p=(1.0 - rate, rate)):
            sc = np.random.choice(STEREOCHEMICAL_GROUPS, p=SC_FREQUENCIES)
            stereochemistry.append(sc)
            mask = SC_MASKS[sc]
            p = AA_FREQUENCIES * mask
            sequence += AMINO_ACIDS[np.random.choice(20, p=p / p.sum())]
        else:
            stereochemistry.append(None)
            sequence += AMINO_ACIDS[np.random.choice(20, p=AA_FREQUENCIES)]

    return sequence, stereochemistry


def _do_speciation(
    parent: Species,
    collection: list[Sequence],
    sequence_tree: nx.DiGraph,
    species_tree: nx.DiGraph,
    config: SimulationConfig,
    sequence_length: int,
) -> int:
    """
    Split one species into two daughter species, mutating each paralog independently.

    Returns the updated sequence length (may grow if insertions occurred).
    """
    n_nodes = len(species_tree.nodes())
    daughter_a = Species(paralogs=[], label=f"sp{n_nodes}")
    daughter_b = Species(paralogs=[], label=f"sp{n_nodes + 1}")

    for paralog_idx, paralog in enumerate(parent.paralogs):
        child_a, gaps = make_mutant(paralog, daughter_a, paralog_idx, config)
        sequence_length += apply_gaps(collection, gaps)
        collection.append(child_a)
        daughter_a.paralogs.append(child_a)
        sequence_tree.add_edge(paralog, child_a)

        child_b, gaps = make_mutant(paralog, daughter_b, paralog_idx, config)
        sequence_length += apply_gaps(collection, gaps)
        collection.append(child_b)
        daughter_b.paralogs.append(child_b)
        sequence_tree.add_edge(paralog, child_b)

    species_tree.add_edge(parent, daughter_a)
    species_tree.add_edge(parent, daughter_b)

    return sequence_length


def _do_gene_duplication(
    species: Species,
    collection: list[Sequence],
    sequence_tree: nx.DiGraph,
    orthologs: list[Sequence],
    current_species: list[Species],
    config: SimulationConfig,
    sequence_length: int,
) -> int:
    """
    Possibly duplicate one paralog within a species to create a new ortholog group.

    The probability of duplication increases as the target number of orthologs
    has not yet been reached. Returns the updated sequence length.
    """
    if config.n_orthologs <= 1 or len(orthologs) >= config.n_orthologs:
        return sequence_length

    p = 2.0 ** (-float(config.n_orthologs - len(orthologs)) / float(len(current_species)))
    if not np.random.choice((True, False), p=(p, 1.0 - p)):
        return sequence_length

    source_idx = (
        0 if len(current_species) == 1 else np.random.randint(len(species.paralogs))
    )
    source = species.paralogs[source_idx]
    new_idx = len(species.paralogs)

    duplicate, gaps = make_mutant(source, species, new_idx, config, duplication=True)
    sequence_length += apply_gaps(collection, gaps)
    collection.append(duplicate)
    species.paralogs.append(duplicate)
    orthologs.append(duplicate)
    sequence_tree.add_edge(source, duplicate)

    return sequence_length
