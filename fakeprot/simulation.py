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
    ROOT_PC_GROUP,
    make_mutant,
    mutation_rate_distribution,
)
from fakeprot.io.output import write_outputs
from fakeprot.models.msa_store import MsaStore
from fakeprot.models.sequence import Sequence
from fakeprot.models.species import Species
from fakeprot.substitution import (
    AA_FREQUENCIES,
    AA_INDEX,
    PC_FREQUENCIES,
    PC_INDEX,
    PC_MASKS,
    PC_NONE,
    PHYSICOCHEMICAL_GROUPS,
)


def run(config: SimulationConfig) -> None:
    """Execute a full simulation and write all output files."""
    root_species = Species(paralogs=[], label="sp0")
    species_tree: nx.DiGraph = nx.DiGraph()
    sequence_tree: nx.DiGraph = nx.DiGraph()

    mutation_rates = mutation_rate_distribution(config.length, config)
    root_chars, root_rates, root_pc = _generate_root_sequence(config.length, mutation_rates)

    store = MsaStore(root_chars, root_rates, root_pc)
    root_sequence = Sequence(row=0, host=root_species, idx=0)
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
            store, sp, collection, sequence_tree,
            orthologs, current_species, config, sequence_length,
        )
        sequence_length = _do_speciation(
            store, sp, collection, sequence_tree, species_tree, config, sequence_length
        )

        current_species = [n for n in species_tree.nodes() if species_tree.out_degree(n) == 0]
        leaves = [n for n in sequence_tree.nodes() if sequence_tree.out_degree(n) == 0]

    if len(orthologs) < config.n_orthologs:
        print(
            f"Warning: too few speciation events to produce {config.n_orthologs} ortholog groups."
        )

    write_outputs(
        config,
        store,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the ancestral root sequence directly as numpy arrays.

    Position 0 is always Met (all natural proteins start with Met).
    Each subsequent site either acquires a physicochemical group (if its mutation
    rate is low) or is drawn freely from background frequencies.
    """
    chars = np.empty(length, dtype=np.uint8)
    rates = np.array(mutation_rates, dtype=np.float32)
    pc    = np.full(length, PC_NONE, dtype=np.int8)

    chars[0] = AA_INDEX[ROOT_AMINO_ACID]
    pc[0]    = PC_INDEX[ROOT_PC_GROUP]

    for i in range(1, length):
        rate = mutation_rates[i]
        if np.random.random() < (1.0 - rate):
            sc = np.random.choice(PHYSICOCHEMICAL_GROUPS, p=PC_FREQUENCIES)
            pc[i] = PC_INDEX[sc]
            mask = PC_MASKS[sc]
            p = AA_FREQUENCIES * mask
            chars[i] = np.random.choice(20, p=p / p.sum())
        else:
            chars[i] = np.random.choice(20, p=AA_FREQUENCIES)

    return chars, rates, pc


def _do_speciation(
    store: MsaStore,
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
        child_a, n = store.commit_child(
            *make_mutant(store, paralog, config), daughter_a, paralog_idx
        )
        sequence_length += n
        collection.append(child_a)
        daughter_a.paralogs.append(child_a)
        sequence_tree.add_edge(paralog, child_a)

        child_b, n = store.commit_child(
            *make_mutant(store, paralog, config), daughter_b, paralog_idx
        )
        sequence_length += n
        collection.append(child_b)
        daughter_b.paralogs.append(child_b)
        sequence_tree.add_edge(paralog, child_b)

    species_tree.add_edge(parent, daughter_a)
    species_tree.add_edge(parent, daughter_b)

    return sequence_length


def _do_gene_duplication(
    store: MsaStore,
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
    if np.random.random() >= p:
        return sequence_length

    source_idx = (
        0 if len(current_species) == 1 else np.random.randint(len(species.paralogs))
    )
    source = species.paralogs[source_idx]
    new_idx = len(species.paralogs)

    duplicate, n = store.commit_child(
        *make_mutant(store, source, config, duplication=True), species, new_idx
    )
    sequence_length += n
    collection.append(duplicate)
    species.paralogs.append(duplicate)
    orthologs.append(duplicate)
    sequence_tree.add_edge(source, duplicate)

    return sequence_length
