"""
Tree construction and ortholog-group detection utilities.
"""

from __future__ import annotations

import string

import networkx as nx
from Bio import Phylo
from Bio.Phylo.BaseTree import Clade
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from fakeprot.models.sequence import Sequence
from fakeprot.models.species import Species


def build_msa(graph: nx.DiGraph, node: Sequence) -> list[SeqRecord]:
    """Collect leaf SeqRecord objects in tree traversal order."""
    if graph.out_degree(node) == 0:
        return []
    records: list[SeqRecord] = []
    for child in graph.successors(node):
        if graph.out_degree(child) == 0:
            records.append(SeqRecord(Seq(child.sequence), id=str(child)))
        else:
            records.extend(build_msa(graph, child))
    return records


def get_branch_length(
    graph: nx.DiGraph, node: Sequence, sequence_length: int
) -> float | None:
    """Branch length as the fraction of mismatched positions to the parent."""
    if graph.in_degree(node) == 0:
        return None
    parent = next(graph.predecessors(node))
    mismatches = sum(1 for x, y in zip(parent.sequence, node.sequence) if x != y)
    return mismatches / sequence_length


def build_gene_tree(
    graph: nx.DiGraph, node: Sequence, sequence_length: int
) -> Clade:
    """Recursively build a Clade tree from the sequence DAG, with branch lengths."""
    clades: list[Clade] = []
    for child in graph.successors(node):
        if graph.out_degree(child) == 0:
            clades.append(
                Clade(
                    name=str(child),
                    branch_length=get_branch_length(graph, child, sequence_length),
                )
            )
        else:
            clades.append(build_gene_tree(graph, child, sequence_length))
    return Clade(
        name=str(node),
        clades=clades,
        branch_length=get_branch_length(graph, node, sequence_length),
    )


def build_species_tree(graph: nx.DiGraph, node: Species) -> Clade:
    """Recursively build a Clade cladogram from the species DAG (no branch lengths)."""
    clades: list[Clade] = []
    for child in graph.successors(node):
        if graph.out_degree(child) == 0:
            clades.append(Clade(name=str(child)))
        else:
            clades.append(build_species_tree(graph, child))
    return Clade(name=str(node), clades=clades)


def og_label(idx: int) -> str:
    """Convert a zero-based ortholog-group index to a human-readable label (A, B, … Z, A1, B1 …)."""
    letter = string.ascii_uppercase[idx % 26]
    suffix = str(idx // 26) if idx >= 26 else ""
    return f"{letter}{suffix}"


def find_ortholog_groups(
    sequence_tree: nx.DiGraph,
    orthologs: list[Sequence],
) -> dict[int, list[Sequence]]:
    """
    Assign each leaf sequence to exactly one ortholog group.

    Traverses descendants of each ortholog anchor, stopping whenever
    another anchor is encountered (i.e. the boundary between OGs).
    """
    ortholog_set = set(orthologs)
    return {
        i: _descendants_until_ortholog(sequence_tree, ancestor, ortholog_set)
        for i, ancestor in enumerate(orthologs)
    }


def _descendants_until_ortholog(
    tree: nx.DiGraph,
    source: Sequence,
    ortholog_set: set[Sequence],
) -> list[Sequence]:
    """DFS from source, not crossing into other ortholog subtrees."""
    result: list[Sequence] = []
    stack = [source]
    while stack:
        node = stack.pop()
        for child in tree.successors(node):
            if child in ortholog_set and child is not source:
                continue
            if tree.out_degree(child) == 0:
                result.append(child)
            else:
                stack.append(child)
    return result
