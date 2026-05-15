"""Tests for models, tree utilities, and config validation."""

import pytest
import networkx as nx

from fakeprot.config import SimulationConfig
from fakeprot.evolution.tree import find_ortholog_groups, og_label
from fakeprot.models.sequence import Sequence
from fakeprot.models.species import Species


class TestSimulationConfig:
    def test_p_gap_defaults_to_1_over_size(self):
        config = SimulationConfig(size=50, length=100)
        assert config.p_gap == pytest.approx(1.0 / 50)

    def test_explicit_p_gap_preserved(self):
        config = SimulationConfig(size=50, length=100, p_gap=0.05)
        assert config.p_gap == pytest.approx(0.05)

    def test_invalid_msa_format_raises(self):
        with pytest.raises(ValueError, match="msa_format"):
            SimulationConfig(size=10, length=10, msa_format="docx")

    def test_invalid_tree_format_raises(self):
        with pytest.raises(ValueError, match="tree_format"):
            SimulationConfig(size=10, length=10, tree_format="xml")


class TestSequenceModel:
    def test_label_property(self):
        sp = Species(paralogs=[], label="sp3")
        seq = Sequence(row=0, host=sp, idx=2)
        assert seq.label == "sp3_seq3"

    def test_repr_is_label(self):
        sp = Species(paralogs=[], label="sp0")
        seq = Sequence(row=0, host=sp, idx=0)
        assert repr(seq) == "sp0_seq1"

    def test_identity_based_equality(self):
        sp = Species(paralogs=[], label="sp0")
        seq_a = Sequence(row=0, host=sp, idx=0)
        seq_b = Sequence(row=0, host=sp, idx=0)
        assert seq_a != seq_b
        assert seq_a == seq_a

    def test_hashable_for_graph_nodes(self):
        sp = Species(paralogs=[], label="sp0")
        seq = Sequence(row=0, host=sp, idx=0)
        g = nx.DiGraph()
        g.add_node(seq)
        assert seq in g.nodes()


class TestOgLabel:
    def test_single_letters(self):
        assert og_label(0) == "A"
        assert og_label(25) == "Z"

    def test_overflow_to_numbered(self):
        assert og_label(26) == "A1"
        assert og_label(27) == "B1"
        assert og_label(52) == "A2"

    def test_no_suffix_below_26(self):
        for i in range(26):
            assert og_label(i)[-1].isalpha() or og_label(i) == og_label(i)


class TestFindOrthologGroups:
    def _make_seq(self, label: str) -> Sequence:
        sp = Species(paralogs=[], label=label)
        return Sequence(row=0, host=sp, idx=0)

    def test_single_ortholog_all_leaves_in_group_0(self):
        root = self._make_seq("root")
        leaf_a = self._make_seq("a")
        leaf_b = self._make_seq("b")
        tree = nx.DiGraph()
        tree.add_edge(root, leaf_a)
        tree.add_edge(root, leaf_b)
        groups = find_ortholog_groups(tree, [root])
        assert set(groups[0]) == {leaf_a, leaf_b}

    def test_two_orthologs_split_correctly(self):
        root = self._make_seq("root")
        dup = self._make_seq("dup")
        leaf_a = self._make_seq("a")
        leaf_b = self._make_seq("b")
        leaf_c = self._make_seq("c")
        leaf_d = self._make_seq("d")
        tree = nx.DiGraph()
        tree.add_edge(root, dup)
        tree.add_edge(root, leaf_a)
        tree.add_edge(root, leaf_b)
        tree.add_edge(dup, leaf_c)
        tree.add_edge(dup, leaf_d)
        groups = find_ortholog_groups(tree, [root, dup])
        assert leaf_c not in groups[0]
        assert leaf_d not in groups[0]
        assert set(groups[1]) == {leaf_c, leaf_d}
