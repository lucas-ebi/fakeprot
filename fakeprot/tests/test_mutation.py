"""Tests for evolution/mutation.py"""

import numpy as np
import pytest

from fakeprot.config import SimulationConfig
from fakeprot.evolution.mutation import (
    apply_gaps,
    make_mutant,
    mutation_rate_distribution,
    wave_shuffle,
)
from fakeprot.models.sequence import Sequence
from fakeprot.models.species import Species
from fakeprot.substitution import AMINO_ACIDS


@pytest.fixture
def config() -> SimulationConfig:
    return SimulationConfig(size=10, length=20, seed=42)


@pytest.fixture
def simple_species() -> Species:
    return Species(paralogs=[], label="sp0")


@pytest.fixture
def simple_sequence(simple_species: Species) -> Sequence:
    np.random.seed(0)
    rates = [0.0] + [0.1] * 19
    stereo = [None] * 20
    return Sequence("M" + "A" * 19, rates, stereo, simple_species, 0)


class TestWaveShuffle:
    def test_preserves_all_values(self):
        values = [0.1, 0.5, 0.3, 0.8, 0.2]
        result = wave_shuffle(values)
        assert sorted(result) == pytest.approx(sorted(values))

    def test_output_length_matches_input(self):
        values = list(range(10))
        assert len(wave_shuffle(values)) == 10

    def test_single_element(self):
        assert wave_shuffle([0.42]) == pytest.approx([0.42])

    def test_returns_list(self):
        assert isinstance(wave_shuffle([1.0, 2.0]), list)


class TestMutationRateDistribution:
    def test_correct_length(self, config: SimulationConfig):
        rates = mutation_rate_distribution(20, config)
        assert len(rates) == 20

    def test_first_rate_is_zero(self, config: SimulationConfig):
        rates = mutation_rate_distribution(20, config)
        assert rates[0] == 0.0

    def test_all_rates_non_negative(self, config: SimulationConfig):
        rates = mutation_rate_distribution(20, config)
        assert all(r >= 0.0 for r in rates)

    def test_duplication_mode_returns_correct_length(self, config: SimulationConfig):
        np.random.seed(1)
        rates = mutation_rate_distribution(20, config, duplication=True)
        assert len(rates) == 20

    def test_seeded_run_is_reproducible(self):
        c1 = SimulationConfig(size=10, length=20, seed=7)
        r1 = mutation_rate_distribution(20, c1)
        # Re-seed by constructing a second config with the same seed
        c2 = SimulationConfig(size=10, length=20, seed=7)
        r2 = mutation_rate_distribution(20, c2)
        assert r1 == pytest.approx(r2)


class TestMakeMutant:
    def test_produces_valid_amino_acids(self, simple_sequence: Sequence, config: SimulationConfig):
        new_host = Species(paralogs=[], label="sp1")
        child, gaps = make_mutant(simple_sequence, new_host, 0, config)
        valid = set(AMINO_ACIDS) | {"-"}
        for aa in child.sequence:
            assert aa in valid

    def test_child_label(self, simple_sequence: Sequence, config: SimulationConfig):
        new_host = Species(paralogs=[], label="spX")
        child, _ = make_mutant(simple_sequence, new_host, 0, config)
        assert child.label == "spX_seq1"

    def test_gaps_list_is_list_of_ints(self, simple_sequence: Sequence, config: SimulationConfig):
        new_host = Species(paralogs=[], label="sp1")
        _, gaps = make_mutant(simple_sequence, new_host, 0, config)
        assert isinstance(gaps, list)
        assert all(isinstance(g, int) for g in gaps)

    def test_duplication_mode(self, simple_sequence: Sequence, config: SimulationConfig):
        np.random.seed(5)
        new_host = Species(paralogs=[], label="sp_dup")
        child, _ = make_mutant(simple_sequence, new_host, 1, config, duplication=True)
        valid = set(AMINO_ACIDS) | {"-"}
        for aa in child.sequence:
            assert aa in valid


class TestApplyGaps:
    def test_inserts_gaps_correctly(self):
        sp = Species(paralogs=[], label="sp0")
        seq = Sequence("ACGT", [0.1, 0.2, 0.3, 0.4], [None] * 4, sp, 0)
        collection = [seq]
        n = apply_gaps(collection, [1])
        assert n == 1
        assert seq.sequence == "AC-GT"
        assert len(seq.mutation_rates) == 5
        assert len(seq.stereochemistry) == 5
        assert seq.mutation_rates[2] == 1.0
        assert seq.stereochemistry[2] is None

    def test_multiple_gaps_applied_in_reverse(self):
        sp = Species(paralogs=[], label="sp0")
        seq = Sequence("ABCDE", [0.1] * 5, [None] * 5, sp, 0)
        n = apply_gaps([seq], [0, 2])
        assert n == 2
        assert len(seq.sequence) == 7

    def test_no_gaps_is_noop(self):
        sp = Species(paralogs=[], label="sp0")
        seq = Sequence("ACGT", [0.1] * 4, [None] * 4, sp, 0)
        n = apply_gaps([seq], [])
        assert n == 0
        assert seq.sequence == "ACGT"
