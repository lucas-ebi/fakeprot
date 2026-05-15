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
from fakeprot.models.msa_store import MsaStore, decode_chars, encode_row
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
def simple_store_and_seq(simple_species: Species) -> tuple[MsaStore, Sequence]:
    np.random.seed(0)
    residues = list("M" + "A" * 19)
    rates = [0.0] + [0.1] * 19
    stereo = [None] * 20
    chars, rates_arr, pc_arr = encode_row(residues, rates, stereo)
    store = MsaStore(chars, rates_arr, pc_arr)
    seq = Sequence(row=0, host=simple_species, idx=0)
    return store, seq


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
        c2 = SimulationConfig(size=10, length=20, seed=7)
        r2 = mutation_rate_distribution(20, c2)
        assert r1 == pytest.approx(r2)


class TestMakeMutant:
    def test_produces_valid_amino_acids(
        self, simple_store_and_seq: tuple[MsaStore, Sequence], config: SimulationConfig
    ):
        store, parent = simple_store_and_seq
        child, _ = store.commit_child(*make_mutant(store, parent, config),
                                      Species(paralogs=[], label="sp1"), 0)
        valid = set(AMINO_ACIDS) | {"-"}
        for aa in decode_chars(store.chars[child.row]):
            assert aa in valid

    def test_child_label(
        self, simple_store_and_seq: tuple[MsaStore, Sequence], config: SimulationConfig
    ):
        store, parent = simple_store_and_seq
        new_host = Species(paralogs=[], label="spX")
        child, _ = store.commit_child(*make_mutant(store, parent, config), new_host, 0)
        assert child.label == "spX_seq1"

    def test_gaps_list_is_list_of_ints(
        self, simple_store_and_seq: tuple[MsaStore, Sequence], config: SimulationConfig
    ):
        store, parent = simple_store_and_seq
        _, _, _, gaps = make_mutant(store, parent, config)
        assert isinstance(gaps, list)
        assert all(isinstance(g, int) for g in gaps)

    def test_duplication_mode(
        self, simple_store_and_seq: tuple[MsaStore, Sequence], config: SimulationConfig
    ):
        store, parent = simple_store_and_seq
        np.random.seed(5)
        child, _ = store.commit_child(*make_mutant(store, parent, config, duplication=True),
                                      Species(paralogs=[], label="sp_dup"), 1)
        valid = set(AMINO_ACIDS) | {"-"}
        for aa in decode_chars(store.chars[child.row]):
            assert aa in valid


class TestApplyGaps:
    def _make_store(self, seq_str: str, rates: list[float], stereo: list) -> MsaStore:
        chars, rates_arr, pc_arr = encode_row(list(seq_str), rates, stereo)
        return MsaStore(chars, rates_arr, pc_arr)

    def test_inserts_gap_correctly(self):
        store = self._make_store("ACGT", [0.1, 0.2, 0.3, 0.4], [None] * 4)
        n = apply_gaps(store, [1])
        assert n == 1
        assert decode_chars(store.chars[0]) == "AC-GT"
        assert store.rates[0, 2] == pytest.approx(1.0)
        assert store.pc[0, 2] == -1

    def test_multiple_gaps(self):
        store = self._make_store("ACGDE", [0.1] * 5, [None] * 5)
        n = apply_gaps(store, [0, 2])
        assert n == 2
        assert store.chars.shape[1] == 7

    def test_no_gaps_is_noop(self):
        store = self._make_store("ACGT", [0.1] * 4, [None] * 4)
        n = apply_gaps(store, [])
        assert n == 0
        assert decode_chars(store.chars[0]) == "ACGT"

    def test_all_rows_updated(self):
        chars1, r1, p1 = encode_row(list("ACGT"), [0.1] * 4, [None] * 4)
        store = MsaStore(chars1, r1, p1)
        chars2, r2, p2 = encode_row(list("MRND"), [0.2] * 4, [None] * 4)
        store.add_row(chars2, r2, p2)
        apply_gaps(store, [1])
        assert store.n_rows == 2
        assert store.length == 5
        assert decode_chars(store.chars[0]) == "AC-GT"
        assert decode_chars(store.chars[1]) == "MR-ND"
