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
    def test_zero_rates_and_no_gaps_preserve_parent(
        self, simple_species: Species
    ):
        residues = list("MARNDCQEGH")
        chars, rates, pc = encode_row(residues, [0.0] * len(residues), [None] * len(residues))
        store = MsaStore(chars, rates, pc)
        parent = Sequence(row=0, host=simple_species, idx=0)
        config = SimulationConfig(size=10, length=len(residues), p_del=0.0, p_ins=0.0, seed=3)

        child_chars, child_rates, child_pc, gaps = make_mutant(store, parent, config)

        assert gaps == []
        assert decode_chars(child_chars) == "".join(residues)
        assert child_rates.tolist() == pytest.approx(rates.tolist())
        assert child_pc.tolist() == pc.tolist()

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

    def test_existing_gaps_are_preserved_when_gap_probability_is_zero(
        self, simple_species: Species
    ):
        chars, rates, pc = encode_row(list("MA--RN"), [0.2] * 6, [None] * 6)
        store = MsaStore(chars, rates, pc)
        parent = Sequence(row=0, host=simple_species, idx=0)
        config = SimulationConfig(size=10, length=6, p_del=0.0, p_ins=0.0, seed=11)

        child_chars, child_rates, child_pc, gaps = make_mutant(store, parent, config)

        assert gaps == []
        assert len(child_chars) == len(chars)
        assert len(child_rates) == len(rates)
        assert len(child_pc) == len(pc)
        assert decode_chars(child_chars)[2:4] == "--"

    def test_output_arrays_have_matching_lengths(
        self, simple_store_and_seq: tuple[MsaStore, Sequence]
    ):
        store, parent = simple_store_and_seq
        config = SimulationConfig(size=10, length=20, p_del=0.25, p_ins=0.05, seed=13)

        child_chars, child_rates, child_pc, _ = make_mutant(store, parent, config)

        assert len(child_chars) == len(child_rates) == len(child_pc)

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

    def test_explicit_branch_length_drives_substitution_rate(
        self, simple_species: Species
    ):
        """Explicit branch_length overrides config and controls divergence."""
        length = 200
        residues = ["M"] + ["A"] * (length - 1)
        rates = [0.0] + [1.0] * (length - 1)
        chars, rates_arr, pc_arr = encode_row(residues, rates, [None] * length)
        store = MsaStore(chars, rates_arr, pc_arr)
        parent = Sequence(row=0, host=simple_species, idx=0)
        # config.branch_length is 0.10 — deliberately between the two t values so that
        # fallback to config would produce ~18% substitution for both, making 5× assert fail
        cfg = SimulationConfig(size=10, length=length, branch_length=0.10,
                               p_del=0.0, p_ins=0.0, seed=99)

        def count_subs(t: float) -> int:
            child_chars, _, _, _ = make_mutant(store, parent, cfg, branch_length=t)
            return int((child_chars != chars).sum())

        n = 300
        mean_low  = np.mean([count_subs(0.01) for _ in range(n)])
        mean_high = np.mean([count_subs(0.20) for _ in range(n)])
        # 1 - exp(-0.20) ≈ 18 × (1 - exp(-0.01)); a 5× margin is very conservative
        assert mean_high > 5 * mean_low


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
