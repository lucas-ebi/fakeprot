"""
Mutation, indel, and rate-distribution logic for sequence evolution.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma

from fakeprot.config import SimulationConfig
from fakeprot.models.msa_store import MsaStore, decode_pc, encode_row
from fakeprot.models.sequence import Sequence
from fakeprot.substitution import (
    AA_FREQUENCIES,
    AA_INDEX,
    AMINO_ACIDS,
    CHAR_GAP,
    PC_FREQUENCIES,
    PC_MASKS,
    PC_SUBS_MATRIX,
    PHYSICOCHEMICAL_GROUPS,
    PHYSICOCHEMICAL_SETS,
    WAG_MATRIX,
)

ROOT_AMINO_ACID = "M"
ROOT_PC_GROUP = "With sulfur"  # Met belongs to the sulfur group


def wave_shuffle(values: list[float]) -> list[float]:
    """
    Arrange values so adjacent elements are similar in magnitude.

    Starts at a random element, then greedily picks the closest remaining
    value at each step. Weights are proportional to (1 - |current - candidate|),
    clamped to zero to guard against gamma PDF ranges that exceed 1.
    """
    n = len(values)
    avail = np.asarray(values, dtype=float).copy()
    result = np.empty(n, dtype=float)
    m = n

    start = np.random.randint(n)
    result[0] = avail[start]
    avail[start] = avail[m - 1]
    m -= 1

    for step in range(1, n):
        distances = np.abs(avail[:m] - result[step - 1])
        weights = np.maximum(0.0, 1.0 - distances)
        total = weights.sum()
        if total == 0.0:
            chosen = np.random.randint(m)
        else:
            chosen = np.random.choice(m, p=weights / total)
        result[step] = avail[chosen]
        avail[chosen] = avail[m - 1]
        m -= 1

    return result.tolist()


def mutation_rate_distribution(
    length: int, config: SimulationConfig, duplication: bool = False
) -> list[float]:
    """
    Draw per-site mutation rates from a gamma distribution.

    In duplication mode the shape and scale are perturbed (must be >= 1),
    modelling the relaxed constraint on a newly duplicated paralog.
    The first position always gets rate 0 (conserved Met start).
    """
    if duplication:
        shape = 0.0
        while shape < 1.0:
            shape = np.random.normal(config.gamma_shape, 1.0)
        scale = 0.0
        while scale < 1.0:
            scale = np.random.normal(config.gamma_scale, 1.0)
    else:
        shape, scale = config.gamma_shape, config.gamma_scale

    x = np.linspace(gamma.ppf(0.01, shape), gamma.ppf(0.99, shape), length - 1)
    rates = gamma.pdf(x, shape, scale=scale).tolist()
    return [0.0] + wave_shuffle(rates)


def make_mutant(
    store: MsaStore,
    parent: Sequence,
    config: SimulationConfig,
    duplication: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Compute child sequence arrays from parent without committing to the store.

    Returns (chars, rates, pc, gaps).  Caller must:
      1. apply_gaps(store, gaps)   — extend existing rows to the new length
      2. row = store.add_row(chars, rates, pc)   — append child at that length
      3. child = Sequence(row, new_host, new_idx)
    """
    residues, rates, stereo = _prepare_evolution_state(store, parent, config, duplication)
    new_seq, new_rates, new_stereo, gaps = _apply_mutations_and_indels(
        residues, rates, stereo, config
    )
    chars, rates_arr, pc_arr = encode_row(list(new_seq), new_rates, new_stereo)
    return chars, rates_arr, pc_arr, gaps


def apply_gaps(store: MsaStore, gaps: list[int]) -> int:
    """
    Insert gap columns into every sequence to preserve alignment after insertions.

    Delegates to MsaStore.insert_gaps which uses np.insert across all rows at once.
    Returns the number of columns added.
    """
    return store.insert_gaps(gaps)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _prepare_evolution_state(
    store: MsaStore,
    parent: Sequence,
    config: SimulationConfig,
    duplication: bool,
) -> tuple[list[str], list[float], list[str | None]]:
    """
    Return (residues, rates, stereochemistry) ready for mutation.

    In duplication mode: re-draws rates and updates physicochemical constraints,
    reflecting relaxed purifying selection on the new copy.
    """
    chars_row = store.chars[parent.row]
    rates     = store.rates[parent.row].tolist()
    stereo    = decode_pc(store.pc[parent.row])
    residues  = [AMINO_ACIDS[int(c)] if c < CHAR_GAP else "-" for c in chars_row]

    if not duplication:
        return residues, rates, stereo

    # Re-draw rates, preserving rank order from the parent
    new_rates = mutation_rate_distribution(len(residues), config, duplication=True)
    ranked_positions = sorted(range(len(residues)), key=lambda i: rates[i])
    sorted_new = sorted(new_rates)
    reranked = dict(zip(ranked_positions, sorted_new))
    current_rates = [reranked[i] for i in range(len(residues))]

    new_stereo: list[str | None] = [stereo[0]]
    for i in range(1, len(residues)):
        rate = current_rates[i]
        if np.random.random() < (1.0 - rate):
            if stereo[i] is None:
                new_sc = np.random.choice(PHYSICOCHEMICAL_GROUPS, p=PC_FREQUENCIES)
            else:
                p_sc = {k: rate * v for k, v in PC_SUBS_MATRIX[stereo[i]].items()}
                p_sc[stereo[i]] = 1.0 - rate
                keys = list(p_sc.keys())
                vals = np.array(list(p_sc.values()))
                new_sc = np.random.choice(keys, p=vals / vals.sum())
            new_stereo.append(new_sc)
            if residues[i] not in PHYSICOCHEMICAL_SETS[new_sc]:
                mask = PC_MASKS[new_sc]
                if residues[i] == "-":
                    p = AA_FREQUENCIES * mask
                else:
                    p = WAG_MATRIX[AA_INDEX[residues[i]]] * mask
                p = p / p.sum()
                residues[i] = AMINO_ACIDS[np.random.choice(20, p=p)]
        else:
            new_stereo.append(None)

    return residues, current_rates, new_stereo


def _sample_aa_unconstrained(rate: float, aa_idx: int) -> str:
    """Draw a new amino acid using WAG probabilities with no physicochemical group constraint."""
    p = WAG_MATRIX[aa_idx] * rate
    p[aa_idx] = 1.0 - rate
    return AMINO_ACIDS[np.random.choice(20, p=p / p.sum())]


def _sample_aa_constrained(rate: float, aa_idx: int, group: str) -> str:
    """Draw a new amino acid restricted to a physicochemical group by WAG probabilities."""
    p = WAG_MATRIX[aa_idx] * PC_MASKS[group]
    p[aa_idx] = 0.0
    total = p.sum()
    if total > 0.0:
        p *= rate / total
    p[aa_idx] = 1.0 - rate
    return AMINO_ACIDS[np.random.choice(20, p=p / p.sum())]


def _apply_mutations_and_indels(
    residues: list[str],
    rates: list[float],
    stereo: list[str | None],
    config: SimulationConfig,
) -> tuple[str, list[float], list[str | None], list[int]]:
    """
    Walk each site of the parent sequence and apply substitutions, deletions,
    and insertions according to the WAG model and gap probability.

    Returns (new_sequence_str, new_rates, new_stereo, gap_positions).
    """
    new_seq: list[str] = []
    new_rates: list[float] = []
    new_stereo: list[str | None] = []
    gaps: list[int] = []
    i = 0

    while i < len(residues):
        if residues[i] == "-":
            i, new_seq, new_rates, new_stereo = _handle_gap_region(
                i, residues, rates, stereo, new_seq, new_rates, new_stereo, config
            )
        else:
            i, new_seq, new_rates, new_stereo, gaps = _handle_residue(
                i, residues, rates, stereo, new_seq, new_rates, new_stereo, gaps, config
            )

    return "".join(new_seq), new_rates, new_stereo, gaps


def _handle_gap_region(
    i: int,
    residues: list[str],
    rates: list[float],
    stereo: list[str | None],
    new_seq: list[str],
    new_rates: list[float],
    new_stereo: list[str | None],
    config: SimulationConfig,
) -> tuple[int, list[str], list[float], list[str | None]]:
    """
    Process a contiguous run of gap characters.

    Each position has a geometrically decreasing probability of being filled.
    Once one gap remains, all subsequent positions in the run stay as gaps.
    """
    start = i
    end = len(residues)
    while i < end:
        if residues[i] != "-":
            end = i
        i += 1
    i = start

    j = 0
    done = False
    while i < end:
        if done:
            new_seq.append("-")
        else:
            p_fill = config.p_gap * (2.0 ** (-float(j)))
            if np.random.random() < p_fill:
                if stereo[i] is None:
                    aa = AMINO_ACIDS[np.random.choice(20, p=AA_FREQUENCIES)]
                else:
                    mask = PC_MASKS[stereo[i]]
                    p = AA_FREQUENCIES * mask
                    aa = AMINO_ACIDS[np.random.choice(20, p=p / p.sum())]
                new_seq.append(aa)
            else:
                new_seq.append("-")
                done = True
        new_rates.append(rates[i])
        new_stereo.append(stereo[i])
        i += 1
        j += 1

    return i, new_seq, new_rates, new_stereo


def _handle_residue(
    i: int,
    residues: list[str],
    rates: list[float],
    stereo: list[str | None],
    new_seq: list[str],
    new_rates: list[float],
    new_stereo: list[str | None],
    gaps: list[int],
    config: SimulationConfig,
) -> tuple[int, list[str], list[float], list[str | None], list[int]]:
    """
    Process a single non-gap residue: possibly delete it, substitute it,
    then possibly insert one or more new residues immediately after it.
    """
    rate = rates[i]
    p_delete = config.p_gap * rate

    if np.random.random() < p_delete:
        new_seq.append("-")
        new_rates.append(rate)
        new_stereo.append(stereo[i])
    else:
        aa_idx = AA_INDEX[residues[i]]
        if stereo[i] is None:
            new_aa = _sample_aa_unconstrained(rate, aa_idx)
        else:
            new_aa = _sample_aa_constrained(rate, aa_idx, stereo[i])
        new_seq.append(new_aa)
        new_rates.append(rate)
        new_stereo.append(stereo[i])

        j = 0
        done = False
        while not done:
            p_insert = rate * config.p_gap * (2.0 ** (-float(j)))
            if np.random.random() < p_insert:
                inserted_aa = AMINO_ACIDS[np.random.choice(20, p=AA_FREQUENCIES)]
                new_seq.append(inserted_aa)
                new_rates.append(1.0)
                new_stereo.append(None)
                insertion = False
                if i < len(residues) - 1:
                    if residues[i + 1] == "-":
                        i += 1
                    else:
                        insertion = True
                else:
                    insertion = True
                if insertion:
                    gaps.append(i)
            else:
                done = True
            j += 1

    i += 1
    return i, new_seq, new_rates, new_stereo, gaps
