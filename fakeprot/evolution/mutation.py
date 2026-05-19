"""
Mutation, indel, and rate-distribution logic for sequence evolution.
"""

from __future__ import annotations

import numpy as np

from fakeprot.config import SimulationConfig
from fakeprot.models.msa_store import MsaStore, decode_chars, decode_pc, encode_row
from fakeprot.models.sequence import Sequence
from fakeprot.substitution import (
    AA_FREQUENCY_CDF,
    AA_INDEX,
    AMINO_ACIDS,
    CHAR_GAP,
    PC_AA_CDFS,
    PC_FREQUENCIES,
    PC_NONE,
    PC_SUBS_MATRIX,
    PC_WAG_CDFS,
    PC_WAG_TOTALS,
    PHYSICOCHEMICAL_GROUPS,
    PHYSICOCHEMICAL_SETS,
    WAG_CDFS,
)

ROOT_AMINO_ACID = "M"
ROOT_PC_GROUP = "With sulfur"  # Met belongs to the sulfur group

# Probability of extending an insertion run by one more residue.
# Decoupled from p_ins so run length ~ Geometric(1 - P_INS_EXTEND), mean ≈ 2.
P_INS_EXTEND = 0.5


def _sample_from_cdf(cdf: np.ndarray) -> int:
    """Sample an index from a cumulative distribution."""
    return int(np.searchsorted(cdf, np.random.random(), side="right"))


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
    else:
        shape = config.gamma_shape

    rates = np.random.gamma(shape, 1.0 / shape, size=length - 1).tolist()
    return [0.0] + wave_shuffle(rates)


def make_mutant(
    store: MsaStore,
    parent: Sequence,
    config: SimulationConfig,
    duplication: bool = False,
    branch_length: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Compute child sequence arrays from parent without committing to the store.

    Returns (chars, rates, pc, gaps).  Caller must:
      1. apply_gaps(store, gaps)   — extend existing rows to the new length
      2. row = store.add_row(chars, rates, pc)   — append child at that length
      3. child = Sequence(row, new_host, new_idx)
    """
    t = branch_length if branch_length is not None else config.branch_length
    if duplication:
        residues, rates, stereo = _prepare_evolution_state(
            store, parent, config, duplication=True, branch_length=t
        )
        chars, rates_arr, pc_arr = encode_row(residues, rates, stereo)
    else:
        chars = store.chars[parent.row]
        rates_arr = store.rates[parent.row]
        pc_arr = store.pc[parent.row]

    return _apply_mutations_and_indels(chars, rates_arr, pc_arr, config, t)


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
    branch_length: float = 0.0,
) -> tuple[list[str], list[float], list[str | None]]:
    """
    Return (residues, rates, stereochemistry) ready for mutation.

    In duplication mode: re-draws rates and updates physicochemical constraints,
    reflecting relaxed purifying selection on the new copy.
    """
    chars_row = store.chars[parent.row]
    rates = store.rates[parent.row].tolist()
    stereo = decode_pc(store.pc[parent.row])
    residues = list(decode_chars(chars_row))

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
        p_conserved = np.exp(-branch_length * rate)
        if np.random.random() < p_conserved:
            if stereo[i] is None:
                new_sc = np.random.choice(PHYSICOCHEMICAL_GROUPS, p=PC_FREQUENCIES)
            else:
                p_change = 1.0 - p_conserved
                p_sc = {k: p_change * v for k, v in PC_SUBS_MATRIX[stereo[i]].items()}
                p_sc[stereo[i]] = p_conserved
                keys = list(p_sc.keys())
                vals = np.array(list(p_sc.values()))
                new_sc = np.random.choice(keys, p=vals / vals.sum())
            new_stereo.append(new_sc)
            if residues[i] not in PHYSICOCHEMICAL_SETS[new_sc]:
                if residues[i] == "-":
                    cdf = PC_AA_CDFS[new_sc]
                else:
                    cdf = PC_WAG_CDFS[new_sc][AA_INDEX[residues[i]]]
                residues[i] = AMINO_ACIDS[_sample_from_cdf(cdf)]
        else:
            new_stereo.append(None)

    return residues, current_rates, new_stereo


def _sample_from_cdf_rows(cdf_rows: np.ndarray) -> np.ndarray:
    """Sample one amino-acid index from each CDF row."""
    r = np.random.random(len(cdf_rows))
    return (cdf_rows <= r[:, None]).sum(axis=1).clip(0, 19).astype(np.uint8)


def _sample_gap_fill(pc_value: int) -> int:
    """Sample a residue for an existing gap slot, respecting its PC constraint."""
    if pc_value == PC_NONE:
        return _sample_from_cdf(AA_FREQUENCY_CDF)
    group = PHYSICOCHEMICAL_GROUPS[pc_value]
    return _sample_from_cdf(PC_AA_CDFS[group])


def _mutation_decisions(
    chars: np.ndarray,
    rates: np.ndarray,
    branch_length: float,
    p_del: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw deletion and substitution masks for all non-gap sites."""
    rates_f = rates.astype(np.float64, copy=False)
    is_gap = chars == CHAR_GAP
    active = ~is_gap

    rand_del = np.random.random(len(chars))
    rand_sub = np.random.random(len(chars))

    deleted = active & (rand_del < rates_f * p_del)
    surviving = active & ~deleted
    p_mut = 1.0 - np.exp(-branch_length * rates_f)
    will_mutate = surviving & (rand_sub < p_mut)
    return is_gap, deleted, will_mutate


def _apply_substitutions(
    chars: np.ndarray,
    pc: np.ndarray,
    will_mutate: np.ndarray,
) -> np.ndarray:
    """Apply vectorised WAG substitutions to the selected non-gap sites."""
    new_chars = chars.copy()

    unconstrained = will_mutate & (pc == PC_NONE)
    if unconstrained.any():
        idx = np.where(unconstrained)[0]
        new_chars[idx] = _sample_from_cdf_rows(WAG_CDFS[chars[idx]])

    constrained = will_mutate & (pc >= 0)
    if constrained.any():
        constrained_idx = np.where(constrained)[0]
        constrained_pc = pc[constrained_idx]
        for group_id in np.unique(constrained_pc):
            group_name = PHYSICOCHEMICAL_GROUPS[int(group_id)]
            selected = constrained_idx[constrained_pc == group_id]
            source_chars = chars[selected]
            can_mutate = PC_WAG_TOTALS[group_name][source_chars] > 0.0
            if can_mutate.any():
                mutable = selected[can_mutate]
                cdf_rows = PC_WAG_CDFS[group_name][source_chars[can_mutate]]
                new_chars[mutable] = _sample_from_cdf_rows(cdf_rows)

    return new_chars


def _append_gap_run(
    start: int,
    end: int,
    rates: np.ndarray,
    pc: np.ndarray,
    p_ins: float,
    out_chars: list[int],
    out_rates: list[float],
    out_pc: list[int],
) -> None:
    """Append an existing gap run, possibly filling its first slots."""
    done = False
    for step, pos in enumerate(range(start, end)):
        if done:
            char = CHAR_GAP
        elif step == 0:
            if np.random.random() < p_ins:
                char = _sample_gap_fill(int(pc[pos]))
            else:
                char = CHAR_GAP
                done = True
        else:
            if np.random.random() < P_INS_EXTEND:
                char = _sample_gap_fill(int(pc[pos]))
            else:
                char = CHAR_GAP
                done = True
        out_chars.append(int(char))
        out_rates.append(float(rates[pos]))
        out_pc.append(int(pc[pos]))


def _apply_mutations_and_indels(
    chars: np.ndarray,
    rates: np.ndarray,
    pc: np.ndarray,
    config: SimulationConfig,
    branch_length: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Apply substitutions, deletions, insertions, and gap filling to encoded rows.

    Deletion and substitution decisions are vectorised in bulk. Output assembly
    remains sequential because insertions can consume existing gap slots and
    may add new alignment columns.

    Note: p_del and p_ins are still calibrated to config.branch_length. A boosted
    duplicate edge therefore has elevated substitutions but unchanged indel probability.
    This is a known inconsistency; making indel rates scale per-edge is left for a
    future commit.
    """
    n = len(chars)
    p_del = config.p_del
    p_ins = config.p_ins
    is_gap, deleted, will_mutate = _mutation_decisions(chars, rates, branch_length, p_del)
    new_chars = _apply_substitutions(chars, pc, will_mutate)

    out_chars: list[int] = []
    out_rates: list[float] = []
    out_pc: list[int] = []
    gaps: list[int] = []

    i = 0
    while i < n:
        if is_gap[i]:
            # Locate end of gap run
            j = i + 1
            while j < n and chars[j] == CHAR_GAP:
                j += 1
            _append_gap_run(i, j, rates, pc, p_ins, out_chars, out_rates, out_pc)
            i = j
            continue

        if deleted[i]:
            out_chars.append(CHAR_GAP)
            out_rates.append(float(rates[i]))
            out_pc.append(int(pc[i]))
            i += 1
            continue

        out_chars.append(int(new_chars[i]))
        out_rates.append(float(rates[i]))
        out_pc.append(int(pc[i]))

        rate_i = float(rates[i])
        if np.random.random() < rate_i * p_ins:
            out_chars.append(_sample_from_cdf(AA_FREQUENCY_CDF))
            out_rates.append(1.0)
            out_pc.append(PC_NONE)
            if i + 1 < n and chars[i + 1] == CHAR_GAP:
                i += 1
            else:
                gaps.append(i)
            while np.random.random() < P_INS_EXTEND:
                out_chars.append(_sample_from_cdf(AA_FREQUENCY_CDF))
                out_rates.append(1.0)
                out_pc.append(PC_NONE)
                if i + 1 < n and chars[i + 1] == CHAR_GAP:
                    i += 1
                else:
                    gaps.append(i)

        i += 1

    return (
        np.array(out_chars, dtype=np.uint8),
        np.array(out_rates, dtype=np.float32),
        np.array(out_pc, dtype=np.int8),
        gaps,
    )
