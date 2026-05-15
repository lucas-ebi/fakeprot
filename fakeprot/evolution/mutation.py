"""
Mutation, indel, and rate-distribution logic for sequence evolution.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.stats import gamma

from fakeprot.config import SimulationConfig
from fakeprot.models.sequence import Sequence
from fakeprot.models.species import Species
from fakeprot.substitution import (
    AA_FREQUENCIES,
    AA_INDEX,
    AMINO_ACIDS,
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
    parent: Sequence,
    new_host: Species,
    new_idx: int,
    config: SimulationConfig,
    duplication: bool = False,
) -> tuple[Sequence, list[int]]:
    """
    Derive a new sequence by evolving parent along one branch.

    Returns the child Sequence and a list of gap positions that must be
    inserted into all other sequences to maintain alignment (see apply_gaps).
    """
    residues, rates, stereo = _prepare_evolution_state(parent, config, duplication)
    new_seq, new_rates, new_stereo, gaps = _apply_mutations_and_indels(
        residues, rates, stereo, config
    )
    child = Sequence(new_seq, new_rates, new_stereo, new_host, new_idx)
    return child, gaps


def apply_gaps(collection: list[Sequence], gaps: list[int]) -> int:
    """
    Insert gap columns into every sequence to preserve alignment after insertions.

    Builds each updated sequence in a single pass (one allocation per sequence)
    instead of one allocation per gap position. Returns the number of columns added.
    """
    if not gaps:
        return 0
    gap_counts = Counter(gaps)
    sorted_positions = sorted(gap_counts)
    for seq in collection:
        s = seq.sequence
        r = seq.mutation_rates
        p = seq.pc_groups
        parts_s: list[str] = []
        parts_r: list[float] = []
        parts_p: list[str | None] = []
        prev = 0
        for g in sorted_positions:
            end = g + 1
            parts_s.append(s[prev:end])
            parts_r += r[prev:end]
            parts_p += p[prev:end]
            count = gap_counts[g]
            parts_s.append("-" * count)
            parts_r += [1.0] * count
            parts_p += [None] * count
            prev = end
        parts_s.append(s[prev:])
        parts_r += r[prev:]
        parts_p += p[prev:]
        seq.sequence = "".join(parts_s)
        seq.mutation_rates = parts_r
        seq.pc_groups = parts_p
    return len(gaps)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _prepare_evolution_state(
    parent: Sequence,
    config: SimulationConfig,
    duplication: bool,
) -> tuple[list[str], list[float], list[str | None]]:
    """
    Return (residues, rates, stereochemistry) ready for mutation.

    In duplication mode: re-draws rates and updates physicochemical constraints,
    reflecting relaxed purifying selection on the new copy.
    """
    if not duplication:
        return list(parent.sequence), list(parent.mutation_rates), list(parent.pc_groups)

    # Re-draw rates, preserving rank order from the parent
    new_rates = mutation_rate_distribution(len(parent.sequence), config, duplication=True)
    ranked_positions = sorted(range(len(parent.sequence)), key=lambda i: parent.mutation_rates[i])
    sorted_new = sorted(new_rates)
    reranked = dict(zip(ranked_positions, sorted_new))
    current_rates = [reranked[i] for i in range(len(parent.sequence))]

    residues = list(parent.sequence)
    stereo: list[str | None] = [parent.pc_groups[0]]

    for i in range(1, len(residues)):
        rate = current_rates[i]
        # With probability (1 - rate) the site acquires / retains a physicochemical class
        if np.random.random() < (1.0 - rate):
            if parent.pc_groups[i] is None:
                new_sc = np.random.choice(PHYSICOCHEMICAL_GROUPS, p=PC_FREQUENCIES)
            else:
                p_sc = {k: rate * v for k, v in PC_SUBS_MATRIX[parent.pc_groups[i]].items()}
                p_sc[parent.pc_groups[i]] = 1.0 - rate
                keys = list(p_sc.keys())
                vals = np.array(list(p_sc.values()))
                new_sc = np.random.choice(keys, p=vals / vals.sum())
            stereo.append(new_sc)
            # Force residue into the new class if it has drifted out
            if residues[i] not in PHYSICOCHEMICAL_SETS[new_sc]:
                mask = PC_MASKS[new_sc]
                if residues[i] == "-":
                    p = AA_FREQUENCIES * mask
                else:
                    p = WAG_MATRIX[AA_INDEX[residues[i]]] * mask
                p = p / p.sum()
                residues[i] = AMINO_ACIDS[np.random.choice(20, p=p)]
        else:
            stereo.append(None)

    return residues, current_rates, stereo


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

        # Insertions after this position, with geometrically decreasing probability
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
