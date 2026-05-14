"""
Mutation, indel, and rate-distribution logic for sequence evolution.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma

from fakeprot.config import SimulationConfig
from fakeprot.models.sequence import Sequence
from fakeprot.models.species import Species
from fakeprot.substitution import (
    AA_FREQUENCIES,
    AA_INDEX,
    AMINO_ACIDS,
    SC_FREQUENCIES,
    SC_MASKS,
    SC_SUBS_MATRIX,
    STEREOCHEMICAL_GROUPS,
    STEREOCHEMICAL_SETS,
    WAG_MATRIX,
)

ROOT_AMINO_ACID = "M"
ROOT_STEREOCHEMISTRY = "With sulfur"  # Met belongs to the sulfur group


def wave_shuffle(values: list[float]) -> list[float]:
    """
    Arrange values so adjacent elements are similar in magnitude.

    Starts at a random element, then greedily picks the closest remaining
    value at each step. Weights are proportional to (1 - |current - candidate|),
    clamped to zero to guard against gamma PDF ranges that exceed 1.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    available = np.ones(n, dtype=bool)
    result = np.empty(n, dtype=float)

    current_idx = np.random.randint(n)
    result[0] = arr[current_idx]
    available[current_idx] = False

    for step in range(1, n):
        avail_idx = np.where(available)[0]
        distances = np.abs(arr[avail_idx] - result[step - 1])
        weights = np.maximum(0.0, 1.0 - distances)
        total = weights.sum()
        if total == 0.0:
            weights = np.ones(len(avail_idx), dtype=float)
            total = float(len(avail_idx))
        weights /= total
        chosen = np.random.choice(len(avail_idx), p=weights)
        current_idx = avail_idx[chosen]
        result[step] = arr[current_idx]
        available[current_idx] = False

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

    Processes positions in reverse order so earlier insertions don't shift
    the indices of later ones. Returns the number of columns added.
    """
    for pos in sorted(gaps, reverse=True):
        for seq in collection:
            seq.sequence = seq.sequence[: pos + 1] + "-" + seq.sequence[pos + 1 :]
            seq.mutation_rates = seq.mutation_rates[: pos + 1] + [1.0] + seq.mutation_rates[pos + 1 :]
            seq.stereochemistry = seq.stereochemistry[: pos + 1] + [None] + seq.stereochemistry[pos + 1 :]
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

    In duplication mode: re-draws rates and updates stereochemical constraints,
    reflecting relaxed purifying selection on the new copy.
    """
    if not duplication:
        return list(parent.sequence), list(parent.mutation_rates), list(parent.stereochemistry)

    # Re-draw rates, preserving rank order from the parent
    new_rates = mutation_rate_distribution(len(parent.sequence), config, duplication=True)
    ranked_positions = sorted(range(len(parent.sequence)), key=lambda i: parent.mutation_rates[i])
    sorted_new = sorted(new_rates)
    reranked = dict(zip(ranked_positions, sorted_new))
    current_rates = [reranked[i] for i in range(len(parent.sequence))]

    residues = list(parent.sequence)
    stereo: list[str | None] = [parent.stereochemistry[0]]

    for i in range(1, len(residues)):
        rate = current_rates[i]
        # With probability (1 - rate) the site acquires / retains a stereochemical class
        if np.random.choice((True, False), p=(1.0 - rate, rate)):
            if parent.stereochemistry[i] is None:
                new_sc = np.random.choice(STEREOCHEMICAL_GROUPS, p=SC_FREQUENCIES)
            else:
                p_sc = {k: rate * v for k, v in SC_SUBS_MATRIX[parent.stereochemistry[i]].items()}
                p_sc[parent.stereochemistry[i]] = 1.0 - rate
                keys = list(p_sc.keys())
                vals = np.array(list(p_sc.values()))
                new_sc = np.random.choice(keys, p=vals / vals.sum())
            stereo.append(new_sc)
            # Force residue into the new class if it has drifted out
            if residues[i] not in STEREOCHEMICAL_SETS[new_sc]:
                mask = SC_MASKS[new_sc]
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
    """Draw a new amino acid using WAG probabilities with no stereochemical constraint."""
    p = WAG_MATRIX[aa_idx] * rate
    p[aa_idx] = 1.0 - rate
    return AMINO_ACIDS[np.random.choice(20, p=p / p.sum())]


def _sample_aa_constrained(rate: float, aa_idx: int, group: str) -> str:
    """Draw a new amino acid restricted to a stereochemical group."""
    mask = SC_MASKS[group].copy()
    mask[aa_idx] = False  # diagonal handled separately
    p = WAG_MATRIX[aa_idx] * mask
    total = p.sum()
    if total > 0.0:
        p = p / total * rate
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
            if np.random.choice((True, False), p=(p_fill, 1.0 - p_fill)):
                if stereo[i] is None:
                    aa = AMINO_ACIDS[np.random.choice(20, p=AA_FREQUENCIES)]
                else:
                    mask = SC_MASKS[stereo[i]]
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

    if np.random.choice((True, False), p=(p_delete, 1.0 - p_delete)):
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
            if np.random.choice((True, False), p=(p_insert, 1.0 - p_insert)):
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
