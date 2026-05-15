"""Columnar MSA storage: three (n_sequences × length) numpy arrays."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from fakeprot.substitution import (
    AA_INDEX,
    AMINO_ACIDS,
    CHAR_GAP,
    PC_INDEX,
    PC_NONE,
    PHYSICOCHEMICAL_GROUPS,
)

if TYPE_CHECKING:
    from fakeprot.models.sequence import Sequence
    from fakeprot.models.species import Species


_CHAR_BYTES = np.frombuffer(("".join(AMINO_ACIDS) + "-").encode("ascii"), dtype="S1")


class MsaStore:
    """
    Three (n_sequences × length) arrays holding all sequence data.

    chars : uint8   — 0–19 = amino acid index, 20 = gap
    rates : float32
    pc    : int8    — –1 = None, 0–17 = physicochemical-group index

    Rows are pre-allocated with capacity doubling to avoid O(n²) copies.
    External code reads via the .chars/.rates/.pc properties, which return
    active-only views so shape[0] always equals the number of committed rows.
    """

    def __init__(
        self,
        chars: np.ndarray,
        rates: np.ndarray,
        pc: np.ndarray,
        capacity: int = 64,
    ) -> None:
        L = len(chars)
        cap = max(capacity, 1)
        self._cap = cap
        self._n = 1
        self._chars = np.empty((cap, L), dtype=np.uint8)
        self._rates = np.empty((cap, L), dtype=np.float32)
        self._pc    = np.full( (cap, L), PC_NONE, dtype=np.int8)
        self._chars[0] = chars
        self._rates[0] = rates
        self._pc[0]    = pc

    # ------------------------------------------------------------------
    # Active-row views — shape[0] == committed rows, never capacity
    # ------------------------------------------------------------------

    @property
    def chars(self) -> np.ndarray:
        return self._chars[:self._n]

    @property
    def rates(self) -> np.ndarray:
        return self._rates[:self._n]

    @property
    def pc(self) -> np.ndarray:
        return self._pc[:self._n]

    @property
    def n_rows(self) -> int:
        return self._n

    @property
    def length(self) -> int:
        return self._chars.shape[1]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_row(self, chars: np.ndarray, rates: np.ndarray, pc: np.ndarray) -> int:
        """Append one sequence row; return its row index."""
        L = self.length
        if len(chars) != L or len(rates) != L or len(pc) != L:
            raise ValueError(
                f"Array lengths ({len(chars)}, {len(rates)}, {len(pc)}) "
                f"!= store length {L}; call insert_gaps (or commit_child) before add_row."
            )
        if self._n == self._cap:
            self._cap *= 2
            L = self._chars.shape[1]
            new_c = np.empty((self._cap, L), dtype=np.uint8)
            new_r = np.empty((self._cap, L), dtype=np.float32)
            new_p = np.full( (self._cap, L), PC_NONE, dtype=np.int8)
            new_c[:self._n] = self._chars[:self._n]
            new_r[:self._n] = self._rates[:self._n]
            new_p[:self._n] = self._pc[:self._n]
            self._chars, self._rates, self._pc = new_c, new_r, new_p
        self._chars[self._n] = chars
        self._rates[self._n] = rates
        self._pc[self._n]    = pc
        row = self._n
        self._n += 1
        return row

    def insert_gaps(self, gaps: list[int]) -> int:
        """Insert gap columns into every active row at once. Returns columns added."""
        if not gaps:
            return 0
        counts = Counter(gaps)
        positions: list[int] = []
        for pos in sorted(counts):
            positions.extend([pos + 1] * counts[pos])
        n = self._n
        new_c = np.insert(self._chars[:n], positions, CHAR_GAP, axis=1)
        new_r = np.insert(self._rates[:n], positions, 1.0,      axis=1)
        new_p = np.insert(self._pc[:n],    positions, PC_NONE,  axis=1)
        new_L = new_c.shape[1]
        full_c = np.empty((self._cap, new_L), dtype=np.uint8)
        full_r = np.empty((self._cap, new_L), dtype=np.float32)
        full_p = np.full( (self._cap, new_L), PC_NONE, dtype=np.int8)
        full_c[:n] = new_c
        full_r[:n] = new_r
        full_p[:n] = new_p
        self._chars, self._rates, self._pc = full_c, full_r, full_p
        return len(gaps)

    def commit_child(
        self,
        chars: np.ndarray,
        rates: np.ndarray,
        pc: np.ndarray,
        gaps: list[int],
        host: "Species",
        idx: int,
    ) -> tuple["Sequence", int]:
        """
        apply_gaps → add_row → Sequence, in the only correct order.

        Returns (child_sequence, columns_added).
        """
        from fakeprot.models.sequence import Sequence
        n = self.insert_gaps(gaps)
        row = self.add_row(chars, rates, pc)
        return Sequence(row, host, idx), n


# ------------------------------------------------------------------
# Encode / decode helpers
# ------------------------------------------------------------------


def decode_chars(chars: np.ndarray) -> str:
    """Convert a uint8 row to an amino acid / gap string."""
    return _CHAR_BYTES[chars].tobytes().decode("ascii")


def decode_pc(pc_row: np.ndarray) -> list[str | None]:
    """Convert an int8 row to a list of physicochemical group names."""
    return [PHYSICOCHEMICAL_GROUPS[int(p)] if p >= 0 else None for p in pc_row]


def encode_row(
    residues: list[str],
    rates: list[float],
    pc: list[str | None],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Python list representations to packed numpy arrays."""
    chars  = np.array([AA_INDEX[r] if r != "-" else CHAR_GAP for r in residues], dtype=np.uint8)
    rates_ = np.array(rates, dtype=np.float32)
    pc_    = np.array([PC_INDEX[p] if p is not None else PC_NONE for p in pc], dtype=np.int8)
    return chars, rates_, pc_
