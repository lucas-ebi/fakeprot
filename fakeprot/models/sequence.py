from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fakeprot.models.species import Species


@dataclass(eq=False)
class Sequence:
    """A node in the evolutionary tree; sequence data lives in MsaStore."""

    row: int
    host: "Species"
    idx: int

    @property
    def label(self) -> str:
        return f"{self.host.label}_seq{self.idx + 1}"

    def __repr__(self) -> str:
        return self.label
