from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fakeprot.models.sequence import Sequence


@dataclass(eq=False)
class Species:
    """A node in the species tree, carrying one or more paralogous sequences."""

    paralogs: list["Sequence"]
    label: str = ""

    def __repr__(self) -> str:
        return self.label
