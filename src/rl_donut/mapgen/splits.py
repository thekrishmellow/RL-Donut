"""Seed splits. Ranges never overlap, so no evaluation map is ever trained on."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    name: str
    start: int
    count: int

    @property
    def seeds(self) -> range:
        return range(self.start, self.start + self.count)


# OOD splits (seeds 30,000+, shifted settings) are defined in Phase 8.
SPLITS = {s.name: s for s in (
    Split("train", 0, 10_000),
    Split("val", 10_000, 1_000),
    Split("test", 20_000, 1_000),
)}
