from dataclasses import dataclass, field

from rl_donut.mapgen import MapConfig


@dataclass(frozen=True)
class EngineConfig:
    # Agents per team: 1 for 1v1, 2 for 2v2.
    team_size: int = 1
    # Physics ticks per second of game time.
    tick_rate: int = 30
    # Agent hitbox in tiles. Must fit inside the generator's clearance box, which is what
    # the map guarantees is passable.
    hitbox_w: float = 0.8
    hitbox_h: float = 1.6
    map: MapConfig = field(default_factory=MapConfig)

    def __post_init__(self):
        if self.team_size < 1:
            raise ValueError("team_size must be >= 1")
        if self.team_size > self.map.spawns.per_team:
            raise ValueError("team_size cannot exceed the map's spawns per team")
        if self.tick_rate < 1:
            raise ValueError("tick_rate must be >= 1")
        clearance = self.map.regions
        if not (0 < self.hitbox_w <= clearance.clearance_w and 0 < self.hitbox_h <= clearance.clearance_h):
            raise ValueError("hitbox must be positive and fit inside the map's clearance box")

    @property
    def n_agents(self) -> int:
        return 2 * self.team_size
