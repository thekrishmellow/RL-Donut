from dataclasses import dataclass, field

from rl_donut.mapgen import MapConfig


@dataclass(frozen=True)
class PhysicsConfig:
    # Accelerations in tiles/s²; y grows downward, so gravity is positive.
    gravity: float = 25.0
    thrust: float = 40.0
    # Speed limits in tiles/s. Each must stay under one tile per tick so collision can't
    # miss a wall; rising has its own limit because a full tank would otherwise reach 38 tiles/s.
    max_rise_speed: float = 12.0
    max_fall_speed: float = 20.0
    # Horizontal speed is set directly by the move action, on the ground and in the air.
    run_speed: float = 6.0
    # Fuel in units: drained while thrusting, refilled while standing and not thrusting.
    # 32/s (not 40) leaves level flight 25% longer than the map generator assumes.
    fuel_capacity: float = 100.0
    fuel_drain: float = 32.0
    fuel_refill: float = 25.0

    def __post_init__(self):
        if min(self.gravity, self.thrust, self.max_rise_speed, self.max_fall_speed, self.run_speed) <= 0:
            raise ValueError("accelerations and speeds must be positive")
        if self.thrust <= self.gravity:
            raise ValueError("thrust must exceed gravity or the jetpack can't lift off")
        if min(self.fuel_capacity, self.fuel_drain, self.fuel_refill) <= 0:
            raise ValueError("fuel settings must be positive")


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
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
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
        p = self.physics
        if max(p.max_rise_speed, p.max_fall_speed, p.run_speed) * self.dt >= 1:
            raise ValueError("every speed limit must stay under one tile per tick")

    @property
    def n_agents(self) -> int:
        return 2 * self.team_size

    @property
    def dt(self) -> float:
        return 1 / self.tick_rate
