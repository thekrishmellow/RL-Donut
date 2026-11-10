from dataclasses import dataclass, field

from rl_donut.mapgen import MapConfig

# Physics scales the engine is tested at; outside them the climb and speed contracts are unchecked.
MIN_SCALE, MAX_SCALE = 0.5, 1.5


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
    # Each match multiplies gravity and thrust by one factor drawn uniformly from this
    # range. Scaling both together keeps level flight per tank unchanged, so maps stay
    # crossable; (1, 1) turns it off.
    scale_range: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self):
        lo, hi = self.scale_range
        if not MIN_SCALE <= lo <= hi <= MAX_SCALE:
            raise ValueError(f"scale_range must satisfy {MIN_SCALE} <= low <= high <= {MAX_SCALE}")
        if min(self.gravity, self.thrust, self.max_rise_speed, self.max_fall_speed, self.run_speed) <= 0:
            raise ValueError("accelerations and speeds must be positive")
        if self.thrust <= self.gravity:
            raise ValueError("thrust must exceed gravity or the jetpack can't lift off")
        if min(self.fuel_capacity, self.fuel_drain, self.fuel_refill) <= 0:
            raise ValueError("fuel settings must be positive")


@dataclass(frozen=True)
class RifleConfig:
    damage: float = 15.0
    # Times in seconds; each must be a whole number of ticks.
    shot_interval: float = 0.2
    reload_time: float = 1.5
    # Reloading starts automatically when the magazine is empty; spare rounds are unlimited.
    magazine: int = 30
    range: float = 30.0
    # Each shot leaves at the aim angle plus a uniform random offset within ± this many degrees.
    spread_deg: float = 2.0

    def __post_init__(self):
        if min(self.damage, self.shot_interval, self.reload_time, self.range) <= 0 or self.magazine < 1:
            raise ValueError("rifle settings must be positive")
        if not 0 <= self.spread_deg < 90:
            raise ValueError("spread_deg must be in [0, 90)")


@dataclass(frozen=True)
class LifeConfig:
    max_health: float = 100.0
    # Health per second, starting once `regen_delay` seconds pass without damage.
    regen_rate: float = 10.0
    # Times in seconds; each must be a whole number of ticks.
    regen_delay: float = 3.0
    respawn_time: float = 2.0
    # No damage for this long after respawning; firing ends it early.
    protection_time: float = 1.0

    def __post_init__(self):
        if min(self.max_health, self.respawn_time) <= 0 or min(self.regen_rate, self.regen_delay, self.protection_time) < 0:
            raise ValueError("health and respawn settings must be positive")


@dataclass(frozen=True)
class MatchConfig:
    # Timed deathmatch: most kills when time runs out wins, equal is a draw.
    duration: float = 60.0
    # Agents decide every this many ticks; each action is held until the next decision.
    decision_ticks: int = 2
    # A pit fall scores for the last enemy to damage the faller within this many seconds.
    fall_credit_window: float = 3.0

    def __post_init__(self):
        if self.duration <= 0 or self.decision_ticks < 1 or self.fall_credit_window < 0:
            raise ValueError("match settings must be positive")


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
    rifle: RifleConfig = field(default_factory=RifleConfig)
    life: LifeConfig = field(default_factory=LifeConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
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
        life = self.life
        for seconds in (self.rifle.shot_interval, self.rifle.reload_time, life.regen_delay,
                        life.respawn_time, life.protection_time, self.match.duration,
                        self.match.fall_credit_window):
            if abs(seconds * self.tick_rate - round(seconds * self.tick_rate)) > 1e-9:
                raise ValueError("rifle, respawn and match times must be whole numbers of ticks")
        if self.ticks(self.match.duration) % self.match.decision_ticks:
            raise ValueError("match duration must be a whole number of decisions")

    def ticks(self, seconds: float) -> int:
        return round(seconds * self.tick_rate)

    @property
    def shot_interval_ticks(self) -> int:
        return self.ticks(self.rifle.shot_interval)

    @property
    def reload_ticks(self) -> int:
        return self.ticks(self.rifle.reload_time)

    @property
    def n_agents(self) -> int:
        return 2 * self.team_size

    @property
    def dt(self) -> float:
        return 1 / self.tick_rate
