"""Exploit watchlist and behavioural anomaly detection (Phase 5 Step 5.6).

Monitors training for common MARL exploits:
- Passive camping (high draw rate)
- Ceiling / corner hovering (excessive hover time)
- Spawn camping (kills within 8 tiles of enemy spawns)
- Mindless shooting spam (low accuracy / high shots per hit)
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class WatchlistThresholds:
    max_draw_rate: float = 0.35            # Alarms if > 35% of completed matches are draws
    max_hover_fraction: float = 0.50       # Alarms if > 50% of time spent hovering in air
    min_engagement_distance: float = 3.0   # Alarms if average hit distance < 3 tiles (face-hugging)
    max_shots_per_hit: float = 8.0         # Alarms if > 8 shots per hit (trigger spam)
    max_spawn_kill_fraction: float = 0.40  # Alarms if > 40% of kills are near enemy spawns


@dataclass
class WatchlistReport:
    draw_rate: float
    hover_fraction: float
    mean_engagement_dist: float
    shots_per_hit: float
    spawn_kill_fraction: float
    alarms: list[str] = field(default_factory=list)

    @property
    def has_alarms(self) -> bool:
        return len(self.alarms) > 0


class WatchlistTracker:
    """Accumulates match statistics and detects strategic exploits."""

    def __init__(self, thresholds: WatchlistThresholds | None = None):
        self.thresholds = thresholds or WatchlistThresholds()
        self.completed_matches = 0
        self.draws = 0
        self.total_kills = 0
        self.spawn_kills = 0
        self.total_shots = 0
        self.total_hits = 0
        self.total_hit_distances: list[float] = []
        self.hover_ticks = 0
        self.total_agent_ticks = 0

    def record_tick(self, vy: np.ndarray, on_ground: np.ndarray, alive: np.ndarray) -> None:
        """Record hover behavior across all agents on one tick."""
        # Hovering: living agent in air with absolute vertical speed < 2.0 tiles/s
        in_air = alive & ~on_ground
        hovering = in_air & (np.abs(vy) < 2.0)
        self.hover_ticks += int(hovering.sum())
        self.total_agent_ticks += int(alive.sum())

    def record_shots(self, shots: int, hits: int, distances: list[float] | None = None) -> None:
        self.total_shots += shots
        self.total_hits += hits
        if distances:
            self.total_hit_distances.extend(distances)

    def record_kill(self, dist_to_enemy_spawn: float) -> None:
        self.total_kills += 1
        if dist_to_enemy_spawn < 8.0:
            self.spawn_kills += 1

    def record_match_outcome(self, outcome: int) -> None:
        self.completed_matches += 1
        if outcome == 0:
            self.draws += 1

    def evaluate(self) -> WatchlistReport:
        """Generate exploit report with active alarm warnings."""
        draw_rate = (self.draws / self.completed_matches) if self.completed_matches > 0 else 0.0
        hover_frac = (self.hover_ticks / self.total_agent_ticks) if self.total_agent_ticks > 0 else 0.0
        mean_dist = float(np.mean(self.total_hit_distances)) if self.total_hit_distances else 10.0
        sph = (self.total_shots / self.total_hits) if self.total_hits > 0 else 1.0
        sk_frac = (self.spawn_kills / self.total_kills) if self.total_kills > 0 else 0.0

        alarms = []
        t = self.thresholds
        if self.completed_matches >= 10 and draw_rate > t.max_draw_rate:
            alarms.append(f"HIGH_DRAW_RATE ({draw_rate:.1%}) - possible passive avoidance")
        if self.total_agent_ticks >= 200 and hover_frac > t.max_hover_fraction:
            alarms.append(f"EXCESSIVE_HOVER ({hover_frac:.1%}) - ceiling/camping exploit")
        if len(self.total_hit_distances) >= 20 and mean_dist < t.min_engagement_distance:
            alarms.append(f"POINT_BLANK_ENGAGEMENT ({mean_dist:.1f} tiles) - face hug exploit")
        if self.total_hits >= 20 and sph > t.max_shots_per_hit:
            alarms.append(f"TRIGGER_SPAM ({sph:.1f} shots/hit) - blind spray exploit")
        if self.total_kills >= 10 and sk_frac > t.max_spawn_kill_fraction:
            alarms.append(f"SPAWN_CAMPING ({sk_frac:.1%}) - spawn entrapment exploit")

        return WatchlistReport(
            draw_rate=draw_rate,
            hover_fraction=hover_frac,
            mean_engagement_dist=mean_dist,
            shots_per_hit=sph,
            spawn_kill_fraction=sk_frac,
            alarms=alarms,
        )

