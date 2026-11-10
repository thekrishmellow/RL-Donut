import numpy as np
from rl_donut.rl.watchlist import WatchlistTracker, WatchlistThresholds


def test_watchlist_hover_and_draw_alarms():
    tracker = WatchlistTracker(WatchlistThresholds(max_draw_rate=0.20, max_hover_fraction=0.30))

    # Record 15 draws out of 20 matches
    for _ in range(15):
        tracker.record_match_outcome(0)  # draw
    for _ in range(5):
        tracker.record_match_outcome(1)  # win

    # Record excessive hover
    vy = np.array([[0.5, 0.2]])
    on_ground = np.array([[False, False]])
    alive = np.array([[True, True]])
    for _ in range(150):
        tracker.record_tick(vy, on_ground, alive)

    report = tracker.evaluate()
    assert report.has_alarms
    assert any("HIGH_DRAW_RATE" in a for a in report.alarms)
    assert any("EXCESSIVE_HOVER" in a for a in report.alarms)


def test_watchlist_clean_behavior():
    tracker = WatchlistTracker()
    for _ in range(20):
        tracker.record_match_outcome(1)
        tracker.record_shots(10, 4, [12.0, 15.0, 10.0, 14.0])
        tracker.record_kill(18.0)

    report = tracker.evaluate()
    assert not report.has_alarms
    assert report.draw_rate == 0.0
    assert report.mean_engagement_dist > 10.0

