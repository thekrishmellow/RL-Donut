"""Build docs/RL-Donut-technical-report.pdf from the decision log and measured runs.

    .venv/bin/python scripts/build_technical_report.py

reportlab is used only by this script. It is not a runtime dependency of the environment.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    ListFlowable,
    ListItem,
    CondPageBreak,
    KeepTogether,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "RL-Donut-technical-report.pdf"

INK = colors.HexColor("#1c1917")
RULE = colors.HexColor("#d6d3d1")
HEAD = colors.HexColor("#292524")
BAND = colors.HexColor("#f5f5f4")
ACCENT = colors.HexColor("#9a3412")


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        "CoverKicker", fontName="Times-Bold", fontSize=9, leading=12,
        textColor=ACCENT, tracking=1.2, spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        "CoverTitle", fontName="Times-Bold", fontSize=22, leading=26,
        textColor=INK, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        "CoverSub", fontName="Times-Italic", fontSize=12, leading=16,
        textColor=INK, spaceAfter=10,
    ))
    base.add(ParagraphStyle(
        "Meta", fontName="Times-Roman", fontSize=9.5, leading=13,
        textColor=INK, spaceAfter=3,
    ))
    base.add(ParagraphStyle(
        "H1", fontName="Times-Bold", fontSize=13, leading=16,
        textColor=INK, spaceBefore=14, spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        "H2", fontName="Times-Bold", fontSize=11.5, leading=14,
        textColor=INK, spaceBefore=10, spaceAfter=4,
    ))
    base.add(ParagraphStyle(
        "Body", fontName="Times-Roman", fontSize=10, leading=13.5,
        textColor=INK, alignment=TA_JUSTIFY, spaceAfter=7,
    ))
    base.add(ParagraphStyle(
        "BulletBody", fontName="Times-Roman", fontSize=10, leading=13.2,
        textColor=INK, alignment=TA_LEFT, leftIndent=0,
    ))
    base.add(ParagraphStyle(
        "Cell", fontName="Times-Roman", fontSize=8, leading=10.5, textColor=INK,
    ))
    base.add(ParagraphStyle(
        "CellHead", fontName="Times-Bold", fontSize=8, leading=10.5, textColor=colors.white,
    ))
    base.add(ParagraphStyle(
        "Caption", fontName="Times-Italic", fontSize=8.5, leading=11,
        textColor=colors.HexColor("#44403c"), spaceBefore=2, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        "Footer", fontName="Times-Roman", fontSize=8, leading=10, textColor=colors.HexColor("#57534e"),
    ))
    return base


S = styles()


def P(text, style="Body"):
    return Paragraph(text, S[style])


def bullets(items):
    flow = []
    for item in items:
        flow.append(ListItem(Paragraph(item, S["BulletBody"]), leftIndent=12, bulletColor=INK))
    return ListFlowable(
        flow, bulletType="bullet", start="-", leftIndent=14,
        bulletFontName="Times-Roman", bulletFontSize=10, spaceBefore=1, spaceAfter=6,
    )


def table(headers, rows, col_widths):
    head = [Paragraph(h, S["CellHead"]) for h in headers]
    body = [[Paragraph(str(c), S["Cell"]) for c in row] for row in rows]
    t = Table([head] + body, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BAND]),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("GRID", (0, 0), (-1, -1), 0.3, RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, HEAD),
    ]))
    return t


def header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(1.5)
    canvas.line(18 * mm, h - 12 * mm, w - 18 * mm, h - 12 * mm)
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(colors.HexColor("#57534e"))
    canvas.drawString(18 * mm, h - 10 * mm, "RL-Donut technical report")
    canvas.drawRightString(w - 18 * mm, h - 10 * mm, "15 November 2026")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(18 * mm, 12 * mm, w - 18 * mm, 12 * mm)
    canvas.drawString(18 * mm, 7.5 * mm, "github.com/thekrishmellow/RL-Donut")
    canvas.drawRightString(w - 18 * mm, 7.5 * mm, f"{doc.page}")
    canvas.restoreState()


def build():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=16 * mm,
        title="RL-Donut: Technical Report",
        author="RL-Donut team",
        subject="Seeded 2D combat environment and partial PPO results",
    )
    W = A4[0] - 36 * mm
    story = []

    story.append(P("TECHNICAL REPORT", "CoverKicker"))
    story.append(P("RL-Donut: learning to fight in seeded caves", "CoverTitle"))
    story.append(P(
        "A Mini Militia-style 2D environment, the decisions behind it, and the training result measured on 15 November 2026.",
        "CoverSub",
    ))
    story.append(P("RL-Donut team. Repository: github.com/thekrishmellow/RL-Donut, branch main.", "Meta"))
    story.append(P("Decision log: DESIGN.md. Current measurements: docs/PROGRESS.md. This PDF supersedes docs/REPORT.md as the document to cite.", "Meta"))
    story.append(Spacer(1, 6))

    story.append(P("Abstract", "H1"))
    story.append(P(
        "RL-Donut is a two-dimensional deathmatch built as a reinforcement-learning environment. "
        "Agents with jetpacks and hitscan rifles fight on caves generated from an integer seed. "
        "The same seed rebuilds the same cave. Observations are 330 numbers, not pixels. "
        "Actions are three discrete choices plus one continuous aim angle. The environment side, "
        "Phases 1-4, is complete and tested: map generator, physics, visibility, scripted bots, "
        "replays. The learning code for Phases 5-10 is on main. Training is not finished. "
        "The measured run, runs/spawn-curriculum-14, trained PPO against a static bot. "
        "With the enemy already in sight at 6-12 tiles, the policy won 8 of 16 validation matches "
        "at update 175. At 12-16 tiles the best check was 4 of 16, and later checks fell to 1 of 16. "
        "The official spawns, which hide the two teams from each other, were not trained. "
        "An earlier sign-off, docs/REPORT.md, is kept in the repository and is not this result."
    ))

    story.append(P("1. Question and method", "H1"))
    story.append(P(
        "The question is whether a policy trained by self-play on procedural caves keeps its skill "
        "on maps and physics it did not train on. The format follows the EvoArena brief's twelve "
        "milestones, grouped into ten phases: maps, engine, observation and action, bots, reward "
        "and PPO, self-play, adaptive opponents, generalization, a dashboard, and a benchmark."
    ))
    story.append(P(
        "Every approved choice is a row in DESIGN.md. A choice is written down with the alternative "
        "that was rejected and the measurement that decided it. Phases 1-4 are logged that way, "
        "decision by decision. Phases 5-10 have code on main. They are not logged as approved or "
        "finished. G3, the choice between 1v1 and 2v2 as the reported format, is still open; "
        "reported matches so far are 1v1."
    ))

    story.append(P("2. Making a map from a seed", "H1"))
    story.append(P(
        "Nothing in the generator is a noise library. M1 writes Perlin noise in NumPy. At each "
        "grid corner a unit gradient is drawn at a random angle. Between corners the gradients "
        "are blended with a quintic fade. M7 uses that quintic because a smoother blend still "
        "leaves a crease on every cell edge. Cells are 16 tiles wide and 10 tall, so the hills "
        "become ledges. Fractional Brownian motion adds four octaves. Each octave is twice as "
        "fine and half as strong as the one before: lacunarity 2, persistence 0.5."
    ))
    story.append(P(
        "M8 turns the noise into rock with a quantile, not a fixed cutoff. Aimed at 40 percent "
        "rock, a fixed cutoff produced anywhere from 31 to 50 percent across 1,000 seeds. The "
        "quantile hits the fraction exactly. Each map draws its rock fraction from 40 to 55 "
        "percent. M9 adds a height bias, 0 to 0.3, so rock prefers the bottom and the sky stays "
        "open. M12 then runs three cellular-automaton passes: a tile becomes rock when 5 of its "
        "8 neighbours are rock, and air when 3 or fewer are. Rough single-tile jags fell from "
        "about 72 per map to about 1. Erosion and dilation were rejected because they cut the "
        "thin bridges agents need. M13 mirrors the left half onto the right before thresholding. "
        "Smoothing a half map first treats the centre as a wall and grows a spine. M14 rejects "
        "a 180 degree rotation: with the height bias, one team would get the ground and the "
        "other the sky."
    ))
    story.append(P(
        "M2 splits one seed with NumPy SeedSequence into named PCG64 streams: terrain, tunnels, "
        "platforms, pits, spawns, and the per-map settings. A new use of randomness cannot change "
        "maps that already exist. A failed map is retried with a new attempt number, not by "
        "adding to the seed. M3 pins NumPy at 2.5.4 and stores a generator version plus golden "
        "hashes, because NumPy promises identical random numbers only on the same build. M41 "
        "draws rock fraction, height bias, and pit count once, at attempt 0. Later attempts "
        "redraw geometry only. Redrawing the settings would have skewed the kept maps toward "
        "layouts that pass more easily."
    ))
    story.append(P("2.1 Playable space", "H2"))
    story.append(P(
        "An agent is 0.8 by 1.6 tiles. M4 does not treat connectivity as \"is this tile air\". "
        "It flood-fills, four directions, only where a 2 by 3 clearance box fits. On raw terrain "
        "that box left 64 percent of the air usable, against 85 percent for a box the agent's "
        "own size, which is why smoothing and tunnels have to recover space. M20 joins "
        "disconnected caves with Dijkstra on those boxes. Each step costs 1, plus 12 for every "
        "rock tile dug, so the path reuses existing air. M30 charges a second vertical dig "
        "triple, which turns shafts into staircases. Fully reachable maps went from 78 percent "
        "to about 94 percent, with about 39 tiles dug per map either way. A plain shaft taller "
        "than the jetpack can climb is a one-way trap."
    ))
    story.append(P(
        "M27 models flight for the generator: climbing one tile costs 2 fuel, sideways costs 1, "
        "falling is free, and the tank is 24, twice the climb budget of 12. M31 accepts a map "
        "when every floor you can reach, you can also leave, and at most 2 percent of floor is "
        "stranded. Demanding 100 percent would reject maps over a two-tile notch. M32 fills "
        "unreachable air only when it is at most 3 percent of the air. Filling everything would "
        "delete whole upper caverns. M23 requires two crossings of the centre, or one opening "
        "at least 12 rows tall, so a single choke cannot be the whole strategy."
    ))
    story.append(P(
        "Pits, M17 and M33-M36, are 0 to 2 per map, 3 to 6 tiles wide, cut from the bottom "
        "before platforms are placed. If the requested count does not fit, the seed is retried. "
        "Quietly dropping a pit would have turned \"one pit\" into \"no pit\" on about 60 percent "
        "of maps. Platforms, M28 and M35, are solid ledges. They may not sit within two tiles "
        "of a pit, or the pit stops teaching anything. Spawns, M37-M39, are six per team, "
        "mirrored, on the main floor, at least 6 tiles from a pit and 16 from the centre, with "
        "no line of sight to an enemy spawn. M40's line of sight is an Amanatides-Woo grid walk "
        "that checks both side tiles at a corner. Sampling the line every quarter tile missed "
        "46 corner clips out of 10,000. The rifle uses the same function, so \"cannot see\" and "
        "\"cannot be shot\" are one fact."
    ))
    story.append(P(
        "M43 allows 30 attempts, then the seed is skipped and listed. In 10,000 training seeds, "
        "3 were skipped. M45 stores each split as a compressed grid array plus a JSON manifest "
        "with a hash per map. M46 commits train, val, and test. M47 freezes them: a generator "
        "change bumps the version and rebuilds. Splits do not overlap."
    ))
    story.append(table(
        ["Split", "Seeds", "Role"],
        [
            ["train", "0-9,999", "Learning"],
            ["val", "10,000-10,999", "Checks during training. Never used to learn."],
            ["test", "20,000-20,999", "Held for a single final evaluation. Not used by the run in section 7."],
            ["OOD", "30,000+", "Shifted generator or physics. Defined for Phase 8. Not the result below."],
        ],
        [28 * mm, 38 * mm, W - 66 * mm],
    ))
    story.append(P("Table 1. Seed splits. Built maps are in data/maps/.", "Caption"))

    story.append(P("3. The match", "H1"))
    story.append(P(
        "P1 stores every match in NumPy arrays of shape (matches, agents). At 64 matches that "
        "ran about 470,000 physics ticks a second, against about 48,000 with ordinary Python "
        "objects and 12,000 with one array per match. P10 integrates with the semi-implicit "
        "Euler step: speed, then position. P4 collides one axis at a time and snaps to the tile "
        "edge. That is exact while nothing moves a full tile in one tick. M16 and P8 keep the "
        "fastest motion at 0.67 tiles per tick: fall speed 20 tiles per second, 30 ticks per "
        "second. There is no jump and no fall damage. The jetpack is the only way up."
    ))
    story.append(P(
        "P8 sets gravity at 25 tiles per second squared and thrust at 40. Run speed is 6 tiles "
        "per second, set directly, on the ground and in the air (P9). The fuel tank is 100, "
        "drains at 32 per second while thrusting, and refills at 25 per second while standing. "
        "Those numbers climb 35.7 tiles and fly 30 tiles level, past the generator's budget of "
        "12 and 24. P11 checks the margin. I17, off by default, scales gravity and thrust by "
        "the same factor so level flight stays 30 tiles. Scaling them independently by 15 percent "
        "dropped level flight to 22 tiles, below the maps."
    ))
    story.append(P(
        "P15 and P16 make the rifle a hitscan ray from the hitbox centre. Damage is 15, so a "
        "kill is 7 hits, about 1.2 seconds at full fire. Shots are 0.2 seconds apart, the "
        "magazine holds 30, reload is 1.5 seconds, range is 30 tiles, and each shot adds a "
        "uniform scatter of plus or minus 2 degrees. Teammates are neither hit nor blocking. "
        "P19 applies every hit on a tick together, so a trade kills both. Resolving in agent "
        "order would hand every tie to one side. P20 gives 100 health and 10 health per second "
        "after 3 seconds without damage. From nearly dead to full is 13 seconds, longer than a "
        "kill, so retreat is a real choice. P21 respawns after 2 seconds on an own spawn no "
        "living enemy can see. P22 adds 1 second of protection, ended by firing. P23 decides "
        "every 2 ticks, 15 decisions a second, so a 60 second match is 900 steps. P24 scores "
        "a timed deathmatch: more kills wins, equal is a draw."
    ))
    story.append(P(
        "P12 and P13 rebuild the generator's own flight, hop by hop, and wait on the ground "
        "until the tank can pay for the next hop. Walking while refuelling had been stepping "
        "bots off ledges. P14 flew 2,000 routes on 200 validation maps. 99.80 percent arrived. "
        "The three falls were hops of 21-24 fuel with a long sideways stretch. The rusher bot "
        "uses this planner. Building it costs about 112 milliseconds per map, so it is cached "
        "on the seed (B2)."
    ))
    story.append(P(
        "I18 measured the full environment at about 35,000 one-versus-one decision steps a "
        "second at 256 matches, and 20,400 in two-versus-two. Visibility is about 45 percent "
        "of that. I19 adds a PettingZoo ParallelEnv over a single match for outside tools, "
        "about 700 steps a second. Training stays on the batched MatchEnv."
    ))

    story.append(P("4. What the learner sees and does", "H1"))
    story.append(P(
        "G1 chooses structured numbers over pixels so a small network can learn. I1 makes aim "
        "one continuous angle, sampled from a Gaussian. Sixteen fixed directions, aimed perfectly, "
        "hit 46 percent of shots at real fight distances. Thirty-two directions hit 71 percent, "
        "64 hit 88 percent, 128 hit 95 percent, and a free angle hit 98 percent. The angle is "
        "not clipped. Zero is right and pi/2 is down. I2 lays the discrete part out as move "
        "(left, none, right), jetpack (off, on), and shoot (no, yes)."
    ))
    story.append(P(
        "I3 sees an enemy when a ray from the agent's centre reaches the enemy's centre or a "
        "hitbox corner. Centre-only sight called 10.6 percent of shootable enemies unseen. "
        "Centre plus corners missed 0.1 percent. I5 runs that walk for every pair at once and "
        "matches the exact Python ray on 100,000 random lines. I6 gives a 21 by 13 window of "
        "the tiles themselves. Twenty-four rays were slower and, at 10 tiles, 2.6 tiles apart, "
        "so they skipped walls. The window is exact and about ten times cheaper. I9 remembers "
        "a seen enemy for 10 seconds and otherwise hides position. I7 mirrors team B's view and "
        "mirrors its actions back, so one network plays either side. I8 always allocates two "
        "enemy slots and one ally slot, with a present flag, so the same network can play 1v1 "
        "or 2v2. The observation is 57 named numbers plus the 273 tiles of the window: 330."
    ))

    story.append(P("5. Scripted opponents", "H1"))
    story.append(P(
        "B1 defines one policy interface. act returns discrete actions and a continuous aim, "
        "in the agent's own frame. Bots may read the true state through the context. That makes "
        "them a yardstick, not a fair opponent. B2 separates the two uses. Shooting uses the "
        "agent's own sight and a 2 degree aim error. Chasing uses the true position of the "
        "nearest living enemy. Over 16 full matches the rusher killed a stationary opponent "
        "every time, 4.88 kills each. In a duel the turret, which never moves, beat the rusher "
        "more often than it lost. B3 stores a replay as the seed and the actions. The engine "
        "is deterministic, so positions do not need to be stored. A matplotlib viewer redraws "
        "the match."
    ))

    story.append(P("6. Reward and PPO", "H1"))
    story.append(P(
        "The learning code on main uses the reward in src/rl_donut/rl/rewards.py. It is a sparse "
        "match outcome plus alpha times a shaping term. A win is +1, a loss is -1, a draw is 0. "
        "Shaping pays +0.3 for a kill and -0.3 for a death, +0.003 per point of damage dealt and "
        "-0.003 per point taken, -0.0005 every decision, and -0.3 for a pit fall. One hundred "
        "damage is another 0.3, the same size as a kill. Nine hundred decisions, a full minute, "
        "cost 0.45 if nothing else happens. Alpha starts at 1 and falls linearly to 0.1 over the "
        "first 60 percent of the scheduled training steps. The point of the fade is that early "
        "updates can follow hits, and later updates have to care about the match."
    ))
    story.append(P(
        "The network is an actor-critic with a hidden width of 256 and tanh layers. Three "
        "categorical heads cover the discrete actions. A Gaussian head predicts the aim mean. "
        "The spread is one learned number, shared across states. PPO uses a clip of 0.2, a "
        "discount of 0.99, GAE lambda 0.95, four epochs, and an entropy bonus of 0.01 on the "
        "discrete heads. Advantages are normalized. Eight matches run together. An update uses "
        "128 decisions. The learner takes team A in even matches and team B in odd matches."
    ))
    story.append(P(
        "During the run in section 7 the aim spread stayed near one radian, about 59 degrees, "
        "because the entropy term included the Gaussian and the spread was initialized at that "
        "width. A standing body is 1.6 tiles tall: about 6 degrees at 15 tiles and about 3 "
        "degrees at the 30-tile rifle limit. The rifle already adds plus or minus 2 degrees. "
        "After that run the code was changed. The spread is initialized at 4 degrees and clamped "
        "to the range 1 to 6 degrees, and the entropy bonus no longer includes the aim. Those "
        "weights are not the checkpoints in section 7. A new network with the 4 degree aim, "
        "before any training, scored about one kill every 30 seconds when the enemy started in "
        "view, because the untrained mean aims to the right and the curriculum places the enemy "
        "on the right. On four official maps the same network dealt no damage."
    ))
    story.append(P(
        "When no enemy is seen or remembered, the learner no longer stands still. Sight is a "
        "line from the body, so turning the gun does not reveal anyone. The agent flies a loop "
        "of the map's reachable floors with the generator's route planner, walks to the far end "
        "of each floor, and does not shoot. The learned aim returns on the decision an enemy "
        "enters the observation. On four official maps that roam covered 22 to 39 tiles in 60 "
        "seconds and never gained a line of sight to the static enemy. The code is in "
        "src/rl_donut/rl/circulate.py. It is later than the measured run."
    ))

    story.append(P("7. The measured run", "H1"))
    story.append(P(
        "runs/spawn-curriculum-14 trains PPO against the static bot and widens the spawn gap "
        "only after a gate. Every 25 updates the current weights play 16 validation matches of "
        "60 seconds, seed 10,000, sides swapped. The gate is 8 wins out of 16. Both rungs that "
        "were actually trained keep the enemy in line of sight and inside rifle range. The run "
        "is 600 updates. It is committed with its metrics and its checkpoints."
    ))
    story.append(table(
        ["Rung", "Best check", "What happened next"],
        [
            ["6-12 tiles, enemy in sight", "Update 175: 8 of 16 wins, 1.81 kills, 273 damage", "Gate opened. Checkpoint ckpt_tiles_8.pt."],
            ["12-16 tiles, still in sight", "4 of 16 wins, about 0.6 kills and 120 damage", "Later checks fell to 1 of 16. Gate stayed shut. ckpt_tiles_14.pt is the first 4-of-16 check, update 225."],
            ["16-28 tiles, then official hidden spawns", "Not run", "The ladder stopped on the 12-16 tile rung."],
        ],
        [38 * mm, 62 * mm, W - 100 * mm],
    ))
    story.append(P(
        "Table 2. Validation during runs/spawn-curriculum-14. Each check is 16 matches of 60 seconds on val seeds.",
        "Caption",
    ))
    story.append(P(
        "The same two checkpoints were scored again, same seed and match length. The 12-16 tile "
        "weights do not beat the 6-12 tile weights on the longer maps: 3 of 16 wins and about "
        "90 damage, against 3 of 16 and about 80 damage. On the short maps the later checkpoint "
        "won 5 of 16. Wilson 95 percent intervals on 16 matches are wide. Eight of 16 has an "
        "interval of about 0.28 to 0.72. Three of 16 has an interval of about 0.07 to 0.43. "
        "The damage gap is the clearer fact: lengthening the shot from about 8 tiles to about "
        "14 tiles dropped a policy that was dealing around 200 damage to one that was dealing "
        "around 80, and further training did not recover it. Entropy at the end of the log is "
        "3.25. A new policy of this shape starts near 3.9, so the discrete actions never sharpened."
    ))
    story.append(P(
        "Mean return at update 600 is 0.15. Alpha has been fading for the whole run, which is "
        "the line on the training chart in scripts/dashboard.py. The dashboard reads metrics.jsonl "
        "from disk and refreshes every few seconds. It does not talk to the trainer. This run "
        "did not log optimizer losses, Elo, or a generalization file. Those tabs are empty on "
        "purpose."
    ))

    story.append(P("8. Code that is on main and not this result", "H1"))
    story.append(P(
        "Phases 6-10 are implemented and tested as software. They are not the numbers in section 7, "
        "and they are not approved in DESIGN.md. A reader who stops at docs/REPORT.md will see a "
        "different claim: that all twelve milestones are fulfilled, and a test-split table of 0 wins "
        "against the scripted bots. That file is an earlier sign-off. Its artifacts, runs/final and "
        "runs/sp-adaptive, are not in this clone. Do not cite it as the outcome of the curriculum."
    ))
    story.append(bullets([
        "Self-play samples a pool: recent self, older snapshots, and the permanent bots. Admission and Elo code are present. K1 asks for 60 second matches. The curriculum in section 7 did not run self-play.",
        "Styles reweight the same shaping: aggressive pays more for damage dealt, defensive pays more for damage taken, sniper adds a bonus for a hit past 15 tiles. No style run is the result above.",
        "Generalization code can score train, val, unseen seeds from 30,000, and physics scaled by 0.75 and 1.25 together. It was not the evaluation of section 7.",
        "The benchmark can play the held-out test seeds once, with Wilson intervals and a Bradley-Terry rating fit. That one-shot test was not used for section 7. The test split is still untouched by the curriculum.",
    ]))

    story.append(P("9. What is actually left", "H1"))
    story.append(P(
        "The environment can host the experiment the brief asks for. The policy that exists can "
        "hit a stationary enemy it can already see from a few tiles, and it loses that when the "
        "shot gets longer, even before the enemy is hidden. Search from the official spawns is "
        "open. Roaming the floors, without being told where the enemy is, did not create a "
        "sighting in the four maps that were checked."
    ))
    story.append(table(
        ["Next measurement", "Why"],
        [
            ["Retrain from the 6 degree aim cap, starting at 6-12 tiles", "The saved checkpoints were trained at a 59 degree spread."],
            ["Stop when validation wins stop rising", "Every rung so far peaked and then declined. The file to keep is the peak, not update 600."],
            ["A smaller step than 8 tiles to 14 tiles", "The 12-16 tile rung peaked at 4 of 16."],
            ["Official spawns, reported as they are", "Hidden starts are the actual task. A 0 is a result if it is measured."],
        ],
        [78 * mm, W - 78 * mm],
    ))
    story.append(P("Table 3. The open measurements. None of these is claimed as done.", "Caption"))

    story.append(P("10. Reproducibility", "H1"))
    story.append(P(
        "Python 3.11 or newer. Dependencies are pinned in requirements.txt, including "
        "numpy==2.5.4, torch==2.14.1, and streamlit==1.67.0. Maps load with "
        "rl_donut.mapgen.load_split. The curriculum is scripts/train_spawn_curriculum.py. "
        "The dashboard is scripts/dashboard.py. 525 tests are collected under tests/. "
        "This report was assembled from DESIGN.md, docs/PROGRESS.md, docs/PLAN.md, and the "
        "committed log runs/spawn-curriculum-14/metrics.jsonl. It does not add a measurement "
        "that those files do not contain."
    ))

    story.append(P("References in the repository", "H1"))
    story.append(bullets([
        "DESIGN.md. Approved decisions G1-G4, E1-E2, M1-M49, P1-P28, I1-I19, B1-B3, and the later notes K1-K7, A1-A5, X1-X2, D1-D2.",
        "docs/PROGRESS.md. Current measurements, 15 November 2026.",
        "docs/PLAN.md. Phase status. Phases 5-10 are code on main, not finished.",
        "docs/mapgen.md, docs/engine.md, docs/interface.md, docs/bots.md. Step-by-step guides.",
        "docs/REPORT.md. Earlier sign-off. Not the curriculum result.",
        "runs/spawn-curriculum-14/. Metrics and checkpoints for section 7.",
    ]))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
