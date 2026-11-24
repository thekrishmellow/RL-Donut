"""Teaching guide for RL-Donut: the roadmap, the training dashboard, and the play page.

    .venv/bin/python scripts/build_learning_guide.py
"""

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


pdfmetrics.registerFont(TTFont("Georgia", "/System/Library/Fonts/Supplemental/Georgia.ttf"))
pdfmetrics.registerFont(TTFont("Georgia-Bold", "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"))
pdfmetrics.registerFont(TTFont("Georgia-Italic", "/System/Library/Fonts/Supplemental/Georgia Italic.ttf"))

INK = HexColor("#1C1917")
PAPER = HexColor("#FBF7F1")
SAND = HexColor("#F4ECDF")
EMBER = HexColor("#E4572E")
EMBER_DEEP = HexColor("#9A3412")
YOU = HexColor("#24596F")
MUTED = HexColor("#6D645B")
LINE = HexColor("#E0D4C4")
ROW = HexColor("#F7F1E8")
CATCH_BG = HexColor("#F8EBE4")
GOOD_BG = HexColor("#E6EEF1")

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
OUT = "docs/RL-Donut-learning-guide.pdf"


def S(name, **kw):
    base = dict(fontName="Helvetica", fontSize=10.5, leading=15, textColor=INK, alignment=TA_JUSTIFY)
    base.update(kw)
    return ParagraphStyle(name, **base)


STYLES = {
    "cover_kicker": S("cover_kicker", fontName="Helvetica", fontSize=9, leading=12, textColor=HexColor("#F0B089"),
                      alignment=TA_LEFT),
    "cover_title": S("cover_title", fontName="Georgia", fontSize=36, leading=40, textColor=SAND, alignment=TA_LEFT),
    "cover_sub": S("cover_sub", fontName="Georgia-Italic", fontSize=14, leading=20, textColor=HexColor("#E7DCCB"),
                   alignment=TA_LEFT),
    "cover_meta": S("cover_meta", fontName="Helvetica", fontSize=9.5, leading=14, textColor=HexColor("#D6CDC2"),
                    alignment=TA_LEFT),
    "h1": S("h1", fontName="Georgia", fontSize=20, leading=24, textColor=INK, alignment=TA_LEFT, spaceBefore=2, spaceAfter=6),
    "h2": S("h2", fontName="Georgia", fontSize=14.5, leading=18, textColor=INK, alignment=TA_LEFT, spaceBefore=11, spaceAfter=4),
    "h3": S("h3", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=INK, alignment=TA_LEFT, spaceBefore=8, spaceAfter=2),
    "body": S("body", spaceAfter=7),
    "small": S("small", fontSize=9, leading=12.5, textColor=MUTED, spaceAfter=4),
    "bullet": S("bullet", alignment=TA_LEFT, leftIndent=0, spaceAfter=2),
    "th": S("th", fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=SAND, alignment=TA_LEFT),
    "td": S("td", fontSize=8.5, leading=11.4, alignment=TA_LEFT),
    "td_em": S("td_em", fontName="Helvetica-Bold", fontSize=8.5, leading=11.4, alignment=TA_LEFT),
    "call_title": S("call_title", fontName="Helvetica-Bold", fontSize=8.5, leading=11, alignment=TA_LEFT, textColor=EMBER_DEEP),
    "call_body": S("call_body", fontSize=9.2, leading=12.6, alignment=TA_LEFT, textColor=INK),
    "toc_part": S("toc_part", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=EMBER, alignment=TA_LEFT, spaceBefore=8),
    "toc_item": S("toc_item", fontName="Georgia", fontSize=12, leading=16, alignment=TA_LEFT),
}


class Bookmark(Flowable):
    """An outline entry with no drawn height."""

    def __init__(self, key, title, level=0):
        super().__init__()
        self.key = key
        self.title = title
        self.level = level
        self.width = 0
        self.height = 0

    def wrap(self, aw, ah):
        return (0, 0)

    def draw(self):
        self.canv.bookmarkPage(self.key)
        self.canv.addOutlineEntry(self.title, self.key, self.level, closed=0)


def P(text, style="body"):
    return Paragraph(text, STYLES[style])


def h1(text, key):
    return KeepTogether([Bookmark(key, text, 0), P(text, "h1"), rule()])


def h2(text, key=None, level=1):
    bits = []
    if key:
        bits.append(Bookmark(key, text, level))
    bits.append(P(text, "h2"))
    return KeepTogether(bits)


def rule():
    return HRFlowable(width="100%", thickness=1, color=EMBER, spaceBefore=0, spaceAfter=8)


def bullets(items):
    flow = []
    for item in items:
        flow.append(Paragraph("•  " + item, STYLES["bullet"]))
    return flow


def callout(kind, title, body):
    if kind == "catch":
        bg, title_color = CATCH_BG, EMBER_DEEP
        label = "THE LIMIT"
    elif kind == "good":
        bg, title_color = GOOD_BG, YOU
        label = "WHY IT STILL EARNS ITS PLACE"
    else:
        bg, title_color = SAND, EMBER_DEEP
        label = "WHY THIS IS HERE"
    title_style = ParagraphStyle("ct", parent=STYLES["call_title"], textColor=title_color)
    data = [[
        Paragraph(label + "  ·  " + title, title_style),
    ], [
        Paragraph(body, STYLES["call_body"]),
    ]]
    width = PAGE_W - 2 * MARGIN
    table = Table(data, colWidths=[width - 16])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 7),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
        ("TOPPADDING", (0, 1), (-1, 1), 1),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("LINEBEFORE", (0, 0), (0, -1), 3, title_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return KeepTogether([Spacer(1, 3), table, Spacer(1, 8)])


def table(headers, rows, widths):
    head = [Paragraph(h, STYLES["th"]) for h in headers]
    body = []
    for row in rows:
        body.append([Paragraph(str(cell), STYLES["td"]) for cell in row])
    data = [head] + body
    tbl = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), SAND),
        ("BACKGROUND", (0, 1), (-1, -1), white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, ROW]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, 0), 0, INK),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, LINE),
    ]
    tbl.setStyle(TableStyle(style))
    return tbl


def draw_cover(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(INK)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(EMBER)
    canvas.rect(0, 0, 7, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(HexColor("#2A241F"))
    canvas.rect(0, 0, PAGE_W, 28 * mm, fill=1, stroke=0)
    canvas.setFillColor(HexColor("#F0B089"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(MARGIN, 14 * mm, "Phases 1-4 are finished. Learning code is on main. Training is not finished.")
    canvas.restoreState()


def draw_body(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.setFillColor(EMBER)
    canvas.rect(0, PAGE_H - 5, PAGE_W, 5, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(MARGIN, PAGE_H - 18, "RL-Donut   ·   a learning guide")
    canvas.drawRightString(PAGE_W - MARGIN, 16, str(doc.page))
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, 28, PAGE_W - MARGIN, 28)
    canvas.restoreState()


def story():
    W = PAGE_W - 2 * MARGIN
    flow = []

    flow.append(Spacer(1, 16 * mm))
    flow.append(P("A TEACHING GUIDE", "cover_kicker"))
    flow.append(Spacer(1, 6))
    flow.append(P("RL-Donut,<br/>taught in the order<br/>the project was built.", "cover_title"))
    flow.append(Spacer(1, 8))
    flow.append(P(
        "What each idea is for, why it is in the design, where it is weak, "
        "and how to read both screens: the training dashboard and the human-versus-agent page.",
        "cover_sub",
    ))
    flow.append(Spacer(1, 10))
    flow.append(P("24 November 2026  ·  Written from the code on main and the logs in runs/.", "cover_meta"))
    flow.append(Spacer(1, 46 * mm))
    flow.append(P("WHAT YOU CAN DO AFTER THIS", "cover_kicker"))
    flow.append(Spacer(1, 6))
    flow.append(P("Walk the ten phases and say what each one protects.", "cover_sub"))
    flow.append(Spacer(1, 3))
    flow.append(P("Read a training chart without mistaking damage for a win.", "cover_sub"))
    flow.append(Spacer(1, 3))
    flow.append(P("Use the play page and know which fighter is the learner.", "cover_sub"))
    flow.append(NextPageTemplate("body"))
    flow.append(PageBreak())

    # ----- how to read -----
    flow.append(h1("How to read this", "read"))
    flow.append(P(
        "This is a lesson, not a status slide. Read Part I as a path: each step exists because the "
        "step before it made the next one possible. Part II is the training dashboard at "
        "http://127.0.0.1:8501. Part III is the play page at http://127.0.0.1:8765. "
        "They can run at the same time. They do not share a log."
    ))
    flow.append(P("Three facts, before any number:"))
    flow.extend(bullets([
        "<b>Phases 1 to 4 are finished.</b> Maps, the game, the 330-number observation, the scripted bots, replays, and the viewer.",
        "<b>Phases 5 to 10 have code on main.</b> They are not marked finished. Training has started and has not cleared the real spawns.",
        "<b>docs/REPORT.md is an earlier sign-off.</b> The measurements in this guide come from runs/spawn-curriculum-14/metrics.jsonl and from the play code. Use docs/PROGRESS.md when the two disagree; this guide follows the log.",
    ]))
    flow.append(Spacer(1, 6))
    flow.append(P("What is inside", "toc_part"))
    for line in [
        "1.  One match, in plain language",
        "2.  The roadmap, and why the order is the design",
        "3.  Maps that a jetpack can actually cross",
        "4.  The engine: the laws of the cave",
        "5.  What the agent sees, and what it is allowed to do",
        "6.  The reward, and PPO, the update that moves the weights",
        "7.  The spawn ladder, and the run that was actually measured",
        "8.  Self-play, and why copying your buttons is dangerous",
        "9.  The training dashboard, control by control",
        "10. The play page, control by control",
    ]:
        flow.append(P(line, "toc_item"))

    # ----- one match -----
    flow.append(h1("1. One match, in plain language", "match"))
    flow.append(P(
        "RL-Donut is a two-team deathmatch in the shape of Mini Militia, built so a learning agent "
        "can play millions of them. One match lasts 60 seconds in training. The play page lets you "
        "choose 12, 20, or 60. The cave is 96 tiles wide and 48 tall. Physics runs at 30 ticks a "
        "second. The fighters decide every 2 ticks, so a full minute is 900 decisions."
    ))
    flow.append(P(
        "Each fighter has 100 health, a jetpack, and a rifle. A shot does 15 damage, so a kill is "
        "7 hits. The rifle reaches 30 tiles and scatters by up to 2 degrees. Health returns at "
        "10 points a second once 3 seconds pass with no damage. That last rule is easy to miss and "
        "it changes the whole game: a burst that deals 50 damage and then stops is not a kill. "
        "The body heals, and the score does not move."
    ))
    flow.append(P(
        "You win by having more kills when the clock ends. Equal kills is a draw. Falling into a "
        "pit can count for the last opponent who hurt you, if that hit was recent. Agents pass "
        "through each other. They do not pass through rock. There is no jump and no fall damage. "
        "The jetpack is the only way up."
    ))
    flow.append(callout(
        "why",
        "A game you can simulate, not a game you have to watch",
        "Pixels would need a camera network and a huge number of matches before the agent noticed "
        "a doorway. Numbers — health, fuel, a small window of tiles — let a small network start "
        "learning on a laptop. The cost of that choice is that the agent never sees what you see. "
        "It sees the 330 numbers in section 5. If those numbers omit a fact, no amount of training invents it.",
    ))

    # ----- roadmap -----
    flow.append(h1("2. The roadmap, and why the order is the design", "roadmap"))
    flow.append(P(
        "The phases are a dependency chain. A later phase that skips an earlier promise trains "
        "on a lie: a map the jetpack cannot leave, a shot the eye cannot see, a reward that pays "
        "for the wrong event. This is the path, in the order it has to be walked."
    ))
    rows = [
        ["0", "Setup", "Done", "One Python environment, exact library versions. The map generator’s random numbers change if NumPy changes, so the pin is part of the science."],
        ["1", "Maps", "Done", "A seed becomes a cave. If this is unfair or uncrossable, every later win rate is about the map, not the agent."],
        ["2", "Engine", "Done", "Physics, rifle, health, the clock. This is the world the reward will score."],
        ["3", "Eyes and hands", "Done", "The 330 numbers and the four controls. This is the contract between the game and the learner."],
        ["4", "Teachers", "Done", "Scripted bots, replays, a viewer. You need an opponent and a way to watch before a learner is worth running."],
        ["5", "PPO vs a bot", "Code, partial training", "Reward, the network, one learner against a fixed teacher. The measured ladder lives here."],
        ["6", "Self-play", "Code, early runs", "The opponent becomes a copy of the learner, then a pool of past copies. This is where a policy stops memorising one bot."],
        ["7", "Styles", "Code", "Aggressive, defensive, sniper opponents, so one trick does not look like skill."],
        ["8", "Unseen maps", "Code", "The held-out caves, and physics scaled together. The question is whether the policy travels."],
        ["9", "Dashboard", "The page exists", "A window onto the log. It does not train anything."],
        ["10", "Report", "Open", "A write-up that matches the measurements. The older report does not."],
    ]
    flow.append(table(
        ["Phase", "Name", "Where it stands", "Why it cannot be skipped"],
        rows,
        [36, 78, 88, W - 202],
    ))
    flow.append(Spacer(1, 8))
    flow.append(P(
        "The handoff in the middle is deliberate. One person built the cave and the contract. "
        "The other built the learner against that contract. A shared test fails if either side "
        "changes the shape of an observation or an action without the other knowing."
    ))
    flow.append(callout(
        "good",
        "The unfinished half is still a real RL project",
        "The environment is the hard part of a research setup: seeded worlds, a fast batch, a legal "
        "action, a reward with a reason, and an evaluation that can say no. PPO, the curriculum, "
        "the opponent pool, and Elo are the learning half, and they step that environment. "
        "What is missing is a policy that wins from the official, hidden spawns. That is a result "
        "gap, not an empty repository.",
    ))

    # ----- maps -----
    flow.append(h1("3. Maps that a jetpack can actually cross", "maps"))
    flow.append(P(
        "A map is a function of a seed. The same seed, the same NumPy build, the same generator "
        "version, the same cave. Training, validation, and the final test are three frozen sets: "
        "9,997 train maps, 1,000 validation, 1,000 held out. Three train seeds failed every retry "
        "and were skipped. Validation and test had no failures. The files live in data/maps/ so a "
        "clone does not regenerate a different world."
    ))
    flow.append(h2("Noise, then a decision", "noise"))
    flow.append(P(
        "Perlin noise, written in NumPy, fills the left half of the map with smooth hills. "
        "Four layers are added, each twice as fine and half as strong (fractional Brownian motion: "
        "lacunarity 2, persistence 0.5). The blend between corners is a quintic curve, because a "
        "simpler blend leaves a crease on every cell edge. Cells are stretched sideways, 16 tiles "
        "by 10, so the hills become ledges a side-view game can stand on."
    ))
    flow.append(P(
        "The hills become rock and air by a quantile: the rockiest fraction of tiles, drawn between "
        "40% and 55%, is rock. A fixed cutoff was measured and dropped. Aimed at 40% rock, it "
        "produced anywhere from 31% to 50% across 1,000 seeds, so two maps with the same setting "
        "were different games. A height bias, also drawn per seed, pushes rock toward the bottom. "
        "Ground below, open air above, and not the same ground on every map."
    ))
    flow.append(h2("Symmetry is the fairness proof", "mirror"))
    flow.append(P(
        "The left half is mirrored to the right, and only then turned into tiles and smoothed. "
        "Smoothing first would treat the centre seam as a wall and grow a spine down the middle. "
        "A 180-degree rotation was the other kind of symmetry. Combined with the height bias it "
        "would give one team the floor and the other the sky. Left-right mirroring gives both "
        "teams the same cave, reflected. Team B’s observation is mirrored the same way, so one "
        "network can play either side."
    ))
    flow.append(h2("A tile is not the same as a place you can stand", "clearance"))
    flow.append(P(
        "The body is 0.8 by 1.6 tiles. Connectivity is a flood fill, four directions, only where a "
        "2 by 3 clearance box fits in air. On raw noise that box left 64% of the air usable, "
        "against 85% for a box the body’s own size. The gap is why the later steps exist: three "
        "passes of a cellular-automaton smoother (rough single-tile jags fell from about 72 per "
        "map to about 1), then tunnels."
    ))
    flow.append(P(
        "Disconnected caves are joined by the cheapest path. Each step costs 1, plus 12 for every "
        "rock tile dug, so the path reuses air that is already there. Digging straight up costs "
        "three times as much, which turns shafts into staircases. A vertical shaft taller than the "
        "jetpack can climb is a one-way trap. That single cost took fully reachable maps from 78% "
        "to about 94%, and the amount of rock dug stayed near 39 tiles per map."
    ))
    flow.append(h2("Fuel is part of the map, not only part of the physics", "fuel"))
    flow.append(P(
        "The generator has its own small flight model, checked later against the real engine. "
        "Climbing one tile costs 2 fuel, sideways costs 1, falling is free. A map is kept when "
        "every floor you can reach, you can also leave, and at most 2% of floor is stranded. "
        "Demanding 100% would throw away maps over a two-tile notch nobody walks. Platforms are "
        "placed only where that model finds a stranded floor, or as a little cover. A ledge is "
        "not allowed to sit on a pit and cancel it."
    ))
    flow.append(P(
        "Pits are 0, 1, or 2 per map, 3 to 6 tiles wide, cut up from the bottom. They have to open "
        "into the main walkable region. If the requested pit does not fit, the map is retried. "
        "Quietly turning “one pit” into “no pit” would have happened on about 60% of maps, and a "
        "curriculum that asked for pits would not have received them."
    ))
    flow.append(h2("Spawns are hidden on purpose", "spawns"))
    flow.append(P(
        "Each team gets 6 floor spawns, mirrored. No line of sight to an enemy spawn, at least "
        "6 tiles from a pit, at least 16 tiles from the centre, at least 6 tiles between teammates. "
        "The line of sight is an exact grid walk (Amanatides–Woo). Sampling the line every quarter "
        "tile missed 46 corner clips out of 10,000. The rifle uses the same function, so “cannot "
        "see” and “cannot be shot” are one fact."
    ))
    flow.append(callout(
        "catch",
        "The fair spawn is also the hardest lesson",
        "Official spawns hide the teams from each other. A policy that has only learned to shoot "
        "a body it can already see will stand at its spawn and deal no damage. That is why training "
        "later moves the two fighters closer, in sight, and only widens the gap after a win rate. "
        "The maps are still the real caves. Only the starting tiles change. The measured run never "
        "reached the official rung.",
    ))
    flow.append(callout(
        "good",
        "Owning the generator is what makes a result comparable",
        "A noise library, a different NumPy, or a seed plus one would each rewrite old maps. "
        "Named random streams (terrain, tunnels, pits, spawns, settings) mean a new use of "
        "randomness cannot move an old cave. Retries use an attempt number. Golden hashes and a "
        "committed cache mean the validation win rate is about the policy.",
    ))

    # ----- engine -----
    flow.append(h1("4. The engine: the laws of the cave", "engine"))
    flow.append(P(
        "The engine has no graphics. It updates many matches at once as NumPy arrays of shape "
        "(matches, agents): position, speed, health, fuel. Measured on the machine this was built "
        "on, 64 matches together ran about 470,000 ticks a second. Ordinary Python objects managed "
        "about 48,000, and one array per match about 12,000, because a NumPy call has a fixed cost "
        "that swamps a tiny array. The full environment, vision included, later measured about "
        "35,000 match-steps a second in 1v1 at a batch of 256."
    ))
    flow.append(h2("Motion that cannot skip a wall", "motion"))
    flow.append(P(
        "Each tick updates speed, then position (semi-implicit Euler). Gravity is 25 tiles per "
        "second squared. Thrust is 40. Run speed is 6 tiles a second, set directly, on the ground "
        "and in the air. There is no horizontal momentum. That is the Mini Militia feel, and it "
        "makes a route check possible: the same input always makes the same arc."
    ))
    flow.append(P(
        "Collision is one axis at a time, horizontal then vertical. If the leading edge enters "
        "rock, the body is placed on the tile edge and that speed becomes zero. This is exact "
        "while nothing moves a full tile in one tick. The speed limits keep the fastest motion at "
        "0.67 tiles per tick. A rising speed limit exists because, without it, a full tank reached "
        "1.27 tiles per tick and could tunnel through the one-tile border."
    ))
    flow.append(P(
        "One full tank climbs 35.7 tiles. The maps only required 12. Level flight is 30 tiles. "
        "The maps required about 29. A scripted flier completed 99.80% of 2,000 routes the "
        "generator claimed were possible. Those three numbers are the bridge between “the map "
        "says you can” and “the game lets you.”"
    ))
    flow.append(h2("The rifle and the clock", "rifle"))
    flow.append(P(
        "A shot is a ray from the body centre, plus a uniform scatter of up to 2 degrees, out to "
        "30 tiles. It hits the nearest enemy body it enters, unless rock is in the way. The "
        "magazine holds 30 rounds, then a 1.5 second reload. Hits that land on the same tick are "
        "applied together, so a trade kills both. Resolving them in agent order would hand every "
        "tie to one side, and a learner would find that."
    ))
    flow.append(P(
        "After death, respawn takes 2 seconds at a hidden spawn, with 1 second of protection that "
        "ends early if you fire. The match is deterministic: the same seed and the same actions "
        "replay the same score. A replay file can therefore store the seed and the buttons, not a video."
    ))
    flow.append(callout(
        "catch",
        "Regeneration punishes a policy that chips and wanders off",
        "Ten health a second, after three quiet seconds, erases a sloppy fight. The curriculum’s "
        "best close-range checkpoint dealt about 273 damage in a 60-second validation match and "
        "still only won 8 of 16. Damage without a finished kill is a near miss, and the reward "
        "section is where that near miss gets paid anyway.",
    ))

    # ----- observation -----
    flow.append(h1("5. What the agent sees, and what it is allowed to do", "obs"))
    flow.append(P(
        "Every decision, the policy receives 330 numbers and returns four controls."
    ))
    flow.append(h2("The hands", "hands"))
    flow.append(P(
        "Move is left, stay, or right. Jetpack is on or off. Shoot is on or off. Aim is one "
        "continuous angle: 0 points right, and the angle grows downward because y grows downward. "
        "Sixteen fixed aim directions, the first design, hit 46% of perfectly aimed shots at real "
        "fight distances. A free angle hit 98%. The network predicts the mean angle. A learned "
        "spread is added when the action is sampled, and that spread is clamped between 1 and 6 "
        "degrees. The rifle already scatters by 2. A standing body at rifle range is only a few "
        "degrees tall, so a spread of a radian — the old starting point, about 57 degrees — puts "
        "almost every shot off the body."
    ))
    flow.append(h2("The eyes", "eyes"))
    flow.append(P(
        "Sight is a line from the body, not a cone in front of the gun. Turning the rifle does "
        "not find anyone. An enemy is seen when a clear line reaches their centre or any corner "
        "of the hitbox. Centre only missed 10.6% of enemies a shot could still hit. Corners as "
        "well miss about 0.1%. In a 2v2, teammates share what they see."
    ))
    flow.append(P(
        "The 330 numbers are 57 named features plus a 21 by 13 window of tiles around the body, "
        "row by row, rock or air. The named part holds your own state, two enemy slots, one "
        "teammate slot, and the score and the clock. An enemy’s position is included only while "
        "they are known, and it is scaled by 30, the rifle range, so the numbers stay near a "
        "small range. A sighting is remembered for 10 seconds, then forgotten. Team B receives "
        "the mirror image, including a mirrored aim, so the same weights play both colours."
    ))
    flow.append(P(
        "A ring of distance rays was the alternative to the tile window. At 10 tiles out, 24 rays "
        "are 2.6 tiles apart and skip walls. The window is the tiles themselves. It is exact inside "
        "its rectangle, and it measured about ten times cheaper."
    ))
    flow.append(callout(
        "catch",
        "The policy is blind to what the critic is allowed to know",
        "During training, the value network may see the true state of the match, including a body "
        "behind a wall. The policy that presses the buttons cannot. That split is standard: the "
        "critic’s job is to score the future, and a better score makes the update less noisy. "
        "It does not give the fighter wall-hack eyes at match time. On the play page, the red "
        "hunter is a different case, and Part III says so.",
    ))
    flow.append(P(
        "There is also a PettingZoo wrapper around one match, for outside tools. It passes that "
        "library’s own tests. Training does not use it. One match at a time was about 700 steps "
        "a second, against about 35,000 for the batched environment."
    ))

    # ----- reward and PPO -----
    flow.append(h1("6. The reward, and the update that moves the weights", "ppo"))
    flow.append(h2("What the agent is paid", "reward"))
    flow.append(P(
        "The reward is a sparse result plus a fading shaping term. Win is +1, loss is -1, draw is 0, "
        "paid when the match ends. While the match is running, a kill is +0.3 and a death is -0.3. "
        "Damage dealt is +0.003 per point, so 100 damage is +0.30, the same size as a kill. Damage "
        "taken is the mirror. Every decision costs 0.0005, which is -0.45 over a 900-decision minute. "
        "A pit fall is -0.3."
    ))
    flow.append(P(
        "Shaping is multiplied by alpha. Alpha starts at 1 and falls to 0.1 across the first 60% of "
        "the scheduled updates, then stays there. In runs/spawn-curriculum-14 it is 0.50 at update "
        "200 and 0.10 from update 375 onward. Early on, the agent is paid to touch the fight. Later, "
        "the win and the loss are most of the cheque."
    ))
    flow.append(callout(
        "catch",
        "Shaping can pay for a fight that does not win",
        "A hundred damage equals one kill in the shaping term, and regeneration deletes that damage "
        "if the shots stop. A policy can look busy — the mean return on the dashboard rises — while "
        "the validation win rate does not. See-enemy and approach bonuses are not in this reward. "
        "They were considered and left out, because a bonus for standing near a body teaches camping.",
    ))
    flow.append(h2("The network", "net"))
    flow.append(P(
        "The model is an actor-critic. Two separate towers, each two layers of 256 units with a "
        "tanh bend, read the 330 numbers. The critic tower ends in one number: how good this "
        "situation looks. The actor tower splits into four heads. Three are categorical: move "
        "(3 choices), jet (2), shoot (2). The fourth is a Gaussian: a mean aim angle, plus one "
        "spread shared by every situation and clamped to 1–6 degrees. Weights start from an "
        "orthogonal initialisation, with the action heads kept small so the first policy is "
        "close to uniform. A uniform choice on the three button heads has entropy "
        "log 3 + log 2 + log 2, about 2.48 nats, and that is the maximum those heads can "
        "reach. The old logs also added the entropy of the aim Gaussian. At a spread of "
        "about one radian that extra term is about 1.4, which is why a fresh policy in "
        "those files sits near 3.9. The current code leaves the aim cloud out of the "
        "entropy number. A later log near 2.0 is the button heads only, and it is not "
        "“half as random” as a 3.5 from the older file."
    ))
    flow.append(h2("PPO, in the order the maths actually runs", "ppo-steps"))
    flow.append(P(
        "Proximal policy optimization is the update. One cycle, called an update on the dashboard, "
        "does this:"
    ))
    flow.extend(bullets([
        "<b>Roll out.</b> The current policy plays a batch of matches for a fixed number of decisions. Each decision stores the observation, the action, the probability of that action, the reward, and the critic’s guess.",
        "<b>Score the future.</b> Generalized advantage estimation (GAE) walks backward through the rollout. Gamma is 0.99, so a reward 100 steps away is heavily discounted. Lambda is 0.95, which blends a one-step surprise with a longer look. The result is an advantage: did this action beat the critic’s expectation?",
        "<b>Fit, but not too far.</b> For 4 epochs over 4 minibatches, the policy is nudged to make high-advantage actions more likely. The probability ratio is clipped at 0.2, so one lucky shot cannot multiply a weight by ten. The critic is trained to guess the return, with weight 0.5. An entropy bonus of 0.01 pays the discrete heads to keep more than one button available. Gradients are clipped at 0.5.",
        "<b>Write a line.</b> The runner records the average of the last 50 finished episode returns, the entropy after the update, and alpha. A validation check, on a timer, plays held-out matches and records wins, kills, and damage.",
    ]))
    flow.append(P(
        "The clip is the “proximal” in the name. A policy gradient without it can take a step so "
        "large that the next rollout is nonsense, and the learner has no copy of the weights that "
        "still worked. The entropy bonus is a different kind of brake. It stops the three button "
        "heads from collapsing to “always shoot, never move” before the caves have been seen. "
        "It does not, by itself, aim the gun. Aim is a separate Gaussian, and paying entropy on "
        "that Gaussian was teaching the spread to widen. The bonus now covers the discrete heads only."
    ))
    flow.append(callout(
        "why",
        "Entropy on the dashboard is a health light, not a score",
        "On spawn-curriculum-14 the number includes the aim cloud, so it lives in the mid 3s. "
        "Update 175 logs 3.47, against about 3.9 for a fresh policy under that same definition. "
        "The policy had learned something — 8 wins out of 16 — and the cloud had not collapsed. "
        "A reading near 0 would mean one button in every situation. The current trainer no longer "
        "adds the aim Gaussian, so compare a new run with 2.48, the discrete maximum, and do not "
        "line it up against 3.47.",
    ))

    # ----- curriculum -----
    flow.append(h1("7. The spawn ladder, and the run that was measured", "ladder"))
    flow.append(P(
        "runs/spawn-curriculum-14 trains PPO against a static target bot. The bot stands where it "
        "spawned. The caves are real validation caves. The two start tiles are moved so the lesson "
        "has a chance. The gate is half of 16 matches: 8 wins. The ladder is:"
    ))
    flow.append(Spacer(1, 2))
    flow.append(table(
        ["Rung", "Gap", "Sight", "What the fighter must learn"],
        [
            ["tiles_8", "6 to 12 tiles", "Required", "The body is in rifle range and in sight. The lesson is aim and trigger."],
            ["tiles_14", "12 to 16 tiles", "Required", "Still in sight, a smaller target in degrees. The gate never opened."],
            ["tiles_20", "16 to 28 tiles", "Required", "Inside or near the edge of rifle range. Not reached in this run."],
            ["tiles_40", "32 to 48 tiles", "Not required", "You must travel before you can shoot. Not reached."],
            ["official", "The cave’s own spawns", "Hidden", "Search, then fight. Not reached."],
        ],
        [62, 88, 70, W - 220],
    ))
    flow.append(Spacer(1, 8))
    flow.append(P(
        "The log has 600 updates. The first seven validation checks are the 6–12 tile rung. "
        "The rest are 12–16. curriculum.json records the stop: final rung tiles_14, official "
        "spawns not finished, gate 50%."
    ))
    flow.append(h2("The close rung, which is the best result so far", "best"))
    flow.append(table(
        ["Update", "Wins", "Kills", "Damage", "Entropy"],
        [
            ["25", "7 / 16", "1.00", "158", "3.68"],
            ["125", "7 / 16", "1.38", "186", "3.22"],
            ["150", "7 / 16", "1.62", "216", "3.47"],
            ["175", "8 / 16", "1.81", "273", "3.47"],
        ],
        [W / 5.0] * 5,
    ))
    flow.append(Spacer(1, 6))
    flow.append(P(
        "Update 175 is the checkpoint saved as ckpt_tiles_8.pt, and it is also ckpt_best.pt for "
        "that run. Eight wins out of sixteen is exactly the gate, so the ladder stepped forward. "
        "A Wilson interval on 8 out of 16, at 95%, runs from about 0.28 to 0.72. The gate opened "
        "on a noisy measurement. That does not erase the trend — wins, kills, and damage were "
        "rising together from update 50 to 175 — and it does mean “half” is a thin claim."
    ))
    flow.append(h2("The next rung, which stalled", "stall"))
    flow.append(P(
        "On 12–16 tiles the best win count is 4 out of 16, and it happens more than once. "
        "The first time is update 225: 4 wins, 0.31 kills, 78 damage. A later 4-win check, "
        "update 325, has more damage: 0.62 kills and 120 damage. ckpt_tiles_14.pt is the first "
        "4-win checkpoint, update 225, not the highest-damage one. After update 325 the checks "
        "fall back toward 1, 2, and 3 wins. The gate stays shut. The run ends at update 600 "
        "still on tiles_14, with entropy 3.25 and alpha already at its floor of 0.1."
    ))
    flow.append(callout(
        "catch",
        "Those checkpoints were aiming with a spread of about 59 degrees",
        "The 6-degree clamp and the decision to stop paying entropy on the aim spread came after "
        "this run. A standing body at 14 tiles is about 7 degrees tall. A 59-degree cloud of "
        "shots mostly misses, and regeneration cleans up the hits that land. Loading one of "
        "these files today clamps the spread at play time, which changes the fighter from the "
        "one that was trained. The weights inside the file were optimised for the wide cloud.",
    ))
    flow.append(callout(
        "good",
        "The ladder is the right shape of lesson even though this climb stopped",
        "A policy thrown onto hidden official spawns deals no damage, so there is nothing for "
        "PPO to reinforce. Starting in sight, at a distance where a correct aim is a kill, gives "
        "the advantage estimator a real signal. The close rung’s curve — more wins, more kills, "
        "more damage, together — is what a working rung looks like. The next rung shows the "
        "other shape: a noisy 4 out of 16 that never becomes 8. Stopping there, instead of "
        "declaring the official spawn solved, is the curriculum doing its job.",
    ))
    flow.append(h2("The teachers that exist before any learner", "bots"))
    flow.append(P(
        "Four scripted policies sit under the learner. Random almost never scored across 128 "
        "matches; beating it only proves that shots land. The static bot stands still. The turret "
        "stands still and shoots what it sees, with 2 degrees of aim error and a short reaction "
        "delay. The rusher flies the generator’s own route planner toward the nearest living "
        "enemy — it is allowed the true position, which a learned policy is not — and shoots on "
        "line of sight. Routes are cached per map. Against a stationary opponent the rusher won "
        "all 16 test matches, 4.88 kills each. In a duel the turret won more often than the rusher: "
        "moving is not the same as winning."
    ))
    flow.append(P(
        "The measured curriculum used the static bot. That choice isolates aim. It also means the "
        "policy never had to lead a moving target. The play page, later, uses the rusher as the "
        "body you fight, for a different reason: a learned policy that waits on its spawn is not "
        "a match."
    ))

    # ----- self-play -----
    flow.append(h1("8. Self-play, and why copying your buttons is dangerous", "selfplay"))
    flow.append(P(
        "Self-play means the opponent is a copy of the policy, frozen at the start of the update, "
        "then refreshed. Both sides move and shoot. There is no standing target. The code for an "
        "opponent pool and for prioritized fictitious self-play is on main: sometimes the latest "
        "self, sometimes a harder past version, sometimes a fixed bot, so a new trick does not "
        "erase the ability to beat an old style. Elo, anchored to the random bot, is there so two "
        "runs can be compared. Those pieces are not what spawn-curriculum-14 ran."
    ))
    flow.append(P(
        "A later log, runs/genuine-selfplay/metrics.jsonl, has 186 updates and walks tiles_8, "
        "tiles_14, and tiles_20. It records mean return, entropy, an aim loss, and how many human "
        "rows were used. It does not record a validation win rate. On that file the return stays "
        "small, and by the last line (update 186, stage tiles_20) the mean return is about -0.05 "
        "and entropy is about 2.0. Read that dashboard tab as a health log of the update, not as "
        "a championship result."
    ))
    flow.append(h2("The failure that shaped the play-page lesson", "nan"))
    flow.append(P(
        "The obvious way to learn from a person is to raise the probability of the buttons they "
        "pressed. For the aim head, that probability is a Gaussian whose spread is about 6 degrees. "
        "The gradient of that log-probability divides by the square of the spread. A human aim that "
        "is off by a large angle produces a huge step. Adam then writes non-finite weights, and "
        "the next categorical head is full of NaNs. That happened in self-play, around the point "
        "where aim imitation was mixed into PPO."
    ))
    flow.append(P(
        "The repair, which the play page uses, has four parts. Rock presses are dropped: if the "
        "move or the jet points into a neighbouring rock tile in the 21 by 13 window, that decision "
        "is not a lesson. When an enemy is actually seen, the aim target is the line to their "
        "offset; a human aim already close to that line is kept. The aim loss is squared angle "
        "error, not the Gaussian log-probability. Discrete buttons, when they are copied, are "
        "copied with a coefficient of 0.05. If any weight stops being a finite number, the update "
        "restores the weights from the start of the step."
    ))
    flow.append(callout(
        "catch",
        "A copied button is not a tactic",
        "Early behaviour cloning reproduced wall-pushes, because the log recorded the key and not "
        "whether the cave allowed it. It also could not be trusted once the aim Gaussian got narrow. "
        "The play page therefore does not treat your match as a script to memorise. It filters the "
        "rows, plays a short self-play practice, and only lightly imitates the legal buttons. "
        "Section 10 says exactly which of the four updates does the imitating.",
    ))
    flow.append(callout(
        "good",
        "A frozen copy is a real opponent",
        "Training against a statue teaches the gun. Training against yesterday’s weights teaches "
        "the rest: a target that moves, a target that shoots back, a target that punishes a habit "
        "you just learned. Freezing the copy for one update keeps the opponent stable while the "
        "learner steps. Refreshing it every update keeps the lesson from going stale. The pool "
        "and Elo exist so that “I beat my latest self” cannot be mistaken for “I beat the game.”",
    ))

    # ----- dashboard -----
    flow.append(h1("9. The training dashboard, control by control", "dash"))
    flow.append(P(
        "Open http://127.0.0.1:8501. The process is "
        "<font face='Courier'>scripts/dashboard.py</font>, a Streamlit page. It does not import "
        "PyTorch, the trainer, or the game. It reads files. Every 3 seconds it looks again, so a "
        "training run that is still writing shows up without a restart."
    ))
    flow.append(P(
        "It looks in two folders. <font face='Courier'>runs/</font> holds one directory per "
        "training attempt. <font face='Courier'>outputs/</font> holds pictures and GIFs from the "
        "preview scripts. The play page writes <font face='Courier'>runs/human-play/</font>, and "
        "that folder has checkpoints, demos, and lessons. It does not write metrics.jsonl. "
        "Select it in the sidebar and the page will tell you the log is empty. That is accurate. "
        "Your matches live on the other screen."
    ))
    flow.append(h2("The frame around the charts", "dash-frame"))
    flow.append(table(
        ["What you see", "What it reads", "What it is for"],
        [
            ["Title “RL-Donut” and the caption", "Nothing from a run", "A reminder that this page is a viewer of runs/ and outputs/, refreshing every 3 seconds."],
            ["Sidebar, “Select run”", "Every subdirectory of runs/ that list_runs accepts", "One training attempt at a time. The names are the folder names: spawn-curriculum-14, genuine-selfplay, the phase5 experiments, human-play."],
            ["“Log updated” caption", "The modification time of metrics.jsonl", "Proof the file changed. A clock time here is the file, not a claim about whose match it was."],
            ["Run configuration, collapsed", "config.json, plus curriculum.json and a slice of clone.json when those files exist", "The settings that produced the lines. For spawn-curriculum-14 there is no config.json. Opening the box shows curriculum.json: gate 0.5, final rung tiles_14, 600 updates, official spawns not finished."],
            ["Empty-log warning", "metrics.jsonl missing or with no valid rows", "The run folder exists and has not logged an update. human-play hits this on purpose."],
        ],
        [120, 150, W - 270],
    ))
    flow.append(Spacer(1, 8))
    flow.append(h2("The five numbers across the top", "dash-metrics"))
    flow.append(P(
        "These are the latest row, except the win rate, which is the best in the file. "
        "They are a snapshot. The charts below them are the history."
    ))
    flow.append(table(
        ["Card", "When it appears", "How to read it"],
        [
            ["Updates", "Always, from the last row’s update index", "How many PPO updates have been written. spawn-curriculum-14 ends at 600. This is not a match count and not a wall-clock time."],
            ["Rung", "When the log has a rung column", "The lesson the latest update was on. tiles_8 means the 6–12 tile, in-sight placement. tiles_14 means 12–16. Seeing tiles_14 in the card means the close gate already opened."],
            ["Env steps", "When there is no rung, and env_steps is logged", "How much simulated time the learner has consumed. A substitute label, not a better score."],
            ["Rows", "When neither rung nor env steps is present", "How many log lines exist. A last resort so the card is never blank."],
            ["Mean return", "Always", "Average total reward of the last 50 matches that finished inside the trainer. Win, damage, step cost, and falls are already mixed in. A rise can be chip damage. It is not a win rate."],
            ["Entropy", "Always", "How spread out the sampled actions still are, in nats, after the latest update. Read it in the units that run logged. spawn-curriculum-14 includes the aim cloud: fresh is near 3.9, and update 175 is 3.47. The current code logs the three button heads only, whose maximum is about 2.48. genuine-selfplay ends near 2.0 on that newer scale. Near 0, on either scale, means one action everywhere."],
            ["Best val win rate", "When any row has eval_win_rate", "The highest validation win rate in the whole file, not the latest. For spawn-curriculum-14 this card reads 50%, which is update 175, 8 of 16. Later rows are worse. The card will not fall when the policy gets worse. Look at the validation tab for the path."],
            ["Policy loss", "Only when the file has no win rate", "The clipped policy-gradient loss of the latest update. A small number is not automatically better. This card is the stand-in for runs that never evaluate."],
        ],
        [100, 145, W - 245],
    ))
    flow.append(Spacer(1, 8))
    flow.append(P(
        "Under the cards, if curriculum.json exists, a line says where the ladder stopped, how "
        "many updates ran, and what the gate was. For the measured run that line is the short "
        "version of section 7: stopped on tiles_14, 600 updates, gate 50%."
    ))
    flow.append(h2("Training tab", "dash-train"))
    flow.append(P(
        "A line chart of whichever of these columns the log actually has: mean return, mean "
        "length, alpha, entropy. spawn-curriculum-14 has return, alpha, and entropy. It does not "
        "log mean length, so that line is absent. Under the chart, the distinct rungs are written "
        "in order. For this run that sentence is tiles_8, then tiles_14."
    ))
    flow.append(table(
        ["Line", "What a point is", "The shape you want, and the shape that lies"],
        [
            ["Mean return", "Average reward of the last 50 finished episodes, one point per update", "A slow rise while the win rate also rises is a real lesson. A rise while wins stay at 1 or 2 is shaping: damage and activity, healed away before the scoreboard."],
            ["Mean length", "Average decisions in those finished episodes", "Useful when episodes can end early. A deathmatch with a fixed clock makes this almost flat. This run does not draw it."],
            ["Alpha", "The weight on the shaping term", "Starts at 1, falls to 0.1, stays. In this log it is 0.5 at update 200 and 0.1 from update 375. After it flattens, a return that is still moving is moving because of wins, deaths, and the small leftover shaping."],
            ["Entropy", "Spread of the actions this file chose to log", "A gentle decline is the policy committing. A crash toward zero early in a rung is collapse. This run stays in the mid 3s and ends at 3.25, on the old scale that includes the aim cloud. Most of that height is the wide gun, not a refusal to pick a button."],
        ],
        [78, 155, W - 233],
    ))
    flow.append(Spacer(1, 8))
    flow.append(h2("Losses tab", "dash-loss"))
    flow.append(P(
        "This tab draws policy loss, value loss, approximate KL, explained variance, and clip "
        "fraction, if the log has them. spawn-curriculum-14 did not write those columns. The tab "
        "says so: “This run did not log optimizer losses.” That is a property of the log, not a "
        "broken chart. When a run does log them, read them as follows."
    ))
    flow.append(table(
        ["Line", "What it represents", "When to worry"],
        [
            ["Policy loss", "The clipped advantage objective. How hard the update pushed the buttons.", "A single spike can be one odd minibatch. A run of spikes with entropy crashing is an update that is eating the policy."],
            ["Value loss", "How wrong the critic’s guess of the return was.", "High and flat means the critic is not explaining the match, so the advantages are noisy and the actor is being coached by a bad teacher."],
            ["Approximate KL", "How far the new policy moved from the one that played the rollout.", "The clip exists to keep this small. A large KL means the step ignored the spirit of PPO, and the next rollout may be garbage."],
            ["Explained variance", "The share of return the critic actually accounts for. 1 is a perfect guess. 0 is a guess no better than the average. Negative is worse than predicting the average.", "Below zero, treat the advantages with suspicion. The actor is being told a story the critic cannot justify."],
            ["Clip fraction", "The share of samples whose probability ratio hit the 0.2 clip.", "Near zero means the clip is idle and steps are small. Near one means almost every sample was censored, so the update wanted to move further than PPO allows. Occasional clipping is the mechanism working."],
        ],
        [90, 175, W - 265],
    ))
    flow.append(Spacer(1, 8))
    flow.append(h2("Validation checks tab", "dash-val"))
    flow.append(P(
        "This is the tab that can promote a rung, and the tab that can embarrass a training curve. "
        "It keeps only rows where eval_win_rate is present. A green line names the best win rate, "
        "the update, and the rung. Then a table, then a chart of win rate, kills, and damage."
    ))
    flow.append(table(
        ["Column", "What one cell is"],
        [
            ["update", "The training update at which a held-out check was played. In this run, every 25 updates."],
            ["rung", "Which spawn placement those matches used."],
            ["eval_wins / eval_matches", "Wins out of the matches played. Here, always out of 16. Sides are part of the evaluation design elsewhere; this column is the count."],
            ["eval_win_rate", "Wins divided by matches. 0.50 is 8 of 16. With 16 matches the Wilson interval is wide: 8 of 16 is about 0.28 to 0.72, and 4 of 16 is about 0.10 to 0.49."],
            ["eval_kills", "Mean kills per validation match. A win needs the kill to finish. 1.81 at update 175, beside a 50% win rate, means the wins were not empty scorelines."],
            ["eval_damage", "Mean damage dealt. 273 at the best close check. Set it next to regeneration: damage that is not pressed into a kill comes back as health."],
        ],
        [130, W - 130],
    ))
    flow.append(Spacer(1, 6))
    flow.append(P(
        "Read the three chart lines together. On tiles_8 they rise together into update 175. "
        "On tiles_14 the win-rate line never clears 0.25, and damage does not establish a new "
        "climb. A win-rate line that jumps once and falls back is the 16-match coin, not a new policy."
    ))
    flow.append(callout(
        "why",
        "Validation is on caves the update did not just train on",
        "The deck is a shuffled set of maps, and the check uses the validation split’s placement "
        "rules for that rung. A policy can memorise a cave. The check is the objection. It is still "
        "a small objection: 16 matches, one opponent (the static bot), and a rung that keeps the "
        "enemy in sight. It is the right number to quote, and it is not the official-spawn result.",
    ))
    flow.append(h2("Other reports tab", "dash-other"))
    flow.append(P(
        "Three optional files, shown only when they exist in the selected run."
    ))
    flow.append(table(
        ["Block", "File", "What it would mean"],
        [
            ["Generalization", "generalization.json", "The same policy on shifted conditions: maps or physics it was not trained on. The question is whether a win travels. This run does not have the file."],
            ["Elo", "elo.json", "A rating per participant, highest first, anchored so different runs share a scale. Beating a weak bot a hundred times should not outrank a policy that trades with a strong one. This run does not have the file."],
            ["Benchmark", "benchmark.json", "A speed or suite dump, shown as raw JSON. Useful when the question is “is the environment the slow part?” This run does not have the file."],
        ],
        [90, 110, W - 200],
    ))
    flow.append(Spacer(1, 6))
    flow.append(P(
        "If none of the three files exist, the tab says so in one line. For spawn-curriculum-14 "
        "that is the whole tab. Do not read the absence as a failed evaluation. Those scripts "
        "were not the script that wrote this run."
    ))
    flow.append(h2("Generated pictures", "dash-pics"))
    flow.append(P(
        "Below the tabs, a select box lists every png, jpg, and gif in outputs/, newest first, "
        "with the time of day in the label. Choosing one draws it. These are the preview scripts: "
        "map sheets, a noise sheet, an observation drawing, a match filmstrip, collision pictures. "
        "They are how you check that a cave looks like a cave, and that a recorded match is the "
        "match you think it is. They are not frames from the live play page. The play page draws "
        "its own canvas in the browser."
    ))
    flow.append(callout(
        "good",
        "A dashboard that cannot train is a dashboard that cannot lie about training",
        "Because the page only reads logs, refreshing it cannot change a weight, and a pretty "
        "chart cannot be produced by the page itself. If metrics.jsonl says 8 of 16, the card "
        "says 50%. If the file has no loss columns, the loss tab admits it. The limit is the "
        "same virtue: a lesson you play in the browser is invisible here until some trainer "
        "writes a metrics line. Today, none does.",
    ))

    # ----- play page -----
    flow.append(h1("10. The play page, control by control", "play"))
    flow.append(P(
        "Open http://127.0.0.1:8765. The process is "
        "<font face='Courier'>scripts/play_human.py</font>. It loads 32 official validation caves, "
        "the first 32, and serves one match at a time to the browser. You are blue. The camera "
        "stays on you. The view starts 64 tiles wide and widens until the other fighter is on "
        "screen. Centring the camera on the midpoint was tried. When the red fighter walked toward "
        "you, you appeared to slide. The camera is on your body for that reason."
    ))
    flow.append(h2("Who the red fighter is", "hunter"))
    flow.append(P(
        "During your match, red is a hunter. It is the rusher: the generator’s route planner, "
        "the true position of the nearest living enemy, and a shot when line of sight exists. "
        "Aim error is 2 degrees, with a one-step reaction. It is not the learned network. "
        "A learned policy that has never been trained to search waits on the official spawn, "
        "and a match against that wait is not a match. The page says this in the sentence under "
        "the title: the red fighter flies the cave toward you and shoots when it can see you."
    ))
    flow.append(P(
        "The learned network enters in two other places. It is the brain that End and teach "
        "updates. After the lesson, an AI bout plays the updated weights against the weights from "
        "before the lesson, with a filter that refuses a step into rock. Nobody is at the keyboard "
        "in that bout. If you only watch the human match, you are watching the rusher. If you only "
        "watch the bout, you are watching the learner. They are different fighters, on purpose."
    ))
    flow.append(callout(
        "catch",
        "The hunter is allowed a fact the learner is not",
        "The rusher paths toward your true position even when a wall is between you. A learned "
        "policy would have to remember a sighting or search. That privilege is why the hunter "
        "closes ground on official spawns, and why beating it, or losing to it, is not a "
        "validation win rate. It is a sparring partner with a map. The learned bout afterwards "
        "is the honest look at the weights.",
    ))
    flow.append(h2("The cave and the scoreboard", "hud"))
    flow.append(table(
        ["Piece", "What it shows", "Why it is on the screen"],
        [
            ["The canvas", "Rock, air, two bodies, shots", "The match, from your camera. The mouse aims. The crosshair is the aim angle sent to the engine."],
            ["You, and a number", "Your kills", "The deathmatch score, team blue. Damage is not on this line. A hit that does not kill leaves this number still."],
            ["Agent, and a number", "Red’s kills", "During the human match this is the hunter’s score. During the bout the names change to the two learned copies."],
            ["The clock", "Seconds elapsed", "Counts up toward the length you picked. The engine’s own match length is the same number."],
            ["Cave note", "The seed, and the sentence “Official spawns…”", "Which of the 32 caves this is, and a reminder that the teams start where that cave placed them, not on the 6–12 tile drill."],
        ],
        [90, 145, W - 235],
    ))
    flow.append(Spacer(1, 8))
    flow.append(h2("The lesson panel, top to bottom", "panel"))
    flow.append(P("<b>Your name.</b> Stored with the match, up to 40 characters. The saved-matches line lists the names it has seen. The default is “you”. The name does not change the loss. It changes the record, so two people can share one file of demonstrations and still be told apart."))
    flow.append(P("<b>Match length.</b> 12, 20, or 60 seconds. Twenty is the default. A shorter match is a smaller lesson and a faster loop. Sixty is the real deathmatch length, and it is long enough for regeneration and a reload to matter. The practice after the match uses the same duration."))
    flow.append(P("<b>Decisions.</b> How many times your keys have been read in this match. One decision is one policy step, every two physics ticks, about 15 per second. A 20-second match that you play to the end is about 300 decisions. A decision that points into rock is still counted here. It is dropped later, when the lesson is built."))
    flow.append(P("<b>Start the match.</b> Picks a cave, resets the score, clears the keys, and starts the loop. Keys are also cleared if the browser tab loses focus, so a stuck key does not fly you into a wall after you click away."))
    flow.append(P("<b>End and teach.</b> Ends the human match and runs the lesson. The button stays disabled until a match has started. What it does, step by step, is the next section."))
    flow.append(P("<b>Skip the AI bout.</b> After a lesson, the two learned copies play and the page replays their frames. Skip stops that replay. It does not undo the lesson and it does not restore the weights."))
    flow.append(P("<b>The lesson sentence.</b> Compares the mean log-probability of your legal saved actions before the practice and after it. Less negative means the network now assigns those buttons a higher probability. The page calls the lesson improved when that number rises by more than 0.0001. If it does not, the sentence says the sample did not sharpen the copy. Read that sentence as a statement about this probability. The weight card below is a different measurement, and the two can disagree: the practice can move weights while your particular buttons stay about as likely as they were."))
    flow.append(P("<b>The AI sentence.</b> Kills and damage from the bout between the updated network and the previous one, plus how many saved controls were left out because they pointed into rock. A 0–0 with no damage means the two learned copies did not finish a fight. It does not mean the hunter failed to chase you. The hunter was the previous match."))
    flow.append(P("<b>Saved matches.</b> How many matches are in the file, how many decisions, whose names, and which brain file was loaded. The file is runs/human-play/demos.npz. It is append-only. It is git-ignored, with the rest of runs/."))
    flow.append(h2("End and teach, one update at a time", "teach"))
    flow.append(P(
        "The button used to wait until all four practice updates finished, then paint one number. "
        "The card looked frozen. It now asks the server for a fresh weight reading after each update. "
        "You should see “Update 1 of 4”, then 2, then 3, then 4, and the Now number should be "
        "allowed to change on each of those lines. The four updates share one optimiser, so the "
        "step is a continuation, not four separate cold starts."
    ))
    flow.append(P("What one update actually does:"))
    flow.extend(bullets([
        "Four caves from the play deck are stepped for 48 decisions, at the match length you chose. The opponent is a frozen copy of the current network, pulled at the start of that update.",
        "PPO runs one update at learning rate 0.0001. That is smaller than the main trainer’s 0.0003, because a lesson typed by a person should not be allowed to wipe a fighter.",
        "An aim fit pulls the gun toward an enemy the observation can see, with coefficient 0.15, using squared angle error. The same fit is applied to the saved demonstration rows.",
        "On the first of the four updates, and only then, the legal buttons are also imitated, at a coefficient of 0.05. The other three updates practice by playing and by aiming. They do not run that button-copy step. The rule in the code is “every 5th update,” and the count starts at zero, so update 1 of 4 is the one that copies.",
        "If a weight becomes NaN, or the rollout throws, that step’s weights are put back. The card would then sit still for that update, which is the guard working.",
    ]))
    flow.append(P(
        "Before any of that, the match is appended to demos.npz and the rows that press into rock "
        "are set aside. The log-probability used in the lesson sentence is computed on the legal "
        "rows only. After the fourth update the new weights are written to ckpt_latest.pt, the "
        "weights from before the lesson are written to ckpt_previous.pt, and a line is appended to "
        "lessons.jsonl. Then the AI bout is recorded for the replay."
    ))
    flow.append(h2("The weight card", "weights"))
    flow.append(P(
        "The card is docked onto the Restore button. It is a fingerprint of the live network, "
        "recomputed whenever the page receives a status: when it loads, when a match starts, "
        "after each practice update, after restore, and after forget."
    ))
    flow.append(table(
        ["Row", "What the number is", "What a change means"],
        [
            ["Original", "The mean absolute value of every floating weight in ckpt_base.pt, the brain from the first launch.", "This number does not move during a lesson. It is the anchor. Six decimal places are shown because a real step is small."],
            ["Now", "The same mean, on the weights currently in memory.", "Each of the four updates can move it. A move means some parameter changed size. It does not say whether you became a better shot."],
            ["Shift", "Now minus Original.", "Positive means the average absolute weight grew. Negative means it shrank. Zero, within a tiny tolerance, means you are back on the original brain."],
            ["The bar", "Grows from the centre. Width is the absolute shift times 500,000, capped at half the track.", "A visual of the shift, so a change of 0.000010 is visible. The cap means a large shift does not leave the card. Colour returns to the blue of “you” when Now matches Original."],
            ["The sentence under the bar", "Whether Now matches Original.", "“A lesson moved the weights” is the fingerprint talking. It can be true on a lesson the log-probability sentence called unimproved."],
            ["The short list", "The last four lines of lessons.jsonl: log-probability before, and after.", "An arrow means that lesson’s probability rose. “Stayed” means it did not. This list is about your buttons, not about the mean absolute weight."],
        ],
        [78, 175, W - 253],
    ))
    flow.append(Spacer(1, 6))
    flow.append(callout(
        "catch",
        "Mean absolute weight is not skill",
        "A network can get worse and move this number, or get better and move it the other way. "
        "It is on the page because you asked to see the weights change while a lesson runs, and "
        "because Restore has to show that they came back. Judge the fighter by the bout’s kills "
        "and damage, and by the validation tab on the other screen. Judge the card by whether "
        "the lesson touched the file.",
    ))
    flow.append(h2("Restore, and Forget", "restore"))
    flow.append(P(
        "They sit on top of each other and they are not the same operation."
    ))
    flow.append(table(
        ["Button", "What it changes", "What it leaves alone"],
        [
            ["Restore the original agent", "Loads ckpt_base.pt into the live network and overwrites ckpt_latest.pt with that same original. Now returns to Original. Shift returns to zero.", "The saved matches. The next lesson can teach them again, onto the original brain. Past lines in lessons.jsonl stay, so the list under the bar can still show old probabilities while the weights are original."],
            ["Forget saved matches", "Deletes the demonstration file. The saved-matches count goes back to zero.", "The weights. Now, Original, and Shift do not move. Forgetting a bad hour of play does not unwind a lesson that already changed the brain. Restore does that."],
        ],
        [110, 185, W - 295],
    ))
    flow.append(Spacer(1, 6))
    flow.append(P(
        "Both buttons refuse to run while a match is in progress and has not been taught. "
        "Finish or teach first. Restore is blocked for the same reason a checkpoint load is "
        "blocked during a match: the live weights and the file would disagree underneath a fight "
        "that is still being recorded."
    ))
    flow.append(h2("The keys", "keys"))
    flow.append(P(
        "A and D, or the left and right arrows, move. Exactly one of them has to be down. "
        "Both down is stay, so the two hands do not fight. W or the up arrow jets. The mouse "
        "aims. J, K, space, or the left mouse button shoots, and holding the key holds the trigger. "
        "The rifle still has its cooldown and its magazine; holding the button does not make it a laser."
    ))
    flow.append(P(
        "The line under the keys is a rule of the observation, repeated where your hands are: "
        "sight is from the body, so turning the gun does not find anyone. If the hunter is behind "
        "a wall, spinning the rifle will not put them on your screen as a known target, and it "
        "will not put them into the 330 numbers either."
    ))
    flow.append(h2("Where the files go", "files"))
    flow.append(table(
        ["File", "Written when", "Read by"],
        [
            ["ckpt_base.pt", "The first launch, from whatever brain was resolved", "Restore, and the Original number. Unchanged by lessons."],
            ["ckpt_latest.pt", "The end of every lesson, and also by Restore", "The next launch, which continues from here unless you pass --restart."],
            ["ckpt_previous.pt", "The end of a lesson, the weights from before it", "The AI bout, as the “before” fighter."],
            ["demos.npz", "Every taught match, appended", "The next lesson, which trains on every legal row, not only the latest match. Forget deletes it."],
            ["lessons.jsonl", "One line per taught match", "The four-line list on the weight card. The training dashboard does not open it."],
        ],
        [100, 175, W - 275],
    ))
    flow.append(Spacer(1, 6))
    flow.append(P(
        "The brain on first launch is chosen in this order: ckpt_latest.pt if a lesson already "
        "exists, otherwise the checkpoint you passed in, otherwise "
        "runs/spawn-curriculum-14/ckpt_tiles_8.pt, otherwise a fresh network. --restart prefers "
        "ckpt_base.pt. The tiles_8 file is the update-175 policy, the best measured close-range "
        "checkpoint, with the limits in section 7."
    ))

    # ----- close -----
    flow.append(h1("What to hold onto", "close"))
    flow.append(P("If you remember one page, remember this."))
    flow.extend(bullets([
        "The cave is generated, mirrored, and checked against a fuel model, because an unfair or trapped map makes every later number meaningless.",
        "The agent sees 330 numbers and outputs move, jet, shoot, and a continuous aim. Sight is a line from the body. The gun does not see.",
        "The reward pays a win, and it also pays damage. Regeneration deletes damage that is not pressed into a kill. Mean return can flatter a policy the win rate will not.",
        "PPO takes a clipped step so one update cannot throw the policy away. Entropy keeps the buttons from collapsing. The aim spread is clamped to 6 degrees because a wide cloud cannot hit a body.",
        "The best measured checkpoint is update 175 of spawn-curriculum-14: 8 of 16 wins, 1.81 kills, about 273 damage, enemy in sight at 6 to 12 tiles. The next rung peaked at 4 of 16. Official hidden spawns were not trained. Eight out of sixteen has a wide confidence interval.",
        "The training dashboard reads metrics.jsonl. Trust the validation tab over the mean-return line. The best-win card remembers the peak, not the present.",
        "The play page’s red fighter, while you are playing, is the rusher. The learned weights move during End and teach, four small updates, and then play each other. The weight card is a fingerprint. Restore puts the fingerprint back and keeps your matches. Forget deletes the matches and keeps the fingerprint.",
        "The two screens run together and do not feed each other. A lesson in the browser does not appear as a new point on the training chart.",
    ]))
    flow.append(Spacer(1, 6))
    flow.append(P(
        "The open work is the same open work as in the progress notes. Retrain from the capped "
        "aim. Find a rung between 8 tiles and 16 that the policy can actually clear. Then search "
        "from the official spawns, where the teams start hidden and a policy that only shoots "
        "what it can already see will stand still. Self-play, a real generalization table, and a "
        "report that quotes these logs are the phases that come after a policy worth measuring. "
        "The code for them is on main. The result is not, yet."
    ))
    return flow


def main():
    doc = BaseDocTemplate(
        OUT,
        pagesize=A4,
        title="RL-Donut — a learning guide",
        author="RL-Donut",
        subject="A step-by-step teaching guide to the project, the training dashboard, and the play page.",
    )
    frame_cover = Frame(MARGIN, 22 * mm, PAGE_W - 2 * MARGIN, PAGE_H - 36 * mm, id="cover", showBoundary=0)
    frame_body = Frame(MARGIN, 16 * mm, PAGE_W - 2 * MARGIN, PAGE_H - 30 * mm, id="body", showBoundary=0)
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame_cover], onPage=draw_cover),
        PageTemplate(id="body", frames=[frame_body], onPage=draw_body),
    ])
    doc.build(story())
    print(OUT)


if __name__ == "__main__":
    main()
