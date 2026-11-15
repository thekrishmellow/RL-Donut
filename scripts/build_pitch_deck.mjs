import pptxgen from "pptxgenjs";

const pres = new pptxgen();
pres.defineLayout({ name: "PITCH", width: 13.333, height: 7.5 });
pres.layout = "PITCH";
pres.title = "RL-Donut pitch";
pres.author = "RL-Donut team";
pres.subject = "Architecture and the measured training result";

const STONE = "1C1917";
const EMBER = "C2410C";
const SAND = "F6F1E7";
const CARD = "FFFCF7";
const INK = "1C1917";
const MUTED = "57534E";
const SAGE = "3F6212";
const LINE = "E7E5E4";

function rail(slide) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.12, h: 7.5, fill: { color: EMBER },
  });
}

function footer(slide, n, total) {
  slide.addText("RL-Donut  ·  15 November 2026", {
    x: 0.5, y: 7.12, w: 8, h: 0.24,
    fontFace: "Calibri", fontSize: 11, color: MUTED, margin: 0,
  });
  slide.addText(`${n}  /  ${total}`, {
    x: 11.3, y: 7.12, w: 1.5, h: 0.24,
    fontFace: "Calibri", fontSize: 11, color: MUTED, align: "right", margin: 0,
  });
}

function contentTitle(slide, title) {
  slide.addText(title, {
    x: 0.5, y: 0.32, w: 12.3, h: 0.55,
    fontFace: "Georgia", fontSize: 32, color: INK, margin: 0,
  });
}

const TOTAL = 10;

// 1. Title
{
  const s = pres.addSlide();
  s.background = { color: STONE };
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.18, h: 7.5, fill: { color: EMBER },
  });
  s.addText("RL-DONUT", {
    x: 0.7, y: 1.7, w: 11, h: 0.32,
    fontFace: "Calibri", fontSize: 14, color: "FDBA74", charSpacing: 3, margin: 0,
  });
  s.addText("Learning to fight\nin seeded caves", {
    x: 0.7, y: 2.15, w: 11, h: 1.9,
    fontFace: "Georgia", fontSize: 48, color: "FAFAF9", margin: 0,
  });
  s.addText("A 2D jetpack deathmatch for reinforcement learning.\nArchitecture, the reward, and the training result we can stand behind.", {
    x: 0.7, y: 4.3, w: 9, h: 0.8,
    fontFace: "Calibri", fontSize: 18, color: "D6D3D1", margin: 0,
  });
  s.addText("15 November 2026", {
    x: 0.7, y: 6.6, w: 6, h: 0.3,
    fontFace: "Calibri", fontSize: 14, color: "A8A29E", margin: 0,
  });
}

// 2. Architecture
{
  const s = pres.addSlide();
  s.background = { color: SAND };
  rail(s);
  contentTitle(s, "From a seed to a policy");
  const steps = [
    ["01", "Seed", "Named random streams. The same integer rebuilds the same cave."],
    ["02", "Map", "Perlin noise, FBM, a quantile of rock, then Dijkstra tunnels."],
    ["03", "Match", "Batched physics, a hitscan rifle, 900 decisions in 60 seconds."],
    ["04", "Learner", "330 numbers in. Move, jetpack, shoot, and one aim angle out."],
    ["05", "PPO", "Eight matches at once. A gate widens the spawn gap after a win."],
  ];
  steps.forEach((step, i) => {
    const x = 0.5 + i * 2.5;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 1.5, w: 2.32, h: 4.7,
      fill: { color: CARD }, rectRadius: 0.08,
    });
    s.addText(step[0], {
      x: x + 0.18, y: 1.7, w: 2, h: 0.4,
      fontFace: "Georgia", fontSize: 20, color: EMBER, margin: 0,
    });
    s.addText(step[1], {
      x: x + 0.18, y: 2.25, w: 2, h: 0.5,
      fontFace: "Georgia", fontSize: 22, color: INK, margin: 0,
    });
    s.addText(step[2], {
      x: x + 0.18, y: 3.0, w: 1.96, h: 2.6,
      fontFace: "Calibri", fontSize: 15, color: MUTED, margin: 0,
    });
  });
  footer(s, 2, TOTAL);
}

// 3. Maps
{
  const s = pres.addSlide();
  s.background = { color: SAND };
  rail(s);
  contentTitle(s, "A cave is a function of one seed");
  const facts = [
    ["Perlin + FBM", "Four octaves. Each is twice as fine and half as strong. Cells are stretched into ledges."],
    ["Quantile rock", "The rockiest 40–55% becomes stone. A fixed cutoff swung from 31% to 50%."],
    ["Cellular automaton", "Three passes. A tile turns to rock at 5 of 8 neighbours. Jags fell from ~72 to ~1."],
    ["Dijkstra tunnels", "Dig cost 12, and a second vertical dig costs triple, so shafts become stairs."],
    ["Amanatides–Woo", "One grid walk for sight and for the rifle. Quarter-tile sampling missed 46 of 10,000 corners."],
    ["Splits never overlap", "Train 0–9,999. Val 10,000–10,999. Test 20,000–20,999 stays untouched."],
  ];
  facts.forEach((f, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = 0.5 + col * 4.15;
    const y = 1.45 + row * 2.55;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y, w: 3.95, h: 2.35,
      fill: { color: CARD }, rectRadius: 0.08,
    });
    s.addText(f[0], {
      x: x + 0.22, y: y + 0.22, w: 3.5, h: 0.45,
      fontFace: "Georgia", fontSize: 18, color: INK, margin: 0,
    });
    s.addText(f[1], {
      x: x + 0.22, y: y + 0.8, w: 3.5, h: 1.25,
      fontFace: "Calibri", fontSize: 15, color: MUTED, margin: 0,
    });
  });
  footer(s, 3, TOTAL);
}

// 4. Match and observation
{
  const s = pres.addSlide();
  s.background = { color: SAND };
  rail(s);
  contentTitle(s, "What a minute of play is");
  const cols = [
    [EMBER, "FAFAF9", "The body", "0.8 × 1.6 tiles. Semi-implicit Euler. Gravity 25, thrust 40. Nothing moves a full tile in one tick."],
    [STONE, "FAFAF9", "The rifle", "15 damage. A kill is 7 hits, about 1.2 seconds. Range 30 tiles. The rifle adds ±2° of scatter."],
    [SAGE, "FAFAF9", "The observation", "330 numbers. A 21×13 tile window. Enemies are remembered for 10 seconds. Team B is a mirror."],
  ];
  cols.forEach((c, i) => {
    const x = 0.5 + i * 4.2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x, y: 1.7, w: 3.95, h: 4.4,
      fill: { color: c[0] }, rectRadius: 0.08,
    });
    s.addText(c[2], {
      x: x + 0.28, y: 2.0, w: 3.4, h: 0.7,
      fontFace: "Georgia", fontSize: 26, color: c[1], margin: 0,
    });
    s.addText(c[3], {
      x: x + 0.28, y: 2.9, w: 3.4, h: 2.6,
      fontFace: "Calibri", fontSize: 18, color: "E7E5E4", margin: 0,
    });
  });
  footer(s, 4, TOTAL);
}

// 5. Reward
{
  const s = pres.addSlide();
  s.background = { color: SAND };
  rail(s);
  contentTitle(s, "The reward is a win, plus a fading hint");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 1.45, w: 12.3, h: 1.35,
    fill: { color: STONE }, rectRadius: 0.08,
  });
  s.addText("reward  =  match outcome  +  alpha  ×  shaping", {
    x: 0.75, y: 1.75, w: 11.8, h: 0.7,
    fontFace: "Georgia", fontSize: 28, color: "FAFAF9", margin: 0,
  });
  const bits = [
    ["+1 / −1", "Win or loss at the end of the minute. A draw is 0."],
    ["±0.3", "A kill, a death, or a fall into a pit."],
    ["±0.003", "Per point of damage. One hundred damage equals another kill."],
    ["alpha", "Starts at 1. Falls toward 0.1, so later updates have to win the match."],
  ];
  bits.forEach((b, i) => {
    const y = 3.1 + i * 0.9;
    s.addText(b[0], {
      x: 0.5, y, w: 2.3, h: 0.7,
      fontFace: "Georgia", fontSize: 22, color: EMBER, margin: 0, valign: "middle",
    });
    s.addText(b[1], {
      x: 3.0, y, w: 9.5, h: 0.7,
      fontFace: "Calibri", fontSize: 18, color: INK, margin: 0, valign: "middle",
    });
  });
  footer(s, 5, TOTAL);
}

// 6. Training
{
  const s = pres.addSlide();
  s.background = { color: SAND };
  rail(s);
  contentTitle(s, "PPO, then a wider gap only after a win");
  const rows = [
    ["Network", "Actor-critic, width 256. Categorical heads for move, jetpack, and shoot. A Gaussian for the aim."],
    ["Batch", "8 matches at once. An update every 128 decisions. The learner plays both sides."],
    ["Opponent", "A static bot. It stands on its spawn and does not shoot. The rusher is the yardstick, not this opponent."],
    ["Gate", "Every 25 updates, 16 validation matches of 60 seconds. 8 wins opens the next distance."],
  ];
  rows.forEach((r, i) => {
    const y = 1.45 + i * 1.3;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
      x: 0.5, y, w: 12.3, h: 1.15,
      fill: { color: CARD }, rectRadius: 0.08,
    });
    s.addText(r[0], {
      x: 0.75, y: y + 0.22, w: 2.4, h: 0.7,
      fontFace: "Georgia", fontSize: 18, color: EMBER, margin: 0, valign: "middle",
    });
    s.addText(r[1], {
      x: 3.3, y: y + 0.18, w: 9.2, h: 0.8,
      fontFace: "Calibri", fontSize: 16, color: INK, margin: 0, valign: "middle",
    });
  });
  footer(s, 6, TOTAL);
}

// 7. Results
{
  const s = pres.addSlide();
  s.background = { color: SAND };
  rail(s);
  contentTitle(s, "What the validation matches showed");
  s.addText("16 matches of 60 seconds. Seeds 10,000 and up. Sides swapped. Run spawn-curriculum-14.", {
    x: 0.5, y: 1.05, w: 12, h: 0.35,
    fontFace: "Calibri", fontSize: 15, color: MUTED, margin: 0,
  });

  // two result cards
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 1.6, w: 6.0, h: 3.55,
    fill: { color: SAGE }, rectRadius: 0.08,
  });
  s.addText("6–12 tiles", {
    x: 0.8, y: 1.85, w: 5.4, h: 0.4,
    fontFace: "Calibri", fontSize: 16, color: "D9F99D", margin: 0,
  });
  s.addText("8 / 16", {
    x: 0.8, y: 2.3, w: 5.4, h: 1.15,
    fontFace: "Georgia", fontSize: 60, color: "FAFAF9", margin: 0,
  });
  s.addText("Update 175. Enemy already in sight.\n1.81 kills. 273 damage. The gate opened.", {
    x: 0.8, y: 3.6, w: 5.4, h: 1.1,
    fontFace: "Calibri", fontSize: 16, color: "ECFCCB", margin: 0,
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 6.8, y: 1.6, w: 6.0, h: 3.55,
    fill: { color: STONE }, rectRadius: 0.08,
  });
  s.addText("12–16 tiles", {
    x: 7.1, y: 1.85, w: 5.4, h: 0.4,
    fontFace: "Calibri", fontSize: 16, color: "FDBA74", margin: 0,
  });
  s.addText("4 / 16", {
    x: 7.1, y: 2.3, w: 5.4, h: 1.15,
    fontFace: "Georgia", fontSize: 60, color: "FAFAF9", margin: 0,
  });
  s.addText("Still in sight. Still inside rifle range.\nLater checks fell to 1 of 16. The gate stayed shut.", {
    x: 7.1, y: 3.6, w: 5.4, h: 1.1,
    fontFace: "Calibri", fontSize: 16, color: "E7E5E4", margin: 0,
  });

  s.addText("Official hidden spawns, about 64 tiles apart, were not trained.", {
    x: 0.5, y: 5.4, w: 12.3, h: 0.4,
    fontFace: "Georgia", fontSize: 18, color: INK, margin: 0,
  });
  footer(s, 7, TOTAL);
}

// 8. Why
{
  const s = pres.addSlide();
  s.background = { color: SAND };
  rail(s);
  contentTitle(s, "The shot got longer. The aim stayed wide.");
  const stats = [
    ["59°", "Aim spread on the saved checkpoints. A body at 14 tiles is about 7° tall."],
    ["3.25", "Entropy at update 600. A new policy starts near 3.9. The actions never locked in."],
    ["3 / 16", "Rescore of the later checkpoint on the longer maps. It did not beat the earlier weights."],
  ];
  stats.forEach((st, i) => {
    const y = 1.5 + i * 1.65;
    s.addText(st[0], {
      x: 0.5, y, w: 2.8, h: 1.3,
      fontFace: "Georgia", fontSize: 36, color: EMBER, margin: 0, valign: "middle",
    });
    s.addText(st[1], {
      x: 3.5, y, w: 9.2, h: 1.3,
      fontFace: "Calibri", fontSize: 20, color: INK, margin: 0, valign: "middle",
    });
  });
  footer(s, 8, TOTAL);
}

// 9. After and open
{
  const s = pres.addSlide();
  s.background = { color: SAND };
  rail(s);
  contentTitle(s, "After the run, and what is still open");
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 0.5, y: 1.45, w: 6.0, h: 5.2,
    fill: { color: CARD }, rectRadius: 0.08,
  });
  s.addText("In the code, after update 600", {
    x: 0.75, y: 1.75, w: 5.5, h: 0.5,
    fontFace: "Georgia", fontSize: 18, color: INK, margin: 0,
  });
  s.addText("Aim is capped at 6 degrees.\nA hidden agent walks the floors of the cave and does not shoot until someone is in view.\nThose changes are not inside the saved checkpoints.", {
    x: 0.75, y: 2.45, w: 5.5, h: 3.5,
    fontFace: "Calibri", fontSize: 18, color: MUTED, margin: 0,
  });

  s.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x: 6.8, y: 1.45, w: 6.0, h: 5.2,
    fill: { color: STONE }, rectRadius: 0.08,
  });
  s.addText("Still open", {
    x: 7.05, y: 1.75, w: 5.5, h: 0.5,
    fontFace: "Georgia", fontSize: 18, color: "FAFAF9", margin: 0,
  });
  s.addText("Retrain with the tight aim.\nStop when the wins stop rising.\nA smaller step than 8 tiles to 14.\nThe official spawns, where the enemy starts behind rock.", {
    x: 7.05, y: 2.45, w: 5.5, h: 3.5,
    fontFace: "Calibri", fontSize: 18, color: "E7E5E4", margin: 0,
  });
  footer(s, 9, TOTAL);
}

// 10. Close
{
  const s = pres.addSlide();
  s.background = { color: STONE };
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: 0.18, h: 7.5, fill: { color: EMBER },
  });
  s.addText("Grade the curriculum,\nnot the earlier sign-off.", {
    x: 0.7, y: 1.3, w: 12, h: 1.7,
    fontFace: "Georgia", fontSize: 36, color: "FAFAF9", margin: 0,
  });
  const links = [
    ["Submission notes", "github.com/thekrishmellow/RL-Donut/blob/main/docs/SUBMISSION.md"],
    ["Technical report", "github.com/thekrishmellow/RL-Donut/blob/main/docs/RL-Donut-technical-report.pdf"],
    ["This run", "runs/spawn-curriculum-14  on  main"],
  ];
  links.forEach((l, i) => {
    const y = 3.4 + i * 0.95;
    s.addText(l[0], {
      x: 0.7, y, w: 3.2, h: 0.7,
      fontFace: "Georgia", fontSize: 18, color: "FDBA74", margin: 0, valign: "middle",
    });
    s.addText(l[1], {
      x: 4.0, y, w: 8.5, h: 0.7,
      fontFace: "Calibri", fontSize: 16, color: "E7E5E4", margin: 0, valign: "middle",
    });
  });
}

await pres.writeFile({ fileName: "docs/RL-Donut-pitch.pptx" });
console.log("wrote docs/RL-Donut-pitch.pptx");
