"""Build the offline interactive predictor: reports/draft_predictor.html.

Run (from anywhere):  python src/build_html.py

Why this works without a server: the shipped model is a plain logistic
regression, so its entire brain is an intercept plus one coefficient per
(champion, where-it-appeared) feature. The win probability is just

    p = 1 / (1 + exp(-(intercept + sum of the active coefficients)))

so we dump those numbers into the page as JSON and re-evaluate them in a few
lines of JavaScript. The result is byte-for-byte the same maths scikit-learn
runs -- verified against predict_proba in verify_export() below -- in a single
self-contained file with no network access, no dependencies, and no hosting
(the project's non-goals rule out a web app; this is a local file you open).
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import features as F  # noqa: E402


def export_model() -> dict:
    """Pull the linear CHAMPION model apart into a JSON-friendly dict.

    Uses model_champions.joblib (Tier-1), not the best model: the interactive
    predictor is the per-champion view, which needs champion coefficients. The
    archetype model scores marginally higher -- see the report for the full
    comparison.
    """
    bundle = joblib.load(config.DATA_PROCESSED / "model_champions.joblib")
    model, columns = bundle["model"], bundle["columns"]
    est = getattr(model, "best_estimator_", model)
    coef = pd.Series(est.coef_[0], index=columns)
    labels = F.load_labels()

    blue, red, ban = {}, {}, {}
    for col, v in coef.items():
        cid = int(col.split("_")[-1])
        if col.startswith("blue_pick_"):
            blue[cid] = round(float(v), 12)
        elif col.startswith("red_pick_"):
            red[cid] = round(float(v), 12)
        elif col.startswith("ban_"):
            ban[cid] = round(float(v), 12)

    df = F.load_matches()
    champions = [{"id": int(cid), "name": name}
                 for cid, name in labels["name"].items()]
    champions.sort(key=lambda c: c["name"])

    # Provenance: state exactly where these numbers came from, so a reader never
    # has to guess which server/rank/patch the model speaks for.
    tiers = (df["seed_tier"].value_counts().to_dict()
             if "seed_tier" in df.columns else {})
    dates = ["", ""]
    if "gameCreation" in df.columns and df["gameCreation"].notna().any():
        span = pd.to_datetime(df["gameCreation"], unit="ms")
        dates = [f"{span.min():%d %b %Y}", f"{span.max():%d %b %Y}"]

    return {
        "intercept": round(float(est.intercept_[0]), 12),
        "blue": blue, "red": red, "ban": ban,
        "champions": champions,
        "meta": {
            "patches": " + ".join(config.TARGET_PATCHES),
            "games": int(len(df)),
            "region": config.REGION_LABEL,
            "platform": config.PLATFORM_LABEL,
            "queue": "Ranked Solo/Duo (420)",
            "tiers": {k: int(v) for k, v in tiers.items()},
            "from": dates[0], "to": dates[1],
        },
    }


def verify_export(data: dict) -> None:
    """Prove the exported numbers reproduce scikit-learn's own prediction.

    If this ever fails the HTML would quietly lie, so it is a hard error.
    """
    import predict

    blue = ["Aatrox", "Vi", "Ahri", "Jinx", "Thresh"]
    red = ["Garen", "Lee Sin", "Orianna", "Caitlyn", "Leona"]
    bans = ["Yasuo", "Zed", "Lulu"]
    labels = F.load_labels()

    logit = data["intercept"]
    for name in blue:
        logit += data["blue"].get(predict.resolve_champion(name, labels), 0.0)
    for name in red:
        logit += data["red"].get(predict.resolve_champion(name, labels), 0.0)
    for name in bans:
        logit += data["ban"].get(predict.resolve_champion(name, labels), 0.0)
    ours = 1 / (1 + np.exp(-logit))
    theirs = predict.predict_draft(blue, red, bans)

    if abs(ours - theirs) > 1e-9:
        raise SystemExit(f"Export mismatch! js-style={ours:.10f} sklearn={theirs:.10f}")
    print(f"  verified: exported maths matches scikit-learn ({ours:.6f})")


HTML = r"""<meta charset="utf-8">
<title>Draft Win-Probability Predictor</title>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb; --panel: #ffffff; --ink: #0b0b0b; --ink-2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --line: #c3c2b7;
    --blue: #2a78d6; --red: #e34948;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --surface: #1a1a19; --panel: #232322; --ink: #ffffff; --ink-2: #c3c2b7;
      --muted: #898781; --grid: #2c2c2a; --line: #383835;
      --blue: #3987e5; --red: #e66767;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 28px 20px 60px;
    background: var(--surface); color: var(--ink);
    font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1060px; margin: 0 auto; }
  h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }
  .sub { color: var(--ink-2); margin: 0 0 26px; max-width: 70ch; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
  @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel); border: 1px solid var(--grid);
    border-radius: 10px; padding: 16px 18px;
  }
  .card h2 { font-size: 12px; letter-spacing: .08em; text-transform: uppercase;
             margin: 0 0 12px; color: var(--muted); }
  .card.blue h2 { color: var(--blue); }
  .card.red h2 { color: var(--red); }
  select {
    width: 100%; margin-bottom: 7px; padding: 7px 9px;
    background: var(--surface); color: var(--ink);
    border: 1px solid var(--line); border-radius: 6px;
    font: inherit; font-size: 14px;
  }
  .readout { margin: 26px 0 8px; }
  .nums { display: flex; justify-content: space-between; align-items: baseline; }
  .pct { font-size: 40px; font-weight: 700; letter-spacing: -0.02em; }
  .pct small { font-size: 13px; font-weight: 600; letter-spacing: .06em;
               text-transform: uppercase; display: block; color: var(--muted); }
  .bar { display: flex; height: 22px; border-radius: 5px; overflow: hidden;
         margin-top: 10px; background: var(--grid); position: relative; }
  .bar i { display: block; height: 100%; }
  .bar .mid { position: absolute; left: 50%; top: -4px; bottom: -4px; width: 1px;
              background: var(--ink-2); opacity: .55; }
  .hint { color: var(--muted); font-size: 12.5px; text-align: center; margin-top: 7px; }
  table { width: 100%; border-collapse: collapse; margin-top: 6px; font-size: 13.5px; }
  th { text-align: left; font-weight: 600; color: var(--muted); font-size: 11.5px;
       letter-spacing: .06em; text-transform: uppercase; padding: 4px 8px 8px 0; }
  td { padding: 4px 8px 4px 0; border-top: 1px solid var(--grid); vertical-align: middle; }
  td.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; width: 76px; }
  .track { position: relative; height: 15px; min-width: 130px; }
  .track .zero { position: absolute; left: 50%; top: 0; bottom: 0; width: 1px; background: var(--line); }
  .track i { position: absolute; top: 3px; height: 9px; border-radius: 2px; }
  .prov { margin-bottom: 22px; }
  .prov dl { display: grid; grid-template-columns: max-content 1fr; gap: 5px 16px;
             margin: 0; font-size: 13.5px; }
  .prov dt { color: var(--muted); }
  .prov dd { margin: 0; color: var(--ink); }
  .note { color: var(--ink-2); font-size: 13px; margin-top: 22px; max-width: 74ch; }
  .foot { color: var(--muted); font-size: 12.5px; margin-top: 12px; }
</style>

<div class="wrap">
  <h1>Draft win-probability predictor</h1>
  <p class="sub">Pick both teams and see what champion select alone says about the
  game, before anyone has moved. Everything runs inside this one file &mdash; no
  internet, no server.</p>

  <div class="card prov">
    <h2>Where this data comes from</h2>
    <dl id="prov"></dl>
  </div>

  <div class="grid">
    <div class="card blue"><h2>Blue team</h2><div id="blue"></div></div>
    <div class="card red"><h2>Red team</h2><div id="red"></div></div>
  </div>

  <div class="card" style="margin-top:18px">
    <h2>Bans (optional)</h2>
    <div class="grid" id="bans"></div>
  </div>

  <div class="readout card">
    <div class="nums">
      <div class="pct" style="color:var(--blue)"><small>Blue team</small><span id="p-blue">50.0%</span></div>
      <div class="pct" style="color:var(--red);text-align:right"><small>Red team</small><span id="p-red">50.0%</span></div>
    </div>
    <div class="bar"><i id="bar-blue" style="background:var(--blue)"></i><i id="bar-red" style="background:var(--red)"></i><span class="mid"></span></div>
    <p class="hint">The centre line is 50/50 &mdash; where the draft tells you nothing.</p>
  </div>

  <div class="card" style="margin-top:18px">
    <h2>What moved the needle</h2>
    <table>
      <thead><tr><th>Champion</th><th>Where</th><th>Effect</th><th style="width:40%">&nbsp;</th></tr></thead>
      <tbody id="contrib"></tbody>
    </table>
  </div>

  <p class="note"><b>How to read this.</b> Effects are in percentage points of win
  chance. A positive number means the blue team wins slightly more often when that
  happens. Notice that even the strongest champion here is worth well under two
  games in a hundred &mdash; and that the total almost never leaves the 40&ndash;60% band.
  That is the finding: <b>the draft barely decides a solo-queue game.</b> What you do
  after champion select matters far more.</p>
  <p class="foot">This is the champion-level model (logistic regression on picks and bans), used
  here so each pick's effect can be shown. A team-archetype model scores a hair higher &mdash; see
  the findings report. Not endorsed by Riot Games.</p>
</div>

<script>
const DATA = __DATA__;
const PP = 25;            // log-odds -> percentage points, near an even game
const DEFAULT_BLUE = ["Aatrox", "Vi", "Ahri", "Jinx", "Thresh"];
const DEFAULT_RED  = ["Garen", "Lee Sin", "Orianna", "Caitlyn", "Leona"];

const byName = {};
DATA.champions.forEach(c => byName[c.name] = c.id);

function makeSelect(container, cls, i, preset) {
  const s = document.createElement("select");
  s.className = cls;
  s.innerHTML = '<option value="">-- none --</option>' +
    DATA.champions.map(c => `<option value="${c.id}">${c.name}</option>`).join("");
  if (preset) s.value = byName[preset];
  s.addEventListener("change", update);
  container.appendChild(s);
}

const blueBox = document.getElementById("blue"), redBox = document.getElementById("red");
for (let i = 0; i < 5; i++) makeSelect(blueBox, "pick-blue", i, DEFAULT_BLUE[i]);
for (let i = 0; i < 5; i++) makeSelect(redBox, "pick-red", i, DEFAULT_RED[i]);
const banBox = document.getElementById("bans");
for (let i = 0; i < 10; i++) makeSelect(banBox, "ban", i, null);

const M = DATA.meta;
const tierLine = Object.entries(M.tiers || {})
  .sort((a, b) => b[1] - a[1])
  .map(([t, n]) => `${t.charAt(0) + t.slice(1).toLowerCase()} ${n.toLocaleString()}`)
  .join(" &middot; ") || "-";
document.getElementById("prov").innerHTML = [
  ["Server",   `${M.region} &mdash; platform <code>${M.platform}</code>`],
  ["Queue",    M.queue],
  ["Ranks",    tierLine + '<br><span style="color:var(--muted)">tier is the rank of the '
               + 'player whose history surfaced the game, so it is approximate</span>'],
  ["Patches",  M.patches],
  ["Games",    `<b>${M.games.toLocaleString()}</b> matches`
               + (M.from ? `, played ${M.from} &ndash; ${M.to}` : "")],
].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");

const vals = cls => [...document.querySelectorAll("." + cls)]
  .map(s => s.value).filter(v => v !== "").map(Number);
const nameOf = id => (DATA.champions.find(c => c.id === id) || {name: "?"}).name;

function update() {
  const blue = vals("pick-blue"), red = vals("pick-red"), bans = vals("ban");

  // Same arithmetic scikit-learn does: intercept + active coefficients.
  let logit = DATA.intercept;
  const rows = [];
  blue.forEach(id => { const c = DATA.blue[id] || 0; logit += c;
                       rows.push([nameOf(id), "on the blue team", c]); });
  red.forEach(id  => { const c = DATA.red[id]  || 0; logit += c;
                       rows.push([nameOf(id), "on the red team", c]); });
  bans.forEach(id => { const c = DATA.ban[id]  || 0; logit += c;
                       rows.push([nameOf(id), "banned", c]); });

  const p = 1 / (1 + Math.exp(-logit));
  document.getElementById("p-blue").textContent = (p * 100).toFixed(1) + "%";
  document.getElementById("p-red").textContent = ((1 - p) * 100).toFixed(1) + "%";
  document.getElementById("bar-blue").style.width = (p * 100) + "%";
  document.getElementById("bar-red").style.width = ((1 - p) * 100) + "%";

  rows.sort((a, b) => b[2] - a[2]);
  const max = Math.max(0.0001, ...rows.map(r => Math.abs(r[2] * PP)));
  document.getElementById("contrib").innerHTML = rows.map(([who, where, c]) => {
    const pp = c * PP;
    const w = Math.abs(pp) / max * 48;                       // % of the track
    const colour = pp >= 0 ? "var(--blue)" : "var(--red)";
    const pos = pp >= 0 ? `left:50%;width:${w}%` : `right:50%;width:${w}%`;
    const shown = c === 0 ? "n/a" : (pp >= 0 ? "+" : "") + pp.toFixed(2);
    return `<tr><td>${who}</td><td style="color:var(--ink-2)">${where}</td>
      <td class="n">${shown}</td>
      <td><div class="track"><span class="zero"></span>
      <i style="${pos};background:${colour}"></i></div></td></tr>`;
  }).join("");
}
update();
</script>
"""


def main() -> None:
    data = export_model()
    verify_export(data)
    out = config.ROOT / "reports" / "draft_predictor.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HTML.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"  wrote {out.relative_to(config.ROOT)}  ({kb:.0f} KB, self-contained)")
    print("  open it directly in a browser -- no server, no internet.")


if __name__ == "__main__":
    main()
