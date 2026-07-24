"""Build the offline findings report: reports/report.html.

Run (from anywhere):  python src/build_report.py

A single self-contained page telling the whole story: where the data came from,
what the model found, and what it means. Figures are embedded as base64 data
URIs, so the file can be moved or emailed and still renders with no internet, no
server and no sibling image folder.

Numbers come from data/processed/headline.json and model_comparison.csv (written
by model.py), so the prose can never drift out of sync with the model.
"""

import base64
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

FRIENDLY_ROWS = {
    "logreg  Tier-2": "Team-composition model",
    "logreg  Tier-1 (M4)": "Champion-identity model",
    "LightGBM Tier-2 (M5)": "Gradient boosting",
    "naive champ-WR": "Naive champion win-rate rule",
    "coin flip (0.5)": "Coin flip",
    "base rate (const)": "Always predict the base rate",
}


def embed(path: Path) -> str:
    """PNG -> data URI, so the report is one portable file."""
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def figure_block(name: str, caption: str) -> str:
    uri = embed(config.FIGURES / name)
    if not uri:
        return f'<p class="missing">[missing figure: {name} -- run the pipeline first]</p>'
    return f'<figure><img src="{uri}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def metrics_table(h: dict) -> str:
    path = config.DATA_PROCESSED / "model_comparison.csv"
    if not path.exists():
        return '<p class="missing">[run python src/model.py first]</p>'
    t = pd.read_csv(path, index_col=0)
    best = h["best_model"]
    rows = []
    for name, r in t.iterrows():
        label = FRIENDLY_ROWS.get(name.strip(), name.strip())
        mark = ' class="best"' if name.strip() == best else ""
        star = " &starf;" if name.strip() == best else ""
        rows.append(
            f"<tr{mark}><td>{label}{star}</td><td class='n'>{r['log_loss']:.4f}</td>"
            f"<td class='n'>{r['roc_auc']:.3f}</td><td class='n'>{r['brier']:.4f}</td>"
            f"<td class='n'>{r['accuracy']:.1%}</td></tr>")
    return ("<table><thead><tr><th>Model</th><th class='n'>Log-loss &darr;</th>"
            "<th class='n'>ROC-AUC</th><th class='n'>Brier &darr;</th>"
            "<th class='n'>Accuracy</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")


def build() -> Path:
    hp = config.DATA_PROCESSED / "headline.json"
    if not hp.exists():
        raise SystemExit("data/processed/headline.json missing -- run python src/model.py first.")
    h = json.loads(hp.read_text(encoding="utf-8"))

    tiers = " &middot; ".join(
        f"{k.capitalize()} {v:,}" for k, v in
        sorted(h.get("tiers", {}).items(), key=lambda kv: -kv[1])) or "-"
    auc_pp = (h["roc_auc"] - 0.5) * 100
    acc_pp = (h["accuracy"] - h["majority_accuracy"]) * 100

    html = TEMPLATE
    for key, val in {
        "__GAMES__": f"{h['games']:,}",
        "__REGION__": h["region"],
        "__PLATFORM__": h["platform"],
        "__PATCHES__": " + ".join(h["patches"]),
        "__TIERS__": tiers,
        "__AUC__": f"{h['roc_auc']:.3f}",
        "__AUC_PP__": f"{auc_pp:+.1f}",
        "__ACC__": f"{h['accuracy']:.1%}",
        "__ACC_PP__": f"{acc_pp:+.1f}",
        "__MAJ__": f"{h['majority_accuracy']:.1%}",
        "__LOGLOSS__": f"{h['log_loss']:.4f}",
        "__COIN__": f"{h['baseline_log_loss']:.4f}",
        "__CONST__": f"{h['constant_log_loss']:.4f}",
        "__PMIN__": f"{h['prob_min']:.0%}",
        "__PMAX__": f"{h['prob_max']:.0%}",
        "__BLUEWR__": f"{h['blue_win_rate']:.1%}",
        "__TABLE__": metrics_table(h),
        "__FIG_SPREAD__": figure_block(
            "winprob_spread.png",
            "Every game the model scored. If champion select decided matches, these would "
            "spread toward 0% and 100% instead of piling up on 50/50."),
        "__FIG_EFFECTS__": figure_block(
            "champion_effects.png",
            "The biggest movers among picks and bans, in percentage points of win chance. "
            "The strongest is worth under two games in a hundred."),
        "__FIG_CALIB__": figure_block(
            "calibration_baseline.png",
            "The model is well calibrated -- when it says 48%, blue really wins about 48% "
            "of the time. It is honest; it just never has much to say."),
        "__FIG_GAMES__": figure_block(
            "my_games.png",
            "My own recent ranked games. The draft called nearly all of them a coin flip."),
        "__FIG_GAP__": figure_block(
            "my_execution_gap.png",
            "What champion select promised me, next to what actually happened."),
        "__FIG_ODDS__": figure_block(
            "my_odds_vs_outcome.png",
            "Wins and losses on the same axis. They overlap instead of separating."),
    }.items():
        html = html.replace(key, val)

    out = config.ROOT / "reports" / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


TEMPLATE = r"""<meta charset="utf-8">
<title>How much does the draft decide a League game?</title>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb; --panel: #ffffff; --ink: #0b0b0b; --ink-2: #52514e;
    --muted: #898781; --grid: #e1e0d9; --line: #c3c2b7;
    --blue: #2a78d6; --accent: #1baf7a;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --surface: #1a1a19; --panel: #232322; --ink: #ffffff; --ink-2: #c3c2b7;
      --muted: #898781; --grid: #2c2c2a; --line: #383835;
      --blue: #3987e5; --accent: #199e70;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 40px 20px 80px; background: var(--surface); color: var(--ink);
         font: 16px/1.65 system-ui, -apple-system, "Segoe UI", sans-serif; }
  .wrap { max-width: 780px; margin: 0 auto; }
  h1 { font-size: 34px; line-height: 1.2; margin: 0 0 10px; letter-spacing: -0.02em; }
  h2 { font-size: 22px; margin: 46px 0 10px; letter-spacing: -0.01em; }
  h3 { font-size: 16px; margin: 28px 0 6px; }
  .lede { font-size: 18px; color: var(--ink-2); margin: 0 0 28px; }
  p { margin: 0 0 14px; }
  .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
          gap: 12px; margin: 26px 0 8px; }
  .kpi { background: var(--panel); border: 1px solid var(--grid); border-radius: 10px;
         padding: 14px 16px; }
  .kpi b { display: block; font-size: 27px; letter-spacing: -0.02em; }
  .kpi span { color: var(--muted); font-size: 12.5px; }
  figure { margin: 22px 0; }
  figure img { width: 100%; height: auto; border: 1px solid var(--grid); border-radius: 10px;
               background: #fcfcfb; }
  figcaption { color: var(--ink-2); font-size: 13.5px; margin-top: 8px; }
  table { width: 100%; border-collapse: collapse; margin: 14px 0; font-size: 14.5px; }
  th { text-align: left; font-size: 11.5px; letter-spacing: .06em; text-transform: uppercase;
       color: var(--muted); padding: 5px 10px 8px 0; }
  td { padding: 7px 10px 7px 0; border-top: 1px solid var(--grid); }
  .n { text-align: right; font-variant-numeric: tabular-nums; }
  tr.best td { font-weight: 700; color: var(--blue); }
  dl.prov { display: grid; grid-template-columns: max-content 1fr; gap: 6px 18px;
            background: var(--panel); border: 1px solid var(--grid); border-radius: 10px;
            padding: 16px 18px; font-size: 14px; margin: 0 0 10px; }
  dl.prov dt { color: var(--muted); }
  dl.prov dd { margin: 0; }
  blockquote { margin: 20px 0; padding: 14px 18px; border-left: 3px solid var(--accent);
               background: var(--panel); border-radius: 0 8px 8px 0; }
  blockquote p:last-child { margin: 0; }
  ul { margin: 0 0 14px; padding-left: 22px; } li { margin-bottom: 7px; }
  .missing { color: #c62828; font-family: monospace; font-size: 13px; }
  footer { margin-top: 54px; padding-top: 18px; border-top: 1px solid var(--grid);
           color: var(--muted); font-size: 13px; }
</style>

<div class="wrap">
<h1>How much does the draft actually decide a League game?</h1>
<p class="lede">A calibrated win-probability model built only from champion select &mdash;
ten champions and their bans, nothing else. The honest answer to the question is the
finding, and the answer is: <b>very little</b>.</p>

<dl class="prov">
  <dt>Server</dt><dd>__REGION__ &mdash; platform <code>__PLATFORM__</code></dd>
  <dt>Queue</dt><dd>Ranked Solo/Duo (420)</dd>
  <dt>Ranks</dt><dd>__TIERS__</dd>
  <dt>Patches</dt><dd>__PATCHES__</dd>
  <dt>Matches</dt><dd><b>__GAMES__</b></dd>
</dl>

<div class="kpis">
  <div class="kpi"><b>__AUC__</b><span>ROC-AUC (__AUC_PP__ pp vs chance)</span></div>
  <div class="kpi"><b>__ACC_PP__ pp</b><span>accuracy vs always guessing the common side</span></div>
  <div class="kpi"><b>__PMIN__&ndash;__PMAX__</b><span>the entire range of predictions made</span></div>
</div>

<h2>The headline</h2>
<blockquote><p>After seeing <b>both complete drafts</b> &mdash; all ten champions and every
ban &mdash; the best model reaches <b>ROC-AUC __AUC__</b>, just __AUC_PP__ points above a
coin flip. Its accuracy of __ACC__ is only __ACC_PP__ points better than blindly always
guessing the more common side (__MAJ__). Across __GAMES__ games, every single prediction it
made fell between __PMIN__ and __PMAX__.</p></blockquote>

<p>That last number is the one worth sitting with. The model is <em>never confident</em>,
because the draft never justifies confidence.</p>

__FIG_SPREAD__

<h2>Which champions move the needle?</h2>
<p>A logistic regression on champion picks lets each champion's effect be read off directly and
converted into percentage points of win chance. The champions below are the most extreme in the
entire dataset &mdash; and the strongest of them shifts the game by less than <b>two games in a
hundred</b>.</p>

__FIG_EFFECTS__

<h2>Is the model any good?</h2>
<p>Evaluation is calibration-forward: the useful question is not "how often is it right" but
"when it says 55%, does blue win 55% of the time?" Every model below was scored by nested
cross-validation on identical folds &mdash; hyperparameters tuned on training rows only, so
no model ever saw the games it is judged on.</p>

__TABLE__

<p>Two results worth stating plainly. First, the <b>gradient-boosted model lost</b> to plain
logistic regression &mdash; the extra complexity found no signal to exploit. Second, among the
linear models the <b>archetype representation</b> (team shape: engage, hard CC, frontline, AD/AP
balance) edged raw <b>champion identity</b>: with enough games, <em>what kind of team you built</em>
carries slightly more signal than <em>which exact champions</em> are on it. And regularisation
mattered more than model choice &mdash; an earlier hand-picked penalty was beaten by a coin flip,
overconfident rather than wrong on average.</p>

__FIG_CALIB__

<h2>My own games</h2>
<p>The same model applied to my recent ranked games. These matches are explicitly held out of
training, so this is a genuine out-of-sample test rather than the model recalling games it
had already memorised.</p>

__FIG_GAMES__
__FIG_GAP__
__FIG_ODDS__

<h2>Limitations</h2>
<ul>
<li><b>One region, one rank band.</b> Apex tiers on __REGION__ only. Nothing here should be
assumed to transfer to other servers or to lower ranks, where the meta and execution differ.</li>
<li><b>Patch scope.</b> Restricted to __PATCHES__ for a coherent meta, so the champion effects
are a snapshot, not permanent truths.</li>
<li><b>Correlational, not causal.</b> The model sees final compositions with no pick order, no
counter-pick context, and no idea who played what. A champion's coefficient reflects who tends
to pick it and into what, not the champion's intrinsic strength.</li>
<li><b>Archetype labels are hand-curated.</b> The engage / enchanter / poke / hard-CC columns are
a judgement call, not official data.</li>
<li><b>Rank labels are approximate.</b> Match-V5 has no game-tier field, so a game's tier is
inferred from the player whose history surfaced it.</li>
<li><b>Solo-queue variance is enormous.</b> Ten players, connection issues, tilt and one early
mistake all swamp the draft. That is precisely the finding.</li>
</ul>

<h2>So what should you take from this?</h2>
<p>Draft matters, but not nearly as much as the discourse around it suggests. If you lost a
solo-queue game, the composition on the loading screen almost certainly was not why. Blue side
won __BLUEWR__ of these games, and knowing all twenty champions barely improves on that.</p>
<p><b>The rest is execution.</b></p>

<footer>Built with the free Riot developer API and open-source tools. Model: L2-regularised
logistic regression on champion picks and bans, evaluated by nested cross-validation.<br>
This project is not endorsed by Riot Games and does not reflect the views or opinions of Riot
Games or anyone officially involved in producing or managing League of Legends.</footer>
</div>
"""


if __name__ == "__main__":
    out = build()
    kb = out.stat().st_size / 1024
    print(f"  wrote {out.relative_to(config.ROOT)}  ({kb:.0f} KB, figures embedded)")
    print("  open it directly in a browser -- no server, no internet.")
