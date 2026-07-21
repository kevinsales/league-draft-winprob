"""Manual draft (10 champions + optional bans) -> win probability. Built LAST.

Usage (from anywhere):
  python src/predict.py --blue "Aatrox,Vi,Ahri,Jinx,Thresh" \
                        --red  "Garen,Lee Sin,Orianna,Caitlyn,Leona" \
                        --bans "Yasuo,Zed,Lulu"          # optional, up to 10

Champion names are matched forgivingly (case/punctuation-insensitive, unique
prefixes OK: "kaisa" -> Kai'Sa, "jarvan" -> Jarvan IV). Requires the trained
model from model.py (data/processed/model.joblib).
"""

import difflib
import re
import sys
from pathlib import Path

import joblib
import viz
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import features as F  # noqa: E402


def _norm(name: str) -> str:
    """'Kai'Sa' -> 'kaisa', 'Nunu & Willump' -> 'nunuwillump'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_champion(raw: str, labels: pd.DataFrame) -> int:
    """User input -> championId. Exact normalized match, then unique prefix,
    then fail with suggestions."""
    by_norm = {_norm(n): cid for cid, n in labels["name"].items()}
    q = _norm(raw)
    if q in by_norm:
        return by_norm[q]
    prefix = [n for n in by_norm if n.startswith(q)]
    if len(prefix) == 1:
        return by_norm[prefix[0]]
    close = difflib.get_close_matches(q, by_norm.keys(), n=3, cutoff=0.5)
    options = [labels.loc[by_norm[n], "name"] for n in (prefix or close)]
    raise SystemExit(f"Unknown champion '{raw}'. Did you mean: {', '.join(options) or '???'}")


def predict_draft(blue: list[str], red: list[str], bans: list[str] = ()) -> float:
    """10 champion names (+ optional bans) -> P(blue side wins)."""
    bundle = joblib.load(config.DATA_PROCESSED / "model.joblib")
    model, columns, tier = bundle["model"], bundle["columns"], bundle["tier"]
    labels = F.load_labels()

    if len(blue) != 5 or len(red) != 5:
        raise SystemExit(f"Need exactly 5 champions per side (got {len(blue)} blue, {len(red)} red).")
    blue_ids = [resolve_champion(c, labels) for c in blue]
    red_ids = [resolve_champion(c, labels) for c in red]
    ban_ids = [resolve_champion(c, labels) for c in bans]

    # One row in the exact shape parse.py produces (win is a dummy label).
    row: dict = {"win": 0}
    row.update({f"blue{i}": c for i, c in enumerate(blue_ids, 1)})
    row.update({f"red{i}": c for i, c in enumerate(red_ids, 1)})
    padded = (list(ban_ids) + [-1] * 10)[:10]
    row.update({f"blue_ban{i}": c for i, c in enumerate(padded[:5], 1)})
    row.update({f"red_ban{i}": c for i, c in enumerate(padded[5:], 1)})
    df = pd.DataFrame([row])

    # Warn about champions the model never saw in training (they contribute 0).
    if tier == "tier1":
        for cid, side in [(c, "blue") for c in blue_ids] + [(c, "red") for c in red_ids]:
            if f"{side}_pick_{cid}" not in columns:
                print(f"  note: {labels.loc[cid, 'name']} ({side}) wasn't in the "
                      "training data -- the model has no opinion on it.")

    X = F.build_aligned(df, tier, columns, labels)
    return float(model.predict_proba(X)[:, 1][0])


def contributions(blue: list[str], red: list[str], bans: list[str] = ()) -> pd.DataFrame:
    """Per-pick contribution to the prediction, in log-odds.

    Only meaningful because the shipped model is LINEAR: the log-odds is just the
    intercept plus each active feature's coefficient, so a draft's probability
    decomposes exactly into 'what each pick contributed'.
    """
    bundle = joblib.load(config.DATA_PROCESSED / "model.joblib")
    model, columns = bundle["model"], bundle["columns"]
    est = getattr(model, "best_estimator_", model)
    if not hasattr(est, "coef_"):
        raise SystemExit("Contribution breakdown needs the linear model (no coef_ found).")
    coef = pd.Series(est.coef_[0], index=columns)
    labels = F.load_labels()

    rows = []
    for names, side in ((blue, "blue"), (red, "red")):
        for raw in names:
            cid = resolve_champion(raw, labels)
            col = f"{side}_pick_{cid}"
            rows.append({"who": labels.loc[cid, "name"], "kind": f"{side} pick",
                         "effect": float(coef.get(col, 0.0))})
    for raw in bans:
        cid = resolve_champion(raw, labels)
        rows.append({"who": labels.loc[cid, "name"], "kind": "ban",
                     "effect": float(coef.get(f"ban_{cid}", 0.0))})
    return pd.DataFrame(rows)


def plot_scorecard(blue, red, bans, prob: float, contrib: pd.DataFrame):
    """B1+B2: one card -- both comps, the probability, and what drove it."""
    import matplotlib.pyplot as plt

    viz.apply_style()
    fig = plt.figure(figsize=(11, 6.4))
    # top= leaves room for the page header so it can't collide with panel titles
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1, 1.4],
                          hspace=0.5, wspace=0.30,
                          top=0.80, bottom=0.10, left=0.05, right=0.97)

    # --- the two comps
    ax_comp = fig.add_subplot(gs[0, 0])
    ax_comp.axis("off")
    ax_comp.text(0, 1.0, "BLUE", color=viz.BLUE_SIDE, fontweight="bold", fontsize=10)
    ax_comp.text(0.52, 1.0, "RED", color=viz.RED_SIDE, fontweight="bold", fontsize=10)
    for i, (b, r) in enumerate(zip(blue, red)):
        ax_comp.text(0, 0.82 - i * 0.17, b, fontsize=10.5, color=viz.INK)
        ax_comp.text(0.52, 0.82 - i * 0.17, r, fontsize=10.5, color=viz.INK)

    # --- the probability bar (the headline number)
    ax_p = fig.add_subplot(gs[1, 0])
    ax_p.barh([0], [prob], color=viz.BLUE_SIDE, height=0.34)
    ax_p.barh([0], [1 - prob], left=[prob], color=viz.RED_SIDE, height=0.34)
    ax_p.axvline(0.5, color=viz.INK_MUTED, lw=1)
    ax_p.set_xlim(0, 1)
    ax_p.set_ylim(-0.8, 0.95)
    ax_p.set_yticks([])
    ax_p.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax_p.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax_p.grid(False)
    for s in ("left", "right", "top"):
        ax_p.spines[s].set_visible(False)
    ax_p.text(0, 0.34, f"{prob:.1%}", color=viz.BLUE_SIDE, fontsize=20,
              fontweight="bold", va="bottom")
    ax_p.text(1, 0.34, f"{1 - prob:.1%}", color=viz.RED_SIDE, fontsize=20,
              fontweight="bold", va="bottom", ha="right")
    ax_p.text(0.5, -0.62, "50% = the draft tells you nothing", fontsize=8.5,
              color=viz.INK_MUTED, ha="center")

    # --- contribution breakdown
    ax_c = fig.add_subplot(gs[:, 1])
    c = contrib[contrib["effect"] != 0].copy()
    if c.empty:
        ax_c.axis("off")
        ax_c.text(0.5, 0.5, "No champion in this draft\nhad a learned effect.",
                  ha="center", va="center", color=viz.INK_SECONDARY)
    else:
        c = c.sort_values("effect")
        colours = [viz.RED_SIDE if v < 0 else viz.BLUE_SIDE for v in c["effect"]]
        ax_c.barh(range(len(c)), c["effect"], color=colours, height=0.7)
        ax_c.set_yticks(range(len(c)))
        ax_c.set_yticklabels([f"{r.who}  ({r.kind})" for r in c.itertuples()], fontsize=9)
        ax_c.axvline(0, color=viz.BASELINE, lw=1)
        viz.despine_x(ax_c)
        ax_c.set_xlabel("effect on blue win probability (log-odds)")
    viz.title_block(ax_c, "What moved the needle",
                    "Blue bars favour blue side, red favour red.")

    fig.text(0.05, 0.945, "Draft win-probability scorecard", fontsize=15,
             fontweight="semibold", color=viz.INK)
    fig.text(0.05, 0.905, f"Master SEA, patches {' + '.join(config.TARGET_PATCHES)}  -  "
             "draft alone rarely moves a solo-queue game far from a coin flip.",
             fontsize=9.5, color=viz.INK_SECONDARY)
    return viz.save(fig, "draft_scorecard.png")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Draft -> win probability (manual input).")
    p.add_argument("--blue", required=True, help="5 blue-side champions, comma-separated")
    p.add_argument("--red", required=True, help="5 red-side champions, comma-separated")
    p.add_argument("--bans", default="", help="up to 10 bans, comma-separated (optional)")
    p.add_argument("--figure", action="store_true",
                   help="also save a scorecard PNG to reports/figures/")
    a = p.parse_args()

    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    blue, red, bans = split(a.blue), split(a.red), split(a.bans)
    prob = predict_draft(blue, red, bans)

    print(f"\n  Blue side win probability: {prob:.1%}")
    print(f"  Red  side win probability: {1 - prob:.1%}")
    print("\n  (Trained on Master SEA, patches "
          f"{' + '.join(config.TARGET_PATCHES)}. Honest reminder: draft alone "
          "rarely moves a solo-queue game far from a coin flip.)")

    if a.figure:
        contrib = contributions(blue, red, bans)
        path = plot_scorecard(blue, red, bans, prob, contrib)
        print(f"\n  Saved {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
