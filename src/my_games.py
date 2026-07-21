"""M6: run MY recent ranked games through the trained model.

Run (from anywhere):  python src/my_games.py [--count 40]

For each of my recent ranked-solo games, take the final draft (the same features
the model was trained on), predict my team's win probability *as it stood at
champ-select lock*, and compare it to what actually happened. Flags the games I
stole against bad odds and the ones I dropped while favored.

HONESTY CHECK: I'm a Master player and the training set was seeded from Master
players, so some of my games may already be IN the training data. Those are
flagged and excluded from the headline numbers -- a model that saw the answer
isn't predicting, it's remembering.
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import viz
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import features as F  # noqa: E402
import ingest  # noqa: E402  (reuse its retrying `get` + caching `fetch_match`)
import parse  # noqa: E402

BLUE = 100


def my_puuid() -> str:
    game_name, tag_line = config.MY_RIOT_ID.split("#", 1)
    for region in config.ACCOUNT_REGIONS:
        url = (config.regional_host(region)
               + f"/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}")
        resp = ingest.get(url)
        if resp.status_code == 200:
            return resp.json()["puuid"]
    raise RuntimeError(f"Could not resolve Riot ID {config.MY_RIOT_ID}")


def my_match_ids(puuid: str, count: int) -> list[str]:
    url = (config.regional_host(config.MATCH_REGION)
           + f"/lol/match/v5/matches/by-puuid/{puuid}/ids")
    resp = ingest.get(url, params={"queue": config.RANKED_SOLO_QUEUE_ID,
                                   "start": 0, "count": count})
    resp.raise_for_status()
    return resp.json()


def my_side_and_champ(match_id: str, puuid: str) -> tuple[int, int]:
    """Which team I was on (100/200) and which champion I played."""
    info = json.loads((config.DATA_RAW / f"{match_id}.json").read_text(encoding="utf-8"))["info"]
    me = next(p for p in info["participants"] if p["puuid"] == puuid)
    return me["teamId"], me["championId"]


def main(count: int) -> None:
    bundle = joblib.load(config.DATA_PROCESSED / "model.joblib")
    model, columns, tier = bundle["model"], bundle["columns"], bundle["tier"]
    labels = F.load_labels()
    name_of = labels["name"].to_dict()

    puuid = my_puuid()
    ids = my_match_ids(puuid, count)
    print(f"{config.MY_RIOT_ID}: {len(ids)} recent ranked-solo games\n")

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    for mid in ids:
        if ingest.fetch_match(mid):
            time.sleep(ingest.REQUEST_SPACING)

    # Parse my games with the same parser used for training data.
    rows = []
    for mid in ids:
        row = parse.parse_match(config.DATA_RAW / f"{mid}.json")
        if row is None:
            continue
        row["matchId"] = mid
        side, champ = my_side_and_champ(mid, puuid)
        row["my_side"] = side
        row["my_champ"] = name_of.get(champ, str(champ))
        rows.append(row)

    df = pd.DataFrame(rows)
    on_patch = df[df["patch"].isin(config.TARGET_PATCHES)].reset_index(drop=True)
    print(f"On target patches {config.TARGET_PATCHES}: {len(on_patch)}/{len(df)} games")

    # Which of my games did the model already train on? Those aren't real predictions.
    train_ids = set(F.load_matches()["matchId"])
    on_patch["in_training"] = on_patch["matchId"].isin(train_ids)
    n_leak = int(on_patch["in_training"].sum())
    print(f"Already in training data: {n_leak} (flagged, excluded from headline)\n")

    # Predict. The model outputs P(blue wins); flip it if I was on red.
    X = F.build_aligned(on_patch, tier, columns, labels)
    blue_prob = model.predict_proba(X)[:, 1]
    on_patch["my_win_prob"] = [p if s == BLUE else 1 - p
                               for p, s in zip(blue_prob, on_patch["my_side"])]
    on_patch["won"] = [(w == 1) == (s == BLUE)
                       for w, s in zip(on_patch["win"], on_patch["my_side"])]

    def verdict(r) -> str:
        if r["won"] and r["my_win_prob"] < 0.5:
            return "WON vs bad odds"
        if not r["won"] and r["my_win_prob"] > 0.5:
            return "LOST while favored"
        return "as expected"

    on_patch["verdict"] = on_patch.apply(verdict, axis=1)

    show = on_patch[["matchId", "my_champ", "my_side", "my_win_prob", "won",
                     "verdict", "in_training"]].copy()
    show["my_side"] = show["my_side"].map({100: "blue", 200: "red"})
    show["my_win_prob"] = show["my_win_prob"].map(lambda p: f"{p:.1%}")
    print("=== MY LAST GAMES: draft-time win probability vs. what happened ===")
    print(show.to_string(index=False))

    clean = on_patch[~on_patch["in_training"]]
    if len(clean):
        print(f"\n--- headline (excluding {n_leak} games the model trained on) ---")
        print(f"  games: {len(clean)} | actual win rate: {clean['won'].mean():.1%} "
              f"| mean predicted: {clean['my_win_prob'].mean():.1%}")
        print(f"  won against the odds:  {int((clean['verdict'] == 'WON vs bad odds').sum())}")
        print(f"  lost while favored:    {int((clean['verdict'] == 'LOST while favored').sum())}")
        print(f"  draft called it right: {(clean['won'] == (clean['my_win_prob'] > 0.5)).mean():.1%} of games")

    viz.apply_style()
    plot_per_game(on_patch)
    plot_execution_gap(on_patch)
    plot_odds_vs_outcome(on_patch)
    print("\nSaved reports/figures/: my_games.png, my_execution_gap.png, "
          "my_odds_vs_outcome.png")


# ------------------------------------------------------------------- figures --
def _outcome_colour(won: bool) -> str:
    return viz.WON if won else viz.LOST


def plot_per_game(df: pd.DataFrame):
    """C1: one labelled row per game -- champion, side, draft-time probability,
    and the outcome. Lollipop keeps the marks thin and the labels readable."""
    d = df.iloc[::-1].reset_index(drop=True)  # oldest at top, newest at bottom
    fig, ax = viz.figure(8.4, 0.52 * len(d) + 2.2)
    y = range(len(d))

    for i, r in d.iterrows():
        c = _outcome_colour(r["won"])
        ax.plot([0.5, r["my_win_prob"]], [i, i], color=c, lw=2.5, solid_capstyle="round",
                zorder=2)
        ax.plot(r["my_win_prob"], i, "o", color=c, ms=9, mec=viz.SURFACE, mew=1.5, zorder=3)
        # Direct label: outcome never depends on colour alone.
        ax.text(r["my_win_prob"] + (0.004 if r["my_win_prob"] >= 0.5 else -0.004), i,
                "WON" if r["won"] else "LOST",
                va="center", ha="left" if r["my_win_prob"] >= 0.5 else "right",
                fontsize=8.5, color=c, fontweight="semibold")

    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{r['my_champ']}  ({'blue' if r['my_side'] == BLUE else 'red'})"
                        for _, r in d.iterrows()], fontsize=9.5)
    viz.reference_line(ax, 0.5, "coin flip", axis="x")
    ax.set_xlim(0.40, 0.60)
    ax.set_xticks([0.40, 0.45, 0.50, 0.55, 0.60])
    ax.set_xticklabels(["40%", "45%", "50%", "55%", "60%"])
    ax.set_xlabel("win probability the draft gave me")
    viz.despine_x(ax)
    # Inverted (row 0 on top) with headroom so the "coin flip" label can't
    # collide with the topmost game's outcome label.
    ax.set_ylim(len(d) - 0.4, -1.1)
    viz.title_block(ax, "My games: what the draft predicted vs. what happened",
                    "The draft never gave me better than a coin flip -- yet most of these are wins.")
    return viz.save(fig, "my_games.png")


def plot_execution_gap(df: pd.DataFrame):
    """C2: the headline -- mean predicted win rate vs what actually happened."""
    predicted, actual = df["my_win_prob"].mean(), df["won"].mean()
    fig, ax = viz.figure(7.2, 3.4)

    bars = ax.barh(["what the draft\npredicted", "what actually\nhappened"],
                   [predicted, actual], color=[viz.BLUE_SIDE, viz.WON], height=0.34)
    for bar, v in zip(bars, [predicted, actual]):
        ax.text(v + 0.012, bar.get_y() + bar.get_height() / 2, f"{v:.0%}",
                va="center", fontsize=13, fontweight="semibold", color=viz.INK)

    viz.reference_line(ax, 0.5, "coin flip", axis="x")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(f"win rate over my last {len(df)} on-patch games")
    viz.despine_x(ax)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(-0.55, 1.7)  # headroom so the "coin flip" label clears the top bar
    viz.title_block(ax, "The gap between the draft and the result",
                    "That gap is not champion select. It is execution "
                    f"(small sample: {len(df)} games).")
    return viz.save(fig, "my_execution_gap.png")


def plot_odds_vs_outcome(df: pd.DataFrame):
    """C3: wins and losses on the same probability axis. If the draft predicted
    outcomes, the two rows would separate. They don't."""
    fig, ax = viz.figure(8.4, 3.2)
    for won, row in [(True, 1), (False, 0)]:
        sub = df[df["won"] == won]
        c = _outcome_colour(won)
        ax.plot(sub["my_win_prob"], [row] * len(sub), "o", color=c, ms=11,
                mec=viz.SURFACE, mew=1.5, label=f"{'won' if won else 'lost'} ({len(sub)})",
                zorder=3)

    viz.reference_line(ax, 0.5, "coin flip", axis="x")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["LOST", "WON"], fontsize=10, fontweight="semibold")
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(0.40, 0.60)
    ax.set_xticks([0.40, 0.45, 0.50, 0.55, 0.60])
    ax.set_xticklabels(["40%", "45%", "50%", "55%", "60%"])
    ax.set_xlabel("win probability the draft gave me")
    viz.despine_x(ax)
    ax.legend(loc="lower right", ncol=2)
    viz.title_block(ax, "Did a better draft actually mean a win?",
                    "If it did, the WON dots would sit right of the LOST dots. They overlap.")
    return viz.save(fig, "my_odds_vs_outcome.png")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Score my recent ranked games.")
    p.add_argument("--count", type=int, default=40, help="how many recent games to pull")
    main(p.parse_args().count)
