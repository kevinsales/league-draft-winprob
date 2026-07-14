"""Tier-1/2/3 feature builders from the parsed match table (data/processed).

Run (from anywhere):  python src/features.py   # prints a summary of each tier

Each builder returns (X, y) with X a feature DataFrame row-aligned to the labels
y = blue/team-100 win (1/0). All features are known at champ-select lock, so
there is no leakage from the match outcome.

  Tier 1  multi-hot champion vectors per side, plus a multi-hot ban block.
  Tier 2  archetype aggregates from champion_labels.csv: per-team counts and
          blue-minus-red diffs (engage/enchanter/poke/tank/mage/marksman, AD/AP
          split, hard-CC total, frontline).
  Tier 3  (OPTIONAL/STRETCH) champion empirical win-rate deltas. Marked optional;
          must be fit on training rows only (see note) and is unreliable at small
          sample sizes.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

BLUE_COLS = [f"blue{i}" for i in range(1, 6)]
RED_COLS = [f"red{i}" for i in range(1, 6)]
BAN_COLS = [f"{side}_ban{i}" for side in ("blue", "red") for i in range(1, 6)]

# Archetype flag columns in champion_labels.csv that we aggregate per team.
FLAG_COLS = ["is_tank", "is_mage", "is_marksman", "is_fighter", "is_assassin",
             "is_support", "engage", "enchanter", "poke", "hard_cc", "frontline"]


def load_matches() -> pd.DataFrame:
    path = config.DATA_PROCESSED / "matches.csv"
    if not path.exists():
        raise RuntimeError("data/processed/matches.csv missing. Run parse.py first.")
    return pd.read_csv(path)


def load_labels() -> pd.DataFrame:
    path = config.ROOT / "champion_labels.csv"
    if not path.exists():
        raise RuntimeError("champion_labels.csv missing. Run build_labels.py first.")
    return pd.read_csv(path).set_index("championId")


# ---------------------------------------------------------------- Tier 1 ------
def tier1_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Multi-hot: for each champion seen, a blue-pick and red-pick indicator;
    for each champion banned, a ban indicator (either team)."""
    pick_ids = sorted(pd.unique(df[BLUE_COLS + RED_COLS].values.ravel()))
    ban_ids = sorted(i for i in pd.unique(df[BAN_COLS].values.ravel()) if i != -1)

    cols = {}
    for cid in pick_ids:
        cols[f"blue_pick_{cid}"] = df[BLUE_COLS].eq(cid).any(axis=1).astype(int)
        cols[f"red_pick_{cid}"] = df[RED_COLS].eq(cid).any(axis=1).astype(int)
    for cid in ban_ids:
        cols[f"ban_{cid}"] = df[BAN_COLS].eq(cid).any(axis=1).astype(int)

    X = pd.DataFrame(cols, index=df.index)
    return X, df["win"]


# ---------------------------------------------------------------- Tier 2 ------
def _team_aggregate(champ_ids: list[int], labels: pd.DataFrame) -> dict:
    """Sum archetype flags + AD/AP counts over one team's 5 champions."""
    known = [c for c in champ_ids if c in labels.index]
    sub = labels.loc[known]
    agg = {flag: int(sub[flag].sum()) for flag in FLAG_COLS}
    agg["ad"] = int((sub["damage_type"] == "AD").sum())
    agg["ap"] = int((sub["damage_type"] == "AP").sum())
    return agg


def tier2_features(df: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Per-team archetype counts (blue_*, red_*) and blue-minus-red diffs (diff_*)."""
    unknown = set(pd.unique(df[BLUE_COLS + RED_COLS].values.ravel())) - set(labels.index)
    if unknown:
        print(f"  Tier2 WARNING: {len(unknown)} champ ids not in labels: {sorted(unknown)}")

    rows = []
    for _, r in df.iterrows():
        blue = _team_aggregate([r[c] for c in BLUE_COLS], labels)
        red = _team_aggregate([r[c] for c in RED_COLS], labels)
        row = {}
        for k in blue:
            row[f"blue_{k}"] = blue[k]
            row[f"red_{k}"] = red[k]
            row[f"diff_{k}"] = blue[k] - red[k]  # blue advantage, signed
        rows.append(row)

    X = pd.DataFrame(rows, index=df.index)
    return X, df["win"]


# ---------------------------------------------------------------- Tier 3 ------
def tier3_features(df: pd.DataFrame, train_idx=None) -> tuple[pd.DataFrame, pd.Series]:
    """OPTIONAL/STRETCH: blue-minus-red sum of per-champion empirical blue-side
    win rates, learned from the data.

    LEAKAGE NOTE: these rates are outcome-derived, so they must be computed on
    TRAINING rows only and then applied to all rows. Pass train_idx (the training
    index) when using this inside a model; if omitted it fits on the whole frame
    (fine for a quick summary, NOT for honest evaluation). At n=50 these rates are
    far too sparse to be reliable -- treat Tier 3 as a scaffold, not a result.
    """
    fit = df.loc[train_idx] if train_idx is not None else df

    # Per-champion blue win rate when that champion is on blue, and red win rate
    # when on red; global mean is the fallback for unseen champions.
    base = fit["win"].mean()
    blue_wr, red_wr = {}, {}
    for cid in pd.unique(df[BLUE_COLS + RED_COLS].values.ravel()):
        on_blue = fit[BLUE_COLS].eq(cid).any(axis=1)
        on_red = fit[RED_COLS].eq(cid).any(axis=1)
        blue_wr[cid] = fit.loc[on_blue, "win"].mean() if on_blue.any() else base
        red_wr[cid] = (1 - fit.loc[on_red, "win"]).mean() if on_red.any() else base

    def team_strength(ids, table):
        return sum(table.get(c, base) for c in ids)

    X = pd.DataFrame({
        "blue_champ_wr_sum": [team_strength([r[c] for c in BLUE_COLS], blue_wr)
                              for _, r in df.iterrows()],
        "red_champ_wr_sum": [team_strength([r[c] for c in RED_COLS], red_wr)
                             for _, r in df.iterrows()],
    }, index=df.index)
    X["diff_champ_wr"] = X["blue_champ_wr_sum"] - X["red_champ_wr_sum"]
    return X, df["win"]


def build_aligned(df: pd.DataFrame, tier: str, columns: list[str],
                  labels: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build features for NEW matches, aligned to a trained model's exact columns.

    Champions the model never saw are dropped; columns it expects but this data
    lacks are filled with 0. Used by my_games.py and predict.py, which must hand
    the saved model the same column layout it was fitted on.
    """
    if tier == "tier1":
        X, _ = tier1_features(df)
    else:
        X, _ = tier2_features(df, labels)
    return X.reindex(columns=columns, fill_value=0)


def main() -> None:
    df = load_matches()
    labels = load_labels()
    print(f"Matches: {len(df)} rows | labels: {len(labels)} champions\n")

    X1, y = tier1_features(df)
    print(f"Tier 1 (multi-hot picks + bans): X={X1.shape}, aligned={len(X1) == len(y)}, "
          f"nonzero/row~{X1.sum(axis=1).mean():.0f}")

    X2, _ = tier2_features(df, labels)
    print(f"Tier 2 (archetype aggregates + diffs): X={X2.shape}, aligned={len(X2) == len(y)}")
    with pd.option_context("display.width", 200):
        print("  example diff columns (blue minus red), first 5 rows:")
        cols = ["diff_engage", "diff_enchanter", "diff_poke", "diff_hard_cc",
                "diff_frontline", "diff_ad", "diff_ap"]
        print(X2[cols].head().to_string())

    X3, _ = tier3_features(df)  # whole-frame fit -> summary only (see leakage note)
    print(f"\nTier 3 (OPTIONAL, whole-frame fit -- not for eval): X={X3.shape}, "
          f"aligned={len(X3) == len(y)}")
    print("  NOTE: n=50 makes per-champion win rates too sparse to trust; scaffold only.")


if __name__ == "__main__":
    main()
