"""Cached raw match JSON (data/raw/) -> one tidy row per match -> data/processed/.

Run (from anywhere):  python src/parse.py

Output: data/processed/matches.csv, one row per ranked-solo match on a
target patch, with the 10 champion ids (by side), both teams' bans, the patch,
and the label `win` (1 = blue/team-100 won). All of these are known at champ
select, so there is no leakage by construction.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

BLUE, RED = 100, 200  # Riot teamId constants


def patch_of(game_version: str) -> str:
    """'16.13.791.5903' -> '16.13' (major.minor identifies the meta)."""
    return ".".join(game_version.split(".")[:2])


def parse_match(path: Path) -> dict | None:
    """One cached match file -> a flat dict of columns, or None to skip it.

    Skips non-ranked-solo games and any match missing the full 5v5 shape.
    Champions are kept in participant order (roughly role order); side identity
    is what matters for the model, not the slot index.
    """
    info = json.loads(path.read_text(encoding="utf-8"))["info"]
    if info.get("queueId") != config.RANKED_SOLO_QUEUE_ID:
        return None

    parts = info["participants"]
    blue = [p["championId"] for p in parts if p["teamId"] == BLUE]
    red = [p["championId"] for p in parts if p["teamId"] == RED]
    if len(blue) != 5 or len(red) != 5:
        return None  # remakes / malformed games

    teams = {t["teamId"]: t for t in info["teams"]}
    # Bans: list of {championId, pickTurn}; championId -1 means no ban. Pad to 5.
    blue_bans = [b["championId"] for b in teams[BLUE]["bans"]]
    red_bans = [b["championId"] for b in teams[RED]["bans"]]
    blue_bans = (blue_bans + [-1] * 5)[:5]
    red_bans = (red_bans + [-1] * 5)[:5]

    row = {
        "matchId": info.get("gameId"),
        "patch": patch_of(info["gameVersion"]),
        "gameVersion": info["gameVersion"],
        # ms epoch -- kept so we can state the date range the data covers.
        "gameCreation": info.get("gameCreation"),
        "win": int(bool(teams[BLUE]["win"])),  # 1 = blue (team 100) win
    }
    row.update({f"blue{i}": c for i, c in enumerate(blue, 1)})
    row.update({f"red{i}": c for i, c in enumerate(red, 1)})
    row.update({f"blue_ban{i}": c for i, c in enumerate(blue_bans, 1)})
    row.update({f"red_ban{i}": c for i, c in enumerate(red_bans, 1)})
    return row


def build_dataframe() -> pd.DataFrame:
    files = sorted(config.DATA_RAW.glob("*.json"))
    if not files:
        raise RuntimeError("No cached matches in data/raw/. Run ingest.py first.")

    rows, skipped = [], 0
    for f in files:
        # matchId from the filename (e.g. SG2_162048586) is the reliable unique key.
        parsed = parse_match(f)
        if parsed is None:
            skipped += 1
            continue
        parsed["matchId"] = f.stem
        rows.append(parsed)

    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(subset="matchId").reset_index(drop=True)

    # Attach the tier of the seed player who surfaced each match (approximate --
    # match-v5 has no game-tier field). Written by ingest.py.
    tiers_path = config.DATA_PROCESSED / "seed_tiers.csv"
    if tiers_path.exists():
        tiers = pd.read_csv(tiers_path)
        df = df.merge(tiers, on="matchId", how="left")
        df["seed_tier"] = df["seed_tier"].fillna("UNKNOWN")
    else:
        df["seed_tier"] = "UNKNOWN"

    # Hold out my own games. my_games.py caches them into the same data/raw/, so
    # without this they would leak into training and M6 would be scoring the
    # model on games it had already memorised.
    holdout_path = config.DATA_PROCESSED / "my_match_ids.txt"
    n_holdout = 0
    if holdout_path.exists():
        mine = set(holdout_path.read_text(encoding="utf-8").split())
        n_holdout = int(df["matchId"].isin(mine).sum())
        df = df[~df["matchId"].isin(mine)].reset_index(drop=True)

    kept = df[df["patch"].isin(config.TARGET_PATCHES)].reset_index(drop=True)

    print(f"Parsed {len(files)} files: {before} valid, {skipped} skipped (non-420/malformed).")
    if n_holdout:
        print(f"Held out {n_holdout} of my own games (never trained on -- see my_games.py).")
    print(f"Patch filter {config.TARGET_PATCHES}: {len(kept)}/{len(df)} kept.")
    return kept


def main() -> None:
    df = build_dataframe()
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = config.DATA_PROCESSED / "matches.csv"
    df.to_csv(out, index=False)

    print(f"\nSaved {out.relative_to(config.ROOT)}  shape={df.shape}")
    print(f"win balance (blue win rate): {df['win'].mean():.1%}")
    print(f"unique matchIds: {df['matchId'].nunique()} (dupes: {len(df) - df['matchId'].nunique()})")
    print(f"seed tiers: {df['seed_tier'].value_counts().to_dict()}")
    if df["gameCreation"].notna().any():
        span = pd.to_datetime(df["gameCreation"], unit="ms")
        print(f"games played between {span.min():%Y-%m-%d} and {span.max():%Y-%m-%d}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\nhead:")
        print(df[["matchId", "patch", "win", "blue1", "blue2", "blue3", "blue4", "blue5",
                  "red1", "red2", "red3", "red4", "red5", "blue_ban1", "red_ban1"]].head())


if __name__ == "__main__":
    main()
