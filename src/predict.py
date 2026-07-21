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


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Draft -> win probability (manual input).")
    p.add_argument("--blue", required=True, help="5 blue-side champions, comma-separated")
    p.add_argument("--red", required=True, help="5 red-side champions, comma-separated")
    p.add_argument("--bans", default="", help="up to 10 bans, comma-separated (optional)")
    a = p.parse_args()

    split = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
    prob = predict_draft(split(a.blue), split(a.red), split(a.bans))

    print(f"\n  Blue side win probability: {prob:.1%}")
    print(f"  Red  side win probability: {1 - prob:.1%}")
    print("\n  (Trained on Master SEA, patches "
          f"{' + '.join(config.TARGET_PATCHES)}. Honest reminder: draft alone "
          "rarely moves a solo-queue game far from a coin flip.)")


if __name__ == "__main__":
    main()
