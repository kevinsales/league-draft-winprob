"""Build champion_labels.csv from free Data Dragon data (one-off, re-runnable).

Run (from anywhere):  python src/build_labels.py

Two layers of columns:
  OBJECTIVE (straight from Data Dragon, reproducible):
    championId, name, tags, damage_type*, is_tank/mage/marksman/fighter/assassin/
    support, frontline
    (*damage_type is a documented heuristic -- see damage_type() below.)
  CURATED (hand-labeled starting point -- REVIEW/EDIT THESE):
    engage, enchanter, poke, hard_cc
    These four archetypes aren't in any free API, so they're seeded from the
    conservative, clear-cut lists below. Edit champion_labels.csv (or these sets)
    to refine them; the model's Tier-2 features are only as good as these labels.
"""

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

DDRAGON_VERSION = "16.13.1"  # matches our data patch (16.13)

# --- Curated archetype sets (Data Dragon 'name' spelling). Conservative on
#     purpose: only clear-cut members, so false positives stay low. Curate me. ---
ENGAGE = {
    "Alistar", "Amumu", "Diana", "Galio", "Gragas", "Hecarim", "Jarvan IV",
    "Kennen", "Leona", "Malphite", "Maokai", "Nautilus", "Nocturne", "Ornn",
    "Rakan", "Rell", "Sejuani", "Sion", "Skarner", "Thresh", "Vi", "Wukong", "Zac",
}
ENCHANTER = {
    "Janna", "Karma", "Lulu", "Milio", "Nami", "Renata Glasc", "Seraphine",
    "Sona", "Soraka", "Taric", "Yuumi", "Ivern",
}
POKE = {
    "Caitlyn", "Corki", "Ezreal", "Jayce", "Lux", "Nidalee", "Varus",
    "Vel'Koz", "Xerath", "Ziggs", "Zoe",
}
HARD_CC = {  # reliable, easy-to-land or point-and-click lockdown
    "Alistar", "Amumu", "Annie", "Ashe", "Blitzcrank", "Braum", "Galio", "Gnar",
    "Jarvan IV", "Leona", "Lissandra", "Malphite", "Maokai", "Morgana", "Nautilus",
    "Nunu & Willump", "Ornn", "Poppy", "Rakan", "Rammus", "Rell", "Sejuani",
    "Sion", "Skarner", "Taric", "Thresh", "Twisted Fate", "Urgot", "Veigar",
    "Vi", "Warwick", "Wukong", "Zac",
}


def damage_type(tags: list[str], info: dict) -> str:
    """Heuristic AD/AP/mixed. Marksman->AD, Mage->AP, else use the attack vs
    magic rating (Data Dragon 'info'), calling it 'mixed' when they tie."""
    if "Marksman" in tags:
        return "AD"
    if "Mage" in tags:
        return "AP"
    if info["magic"] > info["attack"]:
        return "AP"
    if info["attack"] > info["magic"]:
        return "AD"
    return "mixed"


def main() -> None:
    url = f"https://ddragon.leagueoflegends.com/cdn/{DDRAGON_VERSION}/data/en_US/champion.json"
    data = requests.get(url, timeout=15).json()["data"]

    rows = []
    for c in data.values():
        tags, name = c["tags"], c["name"]
        rows.append({
            "championId": int(c["key"]),
            "name": name,
            "tags": "|".join(tags),
            "damage_type": damage_type(tags, c["info"]),
            "is_tank": int("Tank" in tags),
            "is_mage": int("Mage" in tags),
            "is_marksman": int("Marksman" in tags),
            "is_fighter": int("Fighter" in tags),
            "is_assassin": int("Assassin" in tags),
            "is_support": int("Support" in tags),
            "frontline": int("Tank" in tags or "Fighter" in tags),
            "engage": int(name in ENGAGE),
            "enchanter": int(name in ENCHANTER),
            "poke": int(name in POKE),
            "hard_cc": int(name in HARD_CC),
        })

    df = pd.DataFrame(rows).sort_values("championId").reset_index(drop=True)
    out = config.ROOT / "champion_labels.csv"
    df.to_csv(out, index=False)

    # Safety: flag any curated name that didn't match a real champion (typos).
    all_names = {c["name"] for c in data.values()}
    for label, names in [("ENGAGE", ENGAGE), ("ENCHANTER", ENCHANTER),
                         ("POKE", POKE), ("HARD_CC", HARD_CC)]:
        missing = names - all_names
        if missing:
            print(f"  WARNING: {label} names not found in Data Dragon: {sorted(missing)}")

    print(f"Wrote {out.name}: {len(df)} champions (DDragon {DDRAGON_VERSION}).")
    print("column sums (how many champs carry each flag):")
    flag_cols = ["is_tank", "is_mage", "is_marksman", "is_fighter", "is_assassin",
                 "is_support", "frontline", "engage", "enchanter", "poke", "hard_cc"]
    print(df[flag_cols].sum().to_string())
    print("\ndamage_type counts:", df["damage_type"].value_counts().to_dict())


if __name__ == "__main__":
    main()
