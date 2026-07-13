"""Routing, patch pinning, paths, and API-key loading from .env.

All Riot API routing lives here so nothing else in the project hardcodes a host.
Riot uses two kinds of routing (these values change over time -- verify against
https://developer.riotgames.com/ if calls start 404-ing):

  - PLATFORM host  (e.g. "sg2"): region-specific. Used by league-v4 (ranked
    entries). SEA is split across several platforms, so we probe SEA_PLATFORMS
    to find the one holding an account's entries.
  - REGIONAL host  (e.g. "sea" / "asia"): broad cluster. Used by match-v5
    (matches) and account-v1 (Riot ID -> puuid).

Region is SEA only by design (see the build spec).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths (absolute, so scripts work from any working directory) ---
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

# --- Secrets (loaded from .env, which is gitignored and never committed) ---
load_dotenv(ROOT / ".env")
RIOT_API_KEY = os.getenv("RIOT_API_KEY", "").strip()
MY_RIOT_ID = os.getenv("MY_RIOT_ID", "").strip()  # "gameName#tagLine"

# --- Routing (SEA only) ---
# Platform hosts for league-v4. sg2 (Singapore) is the default; the setup check
# probes the rest until it finds the account's ranked entries, then we lock it.
PLATFORM = "sg2"
SEA_PLATFORMS = ["sg2", "ph2", "th2", "tw2", "vn2", "oc1"]

# Regional host for match-v5. SEA got its own "sea" route.
MATCH_REGION = "sea"

# Regional clusters for account-v1 (Riot ID -> puuid). account-v1 is effectively
# global, so we try the nearest first and fall back.
ACCOUNT_REGIONS = ["asia", "americas", "europe"]

# --- Queue & patch ---
RANKED_SOLO_QUEUE_ID = 420        # match-v5 queue filter
RANKED_SOLO = "RANKED_SOLO_5x5"   # league-v4 queue name
# Patch pinning: restrict to one or two recent patches for a coherent meta.
# The M1 sample was entirely 16.13; widen this list if a later pull spans patches.
TARGET_PATCHES: list[str] = ["16.13"]


def api_key_or_die() -> str:
    """Return the Riot API key, or raise a clear error if it's missing.

    A dev key expires every 24h, so a blank/stale key is the most common
    start-of-session failure. Fail loudly here instead of a cryptic 401 later.
    """
    if not RIOT_API_KEY:
        raise RuntimeError(
            "RIOT_API_KEY is empty in .env. Paste a fresh dev key "
            "(they expire every 24h): https://developer.riotgames.com/"
        )
    return RIOT_API_KEY


def platform_host(platform: str | None = None) -> str:
    return f"https://{platform or PLATFORM}.api.riotgames.com"


def regional_host(region: str) -> str:
    return f"https://{region}.api.riotgames.com"
