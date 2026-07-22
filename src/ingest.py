"""Seed players -> match IDs -> match JSON -> local cache in data/raw/.

Run it (from anywhere):  python src/ingest.py

Seeds from the apex tiers (Master / Grandmaster / Challenger) on the SEA server
and caches raw match JSON to data/raw/, keyed by match id, so re-running
re-fetches nothing already on disk -- a long pull can be interrupted and resumed
for free. Each match is also tagged with the tier of the seed player that
surfaced it (data/processed/seed_tiers.csv).
"""

import json
import random
import sys
import time
from pathlib import Path

import requests

# Make `import config` work no matter the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

# Polite pacing between calls. Dev keys allow ~20 req/s but only 100 req / 2 min,
# so ~1.2s/request keeps us comfortably inside the 2-minute window; 429s (if any)
# are still handled by backoff below.
REQUEST_SPACING = 1.2


def get(url: str, params: dict | None = None) -> requests.Response:
    """GET with the API key header, retrying on 429/5xx and transient network
    errors (a long bulk pull will hit the occasional dropped connection)."""
    headers = {"X-Riot-Token": config.api_key_or_die()}
    resp = None
    for attempt in range(6):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=15)
        except requests.exceptions.RequestException as e:
            wait = 2 * (attempt + 1)  # linear backoff on connection reset/timeout
            print(f"  ...network error ({type(e).__name__}), waiting {wait}s (attempt {attempt + 1})")
            time.sleep(wait)
            continue
        if resp.status_code in (401, 403):
            raise RuntimeError(
                f"Key rejected ({resp.status_code}). Dev keys expire every 24h -- "
                "paste a fresh one into .env: https://developer.riotgames.com/"
            )
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = int(resp.headers.get("Retry-After", 2))
            print(f"  ...{resp.status_code}, waiting {wait}s (attempt {attempt + 1})")
            time.sleep(wait)
            continue
        return resp
    if resp is None:
        raise RuntimeError(f"Gave up after repeated network errors: {url}")
    return resp


# Apex tiers have no divisions, so each gets a dedicated league-v4 endpoint.
# Master-heavy by design: that is the account's own rank band.
APEX_ENDPOINTS = {
    "MASTER": "masterleagues",
    "GRANDMASTER": "grandmasterleagues",
    "CHALLENGER": "challengerleagues",
}
DEFAULT_SEEDS_PER_TIER = {"MASTER": 600, "GRANDMASTER": 400, "CHALLENGER": 300}


def get_apex_puuids(seeds_per_tier: dict[str, int]) -> list[tuple[str, str]]:
    """Return [(puuid, tier)] across Master/Grandmaster/Challenger, top-LP first.

    The tier is the *seed player's* rank, which we carry through as an
    approximate label for the games they surface (see the note in ingest()).
    """
    seeds: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tier, endpoint in APEX_ENDPOINTS.items():
        n = seeds_per_tier.get(tier, 0)
        if n <= 0:
            continue
        url = config.platform_host() + f"/lol/league/v4/{endpoint}/by-queue/{config.RANKED_SOLO}"
        resp = get(url)
        resp.raise_for_status()
        entries = resp.json().get("entries", [])
        entries.sort(key=lambda e: e.get("leaguePoints", 0), reverse=True)
        picked = [e["puuid"] for e in entries if e.get("puuid") and e["puuid"] not in seen]
        picked = picked[:n]
        seen.update(picked)
        seeds.extend((p, tier) for p in picked)
        print(f"  {tier}: {len(entries)} players in league, seeding {len(picked)}")
        time.sleep(REQUEST_SPACING)
    if not seeds:
        raise RuntimeError("No apex puuids returned -- endpoint/migration may have changed.")
    return seeds


def get_match_ids(puuid: str, count: int) -> list[str]:
    """Recent ranked-solo (queue 420) match ids for one puuid, via match-v5 (sea)."""
    url = config.regional_host(config.MATCH_REGION) + f"/lol/match/v5/matches/by-puuid/{puuid}/ids"
    resp = get(url, params={"queue": config.RANKED_SOLO_QUEUE_ID, "start": 0, "count": count})
    resp.raise_for_status()
    return resp.json()


def fetch_match(match_id: str) -> bool:
    """Fetch one match and cache it to data/raw/{match_id}.json.

    Returns True if it hit the network, False if it was already cached (so the
    caller can prove re-runs do zero re-fetches).
    """
    path = config.DATA_RAW / f"{match_id}.json"
    if path.exists():
        return False
    url = config.regional_host(config.MATCH_REGION) + f"/lol/match/v5/matches/{match_id}"
    resp = get(url)
    resp.raise_for_status()
    path.write_text(json.dumps(resp.json()), encoding="utf-8")
    return True


def save_seed_tiers(match_tier: dict[str, str]) -> None:
    """Persist matchId -> seed tier, merging with anything already recorded."""
    path = config.DATA_PROCESSED / "seed_tiers.csv"
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    merged = dict(match_tier)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            if "," in line:
                mid, tier = line.split(",", 1)
                merged.setdefault(mid, tier)
    lines = ["matchId,seed_tier"] + [f"{m},{t}" for m, t in sorted(merged.items())]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  seed-tier manifest: {len(merged)} matches -> {path.name}")


def ingest(seeds_per_tier: dict[str, int] | None = None, ids_per_seed: int = 60,
           target_matches: int = 12000, shuffle_seed: int = 42) -> None:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    seeds_per_tier = seeds_per_tier or DEFAULT_SEEDS_PER_TIER

    # 1) Seed players -> their recent match ids, deduped across seeds (they share
    #    games). Each match is tagged with the tier of the FIRST seed that
    #    surfaced it. NOTE: match-v5 has no "game tier" field, so this is an
    #    approximation -- an apex game contains a spread of nearby ranks.
    print("Seeding from apex tiers:")
    seeds = get_apex_puuids(seeds_per_tier)
    match_tier: dict[str, str] = {}
    for i, (puuid, tier) in enumerate(seeds, 1):
        time.sleep(REQUEST_SPACING)
        for m in get_match_ids(puuid, ids_per_seed):
            match_tier.setdefault(m, tier)
        if i % 50 == 0 or i == len(seeds):
            print(f"  seed {i}/{len(seeds)} ({tier}) -> {len(match_tier)} unique ids so far")

    # Shuffle before slicing: the seed list is ordered by tier, so taking the
    # first N unsliced would return only Challenger games.
    match_ids = list(match_tier)
    random.Random(shuffle_seed).shuffle(match_ids)
    match_ids = match_ids[:target_matches]
    save_seed_tiers({m: match_tier[m] for m in match_ids})
    print(f"\nFetching up to {len(match_ids)} unique matches...")

    # 2) Fetch match objects, caching to data/raw/. Skip anything already on disk.
    fetched = cached = 0
    for j, mid in enumerate(match_ids, 1):
        if fetch_match(mid):
            fetched += 1
            time.sleep(REQUEST_SPACING)  # only pace real network calls
        else:
            cached += 1
        if j % 200 == 0 or j == len(match_ids):
            print(f"  {j}/{len(match_ids)}  (fetched {fetched}, already cached {cached})")

    on_disk = len(list(config.DATA_RAW.glob("*.json")))
    print(f"\nDone. Newly fetched: {fetched} | already cached: {cached}")
    print(f"data/raw/ now holds {on_disk} unique cached matches.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Pull & cache apex ranked-solo matches.")
    p.add_argument("--master", type=int, default=600, help="Master seed players")
    p.add_argument("--grandmaster", type=int, default=400, help="Grandmaster seed players")
    p.add_argument("--challenger", type=int, default=300, help="Challenger seed players")
    p.add_argument("--ids-per-seed", type=int, default=60, help="recent match ids per seed")
    p.add_argument("--target", type=int, default=12000, help="max unique matches to fetch")
    a = p.parse_args()
    ingest(seeds_per_tier={"MASTER": a.master, "GRANDMASTER": a.grandmaster,
                           "CHALLENGER": a.challenger},
           ids_per_seed=a.ids_per_seed, target_matches=a.target)
