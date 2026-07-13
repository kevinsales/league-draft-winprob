"""Seed players -> match IDs -> match JSON -> local cache in data/raw/.

Run it (from anywhere):  python src/ingest.py

M1 pulls a *small* sample (a few Master seed players, ~50 unique matches) to
verify the pipeline before scaling. Raw match JSON is cached to data/raw/ keyed
by match id, so re-running re-fetches nothing already on disk.
"""

import json
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
    """GET with the API key header and 429/5xx backoff (honors Retry-After)."""
    headers = {"X-Riot-Token": config.api_key_or_die()}
    for attempt in range(5):
        resp = requests.get(url, headers=headers, params=params, timeout=15)
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
    return resp


def get_master_puuids(n_seeds: int) -> list[str]:
    """Return puuids of the top n_seeds Master ranked-solo players.

    Master is an apex tier (no divisions), so we use the dedicated masterleagues
    endpoint rather than entries/{tier}/{division}. Sort by LP for a stable set.
    """
    url = config.platform_host() + f"/lol/league/v4/masterleagues/by-queue/{config.RANKED_SOLO}"
    resp = get(url)
    resp.raise_for_status()
    entries = resp.json().get("entries", [])
    entries.sort(key=lambda e: e.get("leaguePoints", 0), reverse=True)
    puuids = [e["puuid"] for e in entries if e.get("puuid")]
    if not puuids:
        raise RuntimeError("Master league returned no puuids -- endpoint/migration may have changed.")
    print(f"Master league: {len(entries)} players; seeding with top {n_seeds} by LP.")
    return puuids[:n_seeds]


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


def ingest(n_seeds: int = 5, ids_per_seed: int = 20, target_matches: int = 50) -> None:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    # 1) Seed players -> their recent match ids, deduped across seeds (they share games).
    seeds = get_master_puuids(n_seeds)
    match_ids: list[str] = []
    seen: set[str] = set()
    for i, puuid in enumerate(seeds, 1):
        time.sleep(REQUEST_SPACING)
        ids = get_match_ids(puuid, ids_per_seed)
        new = [m for m in ids if m not in seen]
        seen.update(new)
        match_ids.extend(new)
        print(f"  seed {i}/{len(seeds)}: {len(ids)} ids ({len(new)} new) -> {len(match_ids)} unique total")

    match_ids = match_ids[:target_matches]
    print(f"\nFetching up to {len(match_ids)} unique matches...")

    # 2) Fetch match objects, caching to data/raw/. Skip anything already on disk.
    fetched = cached = 0
    for j, mid in enumerate(match_ids, 1):
        if fetch_match(mid):
            fetched += 1
            time.sleep(REQUEST_SPACING)  # only pace real network calls
        else:
            cached += 1
        if j % 10 == 0 or j == len(match_ids):
            print(f"  {j}/{len(match_ids)}  (fetched {fetched}, already cached {cached})")

    on_disk = len(list(config.DATA_RAW.glob("*.json")))
    print(f"\nDone. Newly fetched: {fetched} | already cached: {cached}")
    print(f"data/raw/ now holds {on_disk} unique cached matches.")


if __name__ == "__main__":
    ingest()
