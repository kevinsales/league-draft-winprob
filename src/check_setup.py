"""M0 sanity check: resolve my Riot ID -> puuid -> current ranked-solo rank.

Run it (from anywhere):  python src/check_setup.py

Its job is only to prove the key + routing work end to end and to print the
tier/division we'll train on. It also reports which SEA platform host actually
held the ranked entries, so we can lock PLATFORM in config.py.
"""

import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

# Make `import config` work no matter the current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def get(url: str) -> requests.Response:
    """GET with the API key header and a simple 429/5xx backoff.

    Returns the final Response (caller inspects status_code). Retries a few
    times on rate-limit (429) or transient server errors, honoring Retry-After.
    """
    headers = {"X-Riot-Token": config.api_key_or_die()}
    for attempt in range(4):
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = int(resp.headers.get("Retry-After", 2))
            print(f"  ...{resp.status_code}, waiting {wait}s (attempt {attempt + 1})")
            time.sleep(wait)
            continue
        return resp
    return resp  # last response after exhausting retries


def die(msg: str) -> None:
    print(f"\nERROR: {msg}")
    sys.exit(1)


def resolve_puuid(game_name: str, tag_line: str) -> str:
    """Riot ID -> puuid via account-v1, trying regional clusters in order."""
    path = f"/riot/account/v1/accounts/by-riot-id/{quote(game_name)}/{quote(tag_line)}"
    for region in config.ACCOUNT_REGIONS:
        url = config.regional_host(region) + path
        resp = get(url)
        if resp.status_code == 200:
            print(f"account-v1: found via '{region}' cluster")
            return resp.json()["puuid"]
        if resp.status_code in (401, 403):
            die(
                f"key rejected ({resp.status_code}). Dev keys expire every 24h -- "
                "paste a fresh one into .env: https://developer.riotgames.com/"
            )
        if resp.status_code != 404:
            print(f"  account-v1 on '{region}': unexpected {resp.status_code} {resp.text[:120]}")
    die(f"Riot ID '{game_name}#{tag_line}' not found on any account cluster.")


def find_ranked_entry(puuid: str):
    """league-v4 entries/by-puuid, probing SEA platforms until one has entries."""
    for platform in config.SEA_PLATFORMS:
        url = config.platform_host(platform) + f"/lol/league/v4/entries/by-puuid/{puuid}"
        resp = get(url)
        if resp.status_code == 200:
            entries = resp.json()
            if entries:  # empty list = account exists on this platform but no ranked data
                return platform, entries
        elif resp.status_code in (401, 403):
            die(f"key rejected ({resp.status_code}) on league-v4. Regenerate the dev key.")
    return None, []


def main() -> None:
    if "#" not in config.MY_RIOT_ID:
        die(f"MY_RIOT_ID should look like 'gameName#tagLine', got: {config.MY_RIOT_ID!r}")
    game_name, tag_line = config.MY_RIOT_ID.split("#", 1)
    print(f"Looking up {game_name}#{tag_line} on SEA...\n")

    puuid = resolve_puuid(game_name, tag_line)
    print(f"puuid: {puuid[:16]}... (len {len(puuid)})\n")

    platform, entries = find_ranked_entry(puuid)
    if not entries:
        die("no ranked entries found on any SEA platform (account may be unranked).")
    print(f"league-v4: entries found on platform '{platform}'")

    solo = next((e for e in entries if e.get("queueType") == config.RANKED_SOLO), None)
    if solo is None:
        queues = [e.get("queueType") for e in entries]
        die(f"no {config.RANKED_SOLO} entry (has: {queues}). Play ranked solo, or adjust the queue.")

    tier, division = solo["tier"].title(), solo["rank"]
    wins, losses = solo["wins"], solo["losses"]
    total = wins + losses
    winrate = f"{wins / total:.0%}" if total else "n/a"
    print("\n=== YOUR CURRENT RANK (Ranked Solo/Duo) ===")
    print(f"  {tier} {division}  -  {solo['leaguePoints']} LP")
    print(f"  {wins}W / {losses}L  ({winrate} over {total} games)")
    print(f"\nLock these into config.py: PLATFORM = \"{platform}\"")


if __name__ == "__main__":
    main()
