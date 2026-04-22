"""American Soccer Analysis v1 API audit.

Confirms the three team-level endpoints we rely on for trend context:
    - /nwsl has analogous endpoints, but we target MLS: /mls/teams/{xgoals,goals-added,xpass}
    - Saves raw JSON per endpoint to data/raw/asa/
    - Prints a top-level schema of each response

Run:
    python -m pressured_progression.ingest.asa
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://app.americansocceranalysis.com/api/v1"
LEAGUE = "mls"
ENDPOINTS = [
    "teams/xgoals",
    "teams/goals-added",
    "teams/xpass",
]

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "asa"
TIMEOUT = 30


def fetch(endpoint: str) -> tuple[int, Any]:
    url = f"{BASE_URL}/{LEAGUE}/{endpoint}"
    logger.info("GET %s", url)
    r = requests.get(url, timeout=TIMEOUT)
    try:
        payload = r.json()
    except ValueError:
        payload = {"_raw_text": r.text[:500]}
    return r.status_code, payload


def describe(payload: Any) -> dict[str, Any]:
    """Return a compact structural description of a JSON payload."""
    if isinstance(payload, list):
        sample = payload[0] if payload else None
        return {
            "type": "list",
            "length": len(payload),
            "sample_keys": sorted(sample.keys()) if isinstance(sample, dict) else None,
            "sample_types": (
                {k: type(v).__name__ for k, v in sample.items()}
                if isinstance(sample, dict)
                else None
            ),
        }
    if isinstance(payload, dict):
        return {
            "type": "dict",
            "keys": sorted(payload.keys()),
            "types": {k: type(v).__name__ for k, v in payload.items()},
        }
    return {"type": type(payload).__name__, "value_preview": str(payload)[:200]}


def save_json(endpoint: str, payload: Any) -> Path:
    safe = endpoint.replace("/", "__")
    out = RAW_DIR / f"{LEAGUE}__{safe}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for ep in ENDPOINTS:
        try:
            status, payload = fetch(ep)
        except requests.RequestException as e:
            logger.error("Request failed for %s: %s", ep, e)
            results[ep] = {"status": "network_error", "error": str(e)}
            continue

        if status != 200:
            logger.warning("%s returned %s", ep, status)
            results[ep] = {"status": status, "schema": describe(payload)}
            continue

        path = save_json(ep, payload)
        schema = describe(payload)
        results[ep] = {"status": 200, "path": str(path), "schema": schema}
        print(f"\n--- {ep} ---")
        print(f"Saved: {path}")
        print(f"Schema: {json.dumps(schema, indent=2)}")

    # Summary index
    (RAW_DIR / "_index.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print("\n=== ASA audit summary ===")
    print(json.dumps({k: v.get("status") for k, v in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
