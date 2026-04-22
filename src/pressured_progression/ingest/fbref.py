"""FBref team-season audit.

Pulls one MLS team-season page and one Premier League team-season page under a
3 requests/minute rate limit with exponential backoff on failures. Saves HTML
and extracted tables under data/raw/fbref/. Logs which stat columns appear on
both pages and which don't — we need that overlap to populate the 8-dim style
vector for MLS ↔ European analog matching.

Run:
    python -m pressured_progression.ingest.fbref
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections import deque
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "fbref"
USER_AGENT = (
    "pressured-progression/0.1 (+research; contact via project repo) "
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
)
TIMEOUT = 30
MAX_RETRIES = 5

# Canonical team-season pages. Update squad IDs here if FBref rewrites them.
TARGETS: list[dict[str, str]] = [
    {
        "label": "mls__philadelphia_union_2023",
        "url": "https://fbref.com/en/squads/1ebc1d66/2023/Philadelphia-Union-Stats",
    },
    {
        "label": "epl__brighton_2023_2024",
        "url": "https://fbref.com/en/squads/d07537b9/2023-2024/Brighton-and-Hove-Albion-Stats",
    },
]


@dataclass
class RateLimiter:
    """Sliding-window limiter: at most `max_calls` calls per `window_s` seconds."""

    max_calls: int = 3
    window_s: float = 60.0
    _calls: deque[float] = field(default_factory=deque)

    def wait(self) -> None:
        now = time.monotonic()
        while self._calls and now - self._calls[0] > self.window_s:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            sleep_s = self.window_s - (now - self._calls[0]) + 0.25
            logger.info("Rate limit reached; sleeping %.1fs", sleep_s)
            time.sleep(max(sleep_s, 0))
            return self.wait()
        self._calls.append(time.monotonic())


def fetch(url: str, limiter: RateLimiter) -> str | None:
    """Fetch a URL respecting rate limits with exponential backoff."""
    headers = {"User-Agent": USER_AGENT}
    delay = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        limiter.wait()
        try:
            r = requests.get(url, headers=headers, timeout=TIMEOUT)
        except requests.RequestException as e:
            logger.warning("attempt %d: request error for %s: %s", attempt, url, e)
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
            continue

        if r.status_code == 200:
            return r.text
        if r.status_code in (429, 503):
            logger.warning("attempt %d: %s %s; backing off", attempt, r.status_code, url)
            time.sleep(delay + random.uniform(0, 1))
            delay *= 2
            continue
        logger.error("attempt %d: %s %s; not retrying", attempt, r.status_code, url)
        return None
    logger.error("Exhausted %d retries for %s", MAX_RETRIES, url)
    return None


def extract_tables(html: str) -> dict[str, pd.DataFrame]:
    """Return tables keyed by table id. FBref hides many tables inside HTML comments."""
    soup = BeautifulSoup(html, "lxml")
    raw_html_chunks: list[str] = [str(soup)]
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        if "<table" in c:
            raw_html_chunks.append(str(c))

    tables: dict[str, pd.DataFrame] = {}
    for chunk in raw_html_chunks:
        sub = BeautifulSoup(chunk, "lxml")
        for tbl in sub.find_all("table"):
            tid = tbl.get("id")
            if not tid or tid in tables:
                continue
            try:
                df_list = pd.read_html(StringIO(str(tbl)))
            except ValueError:
                continue
            if not df_list:
                continue
            df = df_list[0]
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    "_".join(str(p) for p in tup if str(p) and not str(p).startswith("Unnamed"))
                    for tup in df.columns
                ]
            tables[tid] = df
    return tables


def safe_label(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def save_artifacts(label: str, html: str, tables: dict[str, pd.DataFrame]) -> Path:
    outdir = RAW_DIR / label
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "page.html").write_text(html, encoding="utf-8")
    for tid, df in tables.items():
        df.to_csv(outdir / f"{safe_label(tid)}.csv", index=False)
    return outdir


def flatten_columns(tables: dict[str, pd.DataFrame]) -> set[str]:
    """Union of (table_id::column) identifiers across all tables on a page."""
    out: set[str] = set()
    for tid, df in tables.items():
        for col in df.columns:
            out.add(f"{tid}::{col}")
    return out


def log_overlap(a_label: str, a_cols: set[str], b_label: str, b_cols: set[str]) -> None:
    both = sorted(a_cols & b_cols)
    only_a = sorted(a_cols - b_cols)
    only_b = sorted(b_cols - a_cols)
    logger.info("=== Column overlap ===")
    logger.info("On BOTH (%d):", len(both))
    for c in both:
        logger.info("  %s", c)
    logger.info("Only on %s (%d):", a_label, len(only_a))
    for c in only_a[:40]:
        logger.info("  %s", c)
    if len(only_a) > 40:
        logger.info("  ... +%d more", len(only_a) - 40)
    logger.info("Only on %s (%d):", b_label, len(only_b))
    for c in only_b[:40]:
        logger.info("  %s", c)
    if len(only_b) > 40:
        logger.info("  ... +%d more", len(only_b) - 40)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter()

    per_page_cols: dict[str, set[str]] = {}
    for target in TARGETS:
        label, url = target["label"], target["url"]
        html = fetch(url, limiter)
        if html is None:
            logger.error("Skipping %s — fetch failed", label)
            per_page_cols[label] = set()
            continue
        tables = extract_tables(html)
        outdir = save_artifacts(label, html, tables)
        logger.info("%s: %d tables saved under %s", label, len(tables), outdir)
        per_page_cols[label] = flatten_columns(tables)

    labels = list(per_page_cols.keys())
    if len(labels) == 2 and all(per_page_cols.values()):
        a, b = labels
        log_overlap(a, per_page_cols[a], b, per_page_cols[b])
    else:
        logger.warning("Overlap report skipped — one or more pages returned no tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
