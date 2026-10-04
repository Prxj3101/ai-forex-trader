r"""
LIVE news calendar for the bot (this week's high-impact events).

Source: ForexFactory weekly JSON (free, no API key):
    https://nfs.faireconomy.media/ff_calendar_thisweek.json
They rate-limit, so we download at most once per hour and keep a copy in
data/news/ff_thisweek.json (bot keeps working from the copy if the internet is down).

Note: Funding Pips uses ITS OWN calendar (FP dashboard). Times are almost always the same as
ForexFactory, which is why the prop profile keeps a few extra minutes of margin.

Run (to check what the bot sees this week):
    python src\data\news_live.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features import news  # noqa: E402

URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE = ROOT / "data" / "news" / "ff_thisweek.json"
MAX_AGE_S = 3600


def fetch(force: bool = False, log=print) -> list[dict]:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    fresh = CACHE.exists() and time.time() - CACHE.stat().st_mtime < MAX_AGE_S
    if fresh and not force:
        return json.loads(CACHE.read_text())
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "ai-forex-trader/1.0"})
        with urllib.request.urlopen(req, timeout=20, context=_ssl_context()) as r:
            items = json.loads(r.read().decode())
        CACHE.write_text(json.dumps(items))
        log(f"news calendar downloaded: {len(items)} events this week")
        return items
    except Exception as e:  # network down / rate limited -> use the old copy
        if CACHE.exists():
            log(f"news calendar download failed ({e}); using saved copy")
            return json.loads(CACHE.read_text())
        # koi copy nahi: bot band mat karo, khaali calendar se chalo, har ghante dobara try hoga
        log(f"WARNING: news calendar download failed ({e}) and no saved copy -> NO news block until next hourly retry")
        return []


def _ssl_context():
    """Naye Windows server pe root certificates poore nahi hote -> certifi ki list use karo."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def live_events(force: bool = False, log=print):
    return news.parse_ff_week(fetch(force, log))


if __name__ == "__main__":
    import pandas as pd

    from src import config
    from src.risk.news_rules import NewsGuard

    ev = live_events(force=True)
    ny = ev.assign(ny=ev.time.dt.tz_convert("America/New_York").dt.strftime("%a %d %b %H:%M NY"))
    print(ny[["ny", "currency", "event", "duration_min"]].to_string(index=False))
    g = NewsGuard(ev, config.profile())
    now = pd.Timestamp.now(tz="UTC")
    print(f"\nprofile: {g.p['name']} (news rules {'ON' if g.on else 'OFF'})")
    for s in config.CORE_SYMBOLS:
        w = g.windows(s)
        w = w[w.end >= now]
        nxt = "none" if w.empty else (f"{w.start.iloc[0].tz_convert('America/New_York'):%a %H:%M} - "
                                      f"{w.end.iloc[0].tz_convert('America/New_York'):%H:%M} NY  {w.event.iloc[0]}")
        print(f"{s}: blocked now = {g.blocked(s, now) or 'no'} | next window: {nxt}")
