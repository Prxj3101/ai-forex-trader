"""
News calendar loader (high-impact events) for the news filter.

Source: data/news/forex_factory_calendar.csv
(Hugging Face Ehsanrs2/Forex_Factory_Calendar, scraped from ForexFactory, 2007 - Apr 2025.
Personal research/backtesting only.)

PROBLEM WITH THE RAW TIMES (checked 1 Oct 2026):
- ~30-48% of high-impact rows have NO time (stored as 00:00 local), e.g. every NFP since 2019.
- Rows that do have a time are mostly right, but 2023+ summer rows are 1 hour early
  (Iran dropped DST, the scraper did not).
- The DATE is always right (checked: all NFP dates 2023-2025 correct).

FIX (priority order):
1. CSV time if the row has one and is before 2023 (checked vs schedule: ~all exact).
2. Official release schedule (SCHEDULE below, local time, DST via tz names) + CSV date.
3. CSV time for 2023+ rows with no schedule (speeches), with the 1h summer fix (may be off 1h).
Otherwise dropped (counted in the log). US elections: 18:00 NY on election day.

Output of load_events(): DataFrame [time (UTC), currency, event, source, duration_min]
  source = "schedule" | "csv" | "election"
  duration_min = how long the event lasts (0 for a data release, ~60 for speeches / press
  conferences, overnight for US elections). Prop firm speech windows run until the speech ENDS,
  and calendars only give the start, so we assume SPEECH_MINUTES.
"""
from __future__ import annotations

from datetime import time as dtime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "news" / "forex_factory_calendar.csv"

TZ = {"USD": "America/New_York", "EUR": "Europe/Berlin", "GBP": "Europe/London", "JPY": "Asia/Tokyo"}

# Which news currencies matter for each pair
PAIR_CCY = {
    "EURUSD": ("EUR", "USD"),
    "GBPUSD": ("GBP", "USD"),
    "USDJPY": ("USD", "JPY"),
    "USDCAD": ("USD",),
    "AUDUSD": ("USD",),
}

# Official local release times. Value = "HH:MM" or a function(date) -> "HH:MM".
def _ecb_decision(d):  # ECB moved the decision from 13:45 to 14:15 CET in July 2022
    return "14:15" if d >= pd.Timestamp("2022-07-21").date() else "13:45"


def _ecb_press(d):
    return "14:45" if d >= pd.Timestamp("2022-07-21").date() else "14:30"


SCHEDULE: dict[str, dict] = {
    "USD": {
        "_default": "08:30",  # almost all US data
        "ADP Non-Farm Employment Change": "08:15",
        "ISM Manufacturing PMI": "10:00", "ISM Services PMI": "10:00",
        "CB Consumer Confidence": "10:00", "JOLTS Job Openings": "10:00",
        "Prelim UoM Consumer Sentiment": "10:00", "Revised UoM Consumer Sentiment": "10:00",
        "Prelim UoM Inflation Expectations": "10:00",
        "Pending Home Sales m/m": "10:00", "New Home Sales": "10:00",
        "Flash Services PMI": "09:45", "Flash Manufacturing PMI": "09:45",
        "Final Manufacturing PMI": "09:45", "Chicago PMI": "09:45",
        "S&P/CS Composite-20 HPI y/y": "09:00",
        "Crude Oil Inventories": "10:30",
        "FOMC Statement": "14:00", "Federal Funds Rate": "14:00",
        "FOMC Economic Projections": "14:00", "FOMC Meeting Minutes": "14:00",
        "FOMC Press Conference": "14:30",
        "30-y Bond Auction": "13:00", "10-y Bond Auction": "13:00",
        # no fixed time -> csv time or drop
        "Fed Chair Powell Speaks": None, "Fed Chair Powell Testifies": None, "Fed Announcement": None,
        "Treasury Currency Report": None,
    },
    "EUR": {
        "_default": None,
        "French Flash Services PMI": "09:15", "French Flash Manufacturing PMI": "09:15",
        "German Flash Services PMI": "09:30", "German Flash Manufacturing PMI": "09:30",
        "Flash Manufacturing PMI": "10:00", "Flash Services PMI": "10:00",
        "Main Refinancing Rate": _ecb_decision, "Monetary Policy Statement": _ecb_decision,
        "ECB Press Conference": _ecb_press,
        "German Prelim CPI m/m": "14:00",
        "CPI Flash Estimate y/y": "11:00", "Core CPI Flash Estimate y/y": "11:00",
        "Spanish Flash CPI y/y": "09:00",
        "German ifo Business Climate": "10:00", "German ZEW Economic Sentiment": "11:00",
        "German Prelim GDP q/q": "08:00",
        "ECB Monetary Policy Meeting Accounts": "13:30",
    },
    "GBP": {
        "_default": "07:00",  # ONS data
        "Flash Services PMI": "09:30", "Flash Manufacturing PMI": "09:30", "Final Services PMI": "09:30",
        "Official Bank Rate": "12:00", "Monetary Policy Summary": "12:00",
        "MPC Official Bank Rate Votes": "12:00", "BOE Monetary Policy Report": "12:00",
        "MPC Asset Purchase Facility Votes": "12:00", "Asset Purchase Facility": "12:00",
        "Bank Stress Test Results": "07:00",
        "Annual Budget Release": None, "Autumn Forecast Statement": None,
        "Monetary Policy Report Hearings": None, "Parliament Brexit Vote": None,
        "Govt Confidence Vote": None,
    },
    "JPY": {
        "_default": None,  # BOJ statement has no fixed time
        "BOJ Press Conference": "15:30",
        "Tokyo Core CPI y/y": "08:30",
    },
}

US_ELECTIONS = ("Presidential Election", "Congressional Elections")
SPEECH_WORDS = ("Speaks", "Testifies", "Press Conference", "Hearings")
SPEECH_MINUTES = 60
ELECTION_MINUTES = 18 * 60


def duration_for(event: str) -> int:
    if any(e in event for e in US_ELECTIONS):
        return ELECTION_MINUTES
    if any(w in event for w in SPEECH_WORDS):
        return SPEECH_MINUTES
    return 0


def _sched(ccy: str, event: str, day):
    table = SCHEDULE.get(ccy, {})
    if event in table:
        v = table[event]
    elif any(w in event for w in ("Speaks", "Testifies", "Summit", "Election", "Meeting", "Vote", "Ruling", "Forecasts")):
        v = None
    else:
        v = table.get("_default")
    return v(day) if callable(v) else v


def _csv_time_utc(raw: pd.Series) -> pd.Series:
    """CSV time -> UTC, with the 2023+ summer 1h fix. NaT where the row has no time."""
    t = pd.to_datetime(raw, utc=True, format="ISO8601")
    no_time = raw.str[11:16].isin(["00:00", "23:59"])
    ny = t.dt.tz_convert("America/New_York")
    is_dst = pd.Series([bool(x.dst()) for x in ny], index=raw.index)
    fix = (t.dt.year >= 2023) & is_dst
    t = t + pd.to_timedelta(fix.astype(int), unit="h")
    return t.where(~no_time)


def load_events(path: Path = CSV, currencies=("USD", "EUR", "GBP", "JPY"), start="2015-01-01",
                log=print) -> pd.DataFrame:
    raw = pd.read_csv(path, usecols=["DateTime", "Currency", "Impact", "Event"])
    d = raw[raw.Impact.str.startswith("High") & raw.Currency.isin(currencies) & (raw.DateTime >= start)].copy()
    d["day"] = pd.to_datetime(d.DateTime.str[:10]).dt.date
    csv_utc = _csv_time_utc(d.DateTime)

    rows, dropped = [], 0
    for (i, r), cu in zip(d.iterrows(), csv_utc):
        ccy, ev, day = r.Currency, r.Event, r.day
        if ccy == "USD" and ev in US_ELECTIONS:
            # results come in overnight: block 18:00 NY on election day (+18h window set by caller)
            t = pd.Timestamp(f"{day} 18:00").tz_localize(TZ["USD"]).tz_convert("UTC")
            rows.append((t, ccy, ev, "election"))
            continue
        hm = _sched(ccy, ev, day)
        if pd.notna(cu) and cu.year < 2023:
            # pre-2023 CSV times checked against the schedule: ~all exact, and they also catch
            # real schedule changes (UK ONS data was 09:30 before ~2019, holiday-shifted releases)
            rows.append((cu, ccy, ev, "csv"))
        elif hm:
            if ccy == "GBP" and hm == "07:00" and day < pd.Timestamp("2019-01-01").date():
                hm = "09:30"
            t = pd.Timestamp(f"{day} {hm}").tz_localize(TZ[ccy]).tz_convert("UTC")
            rows.append((t, ccy, ev, "schedule"))
        elif pd.notna(cu):
            rows.append((cu, ccy, ev, "csv"))
        else:
            dropped += 1

    out = pd.DataFrame(rows, columns=["time", "currency", "event", "source"])
    out = merge_same_time(out)
    log(f"news: {len(out):,} high-impact event times {out.time.min():%Y-%m-%d} -> {out.time.max():%Y-%m-%d} "
        f"({dropped} rows without a known time dropped)")
    return out


def merge_same_time(out: pd.DataFrame) -> pd.DataFrame:
    """Several events at the same minute (NFP + Unemployment Rate + AHE) -> one row, names joined."""
    out = out.assign(duration_min=out.event.map(duration_for))
    return (out.groupby(["time", "currency"], as_index=False)
               .agg(event=("event", " + ".join), source=("source", "first"), duration_min=("duration_min", "max"))
               .sort_values("time").reset_index(drop=True))


def parse_ff_week(items: list[dict], currencies=("USD", "EUR", "GBP", "JPY"), impacts=("High",)) -> pd.DataFrame:
    """LIVE calendar: ForexFactory weekly JSON (nfs.faireconomy.media/ff_calendar_thisweek.json).
    Each item: {"title", "country", "date": "2026-10-02T08:30:00-04:00", "impact": "High", ...}.
    Times there are real timestamps with offset, so no schedule fix needed."""
    rows = [(pd.Timestamp(i["date"]).tz_convert("UTC"), i["country"], i["title"], "ff_live")
            for i in items if i.get("impact") in impacts and i.get("country") in currencies]
    out = pd.DataFrame(rows, columns=["time", "currency", "event", "source"])
    if out.empty:
        return out.assign(duration_min=pd.Series(dtype=int))
    return merge_same_time(out)


def events_for(symbol: str, events: pd.DataFrame) -> pd.DataFrame:
    return events[events.currency.isin(PAIR_CCY.get(symbol, ("USD",)))]


def check_schedule(path: Path = CSV) -> pd.DataFrame:
    """Validation: for rows that DO have a CSV time and a schedule time, minutes difference."""
    raw = pd.read_csv(path, usecols=["DateTime", "Currency", "Impact", "Event"])
    d = raw[raw.Impact.str.startswith("High") & raw.Currency.isin(TZ) & (raw.DateTime >= "2015")].copy()
    d["csv"] = _csv_time_utc(d.DateTime)
    d = d.dropna(subset=["csv"])
    out = []
    for _, r in d.iterrows():
        day = pd.Timestamp(r.DateTime[:10]).date()
        hm = _sched(r.Currency, r.Event, day)
        if hm and r.Event not in US_ELECTIONS:
            s = pd.Timestamp(f"{day} {hm}").tz_localize(TZ[r.Currency]).tz_convert("UTC")
            out.append((r.Currency, r.Event, r.csv.year, (r.csv - s).total_seconds() / 60))
    return pd.DataFrame(out, columns=["currency", "event", "year", "diff_min"])


if __name__ == "__main__":
    ev = load_events()
    print(ev.source.value_counts().to_string())
    c = check_schedule()
    print(f"schedule vs csv time: {len(c)} rows, |diff|<=5 min: {(c.diff_min.abs() <= 5).mean():.1%}")
    print(c[c.diff_min.abs() > 5].groupby(["currency", "event"]).diff_min.agg(["count", "median"]).to_string())
