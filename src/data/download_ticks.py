r"""Download historical bid/ask ticks in monthly files, independently of the bot.

Run from CMD: .venv\Scripts\python.exe src\data\download_ticks.py
Requires Node.js/npx. Completed months resume using a validated manifest.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SYMBOLS = ("eurusd", "gbpusd", "usdjpy")


def periods(start: date, end: date):
    while start < end:
        following = date(start.year + (start.month == 12), start.month % 12 + 1, 1)
        stop = min(following, end)
        yield start, stop
        start = stop


def validate(path: Path, start: date, end: date) -> dict:
    lower = int(datetime.combine(start, datetime.min.time(), timezone.utc).timestamp() * 1000)
    upper = int(datetime.combine(end, datetime.min.time(), timezone.utc).timestamp() * 1000)
    count, first, previous = 0, None, None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"timestamp", "bidPrice", "askPrice"}.issubset(reader.fieldnames or []):
            raise ValueError("Missing timestamp / bidPrice / askPrice columns")
        for row in reader:
            timestamp = int(row["timestamp"])
            bid, ask = float(row["bidPrice"]), float(row["askPrice"])
            if not (math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask):
                raise ValueError(f"Invalid bid/ask at row {count + 1}")
            if not lower <= timestamp <= upper or (previous is not None and timestamp < previous):
                raise ValueError(f"Invalid date/order at row {count + 1}")
            if first is None:
                first = timestamp
            previous = timestamp
            count += 1
    if not count:
        raise ValueError("Empty tick file")
    return {"rows": count, "first_ms": first, "last_ms": previous, "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2019, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 1), help="Exclusive end date")
    parser.add_argument("--symbols", nargs="+", choices=SYMBOLS, default=list(SYMBOLS))
    parser.add_argument("--plan", action="store_true", help="Show download plan without downloading")
    parser.add_argument("--batch-size", type=int, default=1, help="Concurrent hourly downloads (1-10)")
    parser.add_argument("--batch-pause", type=int, default=2000, help="Pause between batches in milliseconds")
    parser.add_argument("--cache", action="store_true", help="Cache hourly files so interrupted months can resume")
    parser.add_argument("--retries", type=int, default=3, help="Retries per failed hourly request (0-10)")
    parser.add_argument("--retry-pause", type=int, default=60000, help="Pause between failed-request retries in milliseconds")
    args = parser.parse_args()
    if args.start >= args.end:
        parser.error("start must be before end")
    if not 1 <= args.batch_size <= 10 or args.batch_pause < 0:
        parser.error("batch-size must be 1-10 and batch-pause must be non-negative")
    if not 0 <= args.retries <= 10 or args.retry_pause < 0:
        parser.error("retries must be 0-10 and retry-pause must be non-negative")
    tasks = [(symbol, start, end) for start, end in periods(args.start, args.end) for symbol in args.symbols]
    out = ROOT / "data" / "ticks" / "monthly"
    print(f"{len(tasks)} files: {args.start} to {args.end} (end exclusive). Output: {out}", flush=True)
    print(f"Batch size: {args.batch_size}, pause: {args.batch_pause} ms, hourly cache: {args.cache}", flush=True)
    print(f"Failed-request retries: {args.retries}, retry pause: {args.retry_pause} ms", flush=True)
    if args.plan:
        for symbol, start, end in tasks:
            print(f"{symbol}: {start} -> {end}")
        return
    npx = shutil.which("npx.cmd") or shutil.which("npx")
    if not npx:
        raise SystemExit("npx not found. Run this from the same CMD where your sample download worked.")
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "download_manifest.json"
    completed = json.loads(manifest.read_text()) if manifest.exists() else {}
    for index, (symbol, start, end) in enumerate(tasks, 1):
        filename = f"{symbol}-tick-{start}-{end}.csv"
        destination = out / filename
        saved = completed.get(filename)
        if saved and destination.exists() and destination.stat().st_size == saved.get("bytes"):
            print(f"[{index}/{len(tasks)}] SKIP validated {filename}", flush=True)
            continue
        staging = out / "pending" / filename.removesuffix(".csv")
        staging.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(tasks)}] DOWNLOAD {filename}", flush=True)
        command = [npx, "--yes", "dukascopy-node", "-i", symbol, "-from", str(start),
                   "-to", str(end), "-t", "tick", "-f", "csv", "-dir", str(staging),
                   "-bs", str(args.batch_size), "-bp", str(args.batch_pause),
                   "-r", str(args.retries), "-rp", str(args.retry_pause)]
        if args.cache:
            command.extend(["-ch", "-chpath", str(ROOT / "data" / "ticks" / "cache")])
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            raise SystemExit(f"Download failed ({result.returncode}). Completed months preserved. Rerun to resume.")
        source = staging / filename
        metadata = validate(source, start, end)
        os.replace(source, destination)
        completed[filename] = metadata
        temp = manifest.with_suffix(".tmp")
        temp.write_text(json.dumps(completed, indent=2))
        os.replace(temp, manifest)
        print(f"VALIDATED {metadata['rows']:,} ticks, {metadata['bytes'] / 1e6:.1f} MB", flush=True)
    print("All requested files downloaded and structurally validated. Tick coverage still needs backtest audit.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("Stopped. Completed months preserved; rerun the same command to resume.")
