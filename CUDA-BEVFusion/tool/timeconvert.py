#!/usr/bin/env python3
"""
Convert timestamps of the form "YYYY-MM-DD H:MM:SS.sss AM/PM PDT" to Unix time in seconds.

Usage:
    python3 timestamp_to_unix.py "2025-04-07 4:52:16.773 PM PDT"

Or, to convert multiple lines from a file:
    cat timestamps.txt | python3 timestamp_to_unix.py
"""
import sys
from datetime import datetime
import pytz

def parse_timestamp(ts_str: str) -> float:
    # Expect format: '2025-04-07 4:52:16.773 PM PDT'
    # Strip timezone abbreviation and parse offset-aware
    # PDT is UTC-7
    ts_str = ts_str.strip()
    parts = ts_str.rsplit(' ', 2)
    if len(parts) != 3:
        raise ValueError(f"Invalid format: '{ts_str}'")
    datetime_part, ampm, tzabbrev = parts
    # Reassemble full datetime string
    fmt_str = '%Y-%m-%d %I:%M:%S.%f %p'
    # Parse without timezone
    dt_naive = datetime.strptime(f"{datetime_part} {ampm}", fmt_str)
    # Attach PDT timezone (UTC-7)
    tz = pytz.timezone('America/Los_Angeles')
    dt_local = tz.localize(dt_naive)
    # Convert to UTC
    dt_utc = dt_local.astimezone(pytz.utc)
    # Return Unix timestamp with fractional seconds
    return dt_utc.timestamp()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Single argument
        try:
            ts = sys.argv[1]
            print(parse_timestamp(ts))
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Read from stdin
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                print(parse_timestamp(line))
            except Exception as e:
                print(f"Error processing '{line}': {e}", file=sys.stderr)
                continue
