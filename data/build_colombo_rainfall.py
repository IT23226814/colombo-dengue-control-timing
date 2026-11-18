"""Build the weekly rainfall series for Colombo District from the Open-Meteo
historical weather API (ERA5-based reanalysis, CC BY 4.0).

Daily precipitation (mm) is downloaded for three points spread across the
district (west coast, centre, east), averaged over the points, and summed over
the same Monday-to-Sunday weeks as the case data (ISO 8601 weeks).

The series starts in ISO week 40 of 2025 so that rainfall in the weeks before
2026 is available to the model's lagged mosquito-habitat term.

Output: colombo_weekly_rainfall.csv with columns
    year, week, week_start, week_end, rain_mm

Usage (from the repository root):
    pip install pandas
    python data/build_colombo_rainfall.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "raw"  # downloaded JSON (git-ignored)
API = "https://archive-api.open-meteo.com/v1/archive"
POINTS = {  # name: (latitude, longitude)
    "Colombo (west coast)": (6.93, 79.86),
    "Homagama (centre)": (6.84, 80.00),
    "Avissawella (east)": (6.95, 80.21),
}
START, END = "2025-09-29", "2026-11-08"  # Monday of 2025 week 40 to Sunday of 2026 week 45


def fetch(lat: float, lon: float) -> pd.Series:
    url = (f"{API}?latitude={lat}&longitude={lon}&start_date={START}&end_date={END}"
           "&daily=precipitation_sum&timezone=Asia%2FColombo")
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"open_meteo_{lat}_{lon}_{START}_{END}.json"
    if not cached.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            cached.write_bytes(r.read())
    daily = json.loads(cached.read_text())["daily"]
    return pd.Series(daily["precipitation_sum"], index=pd.to_datetime(daily["time"]), dtype=float)


def main() -> None:
    daily = pd.concat({name: fetch(*ll) for name, ll in POINTS.items()}, axis=1)
    assert daily.notna().all().all(), "missing daily values"
    mean = daily.mean(axis=1)
    weekly = mean.resample("W-SUN").sum()                      # weeks ending on Sunday
    iso = weekly.index.isocalendar()
    out = pd.DataFrame({
        "year": iso.year.values,
        "week": iso.week.values,
        "week_start": (weekly.index - pd.Timedelta(days=6)).date,
        "week_end": weekly.index.date,
        "rain_mm": weekly.round(1).values,
    })
    assert (pd.to_datetime(out.week_start).dt.day_of_week == 0).all()
    out.to_csv(HERE / "colombo_weekly_rainfall.csv", index=False)
    print(f"{len(out)} weeks, {out.week_start.iloc[0]} to {out.week_end.iloc[-1]}; "
          f"total {out.rain_mm.sum():,.0f} mm")


if __name__ == "__main__":
    main()
