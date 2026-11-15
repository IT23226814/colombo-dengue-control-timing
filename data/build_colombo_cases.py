"""Build the weekly Colombo District dengue case series from the official
Weekly Dengue Update reports of the National Dengue Control Unit (NDCU).

Each weekly report (PDF) has a district table "Reported dengue cases in week
w-1 and week w" with six numbers per district:

    prev-year week w-1, prev-year week w, this-year week w-1, this-year week w,
    prev-year cumulative to week w, this-year cumulative to week w

This script downloads every report listed in sources/weekly_report_urls.txt,
pulls out the Colombo row, and writes two files:

    colombo_report_rows.csv   one row per report, exactly as published
    colombo_weekly_cases.csv  tidy weekly series (year, week, dates, cases)

Revision rule: a week's count is taken from the *following* report's
"week w-1" column when that report exists, because it includes cases that
were reported late. Otherwise the count from the week's own report is used.

Usage (from the repository root):
    pip install pypdf pandas
    python data/build_colombo_cases.py
"""

from __future__ import annotations

import datetime as dt
import re
import urllib.request
from io import BytesIO
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
URLS = HERE / "sources" / "weekly_report_urls.txt"
CACHE = HERE / "raw"  # downloaded PDFs (git-ignored, not redistributed)
DISTRICT = "Colombo"
REPORT_YEAR = 2026

TOKEN = r"(Nil|\d+)\*?"
ROW = re.compile(rf"^{DISTRICT}\*?\s+" + r"\s+".join([TOKEN] * 6) + r"\s*$", re.M)
WEEK = re.compile(r"week-(\d+)", re.I)


def fetch(url: str) -> bytes:
    """Return the PDF bytes, using a local cache so reruns are offline."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / url.rsplit("/", 1)[-1]
    if not path.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
    return path.read_bytes()


def parse_row(pdf: bytes) -> list[int]:
    """Extract the six Colombo numbers from a report ("Nil" -> 0)."""
    text = "\n".join(p.extract_text() or "" for p in PdfReader(BytesIO(pdf)).pages)
    m = ROW.search(text)
    if m is None:
        raise ValueError(f"{DISTRICT} row not found")
    return [0 if g == "Nil" else int(g) for g in m.groups()]


def week_start(year: int, week: int) -> dt.date:
    """Monday of the ISO week (the reports run Monday to Sunday)."""
    return dt.date.fromisocalendar(year, week, 1)


def main() -> None:
    urls = [u.strip() for u in URLS.read_text().splitlines() if u.strip()]
    rows = []
    for url in urls:
        w = int(WEEK.search(url.rsplit("/", 1)[-1]).group(1))
        vals = parse_row(fetch(url))
        rows.append(dict(zip(
            ["report_week", "prev_year_wk_before", "prev_year_wk",
             "this_year_wk_before", "this_year_wk", "prev_year_cumulative",
             "this_year_cumulative"], [w, *vals])) | {"source_url": url})
    reports = pd.DataFrame(rows).sort_values("report_week").reset_index(drop=True)
    weeks = reports["report_week"].tolist()
    assert weeks == list(range(1, len(weeks) + 1)), f"missing report weeks: {weeks}"
    reports.to_csv(HERE / "colombo_report_rows.csv", index=False)

    by_week = reports.set_index("report_week")
    series = []
    for year, own_col, before_col in [
        (REPORT_YEAR, "this_year_wk", "this_year_wk_before"),
        (REPORT_YEAR - 1, "prev_year_wk", "prev_year_wk_before"),
    ]:
        for w in weeks:
            revised = (w + 1) in by_week.index
            cases = by_week.at[w + 1, before_col] if revised else by_week.at[w, own_col]
            series.append((year, w, cases, w + 1 if revised else w, revised))
    # Week 1 report also gives the last week of the previous year.
    last = dt.date(REPORT_YEAR - 1, 12, 28).isocalendar().week
    series.append((REPORT_YEAR - 1, last, by_week.at[1, "this_year_wk_before"], 1, True))

    df = pd.DataFrame(series, columns=["year", "week", "cases", "source_report_week", "revised"])
    df["week_start"] = [week_start(y, w) for y, w in zip(df.year, df.week)]
    df["week_end"] = df["week_start"] + dt.timedelta(days=6)
    df = df.sort_values(["year", "week"])[
        ["year", "week", "week_start", "week_end", "cases", "source_report_week", "revised"]]
    df.to_csv(HERE / "colombo_weekly_cases.csv", index=False)

    # Consistency checks against the published cumulative totals.
    last_w = weeks[-1]
    for year, col in [(REPORT_YEAR, "this_year_cumulative"), (REPORT_YEAR - 1, "prev_year_cumulative")]:
        total = df[(df.year == year) & (df.week <= last_w)].cases.sum()
        published = by_week.at[last_w, col]
        print(f"{year}: sum of weekly counts to week {last_w} = {total}, "
              f"published cumulative = {published} ({100 * (total - published) / published:+.2f}%)")
    print(f"Wrote {len(reports)} report rows and {len(df)} weekly counts.")


if __name__ == "__main__":
    main()
