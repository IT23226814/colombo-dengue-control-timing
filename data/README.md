# Data

| File | Description | Source (with link) | Licence | Date accessed |
|---|---|---|---|---|
| `colombo_weekly_cases.csv` | Weekly suspected dengue cases in Colombo District: 2026 weeks 1–45, 2025 weeks 1–45 and 2025 week 52 | National Dengue Control Unit, Ministry of Health, Sri Lanka: *Weekly Dengue Update* reports, [dengue.health.gov.lk/weekly-report](https://www.dengue.health.gov.lk/weekly-report/) (data from NaDSys, the National Dengue Surveillance System) | Public government reports; no explicit licence. Aggregated counts reproduced with attribution | 2026-11-15 |
| `colombo_report_rows.csv` | The Colombo row from each weekly report as published (six numbers), with the report URL | Same as above | Same as above | 2026-11-15 |
| `colombo_population.csv` | Colombo District population, 2,375,415 | Department of Census and Statistics, *Census of Population and Housing 2024*, final report, [statistics.gov.lk](https://www.statistics.gov.lk/Blog/Blog_details) | Public government statistics | 2026-11-15 |
| `sources/weekly_report_urls.txt` | URLs of the 45 weekly report PDFs used | NDCU website | — | 2026-11-15 |
| `build_colombo_cases.py` | Script that downloads the reports and rebuilds both case CSVs | Own code | MIT | — |

## Columns of `colombo_weekly_cases.csv`

| Column | Meaning |
|---|---|
| `year`, `week` | Reporting week; weeks run Monday to Sunday and match ISO 8601 week numbers (e.g. 2026 week 45 = 2–8 November 2026, as printed on the report) |
| `week_start`, `week_end` | First (Monday) and last (Sunday) day of the week |
| `cases` | Suspected dengue cases reported in Colombo District that week (includes the Colombo Municipal Council area) |
| `source_report_week` | Which 2026 weekly report the number was taken from |
| `revised` | `True` if taken from the following week's report (see below) |

## How the series was built

Each weekly report has a district table with six numbers: the previous year's counts for weeks *w*−1 and *w*, this year's counts for weeks *w*−1 and *w*, and both years' cumulative totals up to week *w*. The figure for week *w*−1 is printed again in the next report, often slightly changed because late notifications have been added and duplicates removed. A week's count is therefore taken from the **following** report when there is one (`revised = True`), and otherwise from its own report (only week 45 of each year).

The 2025 counts come from the "previous year" columns of the 2026 reports. Week 52 of 2025 comes from the "week *w*−1" column of the 2026 week 1 report. 2025 weeks 46–51 are not covered by these reports and are missing.

To rebuild the files (the PDFs are downloaded into `data/raw/`, which is not committed):

```bash
pip install pypdf pandas
python data/build_colombo_cases.py
```

## Quality checks

- The parsed numbers were checked against a second, independent PDF text extractor (macOS PDFKit): all 45 report rows agree.
- Sum of weekly counts compared with each report's published cumulative total at week 45: **2026: 25,511 vs 25,229 (+1.1%)**, **2025: 10,934 vs 10,862 (+0.7%)**. The reports note that week totals "may vary slightly" because duplicate entries are deleted and patients are transferred between districts, so cumulative totals are revised downward over time. The weekly series is used as is; this ~1% difference is small next to the reporting uncertainty handled by the model's reporting rate.

## Caveats

- These are **suspected** (notified) cases, not laboratory-confirmed infections; many infections are mild or asymptomatic and never reported. The model handles this with a reporting rate parameter.
- Late reporting means the most recent week (week 45) is likely to be revised upward.
- Counts are aggregates only; no personal data is included.
