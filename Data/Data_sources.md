# Data Sources

## Primary Dataset
- **Name:** All Space Missions from 1957 (Space_Corrected.csv)
- **Source:** Kaggle — https://www.kaggle.com/datasets/agirlcoding/all-space-missions-from-1957
- **License:** Refer to Kaggle dataset page (CC0/public domain, verify at source)
- **Rows:** 4,324 launches, 1957–2020
- **Columns used:** Company Name, Location, Datum (date), Detail (rocket + payload),
  Status Rocket, Rocket (cost, ~22% non-null), Status Mission (target)

## Known Data Quality Issues
- `Rocket` (cost in USD millions) is only ~22% populated — mostly missing for
  non-commercial/government launches. Not used as a primary feature; documented as a
  limitation rather than imputed, to avoid introducing fabricated signal.
- No payload mass column exists in this dataset (unlike some JSR/McDowell variants).
  Addressed via a static public-knowledge lookup table of approximate rocket family
  payload capacities — documented explicitly in `src/feature_engineering.py`.
- `Status Mission` has 4 classes (Success, Failure, Partial Failure, Prelaunch Failure),
  collapsed to binary for modeling. Partial Failures are treated as failures — a
  simplification worth noting, since a partial failure (e.g., wrong orbit) is not
  operationally identical to a total loss.
- Older/Soviet-era entries (pre-1980) have less granular location strings, which
  can affect site-level grouping.
- No weather data included in the base dataset; optional enrichment via NOAA
  historical archives, documented separately if implemented.
