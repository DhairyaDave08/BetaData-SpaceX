# Data Sources

## Primary Dataset

- **Name:** All Space Missions from 1957 (`Space_Corrected.csv`)
- **Source:** Kaggle — https://www.kaggle.com/datasets/agirlcoding/all-space-missions-from-1957
- **License:** Refer to Kaggle dataset page (verify current license terms at source)
- **Rows:** 4,324 raw launches → 4,198 after cleaning (126 dropped for missing target/date)
- **Coverage:** 1957-10-04 to 2020-08-07
- **Columns used:** `Company Name`, `Location`, `Datum` (launch date), `Detail` (rocket + payload string), `Status Rocket`, ` Rocket` (cost, mostly missing), `Status Mission` (target)

## Weather Enrichment

- **Source:** Open-Meteo Historical Weather API — https://open-meteo.com/
- **Dataset used:** ERA5 reanalysis (archive endpoint), free, no API key required, global coverage from 1940 onward
- **License:** CC BY 4.0 — https://open-meteo.com/en/license
- **Scope:** Top 10 launch facilities by launch volume only (Baikonur Cosmodrome, Kennedy Space Center / Cape Canaveral, Guiana Space Centre, Plesetsk Cosmodrome, and others — see `SITE_COORDINATES` in `src/weather_enrichment.py` for the full list and coordinates used)
- **Fields retrieved:** daily max wind speed (km/h), daily max temperature (°C), daily precipitation (mm)
- **Actual coverage achieved:** 759 / 4,198 launches (18.1%) have real weather data. This is a deliberate scoping decision — fetching weather for all 126 raw launch site codes (many with only 1-2 historical launches) was not a worthwhile use of API calls or development time relative to the payoff. Launches outside the top 10 facilities are explicitly flagged `weather_available = False` with `NaN` weather fields, rather than being imputed or fabricated.
- **Fetch method:** decade-chunked requests per facility (not per-launch, not a single multi-decade request), cached to `data/raw/weather_cache/*.json` after first fetch — this avoids both oversized/timing-out API responses and unnecessary repeated calls on every notebook re-run.
- **Known gap:** Plesetsk Cosmodrome weather data has partial decade coverage due to Open-Meteo API rate limiting (HTTP 429) encountered during one fetch session; retry logic exists in `src/weather_enrichment.py` (`_fetch_decade_chunk`) but not all chunks were successfully backfilled given hackathon time constraints.

## Known Data Quality Issues

- **`Rocket` (cost, USD millions) is ~78% missing** (only 964 of 4,324 rows populated). Not used as a model feature; documented here rather than imputed, to avoid introducing fabricated signal into a small dataset.
- **No payload mass column exists** in this dataset (unlike some JSR/McDowell dataset variants). Addressed via a static, documented public-knowledge lookup table of approximate rocket family LEO payload capacities (see `ROCKET_PAYLOAD_CAPACITY_KG` in `src/feature_engineering.py`). Rocket families not found in the lookup receive a median fallback value and are explicitly flagged via `payload_capacity_known = False`, so this assumption is traceable rather than silently blended into "real" data.
- **`Status Mission` has 4 raw classes** (Success, Failure, Partial Failure, Prelaunch Failure), collapsed to a binary target for modeling. Partial Failures are treated as failures — a simplification worth noting, since a partial failure (e.g., incorrect orbit insertion) is not operationally identical to a total mission loss, but per-class sample sizes don't support finer-grained modeling reliably.
- **Rocket family naming is inconsistent** across eras — extracted via regex from the free-text `Detail` field (text before the first `|`, with block/version suffixes like "Block 5" stripped). 339 raw unique values were reduced to 92 groups (`min_count=10` threshold) to avoid overfitting a high-cardinality categorical on a ~4,200-row dataset — see `notebooks/02_feature_engineering.ipynb` for the explicit threshold comparison (10 / 15 / 20 / 25) that justified this choice.
- **Launch site strings are pad-level codes** (e.g., `"Site 41/1"`, `"SLC-40"`), not facility names — weather enrichment required matching these against the fuller `Location` string to correctly identify the parent facility (e.g., multiple Baikonur pad codes all map to one facility for weather-fetching purposes).
- **Older/Soviet-era entries (pre-1980) have less granular or differently formatted location strings**, which occasionally affects site-level grouping precision for very old launches.
- **Historical data is volume-skewed toward Soviet/Russian vehicles** — e.g., Cosmos-3M alone accounts for 422 of 4,198 launches (10%). Success-rate patterns learned from high-volume legacy vehicles may not transfer cleanly to modern commercial launch providers with different engineering and operational practices.
