import pandas as pd
import numpy as np
import requests
import json
import os
import time

.
SITE_COORDINATES = {
    "Cape Canaveral": (28.5721, -80.6480),
    "Kennedy Space Center": (28.5721, -80.6480),
    "Baikonur Cosmodrome": (45.9650, 63.3050),
    "Plesetsk Cosmodrome": (62.9250, 40.5770),
    "Vandenberg": (34.7420, -120.5720),
    "Guiana Space Centre": (5.2390, -52.7680),
    "Kourou": (5.2390, -52.7680),
    "Xichang": (28.2460, 102.0260),
    "Jiuquan": (40.9580, 100.2910),
    "Taiyuan": (38.8490, 111.6080),
    "Tanegashima": (30.4000, 130.9700),
    "Satish Dhawan": (13.7330, 80.2350),
    "Sriharikota": (13.7330, 80.2350),
    "Wallops": (37.9400, -75.4660),
}

CACHE_DIR = "data/raw/weather_cache"
DAILY_VARS = "windspeed_10m_max,temperature_2m_max,precipitation_sum"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def match_known_site(location_str: str):
    """Returns (site_key, lat, lon) if the location string matches a known site, else None."""
    if pd.isna(location_str):
        return None
    location_str = str(location_str).lower()
    for key, (lat, lon) in SITE_COORDINATES.items():
        if key.lower() in location_str:
            return key, lat, lon
    return None


def get_top_launch_sites(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Identifies which of the dataset's top launch sites (by launch count)
    we have coordinates for. Looks a bit wider than top_n since not every
    high-volume site will have a coordinate match.
    """
    site_counts = df["launch_site"].value_counts().head(top_n * 3)
    matched = []
    for site_name, count in site_counts.items():
        match = match_known_site(site_name)
        if match:
            matched.append({
                "raw_site_name": site_name,
                "site_key": match[0],
                "lat": match[1],
                "lon": match[2],
                "launch_count": count,
            })
    matched_df = pd.DataFrame(matched)
    if matched_df.empty:
        return matched_df
    return matched_df.sort_values("launch_count", ascending=False).head(top_n).reset_index(drop=True)


def _fetch_decade_chunk(site_key: str, lat: float, lon: float,
                         chunk_start: int, chunk_end: int) -> pd.DataFrame:
    """Fetches (or loads from cache) a single decade's worth of daily weather for one site."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{site_key.replace(' ', '_')}_{chunk_start}s.json")

    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return pd.DataFrame(json.load(f))

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{chunk_start}-01-01",
        "end_date": f"{chunk_end}-12-31",
        "daily": DAILY_VARS,
        "timezone": "UTC",
    }

    try:
        resp = requests.get(ARCHIVE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})

        chunk_df = pd.DataFrame({
            "date": daily.get("time", []),
            "wind_speed_max_kmh": daily.get("windspeed_10m_max", []),
            "temp_max_c": daily.get("temperature_2m_max", []),
            "precipitation_mm": daily.get("precipitation_sum", []),
        })

        chunk_df.to_json(cache_path, orient="records")
        print(f"  Fetched {site_key} {chunk_start}-{chunk_end}: {len(chunk_df)} days")
        time.sleep(0.3)  # polite delay between requests
        return chunk_df

    except Exception as e:
        print(f"  WARNING: fetch failed for {site_key} {chunk_start}-{chunk_end}: {e}")
        return pd.DataFrame(columns=["date", "wind_speed_max_kmh", "temp_max_c", "precipitation_mm"])


def fetch_site_weather(site_key: str, lat: float, lon: float,
                        start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches a site's full weather history, chunked by decade.
    Returns a single concatenated DataFrame indexed conceptually by date.
    """
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year

    print(f"Fetching weather for {site_key} ({start_year}-{end_year})...")

    all_chunks = []
    for decade_start in range(start_year - (start_year % 10), end_year + 1, 10):
        chunk_start = max(decade_start, start_year)
        chunk_end = min(decade_start + 9, end_year)
        if chunk_start > chunk_end:
            continue
        chunk_df = _fetch_decade_chunk(site_key, lat, lon, chunk_start, chunk_end)
        all_chunks.append(chunk_df)

    if not all_chunks:
        return pd.DataFrame(columns=["date", "wind_speed_max_kmh", "temp_max_c", "precipitation_mm"])

    combined = pd.concat(all_chunks, ignore_index=True)
    combined = combined.dropna(subset=["date"]).drop_duplicates(subset=["date"])
    return combined


def enrich_with_weather(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Main entry point. Adds real weather columns for launches at recognized
    top sites. Launches at unrecognized/low-volume sites, or dates where the
    fetch failed, get weather_available=False and NaN weather values.
    """
    df = df.copy()
    df["weather_available"] = False
    df["wind_speed_max_kmh"] = np.nan
    df["temp_max_c"] = np.nan
    df["precipitation_mm"] = np.nan

    top_sites = get_top_launch_sites(df, top_n=top_n)

    if top_sites.empty:
        print("No known sites matched — check SITE_COORDINATES against your launch_site strings.")
        return df

    print(f"Matched {len(top_sites)} known sites:")
    print(top_sites[["raw_site_name", "site_key", "launch_count"]].to_string(index=False))
    print()

    df["_date_str"] = pd.to_datetime(df["launch_date"]).dt.strftime("%Y-%m-%d")

    for _, row in top_sites.iterrows():
        site_mask = df["launch_site"] == row["raw_site_name"]
        site_dates = df.loc[site_mask, "launch_date"]
        if site_dates.empty:
            continue

        start_date = site_dates.min().strftime("%Y-%m-%d")
        end_date = site_dates.max().strftime("%Y-%m-%d")

        weather_df = fetch_site_weather(row["site_key"], row["lat"], row["lon"], start_date, end_date)
        if weather_df.empty:
            continue

        weather_df = weather_df.set_index("date")

        for idx in df[site_mask].index:
            date_str = df.at[idx, "_date_str"]
            if date_str in weather_df.index:
                w = weather_df.loc[date_str]
                # .loc can return a Series (single match) — guard against duplicate dates
                if isinstance(w, pd.DataFrame):
                    w = w.iloc[0]
                df.at[idx, "wind_speed_max_kmh"] = w["wind_speed_max_kmh"]
                df.at[idx, "temp_max_c"] = w["temp_max_c"]
                df.at[idx, "precipitation_mm"] = w["precipitation_mm"]
                df.at[idx, "weather_available"] = True

    df = df.drop(columns=["_date_str"])

    coverage = df["weather_available"].mean()
    print(f"\nFinal weather coverage: {df['weather_available'].sum()} / {len(df)} launches ({coverage:.1%})")

    return df


if __name__ == "__main__":
    # Quick standalone test
    df = pd.read_csv("data/processed/launches_clean.csv", parse_dates=["launch_date"])
    enriched = enrich_with_weather(df, top_n=10)
    enriched.to_csv("data/processed/launches_with_weather.csv", index=False)
    print(enriched[["launch_date", "launch_site", "weather_available",
                     "wind_speed_max_kmh", "temp_max_c"]].dropna(subset=["wind_speed_max_kmh"]).head(10))
