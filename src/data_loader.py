import pandas as pd
import numpy as np
import re


def load_raw_data(path: str = "data/raw/space_missions.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # --- Parse launch date (handles both "...UTC" and no-time formats) ---
    df["launch_date"] = pd.to_datetime(
        df["datum"].astype(str).str.replace(" UTC", "", regex=False),
        errors="coerce"
    )
    df["year"] = df["launch_date"].dt.year
    df["decade"] = (df["year"] // 10) * 10

    # --- Extract rocket family from 'detail' column ---
    def extract_rocket_family(detail_str):
        if pd.isna(detail_str):
            return "unknown"
        name = str(detail_str).split("|")[0].strip()
        name = re.sub(r"\bBlock\s*\d+\b", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\s+", " ", name).strip()
        return name if name else "unknown"

    df["rocket_family"] = df["detail"].apply(extract_rocket_family)

    # --- Extract payload/mission description ---
    df["payload_desc"] = df["detail"].apply(
        lambda x: str(x).split("|")[1].strip() if pd.notna(x) and "|" in str(x) else "unknown"
    )

    # --- Parse location into site + country ---
    df["country"] = df["location"].apply(
        lambda x: str(x).split(",")[-1].strip() if pd.notna(x) else "unknown"
    )
    df["launch_site"] = df["location"].apply(
        lambda x: str(x).split(",")[0].strip() if pd.notna(x) else "unknown"
    )

    # --- Clean cost column (mostly missing, stored as comma-separated string) ---
    cost_col = "rocket" if "rocket" in df.columns else "_rocket"
    df["cost_musd"] = (
        df[cost_col].astype(str).str.replace(",", "", regex=False)
    )
    df["cost_musd"] = pd.to_numeric(df["cost_musd"], errors="coerce")

    # --- Collapse target to binary ---
    def to_binary_target(status):
        if pd.isna(status):
            return np.nan
        status = status.lower()
        if status == "success":
            return 1
        elif status in ["failure", "partial failure", "prelaunch failure"]:
            return 0
        return np.nan

    df["mission_success"] = df["status_mission"].apply(to_binary_target)

    before = len(df)
    df = df.dropna(subset=["mission_success", "launch_date"])
    dropped = before - len(df)
    df["mission_success"] = df["mission_success"].astype(int)

    # Sort chronologically — required for leak-free features downstream
    df = df.sort_values("launch_date").reset_index(drop=True)

    print(f"Dropped {dropped} rows with missing target/date out of {before}")
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    """Documents missing/inconsistent data — feeds directly into RESULTS.md."""
    report = {
        "total_rows": len(df),
        "missing_by_column": df.isna().sum().to_dict(),
        "year_range": (int(df["year"].min()), int(df["year"].max())),
        "unique_rocket_families": df["rocket_family"].nunique(),
        "unique_launch_sites": df["launch_site"].nunique(),
        "unique_countries": df["country"].nunique(),
        "success_rate_overall": round(df["mission_success"].mean(), 4),
        "rows_per_decade": df["decade"].value_counts().sort_index().to_dict(),
        "cost_missing_pct": round(df["cost_musd"].isna().mean() * 100, 1),
    }
    return report


if __name__ == "__main__":
    raw = load_raw_data()
    clean = clean_dataset(raw)
    clean.to_csv("data/processed/launches_clean.csv", index=False)

    report = data_quality_report(clean)
    print("\n=== DATA QUALITY REPORT ===")
    for k, v in report.items():
        print(f"{k}: {v}")
