import pandas as pd
from load_data import load_core_tables, PROJECT_ROOT

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIRST_YEAR = 2014

# Only known after the race. Never use these as features.
OUTCOME_COLUMNS = ["position", "positionText", "points", "laps", "time",
                   "milliseconds", "fastestLap", "rank", "fastestLapTime",
                   "fastestLapSpeed", "statusId", "status", "dnf"]


def build_base_table():
    tables = load_core_tables()
    races = tables["races"]
    results = tables["results"]
    qualifying = tables["qualifying"]

    race_info = races[["raceId", "year", "round", "circuitId", "name", "date"]]
    race_info = race_info.rename(columns={"name": "race_name"})

    df = results.merge(race_info, on="raceId", how="inner")
    df = df[df["year"] >= FIRST_YEAR].copy()

    # Duplicate check for the modern era only
    dup_mask = df.duplicated(subset=["raceId", "driverId"], keep=False)
    if dup_mask.sum() > 0:
        print(df[dup_mask][["raceId", "year", "race_name", "driverId", "positionOrder"]])
        raise ValueError("Duplicate (raceId, driverId) rows found in 2014+ data")

    rows_before = len(df)

    df = df.merge(tables["drivers"][["driverId", "driverRef", "dob"]],
                  on="driverId", how="left")
    df = df.merge(tables["constructors"][["constructorId", "constructorRef"]],
                  on="constructorId", how="left")
    df = df.merge(tables["circuits"][["circuitId", "circuitRef"]],
                  on="circuitId", how="left")
    df = df.merge(tables["status"], on="statusId", how="left")

    quali = qualifying[["raceId", "driverId", "position"]]
    quali = quali.rename(columns={"position": "quali_position"})
    df = df.merge(quali, on=["raceId", "driverId"], how="left")

    if len(df) != rows_before:
        raise ValueError(f"Joins changed row count: {rows_before} -> {len(df)}")

    # Grid 0 means pit-lane start: place them behind the last real grid slot
    df["pit_lane_start"] = (df["grid"] == 0).astype(int)
    max_grid_in_race = df.groupby("raceId")["grid"].transform("max")
    df["grid_clean"] = df["grid"].where(df["grid"] > 0, max_grid_in_race + 1)

    # Missing qualifying position: fall back to the grid slot (both known pre-race)
    df["quali_missing"] = df["quali_position"].isna().astype(int)
    df["quali_position"] = df["quali_position"].fillna(df["grid_clean"])

    # Outcome flag (for history features later, never for the same race)
    df["dnf"] = df["position"].isna().astype(int)

    df["finish_position"] = df["positionOrder"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "raceId", "finish_position"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    base = build_base_table()
    out_path = PROCESSED_DIR / "base_table.csv"
    base.to_csv(out_path, index=False)
    print("Saved:", out_path)
    print("Shape:", base.shape)
    print("Missing values in key columns:")
    key_cols = ["grid_clean", "quali_position", "finish_position",
                "driverRef", "constructorRef", "circuitRef"]
    print(base[key_cols].isna().sum().to_string())