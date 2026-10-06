from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

CORE_TABLES = ["races", "results", "qualifying", "drivers",
               "constructors", "circuits", "status"]


def load_table(name):
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Put the Kaggle CSVs in data/raw/")
    # This dataset writes missing values as \N
    return pd.read_csv(path, na_values=["\\N"])


def load_core_tables():
    return {name: load_table(name) for name in CORE_TABLES}


if __name__ == "__main__":
    tables = load_core_tables()

    # 1. Shapes and real column names
    for name, df in tables.items():
        print(f"\n=== {name}: {df.shape[0]} rows, {df.shape[1]} columns")
        print(list(df.columns))

    races = tables["races"]
    results = tables["results"]
    qualifying = tables["qualifying"]
    drivers = tables["drivers"]
    constructors = tables["constructors"]

    # 2. Season coverage
    print("\nSeasons in file:", races["year"].min(), "to", races["year"].max())
    print("\nRaces per season since 2014:")
    print(races[races["year"] >= 2014].groupby("year").size().to_string())

    # 3. Duplicates: should be one row per driver per race
    dups = results.duplicated(subset=["raceId", "driverId"]).sum()
    print("\nDuplicate (raceId, driverId) rows in results:", dups)

    # 4. Join test: row count must NOT change
    race_info = races[["raceId", "year", "round", "circuitId", "name", "date"]]
    race_info = race_info.rename(columns={"name": "race_name"})
    team_info = constructors[["constructorId", "name"]]
    team_info = team_info.rename(columns={"name": "constructor_name"})

    merged = (results
              .merge(race_info, on="raceId", how="left")
              .merge(drivers[["driverId", "driverRef"]], on="driverId", how="left")
              .merge(team_info, on="constructorId", how="left"))
    print("Rows before/after joins:", len(results), len(merged))

    # 5. Missing values in the columns we care about (modern era)
    modern = merged[merged["year"] >= 2014]
    print("\nMissing values, 2014 onward:")
    print(modern[["grid", "position", "positionOrder"]].isna().sum().to_string())
    print("Rows with grid == 0 (usually pit-lane starts):", (modern["grid"] == 0).sum())

    # 6. Does qualifying line up with results?
    quali = qualifying[["raceId", "driverId", "position"]]
    quali = quali.rename(columns={"position": "quali_position"})
    with_quali = modern.merge(quali, on=["raceId", "driverId"], how="left")
    missing_share = with_quali["quali_position"].isna().mean()
    print(f"\nShare of 2014+ result rows with no qualifying match: {missing_share:.3f}")

    print("\nSample rows:")
    cols = ["year", "round", "race_name", "driverRef", "constructor_name",
            "grid", "positionOrder"]
    print(modern[cols].head().to_string(index=False))