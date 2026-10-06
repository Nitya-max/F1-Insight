import pandas as pd
from clean_data import PROCESSED_DIR

TARGET = "finish_position"
ID_COLUMNS = ["raceId", "year", "date", "race_name", "driverRef"]
NUMERIC_FEATURES = [
    "grid_clean", "quali_position", "grid_minus_quali", "pit_lane_start", "round",
    "driver_form_finish5", "driver_dnf_rate10", "driver_circuit_avg_finish",
    "constructor_form_finish5", "driver_points_before", "constructor_points_before",
]
CATEGORICAL_FEATURES = ["constructorRef", "circuitRef"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TRAIN_END_YEAR = 2021
VAL_YEAR = 2022


def build_features(base):
    # Sort by date only. Never sort by outcomes, so row order can't leak them.
    df = base.sort_values(["date", "raceId", "driverId"]).reset_index(drop=True)

    # Pre-race information
    df["grid_minus_quali"] = df["grid_clean"] - df["quali_position"]  # penalties

    # Driver form: shift(1) makes each row see only PREVIOUS races
    by_driver = df.groupby("driverId")
    df["driver_form_finish5"] = by_driver["finish_position"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["driver_dnf_rate10"] = by_driver["dnf"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean())

    # Driver's points earlier in the same season (before this race)
    df["driver_points_before"] = df.groupby(["driverId", "year"])["points"].transform(
        lambda s: s.shift(1).fillna(0).cumsum())

    # Driver's average finish at this circuit in earlier visits
    df["driver_circuit_avg_finish"] = df.groupby(["driverId", "circuitId"])[
        "finish_position"].transform(lambda s: s.shift(1).expanding().mean())
    df["driver_circuit_avg_finish"] = df["driver_circuit_avg_finish"].fillna(
        df["driver_form_finish5"])  # no circuit history: fall back to recent form

    # Constructor features: aggregate both cars first, then look at earlier races
    team_race = (df.groupby(["constructorId", "raceId"])
                   .agg(year=("year", "first"),
                        date=("date", "first"),
                        team_avg_finish=("finish_position", "mean"),
                        team_points=("points", "sum"))
                   .reset_index()
                   .sort_values(["constructorId", "date", "raceId"]))
    team_race["constructor_form_finish5"] = team_race.groupby("constructorId")[
        "team_avg_finish"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    team_race["constructor_points_before"] = team_race.groupby(
        ["constructorId", "year"])["team_points"].transform(
        lambda s: s.shift(1).fillna(0).cumsum())

    df = df.merge(team_race[["constructorId", "raceId",
                             "constructor_form_finish5", "constructor_points_before"]],
                  on=["constructorId", "raceId"], how="left")

    df = df.sort_values(["date", "raceId", "driverId"]).reset_index(drop=True)
    return df


def time_split(df):
    train = df[df["year"] <= TRAIN_END_YEAR]
    val = df[df["year"] == VAL_YEAR]
    test = df[df["year"] > VAL_YEAR]
    return train, val, test


def run_leakage_test(base):
    """Change one race's outcomes. Features of that race and all EARLIER
    races must not change. If they do, a feature is using the outcome."""
    features_a = build_features(base)

    race_ids = base.sort_values(["date", "raceId"])["raceId"].drop_duplicates().tolist()
    test_race = race_ids[len(race_ids) // 2]
    test_date = base.loc[base["raceId"] == test_race, "date"].iloc[0]

    tampered = base.copy()
    mask = tampered["raceId"] == test_race
    tampered.loc[mask, "finish_position"] = 999
    tampered.loc[mask, "points"] = 999
    tampered.loc[mask, "dnf"] = 1
    features_b = build_features(tampered)

    upto = features_a["date"] <= test_date
    same_a = features_a.loc[upto, FEATURE_COLUMNS].reset_index(drop=True)
    same_b = features_b.loc[upto, FEATURE_COLUMNS].reset_index(drop=True)
    pd.testing.assert_frame_equal(same_a, same_b)  # raises if there is leakage

    later_a = features_a.loc[~upto, FEATURE_COLUMNS].reset_index(drop=True)
    later_b = features_b.loc[~upto, FEATURE_COLUMNS].reset_index(drop=True)
    later_changed = not later_a.equals(later_b)
    print("Leakage test PASSED: features up to the tampered race are unchanged.")
    print("Later races changed (should be True, proves history is used):", later_changed)


if __name__ == "__main__":
    base = pd.read_csv(PROCESSED_DIR / "base_table.csv", parse_dates=["date"])
    run_leakage_test(base)

    model_table = build_features(base)
    keep = ID_COLUMNS + FEATURE_COLUMNS + [TARGET]
    model_table = model_table[keep]
    model_table.to_csv(PROCESSED_DIR / "model_table.csv", index=False)
    print("\nSaved model_table.csv, shape:", model_table.shape)

    print("\nMissing values per feature:")
    print(model_table[FEATURE_COLUMNS].isna().sum().to_string())

    train, val, test = time_split(model_table)
    print("\nSplit sizes -> train:", len(train), "| val:", len(val), "| test:", len(test))
    for name, part in [("train", train), ("val", val), ("test", test)]:
        mae = (part[TARGET] - part["grid_clean"]).abs().mean()
        print(f"Baseline MAE (finish = grid), {name}: {mae:.3f}")

    print("\nSpearman correlation with finish position (TRAIN only):")
    corr = train[NUMERIC_FEATURES + [TARGET]].corr(method="spearman")[TARGET]
    corr = corr.drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
    print(corr.round(3).to_string())