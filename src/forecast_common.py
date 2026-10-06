import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from clean_data import PROCESSED_DIR
from load_data import RAW_DIR
from features import build_features, TARGET
from train import make_pipeline
from experiments import REDUCED_FEATURES
from final_evaluation import make_classifier_pipeline

HISTORY_COLUMNS = ["raceId", "year", "round", "date", "race_name", "driverRef",
                   "constructorRef", "circuitRef", "grid_clean", "quali_position",
                   "pit_lane_start", "finish_position", "dnf", "points"]

# Used BEFORE qualifying: no grid or qualifying information
PREQUALI_FEATURES = ["driver_form_finish5", "driver_dnf_rate10", "driver_circuit_avg_finish",
                     "constructor_form_finish5", "driver_points_before",
                     "constructor_points_before"]


def load_new_seasons():
    results = pd.read_csv(RAW_DIR / "jolpica_results.csv")
    quali = pd.read_csv(RAW_DIR / "jolpica_qualifying.csv")
    new = results.merge(quali, on=["year", "round", "driverRef"], how="left")
    if len(new) != len(results):
        raise ValueError("Merging qualifying changed the row count (duplicates?)")
    new["date"] = pd.to_datetime(new["date"])
    new["raceId"] = new["year"] * 100 + new["round"]   # synthetic, cannot clash with Kaggle IDs

    max_grid = new.groupby("raceId")["grid"].transform("max")
    new["pit_lane_start"] = (new["grid"] == 0).astype(int)
    new["grid_clean"] = new["grid"].where(new["grid"] > 0, max_grid + 1)
    new["quali_position"] = new["quali_position"].fillna(new["grid_clean"])
    return new


def build_extended_history():
    old = pd.read_csv(PROCESSED_DIR / "base_table.csv", parse_dates=["date"])
    new = load_new_seasons()

    bad_races = 0
    for _, race in new.groupby("raceId"):
        if sorted(race["finish_position"]) != list(range(1, len(race) + 1)):
            bad_races += 1
    print("Races where finish positions are not exactly 1..N:", bad_races)
    print("Drivers with no history before 2025:", sorted(set(new["driverRef"]) - set(old["driverRef"])))

    return pd.concat([old[HISTORY_COLUMNS], new[HISTORY_COLUMNS]], ignore_index=True)


def find_next_race(schedule, results, season, round_number=None):
    this_season = schedule[schedule["year"] == season]
    if round_number is not None:
        return this_season[this_season["round"] == round_number].iloc[0]
    done = set(results.loc[results["year"] == season, "round"])
    # one day of grace: a race that just finished may not have results published yet
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    candidates = this_season[(~this_season["round"].isin(done)) & (this_season["date"] >= cutoff)]
    if candidates.empty:
        raise ValueError("No upcoming race found on the schedule (season finished?).")
    return candidates.sort_values("date").iloc[0]


def build_upcoming_rows(round_number=None, stage="final"):
    schedule = pd.read_csv(RAW_DIR / "jolpica_schedule.csv", parse_dates=["date"])
    results = pd.read_csv(RAW_DIR / "jolpica_results.csv")
    quali = pd.read_csv(RAW_DIR / "jolpica_qualifying.csv")

    season = int(schedule["year"].max())
    race = find_next_race(schedule, results, season, round_number)
    ordered = results.sort_values(["year", "round"])

    if stage == "final":
        q = quali[(quali["year"] == season) & (quali["round"] == race["round"])]
        if q.empty:
            raise ValueError(f"Qualifying for round {race['round']} ({race['race_name']}) "
                             f"is not available yet.")
        rows = q[["driverRef", "quali_position"]].copy()
        latest_team = ordered.groupby("driverRef")["constructorRef"].last()
        rows["constructorRef"] = rows["driverRef"].map(latest_team)
        if rows["constructorRef"].isna().any():
            missing = rows.loc[rows["constructorRef"].isna(), "driverRef"].tolist()
            raise ValueError(f"No race history for drivers: {missing}.")
        rows["grid_clean"] = rows["quali_position"]   # assumption: no grid penalties
    else:
        # Provisional: assume the same drivers and teams as the latest completed race
        last = ordered.iloc[-1]
        roster = ordered[(ordered["year"] == last["year"]) & (ordered["round"] == last["round"])]
        rows = roster[["driverRef", "constructorRef"]].copy()
        rows["quali_position"] = np.nan
        rows["grid_clean"] = np.nan

    rows["year"] = season
    rows["round"] = int(race["round"])
    rows["date"] = race["date"]
    rows["race_name"] = race["race_name"]
    rows["circuitRef"] = race["circuitRef"]
    rows["raceId"] = season * 100 + int(race["round"])
    rows["pit_lane_start"] = 0
    rows["finish_position"] = np.nan              # unknown: this is what we predict
    rows["dnf"] = 0
    rows["points"] = 0.0
    return rows[HISTORY_COLUMNS]


def build_model_table(include_upcoming=False, round_number=None, stage="final"):
    frames = [build_extended_history()]
    if include_upcoming:
        frames.append(build_upcoming_rows(round_number, stage))
    combined = pd.concat(frames, ignore_index=True)
    combined["driverId"] = combined["driverRef"]
    combined["constructorId"] = combined["constructorRef"]
    combined["circuitId"] = combined["circuitRef"]
    return build_features(combined)


def make_final_estimators():
    # Same model types and settings chosen in Phases 5-6, before any 2025-2026 data was seen
    return {
        "finish": HistGradientBoostingRegressor(loss="absolute_error", max_depth=3,
                                                learning_rate=0.05, max_iter=200,
                                                min_samples_leaf=20, random_state=42),
        "win": LogisticRegression(max_iter=1000),
        "podium": RandomForestClassifier(n_estimators=300, min_samples_leaf=25,
                                         random_state=42, n_jobs=-1),
    }


def fit_final_models(train, features=REDUCED_FEATURES):
    estimators = make_final_estimators()
    X = train[features]
    return {
        "finish": make_pipeline(estimators["finish"], features, []).fit(X, train[TARGET]),
        "win": make_classifier_pipeline(estimators["win"]).fit(X, (train[TARGET] == 1).astype(int)),
        "podium": make_classifier_pipeline(estimators["podium"]).fit(X, (train[TARGET] <= 3).astype(int)),
    }


def predict_all(fitted, rows, features=REDUCED_FEATURES):
    X = rows[features]
    out = rows[["raceId", "year", "round", "race_name", "driverRef",
                "constructorRef", "grid_clean", TARGET]].copy()
    out["pred_finish"] = fitted["finish"].predict(X)
    out["p_win"] = fitted["win"].predict_proba(X)[:, 1]
    out["p_podium"] = fitted["podium"].predict_proba(X)[:, 1]
    return out