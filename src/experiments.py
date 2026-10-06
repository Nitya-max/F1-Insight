import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from clean_data import PROCESSED_DIR
from features import NUMERIC_FEATURES, TARGET
from train import make_pipeline
from evaluate import evaluate_predictions

FOLD_YEARS = [2018, 2019, 2020, 2021, 2022]   # 2023-2024 test set stays untouched
REDUCED_FEATURES = [f for f in NUMERIC_FEATURES
                    if f not in ("round", "grid_minus_quali")]
BASELINE_NAME = "Baseline (finish = grid)"


def make_rf(leaf):
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=leaf,
                                 random_state=42, n_jobs=-1)


def get_experiments():
    return {
        "LinearReg (full)": (LinearRegression(), NUMERIC_FEATURES),
        "LinearReg (reduced)": (LinearRegression(), REDUCED_FEATURES),
        "RF leaf=10 (full)": (make_rf(10), NUMERIC_FEATURES),
        "RF leaf=10 (reduced)": (make_rf(10), REDUCED_FEATURES),
        "RF leaf=25 (reduced)": (make_rf(25), REDUCED_FEATURES),
        "RF leaf=50 (reduced)": (make_rf(50), REDUCED_FEATURES),
        "HistGradBoost (reduced)": (
            HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                          max_iter=200, min_samples_leaf=20,
                                          random_state=42),
            REDUCED_FEATURES),
    }


def walk_forward_predictions(model_table, model, feature_cols):
    parts = []
    for year in FOLD_YEARS:
        train = model_table[model_table["year"] < year]
        valid = model_table[model_table["year"] == year]
        pipeline = make_pipeline(clone(model), feature_cols, [])
        pipeline.fit(train[feature_cols], train[TARGET])
        part = valid[["raceId", "year", "race_name", "driverRef",
                      "grid_clean", TARGET]].copy()
        part["pred"] = pipeline.predict(valid[feature_cols])
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def mae_by_year(preds):
    errors = (preds[TARGET] - preds["pred"]).abs()
    return errors.groupby(preds["year"]).mean()


if __name__ == "__main__":
    model_table = pd.read_csv(PROCESSED_DIR / "model_table.csv", parse_dates=["date"])
    experiments = get_experiments()

    all_preds = {}
    rows = []
    for name, (model, cols) in experiments.items():
        print("Running:", name)
        preds = walk_forward_predictions(model_table, model, cols)
        all_preds[name] = preds
        rows.append({"model": name, **evaluate_predictions(preds, "pred")})

    baseline_preds = next(iter(all_preds.values())).copy()
    baseline_preds["pred"] = baseline_preds["grid_clean"]
    rows.insert(0, {"model": BASELINE_NAME,
                    **evaluate_predictions(baseline_preds, "pred")})

    results = pd.DataFrame(rows)
    n_races = baseline_preds["raceId"].nunique()
    print(f"\n=== Pooled walk-forward results ({n_races} races, folds 2018-2022) ===")
    print(results.round(3).to_string(index=False))

    per_year = {BASELINE_NAME: mae_by_year(baseline_preds)}
    for name, preds in all_preds.items():
        per_year[name] = mae_by_year(preds)
    print("\n=== MAE by validation year ===")
    print(pd.DataFrame(per_year).T.round(3).to_string())

    # Pick best model (lowest pooled MAE) for error analysis
    candidates = results[results["model"] != BASELINE_NAME]
    best_name = candidates.sort_values("MAE").iloc[0]["model"]
    print("\nLowest pooled MAE:", best_name)

    # ---------- Error analysis ----------
    base = pd.read_csv(PROCESSED_DIR / "base_table.csv",
                       usecols=["raceId", "driverRef", "dnf"])
    best = all_preds[best_name].merge(base, on=["raceId", "driverRef"], how="left")
    assert len(best) == len(all_preds[best_name])

    best["model_error"] = (best[TARGET] - best["pred"]).abs()
    best["baseline_error"] = (best[TARGET] - best["grid_clean"]).abs()
    best["grid_group"] = pd.cut(best["grid_clean"], bins=[0, 3, 10, 25],
                                labels=["P1-3", "P4-10", "P11+"])
    best["outcome"] = best["dnf"].map({0: "classified", 1: "not classified"})

    for column in ["grid_group", "outcome"]:
        summary = best.groupby(column, observed=True).agg(
            rows=("model_error", "size"),
            model_MAE=("model_error", "mean"),
            baseline_MAE=("baseline_error", "mean"))
        print(f"\n=== Error by {column} ({best_name}) ===")
        print(summary.round(3).to_string())

    print("\n=== 8 largest misses ===")
    worst = best.sort_values("model_error", ascending=False).head(8)
    print(worst[["year", "race_name", "driverRef", "grid_clean",
                 TARGET, "pred", "outcome"]].round(1).to_string(index=False))

    # ---------- Permutation importance on the last fold (2022) ----------
    best_model, best_cols = experiments[best_name]
    train = model_table[model_table["year"] < 2022]
    valid = model_table[model_table["year"] == 2022]
    pipeline = make_pipeline(clone(best_model), best_cols, [])
    pipeline.fit(train[best_cols], train[TARGET])
    perm = permutation_importance(pipeline, valid[best_cols], valid[TARGET],
                                  scoring="neg_mean_absolute_error",
                                  n_repeats=10, random_state=42)
    importance = pd.Series(perm.importances_mean, index=best_cols)
    print("\n=== Permutation importance (increase in MAE when shuffled), 2022 ===")
    print(importance.sort_values(ascending=False).round(3).to_string())

    best.to_csv(PROCESSED_DIR / "walkforward_best_predictions.csv", index=False)