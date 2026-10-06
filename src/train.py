import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from clean_data import PROCESSED_DIR
from features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, time_split
from evaluate import evaluate_predictions


def make_pipeline(model, numeric_cols, categorical_cols):
    transformers = [("num", SimpleImputer(strategy="median"), numeric_cols)]
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    preprocessor = ColumnTransformer(transformers)
    return Pipeline([("prep", preprocessor), ("model", model)])


def score_split(df, predictions):
    scored = df[["raceId", TARGET]].copy()
    scored["pred"] = predictions
    return evaluate_predictions(scored, "pred")


if __name__ == "__main__":
    model_table = pd.read_csv(PROCESSED_DIR / "model_table.csv", parse_dates=["date"])
    train, val, _test_not_used = time_split(model_table)  # test set stays untouched

    experiments = {
        "Linear Regression (all features)": (
            LinearRegression(), NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        "Random Forest (all features)": (
            RandomForestRegressor(n_estimators=300, min_samples_leaf=10,
                                  random_state=42, n_jobs=-1),
            NUMERIC_FEATURES, CATEGORICAL_FEATURES),
        "Random Forest (numeric only)": (
            RandomForestRegressor(n_estimators=300, min_samples_leaf=10,
                                  random_state=42, n_jobs=-1),
            NUMERIC_FEATURES, []),
    }

    rows = []

    # Baseline: finish = grid (no learning)
    baseline_train = score_split(train, train["grid_clean"])
    baseline_val = score_split(val, val["grid_clean"])
    rows.append({"model": "Baseline (finish = grid)",
                 "train_MAE": baseline_train["MAE"], **{f"val_{k}": v for k, v in baseline_val.items()}})

    fitted = {}
    for name, (model, num_cols, cat_cols) in experiments.items():
        pipeline = make_pipeline(model, num_cols, cat_cols)
        pipeline.fit(train[num_cols + cat_cols], train[TARGET])
        train_pred = pipeline.predict(train[num_cols + cat_cols])
        val_pred = pipeline.predict(val[num_cols + cat_cols])

        val_scores = score_split(val, val_pred)
        rows.append({"model": name,
                     "train_MAE": np.abs(train[TARGET] - train_pred).mean(),
                     **{f"val_{k}": v for k, v in val_scores.items()}})
        fitted[name] = (pipeline, num_cols + cat_cols, val_pred)

    results = pd.DataFrame(rows)
    print("\n=== Validation results (train = 2014-2021, validation = 2022) ===")
    print(results.round(3).to_string(index=False))

    # Feature importance of the best Random Forest (by validation MAE)
    rf_names = [n for n in fitted if n.startswith("Random Forest")]
    best_rf = min(rf_names, key=lambda n: np.abs(val[TARGET] - fitted[n][2]).mean())
    pipeline = fitted[best_rf][0]
    feature_names = pipeline.named_steps["prep"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    top = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
    print(f"\nTop 10 feature importances for: {best_rf}")
    print(top.round(3).to_string())

    # Save validation predictions for error analysis in the next phase
    val_out = val[["raceId", "year", "race_name", "driverRef", "grid_clean", TARGET]].copy()
    val_out["pred_best_rf"] = fitted[best_rf][2]
    val_out.to_csv(PROCESSED_DIR / "val_predictions.csv", index=False)
    print("\nSaved val_predictions.csv")