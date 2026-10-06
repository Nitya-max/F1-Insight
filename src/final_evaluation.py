import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from clean_data import PROCESSED_DIR, PROJECT_ROOT
from features import TARGET
from train import make_pipeline
from evaluate import evaluate_predictions
from experiments import FOLD_YEARS, REDUCED_FEATURES, walk_forward_predictions

MODELS_DIR = PROJECT_ROOT / "models"
FINAL_TRAIN_END = 2022
REG_MARGIN = 0.05     # new regressor must beat LinearReg MAE by more than this
CLF_MARGIN = 0.005    # RF must beat Logistic log loss by more than this


def make_classifier_pipeline(model):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", model)])


def get_classifiers():
    return {
        "Logistic": LogisticRegression(max_iter=1000),
        "RF leaf=25": RandomForestClassifier(n_estimators=300, min_samples_leaf=25,
                                             random_state=42, n_jobs=-1),
    }


def walk_forward_probabilities(model_table, model, target_col):
    parts = []
    for year in FOLD_YEARS:
        train = model_table[model_table["year"] < year]
        valid = model_table[model_table["year"] == year]
        pipeline = make_classifier_pipeline(clone(model))
        pipeline.fit(train[REDUCED_FEATURES], train[target_col])
        part = valid[["raceId", TARGET, "is_win", "is_podium"]].copy()
        part["prob"] = pipeline.predict_proba(valid[REDUCED_FEATURES])[:, 1]
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def evaluate_probabilities(df, target_col):
    winner_hits, podium_overlaps = [], []
    for _, race in df.groupby("raceId"):
        ranked = race.sort_values("prob", ascending=False, kind="mergesort")
        winner_hits.append(int(ranked.iloc[0][TARGET] == 1))
        podium_overlaps.append((ranked.head(3)[TARGET] <= 3).sum() / 3)
    return {"ROC_AUC": roc_auc_score(df[target_col], df["prob"]),
            "log_loss": log_loss(df[target_col], df["prob"]),
            "winner_hit_rate": np.mean(winner_hits),
            "podium_overlap": np.mean(podium_overlaps)}


def bootstrap_mae_gain(df, pred_col, n_boot=1000):
    """95% interval for (baseline MAE - model MAE), resampling whole races."""
    tmp = df.copy()
    tmp["model_err"] = (tmp[TARGET] - tmp[pred_col]).abs()
    tmp["base_err"] = (tmp[TARGET] - tmp["grid_clean"]).abs()
    per_race = tmp.groupby("raceId").agg(model_sum=("model_err", "sum"),
                                         base_sum=("base_err", "sum"),
                                         rows=("model_err", "size"))
    gains = []
    for i in range(n_boot):
        sample = per_race.sample(n=len(per_race), replace=True, random_state=i)
        gains.append((sample["base_sum"].sum() - sample["model_sum"].sum())
                     / sample["rows"].sum())
    return np.percentile(gains, [2.5, 97.5])


if __name__ == "__main__":
    model_table = pd.read_csv(PROCESSED_DIR / "model_table.csv", parse_dates=["date"])
    model_table["is_win"] = (model_table[TARGET] == 1).astype(int)
    model_table["is_podium"] = (model_table[TARGET] <= 3).astype(int)

    # ================= PART A: regression (walk-forward, 2018-2022) =================
    regressors = {
        "LinearReg (reduced)": LinearRegression(),
        "HGB median-loss (reduced)": HistGradientBoostingRegressor(
            loss="absolute_error", max_depth=3, learning_rate=0.05,
            max_iter=200, min_samples_leaf=20, random_state=42),
    }
    reg_preds = {name: walk_forward_predictions(model_table, model, REDUCED_FEATURES)
                 for name, model in regressors.items()}

    baseline = reg_preds["LinearReg (reduced)"].copy()
    baseline["pred"] = baseline["grid_clean"]
    rows = [{"model": "Baseline (finish = grid)", **evaluate_predictions(baseline, "pred")}]
    for name, preds in reg_preds.items():
        rows.append({"model": name, **evaluate_predictions(preds, "pred")})
    reg_table = pd.DataFrame(rows)
    print("=== PART A: regression, pooled walk-forward (2018-2022) ===")
    print(reg_table.round(3).to_string(index=False))

    groups = pd.cut(baseline["grid_clean"], bins=[0, 3, 10, 25],
                    labels=["P1-3", "P4-10", "P11+"])
    by_group = {"Baseline": (baseline[TARGET] - baseline["pred"]).abs()
                .groupby(groups, observed=True).mean()}
    for name, preds in reg_preds.items():
        by_group[name] = ((preds[TARGET] - preds["pred"]).abs()
                          .groupby(groups, observed=True).mean())
    print("\nMAE by starting group (does median-loss fix the front of the grid?):")
    print(pd.DataFrame(by_group).round(3).to_string())

    mae = reg_table.set_index("model")["MAE"]
    if mae["HGB median-loss (reduced)"] < mae["LinearReg (reduced)"] - REG_MARGIN:
        final_reg = "HGB median-loss (reduced)"
    else:
        final_reg = "LinearReg (reduced)"
    low, high = bootstrap_mae_gain(reg_preds[final_reg], "pred")
    print(f"\nFinal regression model by pre-set rule: {final_reg}")
    print(f"Walk-forward MAE gain over baseline, 95% bootstrap interval: "
          f"[{low:.3f}, {high:.3f}]")

    # ================= PART B: probability models (walk-forward) =================
    clf_rows = []
    for target_col in ["is_win", "is_podium"]:
        for name, model in get_classifiers().items():
            preds = walk_forward_probabilities(model_table, model, target_col)
            clf_rows.append({"target": target_col, "model": name,
                             **evaluate_probabilities(preds, target_col)})
        y = preds[target_col]
        ref = log_loss(y, np.full(len(y), y.mean()))
        print(f"\nReference log loss for {target_col} (constant base rate): {ref:.3f}")
    clf_table = pd.DataFrame(clf_rows)
    print("\n=== PART B: classifiers, pooled walk-forward (2018-2022) ===")
    print(clf_table.round(3).to_string(index=False))

    final_clf = {}
    for target_col in ["is_win", "is_podium"]:
        sub = clf_table[clf_table["target"] == target_col].set_index("model")["log_loss"]
        use_rf = sub["RF leaf=25"] < sub["Logistic"] - CLF_MARGIN
        final_clf[target_col] = "RF leaf=25" if use_rf else "Logistic"
    print("\nFinal classifiers by pre-set rule:", final_clf)

    # ================= PART C: FINAL TEST EVALUATION (used once) =================
    train = model_table[model_table["year"] <= FINAL_TRAIN_END]
    test = model_table[model_table["year"] > FINAL_TRAIN_END].copy()
    print(f"\n=== PART C: FINAL TEST (train 2014-{FINAL_TRAIN_END}: {len(train)} rows; "
          f"test {test['year'].min()}-{test['year'].max()}: {len(test)} rows, "
          f"{test['raceId'].nunique()} races) ===")

    reg_pipeline = make_pipeline(clone(regressors[final_reg]), REDUCED_FEATURES, [])
    reg_pipeline.fit(train[REDUCED_FEATURES], train[TARGET])
    test["pred_finish"] = reg_pipeline.predict(test[REDUCED_FEATURES])
    test["baseline_pred"] = test["grid_clean"]

    classifiers = get_classifiers()
    clf_pipelines = {}
    for target_col, prob_col in [("is_win", "p_win"), ("is_podium", "p_podium")]:
        pipeline = make_classifier_pipeline(clone(classifiers[final_clf[target_col]]))
        pipeline.fit(train[REDUCED_FEATURES], train[target_col])
        test[prob_col] = pipeline.predict_proba(test[REDUCED_FEATURES])[:, 1]
        clf_pipelines[target_col] = pipeline

    final_rows = [
        {"model": "Baseline (finish = grid)", **evaluate_predictions(test, "baseline_pred")},
        {"model": final_reg, **evaluate_predictions(test, "pred_finish")},
    ]
    print("\nRegression on the test set:")
    print(pd.DataFrame(final_rows).round(3).to_string(index=False))

    print("\nMAE by test year:")
    for year, part in test.groupby("year"):
        base_mae = (part[TARGET] - part["baseline_pred"]).abs().mean()
        model_mae = (part[TARGET] - part["pred_finish"]).abs().mean()
        print(f"  {year}: baseline {base_mae:.3f} | model {model_mae:.3f}")

    low, high = bootstrap_mae_gain(test, "pred_finish")
    print(f"Test MAE gain over baseline, 95% bootstrap interval: [{low:.3f}, {high:.3f}]")

    win_scores = evaluate_probabilities(test.rename(columns={"p_win": "prob"}), "is_win")
    podium_scores = evaluate_probabilities(
        test.rename(columns={"p_podium": "prob"}), "is_podium")
    clf_final = pd.DataFrame([
        {"model": f"Win classifier ({final_clf['is_win']})", **win_scores},
        {"model": f"Podium classifier ({final_clf['is_podium']})", **podium_scores},
    ])
    print("\nProbability models on the test set:")
    print(clf_final.round(3).to_string(index=False))

    # ================= Save models + predictions for the app =================
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": reg_pipeline, "features": REDUCED_FEATURES,
                 "name": final_reg}, MODELS_DIR / "finish_regressor.joblib")
    joblib.dump({"pipeline": clf_pipelines["is_win"], "features": REDUCED_FEATURES,
                 "name": final_clf["is_win"]}, MODELS_DIR / "win_classifier.joblib")
    joblib.dump({"pipeline": clf_pipelines["is_podium"], "features": REDUCED_FEATURES,
                 "name": final_clf["is_podium"]}, MODELS_DIR / "podium_classifier.joblib")
    test.to_csv(PROCESSED_DIR / "test_predictions.csv", index=False)
    print("\nSaved 3 models to models/ and test_predictions.csv to data/processed/")