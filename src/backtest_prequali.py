import pandas as pd
from forecast_common import build_model_table, fit_final_models, predict_all, PREQUALI_FEATURES
from features import TARGET
from evaluate import evaluate_predictions
from final_evaluation import evaluate_probabilities

FIRST_BACKTEST_YEAR = 2025

if __name__ == "__main__":
    table = build_model_table().dropna(subset=[TARGET])
    order = (table[table["year"] >= FIRST_BACKTEST_YEAR]
             .sort_values(["date", "raceId"])[["raceId", "date"]].drop_duplicates())

    parts = []
    for i, (race_id, race_date) in enumerate(order.itertuples(index=False), start=1):
        train = table[table["date"] < race_date]
        race = table[table["raceId"] == race_id]
        fitted = fit_final_models(train, PREQUALI_FEATURES)
        part = predict_all(fitted, race, PREQUALI_FEATURES)
        # Baseline: championship order (more points before the race = better predicted finish)
        part["baseline_pred"] = (-race["driver_points_before"]).rank(method="first").values
        parts.append(part)
        print(f"Race {i}/{len(order)} done")

    preds = pd.concat(parts, ignore_index=True)
    preds["is_win"] = (preds[TARGET] == 1).astype(int)

    for label, part in [("2025", preds[preds["year"] == 2025]),
                        ("2026 so far", preds[preds["year"] == 2026]),
                        ("2025 + 2026", preds)]:
        if part.empty:
            continue
        n = part["raceId"].nunique()
        reg = pd.DataFrame([
            {"model": "Baseline (championship order)", **evaluate_predictions(part, "baseline_pred")},
            {"model": "Provisional regression (no grid)", **evaluate_predictions(part, "pred_finish")}])
        win = evaluate_probabilities(part.rename(columns={"p_win": "prob"}), "is_win")
        print(f"\n=== BEFORE QUALIFYING, {label}: {n} races ===")
        print(reg.round(3).to_string(index=False))
        print(f"Win classifier picked the winner in {round(win['winner_hit_rate'] * n)} of {n} races")