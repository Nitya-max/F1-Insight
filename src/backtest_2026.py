import pandas as pd
from forecast_common import build_model_table, fit_final_models, predict_all
from clean_data import PROCESSED_DIR
from features import TARGET
from evaluate import evaluate_predictions
from final_evaluation import evaluate_probabilities, bootstrap_mae_gain

FIRST_BACKTEST_YEAR = 2025


def winner_table(preds):
    rows = []
    for _, race in preds.groupby("raceId"):
        actual = race.loc[race[TARGET] == 1, "driverRef"].iloc[0]
        pick = race.sort_values("p_win", ascending=False, kind="mergesort").iloc[0]["driverRef"]
        front = race.sort_values("grid_clean", kind="mergesort").iloc[0]["driverRef"]
        rows.append({"year": race["year"].iloc[0], "round": race["round"].iloc[0],
                     "race": race["race_name"].iloc[0], "actual_winner": actual,
                     "model_pick": pick, "model_hit": pick == actual,
                     "front_row_pick": front, "front_row_hit": front == actual})
    return pd.DataFrame(rows).sort_values(["year", "round"])


if __name__ == "__main__":
    table = build_model_table().dropna(subset=[TARGET])

    order = (table[table["year"] >= FIRST_BACKTEST_YEAR]
             .sort_values(["date", "raceId"])[["raceId", "date"]].drop_duplicates())
    parts = []
    for i, (race_id, race_date) in enumerate(order.itertuples(index=False), start=1):
        train = table[table["date"] < race_date]          # only strictly earlier races
        race = table[table["raceId"] == race_id]
        fitted = fit_final_models(train)
        parts.append(predict_all(fitted, race))
        print(f"Backtest race {i}/{len(order)} done")

    preds = pd.concat(parts, ignore_index=True)
    preds["is_win"] = (preds[TARGET] == 1).astype(int)
    preds["is_podium"] = (preds[TARGET] <= 3).astype(int)
    preds["baseline_pred"] = preds["grid_clean"]

    for label, part in [("2025", preds[preds["year"] == 2025]),
                        ("2026 so far", preds[preds["year"] == 2026]),
                        ("2025 + 2026", preds)]:
        if part.empty:
            continue
        n_races = part["raceId"].nunique()
        reg = pd.DataFrame([
            {"model": "Baseline (finish = grid)", **evaluate_predictions(part, "baseline_pred")},
            {"model": "Final regression", **evaluate_predictions(part, "pred_finish")}])
        win = evaluate_probabilities(part.rename(columns={"p_win": "prob"}), "is_win")
        print(f"\n=== {label}: {n_races} races ===")
        print(reg.round(3).to_string(index=False))
        print(f"Win classifier: picked the winner in {round(win['winner_hit_rate'] * n_races)} "
              f"of {n_races} races | ROC-AUC {win['ROC_AUC']:.3f} | log loss {win['log_loss']:.3f}")

    low, high = bootstrap_mae_gain(preds, "pred_finish")
    print(f"\nMAE gain over baseline (2025+2026), 95% bootstrap interval: [{low:.3f}, {high:.3f}]")

    wt = winner_table(preds)
    print("\n=== 2026 race-by-race winner predictions ===")
    print(wt[wt["year"] == 2026].to_string(index=False))

    preds.to_csv(PROCESSED_DIR / "backtest_2025_2026.csv", index=False)
    print("\nSaved backtest_2025_2026.csv")