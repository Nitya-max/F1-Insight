import sys
import pandas as pd
from forecast_common import build_model_table, fit_final_models, predict_all
from load_data import PROJECT_ROOT
from features import TARGET

PRED_DIR = PROJECT_ROOT / "predictions"

if __name__ == "__main__":
    round_number = int(sys.argv[1]) if len(sys.argv) > 1 else None   # optional override

    table = build_model_table(include_upcoming=True, round_number=round_number)
    upcoming = table[table[TARGET].isna()]
    train = table[table[TARGET].notna()]
    if upcoming.empty:
        raise ValueError("No upcoming rows were built.")

    fitted = fit_final_models(train)
    result = predict_all(fitted, upcoming)
    result["win_chance"] = result["p_win"] / result["p_win"].sum()
    result["predicted_rank"] = result["pred_finish"].rank(method="first").astype(int)
    result = result.sort_values("predicted_rank")

    year = int(result["year"].iloc[0])
    rnd = int(result["round"].iloc[0])
    name = result["race_name"].iloc[0]
    print(f"\n=== Prediction: {year} {name} (round {rnd}) ===")
    print(f"Trained on {len(train)} driver-races; latest race in training data: "
          f"{train['date'].max().date()}")

    show = pd.DataFrame({
        "Predicted rank": result["predicted_rank"],
        "Driver": result["driverRef"],
        "Team": result["constructorRef"],
        "Grid (qualifying)": result["grid_clean"].astype(int),
        "Predicted finish": result["pred_finish"].round(1),
        "Win chance %": (result["win_chance"] * 100).round(1),
        "Podium prob %": (result["p_podium"] * 100).round(1),
    })
    print(show.to_string(index=False))

    winner = result.sort_values("p_win", ascending=False).iloc[0]
    podium = result.sort_values("p_podium", ascending=False).head(3)["driverRef"].tolist()
    print(f"\nPredicted winner: {winner['driverRef']} "
          f"(normalised win chance {winner['win_chance'] * 100:.1f}%)")
    print("Predicted podium:", ", ".join(podium))

    result["generated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    PRED_DIR.mkdir(exist_ok=True)
    slug = name.replace(" ", "_")
    out_path = PRED_DIR / f"{year}_round{rnd:02d}_{slug}.csv"
    result.to_csv(out_path, index=False)
    print("\nSaved:", out_path)
    print("Tip: git add predictions && git commit BEFORE the race starts.")
    