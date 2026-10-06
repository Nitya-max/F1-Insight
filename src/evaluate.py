import numpy as np


def evaluate_predictions(df, pred_col, target_col="finish_position"):
    """df needs columns: raceId, target_col, pred_col. Lower prediction = better finish."""
    errors = df[target_col] - df[pred_col]
    mae = errors.abs().mean()
    rmse = np.sqrt((errors ** 2).mean())

    winner_hits, podium_overlaps, race_spearmans = [], [], []
    for race_id, race in df.groupby("raceId"):
        predicted_order = race.sort_values(pred_col, kind="mergesort")

        predicted_winner = predicted_order.iloc[0]
        winner_hits.append(int(predicted_winner[target_col] == 1))

        predicted_podium = predicted_order.head(3)
        podium_overlaps.append((predicted_podium[target_col] <= 3).sum() / 3)

        race_spearmans.append(race[pred_col].corr(race[target_col], method="spearman"))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "race_spearman": np.nanmean(race_spearmans),
        "winner_hit_rate": np.mean(winner_hits),
        "podium_overlap": np.mean(podium_overlaps),
    }