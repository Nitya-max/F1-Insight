from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, log_loss)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

SOURCES = {
    "Final test set: 2023-2024 (used once)": "test_predictions.csv",
    "Backtest: 2025-2026 (retrained before each race)": "backtest_2025_2026.csv",
}
NEEDED = ["raceId", "finish_position", "grid_clean", "pred_finish", "baseline_pred",
          "p_win", "p_podium", "is_win", "is_podium"]

# Recorded from our own runs of experiments.py and final_evaluation.py
# (walk-forward validation on 2018-2022, 103 races).
REGRESSION_RESULTS = pd.DataFrame([
    ("Baseline rule: finish = grid (no learning)", 3.660, 5.211, 0.595, 0.476, 0.696),
    ("Linear Regression (all numeric features)", 3.415, 4.434, 0.634, 0.515, 0.686),
    ("Linear Regression (reduced features)", 3.418, 4.435, 0.633, 0.495, 0.689),
    ("Random Forest, leaf 10 (all numeric features)", 3.387, 4.436, 0.632, 0.359, 0.683),
    ("Random Forest, leaf 10 (reduced features)", 3.391, 4.438, 0.630, 0.350, 0.680),
    ("Random Forest, leaf 25 (reduced features)", 3.367, 4.404, 0.638, 0.369, 0.699),
    ("Random Forest, leaf 50 (reduced features)", 3.381, 4.401, 0.637, 0.388, 0.696),
    ("Gradient boosting, squared loss (reduced)", 3.391, 4.434, 0.630, 0.388, 0.680),
    ("Gradient boosting, median loss (reduced)  <- CHOSEN", 3.197, 4.534, 0.639, 0.447, 0.702),
], columns=["Model", "MAE", "RMSE", "Race Spearman", "Winner hit rate", "Podium overlap"])

CLASSIFIER_RESULTS = pd.DataFrame([
    ("Win", "Logistic Regression  <- CHOSEN", 0.938, 0.116),
    ("Win", "Random Forest (leaf 25)", 0.941, 0.114),
    ("Podium", "Logistic Regression", 0.934, 0.221),
    ("Podium", "Random Forest (leaf 25)  <- CHOSEN", 0.941, 0.212),
], columns=["Target", "Model", "ROC-AUC", "Log loss"])


@st.cache_data(show_spinner=False)
def load_eval(filename):
    path = PROCESSED / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if not set(NEEDED).issubset(df.columns):
        return None
    df = df[NEEDED].copy()
    df["pred_rank"] = df.groupby("raceId")["pred_finish"].rank(method="first")
    df["base_rank"] = df.groupby("raceId")["baseline_pred"].rank(method="first")
    return df


def position_metrics(df):
    rows = []
    for label, pred_col, rank_col in [("Baseline (finish = grid)", "baseline_pred", "base_rank"),
                                      ("Our model", "pred_finish", "pred_rank")]:
        error = (df["finish_position"] - df[pred_col]).abs()
        rank_error = (df["finish_position"] - df[rank_col]).abs()
        rows.append({
            "Method": label,
            "Average error (positions)": round(error.mean(), 3),
            "Exactly right %": round((rank_error == 0).mean() * 100, 1),
            "Within 1 place %": round((rank_error <= 1).mean() * 100, 1),
            "Within 3 places %": round((rank_error <= 3).mean() * 100, 1),
        })
    return pd.DataFrame(rows)


def top_k_flags(df, prob_col, k):
    ranks = df.groupby("raceId")[prob_col].rank(method="first", ascending=False)
    return (ranks <= k).astype(int)


def class_metrics(y_true, y_pred):
    return {
        "Accuracy %": round(accuracy_score(y_true, y_pred) * 100, 1),
        "Precision %": round(precision_score(y_true, y_pred, zero_division=0) * 100, 1),
        "Recall %": round(recall_score(y_true, y_pred, zero_division=0) * 100, 1),
        "F1 %": round(f1_score(y_true, y_pred, zero_division=0) * 100, 1),
    }


def show_metrics(metrics):
    st.dataframe(pd.DataFrame({"Metric": list(metrics.keys()),
                               "Value": list(metrics.values())}), hide_index=True)


def draw_confusion(y_true, y_pred, label, title):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    names = [f"Not {label}", label]
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=names, yticklabels=names)
    ax.set_xlabel("What the model said")
    ax.set_ylabel("What really happened")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)
    tn, fp, fn, tp = matrix.ravel()
    return tn, fp, fn, tp


def classification_block(df, target_col, prob_col, k, label, title):
    pred = top_k_flags(df, prob_col, k)
    y_true = df[target_col]
    left, right = st.columns(2)
    with left:
        tn, fp, fn, tp = draw_confusion(y_true, pred, label, title)
    with right:
        metrics = class_metrics(y_true, pred)
        metrics["ROC-AUC (probabilities)"] = round(roc_auc_score(y_true, df[prob_col]), 3)
        metrics["Log loss (probabilities)"] = round(log_loss(y_true, df[prob_col]), 3)
        show_metrics(metrics)
    n_races = df["raceId"].nunique()
    always_no = (1 - y_true.mean()) * 100
    st.write(f"In {n_races} races the model named {k} driver(s) per race as the likely {label}. "
             f"It was right **{tp}** times, wrong **{fp}** times, and missed **{fn}**.")
    st.caption(f"Reading accuracy: always guessing 'No' would already score {always_no:.1f}%, "
               f"so accuracy alone is misleading here. Look at precision and recall.")


def winner_hits(df):
    model_hits, front_hits = 0, 0
    for _, race in df.groupby("raceId"):
        model_hits += int(race.loc[race["p_win"].idxmax(), "is_win"])
        front_hits += int(race.loc[race["grid_clean"].idxmin(), "is_win"])
    return model_hits, front_hits


def evaluation_page():
    st.header("Model evaluation: how good is it?")
    choice = st.radio("Evaluate on", list(SOURCES), horizontal=True)
    df = load_eval(SOURCES[choice])
    if df is None:
        st.warning("Evaluation file not found. Run `python src\\final_evaluation.py` (test set) "
                   "or `python src\\backtest_2026.py` (backtest) first.")
        return

    n_races = df["raceId"].nunique()
    c1, c2 = st.columns(2)
    c1.metric("Races evaluated", n_races)
    c2.metric("Driver-race rows", len(df))
    st.caption("None of these races were used to train the model that predicted them.")

    st.subheader("1. Finishing-position accuracy")
    table = position_metrics(df)
    st.dataframe(table, hide_index=True)
    st.caption("Average error = how many places we are off on average (lower is better). "
               "'Exactly right' compares the predicted order with the real order. "
               "Guessing at random would be exactly right about 5% of the time.")

    st.subheader("2. Who wins the race? (confusion matrix)")
    classification_block(df, "is_win", "p_win", 1, "winner", "Race winner")
    model_hits, front_hits = winner_hits(df)
    if model_hits > front_hits:
        verdict = "better than"
    elif model_hits == front_hits:
        verdict = "the same as"
    else:
        verdict = "slightly lower than"
    st.info(f"**Winner picked correctly:** model {model_hits} of {n_races}. "
            f"Simple rule 'the front-row starter wins': {front_hits} of {n_races}. "
            f"So the model's winner picks are {verdict} the simple rule on these races. "
            f"The model's value is the full predicted order and the probabilities.")

    st.subheader("3. Who reaches the podium? (confusion matrix)")
    classification_block(df, "is_podium", "p_podium", 3, "podium finisher", "Podium (top 3)")

    with st.expander("Try a probability threshold instead of 'top pick per race'"):
        target = st.selectbox("Predict", ["Win", "Podium"])
        cut = st.slider("Say 'yes' when the probability is at least", 0.05, 0.95, 0.30, 0.05)
        target_col, prob_col = ("is_win", "p_win") if target == "Win" else ("is_podium", "p_podium")
        pred = (df[prob_col] >= cut).astype(int)
        a, b = st.columns(2)
        with a:
            draw_confusion(df[target_col], pred, target.lower(), f"{target}, threshold {cut:.2f}")
        with b:
            show_metrics(class_metrics(df[target_col], pred))
        st.caption("Raising the threshold gives fewer, safer predictions (higher precision, lower recall).")

    st.subheader("4. Are the win probabilities honest? (calibration)")
    calib = df.copy()
    calib["Probability range"] = pd.cut(calib["p_win"], bins=[0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0],
                                        include_lowest=True)
    calib_table = calib.groupby("Probability range", observed=True).agg(
        Drivers=("is_win", "size"), Average_predicted=("p_win", "mean"),
        Actual_win_rate=("is_win", "mean")).round(3).reset_index()
    calib_table["Probability range"] = calib_table["Probability range"].astype(str)
    st.dataframe(calib_table, hide_index=True)
    st.caption("If the model is honest, 'Average_predicted' is close to 'Actual_win_rate'. "
               "Groups with few drivers are noisy.")

    st.subheader("5. Models we compared (walk-forward validation, 2018-2022)")
    st.dataframe(REGRESSION_RESULTS, hide_index=True)
    st.bar_chart(REGRESSION_RESULTS.set_index("Model")["MAE"])
    st.caption("Lower MAE is better. Recorded from our own experiment runs.")
    st.dataframe(CLASSIFIER_RESULTS, hide_index=True)
    st.caption("For the classifiers: higher ROC-AUC and lower log loss are better. "
               "A model that ignores the data and predicts the base rate has log loss "
               "0.199 for winning and 0.423 for the podium.")

    st.subheader("Limitations")
    st.write("- Retirements, safety cars, weather and penalties cannot be predicted before the race.\n"
             "- 2026 uses new regulations, but most training data is from earlier rules.\n"
             "- Test sets are small, so small differences between methods can be luck.")


