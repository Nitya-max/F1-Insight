import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from load_data import RAW_DIR
from features import TARGET
from fetch_jolpica import refresh
from forecast_common import (build_model_table, fit_final_models, predict_all,
                             find_next_race, REDUCED_FEATURES, PREQUALI_FEATURES)
from practice import practice_pace
from weekend_pages import completed_page, upcoming_page
from evaluation_pages import evaluation_page

PROCESSED = ROOT / "data" / "processed"
PRED_DIR = ROOT / "predictions"

st.set_page_config(page_title="F1 Insight 2026", layout="wide")
# Reload the page every 10 minutes so new sessions are picked up while it stays open
components.html("<script>setTimeout(function(){window.parent.location.reload();}, 600000);</script>",
                height=0)


@st.cache_data(ttl=600, show_spinner="Checking for new session data...")
def refresh_data(season):
    stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    try:
        refresh([season])
        return "ok", stamp
    except Exception as error:
        return f"refresh failed, using stored data: {error}", stamp


@st.cache_data(ttl=600, show_spinner="Reading practice sessions...")
def get_practice(season, race_date):
    try:
        return practice_pace(season, race_date), None
    except Exception as error:
        return {}, str(error)


@st.cache_resource(show_spinner="Training models on the latest data...")
def compute_prediction(stage, signature):
    features = REDUCED_FEATURES if stage == "final" else PREQUALI_FEATURES
    table = build_model_table(include_upcoming=True, stage=stage)
    upcoming = table[table[TARGET].isna()]
    train = table[table[TARGET].notna()]
    fitted = fit_final_models(train, features)
    result = predict_all(fitted, upcoming, features)
    result["win_chance"] = result["p_win"] / result["p_win"].sum()
    result["predicted_rank"] = result["pred_finish"].rank(method="first").astype(int)
    meta = {"rows": len(train), "latest": str(train["date"].max().date())}
    return result.sort_values("predicted_rank"), meta


def save_snapshot(result, race, stage):
    """Saved once; never overwritten, so it proves what was predicted beforehand."""
    PRED_DIR.mkdir(exist_ok=True)
    slug = race["race_name"].replace(" ", "_")
    path = PRED_DIR / f"{int(race['year'])}_round{int(race['round']):02d}_{slug}_{stage}.csv"
    if not path.exists():
        out = result.copy()
        out["generated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        out.to_csv(path, index=False)
    return path


def winner_summary(df):
    rows = []
    for race_id, race in df.groupby("raceId"):
        actual = race.loc[race["finish_position"] == 1, "driverRef"]
        actual = actual.iloc[0] if len(actual) else "unknown"
        pick = race.sort_values("p_win", ascending=False, kind="mergesort").iloc[0]["driverRef"]
        front = race.sort_values("grid_clean", kind="mergesort").iloc[0]["driverRef"]
        rows.append({"year": race["year"].iloc[0], "raceId": race_id,
                     "race": race["race_name"].iloc[0], "actual_winner": actual,
                     "model_pick": pick, "model_hit": pick == actual,
                     "front_row_pick": front, "front_row_hit": front == actual})
    return pd.DataFrame(rows).sort_values(["year", "raceId"]).reset_index(drop=True)


def show_hits(summary):
    shown = summary.copy()
    shown["model_hit"] = shown["model_hit"].map({True: "✅", False: "❌"})
    shown["front_row_hit"] = shown["front_row_hit"].map({True: "✅", False: "❌"})
    st.dataframe(shown.drop(columns=["raceId"]), hide_index=True)


# ------------------------------------------------------------------ LIVE TAB
def live_tab():
    st.header("Live race-weekend prediction")
    status, fetched_at = refresh_data(pd.Timestamp.today().year)
    left, right = st.columns([4, 1])
    left.caption(f"Data checked {fetched_at} ({status}). Re-checks every 10 minutes.")
    if right.button("Refresh now"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    schedule = pd.read_csv(RAW_DIR / "jolpica_schedule.csv", parse_dates=["date"])
    results = pd.read_csv(RAW_DIR / "jolpica_results.csv")
    quali = pd.read_csv(RAW_DIR / "jolpica_qualifying.csv")
    season = int(schedule["year"].max())

    try:
        race = find_next_race(schedule, results, season)
    except ValueError as error:
        st.info(str(error))
        return

    has_quali = not quali[(quali["year"] == season) & (quali["round"] == race["round"])].empty
    stage = "final" if has_quali else "pre"
    st.subheader(f"Next race: {race['race_name']}  ({race['date'].date()}, round {int(race['round'])})")
    if stage == "final":
        st.success("FINAL prediction: qualifying is complete, so the starting grid is used.")
    else:
        st.warning("PROVISIONAL prediction: qualifying is not published yet. This model has no "
                   "grid information and is less reliable. It upgrades automatically after qualifying.")

    pace, pace_error = get_practice(season, str(race["date"].date()))
    st.markdown("#### Practice pace (shown for information, NOT used by the model)")
    if pace_error:
        st.caption(f"Practice data unavailable: {pace_error}")
    elif not pace:
        st.caption("No finished practice session found for this weekend yet.")
    else:
        for name, table in pace.items():
            with st.expander(name):
                st.dataframe(table, hide_index=True)

    signature = f"{len(results)}-{len(quali)}-{stage}"
    try:
        result, meta = compute_prediction(stage, signature)
    except Exception as error:
        st.error(f"Could not build the prediction: {error}")
        return

    if stage == "pre":
        ranked = result.sort_values("pred_finish")
        top = ranked.iloc[0]
        podium = ranked.head(3)["driverRef"].tolist()
        win_label, win_value = "Method", "Lowest predicted finish"
    else:
        top = result.sort_values("p_win", ascending=False).iloc[0]
        podium = result.sort_values("p_podium", ascending=False).head(3)["driverRef"].tolist()
        win_label, win_value = "Win chance (normalised)", f"{top['win_chance'] * 100:.1f}%"
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted winner", top["driverRef"])
    c2.metric(win_label, win_value)
    c3.metric("Predicted podium", ", ".join(podium))
    st.caption(f"Trained on {meta['rows']} driver-races; latest race in training data: {meta['latest']}")

    view = pd.DataFrame({
        "Predicted rank": result["predicted_rank"], "Driver": result["driverRef"],
        "Team": result["constructorRef"],
        "Grid": result["grid_clean"].round(0),
        "Predicted finish": result["pred_finish"].round(1),
        "Win chance %": (result["win_chance"] * 100).round(1),
        "Podium prob %": (result["p_podium"] * 100).round(1),
    })
    if stage == "pre":
        view = view.drop(columns=["Win chance %", "Podium prob %"])
    st.dataframe(view, hide_index=True)

    path = save_snapshot(result, race, stage)
    frozen = pd.read_csv(path)["generated_at"].iloc[0]
    st.caption(f"Snapshot file: predictions/{path.name} (first created {frozen}). "
               f"Commit it to Git before the race starts.")

    this_season = results[results["year"] == season]
    if not this_season.empty:
        last_round = int(this_season["round"].max())
        files = list(PRED_DIR.glob(f"{season}_round{last_round:02d}_*_final.csv"))
        if files:
            snap = pd.read_csv(files[0])
            predicted = snap.sort_values("p_win", ascending=False).iloc[0]["driverRef"]
            actual = this_season[(this_season["round"] == last_round)
                                 & (this_season["finish_position"] == 1)]["driverRef"].iloc[0]
            st.markdown("#### Last forecast vs reality")
            st.write(f"{snap['race_name'].iloc[0]}: predicted **{predicted}**, actual winner "
                     f"**{actual}** {'✅' if predicted == actual else '❌'} "
                     f"(prediction file created {snap['generated_at'].iloc[0]})")

# ------------------------------------------------------------------ OTHER TABS
def past_tab():
    path = PROCESSED / "test_predictions.csv"
    if not path.exists():
        st.info("Run python src\\final_evaluation.py first.")
        return
    test = pd.read_csv(path)
    summary = winner_summary(test)
    year = st.selectbox("Season", sorted(test["year"].unique()))
    race_name = st.selectbox("Grand Prix", summary[summary["year"] == year]["race"].tolist())
    chosen = test[(test["year"] == year) & (test["race_name"] == race_name)].copy()
    row = summary[(summary["year"] == year) & (summary["race"] == race_name)].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted winner", row["model_pick"])
    c2.metric("Actual winner", row["actual_winner"])
    c3.metric("Result", "✅ Correct" if row["model_hit"] else "❌ Missed")

    chosen["win_chance"] = chosen["p_win"] / chosen["p_win"].sum()
    chosen["rank"] = chosen["pred_finish"].rank(method="first").astype(int)
    st.dataframe(pd.DataFrame({
        "Driver": chosen["driverRef"], "Team": chosen["constructorRef"],
        "Grid": chosen["grid_clean"].astype(int), "Predicted finish": chosen["pred_finish"].round(1),
        "Predicted rank": chosen["rank"], "Win chance %": (chosen["win_chance"] * 100).round(1),
        "Podium prob %": (chosen["p_podium"] * 100).round(1),
        "Actual finish": chosen["finish_position"].astype(int),
    }).sort_values("Predicted rank"), hide_index=True)


def backtest_tab():
    path = PROCESSED / "backtest_2025_2026.csv"
    if not path.exists():
        st.info("Run python src\\backtest_2026.py first.")
        return
    summary = winner_summary(pd.read_csv(path))
    for year, part in summary.groupby("year"):
        st.write(f"**{year}:** model picked the winner in {int(part['model_hit'].sum())} of "
                 f"{len(part)} races; the front-row rule in {int(part['front_row_hit'].sum())}.")
    show_hits(summary)
    st.caption("Each race was predicted using only earlier races (walk-forward), after qualifying.")


#def report_tab():
    path = PROCESSED / "test_predictions.csv"
    if not path.exists():
        st.info("Run python src\\final_evaluation.py first.")
        return
    test = pd.read_csv(path)
    summary = winner_summary(test)
    n = len(summary)
    st.subheader(f"Winner prediction on {n} unseen races (2023-2024)")
    st.write(f"Win classifier: **{int(summary['model_hit'].sum())} of {n}** correct. "
             f"Front-row-starter rule: **{int(summary['front_row_hit'].sum())} of {n}**. "
             f"Random guessing among ~20 drivers: about 5%.")
    model_mae = (test["finish_position"] - test["pred_finish"]).abs().mean()
    base_mae = (test["finish_position"] - test["baseline_pred"]).abs().mean()
    c1, c2 = st.columns(2)
    c1.metric("Baseline MAE (finish = grid)", f"{base_mae:.3f}")
    c2.metric("Our model MAE", f"{model_mae:.3f}", delta=f"{model_mae - base_mae:.3f}",
              delta_color="inverse")

    st.subheader("Calibration: are the win probabilities honest?")
    calib = test.copy()
    calib["range"] = pd.cut(calib["p_win"], bins=[0, 0.05, 0.15, 0.3, 0.5, 0.75, 1.0],
                            include_lowest=True)
    table = calib.groupby("range", observed=True).agg(
        drivers=("is_win", "size"), average_predicted=("p_win", "mean"),
        actual_win_rate=("is_win", "mean")).round(3).reset_index()
    table["range"] = table["range"].astype(str)
    st.dataframe(table, hide_index=True)
    st.subheader("Race by race")
    show_hits(summary)
    st.subheader("Limitations")
    st.write("- Retirements cannot be predicted and cause most large errors.\n"
             "- No weather, tyre strategy or safety-car information.\n"
             "- Models were trained on past regulations; 2026 is a new rule set.\n"
             "- Small test sets: differences between methods can be luck.")

tab_live, tab_upcoming, tab_done, tab_eval, tab_past = st.tabs(
    ["2026 live weekend", "Upcoming race details", "Completed races",
     "Model evaluation", "Race-by-race checks"])
with tab_live:
    live_tab()
with tab_upcoming:
    upcoming_page()
with tab_done:
    completed_page()
with tab_eval:
    evaluation_page()
with tab_past:
    sub_test, sub_backtest = st.tabs(["2023-2024 test races", "2025-2026 backtest"])
    with sub_test:
        past_tab()
    with sub_backtest:
        backtest_tab()