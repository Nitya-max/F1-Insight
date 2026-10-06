from datetime import datetime
import pandas as pd
import requests
import streamlit as st

from forecast_common import find_next_race
from season_data import (load_jolpica, load_names, driver_standings, constructor_standings,
                         weekend_points, season_overview, past_winners, pretty)

LOCAL_TZ = datetime.now().astimezone().tzinfo
TZ_NAME = datetime.now().astimezone().tzname()
FRIENDLY_ERROR = ("Some data is missing. Run `python src\\fetch_jolpica.py` once to download "
                  "the full dataset, then reload this page.")


@st.cache_data(ttl=86400, show_spinner=False)
def track_info(circuit_url):
    """Lead image and short description from the circuit's Wikipedia page."""
    try:
        if not isinstance(circuit_url, str) or "/wiki/" not in circuit_url:
            return None, None
        title = circuit_url.rstrip("/").split("/wiki/")[-1]
        response = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            headers={"User-Agent": "F1-Insight-college-project/1.0"}, timeout=15)
        if response.status_code != 200:
            return None, None
        data = response.json()
        original = data.get("originalimage", {}).get("source")
        thumb = data.get("thumbnail", {}).get("source")
        image = thumb if (original is None or original.lower().endswith(".svg")) else original
        return image, data.get("extract")
    except Exception:
        return None, None


def countdown_text(when):
    seconds = (when - pd.Timestamp.now(tz="UTC")).total_seconds()
    if seconds <= 0:
        return "Already started"
    days, rest = divmod(int(seconds), 86400)
    hours, rest = divmod(rest, 3600)
    return f"{days} d {hours} h {rest // 60} min"


def last_three(results, season, driver):
    rows = results[(results["year"] == season) & (results["driverRef"] == driver)]
    rows = rows.sort_values("round").tail(3)
    text = " · ".join("DNF" if d == 1 else f"P{p}"
                      for p, d in zip(rows["finish_position"], rows["dnf"]))
    return text or "n/a"


# ---------------------------------------------------------------- COMPLETED RACES
def _completed_page():
    data = load_jolpica()
    driver_names, team_names = load_names()
    dn = lambda ref: driver_names.get(ref, pretty(ref))
    tn = lambda ref: team_names.get(ref, pretty(ref))
    results = data["results"]

    st.header("Completed Grands Prix")
    season = st.selectbox("Season", sorted(results["year"].unique(), reverse=True), key="done_season")
    overview = season_overview(data, season, driver_names, team_names)
    st.subheader(f"{season}: {len(overview)} races completed")
    st.dataframe(overview, hide_index=True)

    race_name = st.selectbox("Pick a Grand Prix to see the details",
                             overview["Grand Prix"].tolist()[::-1], key="done_race")
    rnd = int(overview.loc[overview["Grand Prix"] == race_name, "Round"].iloc[0])
    race = results[(results["year"] == season) & (results["round"] == rnd)].sort_values("finish_position")
    winner = race.iloc[0]

    pole_row = overview.loc[overview["Round"] == rnd, "Pole position"].iloc[0]
    fastest = race[pd.to_numeric(race["fastest_lap_rank"], errors="coerce") == 1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Winner", dn(winner["driverRef"]))
    c2.metric("Winning team", tn(winner["constructorRef"]))
    c3.metric("Pole position", pole_row)
    c4.metric("Fastest lap", dn(fastest["driverRef"].iloc[0]) if not fastest.empty else "n/a")

    start = "the pit lane" if winner["grid"] == 0 else f"P{int(winner['grid'])}"
    st.write(f"**{dn(winner['driverRef'])}** won the {race_name} for "
             f"**{tn(winner['constructorRef'])}**, starting from {start}.")

    st.markdown("#### Full results")
    st.dataframe(pd.DataFrame({
        "Finish": race["finish_position"],
        "Driver": race["driverRef"].map(dn),
        "Team": race["constructorRef"].map(tn),
        "Started": race["grid"].apply(lambda g: "Pit lane" if g == 0 else str(int(g))),
        "Points": race["points"].map(lambda p: f"{p:g}"),
        "Race time / gap": race["race_time"].fillna(""),
        "Result": race["status"],
    }), hide_index=True)

    st.markdown("#### Points scored this weekend (race + sprint, if any)")
    points = weekend_points(data, season, rnd).sort_values("total", ascending=False)
    left, right = st.columns(2)
    with left:
        drivers = pd.DataFrame({"Driver": points["driverRef"].map(dn),
                                "Team": points["constructorRef"].map(tn),
                                "Points": points["total"]})
        drivers = drivers[drivers["Points"] > 0]
        st.dataframe(drivers, hide_index=True, column_config={
            "Points": st.column_config.ProgressColumn(
                "Points", min_value=0, max_value=float(max(drivers["Points"].max(), 1)),
                format="%g")})
    with right:
        teams = points.groupby("constructorRef")["total"].sum().sort_values(ascending=False)
        teams = teams[teams > 0]
        st.dataframe(pd.DataFrame({"Team": teams.index.map(tn), "Points": teams.values}),
                     hide_index=True)

    st.markdown(f"#### Championship standings after round {rnd}")
    a, b = st.columns(2)
    with a:
        ds = driver_standings(data, season, upto_round=rnd)
        st.dataframe(pd.DataFrame({
            "Pos": ds["position"], "Driver": ds["driverRef"].map(dn),
            "Team": ds["constructorRef"].map(tn), "Points": ds["points"], "Wins": ds["wins"]}),
            hide_index=True)
    with b:
        cs = constructor_standings(data, season, upto_round=rnd)
        st.dataframe(pd.DataFrame({
            "Pos": cs["position"], "Team": cs["constructorRef"].map(tn),
            "Points": cs["points"], "Wins": cs["wins"]}), hide_index=True)


def completed_page():
    try:
        _completed_page()
    except (FileNotFoundError, KeyError) as error:
        st.warning(f"{FRIENDLY_ERROR} (details: {error})")


# ---------------------------------------------------------------- UPCOMING RACE
def _upcoming_page():
    data = load_jolpica()
    driver_names, team_names = load_names()
    dn = lambda ref: driver_names.get(ref, pretty(ref))
    tn = lambda ref: team_names.get(ref, pretty(ref))
    schedule, results, quali = data["schedule"], data["results"], data["quali"]

    season = int(schedule["year"].max())
    try:
        race = find_next_race(schedule, results, season)
    except ValueError as error:
        st.info(str(error))
        return
    rnd = int(race["round"])

    st.header(f"Next race: {race['race_name']}")
    st.caption(f"Round {rnd} of {season}  |  {race['circuit_name']}, {race['locality']}, {race['country']}")

    if pd.notna(race.get("race")):
        race_when = pd.to_datetime(race["race"], utc=True)
        c1, c2 = st.columns(2)
        c1.metric("Race starts in", countdown_text(race_when))
        c2.metric(f"Race start ({TZ_NAME})",
                  race_when.tz_convert(LOCAL_TZ).strftime("%a %d %b, %I:%M %p"))

    st.markdown("#### Weekend schedule")
    sessions = [("Practice 1", "fp1"), ("Practice 2", "fp2"), ("Practice 3", "fp3"),
                ("Sprint Qualifying", "sprint_quali"), ("Sprint", "sprint"),
                ("Qualifying", "qualifying"), ("Race", "race")]
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for label, column in sessions:
        value = race.get(column)
        if value is None or pd.isna(value):
            continue
        when = pd.to_datetime(value, utc=True)
        rows.append({"Session": label,
                     f"Your time ({TZ_NAME})": when.tz_convert(LOCAL_TZ).strftime("%a %d %b, %I:%M %p"),
                     "UTC": when.strftime("%a %d %b, %H:%M"),
                     "Status": "Upcoming" if now < when else "Started / done",
                     "_when": when})
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("_when").drop(columns=["_when"]),
                     hide_index=True)
    else:
        st.caption("Session times are not published yet.")

    st.markdown("#### The track")
    image, about = track_info(race.get("circuit_url"))
    left, right = st.columns(2)
    with left:
        if image:
            st.image(image, caption=f"{race['circuit_name']} (image: Wikipedia)")
        else:
            st.caption("No track image found for this circuit.")
        if about:
            st.caption(f"{about}  (Source: Wikipedia)")
    with right:
        st.map(pd.DataFrame({"lat": [race["lat"]], "lon": [race["lon"]]}), zoom=11)

    st.markdown("#### Drivers and teams on the grid (current championship)")
    ordered = results.sort_values(["year", "round"])
    last = ordered.iloc[-1]
    roster = ordered[(ordered["year"] == last["year"]) & (ordered["round"] == last["round"])]
    roster = roster[["driverRef", "constructorRef"]]
    q = quali[(quali["year"] == season) & (quali["round"] == rnd)][["driverRef", "quali_position"]]
    table = roster.merge(driver_standings(data, season)[["driverRef", "position", "points", "wins"]],
                         on="driverRef", how="left").merge(q, on="driverRef", how="left")
    table["last3"] = table["driverRef"].map(lambda d: last_three(results, season, d))
    table = table.sort_values("position")
    view = pd.DataFrame({
        "Championship position": table["position"],
        "Driver": table["driverRef"].map(dn),
        "Team": table["constructorRef"].map(tn),
        "Points": table["points"].fillna(0),
        "Wins this season": table["wins"].fillna(0).astype(int),
        "Last 3 races": table["last3"],
    })
    if table["quali_position"].notna().any():
        view["Qualified"] = table["quali_position"]
    st.dataframe(view, hide_index=True)
    st.caption("Driver list is taken from the latest completed race. "
               "Points include sprint races.")

    st.markdown("#### Team championship")
    cs = constructor_standings(data, season)
    st.dataframe(pd.DataFrame({"Pos": cs["position"], "Team": cs["constructorRef"].map(tn),
                               "Points": cs["points"], "Wins": cs["wins"]}), hide_index=True)

    st.markdown("#### Past winners at this circuit (since 2014)")
    winners = past_winners(race["circuitRef"], data, driver_names, team_names)
    if winners.empty:
        st.caption("No earlier races at this circuit in our data.")
    else:
        st.dataframe(winners, hide_index=True)


def upcoming_page():
    try:
        _upcoming_page()
    except (FileNotFoundError, KeyError) as error:
        st.warning(f"{FRIENDLY_ERROR} (details: {error})")