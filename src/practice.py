import time
import requests
import pandas as pd

OPENF1 = "https://api.openf1.org/v1"


def _get(endpoint, params):
    response = requests.get(f"{OPENF1}/{endpoint}", params=params, timeout=30)
    if response.status_code == 404:      # OpenF1 answers 404 when there is no data yet
        return []
    response.raise_for_status()
    return response.json()


def practice_pace(year, race_date):
    """Best lap per driver for each finished practice session of this race weekend."""
    sessions = pd.DataFrame(_get("sessions", {"year": year}))
    if sessions.empty:
        return {}
    sessions["date_end"] = pd.to_datetime(sessions["date_end"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    race_day = pd.Timestamp(race_date).tz_localize("UTC")

    is_practice = sessions["session_name"].astype(str).str.startswith("Practice")
    weekend = sessions[is_practice
                       & (sessions["date_end"] >= race_day - pd.Timedelta(days=4))
                       & (sessions["date_end"] < now)]

    tables = {}
    for _, session in weekend.sort_values("date_end").iterrows():
        key = int(session["session_key"])
        laps = pd.DataFrame(_get("laps", {"session_key": key}))
        time.sleep(0.5)
        drivers = pd.DataFrame(_get("drivers", {"session_key": key}))
        time.sleep(0.5)
        if laps.empty or drivers.empty:
            continue
        laps = laps[laps["lap_duration"].notna()]
        best = laps.groupby("driver_number")["lap_duration"].min().reset_index()
        info = drivers[["driver_number", "name_acronym", "team_name"]].drop_duplicates("driver_number")
        table = best.merge(info, on="driver_number", how="left").sort_values("lap_duration")
        fastest = table["lap_duration"].iloc[0]
        tables[session["session_name"]] = pd.DataFrame({
            "Driver": table["name_acronym"],
            "Team": table["team_name"],
            "Best lap (s)": table["lap_duration"].round(3),
            "Gap to fastest %": ((table["lap_duration"] / fastest - 1) * 100).round(2),
        })
    return tables