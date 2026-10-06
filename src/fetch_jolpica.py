import time
import requests
import pandas as pd
from load_data import RAW_DIR

BASE_URL = "https://api.jolpi.ca/ergast/f1"
SEASONS = [2025, 2026]
SPRINT_COLUMNS = ["year", "round", "driverRef", "constructorRef",
                  "sprint_position", "sprint_points"]


def get_json(url, params):
    for attempt in range(6):
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 429:   # rate limited: wait, then retry
            time.sleep(5 * (attempt + 1))
            continue
        response.raise_for_status()
        return response.json()["MRData"]
    raise RuntimeError("Rate limit hit too many times. Wait a few minutes and run again.")


def download_races(season, resource):
    offset, races = 0, []
    while True:
        data = get_json(f"{BASE_URL}/{season}/{resource}/", {"limit": 100, "offset": offset})
        races.extend(data["RaceTable"]["Races"])
        offset += 100
        if offset >= int(data["total"]):
            break
        time.sleep(0.7)
    return races


def flatten_results(races):
    rows = []
    for race in races:
        for res in race["Results"]:
            driver = res["Driver"]
            rows.append({
                "year": int(race["season"]),
                "round": int(race["round"]),
                "race_name": race["raceName"],
                "date": race["date"],
                "circuitRef": race["Circuit"]["circuitId"],
                "driverRef": driver["driverId"],
                "driver_name": f"{driver['givenName']} {driver['familyName']}",
                "driver_code": driver.get("code"),
                "constructorRef": res["Constructor"]["constructorId"],
                "constructor_name": res["Constructor"]["name"],
                "grid": int(res["grid"]),
                "finish_position": int(res["position"]),
                "dnf": 0 if res["positionText"].isdigit() else 1,
                "points": float(res["points"]),
                "status": res.get("status"),
                "race_time": res.get("Time", {}).get("time"),
                "fastest_lap_rank": res.get("FastestLap", {}).get("rank"),
            })
    return rows


def flatten_qualifying(races):
    rows = []
    for race in races:
        for q in race["QualifyingResults"]:
            rows.append({"year": int(race["season"]), "round": int(race["round"]),
                         "driverRef": q["Driver"]["driverId"],
                         "quali_position": int(q["position"])})
    return rows


def flatten_sprint(races):
    rows = []
    for race in races:
        for res in race["SprintResults"]:
            rows.append({"year": int(race["season"]), "round": int(race["round"]),
                         "driverRef": res["Driver"]["driverId"],
                         "constructorRef": res["Constructor"]["constructorId"],
                         "sprint_position": int(res["position"]),
                         "sprint_points": float(res["points"])})
    return rows


def session_time(block):
    if not block or "date" not in block:
        return None
    return f"{block['date']}T{block.get('time', '00:00:00Z')}"


def flatten_schedule(races):
    rows = []
    for r in races:
        location = r["Circuit"]["Location"]
        rows.append({
            "year": int(r["season"]), "round": int(r["round"]),
            "race_name": r["raceName"], "date": r["date"],
            "circuitRef": r["Circuit"]["circuitId"],
            "circuit_name": r["Circuit"]["circuitName"],
            "locality": location["locality"], "country": location["country"],
            "lat": float(location["lat"]), "lon": float(location["long"]),
            "circuit_url": r["Circuit"].get("url"),
            "fp1": session_time(r.get("FirstPractice")),
            "fp2": session_time(r.get("SecondPractice")),
            "fp3": session_time(r.get("ThirdPractice")),
            "sprint_quali": session_time(r.get("SprintQualifying") or r.get("SprintShootout")),
            "sprint": session_time(r.get("Sprint")),
            "qualifying": session_time(r.get("Qualifying")),
            "race": session_time({"date": r["date"], "time": r.get("time", "00:00:00Z")}),
        })
    return rows


def save_merged(path, new_df, seasons):
    """Replace only the refreshed seasons; keep other seasons already on disk."""
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["year"].isin(seasons)]
        new_df = pd.concat([old, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)
    return new_df


def refresh(seasons=SEASONS):
    results_rows, quali_rows, sprint_rows, schedule_rows = [], [], [], []
    for season in seasons:
        results_rows += flatten_results(download_races(season, "results"))
        time.sleep(0.7)
        quali_rows += flatten_qualifying(download_races(season, "qualifying"))
        time.sleep(0.7)
        sprint_rows += flatten_sprint(download_races(season, "sprint"))
        time.sleep(0.7)
        schedule_rows += flatten_schedule(download_races(season, "races"))
        time.sleep(0.7)

    key = ["year", "round", "driverRef"]
    results = save_merged(RAW_DIR / "jolpica_results.csv",
                          pd.DataFrame(results_rows).drop_duplicates(subset=key), seasons)
    quali = save_merged(RAW_DIR / "jolpica_qualifying.csv",
                        pd.DataFrame(quali_rows).drop_duplicates(subset=key), seasons)
    sprint = save_merged(RAW_DIR / "jolpica_sprint.csv",
                         pd.DataFrame(sprint_rows, columns=SPRINT_COLUMNS).drop_duplicates(subset=key),
                         seasons)
    schedule = save_merged(RAW_DIR / "jolpica_schedule.csv",
                           pd.DataFrame(schedule_rows).drop_duplicates(subset=["year", "round"]),
                           seasons)
    return results, quali, schedule, sprint


def check_standings(season, results, sprint):
    """Compare our own points totals with the official driver standings."""
    data = get_json(f"{BASE_URL}/{season}/driverStandings/", {"limit": 100})
    lists = data["StandingsTable"]["StandingsLists"]
    if not lists:
        print(f"Standings check {season}: no official table returned")
        return
    official = {s["Driver"]["driverId"]: float(s["points"]) for s in lists[0]["DriverStandings"]}
    race_pts = results[results["year"] == season][["driverRef", "points"]]
    race_pts = race_pts.rename(columns={"points": "pts"})
    sprint_pts = sprint[sprint["year"] == season][["driverRef", "sprint_points"]]
    sprint_pts = sprint_pts.rename(columns={"sprint_points": "pts"})
    mine = pd.concat([race_pts, sprint_pts]).groupby("driverRef")["pts"].sum()
    mismatches = [(d, official.get(d, 0), mine.get(d, 0))
                  for d in set(official) | set(mine.index)
                  if official.get(d, 0) != mine.get(d, 0)]
    print(f"Standings check {season}:",
          "OK, matches the official table" if not mismatches
          else f"MISMATCH (driver, official, ours): {mismatches}")


if __name__ == "__main__":
    results, quali, schedule, sprint = refresh(SEASONS)
    print("Saved 4 files to data/raw/")
    for season in SEASONS:
        rounds = sorted(results.loc[results["year"] == season, "round"].unique())
        print(f"{season}: {len(rounds)} races with results, last round = "
              f"{rounds[-1] if rounds else None}")
        check_standings(season, results, sprint)
        time.sleep(0.7)
    print("Last round with qualifying data:", quali.loc[quali["year"] == SEASONS[-1], "round"].max())