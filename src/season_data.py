import pandas as pd
from load_data import RAW_DIR
from clean_data import PROCESSED_DIR


def pretty(ref):
    return str(ref).replace("_", " ").title()


def load_jolpica():
    return {
        "results": pd.read_csv(RAW_DIR / "jolpica_results.csv"),
        "sprint": pd.read_csv(RAW_DIR / "jolpica_sprint.csv"),
        "quali": pd.read_csv(RAW_DIR / "jolpica_qualifying.csv"),
        "schedule": pd.read_csv(RAW_DIR / "jolpica_schedule.csv", parse_dates=["date"]),
    }


def load_names():
    drivers = pd.read_csv(RAW_DIR / "drivers.csv", na_values=["\\N"])
    teams = pd.read_csv(RAW_DIR / "constructors.csv", na_values=["\\N"])
    driver_names = dict(zip(drivers["driverRef"], drivers["forename"] + " " + drivers["surname"]))
    team_names = dict(zip(teams["constructorRef"], teams["name"]))
    results = pd.read_csv(RAW_DIR / "jolpica_results.csv")
    named = results.dropna(subset=["driver_name", "constructor_name"])
    driver_names.update(dict(zip(named["driverRef"], named["driver_name"])))
    team_names.update(dict(zip(named["constructorRef"], named["constructor_name"])))
    return driver_names, team_names


def _season_rows(data, season, upto_round):
    results = data["results"][data["results"]["year"] == season]
    sprint = data["sprint"][data["sprint"]["year"] == season]
    if upto_round is not None:
        results = results[results["round"] <= upto_round]
        sprint = sprint[sprint["round"] <= upto_round]
    return results, sprint


def driver_standings(data, season, upto_round=None):
    results, sprint = _season_rows(data, season, upto_round)
    points = pd.concat([
        results[["driverRef", "points"]].rename(columns={"points": "pts"}),
        sprint[["driverRef", "sprint_points"]].rename(columns={"sprint_points": "pts"}),
    ]).groupby("driverRef")["pts"].sum()
    wins = results[results["finish_position"] == 1].groupby("driverRef").size().rename("wins")
    team = results.sort_values("round").groupby("driverRef")["constructorRef"].last()
    table = pd.DataFrame({"points": points}).join(wins).join(team)
    table["wins"] = table["wins"].fillna(0).astype(int)
    table = table.sort_values(["points", "wins"], ascending=False).reset_index()
    table.insert(0, "position", range(1, len(table) + 1))
    return table


def constructor_standings(data, season, upto_round=None):
    results, sprint = _season_rows(data, season, upto_round)
    points = pd.concat([
        results[["constructorRef", "points"]].rename(columns={"points": "pts"}),
        sprint[["constructorRef", "sprint_points"]].rename(columns={"sprint_points": "pts"}),
    ]).groupby("constructorRef")["pts"].sum()
    wins = results[results["finish_position"] == 1].groupby("constructorRef").size().rename("wins")
    table = pd.DataFrame({"points": points}).join(wins)
    table["wins"] = table["wins"].fillna(0).astype(int)
    table = table.sort_values(["points", "wins"], ascending=False).reset_index()
    table.insert(0, "position", range(1, len(table) + 1))
    return table


def weekend_points(data, season, rnd):
    results, sprint = data["results"], data["sprint"]
    race = results[(results["year"] == season) & (results["round"] == rnd)]
    race = race[["driverRef", "constructorRef", "points"]].rename(columns={"points": "race_points"})
    spr = sprint[(sprint["year"] == season) & (sprint["round"] == rnd)]
    spr = spr[["driverRef", "sprint_points"]]
    merged = race.merge(spr, on="driverRef", how="left")
    merged["sprint_points"] = merged["sprint_points"].fillna(0)
    merged["total"] = merged["race_points"] + merged["sprint_points"]
    return merged


def season_overview(data, season, driver_names, team_names):
    results, quali = data["results"], data["quali"]
    rows = []
    for rnd, race in results[results["year"] == season].groupby("round"):
        winner = race[race["finish_position"] == 1].iloc[0]
        pole = quali[(quali["year"] == season) & (quali["round"] == rnd)
                     & (quali["quali_position"] == 1)]
        pole_name = driver_names.get(pole["driverRef"].iloc[0], pretty(pole["driverRef"].iloc[0])) \
            if not pole.empty else "n/a"
        rows.append({
            "Round": int(rnd),
            "Grand Prix": race["race_name"].iloc[0],
            "Date": race["date"].iloc[0],
            "Winner": driver_names.get(winner["driverRef"], pretty(winner["driverRef"])),
            "Team": team_names.get(winner["constructorRef"], pretty(winner["constructorRef"])),
            "Pole position": pole_name,
        })
    return pd.DataFrame(rows)


def past_winners(circuit_ref, data, driver_names, team_names):
    cols = ["year", "driverRef", "constructorRef", "circuitRef", "finish_position"]
    old = pd.read_csv(PROCESSED_DIR / "base_table.csv", usecols=cols)
    everything = pd.concat([old, data["results"][cols]])
    wins = everything[(everything["circuitRef"] == circuit_ref)
                      & (everything["finish_position"] == 1)].sort_values("year", ascending=False)
    return pd.DataFrame({
        "Year": wins["year"],
        "Winner": wins["driverRef"].map(lambda r: driver_names.get(r, pretty(r))),
        "Team": wins["constructorRef"].map(lambda r: team_names.get(r, pretty(r))),
    })