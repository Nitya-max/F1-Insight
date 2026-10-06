import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from clean_data import PROCESSED_DIR
from load_data import PROJECT_ROOT

FIG_DIR = PROJECT_ROOT / "notebooks" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROCESSED_DIR / "base_table.csv", parse_dates=["date"])

print("Rows:", len(df))
print("Races:", df["raceId"].nunique(),
      "| Drivers:", df["driverRef"].nunique(),
      "| Constructors:", df["constructorRef"].nunique())

# 1. Who has grid == 0, and what happened to them?
pit = df[df["pit_lane_start"] == 1]
print("\nStatus of pit-lane / grid-0 rows (top 8):")
print(pit["status"].value_counts().head(8).to_string())

# 2. DNF rate
print("\nOverall not-classified rate:", round(df["dnf"].mean(), 3))
print("Not-classified rate by year:")
print(df.groupby("year")["dnf"].mean().round(3).to_string())

# 3. How predictive is the grid?
pole = df[df["grid_clean"] == 1]
print("\nPole sitters that won the race:", round((pole["finish_position"] == 1).mean(), 3))
print("Pole sitters that finished on the podium:",
      round((pole["finish_position"] <= 3).mean(), 3))
spearman = df[["grid_clean", "finish_position"]].corr(method="spearman")
print("\nSpearman(grid, finish):", round(spearman.iloc[0, 1], 3))

# 4. BASELINE: predict finish = grid
df["baseline_error"] = (df["finish_position"] - df["grid_clean"]).abs()
print("\nBaseline MAE (finish = grid), all rows:", round(df["baseline_error"].mean(), 3))
print("Baseline MAE by year:")
print(df.groupby("year")["baseline_error"].mean().round(3).to_string())

# ---------- Plots ----------
avg_finish = df.groupby("grid_clean")["finish_position"].mean()
avg_finish = avg_finish[avg_finish.index <= 20]
plt.figure(figsize=(7, 5))
plt.plot(avg_finish.index, avg_finish.values, marker="o", label="Average actual finish")
plt.plot([1, 20], [1, 20], linestyle="--", color="gray", label="finish = grid")
plt.xlabel("Grid position")
plt.ylabel("Average finishing position")
plt.title("Average finish by grid position (2014+)")
plt.legend()
plt.savefig(FIG_DIR / "01_finish_by_grid.png", dpi=120, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 4))
df.groupby("year")["dnf"].mean().plot(kind="bar")
plt.ylabel("Share not classified")
plt.title("Not-classified rate by season")
plt.savefig(FIG_DIR / "02_dnf_by_year.png", dpi=120, bbox_inches="tight")
plt.close()

latest_year = df["year"].max()
team_avg = (df[df["year"] == latest_year]
            .groupby("constructorRef")["finish_position"].mean().sort_values())
plt.figure(figsize=(7, 5))
team_avg.plot(kind="barh")
plt.gca().invert_yaxis()
plt.xlabel("Average finishing position")
plt.title(f"Constructor average finish, {latest_year}")
plt.savefig(FIG_DIR / "03_constructor_avg_finish.png", dpi=120, bbox_inches="tight")
plt.close()

subset = df[(df["grid_clean"] <= 20) & (df["finish_position"] <= 20)]
table = pd.crosstab(subset["grid_clean"], subset["finish_position"], normalize="index")
plt.figure(figsize=(9, 7))
sns.heatmap(table, cmap="Blues")
plt.xlabel("Finishing position")
plt.ylabel("Grid position")
plt.title("Where do drivers finish, given their grid slot? (row-normalised)")
plt.savefig(FIG_DIR / "04_grid_vs_finish_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()

print("\nFigures saved to:", FIG_DIR)