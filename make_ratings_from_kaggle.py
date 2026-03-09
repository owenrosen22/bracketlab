import pandas as pd

DATA_DIR = "kaggle_data"
TEAMS_FILE = f"{DATA_DIR}/MTeams.csv"
RESULTS_FILE = f"{DATA_DIR}/MRegularSeasonCompactResults.csv"

# Change this to the season you want (e.g., 2025, 2026)
SEASON = 2025

teams = pd.read_csv(TEAMS_FILE)  # TeamID, TeamName
results = pd.read_csv(RESULTS_FILE)

df = results[results["Season"] == SEASON].copy()

# Winner rows
w = df[["WTeamID", "WScore", "LScore"]].copy()
w.columns = ["TeamID", "For", "Against"]
w["Win"] = 1
w["Diff"] = w["For"] - w["Against"]

# Loser rows
l = df[["LTeamID", "LScore", "WScore"]].copy()
l.columns = ["TeamID", "For", "Against"]
l["Win"] = 0
l["Diff"] = l["For"] - l["Against"]

all_games = pd.concat([w, l], ignore_index=True)

agg = (
    all_games
    .groupby("TeamID")
    .agg(
        Games=("Win", "count"),
        WinPct=("Win", "mean"),
        AvgDiff=("Diff", "mean"),
    )
    .reset_index()
)

# Merge team names
out = agg.merge(teams, on="TeamID", how="left")

# ratings.csv format:
# Rating = AvgDiff (what your model expects)
out = out.rename(columns={"TeamName": "Team"})
out = out[["Team", "AvgDiff", "WinPct", "Games"]].sort_values("AvgDiff", ascending=False)

# Save
out.to_csv("ratings.csv", index=False)
print("Wrote ratings.csv with", len(out), "teams for season", SEASON)
print(out.head(15).to_string(index=False))
