import pandas as pd

DATA_DIR = "kaggle_data"
SEASON = 2025

teams = pd.read_csv(f"{DATA_DIR}/MTeams.csv")  # TeamID, TeamName
results = pd.read_csv(f"{DATA_DIR}/MRegularSeasonCompactResults.csv")

df = results[results["Season"] == SEASON].copy()

# winner rows
w = df[["WTeamID", "WScore", "LScore"]].copy()
w.columns = ["TeamID", "For", "Against"]
w["Win"] = 1
w["Diff"] = w["For"] - w["Against"]

# loser rows
l = df[["LTeamID", "LScore", "WScore"]].copy()
l.columns = ["TeamID", "For", "Against"]
l["Win"] = 0
l["Diff"] = l["For"] - l["Against"]

all_games = pd.concat([w, l], ignore_index=True)

agg = (
    all_games.groupby("TeamID")
    .agg(Games=("Win", "count"), WinPct=("Win", "mean"), AvgDiff=("Diff", "mean"))
    .reset_index()
)

out = agg.merge(teams, on="TeamID", how="left")
out = out.rename(columns={"TeamName": "Team"})
out = out[["Team", "AvgDiff", "WinPct", "Games"]].sort_values("AvgDiff", ascending=False)

out.to_csv("ratings.csv", index=False)
print("Wrote ratings.csv with", len(out), "teams for season", SEASON)
