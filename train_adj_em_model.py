import pandas as pd
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

print("Loading Kaggle data...")

games = pd.read_csv("kaggle_data/MNCAATourneyCompactResults.csv")

# Create rating proxies from regular season
reg = pd.read_csv("kaggle_data/MRegularSeasonCompactResults.csv")

print("Building team ratings...")

stats = []

for season in reg.Season.unique():
    season_games = reg[reg.Season == season]

    teams = pd.unique(season_games[['WTeamID','LTeamID']].values.ravel())

    for team in teams:
        wins = season_games[season_games.WTeamID == team]
        losses = season_games[season_games.LTeamID == team]

        games_played = len(wins) + len(losses)
        if games_played == 0:
            continue

        avg_diff = (
            wins.WScore.sum() - wins.LScore.sum()
            + losses.LScore.sum() - losses.WScore.sum()
        ) / games_played

        win_pct = len(wins) / games_played

        stats.append({
            "Season": season,
            "TeamID": team,
            "AvgDiff": avg_diff,
            "WinPct": win_pct
        })

ratings = pd.DataFrame(stats)

print("Building training rows...")

rows = []

for _, game in games.iterrows():

    season = game.Season
    w = game.WTeamID
    l = game.LTeamID

    r_w = ratings[(ratings.Season == season) & (ratings.TeamID == w)]
    r_l = ratings[(ratings.Season == season) & (ratings.TeamID == l)]

    if r_w.empty or r_l.empty:
        continue

    r_w = r_w.iloc[0]
    r_l = r_l.iloc[0]

    rows.append({
        "d_AvgDiff": r_w.AvgDiff - r_l.AvgDiff,
        "d_WinPct": r_w.WinPct - r_l.WinPct,
        "result": 1
    })

    rows.append({
        "d_AvgDiff": r_l.AvgDiff - r_w.AvgDiff,
        "d_WinPct": r_l.WinPct - r_w.WinPct,
        "result": 0
    })

df = pd.DataFrame(rows)

print("Training model on", len(df), "examples")

X = df[["d_AvgDiff","d_WinPct"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict_proba(X_test)[:,1]

print("Holdout log loss:", log_loss(y_test, pred))
print("Holdout AUC:", roc_auc_score(y_test, pred))

out = {
    "intercept": float(model.intercept_[0]),
    "coef": {
        "d_AvgDiff": float(model.coef_[0][0]),
        "d_WinPct": float(model.coef_[0][1])
    }
}

with open("model.json","w") as f:
    json.dump(out,f)

print("Saved improved model.json")
print(out)
