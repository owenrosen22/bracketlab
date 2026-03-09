import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

print("Loading Kaggle tournament results...")
games = pd.read_csv("kaggle_data/MNCAATourneyCompactResults.csv")

print("Loading regular season results...")
reg = pd.read_csv("kaggle_data/MRegularSeasonCompactResults.csv")

print("Building season team ratings from regular season...")

rows = []

for season in sorted(reg["Season"].unique()):

    season_games = reg[reg["Season"] == season]

    team_stats = []

    teams = pd.unique(season_games[["WTeamID", "LTeamID"]].values.ravel())

    for team in teams:

        wins = season_games[season_games["WTeamID"] == team]
        losses = season_games[season_games["LTeamID"] == team]

        gp = len(wins) + len(losses)
        if gp == 0:
            continue

        avg_diff = (
            (wins["WScore"] - wins["LScore"]).sum()
            + (losses["LScore"] - losses["WScore"]).sum()
        ) / gp

        team_stats.append({
            "Season": season,
            "TeamID": team,
            "AdjEM_proxy": avg_diff
        })

    season_ratings = pd.DataFrame(team_stats)

    season_tourney = games[games["Season"] == season]

    for _, game in season_tourney.iterrows():

        w = game["WTeamID"]
        l = game["LTeamID"]

        rw = season_ratings[season_ratings["TeamID"] == w]
        rl = season_ratings[season_ratings["TeamID"] == l]

        if rw.empty or rl.empty:
            continue

        rw = rw.iloc[0]
        rl = rl.iloc[0]

        # winner row
        rows.append({
            "d_AdjEM": rw["AdjEM_proxy"] - rl["AdjEM_proxy"],
            "result": 1
        })

        # loser row
        rows.append({
            "d_AdjEM": rl["AdjEM_proxy"] - rw["AdjEM_proxy"],
            "result": 0
        })

df = pd.DataFrame(rows)

print("Training on", len(df), "examples")

X = df[["d_AdjEM"]]
y = df["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict_proba(X_test)[:, 1]

print("Holdout log loss:", round(log_loss(y_test, pred), 4))
print("Holdout AUC:", round(roc_auc_score(y_test, pred), 4))

out = {
    "intercept": float(model.intercept_[0]),
    "coef": {
        "d_AdjEM": float(model.coef_[0][0])
    }
}

with open("model_adj_em.json", "w") as f:
    json.dump(out, f)

print("Saved model_adj_em.json")
print(out)
