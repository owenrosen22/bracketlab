import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

DATA_DIR = "kaggle_data"
REG_SEASON = f"{DATA_DIR}/MRegularSeasonCompactResults.csv"

def build_team_strength(df: pd.DataFrame) -> pd.DataFrame:
    # Winner rows
    w = df[["Season", "WTeamID", "WScore", "LScore"]].copy()
    w.columns = ["Season", "TeamID", "For", "Against"]
    w["Win"] = 1
    w["Diff"] = w["For"] - w["Against"]

    # Loser rows
    l = df[["Season", "LTeamID", "LScore", "WScore"]].copy()
    l.columns = ["Season", "TeamID", "For", "Against"]
    l["Win"] = 0
    l["Diff"] = l["For"] - l["Against"]

    all_games = pd.concat([w, l], ignore_index=True)

    agg = (
        all_games
        .groupby(["Season", "TeamID"])
        .agg(
            Games=("Win", "count"),
            WinPct=("Win", "mean"),
            AvgDiff=("Diff", "mean"),
        )
        .reset_index()
    )
    return agg

def make_rows(results: pd.DataFrame, strength: pd.DataFrame) -> pd.DataFrame:
    s = strength.set_index(["Season", "TeamID"])
    rows = []

    for _, r in results.iterrows():
        season = int(r["Season"])
        w = int(r["WTeamID"])
        l = int(r["LTeamID"])

        sw = s.loc[(season, w)]
        sl = s.loc[(season, l)]

        # Row: A=winner, B=loser (y=1)
        rows.append({
            "d_AvgDiff": float(sw["AvgDiff"] - sl["AvgDiff"]),
            "d_WinPct": float(sw["WinPct"] - sl["WinPct"]),
            "y": 1
        })
        # Row: A=loser, B=winner (y=0)
        rows.append({
            "d_AvgDiff": float(sl["AvgDiff"] - sw["AvgDiff"]),
            "d_WinPct": float(sl["WinPct"] - sw["WinPct"]),
            "y": 0
        })

    return pd.DataFrame(rows)

def main():
    print("Loading results...")
    results = pd.read_csv(REG_SEASON)

    print("Building per-team season strength...")
    strength = build_team_strength(results)

    print("Building training table...")
    train_df = make_rows(results, strength)

    X = train_df[["d_AvgDiff", "d_WinPct"]].values
    y = train_df["y"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training logistic regression...")
    model = LogisticRegression(max_iter=3000)
    model.fit(X_train, y_train)

    p = model.predict_proba(X_test)[:, 1]
    print("Holdout log loss:", round(log_loss(y_test, p), 4))
    print("Holdout AUC:", round(roc_auc_score(y_test, p), 4))

    out = {
        "intercept": float(model.intercept_[0]),
        "coef": {
            "d_AvgDiff": float(model.coef_[0][0]),
            "d_WinPct": float(model.coef_[0][1]),
        }
    }

    with open("model.json", "w") as f:
        json.dump(out, f, indent=2)

    print("Saved model.json")
    print(out)

if __name__ == "__main__":
    main()
    