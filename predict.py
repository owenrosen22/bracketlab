import pandas as pd
import numpy as np

BETA = 0.17

def logistic(z):
    return 1 / (1 + np.exp(-z))

df = pd.read_csv("ratings.csv")
df["Team"] = df["Team"].astype(str).str.strip()

ratings = dict(zip(df["Team"], df["Rating"]))

def win_prob(team_a, team_b):

    if team_a not in ratings:
        raise ValueError(f"Team not found: {team_a}")

    if team_b not in ratings:
        raise ValueError(f"Team not found: {team_b}")

    ra = ratings[team_a]
    rb = ratings[team_b]

    z = BETA * (ra - rb)

    return logistic(z)

while True:

    print("\nEnter two teams (or type quit)")

    team_a = input("Team A: ").strip()

    if team_a.lower() == "quit":
        break

    team_b = input("Team B: ").strip()

    if team_b.lower() == "quit":
        break

    probability = win_prob(team_a, team_b)

    print(f"\nProbability {team_a} beats {team_b}: {round(probability,3)}")
    