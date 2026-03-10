import json
import numpy as np
import pandas as pd

from ratings_utils import canonical_team, prepare_ratings

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

with open("model_adj_em.json", "r") as f:
    M = json.load(f)

INTERCEPT = float(M["intercept"])
COEF_ADJEM = float(M["coef"]["d_AdjEM"])

df = pd.read_csv("master_ratings.csv")
df = prepare_ratings(df)

adj_em = dict(zip(df["Team"], df["AdjEM_current"]))
DEFAULT_ADJEM = float(pd.Series(list(adj_em.values())).dropna().mean())

_missing_teams = set()

def win_prob(team_a: str, team_b: str) -> float:
    a_raw = str(team_a).strip()
    b_raw = str(team_b).strip()

    a = canonical_team(a_raw)
    b = canonical_team(b_raw)

    if a not in adj_em and a.endswith(" St") and (a + ".") in adj_em:
        a = a + "."
    if b not in adj_em and b.endswith(" St") and (b + ".") in adj_em:
        b = b + "."
    if a not in adj_em and a == "NC State" and "N.C. State" in adj_em:
        a = "N.C. State"
    if b not in adj_em and b == "NC State" and "N.C. State" in adj_em:
        b = "N.C. State"

    if a not in adj_em:
        _missing_teams.add(a_raw)
    if b not in adj_em:
        _missing_teams.add(b_raw)

    a_em = float(adj_em.get(a, DEFAULT_ADJEM))
    b_em = float(adj_em.get(b, DEFAULT_ADJEM))

    d_em = a_em - b_em
    z = INTERCEPT + COEF_ADJEM * d_em

    return float(sigmoid(z))

def get_missing_teams():
    return sorted(_missing_teams)
