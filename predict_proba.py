import json
import numpy as np
import pandas as pd

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))

with open("model_adj_em.json", "r") as f:
    M = json.load(f)

INTERCEPT = float(M["intercept"])
COEF_ADJEM = float(M["coef"]["d_AdjEM"])

df = pd.read_csv("master_ratings.csv")
df["Team"] = df["Team"].astype(str).str.strip()

adj_em = dict(zip(df["Team"], df["AdjEM_blend"]))
DEFAULT_ADJEM = float(pd.Series(list(adj_em.values())).dropna().mean())

_missing_teams = set()

def load_name_map():
    try:
        m = pd.read_csv("name_map.csv")
        m["From"] = m["From"].astype(str).str.strip()
        m["To"] = m["To"].astype(str).str.strip()
        return dict(zip(m["From"], m["To"]))
    except:
        return {}

NAME_MAP = load_name_map()

ALIASES = {
    "ETSU": "East Tennessee St.",
    "East Tennessee State": "East Tennessee St.",
    "LIU Brooklyn": "LIU",
    "LIU": "LIU",
}

def canonical_team(name: str) -> str:
    s = str(name).strip().replace("*", "")
    s = " ".join(s.split())
    s = NAME_MAP.get(s, s)
    s = ALIASES.get(s, s)

    if s in adj_em:
        return s

    if s.endswith(" St") and (s + ".") in adj_em:
        return s + "."

    if s == "NC State" and "N.C. State" in adj_em:
        return "N.C. State"

    return s

def win_prob(team_a: str, team_b: str) -> float:
    a_raw = str(team_a).strip()
    b_raw = str(team_b).strip()

    a = canonical_team(a_raw)
    b = canonical_team(b_raw)

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
