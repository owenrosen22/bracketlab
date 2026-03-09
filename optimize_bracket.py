import pandas as pd
from predict_proba import win_prob

ROUND_POINTS = {
    "R64": 10,
    "R32": 20,
    "S16": 40,
    "E8": 80,
    "F4": 160,
    "Title": 320,
}

def load_name_map():
    try:
        df = pd.read_csv("name_map.csv")
        df["From"] = df["From"].astype(str).str.strip()
        df["To"] = df["To"].astype(str).str.strip()
        return dict(zip(df["From"], df["To"]))
    except:
        return {}

def clean_name(name, name_map):
    name = str(name).strip()
    name = name.replace("*", "")
    name = " ".join(name.split())
    return name_map.get(name, name)

def load_playins():
    name_map = load_name_map()
    df = pd.read_csv("play_in.csv")
    df["Team1"] = df["Team1"].apply(lambda x: clean_name(x, name_map))
    df["Team2"] = df["Team2"].apply(lambda x: clean_name(x, name_map))
    return list(zip(df["Slot"], df["Team1"], df["Team2"]))

def load_round1():
    name_map = load_name_map()
    df = pd.read_csv("bracket_round1.csv")
    df["TeamA"] = df["TeamA"].apply(lambda x: clean_name(x, name_map))
    df["TeamB"] = df["TeamB"].apply(lambda x: clean_name(x, name_map))
    return list(zip(df["TeamA"], df["TeamB"]))

def most_likely(a, b):
    p = win_prob(a, b)
    if p >= 0.5:
        return a, p
    return b, 1 - p

def expected_pick(a, b, points):
    p = win_prob(a, b)
    ev_a = points * p
    ev_b = points * (1 - p)
    if ev_a >= ev_b:
        return a, p, ev_a
    else:
        return b, 1 - p, ev_b

def substitute_slots(games, winners):
    out = []
    for a, b in games:
        out.append((winners.get(a, a), winners.get(b, b)))
    return out

def optimize():

    playins = load_playins()
    round1 = load_round1()

    rows = []
    slot_winners = {}

    for slot, a, b in playins:
        w, p = most_likely(a, b)
        slot_winners[slot] = w
        rows.append({
            "Round":"PLAYIN",
            "TeamA":a,
            "TeamB":b,
            "Pick":w,
            "Prob":round(p,3),
            "ExpPoints":""
        })

    games = substitute_slots(round1, slot_winners)

    winners = []

    for a, b in games:
        pick, p, ev = expected_pick(a, b, ROUND_POINTS["R64"])
        winners.append(pick)
        rows.append({
            "Round":"R64",
            "TeamA":a,
            "TeamB":b,
            "Pick":pick,
            "Prob":round(p,3),
            "ExpPoints":round(ev,2)
        })

    rounds = ["R32","S16","E8","F4","Title"]

    for r in rounds:

        games = list(zip(winners[0::2], winners[1::2]))
        winners = []

        for a,b in games:

            pick,p,ev = expected_pick(a,b,ROUND_POINTS[r])
            winners.append(pick)

            rows.append({
                "Round":r,
                "TeamA":a,
                "TeamB":b,
                "Pick":pick,
                "Prob":round(p,3),
                "ExpPoints":round(ev,2)
            })

    df = pd.DataFrame(rows)
    df.to_csv("optimal_bracket.csv",index=False)

    print("Optimal bracket saved to optimal_bracket.csv")

if __name__ == "__main__":
    optimize()
    