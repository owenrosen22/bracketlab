import pandas as pd
import numpy as np
from predict_proba import win_prob

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

def simulate_game(a, b, rng):
    p = win_prob(a, b)
    return a if rng.random() < p else b

def resolve_playins(playins, rng):
    winners = {}
    for slot, a, b in playins:
        winners[slot] = simulate_game(a, b, rng)
    return winners

def substitute_slots(round1, winners):
    new_games = []
    for a, b in round1:
        a = winners.get(a, a)
        b = winners.get(b, b)
        new_games.append((a, b))
    return new_games

def simulate_tournament(games, rng):
    teams = set([t for g in games for t in g])
    depth = {t: 0 for t in teams}

    current = games
    r = 1

    while True:
        winners = []

        for a, b in current:
            w = simulate_game(a, b, rng)
            winners.append(w)
            depth[w] = max(depth[w], r)

        if len(winners) == 1:
            depth[winners[0]] = 6
            break

        current = list(zip(winners[0::2], winners[1::2]))
        r += 1

    return depth

def main():
    N = 100000
    rng = np.random.default_rng(42)

    playins = load_playins()
    round1 = load_round1()

    sweet16 = {}
    elite8 = {}
    final4 = {}
    champ = {}

    for _ in range(N):
        winners = resolve_playins(playins, rng)
        games = substitute_slots(round1, winners)
        depth = simulate_tournament(games, rng)

        for t, d in depth.items():
            if d >= 2:
                sweet16[t] = sweet16.get(t, 0) + 1
            if d >= 3:
                elite8[t] = elite8.get(t, 0) + 1
            if d >= 4:
                final4[t] = final4.get(t, 0) + 1
            if d == 6:
                champ[t] = champ.get(t, 0) + 1

    teams = sorted(set(list(sweet16.keys()) + list(elite8.keys()) + list(final4.keys()) + list(champ.keys())))

    rows = []
    for t in teams:
        s16 = sweet16.get(t, 0) / N
        e8 = elite8.get(t, 0) / N
        f4 = final4.get(t, 0) / N
        ch = champ.get(t, 0) / N

        # Cinderella score:
        # favors teams with real Sweet16/Elite8 paths,
        # but discounts teams that are just obvious title favorites
        score = (2.5 * s16) + (3.0 * e8) + (2.0 * f4) + (1.0 * ch) - (1.5 * ch)

        rows.append({
            "Team": t,
            "Sweet16Prob": round(s16, 3),
            "Elite8Prob": round(e8, 3),
            "Final4Prob": round(f4, 3),
            "ChampProb": round(ch, 3),
            "CinderellaScore": round(score, 3)
        })

    df = pd.DataFrame(rows)

    # remove obvious mega-favorites from Cinderella ranking
    df = df[df["ChampProb"] < 0.08].copy()

    df = df.sort_values("CinderellaScore", ascending=False)

    df.to_csv("cinderella_rankings.csv", index=False)

    print("\n=== Top Cinderella Candidates ===")
    print(df.head(15).to_string(index=False))

    print("\nSaved cinderella_rankings.csv")

if __name__ == "__main__":
    main()
    