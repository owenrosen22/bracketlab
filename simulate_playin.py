import pandas as pd
import numpy as np
from predict_proba import win_prob, get_missing_teams


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


def simulate_playins(playins, rng):
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


def simulate_tournament(round1_games, rng):

    teams = set([t for g in round1_games for t in g])
    depth = {t: 0 for t in teams}

    current = round1_games
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


def summarize(d, N, k):
    items = sorted(d.items(), key=lambda x: -x[1])[:k]
    return [(t, c / N) for t, c in items]


def most_likely(a, b):
    p = win_prob(a, b)
    if p >= 0.5:
        return a, p
    return b, 1 - p


def build_bracket(round1, playins):

    rows = []
    slot_winners = {}

    for slot, a, b in playins:
        w, p = most_likely(a, b)
        slot_winners[slot] = w

    games = substitute_slots(round1, slot_winners)

    winners = []

    for a, b in games:
        w, p = most_likely(a, b)
        winners.append(w)
        rows.append({"Round": "R64", "TeamA": a, "TeamB": b, "Pick": w, "Prob": round(p, 3)})

    rounds = ["R32", "S16", "E8", "F4", "Title"]

    for r in rounds:

        games = list(zip(winners[0::2], winners[1::2]))
        winners = []

        for a, b in games:
            w, p = most_likely(a, b)
            winners.append(w)
            rows.append({"Round": r, "TeamA": a, "TeamB": b, "Pick": w, "Prob": round(p, 3)})

    return pd.DataFrame(rows)


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

        winners = simulate_playins(playins, rng)
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

    print("\n=== Most likely Sweet 16 teams ===")
    for t, p in summarize(sweet16, N, 20):
        print(f"{t}: {p:.3f}")

    print("\n=== Most likely Elite 8 teams ===")
    for t, p in summarize(elite8, N, 16):
        print(f"{t}: {p:.3f}")

    print("\n=== Most likely Final Four teams ===")
    for t, p in summarize(final4, N, 10):
        print(f"{t}: {p:.3f}")

    print("\n=== Championship odds ===")
    for t, p in summarize(champ, N, 10):
        print(f"{t}: {p:.3f}")

    print("\n=== Round 1 upset watch ===")

    slot_winners = {}
    for slot, a, b in playins:
        w, _ = most_likely(a, b)
        slot_winners[slot] = w

    resolved = substitute_slots(round1, slot_winners)

    closeness = []

    for a, b in resolved:
        pa = win_prob(a, b)
        closeness.append((abs(pa - 0.5), a, b, pa, 1 - pa))

    closeness.sort()

    for _, a, b, pa, pb in closeness[:12]:
        print(f"{a} vs {b}: P({a})={pa:.3f}, P({b})={pb:.3f}")

    bracket = build_bracket(round1, playins)
    bracket.to_csv("bracket_picks.csv", index=False)

    print("\nBracket written to bracket_picks.csv")

    missing = get_missing_teams()

    if missing:
        print("\nTeams using default rating:")
        for t in missing:
            print("-", t)


if __name__ == "__main__":
    main()
