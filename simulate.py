import pandas as pd
import numpy as np

from predict_proba import win_prob, get_missing_teams


def load_name_map(path="name_map.csv"):
    """
    Optional file to map bracket names -> ratings.csv names.
    Format:
      From,To
      East Tennessee State,ETSU
      BYU*,BYU
    """
    try:
        m = pd.read_csv(path)
        m["From"] = m["From"].astype(str).str.strip()
        m["To"] = m["To"].astype(str).str.strip()
        return dict(zip(m["From"], m["To"]))
    except FileNotFoundError:
        return {}


def clean_team_name(name: str, name_map: dict) -> str:
    """
    Normalize bracket team strings to match ratings.csv names.
    - Strips whitespace
    - Removes '*'
    - Collapses multiple spaces
    - Converts 'A / B' -> 'A' (temporary play-in handling)
    - Applies name_map at the end
    """
    s = str(name).replace("\u00a0", " ").strip()
    s = " ".join(s.split())
    s = s.replace("*", "")

    # If bracketology gives play-in placeholder like "SMU / New Mexico"
    if " / " in s:
        s = s.split(" / ")[0].strip()

    return name_map.get(s, s)


def load_bracket(path="bracket_round1.csv"):
    name_map = load_name_map()
    df = pd.read_csv(path)
    df["TeamA"] = df["TeamA"].apply(lambda x: clean_team_name(x, name_map))
    df["TeamB"] = df["TeamB"].apply(lambda x: clean_team_name(x, name_map))
    games = list(zip(df["TeamA"], df["TeamB"]))
    return games


def simulate_game(team_a, team_b, rng):
    p = win_prob(team_a, team_b)
    return team_a if rng.random() < p else team_b


def simulate_tournament_depth(round1_games, rng):
    """
    Simulate the full bracket and return dict of team -> depth.

    depth meanings:
      0 = lost Round of 64
      1 = won Round of 64 (reached Round of 32)
      2 = won Round of 32 (reached Sweet 16)
      3 = won Sweet 16 (reached Elite 8)
      4 = won Elite 8 (reached Final Four)
      5 = won Final Four (reached Title game)
      6 = won Title (champion)
    """
    teams = set([t for g in round1_games for t in g])
    depth = {t: 0 for t in teams}

    current = round1_games
    round_index = 1

    while True:
        winners = []
        for a, b in current:
            w = simulate_game(a, b, rng)
            winners.append(w)
            depth[w] = max(depth[w], round_index)

        if len(winners) == 1:
            depth[winners[0]] = 6
            break

        current = list(zip(winners[0::2], winners[1::2]))
        round_index += 1

    return depth


def summarize_counts(counts, N, k):
    items = sorted(counts.items(), key=lambda x: -x[1])[:k]
    return [(t, c / N) for t, c in items]


def main():
    N = 100000
    rng = np.random.default_rng(42)

    round1_games = load_bracket()

    sweet16 = {}
    elite8 = {}
    final4 = {}
    champ = {}

    for _ in range(N):
        depth = simulate_tournament_depth(round1_games, rng)
        for team, d in depth.items():
            if d >= 2:
                sweet16[team] = sweet16.get(team, 0) + 1
            if d >= 3:
                elite8[team] = elite8.get(team, 0) + 1
            if d >= 4:
                final4[team] = final4.get(team, 0) + 1
            if d == 6:
                champ[team] = champ.get(team, 0) + 1

    print("\n=== Most likely Sweet 16 teams ===")
    for t, p in summarize_counts(sweet16, N, 20):
        print(f"{t}: {p:.3f}")

    print("\n=== Most likely Elite 8 teams ===")
    for t, p in summarize_counts(elite8, N, 16):
        print(f"{t}: {p:.3f}")

    print("\n=== Most likely Final Four teams ===")
    for t, p in summarize_counts(final4, N, 10):
        print(f"{t}: {p:.3f}")

    print("\n=== Championship odds ===")
    for t, p in summarize_counts(champ, N, 10):
        print(f"{t}: {p:.3f}")

    print("\n=== Round 1 upset watch (closest games) ===")
    closeness = []
    for a, b in round1_games:
        pa = win_prob(a, b)
        closeness.append((abs(pa - 0.5), a, b, pa, 1 - pa))
    closeness.sort()

    for _, a, b, pa, pb in closeness[:12]:
        print(f"{a} vs {b}: P({a})={pa:.3f}, P({b})={pb:.3f}")

    missing = get_missing_teams()
    if missing:
        print("\nNOTE: These teams were missing from ratings.csv and used default average strength:")
        for t in missing:
            print(" -", t)


if __name__ == "__main__":
    main()
