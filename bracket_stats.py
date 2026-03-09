import pandas as pd
import numpy as np
from predict_proba import win_prob

# ESPN-style scoring
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

def substitute_slots(games, winners):
    out = []
    for a, b in games:
        out.append((winners.get(a, a), winners.get(b, b)))
    return out

def build_pick_bracket():
    """
    Build the deterministic 'most likely' bracket:
    pick the higher-probability winner in each game.
    """
    playins = load_playins()
    round1 = load_round1()

    slot_winners = {}
    rows = []

    for slot, a, b in playins:
        w, p = most_likely(a, b)
        slot_winners[slot] = w
        rows.append({
            "Round": "PLAYIN",
            "TeamA": a,
            "TeamB": b,
            "Pick": w,
            "PickProb": p
        })

    games = substitute_slots(round1, slot_winners)

    winners = []
    for a, b in games:
        w, p = most_likely(a, b)
        winners.append(w)
        rows.append({
            "Round": "R64",
            "TeamA": a,
            "TeamB": b,
            "Pick": w,
            "PickProb": p
        })

    rounds = ["R32", "S16", "E8", "F4", "Title"]

    for r in rounds:
        games = list(zip(winners[0::2], winners[1::2]))
        winners = []

        for a, b in games:
            w, p = most_likely(a, b)
            winners.append(w)
            rows.append({
                "Round": r,
                "TeamA": a,
                "TeamB": b,
                "Pick": w,
                "PickProb": p
            })

    return pd.DataFrame(rows)

def main():
    bracket = build_pick_bracket()

    # Ignore play-ins for "perfect bracket" unless you want them included
    tournament = bracket[bracket["Round"] != "PLAYIN"].copy()

    # Expected correct picks = sum of pick probabilities
    expected_correct = tournament["PickProb"].sum()

    # Expected points
    tournament["RoundPoints"] = tournament["Round"].map(ROUND_POINTS)
    tournament["ExpectedPoints"] = tournament["PickProb"] * tournament["RoundPoints"]
    expected_points = tournament["ExpectedPoints"].sum()

    # Perfect bracket probability = product of all pick probabilities
    perfect_prob = np.prod(tournament["PickProb"].values)

    # Also compute log10 scale because perfect probs are tiny
    log10_prob = np.log10(perfect_prob) if perfect_prob > 0 else float("-inf")

    print("\n=== Bracket Accuracy Stats ===")
    print(f"Expected correct picks: {expected_correct:.2f} out of {len(tournament)}")
    print(f"Expected ESPN-style points: {expected_points:.2f}")

    print("\n=== Perfect Bracket Odds ===")
    print(f"Probability of this exact bracket being perfect: {perfect_prob:.12e}")
    print(f"That is about 1 in {1/perfect_prob:,.0f}" if perfect_prob > 0 else "Too small to compute")
    print(f"log10(probability) = {log10_prob:.2f}")

    # Best and weakest picks in your bracket
    weakest = tournament.sort_values("PickProb").head(10)
    strongest = tournament.sort_values("PickProb", ascending=False).head(10)

    print("\n=== Weakest Picks In Your Bracket ===")
    print(weakest[["Round", "TeamA", "TeamB", "Pick", "PickProb"]].to_string(index=False))

    print("\n=== Strongest Picks In Your Bracket ===")
    print(strongest[["Round", "TeamA", "TeamB", "Pick", "PickProb"]].to_string(index=False))

    tournament.to_csv("bracket_stats_detail.csv", index=False)
    print("\nSaved bracket_stats_detail.csv")

if __name__ == "__main__":
    main()
    