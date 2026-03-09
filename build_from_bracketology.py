import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Source bracket text (accessible + includes seeds/matchups)
URL = "https://bleacherreport.com/articles/25401867-latest-2026-mens-ncaa-tournament-bracket-projections"

# Seed -> rating mapping (fast + consistent)
SEED_TO_RATING = {
    1: 28, 2: 26, 3: 24, 4: 22,
    5: 20, 6: 18, 7: 16, 8: 14,
    9: 13, 10: 12, 11: 11, 12: 10,
    13: 9, 14: 8, 15: 7, 16: 6
}

def clean_team_name(name: str) -> str:
    # normalize common punctuation
    name = name.strip()
    name = name.replace("\u00a0", " ")
    name = re.sub(r"\s+", " ", name)
    return name

def main():
    print("Downloading bracketology page...")
    html = requests.get(URL, timeout=30).text
    soup = BeautifulSoup(html, "lxml")

    text = soup.get_text("\n")
    lines = [clean_team_name(l) for l in text.split("\n") if l.strip()]

    # Match lines like: "No. 1 Duke vs. No. 16 Tennessee State"
    game_pat = re.compile(
        r"^No\.\s*(\d{1,2})\s+(.+?)\s+vs\.\s+No\.\s*(\d{1,2})\s+(.+)$"
    )

    games = []
    team_seed = {}

    for line in lines:
        m = game_pat.match(line)
        if not m:
            continue
        seed_a = int(m.group(1))
        team_a = clean_team_name(m.group(2))
        seed_b = int(m.group(3))
        team_b = clean_team_name(m.group(4))

        # Basic sanity: only keep 1–16 seeds
        if not (1 <= seed_a <= 16 and 1 <= seed_b <= 16):
            continue

        games.append((team_a, team_b))
        team_seed[team_a] = seed_a
        team_seed[team_b] = seed_b

    if len(games) < 20:
        print("WARNING: extracted fewer games than expected.")
        print("First 30 extracted lines that matched 'No. x ... vs. No. y ...':")
        for g in games[:10]:
            print("  ", g)
        raise SystemExit("Could not extract enough games. The page format may have changed.")

    # Write bracket_round1.csv
    bracket_df = pd.DataFrame(games, columns=["TeamA", "TeamB"])
    bracket_df.to_csv("bracket_round1.csv", index=False)
    print(f"Wrote bracket_round1.csv with {len(bracket_df)} games.")

    # Build ratings.csv from seeds
    ratings_rows = []
    for team, seed in sorted(team_seed.items(), key=lambda x: (x[1], x[0])):
        rating = SEED_TO_RATING.get(seed, 0)
        ratings_rows.append((team, rating))

    ratings_df = pd.DataFrame(ratings_rows, columns=["Team", "Rating"])
    ratings_df.to_csv("ratings.csv", index=False)
    print(f"Wrote ratings.csv with {len(ratings_df)} teams.")

    # Quick check
    unique_teams = set(bracket_df["TeamA"]) | set(bracket_df["TeamB"])
    if len(unique_teams) != len(ratings_df):
        print("NOTE: team count mismatch (this can happen if play-in formatting appears).")
        print("Unique bracket teams:", len(unique_teams))
        print("Ratings teams:", len(ratings_df))

    print("\nDone. Next: run your simulator using bracket_round1.csv + ratings.csv.")

if __name__ == "__main__":
    main()
    