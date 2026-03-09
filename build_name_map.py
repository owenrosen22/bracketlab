import pandas as pd
from rapidfuzz import process, fuzz

DATA_DIR = "kaggle_data"
TEAMS_FILE = f"{DATA_DIR}/MTeams.csv"

def normalize(s: str) -> str:
    s = str(s).strip()
    s = s.replace("*", "")
    s = s.replace("\u00a0", " ")
    s = " ".join(s.split())

    # remove punctuation-ish
    for ch in [".", "'", "’", ",", "(", ")", "&"]:
        s = s.replace(ch, "")

    # common words
    s = s.replace("St ", "Saint ")
    s = s.replace("St-", "Saint-")
    s = s.replace("Mt ", "Mount ")

    # collapse again
    s = " ".join(s.split())
    return s.lower()

def main():
    b = pd.read_csv("bracket_round1.csv")
    teams = pd.read_csv(TEAMS_FILE)

    bracket_raw = set(b["TeamA"].astype(str)) | set(b["TeamB"].astype(str))

    # handle play-in placeholders "A / B" by splitting
    expanded = set()
    for t in bracket_raw:
        t = str(t).strip()
        if " / " in t:
            a, b2 = [x.strip() for x in t.split(" / ", 1)]
            expanded.add(a)
            expanded.add(b2)
        else:
            expanded.add(t)

    kaggle_names = teams["TeamName"].astype(str).tolist()

    kaggle_norm = {normalize(n): n for n in kaggle_names}
    kaggle_norm_keys = list(kaggle_norm.keys())

    missing = []
    for t in sorted(expanded):
        nt = normalize(t)
        if nt not in kaggle_norm:
            missing.append(t)

    print("\nBracket teams NOT found in Kaggle team names (after normalization):")
    for t in missing:
        print(" -", t)

    suggestions = []
    for t in missing:
        nt = normalize(t)
        match = process.extractOne(
            nt,
            kaggle_norm_keys,
            scorer=fuzz.WRatio
        )
        if match:
            best_norm, score, _ = match
            best_name = kaggle_norm[best_norm]
            suggestions.append((t, best_name, score))

    sug_df = pd.DataFrame(suggestions, columns=["From", "To", "Score"])
    sug_df = sug_df.sort_values("Score", ascending=False)

    print("\nTop suggested mappings:")
    print(sug_df.head(30).to_string(index=False))

    sug_df.to_csv("name_map_suggestions.csv", index=False)
    print("\nWrote name_map_suggestions.csv")
    print("Open it, and copy the good rows into name_map.csv (keep header: From,To).")

if __name__ == "__main__":
    main()
    