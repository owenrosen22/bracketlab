import pandas as pd

YEAR = 2026  # change if needed
URL = f"https://barttorvik.com/{YEAR}_team_results.csv"

print("Downloading:", URL)
df = pd.read_csv(URL)

# standardize column names to lowercase
df.columns = [c.lower() for c in df.columns]

needed = ["team", "adjoe", "adjde", "adjt", "barthag"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in Torvik CSV: {missing}\nHave: {df.columns.tolist()}")

out = df[needed].copy()
out.to_csv("torvik_ratings.csv", index=False)

print("Saved torvik_ratings.csv with columns:", out.columns.tolist())
print("Teams:", len(out))
print(out.head(5).to_string(index=False))
