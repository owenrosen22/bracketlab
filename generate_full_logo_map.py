import pandas as pd

teams = pd.read_csv("master_ratings.csv")

rows = []

for team in teams["Team"].unique():

    name = team.lower().replace(" ", "-").replace(".", "").replace("&", "and")

    url = f"https://a.espncdn.com/i/teamlogos/ncaa/500/{name}.png"

    rows.append({
        "Team": team,
        "LogoURL": url
    })

df = pd.DataFrame(rows)
df.to_csv("team_logos.csv", index=False)

print("team_logos.csv rebuilt")
