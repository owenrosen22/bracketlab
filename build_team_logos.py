import pandas as pd

teams = pd.read_csv("master_ratings.csv")

# ESPN logo IDs for most common teams
# If team not here it will use fallback monogram
espn_ids = {
"Michigan":130,
"Duke":150,
"Arizona":12,
"Illinois":356,
"Florida":57,
"Houston":248,
"Purdue":2509,
"Connecticut":41,
"Texas Tech":2641,
"Iowa St.":66,
"Michigan St.":127,
"Tennessee":2633,
"Alabama":333,
"Louisville":97,
"Arkansas":8,
"Gonzaga":2250,
"Vanderbilt":238,
"Virginia":258,
"Kansas":2305,
"Nebraska":158,
"Wisconsin":275,
"St John's":2599,
"Ohio St.":194,
"North Carolina":153,
"UCLA":26,
"Kentucky":96,
"Miami FL":2390,
"Villanova":222,
"BYU":252,
"Texas":251,
"Creighton":156,
"Marquette":269,
"Xavier":2752,
"Oregon":2483,
"Notre Dame":87,
"Georgetown":46
}

rows = []

for t in teams["Team"].unique():
    if t in espn_ids:
        url = f"https://a.espncdn.com/i/teamlogos/ncaa/500/{espn_ids[t]}.png"
    else:
        url = ""
    rows.append({"Team": t, "LogoURL": url})

df = pd.DataFrame(rows)
df.to_csv("team_logos.csv", index=False)

print("team_logos.csv built")
