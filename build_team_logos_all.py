import pandas as pd
import requests

API_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/"
    "basketball/mens-college-basketball/teams?limit=1000"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


def normalize(name: str) -> str:
    if pd.isna(name):
        return ""
    name = str(name).strip()
    replacements = {
        "St.": "St",
        "Saint": "St",
        "N.C.": "N.C.",
        "&": "&",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    return " ".join(name.split())


def build_alias_map() -> dict[str, str]:
    return {
        "UConn": "Connecticut",
        "UCF": "UCF",
        "USC": "USC",
        "SMU": "SMU",
        "VCU": "VCU",
        "BYU": "BYU",
        "TCU": "TCU",
        "LIU": "LIU",
        "N.C. State": "N.C. State",
        "St. John's": "St John's",
        "Saint Mary's": "St Mary's CA",
        "St. Mary's": "St Mary's CA",
        "Saint Louis": "St Louis",
        "Utah State": "Utah St.",
        "Iowa State": "Iowa St.",
        "Michigan State": "Michigan St.",
        "Ohio State": "Ohio St.",
        "Penn State": "Penn St.",
        "Boise State": "Boise St.",
        "San Diego State": "San Diego St.",
        "Colorado State": "Colorado St.",
        "Mississippi State": "Mississippi St.",
        "Kansas State": "Kansas St.",
        "Washington State": "Washington St.",
        "North Dakota State": "North Dakota St.",
        "Wright State": "Wright St.",
        "East Tennessee State": "East Tennessee St.",
        "Florida State": "Florida St.",
        "Arizona State": "Arizona St.",
        "Oklahoma State": "Oklahoma St.",
        "Montana State": "Montana St.",
        "Missouri State": "Missouri St.",
        "South Dakota State": "South Dakota St.",
        "San Jose State": "San Jose St.",
        "Texas State": "Texas St.",
        "New Mexico State": "New Mexico St.",
        "Jacksonville State": "Jacksonville St.",
        "Tennessee State": "Tennessee St.",
        "Ball State": "Ball St.",
        "Nicholls": "Nicholls St.",
        "Charleston Southern": "Charleston Southern",
        "Long Beach State": "Long Beach St.",
        "Mount St. Mary's": "Mount St. Mary's",
        "Idaho State": "Idaho St.",
        "Morehead State": "Morehead St.",
        "Texas A&M-Corpus Christi": "Texas A&M Corpus Chris",
        "Southeast Missouri State": "Southeast Missouri St.",
        "SIU-Edwardsville": "SIU Edwardsville",
        "Cal State Bakersfield": "Cal St. Bakersfield",
        "Cal State Fullerton": "Cal St. Fullerton",
        "UC Santa Barbara": "UC Santa Barbara",
        "UC San Diego": "UC San Diego",
        "UC Davis": "UC Davis",
        "UC Irvine": "UC Irvine",
        "UNC Wilmington": "UNC Wilmington",
        "UNC Asheville": "UNC Asheville",
        "UNC Greensboro": "UNC Greensboro",
        "USC Upstate": "USC Upstate",
        "Purdue Fort Wayne": "Purdue Fort Wayne",
        "Nebraska Omaha": "Nebraska Omaha",
        "Western Michigan": "Western Michigan",
        "Northern Illinois": "Northern Illinois",
        "Southern Illinois": "Southern Illinois",
        "Northern Iowa": "Northern Iowa",
        "Northern Kentucky": "Northern Kentucky",
        "Northern Colorado": "Northern Colorado",
        "Northern Arizona": "Northern Arizona",
        "Northwestern State": "Northwestern St.",
        "South Carolina State": "South Carolina St.",
        "Alcorn State": "Alcorn St.",
        "Coppin State": "Coppin St.",
        "Delaware State": "Delaware St.",
        "Grambling": "Grambling St.",
        "Morgan State": "Morgan St.",
        "Norfolk State": "Norfolk St.",
        "Prairie View": "Prairie View A&M",
        "Chicago State": "Chicago St.",
        "Bellarmine": "Bellarmine",
        "Le Moyne": "Le Moyne",
        "Stonehill": "Stonehill",
        "New Haven": "New Haven",
        "Mercyhurst": "Mercyhurst",
        "Queens University": "Queens",
        "St. Thomas-Minnesota": "St. Thomas",
        "IU Indianapolis": "IU Indy",
        "UMKC": "UMKC",
        "UT Rio Grande Valley": "UT Rio Grande Valley",
        "Abilene Christian": "Abilene Christian",
        "Akron": "Akron",
        "Georgetown": "Georgetown",
    }


def fetch_espn_teams():
    resp = requests.get(API_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    teams = []

    def add_team_obj(team_obj):
        if not isinstance(team_obj, dict):
            return
        team = team_obj.get("team", team_obj)

        display_name = team.get("displayName") or team.get("shortDisplayName") or team.get("name")
        if not display_name:
            return

        logos = team.get("logos") or []
        logo_url = ""
        if logos and isinstance(logos, list):
            logo_url = logos[0].get("href", "") or logos[0].get("url", "")

        team_id = str(team.get("id", "")).strip()
        if not logo_url and team_id:
            logo_url = f"https://a.espncdn.com/i/teamlogos/ncaa/500/{team_id}.png"

        teams.append({
            "espn_name": display_name,
            "espn_id": team_id,
            "LogoURL": logo_url,
        })

    # Try common ESPN response shapes
    if "sports" in data:
        for sport in data.get("sports", []):
            for league in sport.get("leagues", []):
                for item in league.get("teams", []):
                    add_team_obj(item)

    if not teams and "teams" in data:
        for item in data.get("teams", []):
            add_team_obj(item)

    return pd.DataFrame(teams).drop_duplicates(subset=["espn_name"])


def main():
    ratings = pd.read_csv("master_ratings.csv")
    ratings["Team"] = ratings["Team"].astype(str).str.strip()

    espn = fetch_espn_teams()
    alias_map = build_alias_map()

    # Build name lookup
    name_to_logo = {}
    for _, row in espn.iterrows():
        espn_name = normalize(row["espn_name"])
        logo = str(row["LogoURL"]).strip()
        if espn_name and logo:
            name_to_logo[espn_name] = logo

    rows = []
    missing = []

    for team in sorted(ratings["Team"].unique()):
        key = normalize(team)
        logo = name_to_logo.get(key, "")

        if not logo:
            # Try reverse alias lookup: ESPN name -> local name mapping
            for espn_name, local_name in alias_map.items():
                if local_name == team:
                    logo = name_to_logo.get(normalize(espn_name), "")
                    if logo:
                        break

        rows.append({"Team": team, "LogoURL": logo})
        if not logo:
            missing.append(team)

    out = pd.DataFrame(rows)
    out.to_csv("team_logos.csv", index=False)

    print(f"Created team_logos.csv with {len(out)} teams")
    print(f"Matched logos: {(out['LogoURL'] != '').sum()}")
    print(f"Missing logos: {(out['LogoURL'] == '').sum()}")

    if missing:
        print("\\nStill missing:")
        for team in missing[:100]:
            print("-", team)


if __name__ == "__main__":
    main()
    