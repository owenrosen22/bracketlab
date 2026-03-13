import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ratings_utils import canonical_team


ROOT = Path(__file__).resolve().parent


def run_step(cmd):
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name)


def logo_coverage(master: pd.DataFrame) -> tuple[int, int]:
    logos = load_csv("team_logos.csv")
    logo_teams = set(logos["Team"].astype(str).map(canonical_team))
    master_teams = set(master["Team"].astype(str).map(canonical_team))
    missing = sorted(master_teams - logo_teams)
    return len(missing), len(master_teams)


def print_summary():
    master = load_csv("master_ratings.csv")
    round1 = load_csv("bracket_round1.csv")
    play_in = load_csv("play_in.csv")
    injuries = load_csv("injury_reports.csv") if (ROOT / "injury_reports.csv").exists() else pd.DataFrame()

    source_counts = {
        "Torvik": int(master["AdjEM_torvik"].notna().sum()) if "AdjEM_torvik" in master.columns else 0,
        "KenPom": int(master["AdjEM_kenpom"].notna().sum()) if "AdjEM_kenpom" in master.columns else 0,
        "EvanMiya": int(master["AdjEM_evanmiya"].notna().sum()) if "AdjEM_evanmiya" in master.columns else 0,
    }
    zero_scores = int((pd.to_numeric(master.get("AdjEM_current"), errors="coerce").fillna(0) == 0).sum())
    missing_logos, total_teams = logo_coverage(master)

    print("\nSelection Sunday readiness\n")
    print(f"Round of 64 games: {len(round1)}")
    print(f"Play-in games: {len(play_in)}")
    print(f"Teams in master_ratings.csv: {len(master)}")
    print(f"Source coverage: T {source_counts['Torvik']} · K {source_counts['KenPom']} · E {source_counts['EvanMiya']}")
    print(f"Injury rows: {len(injuries)}")
    print(f"Zero current ratings: {zero_scores}")
    print(f"Missing logos: {missing_logos} / {total_teams}")

    if len(round1) != 32:
        print("WARN bracket_round1.csv should have 32 games.")
    if len(play_in) != 4:
        print("WARN play_in.csv should have 4 games.")
    if len(master) != 365:
        print("WARN master_ratings.csv should normally have 365 teams.")
    if zero_scores:
        print("WARN some teams still have zero current ratings.")
    if missing_logos:
        print("WARN some teams are still missing logos.")

    print("\nNext step: streamlit run app.py")


def main():
    parser = argparse.ArgumentParser(description="Run the full Selection Sunday BracketLab refresh workflow.")
    parser.add_argument("--skip-torvik-fetch", action="store_true", help="Skip pulling fresh Torvik data")
    args = parser.parse_args()

    cmd = [sys.executable, "refresh_data.py"]
    if args.skip_torvik_fetch:
        cmd.append("--skip-torvik-fetch")

    run_step(cmd)
    print_summary()


if __name__ == "__main__":
    main()
