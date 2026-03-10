import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = {
    "torvik_ratings.csv": ["Team", "AdjEM_torvik", "Tempo_torvik"],
    "kenpom_ratings.csv": ["Team", "AdjEM", "Tempo"],
    "evanmiya_ratings.csv": ["Team", "AdjEM", "Tempo"],
    "injury_adjustments.csv": ["Team", "Player", "AdjEM_delta", "Status", "Note"],
    "bracket_round1.csv": ["TeamA", "TeamB"],
    "play_in.csv": ["Slot", "Team1", "Team2"],
}


def check_file(name: str):
    path = ROOT / name
    if not path.exists():
        return False, f"missing file: {name}"

    if path.stat().st_size == 0:
        return False, f"empty file: {name}"

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        return False, f"could not read {name}: {exc}"

    required = REQUIRED_FILES[name]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return False, f"{name} missing columns: {', '.join(missing)}"

    if name in {"kenpom_ratings.csv", "evanmiya_ratings.csv"} and df.empty:
        return False, f"{name} has headers only; paste current ratings first"

    return True, f"{name}: ok ({len(df)} rows)"


def run_step(cmd):
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="Validate BracketLab inputs and rebuild ratings.")
    parser.add_argument("--skip-torvik-fetch", action="store_true", help="Skip fetching fresh Torvik ratings")
    parser.add_argument("--skip-rebuild", action="store_true", help="Skip rebuilding master_ratings.csv")
    args = parser.parse_args()

    print("Checking input files...\n")
    failed = False
    for name in REQUIRED_FILES:
        ok, msg = check_file(name)
        print(("OK   " if ok else "WARN ") + msg)
        if not ok and name in {"kenpom_ratings.csv", "evanmiya_ratings.csv", "bracket_round1.csv", "play_in.csv"}:
            failed = True

    if failed:
        print("\nFix the warnings above, then rerun this command.")
        sys.exit(1)

    if not args.skip_torvik_fetch:
        print("\nFetching fresh Torvik ratings...")
        run_step([sys.executable, "fetch_torvik.py"])

    if not args.skip_rebuild:
        print("\nRebuilding master ratings...")
        run_step([sys.executable, "build_master_ratings.py"])

    print("\nDone.")
    print("Next step: streamlit run app.py")


if __name__ == "__main__":
    main()
