import argparse
import csv
import html
import re
import urllib.request
from pathlib import Path

import pandas as pd

from ratings_utils import ALIASES, NAME_MAP, canonical_team, injury_status_delta


DEFAULT_URL = "https://www.covers.com/sport/basketball/ncaab/injuries"
OUTPUT_COLUMNS = [
    "Team",
    "Player",
    "Pos",
    "Status",
    "Reported",
    "ImpactMultiplier",
    "AutoAdjEM_delta",
    "Source",
    "Note",
]


def fetch_url(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def html_to_text(raw: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", "\n", text)
    text = html.unescape(text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def normalize_raw_line(line: str) -> str:
    return " ".join(str(line).strip().split())


def team_variants() -> list[tuple[str, str]]:
    variants = {}
    for path, col in [
        ("master_ratings.csv", "Team"),
        ("kenpom_ratings.csv", "Team"),
        ("evanmiya_ratings.csv", "Team"),
        ("torvik_ratings.csv", "team"),
    ]:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if col not in df.columns:
            continue
        for value in df[col].dropna().astype(str):
            variants[value.strip()] = canonical_team(value)
    for src, dst in NAME_MAP.items():
        variants[src] = canonical_team(dst)
        variants[dst] = canonical_team(dst)
    for src, dst in ALIASES.items():
        variants[src] = canonical_team(dst)
        variants[dst] = canonical_team(dst)
    return sorted(variants.items(), key=lambda item: len(item[0]), reverse=True)


def line_key(value: str) -> str:
    s = normalize_raw_line(value).lower()
    s = s.replace("&", "and")
    s = s.replace("st.", "state")
    s = s.replace("st ", "state ")
    s = s.replace("saint", "state")
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_team_header(line: str, variants: list[tuple[str, str]]) -> str | None:
    key = line_key(line)
    for variant, canon in variants:
        vkey = line_key(variant)
        if key == vkey or key.startswith(vkey + " " ) or key.startswith(vkey):
            return canon
    return None


ENTRY_RE = re.compile(
    r"^(?P<player>.+?)\s*(?:\|\s*)?(?P<pos>[A-Z]{1,3}(?:-[A-Z]{1,3})?)\s*(?:\|\s*)?(?P<status>(?:Out|Questionable|Doubtful|Probable|Available|Suspended).+?)\s*\|?$"
)
TABLE_ENTRY_RE = re.compile(
    r"^(?P<player>.+?)\s+(?P<pos>[A-Z](?:/[A-Z])?(?:/[A-Z])?)\s+"
    r"(?P<status>Out for season|Out indefinitely|Out|Questionable|Doubtful|Probable)\s+"
    r"(?P<rest>.+?)\s+(?P<updated>[A-Z][a-z]{2} \d{1,2})$"
)
COMPACT_ENTRY_RE = re.compile(
    r"^(?P<player>.+?)\s+(?P<pos>[A-Z](?:/[A-Z])?(?:/[A-Z])?)\s+"
    r"(?P<status>Out for season|Out indefinitely|Out|Questionable|Doubtful|Probable)(?:\s*-\s*(?P<note>.+))?$"
)


def is_header_window(lines: list[str], idx: int) -> bool:
    window = " ".join(lines[idx : idx + 5]).lower()
    return "player" in window and "status" in window


def parse_lines(lines: list[str], source: str) -> list[dict]:
    if any(normalize_raw_line(line).lower() == "player pos status" for line in lines):
        return parse_compact_lines(lines, source)
    if (
        any(normalize_raw_line(line).lower() == "name pos status injury update updated" for line in lines)
        or (
            any("Team Icon" in line for line in lines)
            and any(TABLE_ENTRY_RE.match(line) for line in lines)
        )
    ):
        return parse_table_lines(lines, source)

    variants = team_variants()
    rows = []
    current_team = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue

        team = match_team_header(line, variants)
        if team and i + 2 < len(lines) and re.fullmatch(r"\(\d+\)", lines[i + 1]) and is_header_window(lines, i + 2):
            current_team = team
            i += 2
            while i < len(lines) and not is_header_window(lines, i):
                i += 1
            i += 1
            continue

        if current_team:
            match = ENTRY_RE.match(line)
            if match:
                player = normalize_raw_line(match.group("player"))
                status = normalize_raw_line(match.group("status"))
                reported = ""
                note_lines = []
                j = i + 1
                while j < len(lines):
                    nxt = lines[j]
                    if not nxt:
                        j += 1
                        continue
                    if ENTRY_RE.match(nxt):
                        break
                    if match_team_header(nxt, variants) and j + 1 < len(lines) and re.fullmatch(r"\(\d+\)", lines[j + 1]):
                        break
                    if re.fullmatch(r"\([^)]*\)", nxt) and not reported:
                        reported = nxt.strip("() ")
                    else:
                        note_lines.append(nxt)
                    j += 1
                note = " ".join(note_lines).strip()
                rows.append(
                    {
                        "Team": current_team,
                        "Player": player,
                        "Pos": normalize_raw_line(match.group("pos")),
                        "Status": status,
                        "Reported": reported,
                        "ImpactMultiplier": "",
                        "AutoAdjEM_delta": injury_status_delta(status, note),
                        "Source": source,
                        "Note": note,
                    }
                )
                i = j
                continue

        i += 1

    deduped = []
    seen = set()
    for row in rows:
        key = (row["Team"], row["Player"], row["Status"], row["Reported"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def parse_compact_lines(lines: list[str], source: str) -> list[dict]:
    variants = team_variants()
    rows = []
    current_team = None
    i = 0

    while i < len(lines):
        line = normalize_raw_line(lines[i])
        if not line:
            i += 1
            continue

        if line.lower() == "player pos status":
            i += 1
            continue

        if line == "No injuries to report.":
            i += 1
            continue

        team = match_team_header(line, variants)
        if team:
            current_team = team
            i += 1
            continue

        if re.fullmatch(r"[A-Z .&'()-]+ \(\d+\)", line):
            i += 1
            continue

        if current_team:
            match = COMPACT_ENTRY_RE.match(line)
            if match:
                note = normalize_raw_line(match.group("note") or "")
                reported = ""
                if i + 1 < len(lines):
                    nxt = normalize_raw_line(lines[i + 1])
                    if re.fullmatch(r"\(\s*[A-Z][a-z]{2},\s*[A-Z][a-z]{2}\s+\d{1,2}\)", nxt):
                        reported = nxt.strip("() ")
                        i += 1

                rows.append(
                    {
                        "Team": current_team,
                        "Player": normalize_raw_line(match.group("player")),
                        "Pos": normalize_raw_line(match.group("pos")),
                        "Status": normalize_raw_line(match.group("status")),
                        "Reported": reported,
                        "ImpactMultiplier": "",
                        "AutoAdjEM_delta": injury_status_delta(match.group("status"), note),
                        "Source": source,
                        "Note": note,
                    }
                )
        i += 1

    deduped = []
    seen = set()
    for row in rows:
        key = (row["Team"], row["Player"], row["Status"], row["Reported"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def extract_table_team(line: str, variants: list[tuple[str, str]]) -> str | None:
    if "Team Icon" not in line:
        return None
    left, right = line.split("Team Icon", 1)
    for candidate in [right.strip(), left.strip()]:
        if not candidate:
            continue
        team = match_team_header(candidate, variants)
        if team:
            return team
    return None


def parse_table_lines(lines: list[str], source: str) -> list[dict]:
    variants = team_variants()
    rows = []
    current_team = None

    for line in lines:
        team = extract_table_team(line, variants)
        if team:
            current_team = team
            continue

        if normalize_raw_line(line).lower() == "name pos status injury update updated":
            continue

        if line == "No injuries to report.":
            continue

        if not current_team:
            continue

        match = TABLE_ENTRY_RE.match(line)
        if not match:
            continue

        player = normalize_raw_line(match.group("player"))
        status = normalize_raw_line(match.group("status"))
        rest = normalize_raw_line(match.group("rest"))
        updated = normalize_raw_line(match.group("updated"))
        rows.append(
            {
                "Team": current_team,
                "Player": player,
                "Pos": normalize_raw_line(match.group("pos")),
                "Status": status,
                "Reported": updated,
                "ImpactMultiplier": "",
                "AutoAdjEM_delta": injury_status_delta(status, rest),
                "Source": source,
                "Note": rest,
            }
        )

    deduped = []
    seen = set()
    for row in rows:
        key = (row["Team"], row["Player"], row["Status"], row["Reported"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def write_rows(rows: list[dict], output_path: Path):
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})


def main():
    parser = argparse.ArgumentParser(description="Fetch or parse Covers NCAAB injuries into injury_reports.csv.")
    parser.add_argument("--input", help="Optional raw text/html file copied from Covers")
    parser.add_argument("--output", default="injury_reports.csv", help="Output CSV path")
    parser.add_argument("--url", default=DEFAULT_URL, help="Source URL when --input is not provided")
    args = parser.parse_args()

    if args.input:
        raw = Path(args.input).read_text()
        source = f"covers_raw:{Path(args.input).name}"
    else:
        raw = fetch_url(args.url)
        source = args.url

    text = html_to_text(raw) if "<html" in raw.lower() else raw
    lines = [normalize_raw_line(line) for line in text.splitlines() if normalize_raw_line(line)]
    rows = parse_lines(lines, source)
    write_rows(rows, Path(args.output))
    print(f"Wrote {len(rows)} injuries to {args.output}")


if __name__ == "__main__":
    main()
