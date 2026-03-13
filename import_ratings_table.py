import argparse
import csv
import re
from pathlib import Path


def clean_team_name(name: str) -> str:
    text = re.sub(r"[^\x00-\x7F]+", "", str(name).strip())
    text = re.sub(r"\s+\d+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_delimiter(sample: str) -> str:
    if "\t" in sample:
        return "\t"
    return ","


def parse_rows(text: str):
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    delimiter = detect_delimiter(lines[0])
    return list(csv.reader(lines, delimiter=delimiter))


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def find_column(header, candidates):
    normalized = [normalize_header(col) for col in header]
    for idx, col in enumerate(normalized):
        if col in candidates:
            return idx
    return None


def extract_kenpom(rows):
    header = rows[0]
    sample_data_row = next((row for row in rows[1:] if row and row[0].strip().isdigit()), [])
    expanded_kenpom_copy = len(sample_data_row) >= 10 and len(sample_data_row) > len(header)

    if expanded_kenpom_copy:
        rank_idx = 0
        team_idx = 1
        adj_idx = 4
        tempo_idx = 9
    else:
        rank_idx = find_column(header, {"rk", "rank"})
        team_idx = find_column(header, {"team"})
        adj_idx = find_column(header, {"netrtg", "adjem"})
        tempo_idx = find_column(header, {"adjt", "tempo"})
        if None in {team_idx, adj_idx, tempo_idx}:
            raise ValueError("Could not find Team / NetRtg or AdjEM / AdjT or Tempo columns in KenPom paste.")

    out = []
    for row in rows[1:]:
        if len(row) <= max(team_idx, adj_idx, tempo_idx):
            continue
        if row[team_idx].strip() == "Team":
            continue
        if row[0].strip() == "Strength of Schedule":
            continue
        if rank_idx is not None and not row[rank_idx].strip().isdigit():
            continue
        team = clean_team_name(row[team_idx])
        adj = row[adj_idx].replace("+", "").strip()
        tempo = row[tempo_idx].strip()
        if not team or not adj or not tempo:
            continue
        out.append((team, adj, tempo))
    return out


def extract_evanmiya(rows):
    if rows and len(rows[0]) == 1:
        return extract_evanmiya_line_blocks(rows)

    header = rows[0]
    rank_idx = find_column(header, {"rk", "rank"})
    team_idx = find_column(header, {"team"})
    adj_idx = find_column(header, {"adjem", "teamrating", "rating"})
    tempo_idx = find_column(header, {"tempo", "adjt", "pace"})
    if None in {team_idx, adj_idx, tempo_idx}:
        raise ValueError("Could not find Team / AdjEM or Rating / Tempo columns in EvanMiya paste.")

    out = []
    for row in rows[1:]:
        if len(row) <= max(team_idx, adj_idx, tempo_idx):
            continue
        if row[team_idx].strip() == "Team":
            continue
        if rank_idx is not None and not row[rank_idx].strip().isdigit():
            continue
        team = clean_team_name(row[team_idx])
        adj = row[adj_idx].replace("+", "").strip()
        tempo = row[tempo_idx].strip()
        if not team or not adj or not tempo:
            continue
        out.append((team, adj, tempo))
    return out


def extract_evanmiya_line_blocks(rows):
    tokens = [row[0].strip() for row in rows if row and row[0].strip()]
    if tokens:
        m = re.search(r"(\d+)\s*$", tokens[0])
        if m and not tokens[0].strip().isdigit():
            tokens[0] = m.group(1)
    out = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.isdigit():
            i += 1
            continue

        if i + 10 >= len(tokens):
            break

        raw_team = tokens[i + 1]
        if not re.search(r"[A-Za-z]", raw_team):
            i += 1
            continue

        team = clean_team_name(raw_team)
        try:
            adj = tokens[i + 4].replace("+", "").strip()
            tempo = tokens[i + 9].strip()
            float(adj)
            float(tempo)
        except ValueError:
            i += 1
            continue

        out.append((team, adj, tempo))
        i += 21

    if not out:
        raise ValueError("Could not parse EvanMiya line-block paste.")

    return out


def write_output(rows, output_path: Path):
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Team", "AdjEM", "Tempo"])
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert pasted ratings table text into Team,AdjEM,Tempo CSV.")
    parser.add_argument("--source", choices=["kenpom", "evanmiya"], required=True)
    parser.add_argument("--input", required=True, help="Path to pasted raw text file")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    args = parser.parse_args()

    raw_text = Path(args.input).read_text()
    rows = parse_rows(raw_text)
    if not rows:
        raise SystemExit("Input file is empty.")

    extracted = extract_kenpom(rows) if args.source == "kenpom" else extract_evanmiya(rows)
    if not extracted:
        raise SystemExit("No data rows found.")

    write_output(extracted, Path(args.output))
    print(f"Wrote {len(extracted)} rows to {args.output}")


if __name__ == "__main__":
    main()
