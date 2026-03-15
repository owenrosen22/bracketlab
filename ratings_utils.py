import re

import pandas as pd

BLEND_WEIGHTS = {
    "AdjEM_torvik": 0.375,
    "AdjEM_kenpom": 0.375,
    "AdjEM_evanmiya": 0.25,
}

TEMPO_WEIGHTS = {
    "Tempo_torvik": 0.375,
    "Tempo_kenpom": 0.375,
    "Tempo_evanmiya": 0.25,
}

INJURY_FILE = "injury_adjustments.csv"
AUTO_INJURY_FILE = "injury_reports.csv"


INJURY_STATUS_SCALE = {
    "questionable": -0.015,
    "doubtful": -0.045,
    "out": -0.08,
    "out_indefinitely": -0.11,
    "out_for_season": -0.16,
}

AUTO_INJURY_POSITION_MULTIPLIER = {
    "G": 1.08,
    "G/F": 1.02,
    "F/G": 1.02,
    "F": 0.95,
    "F/C": 0.91,
    "C/F": 0.91,
    "C": 0.87,
}

AUTO_INJURY_DIMINISHING_FACTORS = [1.00, 0.85, 0.70, 0.58, 0.48, 0.40]
AUTO_INJURY_TEAM_CAP = -0.42


def load_name_map(path="name_map.csv"):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return {}

    if "From" not in df.columns or "To" not in df.columns:
        return {}

    df["From"] = df["From"].astype(str).str.strip()
    df["To"] = df["To"].astype(str).str.strip()
    return dict(zip(df["From"], df["To"]))


NAME_MAP = load_name_map()


ALIASES = {
    "Arkansas-Little Rock": "Little Rock",
    "Arkansas-Pine Bluff": "Arkansas Pine Bluff",
    "Cal St. Northridge": "CSUN",
    "Cal State Northridge": "CSUN",
    "California Baptist": "Cal Baptist",
    "College of Charleston": "Charleston",
    "Detroit": "Detroit Mercy",
    "ETSU": "East Tennessee St.",
    "East Tennessee State": "East Tennessee St.",
    "FIU": "FIU",
    "Florida International": "FIU",
    "Fort Wayne": "Purdue Fort Wayne",
    "Gardner-Webb": "Gardner Webb",
    "Grambling": "Grambling St.",
    "Illinois-Chicago": "Illinois Chicago",
    "LIU Brooklyn": "LIU",
    "LIU": "LIU",
    "Louisiana-Lafayette": "Louisiana",
    "Louisiana-Monroe": "Louisiana Monroe",
    "Loyola Maryland": "Loyola MD",
    "Maryland-Eastern Shore": "Maryland Eastern Shore",
    "McNeese St.": "McNeese",
    "Miami (Fla.)": "Miami FL",
    "Mississippi State": "Mississippi St.",
    "Missouri-Kansas City": "Kansas City",
    "Nicholls St.": "Nicholls",
    "Ohio State": "Ohio St.",
    "Ole Miss": "Mississippi",
    "Omaha": "Nebraska Omaha",
    "Prairie View": "Prairie View A&M",
    "San Diego State": "San Diego St.",
    "South Carolina Upstate": "USC Upstate",
    "Southern Mississippi": "Southern Miss",
    "Southeast Missouri St.": "Southeast Missouri",
    "Southeast Missouri State": "Southeast Missouri",
    "St. Francis": "Saint Francis",
    "St. Francis (PA)": "Saint Francis",
    "St Thomas": "St. Thomas",
    "St. Thomas (MN)": "St. Thomas",
    "Tennessee-Martin": "Tennessee Martin",
    "Texas A&M-Corpus Christi": "Texas A&M Corpus Chris",
    "Texas-Rio Grande Valley": "UT Rio Grande Valley",
    "UMKC": "Kansas City",
    "Washington State": "Washington St.",
    "SIUE": "SIU Edwardsville",
}


def normalize_common_abbreviations(name: str) -> str:
    s = str(name)
    # Normalize common "State" variants so Torvik/KenPom and EvanMiya merge.
    s = re.sub(r"\bState\b", "St.", s)
    s = re.sub(r"\bSt\b(?!\.)", "St.", s)
    s = re.sub(r"\bSaint\b", "St.", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def canonical_team(name: str) -> str:
    s = str(name).strip().replace("*", "")
    s = " ".join(s.split())
    s = NAME_MAP.get(s, s)
    s = normalize_common_abbreviations(s)
    s = NAME_MAP.get(s, s)
    s = ALIASES.get(s, s)
    return s


def weighted_average(row, weights):
    num = 0.0
    den = 0.0
    used = []
    for col, weight in weights.items():
        if col in row and pd.notna(row[col]):
            num += float(row[col]) * weight
            den += weight
            used.append(col.split("_", 1)[1])
    value = num / den if den else pd.NA
    return value, ",".join(used)


def _load_manual_injury_adjustments(path=INJURY_FILE):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Team", "Player", "Pos", "AdjEM_delta", "Status", "Note", "SourceType"])

    required = {"Team", "AdjEM_delta"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["Team", "Player", "Pos", "AdjEM_delta", "Status", "Note", "SourceType"])

    df["Team"] = df["Team"].astype(str).map(canonical_team)
    if "Player" not in df.columns:
        df["Player"] = ""
    if "Pos" not in df.columns:
        df["Pos"] = ""
    if "Status" not in df.columns:
        df["Status"] = "Manual"
    if "Note" not in df.columns:
        df["Note"] = ""
    df["Player"] = df["Player"].astype(str).str.strip()
    df["Pos"] = df["Pos"].astype(str).str.strip()
    df["Status"] = df["Status"].astype(str).str.strip()
    df["Note"] = df["Note"].astype(str).str.strip()
    df["AdjEM_delta"] = pd.to_numeric(df["AdjEM_delta"], errors="coerce").fillna(0.0)
    df["SourceType"] = "manual"
    return df[["Team", "Player", "Pos", "AdjEM_delta", "Status", "Note", "SourceType"]]


def injury_status_delta(status: str, note: str = "") -> float:
    text = f"{status} {note}".lower()
    if "redshirt" in text:
        return 0.0
    if "probable" in text or "expected to return" in text:
        return 0.0
    if "out for season" in text or "season-ending" in text or "out for the season" in text:
        return INJURY_STATUS_SCALE["out_for_season"]
    if "questionable" in text or "uncertain" in text or "day-to-day" in text:
        return INJURY_STATUS_SCALE["questionable"]
    if "doubtful" in text:
        return INJURY_STATUS_SCALE["doubtful"]
    if "out" in text:
        if any(term in text for term in ["remainder", "entire season", "rest of the season", "miss the 2025-26 season", "miss the 2025-2026 season", "torn acl", "achilles", "surgery"]):
            return INJURY_STATUS_SCALE["out_for_season"]
        if "no timetable" in text or "indefinite" in text:
            return INJURY_STATUS_SCALE["out_indefinitely"]
        return INJURY_STATUS_SCALE["out"]
    return 0.0


def _position_multiplier(pos: str) -> float:
    return AUTO_INJURY_POSITION_MULTIPLIER.get(str(pos).strip().upper(), 0.96)


def _note_multiplier(note: str) -> float:
    text = str(note).lower()
    if "redshirt" in text:
        return 0.0
    if "left team" in text:
        return 0.60
    if "personal" in text:
        return 0.65
    if "illness" in text:
        return 0.75
    if "concussion" in text:
        return 0.90
    if any(term in text for term in ["acl", "achilles", "knee", "foot", "ankle", "back", "shoulder", "wrist", "arm", "leg", "hip", "lower body", "groin"]):
        return 1.05
    return 1.0


def _apply_auto_team_context(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    weighted_groups = []
    for _, group in df.groupby("Team", sort=False):
        working = group.copy()
        working["_abs"] = working["AdjEM_delta"].abs()
        working = working.sort_values("_abs", ascending=False).reset_index(drop=True)

        factors = AUTO_INJURY_DIMINISHING_FACTORS
        working["AdjEM_delta"] = [
            float(delta) * factors[idx] if idx < len(factors) else float(delta) * factors[-1]
            for idx, delta in enumerate(working["AdjEM_delta"])
        ]

        total = float(working["AdjEM_delta"].sum())
        if total < AUTO_INJURY_TEAM_CAP:
            scale = AUTO_INJURY_TEAM_CAP / total
            working["AdjEM_delta"] = working["AdjEM_delta"] * scale

        weighted_groups.append(working.drop(columns="_abs"))

    return pd.concat(weighted_groups, ignore_index=True)


def _load_auto_injury_reports(path=AUTO_INJURY_FILE):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["Team", "Player", "Pos", "AdjEM_delta", "Status", "Note", "SourceType"])

    required = {"Team", "Player", "Status"}
    if not required.issubset(df.columns):
        return pd.DataFrame(columns=["Team", "Player", "Pos", "AdjEM_delta", "Status", "Note", "SourceType"])

    df["Team"] = df["Team"].astype(str).map(canonical_team)
    df["Player"] = df["Player"].astype(str).str.strip()
    if "Pos" not in df.columns:
        df["Pos"] = ""
    df["Pos"] = df["Pos"].astype(str).str.strip()
    df["Status"] = df["Status"].astype(str).str.strip()
    if "Note" not in df.columns:
        df["Note"] = ""
    df["Note"] = df["Note"].astype(str).str.strip()
    if "ImpactMultiplier" not in df.columns:
        df["ImpactMultiplier"] = df["Pos"].map(_position_multiplier)
    else:
        df["ImpactMultiplier"] = pd.to_numeric(df["ImpactMultiplier"], errors="coerce")
        missing_multiplier = df["ImpactMultiplier"].isna() | (df["ImpactMultiplier"] <= 0)
        df.loc[missing_multiplier, "ImpactMultiplier"] = df.loc[missing_multiplier, "Pos"].map(_position_multiplier)
    df["ImpactMultiplier"] = pd.to_numeric(df["ImpactMultiplier"], errors="coerce").fillna(0.94).clip(lower=0.0)
    if "AutoAdjEM_delta" in df.columns:
        df["AdjEM_delta"] = pd.to_numeric(df["AutoAdjEM_delta"], errors="coerce")
    else:
        df["AdjEM_delta"] = pd.NA
    missing = df["AdjEM_delta"].isna()
    df.loc[missing, "AdjEM_delta"] = df.loc[missing].apply(
        lambda row: injury_status_delta(row.get("Status", ""), row.get("Note", "")) * float(row.get("ImpactMultiplier", 1.0)),
        axis=1,
    )
    df["AdjEM_delta"] = pd.to_numeric(df["AdjEM_delta"], errors="coerce").fillna(0.0)
    df["SourceType"] = "auto"
    df["AdjEM_delta"] = df["AdjEM_delta"] * df["ImpactMultiplier"] * df["Note"].map(_note_multiplier)
    df = _apply_auto_team_context(df)
    return df[["Team", "Player", "Pos", "AdjEM_delta", "Status", "Note", "SourceType"]]


def load_injury_adjustments_table():
    auto_df = _load_auto_injury_reports()
    manual_df = _load_manual_injury_adjustments()

    if auto_df.empty and manual_df.empty:
        return pd.DataFrame(columns=["Team", "Player", "Pos", "AdjEM_delta", "Status", "Note", "SourceType"])

    if not auto_df.empty and not manual_df.empty:
        manual_keys = set(
            zip(
                manual_df["Team"].astype(str).str.lower(),
                manual_df["Player"].astype(str).str.lower(),
            )
        )
        auto_df = auto_df[
            ~auto_df.apply(
                lambda row: (str(row["Team"]).lower(), str(row["Player"]).lower()) in manual_keys,
                axis=1,
            )
        ]

    return pd.concat([auto_df, manual_df], ignore_index=True)


def load_injury_adjustments(path=INJURY_FILE):
    df = load_injury_adjustments_table()
    if df.empty:
        return {}
    grouped = df.groupby("Team", as_index=False)["AdjEM_delta"].sum()
    return dict(zip(grouped["Team"], grouped["AdjEM_delta"]))


def prepare_ratings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Team" not in out.columns:
        raise ValueError("master_ratings.csv must include a Team column")

    out["Team"] = out["Team"].astype(str).map(canonical_team)

    has_adj_sources = any(col in out.columns for col in BLEND_WEIGHTS)
    if has_adj_sources:
        blends = out.apply(lambda row: weighted_average(row, BLEND_WEIGHTS), axis=1, result_type="expand")
        out["AdjEM_blend"] = pd.to_numeric(blends[0], errors="coerce")
        out["sources_used"] = blends[1]
    else:
        out["AdjEM_blend"] = pd.to_numeric(out.get("AdjEM_blend"), errors="coerce")
        if "sources_used" not in out.columns:
            out["sources_used"] = ""

    has_tempo_sources = any(col in out.columns for col in TEMPO_WEIGHTS)
    if has_tempo_sources:
        tempos = out.apply(lambda row: weighted_average(row, TEMPO_WEIGHTS), axis=1, result_type="expand")
        out["Tempo_blend"] = pd.to_numeric(tempos[0], errors="coerce")
    elif "Tempo_blend" in out.columns:
        out["Tempo_blend"] = pd.to_numeric(out["Tempo_blend"], errors="coerce")

    injury = load_injury_adjustments()
    out["InjuryAdj"] = out["Team"].map(injury).fillna(0.0)
    out["AdjEM_current"] = pd.to_numeric(out["AdjEM_blend"], errors="coerce").fillna(0.0) + out["InjuryAdj"]

    if "sources_used" not in out.columns:
        out["sources_used"] = ""
    out["source_count"] = out["sources_used"].fillna("").map(lambda s: 0 if not s else len([p for p in str(s).split(",") if p]))

    return out
