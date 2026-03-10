import pandas as pd

BLEND_WEIGHTS = {
    "AdjEM_torvik": 0.40,
    "AdjEM_kenpom": 0.40,
    "AdjEM_evanmiya": 0.20,
}

TEMPO_WEIGHTS = {
    "Tempo_torvik": 0.40,
    "Tempo_kenpom": 0.40,
    "Tempo_evanmiya": 0.20,
}

INJURY_FILE = "injury_adjustments.csv"


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
    "ETSU": "East Tennessee St.",
    "East Tennessee State": "East Tennessee St.",
    "LIU Brooklyn": "LIU",
    "LIU": "LIU",
}


def canonical_team(name: str) -> str:
    s = str(name).strip().replace("*", "")
    s = " ".join(s.split())
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


def load_injury_adjustments(path=INJURY_FILE):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return {}

    required = {"Team", "AdjEM_delta"}
    if not required.issubset(df.columns):
        return {}

    df["Team"] = df["Team"].astype(str).map(canonical_team)
    df["AdjEM_delta"] = pd.to_numeric(df["AdjEM_delta"], errors="coerce").fillna(0.0)
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
