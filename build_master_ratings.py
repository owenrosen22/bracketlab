import pandas as pd
from ratings_utils import canonical_team, prepare_ratings

# ---- weights (edit later if you want) ----
W_TORVIK = 0.40
W_KENPOM = 0.40
W_EVANMIYA = 0.20


def norm_team(s: str) -> str:
    return canonical_team(s).replace("St.", "St").replace("St  ", "St ")

def safe_read_csv(path: str):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

def weighted_avg(row, cols, weights):
    num = 0.0
    den = 0.0
    used = []
    for c, w, tag in cols:
        if c in row and pd.notna(row[c]):
            num += float(row[c]) * w
            den += w
            used.append(tag)
    return (num / den if den > 0 else pd.NA), ",".join(used)

def main():
    # -----------------------
    # TORVIK
    # -----------------------
    torvik = pd.read_csv("torvik_ratings.csv")
    torvik.columns = [c.lower() for c in torvik.columns]
    torvik["Team"] = torvik["team"].map(norm_team)
    torvik["AdjEM_torvik"] = torvik["adjoe"] - torvik["adjde"]
    torvik["Tempo_torvik"] = torvik["adjt"]
    torvik = torvik[["Team", "AdjEM_torvik", "Tempo_torvik"]]

    base = torvik.copy()

    # -----------------------
    # KENPOM (optional)
    # expected columns (you can change later): Team, AdjEM, Tempo
    # -----------------------
    kp = safe_read_csv("kenpom_ratings.csv")
    if kp is not None:
        kp["Team"] = kp["Team"].map(norm_team)
        if "AdjEM" not in kp.columns:
            raise ValueError("kenpom_ratings.csv must include column AdjEM")
        kp2 = kp[["Team", "AdjEM"]].rename(columns={"AdjEM": "AdjEM_kenpom"})
        if "Tempo" in kp.columns:
            kp2["Tempo_kenpom"] = kp["Tempo"]
        base = base.merge(kp2, on="Team", how="outer")

    # -----------------------
    # EVANMIYA (optional)
    # expected columns: Team, AdjEM (Tempo optional)
    # -----------------------
    em = safe_read_csv("evanmiya_ratings.csv")
    if em is not None:
        # If EvanMiya uses different column names, we’ll adjust once you show the header.
        if "Team" not in em.columns:
            raise ValueError("evanmiya_ratings.csv must include column Team")
        if "AdjEM" not in em.columns:
            raise ValueError("evanmiya_ratings.csv must include column AdjEM")
        em["Team"] = em["Team"].map(norm_team)
        em2 = em[["Team", "AdjEM"]].rename(columns={"AdjEM": "AdjEM_evanmiya"})
        if "Tempo" in em.columns:
            em2["Tempo_evanmiya"] = em["Tempo"]
        base = base.merge(em2, on="Team", how="outer")

    # -----------------------
    # KAGGLE (optional)
    # expected columns: Team, AvgDiff, WinPct
    # We'll treat AvgDiff as a rough strength proxy if you want it later
    # -----------------------
    kg = safe_read_csv("kaggle_ratings.csv")
    if kg is not None:
        if "Team" not in kg.columns:
            raise ValueError("kaggle_ratings.csv must include column Team")
        kg["Team"] = kg["Team"].map(norm_team)
        base = base.merge(kg, on="Team", how="left")

    # -----------------------
    # BLEND AdjEM + Tempo
    # -----------------------
    def blend_row(r):
        adj, used = weighted_avg(
            r,
            cols=[
                ("AdjEM_torvik", W_TORVIK, "torvik"),
                ("AdjEM_kenpom", W_KENPOM, "kenpom"),
                ("AdjEM_evanmiya", W_EVANMIYA, "evanmiya"),
            ],
            weights=None
        )
        # tempo: same weights, but only if columns exist
        tempo, used_t = weighted_avg(
            r,
            cols=[
                ("Tempo_torvik", W_TORVIK, "torvik"),
                ("Tempo_kenpom", W_KENPOM, "kenpom"),
                ("Tempo_evanmiya", W_EVANMIYA, "evanmiya"),
            ],
            weights=None
        )
        return pd.Series({"AdjEM_blend": adj, "Tempo_blend": tempo, "sources_used": used})

    blended = base.apply(blend_row, axis=1)
    out = pd.concat([base, blended], axis=1)

    out = prepare_ratings(out)
    out = out.sort_values("AdjEM_current", ascending=False)
    out.to_csv("master_ratings.csv", index=False)

    print("Created master_ratings.csv with", len(out), "teams")
    print(out[["Team", "AdjEM_blend", "AdjEM_current", "sources_used"]].head(20).to_string(index=False))

if __name__ == "__main__":
    main()
    
