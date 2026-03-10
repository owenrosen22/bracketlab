import urllib.parse
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import streamlit.components.v1 as components

from predict_proba import win_prob

st.set_page_config(page_title="BracketLab", page_icon="🏀", layout="wide")

REGION_NAMES = ["East", "West", "South", "Midwest"]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', system-ui, sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(59,130,246,0.18), transparent 28%),
            radial-gradient(circle at top right, rgba(16,185,129,0.14), transparent 24%),
            linear-gradient(180deg, #081120 0%, #0b1220 35%, #0f172a 100%);
    }

    .block-container {
        max-width: 1520px;
        padding-top: 1.35rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .stDeployButton, header[data-testid="stHeader"], #MainMenu, footer {
        visibility: hidden;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(12,19,33,0.96), rgba(9,14,24,0.96));
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #f8fafc;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.10);
        background:
            linear-gradient(135deg, rgba(37,99,235,0.30), rgba(16,185,129,0.18)),
            linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        border-radius: 28px;
        padding: 1.45rem 1.5rem 1.3rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow:
            0 20px 60px rgba(0,0,0,0.35),
            0 0 60px rgba(59,130,246,0.15);
    }

    .hero:before {
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 80% 20%, rgba(255,255,255,0.12), transparent 22%);
        pointer-events: none;
    }

    .hero-kicker {
        margin: 0;
        color: #bfdbfe;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.78rem;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0.18rem 0 0 0;
        line-height: 1.02;
    }

    .hero-sub {
        margin-top: 0.65rem;
        margin-bottom: 0;
        color: #dbeafe;
        font-size: 1rem;
        line-height: 1.45;
        max-width: 920px;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.95rem;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        color: #e2e8f0;
        border-radius: 999px;
        padding: 0.45rem 0.75rem;
        font-size: 0.86rem;
        font-weight: 600;
    }

    .section-card {
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.02));
        border-radius: 22px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.18);
        backdrop-filter: blur(8px);
    }

    .section-title {
        margin: 0 0 0.2rem 0;
        font-size: 1.12rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }

    .section-sub {
        margin: 0;
        color: #94a3b8;
        font-size: 0.88rem;
    }

    .insight-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin-bottom: 1rem;
    }

    .insight-card {
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.025));
        padding: 0.95rem 1rem;
        box-shadow: 0 12px 28px rgba(0,0,0,0.25);
    }

    .insight-label {
        color: #93c5fd;
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .insight-value {
        margin-top: 0.35rem;
        font-size: 1.02rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.2;
    }

    .insight-sub {
        margin-top: 0.25rem;
        color: #94a3b8;
        font-size: 0.82rem;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 0.8rem 0.9rem;
        box-shadow: 0 10px 24px rgba(0,0,0,0.18);
    }

    .sidebar-brand {
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(135deg, rgba(59,130,246,0.18), rgba(16,185,129,0.12));
        border-radius: 20px;
        padding: 1rem;
        margin-bottom: 0.9rem;
    }

    .sidebar-brand h3 {
        margin: 0 0 0.25rem 0;
        font-size: 1.1rem;
        font-weight: 800;
    }

    .sidebar-brand p {
        margin: 0;
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.4;
    }

    .mini-guide {
        margin-top: 1rem;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 0.9rem;
        background: rgba(255,255,255,0.03);
    }

    .mini-guide h4 {
        margin: 0 0 0.45rem 0;
        font-size: 0.95rem;
        font-weight: 800;
    }

    .mini-guide p {
        margin: 0.2rem 0;
        color: #cbd5e1;
        font-size: 0.82rem;
    }

    .stButton > button, .stDownloadButton > button {
        width: 100%;
        border-radius: 14px;
        border: none;
        background: linear-gradient(135deg,#2563eb,#10b981);
        color: white;
        font-weight: 700;
        padding: 0.75rem 1rem;
        box-shadow: 0 12px 28px rgba(0,0,0,0.25);
        transition: all 0.15s ease;
    }

    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 40px rgba(0,0,0,0.35);
    }

    .footer-note {
        color: #94a3b8;
        text-align: center;
        font-size: 0.8rem;
        margin-top: 1rem;
    }

    .matchup-card {
        padding: 1.2rem 1.25rem;
    }

    .matchup-grid {
        display: grid;
        grid-template-columns: 1.55fr 220px 1.55fr;
        align-items: center;
        gap: 24px;
    }

    .matchup-team {
        display: flex;
        align-items: center;
        gap: 16px;
        min-width: 0;
    }

    .matchup-team.right {
        justify-content: flex-end;
    }

    .matchup-logo {
        width: 96px;
        height: 96px;
        border-radius: 999px;
        object-fit: cover;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        box-shadow: 0 10px 24px rgba(0,0,0,0.22);
        flex-shrink: 0;
    }

    .matchup-name {
        font-size: 1.3rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.14;
        word-break: break-word;
    }

    .matchup-sub {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 4px;
    }

    .matchup-center {
        text-align: center;
    }

    .matchup-vs {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 86px;
        height: 86px;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(37,99,235,0.78), rgba(16,185,129,0.70));
        border: 1px solid rgba(255,255,255,0.12);
        font-weight: 800;
        color: #fff;
        font-size: 1.1rem;
        box-shadow: 0 12px 28px rgba(0,0,0,0.24);
    }

    .matchup-edge-label {
        margin-top: 12px;
        font-size: 0.78rem;
        color: #93c5fd;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .matchup-prob {
        font-size: 1.95rem;
        font-weight: 800;
        color: #f8fafc;
    }

    .matchup-bar {
        height: 10px;
        background: rgba(255,255,255,0.08);
        border-radius: 999px;
        overflow: hidden;
        margin-top: 10px;
    }

    .matchup-bar-fill {
        height: 100%;
        background: linear-gradient(90deg,#2563eb,#10b981);
    }

    .matchup-split {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 8px;
    }

    .matchup-text.right {
        text-align: right;
    }

    .mobile-bracket-note {
        display: none;
        margin-top: 0.5rem;
        color: #94a3b8;
        font-size: 0.82rem;
    }

    .desktop-only {
        display: block;
    }

    .tablet-only, .mobile-only {
        display: none;
    }

    @media (max-width: 1450px) {
        .block-container {
            max-width: 100%;
            padding-left: 0.9rem;
            padding-right: 0.9rem;
        }

        .hero-title {
            font-size: 2.15rem;
        }

        .insight-strip {
            grid-template-columns: 1fr 1fr;
        }

        .matchup-grid {
            grid-template-columns: 1.3fr 180px 1.3fr;
            gap: 18px;
        }

        .matchup-logo {
            width: 76px;
            height: 76px;
        }

        .matchup-name {
            font-size: 1.08rem;
        }

        .matchup-vs {
            width: 72px;
            height: 72px;
            font-size: 1rem;
        }

        .matchup-prob {
            font-size: 1.6rem;
        }

        .desktop-only {
            display: none;
        }

        .tablet-only {
            display: block;
        }
    }

    @media (max-width: 1180px) {
        .hero-title {
            font-size: 1.95rem;
        }

        .matchup-card {
            padding: 1rem 1rem;
        }

        .matchup-grid {
            grid-template-columns: 1fr;
            gap: 16px;
        }

        .matchup-center {
            order: -1;
        }

        .matchup-team,
        .matchup-team.right {
            justify-content: center;
            gap: 12px;
        }

        .matchup-logo {
            width: 72px;
            height: 72px;
        }

        .matchup-name {
            font-size: 1rem;
        }

        .matchup-prob {
            font-size: 1.5rem;
        }
    }

    @media (max-width: 900px) {
        .hero {
            padding: 1.1rem 1rem;
            border-radius: 22px;
        }

        .hero-title {
            font-size: 1.65rem;
        }

        .hero-sub {
            font-size: 0.92rem;
        }

        .pill {
            font-size: 0.78rem;
            padding: 0.38rem 0.62rem;
        }

        .section-card {
            padding: 0.9rem 0.95rem;
            border-radius: 18px;
        }

        .insight-strip {
            grid-template-columns: 1fr;
        }

        .matchup-card {
            padding: 1rem 0.9rem;
        }

        .matchup-grid {
            grid-template-columns: 1fr;
            gap: 16px;
        }

        .matchup-center {
            order: -1;
        }

        .matchup-team,
        .matchup-team.right {
            justify-content: center;
            flex-direction: column;
            text-align: center;
            gap: 12px;
        }

        .matchup-text,
        .matchup-text.right {
            text-align: center;
        }

        .matchup-logo {
            width: 82px;
            height: 82px;
        }

        .matchup-name {
            font-size: 1.08rem;
        }

        .matchup-sub {
            font-size: 0.82rem;
        }

        .matchup-vs {
            width: 70px;
            height: 70px;
            font-size: 0.98rem;
        }

        .matchup-edge-label {
            font-size: 0.72rem;
            margin-top: 10px;
        }

        .matchup-prob {
            font-size: 1.62rem;
        }

        .matchup-bar {
            height: 9px;
        }

        .matchup-split {
            font-size: 0.75rem;
        }

        .mobile-bracket-note {
            display: block;
        }

        .tablet-only {
            display: none;
        }

        .mobile-only {
            display: block;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_name_map():
    try:
        df = pd.read_csv("name_map.csv")
        df["From"] = df["From"].astype(str).str.strip()
        df["To"] = df["To"].astype(str).str.strip()
        return dict(zip(df["From"], df["To"]))
    except Exception:
        return {}


def clean_name(name, name_map):
    name = str(name).strip()
    name = name.replace("*", "")
    name = " ".join(name.split())
    return name_map.get(name, name)


def load_team_logos():
    try:
        df = pd.read_csv("team_logos.csv")
        df["Team"] = df["Team"].astype(str).str.strip()
        df["LogoURL"] = df["LogoURL"].astype(str).str.strip()
        df = df[df["LogoURL"] != ""]
        return dict(zip(df["Team"], df["LogoURL"]))
    except Exception:
        return {}


TEAM_LOGOS = load_team_logos()


def monogram_logo_data_uri(team: str):
    parts = [p for p in str(team).replace(".", "").split() if p]
    initials = "".join(p[0] for p in parts[:2]).upper() or "TM"
    svg = f"""
    <svg xmlns='http://www.w3.org/2000/svg' width='84' height='84' viewBox='0 0 84 84'>
      <defs>
        <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
          <stop offset='0%' stop-color='#2563eb'/>
          <stop offset='100%' stop-color='#10b981'/>
        </linearGradient>
      </defs>
      <circle cx='42' cy='42' r='39' fill='url(#g)'/>
      <circle cx='42' cy='42' r='36' fill='none' stroke='rgba(255,255,255,0.40)' stroke-width='2.5'/>
      <text x='42' y='50' text-anchor='middle' font-family='Inter, Arial, sans-serif' font-size='30' font-weight='800' fill='white'>{initials}</text>
    </svg>
    """
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)


def logo_url(team: str):
    team = str(team).strip()
    return TEAM_LOGOS.get(team, "")


def logo_src(team: str):
    url = logo_url(team)
    return url if url else monogram_logo_data_uri(team)


@st.cache_data
def load_master_ratings():
    df = pd.read_csv("master_ratings.csv")
    df["Team"] = df["Team"].astype(str).str.strip()
    return df


@st.cache_data
def load_playins():
    name_map = load_name_map()
    df = pd.read_csv("play_in.csv")
    df["Team1"] = df["Team1"].apply(lambda x: clean_name(x, name_map))
    df["Team2"] = df["Team2"].apply(lambda x: clean_name(x, name_map))
    return list(zip(df["Slot"], df["Team1"], df["Team2"]))


@st.cache_data
def load_round1():
    name_map = load_name_map()
    df = pd.read_csv("bracket_round1.csv")
    df["TeamA"] = df["TeamA"].apply(lambda x: clean_name(x, name_map))
    df["TeamB"] = df["TeamB"].apply(lambda x: clean_name(x, name_map))
    return list(zip(df["TeamA"], df["TeamB"]))


def simulate_game(a, b, rng):
    p = win_prob(a, b)
    return a if rng.random() < p else b


def resolve_playins(playins, rng):
    winners = {}
    for slot, a, b in playins:
        winners[slot] = simulate_game(a, b, rng)
    return winners


def substitute_slots(round1, winners):
    new_games = []
    for a, b in round1:
        a = winners.get(a, a)
        b = winners.get(b, b)
        new_games.append((a, b))
    return new_games


def simulate_tournament(games, rng):
    teams = set([t for g in games for t in g])
    depth = {t: 0 for t in teams}
    current = games
    r = 1
    while True:
        winners = []
        for a, b in current:
            w = simulate_game(a, b, rng)
            winners.append(w)
            depth[w] = max(depth[w], r)
        if len(winners) == 1:
            depth[winners[0]] = 6
            break
        current = list(zip(winners[0::2], winners[1::2]))
        r += 1
    return depth


def run_simulations(n_sims):
    rng = np.random.default_rng(42)
    playins = load_playins()
    round1 = load_round1()
    sweet16, elite8, final4, champ = {}, {}, {}, {}
    for _ in range(n_sims):
        winners = resolve_playins(playins, rng)
        games = substitute_slots(round1, winners)
        depth = simulate_tournament(games, rng)
        for t, d in depth.items():
            if d >= 2:
                sweet16[t] = sweet16.get(t, 0) + 1
            if d >= 3:
                elite8[t] = elite8.get(t, 0) + 1
            if d >= 4:
                final4[t] = final4.get(t, 0) + 1
            if d == 6:
                champ[t] = champ.get(t, 0) + 1

    def to_df(d, label):
        rows = [{"Team": k, label: v / n_sims} for k, v in d.items()]
        return pd.DataFrame(rows).sort_values(label, ascending=False).reset_index(drop=True)

    return to_df(sweet16, "Sweet16Prob"), to_df(elite8, "Elite8Prob"), to_df(final4, "Final4Prob"), to_df(champ, "ChampProb")


def build_upset_watch():
    playins = load_playins()
    round1 = load_round1()
    slot_winners = {}
    for slot, a, b in playins:
        p = win_prob(a, b)
        slot_winners[slot] = a if p >= 0.5 else b
    resolved = substitute_slots(round1, slot_winners)
    rows = []
    for a, b in resolved:
        pa = win_prob(a, b)
        rows.append({
            "Matchup": f"{a} vs {b}",
            "TeamA": a,
            "TeamB": b,
            "P(TeamA)": round(pa, 3),
            "P(TeamB)": round(1 - pa, 3),
            "Closeness": abs(pa - 0.5),
        })
    return pd.DataFrame(rows).sort_values("Closeness").reset_index(drop=True)


def choose_pick(a, b, mode, rng):
    p = float(win_prob(a, b))
    if mode == "Safe":
        pick_a_prob = p
    elif mode == "Balanced":
        if p >= 0.72:
            pick_a_prob = p
        else:
            pick_a_prob = 0.5 + (p - 0.5) * 0.65
    elif mode == "Chaos":
        pick_a_prob = 0.5 + (p - 0.5) * 0.35
    elif mode == "Upset-heavy":
        pick_a_prob = 0.5 + (p - 0.5) * 0.15
    else:
        pick_a_prob = p
    pick_a_prob = max(0.02, min(0.98, pick_a_prob))
    winner = a if rng.random() < pick_a_prob else b
    confidence = p if winner == a else 1 - p
    return winner, confidence, p


def build_pick_bracket(mode="Safe"):
    playins = load_playins()
    round1 = load_round1()
    rng = np.random.default_rng()
    rows = []
    slot_winners = {}
    for slot, a, b in playins:
        w, conf, base_p = choose_pick(a, b, mode, rng)
        slot_winners[slot] = w
        rows.append({"Round": "PLAYIN", "TeamA": a, "TeamB": b, "Pick": w, "Prob": round(conf, 3), "BaseProbA": round(base_p, 3), "Style": mode})
    games = substitute_slots(round1, slot_winners)
    winners = []
    for idx, (a, b) in enumerate(games):
        w, conf, base_p = choose_pick(a, b, mode, rng)
        winners.append(w)
        rows.append({"Round": "R64", "GameIndex": idx, "TeamA": a, "TeamB": b, "Pick": w, "Prob": round(conf, 3), "BaseProbA": round(base_p, 3), "Style": mode})
    rounds = ["R32", "S16", "E8", "F4", "Title"]
    for r in rounds:
        games = list(zip(winners[0::2], winners[1::2]))
        winners = []
        for idx, (a, b) in enumerate(games):
            w, conf, base_p = choose_pick(a, b, mode, rng)
            winners.append(w)
            rows.append({"Round": r, "GameIndex": idx, "TeamA": a, "TeamB": b, "Pick": w, "Prob": round(conf, 3), "BaseProbA": round(base_p, 3), "Style": mode})
    return pd.DataFrame(rows)


def metric_cards(champ_df, upset_df):
    top_team = champ_df.iloc[0]["Team"] if len(champ_df) else "-"
    top_prob = champ_df.iloc[0]["ChampProb"] if len(champ_df) else 0
    second_team = champ_df.iloc[1]["Team"] if len(champ_df) > 1 else "-"
    second_prob = champ_df.iloc[1]["ChampProb"] if len(champ_df) > 1 else 0
    closest = upset_df.iloc[0]["Matchup"] if len(upset_df) else "-"
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Top Title Pick", top_team, f"{top_prob:.1%}")
    with c2:
        st.metric("2nd Title Pick", second_team, f"{second_prob:.1%}")
    with c3:
        st.metric("Best Upset Watch", closest)


def probability_chart(df, col, title, color="#34d399"):
    top = df.head(12).copy()
    if top.empty:
        return None
    return (
        alt.Chart(top)
        .mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10, size=28)
        .encode(
            x=alt.X(col, title="Probability", axis=alt.Axis(format="%", labelColor="#cbd5e1", titleColor="#cbd5e1")),
            y=alt.Y("Team:N", sort="-x", title="", axis=alt.Axis(labelColor="#f1f5f9", labelFontSize=13)),
            tooltip=["Team", alt.Tooltip(col, format=".1%")],
            color=alt.value(color),
        )
        .properties(title=title, height=420)
        .configure(background="transparent")
        .configure_title(fontSize=20, color="#f8fafc", anchor="start")
        .configure_view(strokeOpacity=0)
    )


def featured_insights(champ_df, upset_df, sweet16_df):
    if champ_df.empty or upset_df.empty or sweet16_df.empty:
        return
    champ_team = champ_df.iloc[0]["Team"]
    champ_prob = champ_df.iloc[0]["ChampProb"]
    second_team = champ_df.iloc[1]["Team"] if len(champ_df) > 1 else champ_team
    second_prob = champ_df.iloc[1]["ChampProb"] if len(champ_df) > 1 else champ_prob
    upset_game = upset_df.iloc[0]["Matchup"]
    upset_prob = max(upset_df.iloc[0]["P(TeamA)"], upset_df.iloc[0]["P(TeamB)"])
    cinderella_team = sweet16_df.iloc[7]["Team"] if len(sweet16_df) > 7 else sweet16_df.iloc[-1]["Team"]
    cinderella_prob = sweet16_df.iloc[7]["Sweet16Prob"] if len(sweet16_df) > 7 else sweet16_df.iloc[-1]["Sweet16Prob"]
    st.markdown(f"""
        <div class="insight-strip">
            <div class="insight-card"><div class="insight-label">Safest Champion</div><div class="insight-value">{champ_team}</div><div class="insight-sub">Title odds: {champ_prob:.1%}</div></div>
            <div class="insight-card"><div class="insight-label">Best Challenger</div><div class="insight-value">{second_team}</div><div class="insight-sub">Title odds: {second_prob:.1%}</div></div>
            <div class="insight-card"><div class="insight-label">Featured Upset</div><div class="insight-value">{upset_game}</div><div class="insight-sub">Closest edge on the board: {upset_prob:.1%}</div></div>
            <div class="insight-card"><div class="insight-label">Cinderella Signal</div><div class="insight-value">{cinderella_team}</div><div class="insight-sub">Sweet 16 path: {cinderella_prob:.1%}</div></div>
        </div>
    """, unsafe_allow_html=True)


def split_bracket_frames(bracket_df):
    r64 = bracket_df[bracket_df["Round"] == "R64"].reset_index(drop=True)
    r32 = bracket_df[bracket_df["Round"] == "R32"].reset_index(drop=True)
    s16 = bracket_df[bracket_df["Round"] == "S16"].reset_index(drop=True)
    e8 = bracket_df[bracket_df["Round"] == "E8"].reset_index(drop=True)
    f4 = bracket_df[bracket_df["Round"] == "F4"].reset_index(drop=True)
    title = bracket_df[bracket_df["Round"] == "Title"].reset_index(drop=True)
    regions = []
    for i in range(4):
        regions.append({
            "name": REGION_NAMES[i],
            "r64": r64.iloc[i * 8:(i + 1) * 8].copy(),
            "r32": r32.iloc[i * 4:(i + 1) * 4].copy(),
            "s16": s16.iloc[i * 2:(i + 1) * 2].copy(),
            "e8": e8.iloc[i:i + 1].copy(),
        })
    final_left = f4.iloc[0] if len(f4) > 0 else None
    final_right = f4.iloc[1] if len(f4) > 1 else None
    champ = title.iloc[0] if len(title) > 0 else None
    return regions, final_left, final_right, champ


def render_bracket_board(bracket_df, mode="desktop"):
    regions, final_left, final_right, champ = split_bracket_frames(bracket_df)

    if mode == "desktop":
        card_w = 196
        line_w = 58
        final_gap = 228
        r64_space = (8, 8)
        r32_space = (34, 34)
        s16_space = (72, 72)
        e8_space = (136, 136)
        height = 1460
        min_width = 1680
    else:
        card_w = 156
        line_w = 44
        final_gap = 184
        r64_space = (4, 4)
        r32_space = (20, 20)
        s16_space = (42, 42)
        e8_space = (82, 82)
        height = 1080
        min_width = 1220

    spacing = {"R64": r64_space, "R32": r32_space, "S16": s16_space, "E8": e8_space}
    box_h = 68 if mode == "desktop" else 56
    logo = 30 if mode == "desktop" else 24
    title_font = "0.9rem" if mode == "desktop" else "0.8rem"
    sub_font = "0.72rem" if mode == "desktop" else "0.64rem"
    team_font = "0.9rem" if mode == "desktop" else "0.8rem"

    def card(row, mt, mb):
        return f'''
        <div style="margin-top:{mt}px;margin-bottom:{mb}px;border:1px solid rgba(255,255,255,0.10);background:linear-gradient(180deg, rgba(15,23,42,0.96), rgba(15,23,42,0.90));border-radius:16px;padding:12px 13px;min-height:{box_h}px;display:flex;flex-direction:column;justify-content:center;box-shadow:0 10px 22px rgba(0,0,0,0.22);">
            <div style="display:flex;align-items:center;gap:10px;">
                <img src="{logo_src(row['Pick'])}" style="width:{logo}px;height:{logo}px;border-radius:999px;object-fit:cover;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);" />
                <div style="font-weight:800;font-size:{team_font};line-height:1.15;color:#f8fafc;">{row['Pick']}</div>
            </div>
            <div style="margin-top:5px;font-size:{title_font};color:#93c5fd;">Win prob: {float(row['Prob']):.1%}</div>
            <div style="margin-top:4px;font-size:{sub_font};color:#94a3b8;line-height:1.25;">{row['TeamA']} vs {row['TeamB']}</div>
        </div>
        '''

    def centers(count, stage_name):
        mt, mb = spacing[stage_name]
        y = 0
        vals = []
        for _ in range(count):
            y += mt
            vals.append(y + box_h / 2)
            y += box_h + mb + 8
        return vals, max(40, int(y))

    def line_svg(stage: str, winners_count: int, reverse: bool = False):
        prev_stage = {"R32": "R64", "S16": "R32", "E8": "S16"}[stage]
        prev_centers, prev_h = centers(winners_count * 2, prev_stage)
        curr_centers, curr_h = centers(winners_count, stage)
        height_svg = max(prev_h, curr_h)
        mid = line_w / 2
        x1, x2 = (0, line_w) if not reverse else (line_w, 0)
        stroke = "#315fcb"
        paths = []
        for i, cy in enumerate(curr_centers):
            p1 = prev_centers[2 * i]
            p2 = prev_centers[2 * i + 1]
            paths.append(f'<path d="M {x1} {p1} L {mid} {p1} L {mid} {cy} L {x2} {cy}" fill="none" stroke="{stroke}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/>')
            paths.append(f'<path d="M {x1} {p2} L {mid} {p2} L {mid} {cy} L {x2} {cy}" fill="none" stroke="{stroke}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/>')
        return f'<svg width="{line_w}" height="{height_svg}" style="overflow:visible">{"".join(paths)}</svg>'

    def region_html(region, reverse=False):
        data_map = {"R64": region["r64"], "R32": region["r32"], "S16": region["s16"], "E8": region["e8"]}
        order = ["R64", "R32", "S16", "E8"] if not reverse else ["E8", "S16", "R32", "R64"]
        template = f"{card_w}px {line_w}px {card_w}px {line_w}px {card_w}px {line_w}px {card_w}px" if not reverse else f"{card_w}px {line_w}px {card_w}px {line_w}px {card_w}px {line_w}px {card_w}px"
        html = [f'<div><div style="text-align:center;color:#93c5fd;font-weight:800;margin-bottom:12px;font-size:{"1rem" if mode=="desktop" else "0.9rem"};">{region["name"]}</div><div style="display:grid;grid-template-columns:{template};gap:0;align-items:start;">']
        for idx, stage in enumerate(order):
            df = data_map[stage]
            if reverse:
                html.append('<div style="display:flex;flex-direction:column;gap:8px;">')
                html.append(f'<div style="text-align:center;font-weight:700;color:#e2e8f0;font-size:{"0.92rem" if mode=="desktop" else "0.82rem"};margin-bottom:6px;">{stage}</div>')
                mt, mb = spacing[stage]
                for _, row in df.iterrows():
                    html.append(card(row, mt, mb))
                html.append('</div>')
                if stage != "R64":
                    html.append(f'<div style="display:flex;justify-content:center;align-items:flex-start;padding-top:26px;">{line_svg(stage, len(df), reverse=True)}</div>')
            else:
                if idx > 0:
                    html.append(f'<div style="display:flex;justify-content:center;align-items:flex-start;padding-top:26px;">{line_svg(stage, len(df), reverse=False)}</div>')
                html.append('<div style="display:flex;flex-direction:column;gap:8px;">')
                html.append(f'<div style="text-align:center;font-weight:700;color:#e2e8f0;font-size:{"0.92rem" if mode=="desktop" else "0.82rem"};margin-bottom:6px;">{stage}</div>')
                mt, mb = spacing[stage]
                for _, row in df.iterrows():
                    html.append(card(row, mt, mb))
                html.append('</div>')
        html.append('</div></div>')
        return ''.join(html)

    left_html = region_html(regions[0], False) + region_html(regions[1], False)
    right_html = region_html(regions[2], True) + region_html(regions[3], True)

    center_logo = 38 if mode == "desktop" else 30
    center_team_font = "0.96rem" if mode == "desktop" else "0.84rem"
    center_prob_font = "0.8rem" if mode == "desktop" else "0.7rem"
    center_box_min_h = 72 if mode == "desktop" else 60

    def center_box(row):
        return f'''
        <div style="width:100%;border:1px solid rgba(255,255,255,0.10);background:linear-gradient(180deg, rgba(15,23,42,0.96), rgba(15,23,42,0.90));border-radius:16px;padding:12px 14px;min-height:{center_box_min_h}px;display:flex;flex-direction:column;justify-content:center;box-shadow:0 10px 22px rgba(0,0,0,0.22);">
            <div style="display:flex;align-items:center;gap:10px;">
                <img src="{logo_src(row['Pick'])}" style="width:{center_logo}px;height:{center_logo}px;border-radius:999px;object-fit:cover;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);" />
                <div style="font-weight:800;font-size:{center_team_font};line-height:1.15;color:#f8fafc;">{row['Pick']}</div>
            </div>
            <div style="margin-top:5px;font-size:{center_prob_font};color:#93c5fd;">Win prob: {float(row['Prob']):.1%}</div>
            <div style="margin-top:4px;font-size:{"0.72rem" if mode=="desktop" else "0.64rem"};color:#94a3b8;line-height:1.2;">{row['TeamA']} vs {row['TeamB']}</div>
        </div>
        '''

    center_html = ['<div style="display:flex;flex-direction:column;align-items:center;gap:14px;padding-top:28px;">', f'<div style="text-align:center;font-weight:800;color:#f8fafc;letter-spacing:0.02em;font-size:{"0.96rem" if mode=="desktop" else "0.84rem"};">Final Four & Title</div>']
    if final_left is not None:
        center_html.append(center_box(final_left))
    center_html.append(f'<div style="width:{final_gap/1.7:.0f}px;height:{52 if mode=="desktop" else 42}px;display:flex;align-items:center;justify-content:center;"><svg width="{final_gap/1.7:.0f}" height="{52 if mode=="desktop" else 42}"><path d="M 0 10 L {(final_gap/3.4):.0f} 10 L {(final_gap/3.4):.0f} {(26 if mode=="desktop" else 21)} L {(final_gap/1.7):.0f} {(26 if mode=="desktop" else 21)}" fill="none" stroke="#315fcb" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/><path d="M 0 {(42 if mode=="desktop" else 32)} L {(final_gap/3.4):.0f} {(42 if mode=="desktop" else 32)} L {(final_gap/3.4):.0f} {(26 if mode=="desktop" else 21)} L {(final_gap/1.7):.0f} {(26 if mode=="desktop" else 21)}" fill="none" stroke="#315fcb" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/></svg></div>')
    if final_right is not None:
        center_html.append(center_box(final_right))
    if champ is not None:
        center_html.append(f'<div style="width:{final_gap/2:.0f}px;height:{36 if mode=="desktop" else 30}px;display:flex;align-items:center;justify-content:center;"><svg width="{final_gap/2:.0f}" height="{36 if mode=="desktop" else 30}"><path d="M 0 {(18 if mode=="desktop" else 15)} L {(final_gap/2):.0f} {(18 if mode=="desktop" else 15)}" fill="none" stroke="#315fcb" stroke-width="2.2" stroke-linecap="round" opacity="0.95"/></svg></div>')
        center_html.append(f'''<div style="width:100%;background:linear-gradient(135deg, rgba(234,179,8,0.22), rgba(59,130,246,0.20));border:1px solid rgba(255,255,255,0.18);border-radius:18px;padding:14px;text-align:center;box-shadow:0 16px 30px rgba(0,0,0,0.22);"><div style="font-weight:700;color:#e2e8f0;font-size:{"0.88rem" if mode=="desktop" else "0.78rem"};margin-bottom:2px;">Champion</div><img src="{logo_src(champ['Pick'])}" style="width:{52 if mode=="desktop" else 42}px;height:{52 if mode=="desktop" else 42}px;border-radius:999px;object-fit:cover;margin:8px auto 10px auto;display:block;background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);" /><div style="font-weight:800;font-size:{"1.06rem" if mode=="desktop" else "0.94rem"};line-height:1.15;color:#f8fafc;">{champ['Pick']}</div><div style="margin-top:5px;font-size:{"0.76rem" if mode=="desktop" else "0.68rem"};color:#93c5fd;">Title win prob: {float(champ['Prob']):.1%}</div></div>''')
    center_html.append('</div>')

    board_cols = f"1fr {final_gap}px 1fr"
    html = f'''
    <html><head><style>
    body {{ margin:0; background:transparent; font-family:Inter,system-ui,sans-serif; }}
    .shell {{ background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); border:1px solid rgba(255,255,255,0.08); border-radius:24px; padding:{14 if mode=="desktop" else 12}px; overflow-x:auto; overflow-y:hidden; }}
    .board {{ min-width:{min_width}px; display:grid; grid-template-columns:{board_cols}; gap:{28 if mode=="desktop" else 18}px; align-items:start; color:#f8fafc; transform:scale(1); transform-origin:top left; }}
    </style></head><body><div class="shell"><div class="board"><div>{left_html}</div>{''.join(center_html)}<div>{right_html}</div></div></div></body></html>
    '''
    components.html(html, height=height, scrolling=True)


def render_region_cards(bracket_df):
    regions, final_left, final_right, champ = split_bracket_frames(bracket_df)
    tabs = st.tabs([r["name"] for r in regions] + ["Final Four"])
    for i, region in enumerate(regions):
        with tabs[i]:
            for stage_name, df in [("Round of 64", region["r64"]), ("Round of 32", region["r32"]), ("Sweet 16", region["s16"]), ("Elite 8", region["e8"] )]:
                st.markdown(f'<div class="section-card"><p class="section-title">{stage_name}</p><p class="section-sub">{region["name"]} region picks.</p></div>', unsafe_allow_html=True)
                for _, row in df.iterrows():
                    st.markdown(f'''
                    <div class="section-card" style="padding:0.85rem 0.95rem; margin-bottom:0.7rem;">
                        <div style="display:flex; align-items:center; gap:12px;">
                            <img src="{logo_src(row['Pick'])}" style="width:42px;height:42px;border-radius:999px;object-fit:cover;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);" />
                            <div>
                                <div style="font-weight:800; font-size:1rem; color:#f8fafc;">{row['Pick']}</div>
                                <div style="font-size:0.84rem; color:#93c5fd; margin-top:2px;">Win prob: {float(row['Prob']):.1%}</div>
                                <div style="font-size:0.78rem; color:#94a3b8; margin-top:2px;">{row['TeamA']} vs {row['TeamB']}</div>
                            </div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
    with tabs[-1]:
        if final_left is not None:
            st.markdown(f'''<div class="section-card"><p class="section-title">Final Four Pick 1</p><p class="section-sub">{final_left['Pick']} over {final_left['TeamA'] if final_left['Pick'] != final_left['TeamA'] else final_left['TeamB']}.</p></div>''', unsafe_allow_html=True)
        if final_right is not None:
            st.markdown(f'''<div class="section-card"><p class="section-title">Final Four Pick 2</p><p class="section-sub">{final_right['Pick']} over {final_right['TeamA'] if final_right['Pick'] != final_right['TeamA'] else final_right['TeamB']}.</p></div>''', unsafe_allow_html=True)
        if champ is not None:
            st.markdown(f'''<div class="section-card"><p class="section-title">Champion</p><p class="section-sub">{champ['Pick']} is your title pick with {float(champ['Prob']):.1%} game win probability in the final.</p></div>''', unsafe_allow_html=True)
            st.markdown(f'''<div class="section-card" style="text-align:center;"><img src="{logo_src(champ['Pick'])}" style="width:74px;height:74px;border-radius:999px;object-fit:cover;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);margin-bottom:10px;" /><div style="font-weight:800;font-size:1.28rem;">{champ['Pick']}</div></div>''', unsafe_allow_html=True)


ratings = load_master_ratings()
teams = sorted(ratings["Team"].dropna().unique().tolist())

st.markdown(
    """
    <div class="hero">
        <p class="hero-kicker">BracketLab • share-ready analytics</p>
        <h1 class="hero-title">BracketLab</h1>
        <p class="hero-sub">A polished March Madness dashboard for building, simulating, and sharing smarter brackets. Compare matchups, run tournament sims, surface upset candidates, and visualize the full bracket path in one place.</p>
        <div class="pill-row">
            <div class="pill">🏀 Matchup Predictor</div>
            <div class="pill">📈 Simulation Dashboard</div>
            <div class="pill">⚡ Featured Insights</div>
            <div class="pill">🧩 Bracket Board</div>
            <div class="pill">⬇️ Exportable Picks</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h3>🏀 BracketLab</h3>
            <p>Professional-looking bracket analysis built for simulation, storytelling, and sharing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.header("Controls")
    n_sims = st.slider("Number of simulations", min_value=1000, max_value=100000, step=1000, value=20000)
    bracket_style = st.selectbox("Bracket style", ["Safe", "Balanced", "Chaos", "Upset-heavy"], index=1, help="Safe hugs favorites. Balanced mixes value and realism. Chaos creates a wilder bracket. Upset-heavy hunts for separation.")
    run_button = st.button("Run tournament simulations", use_container_width=True)
    st.markdown(
        """
        <div class="mini-guide">
            <h4>How to use it</h4>
            <p>1. Compare a matchup</p>
            <p>2. Run sims</p>
            <p>3. Review featured insights</p>
            <p>4. Build a bracket and share it</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

left, right = st.columns([1.12, 0.88], gap="large")
with left:
    st.markdown('<div class="section-card"><p class="section-title">Single Game Predictor</p><p class="section-sub">Check any matchup on demand.</p></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Team A", teams, index=0)
    with col2:
        team_b = st.selectbox("Team B", teams, index=1)
    if team_a == team_b:
        st.info("Choose two different teams.")
    else:
        p = win_prob(team_a, team_b)
        left_logo = logo_src(team_a)
        right_logo = logo_src(team_b)
        st.markdown(f'''
            <div class="section-card matchup-card">
                <div class="matchup-grid">
                    <div class="matchup-team">
                        <img src="{left_logo}" class="matchup-logo" />
                        <div class="matchup-text">
                            <div class="matchup-name">{team_a}</div>
                            <div class="matchup-sub">Selected team A</div>
                        </div>
                    </div>
                    <div class="matchup-center">
                        <div class="matchup-vs">VS</div>
                        <div class="matchup-edge-label">Win edge</div>
                        <div class="matchup-prob">{p:.1%}</div>
                        <div class="matchup-bar"><div class="matchup-bar-fill" style="width:{p*100:.1f}%;"></div></div>
                        <div class="matchup-split"><span>{p:.1%}</span><span>{1-p:.1%}</span></div>
                    </div>
                    <div class="matchup-team right">
                        <img src="{right_logo}" class="matchup-logo" />
                        <div class="matchup-text right">
                            <div class="matchup-name">{team_b}</div>
                            <div class="matchup-sub">Selected team B</div>
                        </div>
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        c1, c2 = st.columns([0.9, 1.1])
        with c1:
            st.metric(f"P({team_a} wins)", f"{p:.1%}")
        with c2:
            st.markdown(f'''<div class="section-card" style="height:100%;"><p class="section-title" style="font-size:1rem;">Quick take</p><p class="section-sub"><b>{team_a}</b> gets a <b>{p:.1%}</b> win probability over <b>{team_b}</b> from the current model.</p></div>''', unsafe_allow_html=True)
with right:
    st.markdown('<div class="section-card"><p class="section-title">Quick Actions</p><p class="section-sub">Jump straight to the outputs people want to see.</p></div>', unsafe_allow_html=True)
    st.markdown(f'''<div class="section-card" style="padding:0.9rem 1rem;"><p class="section-title" style="font-size:0.98rem;">Generate My Bracket</p><p class="section-sub">Current style: <b>{bracket_style}</b></p><div style="margin-top:8px;color:#cbd5e1;font-size:0.84rem;line-height:1.4;">Safe = chalky and stable. Balanced = realistic pool-friendly mix. Chaos = more volatility. Upset-heavy = hunts for bracket separation.</div></div>''', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        show_upsets = st.button("Show upset watch", use_container_width=True)
    with col4:
        build_bracket = st.button("Generate my bracket", use_container_width=True)

if show_upsets:
    upset_df = build_upset_watch()
    st.markdown('<div class="section-card"><p class="section-title">Upset Watch</p><p class="section-sub">The smallest model edges in the bracket — these are the pressure points.</p></div>', unsafe_allow_html=True)
    st.dataframe(upset_df.head(20), use_container_width=True, height=520)

if build_bracket:
    bracket_df = build_pick_bracket(bracket_style)
    st.markdown(f'<div class="section-card"><p class="section-title">Bracket Board</p><p class="section-sub">A responsive bracket system tuned for desktop, split-screen, and phone. Generated using the <b>{bracket_style}</b> style.</p><p class="mobile-bracket-note">On smaller screens, region view is easier to read than the full board.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="desktop-only">', unsafe_allow_html=True)
    render_bracket_board(bracket_df, "desktop")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="tablet-only">', unsafe_allow_html=True)
    render_bracket_board(bracket_df, "tablet")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="mobile-only">', unsafe_allow_html=True)
    render_region_cards(bracket_df)
    st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("Show bracket table"):
        st.dataframe(bracket_df, use_container_width=True)
    st.download_button("Download bracket_picks.csv", bracket_df.to_csv(index=False).encode("utf-8"), file_name="bracket_picks.csv", mime="text/csv")

if run_button:
    with st.spinner("Running simulations..."):
        sweet16_df, elite8_df, final4_df, champ_df = run_simulations(n_sims)
        upset_df = build_upset_watch()
    featured_insights(champ_df, upset_df, sweet16_df)
    metric_cards(champ_df, upset_df)
    c1, c2 = st.columns(2)
    with c1:
        chart = probability_chart(champ_df, "ChampProb", "Top Championship Odds", "#34d399")
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
    with c2:
        chart = probability_chart(final4_df, "Final4Prob", "Top Final Four Odds", "#60a5fa")
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
    tabs = st.tabs(["Sweet 16", "Elite 8", "Final Four", "Champion", "Upset Watch"])
    with tabs[0]:
        st.dataframe(sweet16_df.head(20), use_container_width=True, height=500)
    with tabs[1]:
        st.dataframe(elite8_df.head(20), use_container_width=True, height=500)
    with tabs[2]:
        st.dataframe(final4_df.head(15), use_container_width=True, height=500)
    with tabs[3]:
        st.dataframe(champ_df.head(15), use_container_width=True, height=500)
        st.download_button("Download championship_odds.csv", champ_df.to_csv(index=False).encode("utf-8"), file_name="championship_odds.csv", mime="text/csv")
    with tabs[4]:
        st.dataframe(upset_df.head(20), use_container_width=True, height=500)

st.markdown('<div class="section-card"><p class="section-title">Current Top Teams By Blended Rating</p><p class="section-sub">The rating table driving BracketLab right now.</p></div>', unsafe_allow_html=True)
ratings_view = ratings.head(20).copy()
if not ratings_view.empty:
    rating_rows = []
    for _, row in ratings_view.iterrows():
        sources_text = row.get("sources_used", "") if "sources_used" in ratings_view.columns else ""
        rating_rows.append(f'''<div style="display:grid;grid-template-columns:72px 1.8fr 0.7fr 1fr;gap:16px;align-items:center;padding:14px 16px;border-bottom:1px solid rgba(255,255,255,0.06);"><div><img src="{logo_src(row['Team'])}" style="width:48px;height:48px;border-radius:999px;object-fit:cover;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.10);" /></div><div style="font-weight:700;color:#f8fafc;font-size:1rem;">{row['Team']}</div><div style="text-align:right;color:#e2e8f0;">{row['AdjEM_blend']:.2f}</div><div style="color:#94a3b8;">{sources_text}</div></div>''')
    ratings_html = '<html><body style="margin:0;background:transparent;font-family:Inter,system-ui,sans-serif;"><div style="border:1px solid rgba(255,255,255,0.08);border-radius:22px;overflow:hidden;background:linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.02));"><div style="display:grid;grid-template-columns:72px 1.8fr 0.7fr 1fr;gap:16px;align-items:center;padding:14px 16px;border-bottom:1px solid rgba(255,255,255,0.08);color:#94a3b8;font-weight:700;"><div>Logo</div><div>Team</div><div style="text-align:right;">AdjEM</div><div>Sources</div></div>' + ''.join(rating_rows) + '</div></body></html>'
    components.html(ratings_html, height=1050, scrolling=False)

st.markdown('<p class="footer-note">BracketLab • designed for shareable bracket analysis and polished tournament storytelling</p>', unsafe_allow_html=True)
