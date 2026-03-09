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
        max-width: 1480px;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
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
        padding: 1.5rem 1.6rem 1.4rem 1.6rem;
        margin-bottom: 1.1rem;
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
        font-size: 2.45rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0.15rem 0 0 0;
        line-height: 1.02;
    }

    .hero-sub {
        margin-top: 0.65rem;
        margin-bottom: 0;
        color: #dbeafe;
        font-size: 1rem;
        line-height: 1.45;
        max-width: 900px;
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
        box-shadow:
            0 12px 28px rgba(0,0,0,0.25),
            0 0 25px rgba(56,189,248,0.05);
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

    [data-testid="stMetricLabel"] {
        color: #cbd5e1;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
        font-weight: 800;
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

    .stSelectbox > div > div,
    .stSlider > div > div {
        border-radius: 14px !important;
    }

    .footer-note {
        color: #94a3b8;
        text-align: center;
        font-size: 0.8rem;
        margin-top: 1rem;
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

    sweet16 = {}
    elite8 = {}
    final4 = {}
    champ = {}

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

    return (
        to_df(sweet16, "Sweet16Prob"),
        to_df(elite8, "Elite8Prob"),
        to_df(final4, "Final4Prob"),
        to_df(champ, "ChampProb"),
    )


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
        rows.append(
            {
                "Matchup": f"{a} vs {b}",
                "TeamA": a,
                "TeamB": b,
                "P(TeamA)": round(pa, 3),
                "P(TeamB)": round(1 - pa, 3),
                "Closeness": abs(pa - 0.5),
            }
        )

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
        rows.append({
            "Round": "PLAYIN",
            "TeamA": a,
            "TeamB": b,
            "Pick": w,
            "Prob": round(conf, 3),
            "BaseProbA": round(base_p, 3),
            "Style": mode,
        })

    games = substitute_slots(round1, slot_winners)

    winners = []
    for idx, (a, b) in enumerate(games):
        w, conf, base_p = choose_pick(a, b, mode, rng)
        winners.append(w)
        rows.append({
            "Round": "R64",
            "GameIndex": idx,
            "TeamA": a,
            "TeamB": b,
            "Pick": w,
            "Prob": round(conf, 3),
            "BaseProbA": round(base_p, 3),
            "Style": mode,
        })

    rounds = ["R32", "S16", "E8", "F4", "Title"]
    for r in rounds:
        games = list(zip(winners[0::2], winners[1::2]))
        winners = []
        for idx, (a, b) in enumerate(games):
            w, conf, base_p = choose_pick(a, b, mode, rng)
            winners.append(w)
            rows.append({
                "Round": r,
                "GameIndex": idx,
                "TeamA": a,
                "TeamB": b,
                "Pick": w,
                "Prob": round(conf, 3),
                "BaseProbA": round(base_p, 3),
                "Style": mode,
            })

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

    chart = (
        alt.Chart(top)
        .mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10, size=28)
        .encode(
            x=alt.X(
                col,
                title="Probability",
                axis=alt.Axis(format="%", labelColor="#cbd5e1", titleColor="#cbd5e1")
            ),
            y=alt.Y(
                "Team:N",
                sort="-x",
                title="",
                axis=alt.Axis(labelColor="#f1f5f9", labelFontSize=13)
            ),
            tooltip=["Team", alt.Tooltip(col, format=".1%")],
            color=alt.value(color),
        )
        .properties(title=title, height=420)
        .configure(background="transparent")
        .configure_title(fontSize=20, color="#f8fafc", anchor="start")
        .configure_view(strokeOpacity=0)
    )

    return chart


def render_actual_bracket(bracket_df):
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

    def styled_box(row, mt=0, mb=0):
        return f'''
        <div style="position:relative;margin-top:{mt}px;margin-bottom:{mb}px;border:1px solid rgba(255,255,255,0.12);background:linear-gradient(180deg, rgba(15,23,42,0.96), rgba(15,23,42,0.88));border-radius:18px;padding:18px 20px;min-height:108px;display:flex;flex-direction:column;justify-content:center;box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 14px 30px rgba(0,0,0,0.22);">
            <div style="display:flex;align-items:center;gap:14px;">
                <img src="{logo_src(row['Pick'])}" style="width:54px;height:54px;border-radius:999px;object-fit:cover;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);" />
                <div style="font-weight:800;font-size:1.18rem;line-height:1.15;color:#f8fafc;">{row['Pick']}</div>
            </div>
            <div style="margin-top:7px;font-size:0.96rem;color:#93c5fd;">Win prob: {float(row['Prob']):.1%}</div>
            <div style="margin-top:6px;font-size:0.88rem;color:#94a3b8;line-height:1.3;">{row['TeamA']} vs {row['TeamB']}</div>
        </div>
        '''

    def ff_box(row):
        return f'''
        <div style="width:100%;border:1px solid rgba(255,255,255,0.12);background:linear-gradient(180deg, rgba(15,23,42,0.96), rgba(15,23,42,0.88));border-radius:18px;padding:18px 20px;min-height:108px;display:flex;flex-direction:column;justify-content:center;box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 14px 30px rgba(0,0,0,0.22);">
            <div style="display:flex;align-items:center;gap:14px;">
                <img src="{logo_src(row['Pick'])}" style="width:54px;height:54px;border-radius:999px;object-fit:cover;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);" />
                <div style="font-weight:800;font-size:1.18rem;line-height:1.15;color:#f8fafc;">{row['Pick']}</div>
            </div>
            <div style="margin-top:7px;font-size:0.96rem;color:#93c5fd;">Win prob: {float(row['Prob']):.1%}</div>
            <div style="margin-top:6px;font-size:0.88rem;color:#94a3b8;line-height:1.3;">{row['TeamA']} vs {row['TeamB']}</div>
        </div>
        '''

    def line_svg(stage: str, winners_count: int, reverse: bool = False):
        box_h = 108
        gap = 12
        pad = {"R64": 28, "R32": 120, "S16": 240, "E8": 430}
        prev_stage = {"R32": "R64", "S16": "R32", "E8": "S16"}[stage]
        prev_count = winners_count * 2
        width = 110 if stage != "E8" else 130

        def centers(count, stage_name):
            mt = pad[stage_name]
            mb = pad[stage_name]
            y = 0
            vals = []
            for _ in range(count):
                y += mt
                vals.append(y + box_h / 2)
                y += box_h + mb + gap
            return vals, max(40, int(y))

        prev_centers, prev_h = centers(prev_count, prev_stage)
        curr_centers, curr_h = centers(winners_count, stage)
        height = max(prev_h, curr_h)
        mid = width / 2
        x1, x2 = (0, width) if not reverse else (width, 0)
        stroke = "#315fcb"
        paths = []
        for i, cy in enumerate(curr_centers):
            p1 = prev_centers[2 * i]
            p2 = prev_centers[2 * i + 1]
            paths.append(f'<path d="M {x1} {p1} L {mid} {p1} L {mid} {cy} L {x2} {cy}" fill="none" stroke="{stroke}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/>')
            paths.append(f'<path d="M {x1} {p2} L {mid} {p2} L {mid} {cy} L {x2} {cy}" fill="none" stroke="{stroke}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/>')
        return f'<svg width="{width}" height="{height}" style="overflow:visible">{"".join(paths)}</svg>'

    def region_html(region_name, r64_df, r32_df, s16_df, e8_df, reverse=False):
        data_map = {"R64": r64_df, "R32": r32_df, "S16": s16_df, "E8": e8_df}
        order = ["R64", "R32", "S16", "E8"] if not reverse else ["E8", "S16", "R32", "R64"]
        spacing = {"R64": (28, 28), "R32": (120, 120), "S16": (240, 240), "E8": (430, 430)}
        template = "290px 130px 290px 130px 290px 150px 290px" if not reverse else "290px 150px 290px 130px 290px 130px 290px"
        html = [f'<div><div style="text-align:center;color:#93c5fd;font-weight:800;margin-bottom:18px;letter-spacing:0.02em;font-size:1.22rem;">{region_name}</div><div style="display:grid;grid-template-columns:{template};gap:0;align-items:start;">']
        for idx, stage in enumerate(order):
            if reverse:
                df = data_map[stage]
                html.append('<div style="display:flex;flex-direction:column;gap:12px;">')
                html.append(f'<div style="text-align:center;font-weight:700;color:#e2e8f0;font-size:1.12rem;margin-bottom:14px;">{stage}</div>')
                mt, mb = spacing[stage]
                for _, row in df.iterrows():
                    html.append(styled_box(row, mt, mb))
                html.append('</div>')
                if stage != "R64":
                    html.append(f'<div style="display:flex;justify-content:center;align-items:flex-start;padding-top:70px;">{line_svg(stage, len(df), reverse=True)}</div>')
            else:
                if idx > 0:
                    df = data_map[stage]
                    html.append(f'<div style="display:flex;justify-content:center;align-items:flex-start;padding-top:70px;">{line_svg(stage, len(df), reverse=False)}</div>')
                df = data_map[stage]
                html.append('<div style="display:flex;flex-direction:column;gap:12px;">')
                html.append(f'<div style="text-align:center;font-weight:700;color:#e2e8f0;font-size:1.12rem;margin-bottom:14px;">{stage}</div>')
                mt, mb = spacing[stage]
                for _, row in df.iterrows():
                    html.append(styled_box(row, mt, mb))
                html.append('</div>')
        html.append('</div></div>')
        return ''.join(html)

    left_html = region_html(regions[0]["name"], regions[0]["r64"], regions[0]["r32"], regions[0]["s16"], regions[0]["e8"], False)
    left_html += region_html(regions[1]["name"], regions[1]["r64"], regions[1]["r32"], regions[1]["s16"], regions[1]["e8"], False)
    right_html = region_html(regions[2]["name"], regions[2]["r64"], regions[2]["r32"], regions[2]["s16"], regions[2]["e8"], True)
    right_html += region_html(regions[3]["name"], regions[3]["r64"], regions[3]["r32"], regions[3]["s16"], regions[3]["e8"], True)

    final_left = f4.iloc[0] if len(f4) > 0 else None
    final_right = f4.iloc[1] if len(f4) > 1 else None
    champ = title.iloc[0] if len(title) > 0 else None

    center_html = ['<div style="display:flex;flex-direction:column;align-items:center;gap:22px;padding-top:52px;">', '<div style="text-align:center;font-weight:800;color:#f8fafc;letter-spacing:0.02em;font-size:1.14rem;">Final Four & Title</div>']
    if final_left is not None:
        center_html.append(ff_box(final_left))
    center_html.append('<div style="width:150px;height:70px;display:flex;align-items:center;justify-content:center;"><svg width="150" height="70"><path d="M 0 12 L 75 12 L 75 35 L 150 35" fill="none" stroke="#315fcb" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/><path d="M 0 58 L 75 58 L 75 35 L 150 35" fill="none" stroke="#315fcb" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round" opacity="0.95"/></svg></div>')
    if final_right is not None:
        center_html.append(ff_box(final_right))
    if champ is not None:
        center_html.append('<div style="width:140px;height:52px;display:flex;align-items:center;justify-content:center;"><svg width="140" height="52"><path d="M 0 26 L 140 26" fill="none" stroke="#315fcb" stroke-width="2.8" stroke-linecap="round" opacity="0.95"/></svg></div>')
        champ_logo = logo_src(champ["Pick"])
        center_html.append(
            f'''<div style="width:100%;background:linear-gradient(135deg, rgba(234,179,8,0.22), rgba(59,130,246,0.20));border:1px solid rgba(255,255,255,0.18);border-radius:20px;padding:22px;text-align:center;box-shadow:0 16px 36px rgba(0,0,0,0.24);">
                    <div style="text-align:center;font-weight:700;color:#e2e8f0;font-size:1rem;margin-bottom:2px;">Champion</div>
                    <img src="{champ_logo}" style="width:76px;height:76px;border-radius:999px;object-fit:cover;margin:10px auto 12px auto;display:block;background:rgba(255,255,255,0.07);border:1px solid rgba(255,255,255,0.12);" />
                    <div style="font-weight:800;font-size:1.34rem;line-height:1.15;color:#f8fafc;">{champ['Pick']}</div>
                    <div style="margin-top:6px;font-size:0.86rem;color:#93c5fd;">Title win prob: {float(champ['Prob']):.1%}</div>
                    <div style="margin-top:4px;font-size:0.8rem;color:#94a3b8;line-height:1.25;">{champ['TeamA']} vs {champ['TeamB']}</div>
                </div>'''
        )
    center_html.append('</div>')

    html = f'''
    <html>
    <head>
    <style>
      body {{ margin: 0; background: transparent; font-family: Inter, system-ui, sans-serif; }}
      .bracket-shell {{ background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); border: 1px solid rgba(255,255,255,0.08); border-radius: 24px; padding: 18px; overflow-x: auto; }}
      .bracket-board {{ min-width: 3000px; display: grid; grid-template-columns: 1fr 420px 1fr; gap: 56px; align-items: start; color: #f8fafc; }}
    </style>
    </head>
    <body>
      <div class="bracket-shell">
        <div class="bracket-board">
          <div>{left_html}</div>
          {''.join(center_html)}
          <div>{right_html}</div>
        </div>
      </div>
    </body>
    </html>
    '''
    components.html(html, height=2800, scrolling=True)


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

    st.markdown(
        f"""
        <div class="insight-strip">
            <div class="insight-card">
                <div class="insight-label">Safest Champion</div>
                <div class="insight-value">{champ_team}</div>
                <div class="insight-sub">Title odds: {champ_prob:.1%}</div>
            </div>
            <div class="insight-card">
                <div class="insight-label">Best Challenger</div>
                <div class="insight-value">{second_team}</div>
                <div class="insight-sub">Title odds: {second_prob:.1%}</div>
            </div>
            <div class="insight-card">
                <div class="insight-label">Featured Upset</div>
                <div class="insight-value">{upset_game}</div>
                <div class="insight-sub">Closest edge on the board: {upset_prob:.1%}</div>
            </div>
            <div class="insight-card">
                <div class="insight-label">Cinderella Signal</div>
                <div class="insight-value">{cinderella_team}</div>
                <div class="insight-sub">Sweet 16 path: {cinderella_prob:.1%}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    bracket_style = st.selectbox(
        "Bracket style",
        ["Safe", "Balanced", "Chaos", "Upset-heavy"],
        index=1,
        help="Safe hugs the favorites. Balanced mixes value and realism. Chaos creates a wilder bracket. Upset-heavy aggressively chases surprises.",
    )
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

left, right = st.columns([1.2, 0.8])

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
        st.markdown(
            f'''
            <div class="section-card" style="padding:1.3rem 1.4rem;">
                <div style="display:grid;grid-template-columns:1.6fr 260px 1.6fr;align-items:center;gap:34px;">
                    <div style="display:flex;align-items:center;gap:18px;justify-content:flex-start;">
                        <img src="{left_logo}" style="width:110px;height:110px;border-radius:999px;object-fit:cover;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);box-shadow:0 10px 24px rgba(0,0,0,0.22);" />
                        <div>
                            <div style="font-size:1.4rem;font-weight:800;color:#f8fafc;line-height:1.12;">{team_a}</div>
                            <div style="font-size:0.9rem;color:#94a3b8;margin-top:4px;">Selected team A</div>
                        </div>
                    </div>
                    <div style="text-align:center;">
                        <div style="display:inline-flex;align-items:center;justify-content:center;width:96px;height:96px;border-radius:999px;background:linear-gradient(135deg, rgba(37,99,235,0.78), rgba(16,185,129,0.70));border:1px solid rgba(255,255,255,0.12);font-weight:800;color:#fff;font-size:1.2rem;box-shadow:0 12px 28px rgba(0,0,0,0.24);">VS</div>
                        <div style="margin-top:12px;font-size:0.8rem;color:#93c5fd;font-weight:800;letter-spacing:0.1em;text-transform:uppercase;">Win edge</div>
                        <div style="font-size:2.1rem;font-weight:800;color:#f8fafc;">{p:.1%}</div>
                        <div style="height:12px;background:rgba(255,255,255,0.08);border-radius:999px;overflow:hidden;margin-top:10px;">
                            <div style="width:{p*100:.1f}%;height:100%;background:linear-gradient(90deg,#2563eb,#10b981);"></div>
                        </div>
                        <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#94a3b8;margin-top:8px;">
                            <span>{p:.1%}</span><span>{1-p:.1%}</span>
                        </div>
                    </div>
                    <div style="display:flex;align-items:center;gap:18px;justify-content:flex-end;">
                        <div style="text-align:right;">
                            <div style="font-size:1.4rem;font-weight:800;color:#f8fafc;line-height:1.12;">{team_b}</div>
                            <div style="font-size:0.9rem;color:#94a3b8;margin-top:4px;">Selected team B</div>
                        </div>
                        <img src="{right_logo}" style="width:110px;height:110px;border-radius:999px;object-fit:cover;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);box-shadow:0 10px 24px rgba(0,0,0,0.22);" />
                    </div>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([0.85, 1.15])
        with c1:
            st.metric(f"P({team_a} wins)", f"{p:.1%}")
        with c2:
            st.markdown(
                f"""
                <div class="section-card" style="height:100%;">
                    <p class="section-title" style="font-size:1rem;">Quick take</p>
                    <p class="section-sub"><b>{team_a}</b> gets a <b>{p:.1%}</b> win probability over <b>{team_b}</b> from the current model.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

with right:
    st.markdown('<div class="section-card"><p class="section-title">Quick Actions</p><p class="section-sub">Jump straight to the outputs people want to see.</p></div>', unsafe_allow_html=True)
    st.markdown(
        f'''<div class="section-card" style="padding:0.9rem 1rem;">
        <p class="section-title" style="font-size:0.98rem;">Generate My Bracket</p>
        <p class="section-sub">Current style: <b>{bracket_style}</b></p>
        <div style="margin-top:8px;color:#cbd5e1;font-size:0.84rem;line-height:1.4;">
        Safe = chalky and stable. Balanced = realistic pool-friendly mix. Chaos = more volatility. Upset-heavy = hunts for bracket separation.
        </div>
        </div>''',
        unsafe_allow_html=True,
    )
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
    st.markdown(
        f'''<div class="section-card"><p class="section-title">Bracket Board</p><p class="section-sub">A four-region bracket layout with logos, connecting lines, Final Four, and champion. Generated using the <b>{bracket_style}</b> style.</p></div>''',
        unsafe_allow_html=True,
    )
    render_actual_bracket(bracket_df)
    with st.expander("Show bracket table"):
        st.dataframe(bracket_df, use_container_width=True)
    st.download_button(
        "Download bracket_picks.csv",
        bracket_df.to_csv(index=False).encode("utf-8"),
        file_name="bracket_picks.csv",
        mime="text/csv",
    )

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
        st.download_button(
            "Download championship_odds.csv",
            champ_df.to_csv(index=False).encode("utf-8"),
            file_name="championship_odds.csv",
            mime="text/csv",
        )
    with tabs[4]:
        st.dataframe(upset_df.head(20), use_container_width=True, height=500)

st.markdown('<div class="section-card"><p class="section-title">Current Top Teams By Blended Rating</p><p class="section-sub">The rating table driving BracketLab right now.</p></div>', unsafe_allow_html=True)
ratings_view = ratings.head(20).copy()
if not ratings_view.empty:
    rating_rows = []
    for _, row in ratings_view.iterrows():
        sources_text = row.get("sources_used", "") if "sources_used" in ratings_view.columns else ""
        rating_rows.append(
            f'''
            <div style="display:grid;grid-template-columns:72px 1.8fr 0.7fr 1fr;gap:16px;align-items:center;padding:14px 16px;border-bottom:1px solid rgba(255,255,255,0.06);">
                <div><img src="{logo_src(row['Team'])}" style="width:48px;height:48px;border-radius:999px;object-fit:cover;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.10);" /></div>
                <div style="font-weight:700;color:#f8fafc;font-size:1rem;">{row['Team']}</div>
                <div style="text-align:right;color:#e2e8f0;">{row['AdjEM_blend']:.2f}</div>
                <div style="color:#94a3b8;">{sources_text}</div>
            </div>
            '''
        )

    ratings_html = (
        '<html><body style="margin:0;background:transparent;font-family:Inter,system-ui,sans-serif;">'
        '<div style="border:1px solid rgba(255,255,255,0.08);border-radius:22px;overflow:hidden;'
        'background:linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.02));">'
        '<div style="display:grid;grid-template-columns:72px 1.8fr 0.7fr 1fr;gap:16px;align-items:center;'
        'padding:14px 16px;border-bottom:1px solid rgba(255,255,255,0.08);color:#94a3b8;font-weight:700;">'
        '<div>Logo</div><div>Team</div><div style="text-align:right;">AdjEM</div><div>Sources</div></div>'
        + ''.join(rating_rows)
        + '</div></body></html>'
    )
    components.html(ratings_html, height=1050, scrolling=False)

st.markdown('<p class="footer-note">BracketLab • designed for shareable bracket analysis and polished tournament storytelling</p>', unsafe_allow_html=True)
