# ══════════════════════════════════════════════════════════════════════════════
#  DSS REKOMENDASI RUMAH SAKIT — KAWASAN DENPASAR
#  Struktur Data · Graph · Dijkstra · K-Means · AI Groq
#  Teknologi: Python · Streamlit · NetworkX · Folium · Scikit-learn
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import networkx as nx
import heapq
import math
import copy
import time
import urllib.parse
import streamlit.components.v1 as components
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HospFinder DSS — Bali",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS PREMIUM (Dari app.py)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --p:    #3b82f6;
  --pd:   #1d4ed8;
  --acc:  #f59e0b;
  --acc2: #10b981;
  --bg:   #0f0f1a;
  --bg2:  #16162a;
  --bg3:  #1e1e35;
  --card: #1a1a2e;
  --bord: rgba(255,255,255,0.08);
  --txt:  #e2e8f0;
  --txt2: #94a3b8;
  --glow: 0 0 30px rgba(59,130,246,0.25);
}

html, body, .stApp { background: var(--bg) !important; font-family: 'Sora', sans-serif !important; color: var(--txt) !important; }
.stApp > header { background: transparent !important; }
#MainMenu, footer, .stDeployButton { display: none !important; }
.block-container { padding: 1.5rem 2rem 4rem !important; max-width: 1440px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: var(--bg2) !important; border-right: 1px solid var(--bord) !important; }
[data-testid="stSidebar"] * { color: var(--txt) !important; }
[data-testid="stSidebar"] .stSelectbox>div>div, [data-testid="stSidebar"] .stTextInput>div>div, [data-testid="stSidebar"] .stNumberInput>div>div {
  background: var(--bg3) !important; border: 1px solid var(--bord) !important; border-radius: 10px !important; color: var(--txt) !important;
}
[data-testid="stSidebar"] hr { border-color: var(--bord) !important; }

/* ── Hero ── */
.hero {
  background: linear-gradient(135deg, #0f172a 0%, #0f0f1a 40%, #1a1a2e 100%);
  border: 1px solid var(--bord); border-radius: 20px; padding: 2.5rem 3rem; margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.hero::before {
  content: ''; position: absolute; top: -60px; right: -60px; width: 280px; height: 280px;
  background: radial-gradient(circle, rgba(59,130,246,0.18) 0%, transparent 70%); border-radius: 50%;
}
.hero h1 {
  font-size: 2.4rem !important; font-weight: 800 !important; color: white !important; margin: 0 !important;
  background: linear-gradient(135deg, #fff 0%, #93c5fd 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { color: var(--txt2) !important; font-size: 0.95rem !important; margin: 0.5rem 0 0 !important; }
.hero-pill {
  display: inline-block; background: rgba(59,130,246,0.20); color: #93c5fd;
  border: 1px solid rgba(59,130,246,0.35); border-radius: 20px; padding: 3px 14px; font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.06em; margin-bottom: 0.8rem;
}

/* ── Steps ── */
.steps { display: flex; align-items: center; background: var(--card); border: 1px solid var(--bord); border-radius: 50px; padding: 5px; width: fit-content; margin-bottom: 1.5rem; gap: 2px; }
.step { display: flex; align-items: center; gap: 8px; padding: 7px 18px; border-radius: 40px; font-size: 0.8rem; font-weight: 600; color: var(--txt2); transition: all 0.3s; }
.step.active { background: var(--p); color: white; box-shadow: 0 0 16px rgba(59,130,246,0.4); }
.step.done   { color: var(--acc2); }
.step-n { width: 20px; height: 20px; border-radius: 50%; background: rgba(255,255,255,0.08); display: flex; align-items: center; justify-content: center; font-size: 0.7rem; font-weight: 700; }
.step.active .step-n { background: rgba(255,255,255,0.25); }
.step-div { width: 20px; height: 1px; background: var(--bord); }

/* ── Cards & Items ── */
.card { background: var(--card); border: 1px solid var(--bord); border-radius: 16px; padding: 1.4rem; margin-bottom: 1rem; }
.card-p { border-top: 2px solid var(--p); }
.card-g { border-left: 3px solid var(--acc2); }
.kost-card { background: var(--bg3); border: 1px solid var(--bord); border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.6rem; cursor: pointer; transition: all 0.2s; }
.kost-card:hover { border-color: var(--p); box-shadow: var(--glow); }
.kost-card.sel   { border-color: var(--p); background: rgba(59,130,246,0.08); box-shadow: 0 0 0 2px rgba(59,130,246,0.25); }
.kost-name { font-size: 0.9rem; font-weight: 700; color: var(--txt); line-height: 1.3; }
.tag { display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 0.68rem; font-weight: 600; }
.tag-p  { background: rgba(59,130,246,0.2);  color: #93c5fd; }
.tag-g  { background: rgba(16,185,129,0.2);  color: #6ee7b7; }
.tag-y  { background: rgba(245,158,11,0.2);  color: #fcd34d; }
.tag-r  { background: rgba(239,68,68,0.2);   color: #fca5a5; }
.tag-b  { background: rgba(139,92,246,0.2);  color: #c4b5fd; }

/* ── Stat Grid ── */
.sg { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin: 0.8rem 0; }
.sg2 { grid-template-columns: repeat(2,1fr); }
.sg4 { grid-template-columns: repeat(4,1fr); }
.sb { background: var(--bg3); border: 1px solid var(--bord); border-radius: 10px; padding: 0.8rem; text-align: center; }
.sv { font-size: 1.5rem; font-weight: 800; color: var(--p); font-family: 'JetBrains Mono', monospace; }
.sl { font-size: 0.68rem; color: var(--txt2); font-weight: 500; margin-top: 2px; }

/* ── Route Viz ── */
.route { background: linear-gradient(135deg, var(--pd), var(--bg3)); border: 1px solid rgba(59,130,246,0.3); border-radius: 12px; padding: 1rem 1.4rem; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.rn { background: rgba(255,255,255,0.12); color: white; border-radius: 8px; padding: 4px 12px; font-size: 0.78rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
.rn.s { background: rgba(16,185,129,0.35); }
.rn.e { background: rgba(239,68,68,0.35); }
.ra   { color: rgba(255,255,255,0.4); font-size: 1rem; }

.sl-h { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--p); display: flex; align-items: center; gap: 6px; margin-bottom: 0.6rem; }
.ai-box { background: linear-gradient(135deg, #0d1724, #0f172a); border: 1px solid rgba(59,130,246,0.3); border-radius: 14px; padding: 1.4rem 1.6rem; }
.ai-lbl { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #93c5fd; margin-bottom: 0.6rem; display: block; }
.stButton>button[kind="primary"] { background: linear-gradient(135deg, var(--p), var(--pd)) !important; border: none !important; color: white !important; box-shadow: 0 4px 20px rgba(59,130,246,0.35) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  KONSTANTA & GRAPH BASE
# ══════════════════════════════════════════════════════════════════════════════
BASE_NODES = ['Tabanan', 'Mengwi', 'Canggu', 'Ubung', 'Denpasar', 'Kuta', 'Jimbaran', 'Nusa Dua', 'Sanur']
BASE_COORDS = {
    'Tabanan': [-8.5363, 115.1215], 'Mengwi': [-8.5283, 115.1611],
    'Canggu': [-8.6478, 115.1385], 'Ubung': [-8.6360, 115.1979],
    'Denpasar': [-8.6500, 115.2167], 'Kuta': [-8.7233, 115.1723],
    'Jimbaran': [-8.7667, 115.1750], 'Nusa Dua': [-8.8061, 115.2268],
    'Sanur': [-8.6946, 115.2599]
}
BASE_EDGES = [
    ('Tabanan', 'Mengwi', 8.0), ('Tabanan', 'Canggu', 15.0),
    ('Mengwi', 'Denpasar', 12.0), ('Mengwi', 'Ubung', 10.0),
    ('Canggu', 'Kuta', 10.0), ('Canggu', 'Denpasar', 11.0),
    ('Ubung', 'Denpasar', 5.0), ('Denpasar', 'Kuta', 9.0),
    ('Denpasar', 'Jimbaran', 18.0), ('Kuta', 'Jimbaran', 8.0),
    ('Jimbaran', 'Nusa Dua', 12.0), ('Denpasar', 'Sanur', 7.0)
]

TIER_LABELS = ['🥇 Tier S — Prioritas Utama', '🥈 Tier A — Sangat Layak', '🥉 Tier B — Alternatif', '📋 Tier C — Standar']
TIER_SHORT = ['Tier S', 'Tier A', 'Tier B', 'Tier C']
TIER_TAG   = ['tag-y',  'tag-p',  'tag-g',  'tag-b']

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET RUMAH SAKIT
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_hospital_data():
    data = [
        {"nama": "RS Bali International Hospital", "tipe": "A", "lat": -8.6755, "lon": 115.2625, "rating": 4.8, "biaya_admin": 250000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Bali+International+Hospital+Denpasar"},
        {"nama": "RSUP Prof. Dr. I.G.N.G. Ngoerah (Sanglah)", "tipe": "A", "lat": -8.6816, "lon": 115.2078, "rating": 4.5, "biaya_admin": 150000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RSUP+Prof+Dr+IGNG+Ngoerah+Denpasar"},
        {"nama": "Rumah Sakit Mata Bali Mandara", "tipe": "A", "lat": -8.6543, "lon": 115.2185, "rating": 4.8, "biaya_admin": 120000, "gmaps": "https://www.google.com/maps/search/?api=1&query=Rumah+Sakit+Mata+Bali+Mandara+Denpasar"},
        {"nama": "RS Gigi dan Mulut Saraswati Denpasar", "tipe": "B", "lat": -8.6567, "lon": 115.2212, "rating": 4.6, "biaya_admin": 80000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Gigi+dan+Mulut+Saraswati+Denpasar"},
        {"nama": "RS Umum Daerah Bali Mandara", "tipe": "B", "lat": -8.6944, "lon": 115.2536, "rating": 4.8, "biaya_admin": 90000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Umum+Daerah+Bali+Mandara+Denpasar"},
        {"nama": "RS Umum Daerah Wangaya", "tipe": "B", "lat": -8.6472, "lon": 115.2084, "rating": 4.2, "biaya_admin": 80000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Umum+Daerah+Wangaya+Denpasar"},
        {"nama": "RS BaliMed", "tipe": "C", "lat": -8.6635, "lon": 115.1887, "rating": 4.5, "biaya_admin": 210000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+BaliMed+Denpasar"},
        {"nama": "RS Bhayangkara Denpasar", "tipe": "C", "lat": -8.6501, "lon": 115.2195, "rating": 4.4, "biaya_admin": 50000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Bhayangkara+Denpasar"},
        {"nama": "RS Ibu dan Anak Bali Royal", "tipe": "C", "lat": -8.6690, "lon": 115.2255, "rating": 4.6, "biaya_admin": 150000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Ibu+dan+Anak+Bali+Royal+Denpasar"},
        {"nama": "RS Ibu dan Anak Bunda Denpasar", "tipe": "C", "lat": -8.6412, "lon": 115.1956, "rating": 4.4, "biaya_admin": 120000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Ibu+dan+Anak+Bunda+Denpasar"},
        {"nama": "RS Ibu dan Anak Puri Bunda", "tipe": "C", "lat": -8.6385, "lon": 115.2123, "rating": 4.7, "biaya_admin": 140000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Ibu+dan+Anak+Puri+Bunda+Denpasar"},
        {"nama": "RS Khusus Bedah Dharma Usadha Sidhi", "tipe": "C", "lat": -8.6682, "lon": 115.2115, "rating": 4.3, "biaya_admin": 180000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Khusus+Bedah+Dharma+Usadha+Sidhi+Denpasar"},
        {"nama": "RS Khusus Ibu dan Anak Pucuk Permata Hati", "tipe": "C", "lat": -8.6705, "lon": 115.1882, "rating": 4.5, "biaya_admin": 130000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Khusus+Ibu+dan+Anak+Pucuk+Permata+Hati+Denpasar"},
        {"nama": "RS Khusus Mata Ramata", "tipe": "C", "lat": -8.6418, "lon": 115.1915, "rating": 4.8, "biaya_admin": 160000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Khusus+Mata+Ramata+Denpasar"},
        {"nama": "RS TK. II Udayana", "tipe": "C", "lat": -8.6715, "lon": 115.2140, "rating": 4.6, "biaya_admin": 75000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+TK+II+Udayana+Denpasar"},
        {"nama": "RS Umum Bali Royal (BROS)", "tipe": "C", "lat": -8.6690, "lon": 115.2255, "rating": 4.6, "biaya_admin": 200000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Umum+Bali+Royal+Denpasar"},
        {"nama": "RS Umum Dharma Yadnya", "tipe": "C", "lat": -8.6390, "lon": 115.2425, "rating": 4.3, "biaya_admin": 80000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Umum+Dharma+Yadnya+Denpasar"},
        {"nama": "RS Umum Kasih Ibu", "tipe": "C", "lat": -8.6667, "lon": 115.2033, "rating": 4.4, "biaya_admin": 250000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Umum+Kasih+Ibu+Denpasar"},
        {"nama": "RS Umum Manuaba", "tipe": "C", "lat": -8.6493, "lon": 115.2014, "rating": 4.3, "biaya_admin": 120000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Umum+Manuaba+Denpasar"},
        {"nama": "RS Umum Prima Medika", "tipe": "C", "lat": -8.6720, "lon": 115.2040, "rating": 4.5, "biaya_admin": 220000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Umum+Prima+Medika+Denpasar"},
        {"nama": "RS Umum Puri Raharja", "tipe": "C", "lat": -8.6540, "lon": 115.2225, "rating": 4.2, "biaya_admin": 150000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Umum+Puri+Raharja+Denpasar"},
        {"nama": "RS Umum Surya Husadha", "tipe": "C", "lat": -8.6750, "lon": 115.2100, "rating": 4.4, "biaya_admin": 200000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Umum+Surya+Husadha+Pulau+Serangan+Denpasar"},
        {"nama": "RS Umum Bhakti Rahayu", "tipe": "D", "lat": -8.6415, "lon": 115.2047, "rating": 4.1, "biaya_admin": 100000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Umum+Bhakti+Rahayu+Denpasar"},
        {"nama": "RS Umum Surya Husadha Cokroaminoto", "tipe": "D", "lat": -8.6402, "lon": 115.2010, "rating": 4.2, "biaya_admin": 90000, "gmaps": "https://www.google.com/maps/search/?api=1&query=RS+Umum+Surya+Husadha+Cokroaminoto+Denpasar"}
    ]
    return pd.DataFrame(data)

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
def init():
    if 'G' not in st.session_state:
        G = nx.Graph()
        G.add_weighted_edges_from(BASE_EDGES)
        st.session_state.G = G
    if 'all_coords' not in st.session_state:
        st.session_state.all_coords = copy.deepcopy(BASE_COORDS)
    
    defs = dict(
        step=1, role='Guest', df_rs=None, df_tier=None,
        kota_awal=None, tipe_rs_pilihan='Semua Tipe',
        rs_sel=None, rute=None, jarak_total=None,
        log_dijkstra=None, ai_text=None
    )
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v
init()

# ══════════════════════════════════════════════════════════════════════════════
#  UTILS
# ══════════════════════════════════════════════════════════════════════════════
def hav(c1, c2) -> float:
    R = 6371.0
    la1,lo1,la2,lo2 = map(math.radians, [c1[0],c1[1],c2[0],c2[1]])
    a = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return round(R*2*math.atan2(math.sqrt(a), math.sqrt(max(0,1-a))), 3)

def fresh_graph() -> nx.Graph:
    return st.session_state.G.copy()

# ══════════════════════════════════════════════════════════════════════════════
#  K-MEANS TIER
# ══════════════════════════════════════════════════════════════════════════════
def build_tier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    n = min(4, max(2, len(df)))
    
    scaler = MinMaxScaler()
    # Fitur: Rating (makin tinggi makin baik), Biaya (makin rendah makin baik), Jarak (makin dekat makin baik)
    X = scaler.fit_transform(df[['rating','biaya_admin','Jarak (Km)']].assign(
        biaya_inv=lambda d: -d['biaya_admin'],
        jarak_inv=lambda d: -d['Jarak (Km)'],
    )[['rating','biaya_inv','jarak_inv']])

    km = KMeans(n_clusters=n, random_state=42, n_init=10)
    df['Cluster'] = km.fit_predict(X)

    # Ranking berdasarkan Rating rata-rata di tiap cluster
    rank = (df.groupby('Cluster')['rating'].mean().sort_values(ascending=False).index.tolist())
    df['Tier']      = df['Cluster'].map({c: TIER_LABELS[i]  for i,c in enumerate(rank)}).fillna(TIER_LABELS[-1])
    df['TierShort'] = df['Cluster'].map({c: TIER_SHORT[i]   for i,c in enumerate(rank)}).fillna(TIER_SHORT[-1])
    df['TierTag']   = df['Cluster'].map({c: TIER_TAG[i]     for i,c in enumerate(rank)}).fillna(TIER_TAG[-1])
    df.drop(columns=['Cluster'], inplace=True)
    
    return df.sort_values(['TierShort', 'Jarak (Km)'], ascending=[True, True]).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DIJKSTRA
# ══════════════════════════════════════════════════════════════════════════════
def dijkstra(G: nx.Graph, start: str, goal: str):
    adj = {n: [] for n in G.nodes()}
    for u, v, d in G.edges(data=True):
        w = d.get('weight', 1.0)
        adj[u].append((v, w))
        adj[v].append((u, w))

    ctr = 0
    heap = [(0.0, ctr, start)]
    came = {}
    g = {n: float('inf') for n in adj}
    g[start] = 0.0
    vis = set()
    log = []
    step = 1

    while heap:
        dist, _, node = heapq.heappop(heap)
        if node in vis: continue
        vis.add(node)
        log.append({
            'Step': step, 'Node': node,
            'Jarak Kumulatif (km)': round(dist, 3),
            'Status': '✅ Tujuan' if node == goal else '🔄 Eksplorasi'
        })
        step += 1

        if node == goal:
            path, n = [], node
            while n in came:
                path.append(n); n = came[n]
            path.append(start); path.reverse()
            return path, round(g[goal], 3), log

        for nb, w in adj.get(node, []):
            if nb in vis: continue
            t = g[node] + w
            if t < g.get(nb, float('inf')):
                came[nb] = node; g[nb] = t
                ctr += 1
                heapq.heappush(heap, (t, ctr, nb))

    return None, float('inf'), log

# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH SVG VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════
def render_graph(G: nx.Graph, coords: dict, path=None, goal_node=None):
    W, H = 720, 400
    la = [v[0] for v in coords.values() if v]
    lo = [v[1] for v in coords.values() if v]
    if not la: return
    lamin,lamax = min(la),max(la)
    lomin,lomax = min(lo),max(lo)
    M = 65

    def tc(lat, lon):
        fx = (lon-lomin)/(lomax-lomin+1e-9)
        fy = 1-(lat-lamin)/(lamax-lamin+1e-9)
        return round(M+fx*(W-2*M),1), round(M+fy*(H-2*M),1)

    pos = {n: tc(coords[n][0], coords[n][1]) for n in G.nodes() if n in coords and coords[n]}
    path_set = set(path) if path else set()
    path_edges = set()
    if path and len(path)>=2:
        for i in range(len(path)-1):
            path_edges.add((path[i],path[i+1]))
            path_edges.add((path[i+1],path[i]))

    parts = [
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;background:#0f0f1a;border-radius:14px;border:1px solid rgba(255,255,255,0.07)">'
    ]

    for u, v, d in G.edges(data=True):
        if u not in pos or v not in pos: continue
        x1,y1 = pos[u]; x2,y2 = pos[v]
        w = d.get('weight', 1.0)
        hi = (u,v) in path_edges
        clr = '#3b82f6' if hi else 'rgba(255,255,255,0.1)'
        sw  = '2.5' if hi else '1'
        mx,my = (x1+x2)/2,(y1+y2)/2
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{clr}" stroke-width="{sw}"/>')
        if hi:
            parts.append(f'<text x="{mx}" y="{my-5}" text-anchor="middle" font-size="9" fill="#93c5fd" font-family="monospace">{w}km</text>')

    for node,(x,y) in pos.items():
        is_goal  = node == goal_node
        is_start = path and node == path[0]
        is_path  = node in path_set

        if is_goal: fill,stroke,r = '#ef4444','#b91c1c',11
        elif is_start: fill,stroke,r = '#10b981','#059669',10
        elif is_path: fill,stroke,r = '#3b82f6','#1d4ed8',8
        else: fill,stroke,r = '#1e1e35','rgba(255,255,255,0.15)',6

        parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        short = (node[:10]+'..') if len(node)>12 else node
        fw = '700' if (is_path or is_goal) else '400'
        fc = '#ef4444' if is_goal else ('#10b981' if is_start else ('#e2e8f0' if is_path else '#64748b'))
        parts.append(f'<text x="{x}" y="{y-r-4}" text-anchor="middle" font-size="9.5" font-weight="{fw}" fill="{fc}" font-family="Sora,sans-serif">{short}</text>')

    # Legend (Dipindah ke Pojok Kiri Bawah)
    parts += [
        '<rect x="10" y="310" width="130" height="80" rx="8" fill="#16162a" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>',
        '<circle cx="26" cy="330" r="6" fill="#10b981" stroke="#059669" stroke-width="1.5"/>',
        '<text x="38" y="334" font-size="9" fill="#94a3b8" font-family="sans-serif">Lokasi Awal</text>',
        '<circle cx="26" cy="348" r="6" fill="#ef4444" stroke="#b91c1c" stroke-width="1.5"/>',
        '<text x="38" y="352" font-size="9" fill="#94a3b8" font-family="sans-serif">RS Tujuan</text>',
        '<circle cx="26" cy="366" r="5" fill="#3b82f6" stroke="#1d4ed8" stroke-width="1.5"/>',
        '<text x="38" y="370" font-size="9" fill="#94a3b8" font-family="sans-serif">Jalur Dijkstra</text>',
        '</svg>'
    ]
    st.markdown('\n'.join(parts), unsafe_allow_html=True)
    
# ══════════════════════════════════════════════════════════════════════════════
#  STEP INDICATOR
# ══════════════════════════════════════════════════════════════════════════════
def steps_ui(cur: int):
    items = [('1','Pencarian'), ('2','Rekomendasi RS'), ('3','Analisis Medis')]
    html = '<div class="steps">'
    for i,(n,lbl) in enumerate(items):
        idx = i+1
        cls = 'active' if idx==cur else ('done' if idx<cur else '')
        ico = '✓' if idx<cur else n
        html += f'<div class="step {cls}"><div class="step-n">{ico}</div>{lbl}</div>'
        if i < len(items)-1:
            html += '<div class="step-div"></div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 0.8rem">
      <div style="font-size:2.5rem">🏥</div>
      <div style="font-size:1.05rem;font-weight:800;color:#e2e8f0;letter-spacing:-0.02em">HospFinder DSS</div>
      <div style="font-size:0.7rem;color:#64748b;margin-top:3px">Kawasan Denpasar · Bali</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr style="border-color:rgba(255,255,255,0.08)">', unsafe_allow_html=True)

    # Role
    st.markdown('<p style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#64748b;margin:0 0 5px">LOGIN PENGGUNA</p>', unsafe_allow_html=True)
    st.session_state.role = st.selectbox('Peran', ['Guest', 'Paramedis', 'Admin RS'], label_visibility='collapsed')
    
    st.markdown('<hr style="border-color:rgba(255,255,255,0.08)">', unsafe_allow_html=True)

    # API Key
    st.markdown('<p style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#64748b;margin:0 0 5px">AI ENGINE (GROQ)</p>', unsafe_allow_html=True)
    api_key = st.text_input('Groq API Key', type='password', placeholder='gsk_...', label_visibility='collapsed')
    if api_key:
        st.markdown('<div style="background:rgba(16,185,129,0.15);border-radius:8px;padding:6px 12px;font-size:0.75rem;color:#6ee7b7;font-weight:600">✓ AI Aktif — LLaMA 3.1</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:rgba(255,255,255,0.04);border-radius:8px;padding:6px 12px;font-size:0.75rem;color:#475569">AI tidak aktif</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:rgba(255,255,255,0.08)">', unsafe_allow_html=True)

    # Graph Stats
    n_n = st.session_state.G.number_of_nodes()
    n_e = st.session_state.G.number_of_edges()
    st.markdown(f"""
    <p style="font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#64748b;margin:0 0 8px">GRAPH DATABASE</p>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-bottom:10px">
      <div class="sb"><div class="sv" style="font-size:1.3rem;color:#93c5fd">{n_n}</div><div class="sl">Nodes</div></div>
      <div class="sb"><div class="sv" style="font-size:1.3rem;color:#fcd34d">{n_e}</div><div class="sl">Edges</div></div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HALAMAN 1 — FILTER & INPUT
# ══════════════════════════════════════════════════════════════════════════════
def page_filter():
    st.markdown("""
    <div class="hero">
      <div class="hero-pill">DSS · GRAPH · DIJKSTRA · K-MEANS · AI</div>
      <h1>🏥 HospFinder — DSS Rumah Sakit</h1>
      <p>Sistem Pendukung Keputusan berbasis <b>Machine Learning</b> & Algoritma <b>Dijkstra</b><br>
         untuk mencari dan menavigasi Rumah Sakit Tipe A/B/C/D di Kawasan Denpasar.</p>
    </div>
    """, unsafe_allow_html=True)
    steps_ui(1)

    df_rs_all = load_hospital_data()

    col_form, col_graph = st.columns([1, 1.7], gap='large')

    with col_form:
        st.markdown('<div class="card card-p">', unsafe_allow_html=True)
        st.markdown('<div class="sl-h">🗺️ PLANNER EVAKUASI & PENCARIAN</div>', unsafe_allow_html=True)

        kota_awal = st.selectbox('📍 Lokasi / Wilayah Awal', BASE_NODES, index=0)
        tipe_rs_pilihan = st.selectbox('🏥 Tujuan (Filter Tipe RS)', ['Semua Tipe', 'Tipe A', 'Tipe B', 'Tipe C', 'Tipe D'])
        
        with st.expander("ℹ️ Info Tipe Rumah Sakit"):
            st.markdown("""
            * **Tipe A:** Rujukan tertinggi (Pusat/Nasional). Spesialis & subspesialis paling lengkap.
            * **Tipe B:** Rujukan Provinsi. Medik spesialis luas & subspesialis terbatas.
            * **Tipe C:** Rujukan Kabupaten/Kota. Medik spesialis dasar.
            * **Tipe D:** Tingkat pertama (Transisi). Medik dasar umum sebelum dirujuk.
            """)

        if st.button('🔍 Cari Rute & Rekomendasi', type='primary', use_container_width=True):
            with st.spinner('Mengkalkulasi rute Dijkstra ke seluruh RS...'):
                time.sleep(0.3)
                df_rs = df_rs_all.copy()
                
                if tipe_rs_pilihan != "Semua Tipe":
                    tipe_huruf = tipe_rs_pilihan.replace("Tipe ", "") 
                    df_rs = df_rs[df_rs['tipe'] == tipe_huruf]
                    
                if df_rs.empty:
                    st.error(f"Maaf, tidak ada data Rumah Sakit untuk kategori {tipe_rs_pilihan}.")
                else:
                    jarak_list = []
                    for _, row in df_rs.iterrows():
                        # Hubungkan RS ke node wilayah terdekat
                        node_terdekat = min(BASE_NODES, key=lambda n: hav(BASE_COORDS[n], [row['lat'], row['lon']]))
                        dist_terdekat = hav(BASE_COORDS[node_terdekat], [row['lat'], row['lon']])
                        
                        temp_G = st.session_state.G.copy()
                        temp_G.add_node("Temp_RS")
                        temp_G.add_edge(node_terdekat, "Temp_RS", weight=round(dist_terdekat, 2))
                        
                        # Hitung Dijkstra
                        _, dist_dijkstra, _ = dijkstra(temp_G, kota_awal, "Temp_RS")
                        jarak_list.append(dist_dijkstra)
                        
                    df_rs['Jarak (Km)'] = jarak_list
                    
                    # Generate K-Means Tiers
                    st.session_state.df_tier = build_tier(df_rs)
                    st.session_state.kota_awal = kota_awal
                    st.session_state.tipe_rs_pilihan = tipe_rs_pilihan
                    st.session_state.step = 2
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Stats DB
        st.markdown(f"""
        <div class="card">
          <div class="sl-h">📊 STATISTIK DATABASE RS</div>
          <div class="sg">
            <div class="sb"><div class="sv">{len(df_rs_all)}</div><div class="sl">Total RS</div></div>
            <div class="sb"><div class="sv" style="color:#fcd34d">{round(df_rs_all["rating"].mean(),1)}</div><div class="sl">Avg Rating</div></div>
            <div class="sb"><div class="sv" style="color:#6ee7b7">4</div><div class="sl">Kategori</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_graph:
        st.markdown('<div class="card card-p">', unsafe_allow_html=True)
        tab1, tab2, tab3 = st.tabs(['🕸️ Graph Visual', '📋 Edge Database', '🗺️ Peta Wilayah'])

        with tab1:
            st.markdown('<div class="sl-h">WEIGHTED UNDIRECTED GRAPH — KAWASAN DENPASAR</div>', unsafe_allow_html=True)
            render_graph(st.session_state.G, st.session_state.all_coords)

        with tab2:
            edge_df = pd.DataFrame([
                {'Asal': u, 'Tujuan': v, 'Bobot (km)': d.get('weight',0)}
                for u,v,d in st.session_state.G.edges(data=True)
            ])
            st.dataframe(edge_df, hide_index=True, use_container_width=True, height=320)

        with tab3:
            m = folium.Map(location=[-8.6500, 115.2167], zoom_start=11, tiles='OpenStreetMap')
            for nama, coord in BASE_COORDS.items():
                folium.Marker(coord, tooltip=f'📍 {nama}', icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
            for _, row in df_rs_all.iterrows():
                folium.CircleMarker(
                    [row['lat'], row['lon']], radius=4, color='#ef4444', fill=True, fill_opacity=0.8,
                    tooltip=f"🏥 {row['nama']} (Tipe {row['tipe']})"
                ).add_to(m)
            st_folium(m, width='100%', height=350, returned_objects=[])

        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HALAMAN 2 — REKOMENDASI ML
# ══════════════════════════════════════════════════════════════════════════════
def page_rekomen():
    df = st.session_state.df_tier
    awal = st.session_state.kota_awal

    st.markdown(f"""
    <div style="margin-bottom:1rem">
      <h2 style="font-size:1.6rem;font-weight:800;color:#e2e8f0;margin:0">🏆 Rekomendasi Rumah Sakit — K-Means Tier List</h2>
      <p style="color:#64748b;font-size:0.85rem;margin:4px 0 0">
        Lokasi Awal: <b style="color:#93c5fd">{awal}</b> ·
        Kategori: <b style="color:#fcd34d">{st.session_state.tipe_rs_pilihan}</b> ·
        Ditemukan: <b style="color:#e2e8f0">{len(df)} Fasilitas Medis</b>
      </p>
    </div>
    """, unsafe_allow_html=True)
    steps_ui(2)

    c1, c2 = st.columns([1, 2], gap='large')

    with c1:
        if st.button('⬅️ Ubah Parameter'):
            st.session_state.step = 1; st.rerun()

        ts = len(df[df['TierShort']=='Tier S'])
        avr = round(df['rating'].mean(), 2)
        st.markdown(f"""
        <div class="sg sg4" style="grid-template-columns:repeat(3,1fr);margin-bottom:1rem">
          <div class="sb"><div class="sv" style="font-size:1.2rem">{len(df)}</div><div class="sl">RS Cocok</div></div>
          <div class="sb"><div class="sv" style="font-size:1.2rem;color:#fcd34d">{ts}</div><div class="sl">Tier S</div></div>
          <div class="sb"><div class="sv" style="font-size:1.2rem">{avr}</div><div class="sl">Avg Rating</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="sl-h">🎯 PILIH RUMAH SAKIT TUJUAN</div>', unsafe_allow_html=True)
        sel = st.selectbox('Rumah Sakit:', df['nama'].tolist(), label_visibility='collapsed')

        for _, row in df.head(6).iterrows():
            is_s = row['nama'] == sel
            cls  = 'sel' if is_s else ''
            st.markdown(f"""
            <div class="kost-card {cls}">
              <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <div class="kost-name">{row["nama"]}</div>
                <span class="tag {row["TierTag"]}" style="flex-shrink:0;margin-left:6px">{row["TierShort"]}</span>
              </div>
              <div class="kost-meta" style="margin-top:5px">
                <span class="tag tag-r">Tipe {row["tipe"]}</span>
                <span class="tag tag-y">⭐ {row["rating"]}</span>
                <span class="tag tag-g">📍 {row["Jarak (Km)"]} km</span>
                <span class="tag tag-b">Rp {row["biaya_admin"]//1000}k</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        if st.button('🚀 Hitung Navigasi Dijkstra', type='primary', use_container_width=True):
            rs_row = df[df['nama'] == sel].iloc[0]
            coord_rs = [rs_row['lat'], rs_row['lon']]
            nama_rs = rs_row['nama']

            G_tmp = fresh_graph()
            G_tmp.add_node(nama_rs)
            
            node_terdekat = min(BASE_NODES, key=lambda n: hav(BASE_COORDS[n], coord_rs))
            dist_terdekat = hav(BASE_COORDS[node_terdekat], coord_rs)
            G_tmp.add_edge(node_terdekat, nama_rs, weight=round(dist_terdekat, 2))

            coords_tmp = dict(st.session_state.all_coords)
            coords_tmp[nama_rs] = coord_rs

            rute, total, log = dijkstra(G_tmp, awal, nama_rs)

            st.session_state.rs_sel      = rs_row
            st.session_state.rute        = rute
            st.session_state.jarak_total = total
            st.session_state.log_dijk    = log
            st.session_state.G_tmp       = G_tmp
            st.session_state.coords_tmp  = coords_tmp
            st.session_state.ai_text     = None
            st.session_state.step        = 3
            st.rerun()

    with c2:
        st.markdown('<div class="sl-h">🗺️ PETA SEBARAN RUMAH SAKIT TERFILTER</div>', unsafe_allow_html=True)
        m2 = folium.Map(location=[-8.670, 115.210], zoom_start=12, tiles='OpenStreetMap')

        # Node Awal
        c_awal = BASE_COORDS[awal]
        folium.Marker(c_awal, tooltip=f'📍 Titik Awal: {awal}', icon=folium.Icon(color='green', icon='user')).add_to(m2)

        tier_color = {'Tier S':'red','Tier A':'orange','Tier B':'blue','Tier C':'gray'}
        for _, row in df.iterrows():
            clr = tier_color.get(row['TierShort'], 'gray')
            popup = f"<b>{row['nama']}</b><br>Tipe {row['tipe']} · ⭐{row['rating']}<br>{row['Jarak (Km)']} km dari {awal}"
            folium.Marker(
                [row['lat'], row['lon']], tooltip=f"🏥 {row['nama']} ({row['TierShort']})",
                popup=folium.Popup(popup, max_width=250), icon=folium.Icon(color=clr, icon='plus')
            ).add_to(m2)

        st_folium(m2, width='100%', height=530, returned_objects=[])

# ══════════════════════════════════════════════════════════════════════════════
#  HALAMAN 3 — ANALISIS LENGKAP
# ══════════════════════════════════════════════════════════════════════════════
def page_analisis():
    rs    = st.session_state.rs_sel
    rute  = st.session_state.rute
    total = st.session_state.jarak_total

    st.markdown(f"""
    <div style="margin-bottom:1rem">
      <h2 style="font-size:1.6rem;font-weight:800;color:#e2e8f0;margin:0">✅ Analisis Keputusan & Navigasi Medis</h2>
      <p style="color:#64748b;font-size:0.85rem;margin:4px 0 0">
        Evakuasi dari {st.session_state.kota_awal} → {rs["nama"]} · <b style="color:#ef4444">{total} km</b>
      </p>
    </div>
    """, unsafe_allow_html=True)
    steps_ui(3)

    if st.button('⬅️ Kembali ke Menu Rekomendasi'):
        st.session_state.step = 2; st.rerun()

    # Rute Banner
    nodes_html = ''
    for i, node in enumerate(rute):
        cls = 's' if i==0 else ('e' if i==len(rute)-1 else '')
        nodes_html += f'<div class="rn {cls}">{node}</div>'
        if i < len(rute)-1: nodes_html += '<div class="ra">→</div>'
    st.markdown(f'<div class="route" style="margin-bottom:1.2rem">{nodes_html}</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.6], gap='large')

    with col_l:
        tabs = st.tabs(['📋 Detail RS', '🕸️ Graph', '🧮 Log Dijkstra', '🤖 AI Analisis'])

        with tabs[0]:
            st.markdown(f"""
            <div class="card card-g">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1rem">
                <div>
                  <div style="font-size:1.1rem;font-weight:800;color:#e2e8f0">{rs["nama"]}</div>
                  <div style="font-size:0.8rem;color:#ef4444;margin-top:3px;font-weight:600">Rumah Sakit Tipe {rs["tipe"]}</div>
                </div>
                <span class="tag {rs["TierTag"]}" style="flex-shrink:0">{rs["TierShort"]}</span>
              </div>
              <div class="sg" style="margin-bottom:0.8rem">
                <div class="sb"><div class="sv">⭐{rs["rating"]}</div><div class="sl">Rating</div></div>
                <div class="sb"><div class="sv" style="font-size:1.1rem;color:#ef4444">{total}km</div><div class="sl">Jarak Total</div></div>
                <div class="sb"><div class="sv" style="font-size:1.1rem">Rp{rs["biaya_admin"]//1000}k</div><div class="sl">Est. Admin</div></div>
              </div>
              <a href="{rs['gmaps']}" target="_blank"
                style="display:inline-flex;align-items:center;gap:6px;background:#ef4444;color:white;
                       border-radius:8px;padding:8px 16px;font-size:0.8rem;font-weight:600;text-decoration:none">
                🗺️ Navigasi via Google Maps
              </a>
            </div>
            """, unsafe_allow_html=True)

        with tabs[1]:
            st.markdown('<div class="sl-h">VISUALISASI GRAPH + JALUR DIJKSTRA</div>', unsafe_allow_html=True)
            G_viz = st.session_state.get('G_tmp', fresh_graph())
            coords_viz = st.session_state.get('coords_tmp', st.session_state.all_coords)
            render_graph(G_viz, coords_viz, path=rute, goal_node=rs['nama'])

        with tabs[2]:
            st.markdown('<div class="sl-h">LOG LANGKAH KOMPUTASI DIJKSTRA</div>', unsafe_allow_html=True)
            df_log = pd.DataFrame(st.session_state.log_dijk)
            st.dataframe(df_log, hide_index=True, use_container_width=True, height=280)

        with tabs[3]:
            st.markdown('<div class="sl-h">AI MEDICAL LOGISTICS — GROQ LLAMA 3.1</div>', unsafe_allow_html=True)
            if not api_key:
                st.info('💡 Masukkan Groq API Key di sidebar untuk mengaktifkan AI analysis.')
            else:
                if st.button('✨ Minta Analisis Kelayakan AI', type='primary'):
                    with st.spinner('LLaMA 3.1 sedang menganalisis rute medis...'):
                        try:
                            from groq import Groq
                            client = Groq(api_key=api_key)
                            prompt = f"""Kamu adalah asisten logistik medis DSS. Berikan analisis rekomendasi RS dalam 3 paragraf bahasa Indonesia singkat:
Lokasi Awal: {st.session_state.kota_awal}
RS Tujuan: {rs['nama']} (Tipe {rs['tipe']})
Jarak: {total} km (Rute: {" → ".join(rute)})
Rating: {rs['rating']}
Tier ML: {rs['Tier']}
Fokuskan pada: 1) Kapabilitas medis (Tipe RS tsb), 2) Efisiensi jarak tempuh evakuasi, 3) Kesimpulan kelayakan."""
                            res = client.chat.completions.create(
                                messages=[{'role':'user','content':prompt}],
                                model='llama-3.1-8b-instant', max_tokens=500
                            )
                            st.session_state.ai_text = res.choices[0].message.content
                        except Exception as e:
                            st.error(f'Error Groq API: {e}')

                if st.session_state.ai_text:
                    st.markdown(f"""
                    <div class="ai-box">
                      <span class="ai-lbl">🤖 ANALISIS AI DSS</span>
                      <p>{st.session_state.ai_text.replace(chr(10),"<br>")}</p>
                    </div>
                    """, unsafe_allow_html=True)

    with col_r:
        tab_map, tab_gmaps, tab_tier = st.tabs(['🗺️ Peta Jalur Evakuasi', '📍 Google Maps Embed', '📊 K-Means Tier List'])

        with tab_map:
            m3 = folium.Map(location=[-8.660, 115.200], zoom_start=12, tiles='OpenStreetMap')
            coords_tmp = st.session_state.get('coords_tmp', st.session_state.all_coords)
            
            rute_coords = []
            for node in rute:
                c = coords_tmp.get(node)
                if c: rute_coords.append(c)

            if len(rute_coords) >= 2:
                folium.PolyLine(rute_coords, color='#ef4444', weight=5, opacity=0.9, tooltip=f'Jalur: {total} km').add_to(m3)
                for i, (node, coord) in enumerate(zip(rute, rute_coords)):
                    if i == 0:
                        folium.Marker(coord, tooltip=f'📍 START: {node}', icon=folium.Icon(color='green', icon='user')).add_to(m3)
                    elif i == len(rute)-1:
                        folium.Marker(coord, tooltip=f'🏥 GOAL: {node}', icon=folium.Icon(color='red', icon='plus')).add_to(m3)
                    else:
                        folium.Marker(coord, tooltip=f'Lintasan: {node}', icon=folium.Icon(color='blue', icon='info-sign')).add_to(m3)
                        
            st_folium(m3, width='100%', height=440, returned_objects=[])

        with tab_gmaps:
            nama_enc = urllib.parse.quote(f"{rs['nama']} Denpasar Bali")
            iframe   = f"https://maps.google.com/maps?q={nama_enc}&t=&z=16&ie=UTF8&iwloc=&output=embed"
            components.html(f'<iframe width="100%" height="480" src="{iframe}" frameborder="0" scrolling="no" style="border-radius:12px"></iframe>', height=500)

        with tab_tier:
            st.markdown('<div class="sl-h">TIER LIST LENGKAP HASIL K-MEANS</div>', unsafe_allow_html=True)
            df_show = st.session_state.df_tier[['TierShort','nama','tipe','rating','Jarak (Km)','biaya_admin']].copy()
            df_show.columns = ['Tier','Nama RS','Tipe','Rating','Jarak (km)','Est. Biaya']
            st.dataframe(df_show, hide_index=True, use_container_width=True, height=400)

# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
s = st.session_state.get('step', 1)
if   s == 1: page_filter()
elif s == 2: page_rekomen()
elif s == 3: page_analisis()
else:
    st.session_state.step = 1; st.rerun()