import os
import json
import numpy as np
import pandas as pd
try:
    import geopandas as gpd
except ImportError:
    gpd = None
import plotly.graph_objects as go
import streamlit as st

WORK_DIR = os.environ.get("ASEAN_WORK_DIR", ".")

st.set_page_config(
    page_title="Carbon Trading, But for Whom? | ASEAN Spatial Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================== PALETTE & CONSTANTS
BG, SURF, SURF_ALT, INK, MUTE, LINE = "#0a0e17", "#111827", "#1e293b", "#f8fafc", "#94a3b8", "#1f2937"

ARCHETYPE_LABELS = {
    "HIGH_EMISSION_INDUSTRIAL_CORE": "Industrial Emission Source (Offset Buyer)",
    "UPWIND_COMMUNITY_GREEN_SINK": "Upwind Community Green Sink (Offset Provider)",
    "TRANSITIONAL_RESTORATION_ZONE": "Transitional & Restoration Target Zone",
}

ROLE_COLORS = {
    "Industrial Emission Source (Offset Buyer)": "#ff1744",
    "Upwind Community Green Sink (Offset Provider)": "#00e676",
    "Transitional & Restoration Target Zone": "#ff9100",
}

ROLE_DESC = {
    "Industrial Emission Source (Offset Buyer)": "High fossil CO₂, dense built-up footprint, and intense nightlights (Clusters C3 & C7). Primary carbon credit demand centers.",
    "Upwind Community Green Sink (Offset Provider)": "Intact forest matrix, coastal mangroves, and high-leverage island sinks (Clusters C0, C1, C2 & C4).",
    "Transitional & Restoration Target Zone": "Degraded shrub-grasslands and intensive peri-urban cropland buffers (Clusters C5 & C6).",
}

CUSTOM_CLUSTER_DISPLAY = {
    0: "C0: Coastal Mosaic Green Sink",
    1: "C1: Intact Forest Matrix",
    2: "C2: High-Leverage Island Sink Hub",
    3: "C3: Mega Industrial Urban Core",
    4: "C4: Dense Coastal Mangrove Belt",
    5: "C5: Degraded Shrubgrass Restoration Zone",
    6: "C6: Intensive Cropland Peri-Urban Buffer",
    7: "C7: Secondary Industrial Urban Belt",
}

CLUSTER_PALETTE = {
    "C0: Coastal Mosaic Green Sink": "#38bdf8",            # Sky Blue
    "C1: Intact Forest Matrix": "#16a34a",                 # Deep Forest Green
    "C2: High-Leverage Island Sink Hub": "#a3e635",        # Bright Chartreuse Lime
    "C3: Mega Industrial Urban Core": "#ff1744",           # Vivid Crimson Red
    "C4: Dense Coastal Mangrove Belt": "#a855f7",          # Royal Purple
    "C5: Degraded Shrubgrass Restoration Zone": "#facc15", # Bright Gold Yellow
    "C6: Intensive Cropland Peri-Urban Buffer": "#f97316", # Burnt Orange
    "C7: Secondary Industrial Urban Belt": "#ec4899",      # Hot Magenta Pink
}

CLASS_LABEL = {
    "area_treecover": "Tree Cover (Forest)",
    "area_mangrove": "Mangrove Ecosystem",
    "area_cropland": "Cropland Mosaic",
    "area_shrubgrass": "Shrub & Grassland",
}

CLASS_COLORS = {
    "Tree Cover (Forest)": "#16a34a",
    "Mangrove Ecosystem": "#a855f7",
    "Cropland Mosaic": "#f97316",
    "Shrub & Grassland": "#facc15",
}

LISA_LABEL = {
    0: "Not Significant (p ≥ 0.05)",
    1: "High-High (Compounded Emission Hotspot)",
    2: "Low-High (Low Cell Near High Neighbors)",
    3: "Low-Low (Upwind Green Sink Coldspot)",
    4: "High-Low (High Cell Near Low Neighbors)",
}
LISA_COLORS = {
    "Not Significant (p ≥ 0.05)": "#334155",
    "High-High (Compounded Emission Hotspot)": "#ff1744",
    "Low-High (Low Cell Near High Neighbors)": "#facc15",
    "Low-Low (Upwind Green Sink Coldspot)": "#00e676",
    "High-Low (High Cell Near Low Neighbors)": "#38bdf8",
}
LISA_MEANING = {
    "Not Significant (p ≥ 0.05)": "No statistically significant local spatial clustering of spillover scores.",
    "High-High (Compounded Emission Hotspot)": "Cluster of elevated upwind spillover scores linked to higher fossil CO₂ pressure.",
    "Low-High (Low Cell Near High Neighbors)": "Low spillover cell surrounded by high-score neighbors.",
    "Low-Low (Upwind Green Sink Coldspot)": "Statistically significant cluster where upwind vegetation strongly suppresses CO₂.",
    "High-Low (High Cell Near Low Neighbors)": "High spillover cell surrounded by low-score neighbors.",
}

SCALES = {
    "priority": [[0.0, "#1e293b"], [0.35, "#38bdf8"], [0.7, "#a855f7"], [1.0, "#ff1744"]],
    "need": [[0.0, "#1e293b"], [0.4, "#7c3aed"], [0.75, "#f97316"], [1.0, "#facc15"]],
    "sink": [[0.0, "#1e293b"], [0.4, "#0284c7"], [0.75, "#16a34a"], [1.0, "#a3e635"]],
}

COUNTRY_LABEL = {"Burma": "Myanmar"}

WANTED = [
    "grid_id", "country_id", "centroid_lon", "centroid_lat", "land_area_ha", "coastal_sliver",
    "co2_odiac", "ntl_viirs", "area_treecover", "area_mangrove", "area_cropland",
    "area_shrubgrass", "area_builtup", "hotspot_count", "hotspot_frp_sum",
    "slx_spillover_ann", "lisa_I", "lisa_p", "lisa_q",
    "sink_strength_co2", "dominant_class_co2",
    "shap_tot_treecover", "shap_tot_mangrove", "shap_tot_cropland", "shap_tot_shrubgrass",
    "cluster_id", "policy_zone", "cluster_name", "need_rank", "lever_rank", "investment_priority",
]

NAV_TABS = [
    "⊞  01 · Policy Zones & Typology",
    "◈  02 · CO₂ Severity & Priority",
    "◫  03 · Commodity Leverage (SHAP & SLX)",
    "⇄  04 · Source-to-Sink Pairing",
    "◎  05 · Airshed & LISA Spillover",
    "≡  06 · Variable Dictionary & Formulas",
]

# ============================================================== ULTRA-LIGHT TABULAR LOADER
def restore_names(df):
    rename = {}
    for w in WANTED:
        if w not in df.columns and w[:10] in df.columns:
            rename[w[:10]] = w
    if "cluster_na" in df.columns and "cluster_name" not in df.columns:
        rename["cluster_na"] = "cluster_name"
    return df.rename(columns=rename)


@st.cache_resource(show_spinner="Loading spatial dataset...")
def load_data(work_dir):
    p = lambda f: os.path.join(work_dir, f)
    fast_tab_path = p("dashboard_tabular_cache_10km_v6.pkl")
    prof_path = p("05_zone_commodity_profile.csv")

    prof = None
    if os.path.exists(prof_path):
        prof = pd.read_csv(prof_path)
        prof["cluster_id"] = prof["cluster_id"].astype(int)
        prof["label"] = prof["cluster_id"].map(CUSTOM_CLUSTER_DISPLAY)
        prof["role"] = prof["post_hoc_archetype"].map(ARCHETYPE_LABELS).fillna("Transitional & Restoration Target Zone")
        prof["modal_class"] = prof["modal_dominant_class_co2"].map(CLASS_LABEL).fillna(prof["modal_dominant_class_co2"])

    if os.path.exists(fast_tab_path):
        df_tab = pd.read_pickle(fast_tab_path)
    else:
        zones = restore_names(gpd.read_file(p("05_asean_policy_zones_10km.shp")))
        cen = zones.geometry.centroid.to_crs("EPSG:4326")
        zones["clon"] = cen.x.astype("float32").values
        zones["clat"] = cen.y.astype("float32").values
        df_tab = pd.DataFrame(zones.drop(columns="geometry", errors="ignore"))
        df_tab["country"] = df_tab["country_id"].map(lambda c: COUNTRY_LABEL.get(c, c))
        df_tab["cluster_id"] = df_tab["cluster_id"].astype(int)

    if prof is not None:
        idx = prof.set_index("cluster_id")
        df_tab["cluster_label"] = df_tab["cluster_id"].map(idx["label"])
        df_tab["role"] = df_tab["cluster_id"].map(idx["role"])
    else:
        df_tab["cluster_label"] = df_tab["cluster_id"].map(CUSTOM_CLUSTER_DISPLAY)
        df_tab["role"] = df_tab["policy_zone"].map(ARCHETYPE_LABELS).fillna("Transitional & Restoration Target Zone")

    df_tab["class_label"] = df_tab["dominant_class_co2"].map(CLASS_LABEL).fillna("Tree Cover (Forest)")
    df_tab["lisa_label"] = df_tab.get("lisa_q", pd.Series(0, index=df_tab.index)).fillna(0).astype(int).map(LISA_LABEL)
    df_tab["hover"] = (
        "<b>" + df_tab["grid_id"].astype(str) + "</b> (" + df_tab["country"].astype(str) + ")"
        + "<br>" + df_tab["cluster_label"].astype(str)
        + "<br>CO₂: " + df_tab["co2_odiac"].round(0).astype(int).astype(str) + " t"
        + " | Priority: " + df_tab["investment_priority"].round(2).astype(str)
        + "<br>Sink: " + df_tab["class_label"].astype(str)
    )
    return df_tab, prof


@st.cache_resource(show_spinner=False)
def load_corridors(work_dir, top_n_cap=2000):
    cache_c = os.path.join(work_dir, "dashboard_corridors_cache_v6.pkl")
    if os.path.exists(cache_c):
        try:
            return pd.read_pickle(cache_c)
        except Exception:
            pass
    path = os.path.join(work_dir, "02_airshed_corridors_10km.shp")
    if not os.path.exists(path):
        return None
    c = gpd.read_file(path).to_crs("EPSG:4326")
    c = c.sort_values("weight_wij", ascending=False).head(top_n_cap).reset_index(drop=True)
    x0, y0, x1, y1 = [], [], [], []
    for g in c.geometry:
        xs, ys = g.xy
        x0.append(float(xs[0])); y0.append(float(ys[0]))
        x1.append(float(xs[-1])); y1.append(float(ys[-1]))
    df_c = pd.DataFrame({
        "src_id": c["src_id"].values, "dst_id": c["dst_id"].values,
        "weight_wij": c["weight_wij"].values,
        "x0": x0, "y0": y0, "x1": x1, "y1": y1
    })
    try:
        df_c.to_pickle(cache_c)
    except Exception:
        pass
    return df_c


@st.cache_resource(show_spinner=False)
def load_links(work_dir):
    cache_l = os.path.join(work_dir, "dashboard_links_top5000_v6.pkl")
    if os.path.exists(cache_l):
        try:
            return pd.read_pickle(cache_l)
        except Exception:
            pass
    path = os.path.join(work_dir, "05_industrial_community_investment_links.csv")
    if not os.path.exists(path):
        return None
    wanted_cols = [
        "industrial_grid_id", "industrial_country", "industrial_cluster_id",
        "upwind_community_grid_id", "upwind_country", "upwind_cluster_id",
        "dominant_class_co2_j", "restoration_shrubgrass_ha_j", "treecover_ha_j",
        "mangrove_ha_j", "investment_flow_ijk"
    ]
    avail = pd.read_csv(path, nrows=1).columns.tolist()
    use = [c for c in wanted_cols if c in avail]
    d = pd.read_csv(path, usecols=use, nrows=15000)
    d["ind_country"] = d["industrial_country"].map(lambda c: COUNTRY_LABEL.get(c, c))
    d["com_country"] = d["upwind_country"].map(lambda c: COUNTRY_LABEL.get(c, c))
    d["class_label"] = d["dominant_class_co2_j"].map(CLASS_LABEL).fillna("Tree Cover (Forest)")
    d["ind_cluster_label"] = d["industrial_cluster_id"].astype(int).map(CUSTOM_CLUSTER_DISPLAY)
    if "upwind_cluster_id" in d.columns:
        d["upwind_cluster_label"] = d["upwind_cluster_id"].astype(int).map(CUSTOM_CLUSTER_DISPLAY)
    try:
        d.to_pickle(cache_l)
    except Exception:
        pass
    return d


@st.cache_resource(show_spinner=False)
def load_evidence(work_dir):
    imp, slx = None, None
    f1 = os.path.join(work_dir, "04_ml_commodity_importance.csv")
    f2 = os.path.join(work_dir, "03_slx_coefficients.csv")
    if os.path.exists(f1):
        imp = pd.read_csv(f1)
        imp["label"] = imp["commodity_class"].map(CLASS_LABEL)
    if os.path.exists(f2):
        slx = pd.read_csv(f2)
        slx = slx[slx["role"] == "MAIN_POLICY_COMMODITY"].copy()
        slx["kind"] = np.where(slx["variable"].str.startswith("W_"), "Upwind Spillover (θ₁)", "Local Effect (β₁)")
        slx["cls"] = "area_" + slx["variable"].str.extract(r"area_([a-z]+)")[0]
        slx["label"] = slx["cls"].map(CLASS_LABEL)
    return imp, slx


# ============================================================== STYLING (CLEAN TYPOGRAPHY + KATEX MATH SUPPORT)
st.markdown(f"""<style>
.stApp {{ background-color: {BG}; color: {INK}; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
.block-container {{ padding-top: 1.0rem; padding-bottom: 2.2rem; max-width: 1640px; }}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}

.hero {{
    padding: 18px 24px; border-radius: 10px; margin-bottom: 14px; color: {INK};
    border: 1px solid {LINE};
    background: linear-gradient(180deg, #111827 0%, #0d131f 100%);
}}
.hero .tag {{ font-size: 0.70rem; letter-spacing: 0.14em; text-transform: uppercase; color: #38bdf8; font-weight: 600; }}
.hero .ttl {{ font-size: 1.52rem; font-weight: 700; line-height: 1.25; margin: 4px 0 4px 0; color: #f8fafc; letter-spacing: -0.01em; }}
.hero .sub {{ font-size: 0.86rem; color: #cbd5e1; max-width: 1220px; line-height: 1.5; }}

div[data-testid="stSegmentedControl"] button {{
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 6px 14px !important;
}}

.kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin: 6px 0 14px 0; }}
.kpi {{
    border: 1px solid {LINE}; background: {SURF}; border-radius: 8px; padding: 13px 16px;
    border-left: 3px solid #38bdf8;
}}
.kpi .cv {{ font-size: 1.30rem; font-weight: 700; color: {INK}; line-height: 1.2; font-variant-numeric: tabular-nums; }}
.kpi .cl {{ font-size: 0.70rem; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 4px; font-weight: 500; }}

.card {{
    border: 1px solid {LINE}; background: {SURF}; border-radius: 8px; padding: 15px 17px; margin-bottom: 12px;
}}
.card h4 {{
    margin: 0 0 10px 0; font-size: 0.78rem; font-weight: 700; color: #e2e8f0;
    text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid {LINE}; padding-bottom: 7px;
}}
.card p {{ font-size: 0.82rem; line-height: 1.55; margin: 0 0 8px 0; color: #cbd5e1; }}
.card p:last-child {{ margin-bottom: 0; }}

.sec-hdr {{
    font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em;
    color: #f1f5f9; margin: 16px 0 10px 0; padding-bottom: 6px; border-bottom: 1px solid {LINE};
}}

.lrow {{ display: flex; gap: 11px; margin: 0 0 11px 0; align-items: flex-start; }}
.swatch {{
    width: 14px; height: 14px; border-radius: 3px; margin-top: 2px; flex: 0 0 14px;
    border: 1px solid rgba(255,255,255,0.28);
}}
.ltxt {{ flex: 1; min-width: 0; }}
.lt {{ display: flex; justify-content: space-between; gap: 8px; font-size: 0.82rem; color: {INK}; }}
.lt b {{ font-weight: 600; overflow-wrap: anywhere; }}
.lp {{ color: {INK}; font-variant-numeric: tabular-nums; white-space: nowrap; font-size: 0.80rem; font-weight: 600; }}
.ls {{ font-size: 0.74rem; color: {MUTE}; line-height: 1.42; margin: 2px 0 4px 0; }}
.btrack {{
    height: 5px; border-radius: 3px; background: #1e293b; overflow: hidden; margin-top: 4px;
}}
.bfill {{ height: 100%; border-radius: 3px; }}

.mini {{
    font-size: 0.81rem; display: flex; justify-content: space-between; gap: 10px;
    padding: 5.5px 0; border-bottom: 1px solid {LINE}; color: #cbd5e1;
}}
.mini:last-child {{ border-bottom: none; }}
.mini b {{ color: {INK}; font-variant-numeric: tabular-nums; font-weight: 600; text-align: right; }}

/* KaTeX LaTeX Display Styling - Prevent clipping */
div[data-testid="stLatex"] {{
    background: #0d1322;
    border: 1px solid #1f2937;
    border-radius: 6px;
    padding: 8px 10px;
    margin: 6px 0 10px 0;
    overflow-x: auto;
}}
.katex {{ font-size: 0.86em !important; color: #7dd3fc !important; }}

div[data-testid="stPlotlyChart"] {{ border: 1px solid {LINE}; border-radius: 8px; overflow: hidden; background: {SURF}; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: {LINE}!important; border-radius: 8px!important; background: {SURF}; }}
</style>""", unsafe_allow_html=True)


def html(s): st.markdown(s.replace("\n", ""), unsafe_allow_html=True)
def card(title, body): html(f"<div class='card'><h4>{title}</h4>{body}</div>")
def sec_header(title): html(f"<div class='sec-hdr'>{title}</div>")
def chips(items):
    html("<div class='kpis'>" + "".join(
        f"<div class='kpi'><div class='cv'>{v}</div><div class='cl'>{l}</div></div>" for v, l in items
    ) + "</div>")


def formula_card(title, latex_str, bullets_md):
    with st.container(border=True):
        html(f"<div style='font-size:0.78rem;font-weight:700;color:#e2e8f0;text-transform:uppercase;letter-spacing:0.08em;border-bottom:1px solid {LINE};padding-bottom:6px;margin-bottom:8px'>{title}</div>")
        st.latex(latex_str)
        st.markdown(bullets_md)


def legend_rows(rows):
    out = ""
    for label, color, pct, desc in rows:
        desc_html = f"<div class='ls'>{desc}</div>" if desc else ""
        out += (
            f"<div class='lrow'>"
            f"<span class='swatch' style='background:{color}'></span>"
            f"<div class='ltxt'>"
            f"<div class='lt'><b>{label}</b><span class='lp'>{pct:.1f}%</span></div>"
            f"{desc_html}<div class='btrack'><div class='bfill' "
            f"style='width:{min(pct, 100):.1f}%;background:{color}'></div></div></div></div>"
        )
    return out


def shares(df, col, order, colors, desc=None, weight=None):
    if weight is None:
        vc = df[col].value_counts(normalize=True) * 100
    else:
        s = df.groupby(col)[weight].sum()
        vc = s / s.sum() * 100 if s.sum() > 0 else s * 0
    return [
        (c, colors.get(c, "#94a3b8"), float(vc.get(c, 0.0)), (desc or {}).get(c, ""))
        for c in order if c in vc.index
    ]


def mini_list(rows):
    return "".join(f"<div class='mini'><span>{a}</span><b>{b}</b></div>" for a, b in rows)


# ============================================================== INSTANT WEBGL MAP & CHART BUILDERS
BASEMAPS = {
    "Dark Matter (Carto Streets & Borders)": "carto-darkmatter",
    "Minimal Slate (Solid Navy Background)": {"version": 8, "sources": {}, "layers": [
        {"id": "bg", "type": "background", "paint": {"background-color": "#0b1120"}}
    ]},
}
HOVER = dict(align="left", bgcolor="#111827", font=dict(color=INK, size=12), bordercolor="#38bdf8")


def dark_chart(fig, **kw):
    fig.update_layout(paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=INK, size=11.5), **kw)
    fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#334155", linecolor="#334155", automargin=True)
    fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#334155", linecolor="#334155", automargin=True)
    return fig


def view_for(df):
    minx, maxx = float(df["clon"].min()), float(df["clon"].max())
    miny, maxy = float(df["clat"].min()), float(df["clat"].max())
    span = max(maxx - minx, (maxy - miny) * 1.1, 0.5)
    zoom = float(np.clip(np.log2(360.0 / span) - 0.85, 2.5, 8.5))
    return {"lat": (miny + maxy) / 2, "lon": (minx + maxx) / 2}, zoom


def auto_marker_size(df):
    span = max(float(df["clon"].max() - df["clon"].min()), float(df["clat"].max() - df["clat"].min()), 0.5)
    if span > 25:
        return 4.0
    elif span > 12:
        return 5.8
    elif span > 5:
        return 8.5
    return 11.5


def asean_land_base_trace(df):
    """Renders the full ASEAN terrestrial landmass silhouette underneath vector/corridor layers."""
    msize = auto_marker_size(df)
    return go.Scattermap(
        lon=df["clon"],
        lat=df["clat"],
        mode="markers",
        marker=dict(size=msize, color="#1e293b", opacity=0.88),
        hoverinfo="skip",
        name="ASEAN Landmass (10 km Grid)"
    )


def finish_map(fig, df, basemap, height=580, bottom=0):
    center, zoom = view_for(df)
    fig.update_layout(
        map_style=BASEMAPS[basemap], map_center=center, map_zoom=zoom,
        height=height, showlegend=False,
        margin=dict(l=0, r=0, t=0, b=bottom), paper_bgcolor=SURF, hoverlabel=HOVER
    )
    return fig


def categorical_map(df, col, colors, basemap, order=None):
    cats = [c for c in (order or sorted(df[col].dropna().unique())) if (df[col] == c).any()]
    cols = [colors.get(c, "#94a3b8") for c in cats]
    n = max(len(cats), 1)
    z = df[col].map({c: i + 0.5 for i, c in enumerate(cats)}).astype("float32")
    scale = []
    for i, c in enumerate(cols):
        scale += [[i / n, c], [(i + 1) / n, c]]

    msize = auto_marker_size(df)
    fig = go.Figure(go.Scattermap(
        lon=df["clon"], lat=df["clat"], mode="markers",
        marker=dict(size=msize, color=z, colorscale=scale, cmin=0, cmax=n, opacity=0.95),
        text=df["hover"], hovertemplate="%{text}<extra></extra>"
    ))
    return finish_map(fig, df, basemap)


def continuous_map(df, col, scale, title, basemap):
    cbar = dict(
        orientation="h", thickness=10, len=0.42, x=0.5, xanchor="center", y=-0.02, yanchor="top",
        outlinewidth=0, title=dict(text=title, side="right", font=dict(color=MUTE, size=11)),
        tickfont=dict(color=MUTE, size=10), tickvals=[0, 0.25, 0.5, 0.75, 1.0]
    )
    msize = auto_marker_size(df)
    fig = go.Figure(go.Scattermap(
        lon=df["clon"], lat=df["clat"], mode="markers",
        marker=dict(
            size=msize, color=df[col].astype("float32"), colorscale=scale, cmin=0, cmax=1,
            opacity=0.95, showscale=True, colorbar=cbar
        ),
        text=df["hover"], hovertemplate="%{text}<extra></extra>"
    ))
    return finish_map(fig, df, basemap, height=600, bottom=52)


def render_fast_map(fig, key):
    st.plotly_chart(
        fig, use_container_width=True, key=key,
        config={"displayModeBar": False, "scrollZoom": True}
    )


def render_cell_inspector(df_sub, full_df):
    default_id = df_sub.sort_values("investment_priority", ascending=False)["grid_id"].iloc[0]
    cur_id = st.session_state.get("inspected_grid_id", default_id)
    if cur_id not in full_df["grid_id"].values:
        cur_id = default_id

    top_options = df_sub.sort_values("investment_priority", ascending=False)["grid_id"].head(300).tolist()
    if cur_id not in top_options:
        top_options = [cur_id] + top_options

    picked_id = st.selectbox(
        "Inspect Grid Cell ID (Sorted by Priority):",
        options=top_options,
        index=top_options.index(cur_id),
        key="inspector_select"
    )
    st.session_state["inspected_grid_id"] = picked_id
    r = full_df[full_df["grid_id"] == picked_id].iloc[0]

    if r["cluster_id"] in [3, 7]:
        rec_text = (
            "<b>Policy Action — Industrial/Urban Source (Offset Buyer):</b> Mandate carbon offset procurement "
            "directed to upwind community sinks (Clusters C0, C1, C2, C4) alongside point-source abatement."
        )
    elif r["cluster_id"] in [5, 6]:
        rec_text = (
            f"<b>Policy Action — Restoration & Transition Target:</b> Prioritize community reforestation across "
            f"<b>{r.get('area_shrubgrass', 0):,.0f} ha</b> of degraded shrub-grassland and agroforestry buffers."
        )
    else:
        rec_text = (
            f"<b>Policy Action — Upwind Community Green Sink:</b> Channel community-based PES and carbon "
            f"finance into local <b>{r['class_label']}</b> conservation."
        )

    c_hex = CLUSTER_PALETTE.get(r["cluster_label"], "#38bdf8")
    cluster_badge = (
        f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;"
        f"background:{c_hex};margin-right:6px'></span>{r['cluster_label']}"
    )

    body = mini_list([
        ("Grid ID & Country", f"{r['grid_id']} · {r['country']}"),
        ("Centroid (Lat, Lon)", f"{r['clat']:.4f}°N, {r['clon']:.4f}°E"),
        ("Cluster Typology", cluster_badge),
        ("Macro Policy Role", str(r["role"])),
        ("Land Area (coastal_sliver)", f"{r.get('land_area_ha',0):,.0f} ha ({'Yes' if r.get('coastal_sliver',0)==1 else 'No'})"),
        ("Fossil CO₂ (co2_odiac)", f"{r['co2_odiac']:,.1f} t"),
        ("CO₂ Severity (need_rank)", f"{r['need_rank']:.3f} ({r['need_rank']*100:.1f} pct)"),
        ("Sink Strength (sink_strength_co2)", f"{r['sink_strength_co2']:.3f} ({r['lever_rank']*100:.1f} pct)"),
        ("Investment Priority", f"{r['investment_priority']:.3f}"),
        ("Dominant Sink Class", str(r["class_label"])),
        ("Tree Cover / Mangrove", f"{r.get('area_treecover',0):,.0f} / {r.get('area_mangrove',0):,.0f} ha"),
        ("Cropland / Shrubgrass", f"{r.get('area_cropland',0):,.0f} / {r.get('area_shrubgrass',0):,.0f} ha"),
        ("Built-up / Nightlights", f"{r.get('area_builtup',0):,.0f} ha / {r.get('ntl_viirs',0):.2f} nW"),
    ]) + f"<p style='margin-top:8px;padding-top:8px;border-top:1px solid {LINE};color:#6ee7b7'>{rec_text}</p>"

    card("Grid Cell Telemetry (10 × 10 km)", body)


# ============================================================== HEADER & GLOBAL NAVIGATION BAR
html(
    "<div class='hero'>"
    "<div class='tag'>ASEAN 10 × 10 KM SPATIAL ECONOMETRICS & MACHINE LEARNING PIPELINE · H1 2026 · ESRI:102028</div>"
    "<div class='ttl'>Carbon Trading, But for Whom?</div>"
    "<div class='sub'>Executive Decision Support System: Democratizing ASEAN's carbon markets through community-driven "
    "green investment. Integrating ODIAC Fossil CO₂ Severity, Anisotropic Wind-Shed Econometrics (SLX), "
    "TreeSHAP Commodity Decomposition, K-Means++ Typology (k* = 8), and Source-to-Sink Investment Pairing.</div>"
    "</div>"
)

try:
    gdf, prof = load_data(WORK_DIR)
except Exception as e:
    st.error(f"Could not load spatial artifacts from {WORK_DIR}. Ensure Stages 1–5B have been executed.\n\nDetails: {e}")
    st.stop()

tb_nav, tb_filter = st.columns([5.8, 1.2], gap="small", vertical_alignment="center")
with tb_filter:
    with st.popover("⛭  Map & Filter Settings", use_container_width=True):
        countries = sorted(gdf["country"].unique())
        picked = st.multiselect("Filter ASEAN Countries", countries, default=countries)
        drop_sliver = st.checkbox("Exclude coastal slivers (< 1,000 ha)", value=False)
        basemap = st.radio(
            "Basemap Style",
            list(BASEMAPS),
            index=0,
            help="'Dark Matter' displays ASEAN country boundaries, coastlines, and labels."
        )

with tb_nav:
    if hasattr(st, "segmented_control"):
        view = st.segmented_control("Module", NAV_TABS, default=NAV_TABS[0], label_visibility="collapsed") or NAV_TABS[0]
    else:
        view = st.radio("Module", NAV_TABS, horizontal=True, label_visibility="collapsed")

df = gdf[gdf["country"].isin(picked)]
if drop_sliver and "coastal_sliver" in df.columns:
    df = df[df["coastal_sliver"] == 0]
if df.empty:
    st.warning("No grid cells match the current country filter.")
    st.stop()

cluster_order = [CUSTOM_CLUSTER_DISPLAY[i] for i in range(8) if CUSTOM_CLUSTER_DISPLAY[i] in df["cluster_label"].values]

# ============================================================== VIEW 01: POLICY ZONES & TYPOLOGY
if "01 ·" in view:
    ind_cells = int(df["cluster_id"].isin([3, 7]).sum())
    sink_cells = int(df["cluster_id"].isin([0, 1, 2, 4]).sum())
    rest_cells = int(df["cluster_id"].isin([5, 6]).sum())

    chips([
        (f"{len(df):,}", "Active Grid Cells (10 × 10 km)"),
        (f"{ind_cells:,} ({ind_cells/len(df)*100:.1f}%)", "Industrial Emission Cores (C3 & C7)"),
        (f"{sink_cells:,} ({sink_cells/len(df)*100:.1f}%)", "Upwind Green Sinks (C0, C1, C2, C4)"),
        (f"{rest_cells:,} ({rest_cells/len(df)*100:.1f}%)", "Restoration & Transition Zones (C5 & C6)"),
    ])

    left, right = st.columns([3.2, 1.35], gap="medium")
    with right:
        mode_color = st.radio(
            "Spatial Classification Layer:",
            ["8 Data-Driven Clusters (Stage 5B)", "3 Macro Policy Roles"],
            horizontal=False
        )
        if mode_color.startswith("8"):
            card(
                "8-Cluster Distribution (k* = 8)",
                legend_rows(shares(df, "cluster_label", cluster_order, CLUSTER_PALETTE))
            )
        else:
            card(
                "Macro Policy Roles",
                legend_rows(shares(df, "role", list(ROLE_COLORS), ROLE_COLORS, ROLE_DESC))
            )

        formula_card(
            "Centroid Standardization (Stage 5)",
            r"\begin{aligned} Z_{i,m} &= \frac{X_{i,m} - \mu_m}{\sigma_m} \\ k^{*} &= 8, \quad \bar{J}_{\text{boot}} = 0.9767 \end{aligned}",
            "- $\\mathbf{X_{i,m}}$: Physical value of feature $m \\in \\{1,\\dots,8\\}$ in grid cell $i$.\n"
            "- $\\mathbf{\\bar{J}_{\\text{boot}}}$: Hungarian-matched bootstrap Jaccard stability ($B = 100$)."
        )

        render_cell_inspector(df, gdf)

    with left:
        if mode_color.startswith("8"):
            fig_map = categorical_map(df, "cluster_label", CLUSTER_PALETTE, basemap, order=cluster_order)
        else:
            fig_map = categorical_map(df, "role", ROLE_COLORS, basemap, order=list(ROLE_COLORS))
        render_fast_map(fig_map, key="map_zones")

        if prof is not None:
            sec_header("Stage 5B — Post-Hoc Cluster Typology & Physical Centroid Profile")
            cols_needed = [
                "label", "role", "cell_count", "pct_total_cells", "modal_class",
                "mean_co2_odiac", "mean_sink_strength_co2", "mean_investment_priority", "bootstrap_jaccard"
            ]
            avail_p = [c for c in cols_needed if c in prof.columns]
            t = prof.sort_values("cluster_id")[avail_p].copy()
            rename_map = {
                "label": "Cluster Typology (Stage 5B)",
                "role": "Macro Policy Role",
                "cell_count": "Cell Count",
                "pct_total_cells": "Share of ASEAN (%)",
                "modal_class": "Dominant Sink Class",
                "mean_co2_odiac": "Mean CO₂ (ton)",
                "mean_sink_strength_co2": "Mean Sink (SHAP)",
                "mean_investment_priority": "Mean Priority",
                "bootstrap_jaccard": "Jaccard Stability",
            }
            t.columns = [rename_map.get(c, c) for c in t.columns]
            st.dataframe(t.round(3), hide_index=True, use_container_width=True)

            zc = [c for c in prof.columns if c.startswith("Z_")]
            if zc:
                sec_header("Standardized Centroid Z-Scores Across 8 Clustering Features (Clipped at ±3 SD)")
                z_short_labels = {
                    "Z_co2_odiac": "Fossil CO₂",
                    "Z_sink_strength_co2": "Sink (SHAP)",
                    "Z_area_treecover": "Tree Cover",
                    "Z_area_mangrove": "Mangrove",
                    "Z_area_cropland": "Cropland",
                    "Z_area_shrubgrass": "Shrubgrass",
                    "Z_area_builtup": "Built-Up",
                    "Z_ntl_viirs": "Nightlights",
                }
                x_labels = [z_short_labels.get(c, c[2:]) for c in zc]
                y_labels = prof.sort_values("cluster_id")["label"].tolist()
                hm = go.Figure(go.Heatmap(
                    z=prof.sort_values("cluster_id")[zc].clip(-3, 3).values,
                    x=x_labels,
                    y=y_labels,
                    zmin=-3, zmax=3, xgap=4, ygap=4,
                    colorscale=[[0.0, "#38bdf8"], [0.5, "#111827"], [1.0, "#ff1744"]],
                    text=prof.sort_values("cluster_id")[zc].round(2).values,
                    texttemplate="%{text:.2f}", textfont=dict(size=11.5, color=INK),
                    colorbar=dict(
                        thickness=10, len=0.88, tickfont=dict(size=10, color=MUTE),
                        outlinewidth=0, title=dict(text="Z-Score", font=dict(size=10, color=MUTE))
                    )
                ))
                dark_chart(
                    hm, height=130 + 38 * len(prof),
                    margin=dict(l=10, r=20, t=16, b=45),
                    yaxis=dict(autorange="reversed", tickfont=dict(size=11.5, color=INK)),
                    xaxis=dict(side="bottom", tickangle=0, tickfont=dict(size=11.5, color=INK))
                )
                st.plotly_chart(hm, use_container_width=True, config={"displayModeBar": False})

# ============================================================== VIEW 02: CO2 SEVERITY & PRIORITY INDEX
elif "02 ·" in view:
    left, right = st.columns([3.2, 1.35], gap="medium")
    with right:
        layer_choice = st.radio(
            "Select Continuous Index Layer:",
            [
                "Investment Priority Index (Geometric Mean)",
                "Fossil CO₂ Pollution Severity (Need Rank)",
                "Ecological Sink Leverage (Lever Rank)",
            ]
        )
        min_thr = st.slider("Minimum Index Filter:", 0.0, 0.90, 0.0, 0.05)

    col_map = {
        "Investment Priority Index (Geometric Mean)": ("investment_priority", SCALES["priority"], "Priority Index (0–1)"),
        "Fossil CO₂ Pollution Severity (Need Rank)": ("need_rank", SCALES["need"], "CO₂ Need Rank (0–1)"),
        "Ecological Sink Leverage (Lever Rank)": ("lever_rank", SCALES["sink"], "Sink Lever Rank (0–1)"),
    }
    target_col, target_scale, bar_title = col_map[layer_choice]
    sub = df[df[target_col] >= min_thr]
    if sub.empty:
        sub = df

    chips([
        (f"{len(sub):,}", "Displayed Grid Cells"),
        (f"{sub['co2_odiac'].sum()/1e6:,.2f} Mt", "Cumulative Fossil CO₂"),
        (f"{sub['investment_priority'].mean():.3f}", "Mean Priority Score"),
        (f"{(sub[target_col] >= 0.90).mean()*100:.1f}%", "Share in ASEAN Top 10%"),
    ])

    with left:
        fig_cont = continuous_map(sub, target_col, target_scale, bar_title, basemap)
        render_fast_map(fig_cont, key="map_severity")

    with right:
        formula_card(
            "Geometric Priority Index (Stage 5)",
            r"\text{Priority}_i = \sqrt{\,\text{Need}_i \times \text{Lever}_i\,}",
            "- $\\mathbf{\\text{Need}_i} = F_{\\text{emp}}(\\text{CO}_{2,i}) \\in [0, 1]$: Empirical percentile rank of ODIAC fossil $\\text{CO}_2$ emissions.\n"
            "- $\\mathbf{\\text{Lever}_i} = F_{\\text{emp}}(\\text{Sink}_i) \\in [0, 1]$: Empirical percentile rank of total $\\text{CO}_2$-reducing SHAP sink strength."
        )
        top_cells = sub.nlargest(6, target_col)
        card(
            f"Highest Ranked Cells ({bar_title})",
            mini_list([(f"{r.grid_id} · {r.country}", f"{getattr(r, target_col):.3f}") for r in top_cells.itertuples()])
        )
        render_cell_inspector(sub, gdf)

# ============================================================== VIEW 03: GREEN COMMODITY POWER (SHAP & SLX)
elif "03 ·" in view:
    imp, slx = load_evidence(WORK_DIR)
    dom_mode = df["class_label"].mode().iloc[0]

    chips([
        ("Tree Cover (0.2174)", "Highest Mean |SHAP| Leverage"),
        ("Mangrove (94.23%)", "Highest Consistency (% Cells Reducing CO₂)"),
        ("R² = 0.9977", "5-Fold Spatial Block CV (Hist-XGBoost)"),
        (dom_mode, "Modal Dominant Sink Class"),
    ])

    left, right = st.columns([3.2, 1.35], gap="medium")
    with right:
        per = st.radio(
            "SLX Seasonal Wind Window (Stage 3):",
            ["ann", "djf", "jja"],
            format_func=lambda x: {"ann": "Annual (H1 2026)", "djf": "NE Monsoon (DJF)", "jja": "SW Monsoon (JJA)"}[x],
            horizontal=False
        )
        card(
            "Dominant Sink Share Across Cells",
            legend_rows(shares(df, "class_label", list(CLASS_COLORS), CLASS_COLORS))
        )
        formula_card(
            "SLX & TreeSHAP Formulation (Stages 3–4)",
            r"\begin{aligned} Y_i &= \alpha + \mathbf{X}_i\boldsymbol{\beta}_1 + (\mathbf{W}\mathbf{X})_i\boldsymbol{\theta}_1 \\ &\quad + \mathbf{C}_i\boldsymbol{\gamma}_1 + (\mathbf{W}\mathbf{C})_i\boldsymbol{\gamma}_2 + \varepsilon_i \\[4pt] \text{Sink}_i &= -\sum_{k=1}^{4} \min\!\left(0,\, \phi_{i,k}^{\text{loc}} + \phi_{i,k}^{W}\right) \end{aligned}",
            "- $\\boldsymbol{\\beta}_1, \\phi_{i,k}^{\\text{loc}}$: Local within-cell effect of green class $k$ on $\\ln(\\text{CO}_{2,i}+1)$.\n"
            "- $\\boldsymbol{\\theta}_1, \\phi_{i,k}^{W}$: Upwind airshed spillover effect within $250\\text{ km}$."
        )
        render_cell_inspector(df, gdf)

    with left:
        fig_dom = categorical_map(df, "class_label", CLASS_COLORS, basemap, order=list(CLASS_COLORS))
        render_fast_map(fig_dom, key="map_commodity")

        if imp is not None or slx is not None:
            c_chart1, c_chart2 = st.columns(2, gap="medium")
            if imp is not None:
                with c_chart1:
                    sec_header("Stage 4 — TreeSHAP Magnitude Decomposition (Mean |SHAP|)")
                    y_imp = imp["label"].tolist()
                    f1 = go.Figure()
                    f1.add_bar(
                        y=y_imp, x=imp["mean_abs_shap_local"],
                        name="Local Effect (|φ_loc|)", orientation="h", marker_color="#16a34a",
                        text=imp["mean_abs_shap_local"].round(3), textposition="outside",
                        textfont=dict(size=10.5, color="#86efac"), cliponaxis=False
                    )
                    f1.add_bar(
                        y=y_imp, x=imp["mean_abs_shap_upwind"],
                        name="Upwind Spillover (|φ_W|)", orientation="h", marker_color="#38bdf8",
                        text=imp["mean_abs_shap_upwind"].round(3), textposition="outside",
                        textfont=dict(size=10.5, color="#bae6fd"), cliponaxis=False
                    )
                    max_x = float(max(imp["mean_abs_shap_local"].max(), imp["mean_abs_shap_upwind"].max())) * 1.28
                    dark_chart(
                        f1, barmode="group", height=320,
                        margin=dict(l=10, r=48, t=42, b=44),
                        legend=dict(
                            orientation="h", y=1.14, x=0.0, xanchor="left", yanchor="bottom",
                            font=dict(size=11, color=INK), bgcolor="rgba(0,0,0,0)"
                        ),
                        xaxis=dict(
                            range=[0, max_x],
                            title=dict(text="Mean |SHAP| Contribution to ln(CO₂ + 1)", font=dict(size=10.5, color=MUTE))
                        ),
                        yaxis=dict(autorange="reversed", tickfont=dict(size=11.5, color=INK))
                    )
                    st.plotly_chart(f1, use_container_width=True, config={"displayModeBar": False})

            if slx is not None:
                with c_chart2:
                    sec_header(f"Stage 3 — SLX Spatial Econometric Coefficients ({per.upper()}, 95% Conley HAC CI)")
                    s_ = slx[slx["period"] == per]
                    order = list(CLASS_LABEL.values())
                    f2 = go.Figure()
                    for kind, off, colr in [
                        ("Local Effect (β₁)", -0.16, "#16a34a"),
                        ("Upwind Spillover (θ₁)", 0.16, "#38bdf8")
                    ]:
                        d = s_[s_["kind"] == kind].set_index("label").reindex(order)
                        f2.add_trace(go.Scatter(
                            x=d["beta_slx"], y=[i + off for i in range(len(order))], mode="markers", name=kind,
                            marker=dict(size=9, color=colr),
                            error_x=dict(type="data", array=(1.96 * d["se_conley_hac"]).values, color=colr, thickness=1.5),
                            hovertemplate=kind + ": %{x:.4f}<extra></extra>"
                        ))
                    f2.add_vline(x=0, line_color="#64748b", line_dash="dash", line_width=1)
                    dark_chart(
                        f2, height=320,
                        margin=dict(l=10, r=24, t=42, b=44),
                        legend=dict(
                            orientation="h", y=1.14, x=0.0, xanchor="left", yanchor="bottom",
                            font=dict(size=11, color=INK), bgcolor="rgba(0,0,0,0)"
                        ),
                        xaxis=dict(title=dict(text="SLX Elasticity Coefficient (±1.96 Conley HAC SE)", font=dict(size=10.5, color=MUTE))),
                        yaxis=dict(
                            tickmode="array", tickvals=list(range(len(order))),
                            ticktext=order, autorange="reversed", tickfont=dict(size=11.5, color=INK)
                        )
                    )
                    st.plotly_chart(f2, use_container_width=True, config={"displayModeBar": False})

# ============================================================== VIEW 04: SOURCE-TO-SINK INVESTMENT PAIRING
elif "04 ·" in view:
    links = load_links(WORK_DIR)
    if links is None:
        st.info("File 05_industrial_community_investment_links.csv was not found in the work directory.")
        st.stop()

    lk = links[links["ind_country"].isin(picked)]
    left, right = st.columns([3.2, 1.35], gap="medium")

    with right:
        pick_c = st.selectbox("Filter Industrial Buyer Country:", ["All Countries"] + sorted(lk["ind_country"].unique()))
        pick_cl = st.selectbox("Filter Industrial Buyer Cluster:", ["C3 & C7 (All Industrial Cores)", "C3: Mega Industrial Urban Core", "C7: Secondary Industrial Urban Belt"])
        n_links = st.slider("Top Investment Corridors Displayed:", 50, 600, 250, 50)

    if pick_c != "All Countries":
        lk = lk[lk["ind_country"] == pick_c]
    if pick_cl.startswith("C3:"):
        lk = lk[lk["industrial_cluster_id"] == 3]
    elif pick_cl.startswith("C7:"):
        lk = lk[lk["industrial_cluster_id"] == 7]

    top = lk.nlargest(n_links, "investment_flow_ijk")

    chips([
        ("647,226", "Total Active Upwind Corridors (Stage 5B)"),
        (f"{top['industrial_grid_id'].nunique():,}", "Industrial Buyer Cells (Shown)"),
        (f"{top['upwind_community_grid_id'].nunique():,}", "Upwind Community Receiver Cells"),
        (f"{top['restoration_shrubgrass_ha_j'].sum():,.0f} ha", "Upwind Shrubgrass Restoration Area"),
    ])

    with left:
        pos = gdf.set_index("grid_id")[["clon", "clat"]]
        a_ = pos.reindex(top["industrial_grid_id"]).values
        b_ = pos.reindex(top["upwind_community_grid_id"]).values

        # 1. Always draw the ASEAN terrestrial landmass base trace first!
        traces = [asean_land_base_trace(df)]

        # 2. Draw directed investment corridors by dominant green commodity class
        for cls, color in CLASS_COLORS.items():
            m = (top["class_label"] == cls).values
            lons, lats = [], []
            for (x0, y0), (x1, y1) in zip(a_[m], b_[m]):
                lons += [x0, x1, None]
                lats += [y0, y1, None]
            if lons:
                traces.append(go.Scattermap(
                    lon=lons, lat=lats, mode="lines",
                    line=dict(width=2.2, color=color), opacity=0.92, hoverinfo="skip"
                ))

        # 3. Draw Industrial Buyer Nodes (Red) & Upwind Community Nodes (Green)
        traces.append(go.Scattermap(
            lon=a_[:, 0], lat=a_[:, 1], mode="markers",
            marker=dict(size=8.5, color="#ff1744"),
            text=top["industrial_grid_id"] + " (" + top["ind_country"] + ")",
            hovertemplate="<b>Industrial Buyer:</b> %{text}<extra></extra>"
        ))
        traces.append(go.Scattermap(
            lon=b_[:, 0], lat=b_[:, 1], mode="markers",
            marker=dict(size=6.0, color="#00e676"),
            text=top["upwind_community_grid_id"] + " (" + top["com_country"] + ") · " + top["class_label"],
            hovertemplate="<b>Upwind Community Sink:</b> %{text}<extra></extra>"
        ))

        fig_flow = finish_map(go.Figure(traces), df, basemap, height=580)
        render_fast_map(fig_flow, key="map_pairing")

        sec_header("Source-to-Sink Pairing Matrix: Industrial Cluster (X) → Green Commodity (Y) → Upwind Community (Z)")
        group_cols = ["ind_cluster_label", "ind_country", "class_label", "com_country"]
        if "upwind_cluster_label" in top.columns:
            group_cols = ["ind_cluster_label", "ind_country", "class_label", "upwind_cluster_label", "com_country"]

        g = (
            top.groupby(group_cols)
            .agg(
                Corridors=("investment_flow_ijk", "size"),
                Total_Flow=("investment_flow_ijk", "sum"),
                Forest_Ha=("treecover_ha_j", "sum"),
                Mangrove_Ha=("mangrove_ha_j", "sum"),
                Restoration_Shrub_Ha=("restoration_shrubgrass_ha_j", "sum"),
            )
            .reset_index()
            .sort_values("Total_Flow", ascending=False)
            .head(15)
        )
        rename_g = {
            "ind_cluster_label": "Industrial Buyer Cluster (X)",
            "ind_country": "Buyer Country",
            "class_label": "Green Commodity Lever (Y)",
            "upwind_cluster_label": "Upwind Community Cluster (Z)",
            "com_country": "Community Country",
            "Corridors": "Corridors",
            "Total_Flow": "Cumulative Flow Score",
            "Forest_Ha": "Tree Cover (ha)",
            "Mangrove_Ha": "Mangrove (ha)",
            "Restoration_Shrub_Ha": "Shrub Restoration (ha)",
        }
        g.columns = [rename_g.get(c, c) for c in g.columns]
        st.dataframe(g.round(2), hide_index=True, use_container_width=True)

    with right:
        formula_card(
            "Source-to-Sink Pairing Rule (Stage 5B)",
            r"\begin{aligned} \text{Flow}_{i \to j, k} &= \text{Priority}_i \times W_{ij} \\ &\quad \times \mathbb{I}\!\left(\text{Class}_j^{*} = k\right) \end{aligned}",
            "- **Red Nodes ($i$)**: High-emission industrial cores ($\\text{C3}, \\text{C7}$) requiring carbon offsets.\n"
            "- **Green Nodes ($j$)**: Upwind community cells supplying airshed services ($W_{ij} > 0$) to cell $i$."
        )
        card(
            "Investment Flow Share by Commodity",
            legend_rows(shares(top, "class_label", list(CLASS_COLORS), CLASS_COLORS, weight="investment_flow_ijk"))
        )
        card(
            "Top Upwind Receiver Countries",
            mini_list([
                (c, f"{v:.2f}")
                for c, v in top.groupby("com_country")["investment_flow_ijk"].sum().sort_values(ascending=False).head(5).items()
            ])
        )

# ============================================================== VIEW 05: UPWIND AIRSHED & LISA SPILLOVER
elif "05 ·" in view:
    left, right = st.columns([3.2, 1.35], gap="medium")
    with right:
        air_mode = st.radio(
            "Select Spatial Econometric Layer:",
            ["Local Moran's I (LISA Spillover Clusters)", "Strongest Upwind Airshed Corridors (W_ij)"],
            horizontal=False
        )

    if air_mode.startswith("Local"):
        sig_pct = (df["lisa_label"] != "Not Significant (p ≥ 0.05)").mean() * 100
        ll_pct = (df["lisa_label"] == "Low-Low (Upwind Green Sink Coldspot)").mean() * 100
        hh_pct = (df["lisa_label"] == "High-High (Compounded Emission Hotspot)").mean() * 100

        chips([
            (f"{len(df):,}", "Evaluated Grid Cells"),
            (f"{sig_pct:.1f}%", "Statistically Significant (p < 0.05)"),
            (f"{ll_pct:.1f}%", "Low-Low Sink Coldspots (Lower CO₂)"),
            (f"{hh_pct:.1f}%", "High-High Spillover Hotspots"),
        ])

        with left:
            fig_lisa = categorical_map(df, "lisa_label", LISA_COLORS, basemap, order=list(LISA_MEANING.keys()))
            render_fast_map(fig_lisa, key="map_lisa")

        with right:
            formula_card(
                "SLX Spillover & Local Moran's I (Stage 3)",
                r"\begin{aligned} S_i &= \sum_{k=1}^{4} (\mathbf{W}\ln\mathbf{X}_k)_i\,\hat{\theta}_{1,k} \\ I_i &= \frac{z_i}{m_2}\sum_{j=1}^{N} W_{ij}z_j \end{aligned}",
                "- **Low-Low ($I_i > 0, z_i < 0$)**: Regional coldspots where dense upwind green cover significantly suppresses $\\text{CO}_2$.\n"
                "- **High-High ($I_i > 0, z_i > 0$)**: Transboundary corridors with compounded upwind emission pressure."
            )
            card(
                "LISA Cluster Distribution",
                legend_rows(shares(df, "lisa_label", list(LISA_MEANING.keys()), LISA_COLORS, LISA_MEANING))
            )
            render_cell_inspector(df, gdf)

    else:
        cor = load_corridors(WORK_DIR)
        if cor is None:
            st.info("File 02_airshed_corridors_10km.shp was not found in the work directory.")
            st.stop()

        with right:
            n_cor = st.slider("Number of Strongest Airshed Vectors:", 50, min(2000, len(cor)), min(400, len(cor)), 50)

        dst_country = gdf.set_index("grid_id")["country"]
        cc = cor[cor["dst_id"].map(dst_country).isin(picked)].head(n_cor)

        chips([
            (f"{len(cc):,}", "Displayed Upwind Vectors"),
            (f"{cc['weight_wij'].mean():.3f}", "Mean Row-Standardized Weight (W_ij)"),
            ("250 km", "Anisotropic Capture Radius (d_max)"),
            ("α = 1.5", "Distance Decay Exponent"),
        ])

        with left:
            lons, lats = [], []
            for r_c in cc.itertuples():
                lons += [r_c.x0, r_c.x1, None]
                lats += [r_c.y0, r_c.y1, None]

            traces = [
                asean_land_base_trace(df),
                go.Scattermap(lon=lons, lat=lats, mode="lines", line=dict(width=1.9, color="#38bdf8"), opacity=0.88, hoverinfo="skip"),
                go.Scattermap(lon=cc["x1"], lat=cc["y1"], mode="markers", marker=dict(size=5.5, color="#ff1744"), hoverinfo="skip"),
            ]
            fig_cor = finish_map(go.Figure(traces), df, basemap, height=580)
            render_fast_map(fig_cor, key="map_corridors")

        with right:
            formula_card(
                "Anisotropic Wind-Shed Kernel (Stage 2)",
                r"\begin{aligned} w_{ij} &= \frac{V_j \cdot \max\!\left(0,\, \cos(\phi_{ji} - \theta_j)\right)}{d_{ji}^{\alpha}} \\[4pt] W_{ij} &= \frac{w_{ij}}{\sum_{j'} w_{ij'}} \end{aligned}",
                "- $V_j = \\sqrt{u_j^2 + v_j^2}$: ERA5-Land $10\\text{ m}$ wind speed at source cell $j$.\n"
                "- $\\phi_{ji}, \\theta_j$: Geodesic bearing $(j \\to i)$ and wind direction angle ($d_{ji} \\le 250\\text{ km}, \\alpha = 1.5$)."
            )
            render_cell_inspector(df, gdf)

# ============================================================== VIEW 06: VARIABLE DICTIONARY & FORMULAS
else:
    sec_header("Mathematical Formulation of the 5-Stage Spatial Pipeline")
    f_c1, f_c2 = st.columns(2, gap="medium")
    with f_c1:
        formula_card(
            "Stage 2 — Anisotropic Upwind Spatial Weights Matrix",
            r"w_{ij} = \begin{cases} \dfrac{V_j \cos(\phi_{ji} - \theta_j)}{d_{ji}^{\alpha}} & \text{if } 0 < d_{ji} \le d_{\max},\; \cos(\phi_{ji} - \theta_j) > 0 \\[6pt] 0 & \text{otherwise} \end{cases}",
            "Row-standardized as $W_{ij} = w_{ij} / \\sum_{j'} w_{ij'}$ over $N = 50,621$ cells ($d_{\\max} = 250\\text{ km}, \\alpha = 1.5$)."
        )
        formula_card(
            "Stage 3 — Spatial Lag of X (SLX) & Conley Spatial HAC SE",
            r"\begin{aligned} \ln(\text{CO}_{2,i} + 1) &= \alpha + \sum_{k=1}^{4} \beta_{1,k}\ln(X_{i,k}+1) + \sum_{k=1}^{4} \theta_{1,k}(\mathbf{W}\ln\mathbf{X}_k)_i \\ &\quad + \mathbf{C}_i\boldsymbol{\gamma}_1 + (\mathbf{W}\mathbf{C})_i\boldsymbol{\gamma}_2 + \varepsilon_i \end{aligned}",
            "Inference uses Conley (1999) spatial HAC covariance $\\hat{\\mathbf{V}}_{\\text{Conley}} = (\\mathbf{X}'\\mathbf{X})^{-1}\\!\\left(\\sum_i \\sum_j K(d_{ij}/250)\\,u_i u_j \\mathbf{x}_i \\mathbf{x}_j'\\right)\\!(\\mathbf{X}'\\mathbf{X})^{-1}$."
        )
    with f_c2:
        formula_card(
            "Stage 4 — TreeSHAP Decomposition & Ecological Sink Strength",
            r"\begin{aligned} \text{Sink}_i &= -\sum_{k \in \mathcal{K}_{\text{kom}}} \min\!\left(0,\, \phi_{i,k}^{\text{loc}} + \phi_{i,k}^{W}\right) \\ \text{Class}_i^{*} &= \arg\min_{k \in \mathcal{K}_{\text{kom}}} \left(\phi_{i,k}^{\text{loc}} + \phi_{i,k}^{W}\right) \end{aligned}",
            "Decomposes 5-fold Spatial Block CV Hist-XGBoost ($R^2_{\\text{spatial}} = 0.9977$) predictions into local ($\\phi_{i,k}^{\\text{loc}}$) and upwind ($\\phi_{i,k}^{W}$) contributions."
        )
        formula_card(
            "Stage 5 — Geometric Priority & Source-to-Sink Investment Flow",
            r"\begin{aligned} \text{Priority}_i &= \sqrt{\,\text{Need}_i \times \text{Lever}_i\,} \\ \text{Flow}_{i \to j, k} &= \text{Priority}_i \times W_{ij} \times \mathbb{I}\!\left(\text{Class}_j^{*} = k\right) \end{aligned}",
            "Pairs high-emission industrial cores ($i \\in \\{\\text{C3}, \\text{C7}\\}$) with their physical upwind community green sinks ($j$)."
        )

    sec_header("Complete Variable Dictionary & Technical Specifications (Stages 1 to 5B)")
    dict_rows = [
        ("grid_id", "Stage 1", "Unique identifier for each 10 × 10 km terrestrial grid cell across ASEAN (N = 50,621 cells)."),
        ("centroid_lon, centroid_lat", "Stage 1", "Geographic coordinates (WGS84, EPSG:4326) of the grid cell centroid."),
        ("land_area_ha", "Stage 1", "Total non-water terrestrial area within the 10 × 10 km cell in hectares (max 10,000 ha; projected in ESRI:102028)."),
        ("coastal_sliver", "Stage 1", "Binary indicator (1 if land_area_ha < 1,000 ha, i.e., < 10% of a full cell; 0 otherwise) marking coastal/small-island cells."),
        ("co2_odiac", "Stage 1", "Dependent variable (Y): Cumulative fossil fuel CO₂ emissions (tonnes) for H1 2026 derived from ODIAC / calibrated VIIRS-Builtup proxy."),
        ("area_treecover", "Stage 1", "Main green variable (X_kom,1): Tree cover area (hectares, ESA WorldCover Class 10)."),
        ("area_mangrove", "Stage 1", "Main green variable (X_kom,2): Mangrove ecosystem area (hectares, ESA WorldCover Class 95)."),
        ("area_cropland", "Stage 1", "Main green variable (X_kom,3): Cropland mosaic area (hectares, ESA WorldCover Class 40)."),
        ("area_shrubgrass", "Stage 1", "Main green variable (X_kom,4): Shrubland and grassland area (hectares, ESA WorldCover Classes 20 & 30); primary restoration target."),
        ("area_builtup", "Stage 1", "Control variable (C_1): Urban and industrial built-up area (hectares, ESA WorldCover Class 50)."),
        ("ntl_viirs", "Stage 1", "Control variable (C_2): Mean nighttime light radiance from NOAA VIIRS DNB (nW/cm²/sr)."),
        ("hotspot_count, hotspot_frp_sum", "Stage 1", "Control variables (C_3, C_4): Active fire hotspot count (confidence > 50%) and cumulative Fire Radiative Power (MW) from NASA FIRMS."),
        ("u_wind_10m, v_wind_10m", "Stage 1 & 2", "Zonal (eastward) and meridional (northward) 10 m wind velocity components (m/s) from ECMWF ERA5-Land for Annual, DJF, and JJA windows."),
        ("W_ann_250, W_djf_250, W_jja_250", "Stage 2", "Row-standardized anisotropic upwind spatial weights matrices (N × N CSR sparse) with capture radius d_max = 250 km and decay α = 1.5."),
        ("beta_slx (Local β₁)", "Stage 3", "Direct elasticity of local log-transformed land cover ln(X_kom + 1) on local ln(CO₂ + 1) in the Spatial Lag of X (SLX) model."),
        ("beta_slx (Upwind θ₁)", "Stage 3", "Upwind spillover elasticity of wind-weighted land cover W · ln(X_kom + 1) on downwind ln(CO₂ + 1), evaluated with 250 km Conley HAC standard errors."),
        ("slx_spillover_ann", "Stage 3", "Composite upwind green spillover score: (W_ann · ln X_kom) · θ₁."),
        ("lisa_I, lisa_p, lisa_q", "Stage 3", "Local Moran's I statistic, pseudo p-value (199 permutations), and LISA quadrant (1=HH, 2=LH, 3=LL, 4=HL, 0=Insignificant) on slx_spillover_ann."),
        ("shap_loc_*, shap_upw_*, shap_tot_*", "Stage 4", "TreeSHAP contributions (local, upwind, and combined) of each green land-cover class from 5-fold Spatial Block CV Hist-XGBoost."),
        ("sink_strength_co2", "Stage 4", "Total CO₂-reducing SHAP magnitude per cell: -Σ_k min(0, φ_loc,ik + φ_upw,ik)."),
        ("dominant_class_co2", "Stage 4", "Land-cover class delivering the strongest negative (CO₂-reducing) combined SHAP value in cell i: argmin_k (φ_loc,ik + φ_upw,ik)."),
        ("need_rank, lever_rank", "Stage 5", "ASEAN-wide empirical percentile ranks (0 to 1) of co2_odiac (Need) and sink_strength_co2 (Lever)."),
        ("investment_priority", "Stage 5", "Geometric mean index of pollution severity and ecological leverage: √(need_rank × lever_rank)."),
        ("cluster_id, custom_cluster_name", "Stage 5A/5B", "Unsupervised K-Means++ cluster assignment (k* = 8, Bootstrap Jaccard = 0.9767) and post-hoc centroid typology label (C0 to C7)."),
        ("investment_flow_ijk", "Stage 5B", "Directed source-to-sink carbon investment flow from industrial cell i (C3, C7) to upwind community cell j: Priority(i) × W_ij × I(Class_j == k)."),
    ]
    df_dict = pd.DataFrame(dict_rows, columns=["Variable / Artifact Name", "Pipeline Stage", "Operational Definition"])
    st.dataframe(df_dict, hide_index=True, use_container_width=True, height=560)
