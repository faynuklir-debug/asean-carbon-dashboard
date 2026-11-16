
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import plotly.graph_objects as go
import streamlit as st

WORK_DIR = os.environ.get("ASEAN_WORK_DIR", os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Carbon Trading, But for Whom? | 11 ASEAN Countries · XCH4",
    page_icon="🌏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================== palette (dark + pastel accents)
BG, SURF, INK, MUTE, LINE = "#0f1420", "#171d2c", "#e6eaf4", "#8e98b0", "#283148"
NODATA_COLOR = "#252d3f"
NODATA_BORDER = "#38435e"
NODATA_LABEL = "Cloud-masked / No S5P retrieval (n_obs < 3)"
NODATA_DESC = (
    "Equatorial/mountain land cells excluded because Sentinel-5P SWIR had < 3 cloud-free observations "
    "over H1 2026. Left un-imputed to prevent artificial spatial smoothing."
)

ROLE_COLORS = {
    "Anthropogenic & agro methane core": "#f2a3a0",
    "Forest & mangrove conservation belt": "#9fd8b8",
    "Transitional mosaic buffer": "#f6d28f",
    "Background": "#46506a",
}
ROLE_DESC = {
    "Anthropogenic & agro methane core": (
        "High XCH4 (~1,938 ppb), intensive rice paddies/cropland (20.8x), urban built-up (14.4x), and nightlights (2.3x). "
        "Policy action: Direct methane source abatement (clean energy, waste capture, low-methane rice AWD) & offset buyer."
    ),
    "Forest & mangrove conservation belt": (
        "Low XCH4 (~1,905 ppb), dense forest & coastal mangroves (23.3x), minimal cropland, and 4.9x stronger lever association. "
        "Policy action: Upwind conservation protection, blue-carbon mangrove finance, and community benefit-sharing."
    ),
    "Transitional mosaic buffer": (
        "Mixed agro-forestry mosaic between intensive emission corridors and natural forests. "
        "Policy action: Agroforestry buffer restoration and fire prevention."
    ),
    "Background": "Near regional average across indicators. Policy action: Routine satellite monitoring.",
}
ROLE_SHADES = {
    "Anthropogenic & agro methane core": ["#f2a3a0", "#ff7f8e"],
    "Forest & mangrove conservation belt": ["#9fd8b8", "#8ad0d6", "#c3e8a4"],
    "Transitional mosaic buffer": ["#f6c99a", "#e3dc9a", "#f0b5d0"],
    "Background": ["#46506a", "#58637f"],
}

CLASS_LABEL = {
    "area_treecover": "Tree cover",
    "area_mangrove": "Mangrove",
    "area_cropland": "Cropland",
    "area_shrubgrass": "Shrub & grassland",
    "none": "No negative association (SHAP >= 0)",
}
CLASS_COLORS = {
    "Tree cover": "#8fd0a8",
    "Mangrove": "#8ad0d6",
    "Cropland": "#f6c99a",
    "Shrub & grassland": "#e3dc9a",
    "No negative association (SHAP >= 0)": "#5c6684",
}
CLASS_LINE = {k: v for k, v in CLASS_COLORS.items() if k != "No negative association (SHAP >= 0)"}

LISA_LABEL = {0: "Not significant", 1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
LISA_COLORS = {
    "Not significant": "#3a4359",
    "High-High": "#ef9a9a",
    "Low-High": "#f7dcb4",
    "Low-Low": "#8fcfb0",
    "High-Low": "#a9c4ee",
}
LISA_MEANING = {
    "Not significant": "No statistically significant local airshed spillover cluster (p >= 0.05).",
    "High-High": "Transboundary/regional methane hotspot: upwind land cover is associated with higher downwind XCH4.",
    "Low-High": "Low-spillover cell surrounded by high-spillover neighbours (transition edge).",
    "Low-Low": "Clean airshed corridor: upwind forest/mangrove cover is consistently linked to lower downwind XCH4.",
    "High-Low": "Localized high-spillover pocket surrounded by low-spillover neighbours.",
}

SCALES = {
    "priority": [[0, "#26304d"], [0.35, "#5f7fc0"], [0.7, "#c999d8"], [1, "#ffb8aa"]],
    "need": [[0, "#2d2a3d"], [0.4, "#8f5f86"], [0.75, "#f0a584"], [1, "#ffdf9f"]],
    "sink": [[0, "#1f3033"], [0.4, "#3f8a7a"], [0.75, "#8fd5b0"], [1, "#d6f5c2"]],
}

COUNTRY_LABEL = {"Burma": "Myanmar"}
Y_COL = "ghg_ch4_h1"
WANTED = [
    "grid_id", "country_id", "centroid_lon", "centroid_lat", "land_area_ha", "coastal_sliver",
    Y_COL, "ghg_nobs_h1", "ntl_viirs", "area_treecover", "area_mangrove", "area_cropland",
    "area_shrubgrass", "area_builtup", "slx_spillover", "lisa_q", "lever_strength",
    "dominant_class", "cluster_id", "policy_zone", "need_rank", "lever_rank", "investment_priority",
]
NAV = [
    "Policy zones",
    "Investment priority",
    "XCH4 & lever indices",
    "Land-cover class",
    "Investment pairing",
    "Wind-shed",
]

ARCHETYPE_MAP_UI = {
    "HIGH_METHANE_ANTHROPOGENIC_AGRO_CORE": (
        "Anthropogenic & agro-methane core",
        "Anthropogenic & agro methane core",
    ),
    "LOW_METHANE_FOREST_MANGROVE_CONSERVATION_BELT": (
        "Forest & mangrove conservation belt",
        "Forest & mangrove conservation belt",
    ),
    "TRANSITIONAL_MOSAIC_BUFFER_ZONE": (
        "Transitional mosaic buffer",
        "Transitional mosaic buffer",
    ),
    "HIGH_EMISSION_INDUSTRIAL_CORE": (
        "Anthropogenic & agro-methane core",
        "Anthropogenic & agro methane core",
    ),
    "UPWIND_COMMUNITY_GREEN_SINK": (
        "Forest & mangrove conservation belt",
        "Forest & mangrove conservation belt",
    ),
    "TRANSITIONAL_RESTORATION_ZONE": (
        "Transitional mosaic buffer",
        "Transitional mosaic buffer",
    ),
}


def describe_cluster(r):
    arch = str(r.get("post_hoc_archetype", ""))
    if arch in ARCHETYPE_MAP_UI:
        return ARCHETYPE_MAP_UI[arch]
    ind = np.nanmean([r.get("Z_ghg", 0.0), r.get("Z_builtup", 0.0), r.get("Z_ntl", 0.0), r.get("Z_cropland", 0.0)])
    grn = np.nanmean([r.get("Z_lever", 0.0), r.get("Z_treecover", 0.0), r.get("Z_mangrove", 0.0)])
    if ind >= grn and ind > 0.15:
        return "Anthropogenic & agro-methane core", "Anthropogenic & agro methane core"
    if grn > 0.15:
        return "Forest & mangrove conservation belt", "Forest & mangrove conservation belt"
    return "Transitional mosaic buffer", "Transitional mosaic buffer"


def _build_crisp_geojson(gdf_wgs):
    """Keep crisp 10 km square boundaries (tolerance=0.0015, 4-decimal precision) so cells are easy to click when zoomed."""
    simple_geom = shapely.simplify(gdf_wgs.geometry.values, tolerance=0.0015, preserve_topology=True)
    simple_geom = shapely.set_precision(simple_geom, grid_size=0.0001)
    tmp = gpd.GeoDataFrame({"grid_id": gdf_wgs["grid_id"].values}, geometry=simple_geom, crs="EPSG:4326")
    return json.loads(tmp.to_json())


# ============================================================== data loaders
@st.cache_resource(show_spinner="Preparing 11-country ASEAN land mask ...")
def load_asean_land_mask(work_dir):
    cache_path = os.path.join(work_dir, "00_asean_11_land_mask.geojson")
    if os.path.exists(cache_path):
        try:
            land_gdf = gpd.read_file(cache_path)
            return json.loads(land_gdf.to_json()), land_gdf
        except Exception:
            pass

    urls = [
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson",
        "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson",
    ]
    name_map = {
        "Brunei": "Brunei", "Brunei Darussalam": "Brunei", "Cambodia": "Cambodia",
        "Indonesia": "Indonesia", "Laos": "Laos", "Lao PDR": "Laos", "Malaysia": "Malaysia",
        "Myanmar": "Myanmar", "Burma": "Myanmar", "Philippines": "Philippines",
        "Singapore": "Singapore", "Thailand": "Thailand", "Timor-Leste": "Timor-Leste",
        "East Timor": "Timor-Leste", "Vietnam": "Vietnam", "Viet Nam": "Vietnam",
    }
    for u in urls:
        try:
            world = gpd.read_file(u)
            cand_cols = [c for c in ["NAME", "NAME_EN", "ADMIN", "name", "sovereignt"] if c in world.columns]
            matched = []
            for _, row in world.iterrows():
                std_name = None
                for c in cand_cols:
                    val = str(row[c]).strip()
                    if val in name_map:
                        std_name = name_map[val]
                        break
                if std_name:
                    matched.append({"country": std_name, "geometry": row.geometry})
            if matched:
                land = gpd.GeoDataFrame(matched, geometry="geometry", crs="EPSG:4326")
                land = land.dissolve(by="country", as_index=False)
                land["geometry"] = shapely.simplify(land.geometry.values, tolerance=0.010, preserve_topology=True)
                land["geometry"] = shapely.set_precision(land.geometry.values, grid_size=0.001)
                land.to_file(cache_path, driver="GeoJSON")
                return json.loads(land.to_json()), land
        except Exception:
            continue
    return None, None


@st.cache_resource(show_spinner="Loading 11-country ASEAN XCH4 spatial artifacts ...")
def load_data(work_dir):
    p = lambda f: os.path.join(work_dir, f)

    if os.path.exists(p("05_final_grid.pkl")):
        zones = pd.read_pickle(p("05_final_grid.pkl"))
        zones = gpd.GeoDataFrame(zones, geometry="geometry", crs=zones.crs)
    elif os.path.exists(p("05_asean_policy_zones_10km.gpkg")):
        zones = gpd.read_file(p("05_asean_policy_zones_10km.gpkg"))
        for extra in ["03_slx_spillover_10km.gpkg", "04_ml_shap_10km.gpkg", "01_asean_grid_10km.gpkg"]:
            if os.path.exists(p(extra)):
                oth = pd.DataFrame(gpd.read_file(p(extra), ignore_geometry=True))
                new_cols = [c for c in oth.columns if c not in zones.columns and c in WANTED]
                if new_cols:
                    zones = zones.merge(oth[["grid_id"] + new_cols], on="grid_id", how="left")
    else:
        raise FileNotFoundError("Neither 05_final_grid.pkl nor 05_asean_policy_zones_10km.gpkg found.")

    if "centroid_lon" in zones.columns and "centroid_lat" in zones.columns:
        zones["clon"], zones["clat"] = zones["centroid_lon"].values, zones["centroid_lat"].values
    else:
        cen = zones.geometry.centroid.to_crs("EPSG:4326")
        zones["clon"], zones["clat"] = cen.x.values, cen.y.values

    gdf = zones.to_crs("EPSG:4326")
    gdf["country"] = gdf["country_id"].map(lambda c: COUNTRY_LABEL.get(c, c))
    gdf["cluster_id"] = gdf["cluster_id"].astype(int)

    master_gj = _build_crisp_geojson(gdf)
    df_attr = pd.DataFrame(gdf.drop(columns="geometry"))

    prof = None
    if os.path.exists(p("05_zone_commodity_profile.csv")):
        prof = pd.read_csv(p("05_zone_commodity_profile.csv"))
        nr = prof.apply(describe_cluster, axis=1, result_type="expand")
        prof["cname"], prof["role"] = nr[0], nr[1]
        small = prof["pct_total_cells"] < 2.0
        prof["label"] = (
            "C" + prof["cluster_id"].astype(str) + " · " + prof["cname"] + np.where(small, " (small)", "")
        )
        mcol = "modal_dominant_class" if "modal_dominant_class" in prof.columns else "modal_dominant_class_co2"
        prof["modal_class"] = prof[mcol].map(CLASS_LABEL).fillna(prof[mcol])
        idx = prof.set_index("cluster_id")
        df_attr["cluster_label"] = df_attr["cluster_id"].map(idx["label"])
        df_attr["role"] = df_attr["cluster_id"].map(idx["role"])
    else:
        fb = df_attr["policy_zone"].map(lambda a: ARCHETYPE_MAP_UI.get(str(a), ("Zone", "Background")))
        df_attr["cluster_label"] = "C" + df_attr["cluster_id"].astype(str) + " · " + fb.map(lambda t: t[0])
        df_attr["role"] = fb.map(lambda t: t[1])

    dom_col = "dominant_class" if "dominant_class" in df_attr.columns else "dominant_class_co2"
    df_attr["class_label"] = df_attr[dom_col].map(CLASS_LABEL).fillna("No negative association (SHAP >= 0)")
    df_attr["lisa_label"] = df_attr["lisa_q"].fillna(0).astype(int).map(LISA_LABEL)
    ch4_str = df_attr[Y_COL].round(1).astype(str) + " ppb" if Y_COL in df_attr.columns else "n/a"

    df_attr["hover"] = (
        "<b>" + df_attr["grid_id"].astype(str) + " · " + df_attr["country"].astype(str) + "</b>"
        + "<br>Zone: " + df_attr["cluster_label"].astype(str)
        + "<br>Methane (XCH4): <b>" + ch4_str + "</b> (Need pct: "
        + (df_attr["need_rank"] * 100).round(0).astype(int).astype(str) + "%)"
        + "<br>Investment Priority: <b>" + df_attr["investment_priority"].round(2).astype(str)
        + "</b> (Lever pct: " + (df_attr["lever_rank"] * 100).round(0).astype(int).astype(str) + "%)"
        + "<br>Primary Land Lever: <b>" + df_attr["class_label"].astype(str) + "</b>"
        + "<br>Tree: " + df_attr["area_treecover"].round(0).astype(int).astype(str)
        + " ha · Mangrove: " + df_attr["area_mangrove"].round(0).astype(int).astype(str)
        + " ha · Cropland: " + df_attr["area_cropland"].round(0).astype(int).astype(str) + " ha"
        + "<br><i>(Click cell on map to inspect full policy brief in right panel)</i>"
    )
    return df_attr, prof, master_gj


@st.cache_resource(show_spinner=False)
def load_corridors(work_dir, top_n_cap=3000):
    for fname in ["02_airshed_corridors_10km.gpkg", "02_airshed_corridors_10km.shp"]:
        path = os.path.join(work_dir, fname)
        if os.path.exists(path):
            c = gpd.read_file(path).to_crs("EPSG:4326")
            c = c.sort_values("weight_wij", ascending=False).head(top_n_cap).reset_index(drop=True)
            coords = [np.round(np.asarray(g.coords), 3) for g in c.geometry]
            c_df = pd.DataFrame(c.drop(columns="geometry"))
            c_df["coords"] = coords
            return c_df
    return None


@st.cache_resource(show_spinner=False)
def load_links(work_dir):
    path = os.path.join(work_dir, "05_industrial_community_investment_links.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    if "community_grid_id" not in d.columns and "upwind_community_grid_id" in d.columns:
        d = d.rename(columns={
            "upwind_community_grid_id": "community_grid_id",
            "upwind_country": "community_country",
            "dominant_class_co2_j": "dominant_class_j",
        })
    d["ind_country"] = d["industrial_country"].map(lambda c: COUNTRY_LABEL.get(c, c))
    d["com_country"] = d["community_country"].map(lambda c: COUNTRY_LABEL.get(c, c))
    d["class_label"] = d["dominant_class_j"].map(CLASS_LABEL).fillna("No negative association (SHAP >= 0)")
    return d


@st.cache_resource(show_spinner=False)
def load_evidence(work_dir):
    imp, slx, rob = None, None, None
    f1 = os.path.join(work_dir, "04_ml_commodity_importance.csv")
    f2 = os.path.join(work_dir, "03_slx_coefficients.csv")
    f3 = os.path.join(work_dir, "03_robustness.csv")
    if os.path.exists(f1):
        imp = pd.read_csv(f1)
        imp["label"] = imp["commodity_class"].map(CLASS_LABEL)
    if os.path.exists(f2):
        slx = pd.read_csv(f2)
        slx = slx[slx["role"] == "MAIN_POLICY_COMMODITY"].copy()
        slx["kind"] = np.where(slx["variable"].str.startswith("W_"), "Upwind (θ)", "Local (β)")
        slx["cls"] = "area_" + slx["variable"].str.extract(r"area_([a-z]+)")[0]
        slx["label"] = slx["cls"].map(CLASS_LABEL)
    if os.path.exists(f3):
        rob = pd.read_csv(f3)
    return imp, slx, rob


# ============================================================== styling
st.markdown(f"""<style>
.block-container{{padding-top:1.2rem;padding-bottom:2.5rem;max-width:1600px}}
header[data-testid="stHeader"]{{background:transparent}}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"]{{display:none}}
.hero{{padding:16px 22px;border-radius:16px;margin-bottom:14px;color:{INK};border:1px solid {LINE};
  background:radial-gradient(900px 220px at 100% 0%,rgba(143,208,168,.16),transparent),
  linear-gradient(120deg,#182235 0%,#1a2236 60%,#231f3a 120%)}}
.hero .tag{{font-size:.68rem;letter-spacing:.16em;text-transform:uppercase;color:{MUTE}}}
.hero .ttl{{font-size:1.45rem;font-weight:700;line-height:1.25;margin:3px 0 4px 0}}
.hero .sub{{font-size:.88rem;color:{MUTE};max-width:1100px;line-height:1.45}}
.sec{{font-size:.7rem;letter-spacing:.12em;text-transform:uppercase;color:{MUTE};margin:2px 0 8px 0}}
.card-ttl{{font-size:.9rem;font-weight:650;color:{INK};letter-spacing:.01em;margin:0 0 6px 0}}
.policy-box{{border-left:3px solid #8fd0a8;background:#131a29;padding:8px 11px;border-radius:6px;margin:6px 0;font-size:.8rem;color:#d5def0;line-height:1.48}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:2px 0 14px 0}}
.kpi{{border:1px solid {LINE};background:{SURF};border-radius:12px;padding:10px 14px}}
.kpi .cv{{font-size:1.25rem;font-weight:700;color:{INK};line-height:1.2;overflow-wrap:anywhere}}
.kpi .cl{{font-size:.66rem;color:{MUTE};text-transform:uppercase;letter-spacing:.09em;margin-top:2px}}
.lrow{{display:flex;gap:10px;margin:0 0 12px 0}}
.dot{{width:12px;height:12px;border-radius:4px;margin-top:4px;flex:0 0 12px}}
.ltxt{{flex:1;min-width:0}}.lt{{display:flex;justify-content:space-between;gap:8px;font-size:.83rem;color:{INK}}}
.lt b{{font-weight:600;overflow-wrap:anywhere}}
.lp{{color:{MUTE};font-variant-numeric:tabular-nums;white-space:nowrap}}
.ls{{font-size:.73rem;color:{MUTE};line-height:1.4;margin:2px 0 5px 0}}
.btrack{{height:4px;border-radius:3px;background:#222b40;overflow:hidden}}
.bfill{{height:100%;border-radius:3px}}
.mini{{font-size:.79rem;display:flex;justify-content:space-between;align-items:baseline;gap:10px;padding:6px 0;
  border-bottom:1px solid {LINE};color:#c4cbdc}}
.mini span{{min-width:0;overflow-wrap:anywhere}}
.mini small{{color:{MUTE};font-size:.72rem}}
.mini:last-child{{border-bottom:none}}
.mini b{{color:{INK};white-space:nowrap;font-variant-numeric:tabular-nums}}
div[data-testid="stPlotlyChart"]{{border:1px solid {LINE};border-radius:14px;overflow:hidden;background:{SURF}}}
div[data-testid="stVerticalBlockBorderWrapper"]{{border-color:{LINE}!important;border-radius:14px!important;background:{SURF};padding:12px 15px!important;margin-bottom:10px!important}}
div[data-testid="stVerticalBlockBorderWrapper"] p{{font-size:.82rem!important;line-height:1.55!important;color:#c4cbdc!important;margin-bottom:6px!important}}
div[data-testid="stVerticalBlockBorderWrapper"] li{{font-size:.81rem!important;line-height:1.5!important;color:#c4cbdc!important}}
div[data-testid="stLatex"]{{background:#0f1420;border:1px solid {LINE};border-radius:8px;padding:6px 8px;margin:6px 0!important;overflow-x:auto}}
div[data-testid="stLatex"] .katex{{font-size:.85rem!important;color:#bcd0f5!important}}
.katex{{font-size:.93em!important;color:#dce6ff!important}}
h5{{font-size:.95rem!important;color:{INK};margin:16px 0 8px 0!important}}
</style>""", unsafe_allow_html=True)


def html(s):
    st.markdown(s.replace("\n", ""), unsafe_allow_html=True)


def card(title, body=None, latex_eq=None, footer_md=None, raw_html=None):
    with st.container(border=True):
        st.markdown(f"<div class='card-ttl'>{title}</div>", unsafe_allow_html=True)
        if body:
            st.markdown(body)
        if latex_eq:
            st.latex(latex_eq)
        if footer_md:
            st.markdown(footer_md)
        if raw_html:
            html(raw_html)


def legend_rows(rows):
    out = ""
    for label, color, pct, desc in rows:
        pct_str = f"{pct:.1f}%" if isinstance(pct, (int, float)) else str(pct)
        bar_w = min(float(pct), 100.0) if isinstance(pct, (int, float)) else 100.0
        out += (
            f"<div class='lrow'><span class='dot' style='background:{color};border:1px solid {NODATA_BORDER}'></span><div class='ltxt'>"
            f"<div class='lt'><b>{label}</b><span class='lp'>{pct_str}</span></div>"
            f"<div class='ls'>{desc}</div><div class='btrack'><div class='bfill' "
            f"style='width:{bar_w:.1f}%;background:{color}'></div></div></div></div>"
        )
    return out


def nodata_legend_card(extra_note=None):
    row_html = legend_rows([(NODATA_LABEL, NODATA_COLOR, "Masked", NODATA_DESC)])
    card("Data coverage & cloud mask", raw_html=row_html, footer_md=extra_note)


def country_index_summary_card(df_sub, col_name, card_title, is_priority=False, q90_val=None):
    """
    Present index values across countries WITHOUT ranking (alphabetical A-Z)
    and explicitly report the number of valid pixels/cells (n) used in each country.
    """
    grp = (
        df_sub.groupby("country", as_index=False)
        .agg(
            n_pixels=(col_name, "size"),
            mean_idx=(col_name, "mean"),
            mean_ch4=(Y_COL, "mean"),
        )
        .sort_values("country", ascending=True)  # Alphabetical A-Z, NOT ranked
    )
    if is_priority and q90_val is not None:
        top10 = (
            df_sub.assign(_hi=df_sub[col_name] >= q90_val)
            .groupby("country")["_hi"]
            .mean()
            .to_dict()
        )
        rows = [
            (
                f"<b>{r.country}</b> <small>(n = {int(r.n_pixels):,} pixels)</small>",
                f"Mean: {r.mean_idx:.2f} · Top 10%: {top10.get(r.country, 0.0)*100:.0f}%",
            )
            for r in grp.itertuples()
        ]
    else:
        rows = [
            (
                f"<b>{r.country}</b> <small>(n = {int(r.n_pixels):,} pixels)</small>",
                f"Index: {r.mean_idx:.2f} <small>({r.mean_ch4:.0f} ppb)</small>",
            )
            for r in grp.itertuples()
        ]
    note = (
        "Presented alphabetically (not ranked). Averages reflect **only valid cloud-free pixels ($n$)** "
        "within each country (`n_obs >= 3`), not total national land area."
    )
    card(card_title, raw_html=mini_list(rows), footer_md=note)


def render_cell_inspector(df_all, selected_id=None):
    """Interactive Decision-Maker Cell Inspector: triggered by clicking any grid cell on the map or selecting via dropdown."""
    with st.container(border=True):
        html(f"<div class='card-ttl'>🔍 Interactive 10 km Grid Inspector</div>")
        st.caption("Click any grid cell directly on the map (or search below) to view its local profile & policy recommendation.")
        options = ["(Click a cell on the map or pick ID)"] + df_all["grid_id"].tolist()
        idx_default = options.index(selected_id) if (selected_id in options) else 0
        chosen = st.selectbox("Inspect Grid ID", options, index=idx_default, label_visibility="collapsed")
        if chosen != options[0]:
            r = df_all[df_all["grid_id"] == chosen].iloc[0]
            is_c0 = int(r["cluster_id"]) == 0
            action_rec = (
                "<b>Recommended Policy Action (Cluster 0 — Receptor/Source):</b> Prioritize direct methane source mitigation "
                "(alternate wetting & drying in rice paddies, landfill/wastewater capture, industrial leak detection) and "
                "mobilize carbon finance toward upwind Cluster 1 conservation cells."
                if is_c0 else
                "<b>Recommended Policy Action (Cluster 1 — Upwind Conservation Belt):</b> Eligible recipient for community-based "
                "forest & coastal mangrove protection finance; prevent conversion of natural cover into irrigated cropland."
            )
            details = mini_list([
                ("Grid ID & Country", f"{r['grid_id']} · {r['country']}"),
                ("Coordinates (Lon, Lat)", f"{r['clon']:.3f}°, {r['clat']:.3f}°"),
                ("Policy Regime", f"{r['cluster_label']}"),
                ("Column Methane (XCH4)", f"{r[Y_COL]:.1f} ppb (Need pct: {r['need_rank']*100:.0f}%)"),
                ("Investment Priority", f"{r['investment_priority']:.2f} (Lever pct: {r['lever_rank']*100:.0f}%)"),
                ("Primary Land Lever", f"{r['class_label']}"),
                ("Tree / Mangrove Cover", f"{r['area_treecover']:,.0f} ha / {r['area_mangrove']:,.0f} ha"),
                ("Cropland / Built-up", f"{r['area_cropland']:,.0f} ha / {r['area_builtup']:,.0f} ha"),
                ("LISA Spillover Quadrant", f"{r['lisa_label']}"),
            ])
            html(details + f"<div class='policy-box'>{action_rec}</div>")


def shares(df, col, order, colors, desc=None, weight=None):
    if weight is None:
        vc = df[col].value_counts(normalize=True) * 100
    else:
        s = df.groupby(col)[weight].sum()
        vc = s / s.sum() * 100 if s.sum() > 0 else s * 0.0
    return [
        (c, colors.get(c, "#cbd5e1"), float(vc.get(c, 0.0)), (desc or {}).get(c, ""))
        for c in order if c in vc.index
    ]


def chips(items):
    html(
        "<div class='kpis'>"
        + "".join(f"<div class='kpi'><div class='cv'>{v}</div><div class='cl'>{l}</div></div>" for v, l in items)
        + "</div>"
    )


def mini_list(rows):
    return "".join(f"<div class='mini'><span>{a}</span><b>{b}</b></div>" for a, b in rows)


# ============================================================== map helpers (Basemap default = Dark map CARTO)
BASEMAPS = {
    "Dark map": "carto-darkmatter",
    "Plain dark": {
        "version": 8,
        "sources": {},
        "layers": [{"id": "bg", "type": "background", "paint": {"background-color": "#141a29"}}],
    },
}
HOVER = dict(align="left", bgcolor="#1f2638", font=dict(color=INK, size=12), bordercolor="#38425c")


def dark(fig, **kw):
    fig.update_layout(paper_bgcolor=SURF, plot_bgcolor=SURF, font=dict(color=INK, size=11), **kw)
    fig.update_xaxes(gridcolor="#242d44", zerolinecolor="#38425c", linecolor="#38425c")
    fig.update_yaxes(gridcolor="#242d44", zerolinecolor="#38425c", linecolor="#38425c")
    return fig


def view_for(df, focus_country=None):
    if focus_country and focus_country != "All selected countries":
        sub = df[df["country"] == focus_country]
        if not sub.empty:
            df = sub
    minx, maxx = float(df["clon"].min()) - 0.3, float(df["clon"].max()) + 0.3
    miny, maxy = float(df["clat"].min()) - 0.3, float(df["clat"].max()) + 0.3
    span = max(maxx - minx, (maxy - miny) * 1.1, 0.4)
    zoom = float(np.clip(np.log2(360.0 / span) - 0.75, 2.2, 9.5))
    return {"lat": (miny + maxy) / 2, "lon": (minx + maxx) / 2}, zoom


def _nodata_land_trace(land_gj, picked_countries):
    """Background trace for cloud-masked land. hoverinfo='skip' ensures it NEVER blocks clicking/hovering 10 km cells when zoomed."""
    if land_gj is None or not picked_countries:
        return None
    return go.Choroplethmap(
        geojson=land_gj,
        featureidkey="properties.country",
        locations=list(picked_countries),
        z=np.zeros(len(picked_countries)),
        zmin=0,
        zmax=1,
        colorscale=[[0, NODATA_COLOR], [1, NODATA_COLOR]],
        showscale=False,
        marker_line_width=0.7,
        marker_line_color=NODATA_BORDER,
        marker_opacity=0.88,
        hoverinfo="skip",
    )


def _cell_click_target_trace(df):
    """Invisible high-priority centroid target trace so clicking/hovering any 10 km cell when zoomed is effortless and exact."""
    return go.Scattermap(
        lon=df["clon"],
        lat=df["clat"],
        mode="markers",
        customdata=df["grid_id"],
        text=df["hover"],
        hovertemplate="%{text}<extra></extra>",
        marker=dict(size=11, color="#ffffff", opacity=0.01),
        showlegend=False,
    )


def finish(fig, df, basemap, focus_country=None, height=650, bottom=0):
    center, zoom = view_for(df, focus_country=focus_country)
    fig.update_layout(
        map_style=BASEMAPS[basemap],
        map_center=center,
        map_zoom=zoom,
        height=height,
        showlegend=False,
        clickmode="event+select",
        margin=dict(l=0, r=0, t=0, b=bottom),
        paper_bgcolor=SURF,
        hoverlabel=HOVER,
    )
    return fig


def categorical_map(df, gj, col, colors, basemap, order=None, land_gj=None, picked_countries=None, focus_country=None):
    cats = [c for c in (order or sorted(df[col].dropna().unique())) if (df[col] == c).any()]
    cols = [colors.get(c, "#cbd5e1") for c in cats]
    n = max(len(cats), 1)
    z = df[col].map({c: i + 0.5 for i, c in enumerate(cats)})
    scale = []
    for i, c in enumerate(cols):
        scale += [[i / n, c], [(i + 1) / n, c]]
    traces = []
    bg = _nodata_land_trace(land_gj, picked_countries)
    if bg is not None:
        traces.append(bg)
    traces.append(
        go.Choroplethmap(
            geojson=gj,
            featureidkey="properties.grid_id",
            locations=df["grid_id"],
            customdata=df["grid_id"],
            z=z,
            zmin=0,
            zmax=n,
            colorscale=scale,
            showscale=False,
            marker_line_width=0.45,
            marker_line_color="#101624",
            marker_opacity=0.92,
            text=df["hover"],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    traces.append(_cell_click_target_trace(df))
    return finish(go.Figure(traces), df, basemap, focus_country=focus_country)


def continuous_map(df, gj, col, scale, title, basemap, land_gj=None, picked_countries=None, focus_country=None):
    traces = []
    bg = _nodata_land_trace(land_gj, picked_countries)
    if bg is not None:
        traces.append(bg)
    traces.append(
        go.Choroplethmap(
            geojson=gj,
            featureidkey="properties.grid_id",
            locations=df["grid_id"],
            customdata=df["grid_id"],
            z=df[col],
            colorscale=scale,
            zmin=0,
            zmax=1,
            marker_line_width=0.45,
            marker_line_color="#101624",
            marker_opacity=0.93,
            text=df["hover"],
            hovertemplate="%{text}<extra></extra>",
            colorbar=dict(
                orientation="h",
                thickness=9,
                len=0.4,
                x=0.5,
                xanchor="center",
                y=-0.015,
                yanchor="top",
                outlinewidth=0,
                title=dict(text=title, side="right", font=dict(color=MUTE, size=11)),
                tickfont=dict(color=MUTE, size=10),
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
            ),
        )
    )
    traces.append(_cell_click_target_trace(df))
    return finish(go.Figure(traces), df, basemap, focus_country=focus_country, height=670, bottom=56)


def show_interactive_map(fig, key):
    """Render Plotly map with click/box selection enabled and return clicked grid_id if any."""
    event = st.plotly_chart(
        fig,
        width="stretch",
        key=key,
        on_select="rerun",
        selection_mode=("points",),
        config={"displayModeBar": True, "scrollZoom": True},
    )
    try:
        pts = event.selection.points if (event and hasattr(event, "selection") and event.selection) else []
        if pts:
            p0 = pts[0]
            return p0.get("customdata") or p0.get("location")
    except Exception:
        pass
    return None


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False, "scrollZoom": True})


def flow_lines(base_df, gj, basemap, segments, dots, land_gj=None, picked_countries=None, focus_country=None):
    traces = []
    bg = _nodata_land_trace(land_gj, picked_countries)
    if bg is not None:
        traces.append(bg)
    traces.append(
        go.Choroplethmap(
            geojson=gj,
            featureidkey="properties.grid_id",
            locations=base_df["grid_id"],
            customdata=base_df["grid_id"],
            z=np.zeros(len(base_df)),
            colorscale=[[0, "#313b54"], [1, "#313b54"]],
            showscale=False,
            marker_opacity=0.75,
            marker_line_width=0.35,
            marker_line_color="#182032",
            text=base_df["hover"],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    for color, (lons, lats) in segments:
        traces.append(
            go.Scattermap(
                lon=lons, lat=lats, mode="lines", line=dict(width=1.9, color=color), opacity=0.88, hoverinfo="skip"
            )
        )
    for lons, lats, color, size in dots:
        traces.append(
            go.Scattermap(lon=lons, lat=lats, mode="markers", marker=dict(size=size, color=color), hoverinfo="skip")
        )
    return finish(go.Figure(traces), base_df, basemap, focus_country=focus_country)


# ============================================================== load + hero + toolbar
html(
    "<div class='hero'><div class='tag'>11 ASEAN Countries · Sentinel-5P XCH4 (ppb) · Decision-Support Dashboard</div>"
    "<div class='ttl'>Carbon Trading, But for Whom?</div>"
    "<div class='sub'>A spatial decision-support tool for ASEAN ministries, carbon market regulators, and green investors: "
    "identifying where atmospheric methane (XCH4) pressures concentrate, which upwind forest & mangrove ecosystems buffer "
    "regional airsheds, and how to pair high-emission corridors with community conservation sites.</div></div>"
)

try:
    gdf, prof, master_gj = load_data(WORK_DIR)
    land_gj, _ = load_asean_land_mask(WORK_DIR)
except Exception as e:
    st.error(f"Could not load the pipeline artifacts from {WORK_DIR}.\n\n{e}")
    st.stop()

tb_nav, tb_zoom, tb_filter = st.columns([5.0, 1.4, 0.9], gap="small", vertical_alignment="center")
with tb_filter:
    with st.popover("⚙️ Filters"):
        countries = sorted(gdf["country"].unique())
        picked = st.multiselect("Countries (11 ASEAN)", countries, default=countries)
        show_nodata = st.checkbox("Show cloud-masked land (#252d3f)", value=True)
        drop_sliver = st.checkbox("Exclude coastal slivers (< 1,000 ha)", value=False)
        basemap = st.radio("Basemap", list(BASEMAPS), index=0)
with tb_zoom:
    zoom_country = st.selectbox(
        "Quick Zoom",
        ["All selected countries"] + sorted(picked),
        index=0,
        label_visibility="collapsed",
        help="Quickly zoom the map to a specific ASEAN country to inspect & click 10 km grid cells easily.",
    )
with tb_nav:
    if hasattr(st, "segmented_control"):
        view = st.segmented_control("Map", NAV, default=NAV[0], label_visibility="collapsed") or NAV[0]
    else:
        view = st.radio("Map", NAV, horizontal=True, label_visibility="collapsed")

df = gdf[gdf["country"].isin(picked)]
if drop_sliver and "coastal_sliver" in df.columns:
    df = df[df["coastal_sliver"] == 0]
if df.empty:
    st.warning("No cells match the current filters.")
    st.stop()

gj = master_gj
active_land_gj = land_gj if show_nodata else None

if prof is not None:
    cluster_order = prof.sort_values("cluster_id")["label"].tolist()
    used, cluster_colors = {}, {}
    for r in prof.sort_values("cluster_id").itertuples():
        k = used.get(r.role, 0)
        shades = ROLE_SHADES.get(r.role, ROLE_SHADES["Background"])
        cluster_colors[r.label] = shades[k % len(shades)]
        used[r.role] = k + 1
else:
    cluster_order = sorted(df["cluster_label"].unique())
    cluster_colors = {c: ROLE_SHADES["Background"][0] for c in cluster_order}

kpi = st.container()
left, right = st.columns([3.35, 1.3], gap="medium")
with right:
    ctrl = st.container(border=True)
    with ctrl:
        html("<div class='sec'>Controls</div>")

# ============================================================== views
if view == "Policy zones":
    with ctrl:
        by = st.radio("Color by", ["Zone role", "Cluster"], horizontal=True)
    core_share = (df["role"] == "Anthropogenic & agro methane core").mean() * 100
    mean_ch4 = f"{df[Y_COL].mean():.1f} ppb" if Y_COL in df.columns else "n/a"
    with kpi:
        chips([
            (f"{len(df):,}", "valid 10 km pixels"),
            (str(df["country"].nunique()), "ASEAN countries"),
            (mean_ch4, "mean XCH4 (H1 2026)"),
            (f"{core_share:.1f}%", "anthropogenic & agro core (C0)"),
        ])
    with left:
        if by == "Zone role":
            fig_map = categorical_map(
                df, gj, "role", ROLE_COLORS, basemap, order=list(ROLE_COLORS),
                land_gj=active_land_gj, picked_countries=picked, focus_country=zoom_country,
            )
        else:
            fig_map = categorical_map(
                df, gj, "cluster_label", cluster_colors, basemap, order=cluster_order,
                land_gj=active_land_gj, picked_countries=picked, focus_country=zoom_country,
            )
        clicked_id = show_interactive_map(fig_map, key="map_policy_zones")

        if prof is not None:
            st.markdown("##### Executive Summary of ASEAN Spatial Regimes ($k^* = 2$)")
            lev_col = "total_lever_strength" if "total_lever_strength" in prof.columns else "total_sink_strength_co2"
            cols_show = [
                "label", "role", "cell_count", "pct_total_cells", "raw_mean_ghg_ppb",
                "modal_class", "pct_cells_modal_class", lev_col, "bootstrap_jaccard",
            ]
            cols_avail = [c for c in cols_show if c in prof.columns]
            t = prof[cols_avail].copy()
            rename_map = {
                "label": "Cluster",
                "role": "Policy Regime Role",
                "cell_count": "Valid Pixels",
                "pct_total_cells": "% Share",
                "raw_mean_ghg_ppb": "Mean XCH4 (ppb)",
                "modal_class": "Primary Land Lever",
                "pct_cells_modal_class": "% Pixels Supporting",
                lev_col: "Total Lever Score",
                "bootstrap_jaccard": "Stability (Jaccard)",
            }
            t.columns = [rename_map.get(c, c) for c in cols_avail]
            st.dataframe(
                t.round(3),
                hide_index=True,
                width="stretch",
                column_config={
                    "% Share": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
                    "% Pixels Supporting": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                },
            )
            zc = [c for c in prof.columns if c.startswith("Z_")]
            hm = go.Figure(
                go.Heatmap(
                    z=prof[zc].clip(-3, 3).values,
                    x=[c[2:].replace("area_", "").replace("_", " ") for c in zc],
                    y=prof["label"],
                    zmin=-3,
                    zmax=3,
                    xgap=2,
                    ygap=2,
                    colorscale=[[0, "#7f9fd6"], [0.5, "#1c2334"], [1, "#e88f8c"]],
                    text=prof[zc].round(2).values,
                    texttemplate="%{text}",
                    textfont=dict(size=10, color=INK),
                    colorbar=dict(thickness=8, len=0.8, tickfont=dict(size=10, color=MUTE), outlinewidth=0),
                )
            )
            dark(
                hm,
                height=80 + 38 * len(prof),
                margin=dict(l=0, r=10, t=8, b=0),
                yaxis=dict(autorange="reversed"),
                xaxis=dict(side="top"),
            )
            st.markdown("##### Standardized Regime Fingerprint (Z-scores relative to ASEAN mean)")
            show(hm)
    with right:
        render_cell_inspector(df, clicked_id)
        card(
            "📌 Decision-Maker Guide: Policy Zones",
            body=(
                "**How to read this map:** Observed ASEAN land pixels divide into two stable regimes ($\\text{Jaccard} > 0.98$):\n"
                "- **Pink cells (Cluster 0 — 73.8%):** *Anthropogenic & Agro-Methane Core*. High $\\text{XCH}_4$ ($1,938\\text{ ppb}$), "
                "irrigated rice paddies ($20.8\\times$ higher), and urban/industrial activity ($14.4\\times$ higher). **Action:** Target for "
                "on-site methane reduction and **source of conservation funding**.\n"
                "- **Green cells (Cluster 1 — 26.2%):** *Forest & Mangrove Conservation Belt*. Low $\\text{XCH}_4$ ($1,905\\text{ ppb}$) "
                "and $23.3\\times$ more coastal mangroves. **Action:** Target for **receiving green conservation finance**."
            ),
            latex_eq="Z_{i,m} = \\frac{x_{i,m} - \\mu_m}{\\sigma_m}, \\quad k^* = 2 \\;\\; (\\text{Jaccard}_{\\text{OOB}} > 0.98)",
        )
        if by == "Zone role":
            card("Regime roles (valid pixels)", raw_html=legend_rows(shares(df, "role", list(ROLE_COLORS), ROLE_COLORS, ROLE_DESC)))
        else:
            desc = df.groupby("cluster_label")["role"].agg(lambda s: s.mode().iloc[0]).to_dict()
            card("Clusters (valid pixels)", raw_html=legend_rows(shares(df, "cluster_label", cluster_order, cluster_colors, desc)))
        nodata_legend_card()

elif view == "Investment priority":
    with ctrl:
        thr = st.slider("Minimum priority score", 0.0, 0.95, 0.0, 0.05)
    sub = df[df["investment_priority"] >= thr]
    if sub.empty:
        with ctrl:
            st.warning("No cells above this threshold.")
        sub = df
    with kpi:
        chips([
            (f"{len(sub):,}", "valid pixels shown"),
            (f"{sub['investment_priority'].mean():.2f}", "mean priority"),
            (f"{sub['investment_priority'].max():.2f}", "max priority"),
            (str(sub["country"].nunique()), "countries"),
        ])
    with left:
        fig_map = continuous_map(
            sub, gj, "investment_priority", SCALES["priority"], "Priority", basemap,
            land_gj=active_land_gj, picked_countries=picked, focus_country=zoom_country,
        )
        clicked_id = show_interactive_map(fig_map, key="map_priority")
    with right:
        render_cell_inspector(sub, clicked_id)
        card(
            "📌 Decision-Maker Guide: Investment Priority",
            body=(
                "**What this score means for budgeting:**\n"
                "- **High Priority ($> 0.75$, bright pink/peach):** Locations where high atmospheric methane pressure ($\\text{Need}_i$) "
                "overlaps with strong protective vegetation associations ($\\text{Lever}_i$). Investing in forest/mangrove protection or "
                "agro-methane transition here yields the highest co-benefit.\n"
                "- **Low Priority ($< 0.30$, dark blue):** Either methane concentration is already low or local land-cover changes show minimal statistical leverage."
            ),
            latex_eq="\\text{Priority}_i = \\sqrt{\\text{Need}_i \\times \\text{Lever}_i} \\in [0, 1]",
            footer_md="Use the **Minimum priority score** slider above to isolate hotspot pixels for project screening.",
        )
        q90 = gdf["investment_priority"].quantile(0.9)
        country_index_summary_card(
            sub,
            col_name="investment_priority",
            card_title="Priority index & valid pixels by country (A–Z, unranked)",
            is_priority=True,
            q90_val=q90,
        )
        nodata_legend_card()

elif view == "XCH4 & lever indices":
    with ctrl:
        which = st.radio("Index", ["Column XCH4 (need)", "Lever strength (SHAP)"], horizontal=True)
    is_need = which.startswith("Column")
    col = "need_rank" if is_need else "lever_rank"
    with kpi:
        chips([
            (f"{len(df):,}", "valid pixels"),
            (f"{df[col].mean():.2f}", "mean index"),
            (f"{(df[col] >= 0.9).mean() * 100:.1f}%", "pixels in top 10%"),
            (str(df["country"].nunique()), "countries"),
        ])
    with left:
        if is_need:
            fig_map = continuous_map(
                df, gj, "need_rank", SCALES["need"], "XCH4 need rank", basemap,
                land_gj=active_land_gj, picked_countries=picked, focus_country=zoom_country,
            )
        else:
            fig_map = continuous_map(
                df, gj, "lever_rank", SCALES["sink"], "Lever rank", basemap,
                land_gj=active_land_gj, picked_countries=picked, focus_country=zoom_country,
            )
        clicked_id = show_interactive_map(fig_map, key="map_indices")
    with right:
        render_cell_inspector(df, clicked_id)
        if is_need:
            card(
                "📌 Decision-Maker Guide: Methane Need Index",
                body=(
                    "**How to interpret the Need Index:**\n"
                    "- Measures the empirical percentile ($0.0$ to $1.0$) of Sentinel-5P column $\\text{XCH}_4$ across observed ASEAN pixels.\n"
                    "- **Bright yellow/orange pixels ($> 0.80$):** Persistent atmospheric methane accumulation driven by wetland agriculture (irrigated rice), urban/fossil activity, or regional wind transport."
                ),
                latex_eq="\\text{Need}_i = F_{\\text{emp}}\\!\\left(\\text{XCH}_{4,i}\\right) \\in [0, 1]",
            )
            country_index_summary_card(
                df,
                col_name="need_rank",
                card_title="Need index & valid pixels by country (A–Z, unranked)",
            )
        else:
            card(
                "📌 Decision-Maker Guide: Land-Cover Lever Index",
                body=(
                    "**How to interpret the Lever Index:**\n"
                    "- Measures the empirical percentile ($0.0$ to $1.0$) of how strongly local and upwind ecosystems are associated with **lower $\\text{XCH}_4$** in out-of-fold XGBoost SHAP models.\n"
                    "- **Bright green pixels ($> 0.80$):** Conservation buffers where maintaining natural cover (or avoiding flooded cropland expansion) keeps regional methane lower."
                ),
                latex_eq="\\text{Lever}_i = -\\sum_{k=1}^{4} \\min\\!\\left(0,\\; \\phi_{i,k}^{\\text{local}} + \\phi_{i,k}^{\\text{upwind}}\\right)",
            )
            country_index_summary_card(
                df,
                col_name="lever_rank",
                card_title="Lever index & valid pixels by country (A–Z, unranked)",
            )
        nodata_legend_card()

elif view == "Land-cover class":
    imp, slx, _ = load_evidence(WORK_DIR)
    with ctrl:
        per = st.radio("SLX period (chart below map)", ["h1", "djf", "jja"], horizontal=True)
    with kpi:
        chips([
            (f"{len(df):,}", "valid pixels"),
            (df["class_label"].mode().iloc[0], "modal dominant class"),
            (str(df["country"].nunique()), "countries"),
            (f"{(df['class_label'] == 'No negative association (SHAP >= 0)').mean() * 100:.1f}%", "SHAP >= 0 pixels"),
        ])
    with left:
        fig_map = categorical_map(
            df, gj, "class_label", CLASS_COLORS, basemap, order=list(CLASS_COLORS),
            land_gj=active_land_gj, picked_countries=picked, focus_country=zoom_country,
        )
        clicked_id = show_interactive_map(fig_map, key="map_landcover")
        if imp is not None or slx is not None:
            st.markdown("##### Econometric (SLX) & Machine Learning (OOF SHAP) Evidence")
            e1, e2 = st.columns(2, gap="medium")
            if imp is not None:
                with e1:
                    f = go.Figure()
                    f.add_bar(
                        y=imp["label"], x=imp["mean_abs_shap_local"], name="Local", orientation="h", marker_color="#8fd0a8"
                    )
                    f.add_bar(
                        y=imp["label"], x=imp["mean_abs_shap_upwind"], name="Upwind", orientation="h", marker_color="#a9c4ee"
                    )
                    dark(
                        f,
                        barmode="group",
                        height=290,
                        margin=dict(l=0, r=12, t=44, b=8),
                        title=dict(text="Out-of-Fold Mean |SHAP| (XGBoost on Y_adj)", font=dict(size=13, color=INK), x=0.01),
                        legend=dict(orientation="h", y=1.0, x=1, xanchor="right", yanchor="bottom", font=dict(size=10)),
                        yaxis=dict(autorange="reversed"),
                    )
                    show(f)
            if slx is not None:
                with e2:
                    s_ = slx[slx["period"] == per]
                    order = [v for k, v in CLASS_LABEL.items() if k != "none"]
                    f = go.Figure()
                    for kind, off, colr in [("Local (β)", -0.17, "#8fd0a8"), ("Upwind (θ)", 0.17, "#a9c4ee")]:
                        d = s_[s_["kind"] == kind].set_index("label").reindex(order)
                        f.add_trace(
                            go.Scatter(
                                x=d["beta_slx"],
                                y=[i + off for i in range(len(order))],
                                mode="markers",
                                name=kind,
                                marker=dict(size=9, color=colr, line=dict(width=1, color=SURF)),
                                error_x=dict(
                                    type="data", array=(1.96 * d["se_conley_hac"]).values, color=colr, thickness=1.5
                                ),
                                hovertemplate=kind + "<br>coef %{x:.4f}<extra></extra>",
                            )
                        )
                    f.add_vline(x=0, line_color="#5a6580", line_width=1)
                    dark(
                        f,
                        height=290,
                        margin=dict(l=0, r=12, t=44, b=8),
                        title=dict(text=f"SLX coefficients ({per.upper()}), 95% Conley HAC CI", font=dict(size=13, color=INK), x=0.01),
                        legend=dict(orientation="h", y=1.0, x=1, xanchor="right", yanchor="bottom", font=dict(size=10)),
                        yaxis=dict(
                            tickmode="array", tickvals=list(range(len(order))), ticktext=order, autorange="reversed"
                        ),
                    )
                    show(f)
    with right:
        render_cell_inspector(df, clicked_id)
        card(
            "📌 Decision-Maker Guide: Sectoral Intervention",
            body=(
                "**Which sector should policy target in each pixel?**\n"
                "- **Tree Cover (Green):** Prioritize forest protection (REDD+) and upland reforestation.\n"
                "- **Mangrove (Cyan):** Prioritize coastal blue-carbon conservation and hydrology restoration.\n"
                "- **Cropland (Orange):** Indicates that *absence of flooded rice paddies* (in C1) or *sustainable rice management* (in C0) is the strongest lever to keep $\\text{XCH}_4$ low.\n"
                "- **Shrub & Grassland (Yellow):** Target for peat/scrubland fire prevention and rehabilitation."
            ),
            latex_eq="k_i^* = \\arg\\min_{k} \\left(\\phi_{i,k}^{\\text{local}} + \\phi_{i,k}^{\\text{upwind}}\\right) \\quad \\text{s.t.} \\quad \\phi_{i,k}^{\\text{tot}} < 0",
        )
        card("Share of valid pixels by class", raw_html=legend_rows(shares(df, "class_label", list(CLASS_COLORS), CLASS_COLORS)))
        nodata_legend_card()

elif view == "Investment pairing":
    links = load_links(WORK_DIR)
    if links is None:
        with left:
            st.info("05_industrial_community_investment_links.csv not found in the work directory.")
        st.stop()
    lk = links[links["ind_country"].isin(picked)]
    with ctrl:
        pick_c = st.selectbox("Receptor country (C0)", ["All"] + sorted(lk["ind_country"].unique()))
        n = st.slider("Strongest links to draw", 50, 500, 200, 50)
    if pick_c != "All":
        lk = lk[lk["ind_country"] == pick_c]
    top = lk.nlargest(n, "investment_flow_ijk")
    with kpi:
        chips([
            (f"{len(top):,}", "links shown"),
            (f"{top['industrial_grid_id'].nunique():,}", "receptor pixels (C0)"),
            (f"{top['community_grid_id'].nunique():,}", "upwind conservation pixels (C1)"),
            (str(top["com_country"].nunique()), "upwind countries"),
        ])
    with left:
        pos = gdf.set_index("grid_id")[["clon", "clat"]]
        a_ = pos.reindex(top["industrial_grid_id"]).values
        b_ = pos.reindex(top["community_grid_id"]).values
        segs = []
        for cls, color in CLASS_LINE.items():
            m = (top["class_label"] == cls).values
            lons, lats = [], []
            for (x0, y0), (x1, y1) in zip(a_[m], b_[m]):
                if not (np.isnan(x0) or np.isnan(y0) or np.isnan(x1) or np.isnan(y1)):
                    lons += [x0, x1, None]
                    lats += [y0, y1, None]
            if lons:
                segs.append((color, (lons, lats)))
        dots = [(a_[:, 0], a_[:, 1], "#ff8f9c", 6), (b_[:, 0], b_[:, 1], "#9bbcf2", 4)]
        fig_map = flow_lines(
            df, gj, basemap, segs, dots,
            land_gj=active_land_gj, picked_countries=picked, focus_country=zoom_country,
        )
        clicked_id = show_interactive_map(fig_map, key="map_pairing")
        st.markdown("##### Cross-Border & Domestic Conservation Financing Corridors")
        g = (
            top.groupby(["industrial_cluster_id", "ind_country", "class_label", "com_country"])
            .agg(links=("investment_flow_ijk", "size"), flow=("investment_flow_ijk", "sum"))
            .reset_index()
            .sort_values(["ind_country", "com_country"], ascending=[True, True])
        )
        cn = prof.set_index("cluster_id")["label"].to_dict() if prof is not None else {}
        g["industrial_cluster_id"] = g["industrial_cluster_id"].map(lambda c: cn.get(c, f"C{c}"))
        g.columns = [
            "Buyer / Receptor Zone (C0)",
            "Buyer Country",
            "Upwind Conservation Lever",
            "Provider Country (C1)",
            "Corridor Pixels",
            "Priority Flow IndexSum",
        ]
        st.dataframe(g.round(3), hide_index=True, width="stretch")
    with right:
        render_cell_inspector(df, clicked_id)
        card(
            "📌 Decision-Maker Guide: Who Pays Whom?",
            body=(
                "**How to use this map for bilateral/domestic carbon finance:**\n"
                "- **Pink dots (Cluster 0 Buyers/Receptors):** High-methane urban/agricultural pixels that experience high $\\text{XCH}_4$ pressure.\n"
                "- **Blue dots (Cluster 1 Providers):** Upwind forest & mangrove community pixels whose air masses flow directly toward those pink dots within $250\\text{ km}$.\n"
                "- **Policy mechanism:** Direct domestic fiscal transfers or bilateral ASEAN Article 6 carbon finance from **Buyer Countries** to **Provider Countries** along these physical wind corridors."
            ),
            latex_eq="\\text{Flow}_{ij} = \\sqrt{\\text{Need}_i \\times \\text{Lever}_j} \\times \\frac{W_{ij}}{\\max_m W_{im}}",
        )
        card(
            "Financing share by ecosystem lever",
            raw_html=legend_rows(shares(top, "class_label", list(CLASS_LINE), CLASS_LINE, weight="investment_flow_ijk")),
        )
        upw_grp = (
            top.groupby("com_country", as_index=False)
            .agg(n_pix=("community_grid_id", "nunique"), flow_sum=("investment_flow_ijk", "sum"))
            .sort_values("com_country", ascending=True)
        )
        card(
            "Upwind provider countries (A–Z, unranked)",
            raw_html=mini_list([
                (f"<b>{r.com_country}</b> <small>(n = {int(r.n_pix):,} upwind pixels)</small>", f"Flow: {r.flow_sum:.1f}")
                for r in upw_grp.itertuples()
            ]),
        )

else:  # Wind-shed
    _, _, rob = load_evidence(WORK_DIR)
    with ctrl:
        mode = st.radio("Layer", ["Spillover (LISA)", "Airshed corridors"], horizontal=True)
    if mode == "Spillover (LISA)":
        sig = (df["lisa_label"] != "Not significant").mean() * 100
        with kpi:
            chips([
                (f"{len(df):,}", "valid pixels"),
                (f"{sig:.1f}%", "significant (p < 0.05)"),
                (f"{(df['lisa_label'] == 'Low-Low').mean() * 100:.1f}%", "Low-Low (clean buffer)"),
                (f"{(df['lisa_label'] == 'High-High').mean() * 100:.1f}%", "High-High (plume hotspot)"),
            ])
        with left:
            fig_map = categorical_map(
                df, gj, "lisa_label", LISA_COLORS, basemap, order=list(LISA_COLORS),
                land_gj=active_land_gj, picked_countries=picked, focus_country=zoom_country,
            )
            clicked_id = show_interactive_map(fig_map, key="map_lisa")
            if rob is not None:
                st.markdown("##### Scientific Verification: Wind-Shed Robustness Table (`03_robustness.csv`)")
                st.dataframe(rob.round(5), hide_index=True, width="stretch")
        with right:
            render_cell_inspector(df, clicked_id)
            card(
                "📌 Decision-Maker Guide: Transboundary Spillovers",
                body=(
                    "**Why local action alone is not enough:**\n"
                    "- **High-High (Red):** Regional hotspots where upwind land use compounds downwind methane accumulation. Requires **joint cross-district or cross-border mitigation**.\n"
                    "- **Low-Low (Green):** Protected clean-air corridors where upwind forest/mangrove belts consistently suppress regional $\\text{XCH}_4$. Requires **strict conservation zoning**."
                ),
                latex_eq="\\text{Spillover}_i = (\\mathbf{W}\\mathbf{X})_i \\hat{\\boldsymbol{\\theta}}, \\quad I_i = z_i \\sum_j W_{ij} z_j",
            )
            card("Share of valid pixels", raw_html=legend_rows(shares(df, "lisa_label", list(LISA_COLORS), LISA_COLORS, LISA_MEANING)))
            nodata_legend_card()
    else:
        cor = load_corridors(WORK_DIR)
        if cor is None:
            with left:
                st.info("02_airshed_corridors_10km.gpkg not found in the work directory.")
            st.stop()
        with ctrl:
            n = st.slider("Strongest corridors", 50, min(3000, len(cor)), min(500, len(cor)), 50)
        dst_country = gdf.set_index("grid_id")["country"]
        cc = cor[cor["dst_id"].map(dst_country).isin(picked)].head(n)
        with kpi:
            chips([
                (f"{len(cc):,}", "corridors"),
                (f"{cc['weight_wij'].mean():.2f}", "mean weight"),
                (str(df["country"].nunique()), "countries"),
                ("250 km", "max radius"),
            ])
        with left:
            lons, lats, rx, ry = [], [], [], []
            for arr in cc["coords"]:
                lons += arr[:, 0].tolist() + [None]
                lats += arr[:, 1].tolist() + [None]
                rx.append(float(arr[-1, 0]))
                ry.append(float(arr[-1, 1]))
            fig_map = flow_lines(
                df, gj, basemap, [("#f0c36a", (lons, lats))], [(rx, ry, "#ff8f9c", 5)],
                land_gj=active_land_gj, picked_countries=picked, focus_country=zoom_country,
            )
            clicked_id = show_interactive_map(fig_map, key="map_corridors")
        with right:
            render_cell_inspector(df, clicked_id)
            card(
                "📌 Decision-Maker Guide: Physical Wind Corridors",
                body=(
                    "**How atmospheric transport connects jurisdictions:**\n"
                    "- Yellow lines trace the strongest physical wind transport path ($\\le 250\\text{ km}$) feeding into each receptor pixel (pink dot).\n"
                    "- Demonstrates whether a district's atmospheric methane is influenced by domestic neighbours or cross-border ASEAN states."
                ),
                latex_eq="w_{ij} = \\frac{V_j \\cos(\\phi_{ji} - \\theta_j)}{d_{ji}^{1.5}}, \\quad W_{ij} = \\frac{w_{ij}}{\\sum_m w_{im}}",
            )
            nodata_legend_card()

# ============================================================== footer
st.write("")
c1, c2 = st.columns(2)
with c1:
    with st.expander("Claim limits & physical caveats for policy interpretation"):
        st.markdown(
            "1. **Conditional associations, not causal proof:** SLX and SHAP estimates capture spatial associations conditional on fire hotspots, built-up area, nightlights, and 2nd-order geographic trends $P_2(x_i, y_i)$.\n"
            "2. **Satellite $\\text{XCH}_4$ physics & partial pixel coverage:** Sentinel-5P measures column-averaged dry-air methane mixing ratios ($\\text{ppb}$). Because equatorial cloud cover masks many land pixels, country-level index values are reported **alongside the exact number of valid pixels ($n$)** and are **not ranked** across countries.\n"
            "3. **Biogenic methane sources:** Unlike $\\text{CO}_2$, flooded cropland (rice paddies) and coastal wetlands/mangroves can act as biogenic $\\text{CH}_4$ sources. `lever_strength` reflects statistical association with lower $\\text{XCH}_4$ (including absence of agricultural sources), not physical canopy uptake.\n"
            "4. **Static land cover:** ESA WorldCover v200 represents a 2021 baseline snapshot; VIIRS nightlights serve as a 12-month static control."
        )
with c2:
    with st.expander("Methodological workflow in brief (Stages 0–6 across 11 ASEAN countries)"):
        st.markdown(
            "**Stage 1** Equal-area $10\\text{ km}$ grid (`ESRI:102028`) across all 11 ASEAN countries, deduplicated across borders and clipped to land.\n\n"
            "**Stage 2** Sparse directional wind-shed matrices (`h1`, `djf`, `jja`; $d_{\\max} = 250\\text{ km}$, $\\alpha = 1.5$).\n\n"
            "**Stage 3** SLX spatial econometrics with 2nd-order geographic trend control, Conley HAC standard errors, Moran/LISA, and isotropic/placebo wind robustness tests.\n\n"
            "**Stage 4** XGBoost & Random Forest with out-of-fold SHAP values under buffered spatial block CV ($500\\text{ km}$ blocks, $250\\text{ km}$ buffer).\n\n"
            "**Stage 5A/5B** K-Means++ clustering ($k^*=2$, $\\text{Jaccard}_{\\text{OOB}} > 0.98$) and row-max normalized receptor-to-upwind conservation pairing ($W_{ij}^{\\text{rel}} = W_{ij} / \\max_m W_{im}$)."
        )
