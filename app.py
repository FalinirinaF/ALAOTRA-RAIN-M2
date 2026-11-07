from pathlib import Path
import calendar
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Alaotra Rain",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = Path(__file__).resolve().parent
LOGO_PATH = BASE / "assets" / "alaotra_mangoro_logo.png"
TEMPLATE_PATH = BASE / "data" / "modele_import.csv"

MONTHS_FR = {
    1: "Janvier", 2: "Février", 3: "Mars", 4: "Avril", 5: "Mai", 6: "Juin",
    7: "Juillet", 8: "Août", 9: "Septembre", 10: "Octobre",
    11: "Novembre", 12: "Décembre",
}
SHORT_MONTHS = {
    1: "Jan", 2: "Fév", 3: "Mar", 4: "Avr", 5: "Mai", 6: "Jun",
    7: "Jul", 8: "Aoû", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Déc",
}

# =============================================================================
# SIDEBAR - OPTIONS AVANT LE STYLE
# =============================================================================
with st.sidebar:
    st.markdown("### 🌧️ ALAOTRA RAIN")
    st.caption("Prévision des précipitations & aide à la décision rizicole")
    st.markdown("---")
    ui_theme = st.radio(
        "Thème",
        ["🌙 Sombre", "☀️ Clair"],
        horizontal=True,
        label_visibility="collapsed",
    )

DARK = ui_theme.startswith("🌙")

BG = "#061116" if DARK else "#f4f8f6"
PANEL = "#0b1d24" if DARK else "#ffffff"
PANEL2 = "#0e252c" if DARK else "#f7fbf8"
TEXT = "#f5fbf8" if DARK else "#14251e"
MUTED = "#9fb4aa" if DARK else "#5e746a"
BORDER = "rgba(67,209,102,.28)" if DARK else "rgba(28,123,69,.22)"
GRID = "rgba(180,210,195,.12)" if DARK else "rgba(40,80,60,.10)"
PLOT_TEMPLATE = "plotly_dark" if DARK else "plotly_white"

# =============================================================================
# STYLE V5
# =============================================================================
st.markdown(
    f"""
    <style>
    :root {{
        --bg:{BG}; --panel:{PANEL}; --panel2:{PANEL2}; --text:{TEXT};
        --muted:{MUTED}; --border:{BORDER}; --green:#56e35e; --cyan:#20bff3;
        --amber:#ffb000; --purple:#c062f2;
    }}
    .stApp {{background:var(--bg); color:var(--text);}}
    .block-container {{padding-top:.8rem; padding-bottom:2rem; max-width:1560px;}}
    [data-testid="stSidebar"] {{
        background:linear-gradient(180deg,#07161d 0%,#08242a 58%,#071c18 100%);
        border-right:1px solid rgba(74,220,97,.18);
    }}
    [data-testid="stSidebar"] * {{color:#eef8f1;}}
    [data-testid="stSidebar"] hr {{border-color:rgba(255,255,255,.10);}}
    .v5-logo-wrap {{
        text-align:center; margin:.1rem auto .15rem; animation:fadeDown .65s ease both;
    }}
    .v5-title {{
        text-align:center; font-size:clamp(2.2rem,5vw,4.4rem); line-height:1;
        letter-spacing:.035em; font-weight:900; margin:.15rem 0 .35rem;
        background:linear-gradient(90deg,#7cf05b 0%,#31d9b1 52%,#2fa8ff 100%);
        -webkit-background-clip:text; color:transparent;
        animation:fadeUp .65s ease both;
    }}
    .v5-subtitle {{
        text-align:center; color:var(--text); font-size:1.08rem; margin-bottom:1.2rem;
        opacity:.92;
    }}
    .import-box {{
        border:1px solid rgba(77,230,92,.52);
        background:linear-gradient(135deg,rgba(18,64,45,.42),rgba(5,33,41,.72));
        border-radius:18px; padding:1.05rem 1.2rem; margin:.65rem 0 1rem;
        animation:fadeUp .5s ease both;
    }}
    .import-title {{font-weight:800; color:#5af063; font-size:1.02rem; letter-spacing:.03em;}}
    .live-pill {{
        display:inline-flex; align-items:center; gap:.45rem; border-radius:999px;
        border:1px solid rgba(77,230,92,.25); padding:.35rem .68rem;
        background:rgba(45,172,81,.08); color:var(--text); font-size:.82rem;
    }}
    .live-dot {{
        width:8px;height:8px;border-radius:50%;background:#56e35e;
        box-shadow:0 0 0 0 rgba(86,227,94,.45); animation:pulse 1.8s infinite;
    }}
    div[data-testid="stMetric"] {{
        background:linear-gradient(145deg,var(--panel),var(--panel2));
        border:1px solid var(--border); padding:1rem 1.05rem; border-radius:17px;
        min-height:132px; transition:transform .2s ease,box-shadow .2s ease,border-color .2s ease;
        animation:fadeUp .5s ease both;
    }}
    div[data-testid="stMetric"]:hover {{
        transform:translateY(-4px); border-color:rgba(86,227,94,.7);
        box-shadow:0 15px 32px rgba(0,0,0,.20);
    }}
    div[data-testid="stMetricLabel"] {{color:var(--muted);}}
    div[data-testid="stMetricValue"] {{color:var(--text);}}
    div[data-testid="stPlotlyChart"] {{
        border:1px solid var(--border); border-radius:17px; padding:.25rem;
        background:linear-gradient(145deg,var(--panel),var(--panel2));
        animation:fadeUp .5s ease both;
    }}
    .section-head {{
        display:flex;align-items:center;gap:.55rem;margin:.2rem 0 .7rem;
        color:#61ed67;font-weight:800;font-size:1.12rem;letter-spacing:.02em;
    }}
    .advice-grid {{
        display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.75rem;
        border:1px solid var(--border);border-radius:17px;padding:.9rem;
        background:linear-gradient(145deg,var(--panel),var(--panel2));
        margin:.7rem 0 1.1rem;
    }}
    .advice-item {{padding:.45rem .7rem;border-right:1px solid rgba(140,180,160,.15);}}
    .advice-item:last-child {{border-right:0;}}
    .advice-item b {{display:block;margin-bottom:.25rem;}}
    .callout {{
        border-left:4px solid #56e35e;border-radius:12px;padding:.8rem 1rem;
        background:rgba(86,227,94,.075);margin:.55rem 0;
    }}
    .warn {{
        border-left:4px solid #ffb000;border-radius:12px;padding:.8rem 1rem;
        background:rgba(255,176,0,.075);margin:.55rem 0;
    }}
    .tiny {{font-size:.82rem;color:var(--muted);}}
    @keyframes pulse {{
        0% {{box-shadow:0 0 0 0 rgba(86,227,94,.42);}}
        70% {{box-shadow:0 0 0 8px rgba(86,227,94,0);}}
        100% {{box-shadow:0 0 0 0 rgba(86,227,94,0);}}
    }}
    @keyframes fadeUp {{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:none}}}}
    @keyframes fadeDown {{from{{opacity:0;transform:translateY(-8px)}}to{{opacity:1;transform:none}}}}
    @media (max-width:900px) {{
        .advice-grid {{grid-template-columns:1fr 1fr;}}
    }}
    @media (max-width:600px) {{
        .advice-grid {{grid-template-columns:1fr;}}
        .advice-item {{border-right:0;border-bottom:1px solid rgba(140,180,160,.12);}}
    }}
    @media (prefers-reduced-motion:reduce) {{
        *,*::before,*::after{{animation:none!important;transition:none!important;}}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# UTILITAIRES
# =============================================================================
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "filename": "alaotra_rain", "scale": 2},
}

def polish(fig, height=None):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        hoverlabel=dict(font_size=13),
        transition=dict(duration=350, easing="cubic-in-out"),
        legend=dict(orientation="h", y=1.03, x=0),
        margin=dict(l=18, r=18, t=45, b=20),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    if height:
        fig.update_layout(height=height)
    return fig

def show_chart(fig, height=None):
    st.plotly_chart(
        polish(fig, height),
        use_container_width=True,
        config=PLOTLY_CONFIG,
    )

def render_header():
    a, b, c = st.columns([1, 1.05, 1])
    with b:
        st.image(str(LOGO_PATH), use_container_width=True)
    st.markdown('<div class="v5-title">ALAOTRA RAIN</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="v5-subtitle">Prévision des précipitations & conseils pour les riziculteurs</div>',
        unsafe_allow_html=True,
    )

def read_csv_flexible(uploaded_file):
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()

    # Fichier régional avec métadonnées avant la vraie ligne CSV
    header_index = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("PARAMETER,YEAR,LAT,LON"):
            header_index = i
            break

    payload = "\n".join(lines[header_index:]) if header_index is not None else text
    try:
        return pd.read_csv(io.StringIO(payload), sep=None, engine="python")
    except Exception:
        return pd.read_csv(io.StringIO(payload))

def normalize_dataset(raw_df):
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Format long attendu
    long_required = {"year", "month", "latitude", "longitude", "precipitation_mm_month"}
    if long_required.issubset(set(df.columns)):
        if "date" not in df.columns:
            df["date"] = pd.to_datetime(
                dict(year=df["year"].astype(int), month=df["month"].astype(int), day=1)
            )
        else:
            df["date"] = pd.to_datetime(df["date"])
        out = df[["date","year","month","latitude","longitude","precipitation_mm_month"]].copy()
        out["year"] = pd.to_numeric(out["year"], errors="coerce")
        out["month"] = pd.to_numeric(out["month"], errors="coerce")
        out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
        out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
        out["precipitation_mm_month"] = pd.to_numeric(out["precipitation_mm_month"], errors="coerce")
        out = out.dropna().sort_values(["latitude","longitude","date"]).reset_index(drop=True)
        return out, "Format mensuel long reconnu"

    # Variantes de noms simples
    aliases = {
        "LAT": "latitude", "LON": "longitude", "YEAR": "year",
        "MONTH": "month", "PRECIPITATION": "precipitation_mm_month",
        "PRECIPITATION_MM": "precipitation_mm_month",
    }
    rename = {c: aliases[c.upper()] for c in df.columns if c.upper() in aliases}
    if rename:
        df = df.rename(columns=rename)
        if long_required.issubset(set(df.columns)):
            return normalize_dataset(df)

    # Format large : PARAMETER,YEAR,LAT,LON,JAN...DEC
    months_cols = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    needed = {"YEAR","LAT","LON"}
    upper_cols = {c.upper(): c for c in df.columns}
    if needed.issubset(upper_cols) and all(m in upper_cols for m in months_cols):
        param_col = upper_cols.get("PARAMETER")
        if param_col:
            param_values = df[param_col].astype(str).str.upper()
            if param_values.str.contains("_SUM").any():
                df = df[param_values.str.contains("_SUM")].copy()
                is_sum = True
            else:
                is_sum = False
        else:
            is_sum = True

        id_cols = [upper_cols["YEAR"], upper_cols["LAT"], upper_cols["LON"]]
        if param_col:
            id_cols = [param_col] + id_cols

        melted = df.melt(
            id_vars=id_cols,
            value_vars=[upper_cols[m] for m in months_cols],
            var_name="month_code",
            value_name="value",
        )
        month_map = {m:i+1 for i,m in enumerate(months_cols)}
        melted["month"] = melted["month_code"].astype(str).str.upper().map(month_map)
        melted["year"] = pd.to_numeric(melted[upper_cols["YEAR"]], errors="coerce")
        melted["latitude"] = pd.to_numeric(melted[upper_cols["LAT"]], errors="coerce")
        melted["longitude"] = pd.to_numeric(melted[upper_cols["LON"]], errors="coerce")
        melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
        melted = melted.replace(-999, np.nan).dropna(subset=["year","month","latitude","longitude","value"])

        if is_sum:
            melted["precipitation_mm_month"] = melted["value"]
            note = "Format mensuel large reconnu : cumul mensuel utilisé"
        else:
            # moyenne journalière mensuelle -> approximation du cumul mensuel
            melted["precipitation_mm_month"] = melted.apply(
                lambda r: r["value"] * calendar.monthrange(int(r["year"]), int(r["month"]))[1],
                axis=1,
            )
            note = "Format mensuel large reconnu : moyenne journalière convertie en cumul mensuel estimé"

        melted["date"] = pd.to_datetime(
            dict(year=melted["year"].astype(int), month=melted["month"].astype(int), day=1)
        )
        out = melted[["date","year","month","latitude","longitude","precipitation_mm_month"]]
        out = out.sort_values(["latitude","longitude","date"]).reset_index(drop=True)
        return out, note

    raise ValueError(
        "Colonnes non reconnues. Utilisez soit le format long "
        "(year, month, latitude, longitude, precipitation_mm_month), "
        "soit le format mensuel large avec YEAR, LAT, LON, JAN...DEC."
    )

def build_regional(df):
    reg = (
        df.groupby(["date","year","month"], as_index=False)["precipitation_mm_month"]
        .mean()
        .rename(columns={"precipitation_mm_month":"precipitation_mm"})
    )
    reg["month_name"] = reg["month"].map(MONTHS_FR)
    return reg

def build_climatology(reg):
    clim = (
        reg.groupby("month")["precipitation_mm"]
        .agg(
            mean="mean",
            median="median",
            std="std",
            q25=lambda x:x.quantile(.25),
            q75=lambda x:x.quantile(.75),
            min="min",
            max="max",
        )
        .reset_index()
    )
    clim["month_name"] = clim["month"].map(MONTHS_FR)
    return clim

def prepare_ml(df):
    data = df.sort_values(["latitude","longitude","date"]).copy()
    data["month_sin"] = np.sin(2*np.pi*data["month"]/12)
    data["month_cos"] = np.cos(2*np.pi*data["month"]/12)

    for lag in [1,2,3,6,12]:
        data[f"precip_lag_{lag}"] = (
            data.groupby(["latitude","longitude"])["precipitation_mm_month"].shift(lag)
        )
    for w in [3,6,12]:
        data[f"rolling_mean_{w}"] = (
            data.groupby(["latitude","longitude"])["precipitation_mm_month"]
            .transform(lambda s:s.shift(1).rolling(w).mean())
        )
    data["target"] = data["precipitation_mm_month"]
    return data.dropna().reset_index(drop=True)

FEATURES = [
    "year","month","latitude","longitude","month_sin","month_cos",
    "precip_lag_1","precip_lag_2","precip_lag_3","precip_lag_6","precip_lag_12",
    "rolling_mean_3","rolling_mean_6","rolling_mean_12",
]

def model_metrics(y, pred):
    return {
        "MAE": float(mean_absolute_error(y,pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y,pred))),
        "R2": float(r2_score(y,pred)),
    }

def train_all_models(df):
    ml = prepare_ml(df)
    years = sorted(ml["year"].astype(int).unique())
    if len(years) < 6:
        raise ValueError("Il faut au moins 6 années de données pour cette comparaison temporelle.")

    max_year = max(years)
    train_end = max_year - 4
    valid_end = max_year - 2

    train = ml[ml.year <= train_end]
    valid = ml[(ml.year > train_end) & (ml.year <= valid_end)]
    test = ml[ml.year > valid_end]

    if min(len(train),len(valid),len(test)) == 0:
        raise ValueError("Découpage temporel impossible avec les années disponibles.")

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=350, max_depth=12, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=450, learning_rate=.035, max_depth=6,
            subsample=.9, colsample_bytree=.9, random_state=42,
            objective="reg:squarederror"
        ),
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        model.fit(train[FEATURES], train["target"])
        pv = np.maximum(model.predict(valid[FEATURES]),0)
        pt = np.maximum(model.predict(test[FEATURES]),0)
        mv = model_metrics(valid["target"], pv)
        mt = model_metrics(test["target"], pt)
        rows.append({
            "model":name,
            "valid_MAE":mv["MAE"],"valid_RMSE":mv["RMSE"],"valid_R2":mv["R2"],
            "test_MAE":mt["MAE"],"test_RMSE":mt["RMSE"],"test_R2":mt["R2"],
        })
        fitted[name] = model

    comparison = pd.DataFrame(rows).sort_values("valid_RMSE").reset_index(drop=True)
    best_name = comparison.loc[0,"model"]
    return {
        "comparison":comparison,
        "models":fitted,
        "best_name":best_name,
        "best_model":fitted[best_name],
        "ml_data":ml,
        "split":(train_end, valid_end, max_year),
    }

def make_forecast(df, model, forecast_year):
    """Prévision récursive depuis la dernière année observée jusqu'à forecast_year.

    Pour 2028 avec des observations jusqu'en 2025, le modèle calcule successivement
    2026, puis 2027, puis 2028. Les prédictions intermédiaires alimentent les lags
    et moyennes mobiles des années suivantes.
    """
    last_observed_year = int(df["year"].max())
    if forecast_year <= last_observed_year:
        raise ValueError(f"L'année prévue doit être supérieure à {last_observed_year}.")

    forecasts = []
    for (lat, lon), g in df.groupby(["latitude", "longitude"]):
        hist = g.sort_values("date").copy()
        vals = hist["precipitation_mm_month"].astype(float).tolist()
        if len(vals) < 12:
            continue

        for year in range(last_observed_year + 1, int(forecast_year) + 1):
            for month in range(1, 13):
                feat = {
                    "year": year,
                    "month": month,
                    "latitude": lat,
                    "longitude": lon,
                    "month_sin": np.sin(2*np.pi*month/12),
                    "month_cos": np.cos(2*np.pi*month/12),
                    "precip_lag_1": vals[-1],
                    "precip_lag_2": vals[-2],
                    "precip_lag_3": vals[-3],
                    "precip_lag_6": vals[-6],
                    "precip_lag_12": vals[-12],
                    "rolling_mean_3": float(np.mean(vals[-3:])),
                    "rolling_mean_6": float(np.mean(vals[-6:])),
                    "rolling_mean_12": float(np.mean(vals[-12:])),
                }
                pred = max(float(model.predict(pd.DataFrame([feat])[FEATURES])[0]), 0)
                vals.append(pred)
                forecasts.append({
                    "year": year, "month": month,
                    "latitude": lat, "longitude": lon,
                    "prediction_mm": pred,
                    "horizon_years": year - last_observed_year,
                })

    out = pd.DataFrame(forecasts)
    if not out.empty:
        out["month_name"] = out["month"].map(MONTHS_FR)
    return out

def advice_box(title, interpretation, items, warning=None):
    st.markdown(f"#### 🌾 {title}")
    st.markdown(f'<div class="callout"><b>Interprétation :</b> {interpretation}</div>', unsafe_allow_html=True)
    for item in items:
        st.markdown(f"- {item}")
    if warning:
        st.markdown(f'<div class="warn"><b>Prudence :</b> {warning}</div>', unsafe_allow_html=True)

# =============================================================================
# EN-TÊTE + IMPORT CSV OBLIGATOIRE
# =============================================================================
render_header()

st.markdown(
    '<div class="import-box"><div class="import-title">1. IMPORTATION DES DONNÉES CSV</div>'
    '<div class="tiny">Importez votre fichier de précipitations mensuelles avant tout traitement, analyse ou prévision.</div></div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Glissez-déposez votre fichier CSV ici ou cliquez pour parcourir",
    type=["csv"],
    accept_multiple_files=False,
)

if TEMPLATE_PATH.exists():
    with open(TEMPLATE_PATH, "rb") as f:
        st.download_button(
            "⬇️ Télécharger un exemple de structure CSV",
            data=f.read(),
            file_name="modele_import.csv",
            mime="text/csv",
        )

if uploaded is None:
    st.markdown(
        '<div class="warn"><b>Aucune donnée importée.</b> Les graphiques, analyses, modèles et prévisions '
        'restent volontairement désactivés jusqu’à l’importation d’un fichier CSV.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

try:
    raw_df = read_csv_flexible(uploaded)
    df, import_note = normalize_dataset(raw_df)
except Exception as exc:
    st.error(f"Impossible d'utiliser ce fichier : {exc}")
    st.stop()

if df.empty:
    st.error("Le fichier ne contient aucune observation exploitable.")
    st.stop()

st.success(f"✅ {uploaded.name} importé — {len(df):,} observations utilisables. {import_note}")

regional = build_regional(df)
climatology = build_climatology(regional)

# =============================================================================
# SIDEBAR - NAVIGATION + FILTRES APRÈS IMPORT
# =============================================================================
with st.sidebar:
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🏠 Vue d'ensemble",
            "☁️ Climatologie",
            "🗺️ Analyse spatiale",
            "✨ Machine Learning",
            "📅 Prévision",
            "🌾 Conseils rizicoles",
            "📤 Export & Données",
            "ℹ️ À propos",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### PARAMÈTRES")
    show_markers = st.toggle("Afficher davantage de marqueurs", value=True)
    smooth_mode = st.toggle("Courbes lissées", value=True)
    animations = st.toggle("Animations", value=True)

    min_year, max_year = int(df.year.min()), int(df.year.max())
    if min_year < max_year:
        year_range = st.slider("Période", min_year, max_year, (min_year, max_year))
    else:
        year_range = (min_year, max_year)

    seasonal_threshold = st.slider("Seuil début saison (mm/mois)", 50, 350, 150, 10)

    st.markdown("---")
    st.markdown(
        '<div class="tiny">Les résultats sont des outils d’aide à la décision. '
        'Toujours compléter avec l’observation terrain et les bulletins météorologiques officiels.</div>',
        unsafe_allow_html=True,
    )

filtered = regional[(regional.year >= year_range[0]) & (regional.year <= year_range[1])].copy()

# =============================================================================
# PAGE 1 — VUE D'ENSEMBLE
# =============================================================================
if page == "🏠 Vue d'ensemble":
    st.markdown('<div class="live-pill"><span class="live-dot"></span>Données importées et interface interactive</div>', unsafe_allow_html=True)

    annual = filtered.groupby("year", as_index=False)["precipitation_mm"].sum()
    monthly_mean = filtered.groupby("month", as_index=False)["precipitation_mm"].mean()
    wettest = int(monthly_mean.loc[monthly_mean.precipitation_mm.idxmax(),"month"])
    driest = int(monthly_mean.loc[monthly_mean.precipitation_mm.idxmin(),"month"])
    avg_annual = annual.precipitation_mm.mean() if len(annual) else 0

    if len(annual) >= 2:
        x = annual["year"].to_numpy()
        y = annual["precipitation_mm"].to_numpy()
        slope = np.polyfit(x,y,1)[0]
        trend_pct = (slope * (x[-1]-x[0]) / max(y.mean(),1))*100
    else:
        trend_pct = 0

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🌧️ Pluie moyenne annuelle", f"{avg_annual:,.0f} mm")
    c2.metric("📅 Mois le plus pluvieux", MONTHS_FR[wettest], f"{monthly_mean.loc[monthly_mean.month==wettest,'precipitation_mm'].iloc[0]:.0f} mm")
    c3.metric("💧 Mois le plus sec", MONTHS_FR[driest], f"{monthly_mean.loc[monthly_mean.month==driest,'precipitation_mm'].iloc[0]:.0f} mm")
    c4.metric("📈 Tendance période", f"{trend_pct:+.1f}%")

    left,right = st.columns(2)
    with left:
        fig = px.line(annual, x="year", y="precipitation_mm", markers=show_markers, title="ÉVOLUTION DES PRÉCIPITATIONS ANNUELLES")
        fig.update_traces(line_shape="spline" if smooth_mode else "linear", line=dict(width=3))
        show_chart(fig, 410)
    with right:
        fig = px.bar(
            monthly_mean.assign(month_name=monthly_mean.month.map(SHORT_MONTHS)),
            x="month_name", y="precipitation_mm",
            title="RÉPARTITION MOYENNE MENSUELLE"
        )
        show_chart(fig, 410)

    left,right = st.columns(2)
    with left:
        spatial = df.groupby(["latitude","longitude"],as_index=False)["precipitation_mm_month"].mean()
        fig = px.scatter_map(
            spatial,lat="latitude",lon="longitude",
            size="precipitation_mm_month",color="precipitation_mm_month",
            zoom=5.5,map_style="open-street-map",size_max=30,
            title="CARTE DES PRÉCIPITATIONS MOYENNES",
            labels={"precipitation_mm_month":"mm/mois"}
        )
        show_chart(fig, 480)
    with right:
        heat = filtered.pivot_table(index="year",columns="month",values="precipitation_mm",aggfunc="mean")
        heat = heat.reindex(columns=range(1,13))
        heat.columns = [SHORT_MONTHS[m] for m in heat.columns]
        fig = px.imshow(heat,aspect="auto",title="HEATMAP ANNÉE / MOIS",labels=dict(x="Mois",y="Année",color="mm"))
        show_chart(fig, 480)

    oct_dec = monthly_mean[monthly_mean.month.isin([10,11,12])].precipitation_mm.sum()
    dry_season = monthly_mean[monthly_mean.month.isin([6,7,8])].precipitation_mm.mean()
    st.markdown(
        f"""
        <div class="advice-grid">
          <div class="advice-item"><b style="color:#25baff">💧 Saison des pluies</b>Oct–Déc : environ {oct_dec:.0f} mm en moyenne cumulée. Surveiller la régularité des pluies avant les décisions de semis.</div>
          <div class="advice-item"><b style="color:#65e35c">🌱 Gestion de l'eau</b>Préparer le drainage lorsque plusieurs mois humides se succèdent et conserver les ouvrages d'irrigation fonctionnels.</div>
          <div class="advice-item"><b style="color:#ffb000">☀️ Période sèche</b>Juin–Août : moyenne mensuelle d'environ {dry_season:.0f} mm. Anticiper les besoins en eau et l'entretien des canaux.</div>
          <div class="advice-item"><b style="color:#ce6df4">📅 Planification</b>Comparer les tendances récentes avec la climatologie avant d'organiser pépinières, repiquage et fertilisation.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# PAGE 2 — CLIMATOLOGIE
# =============================================================================
elif page == "☁️ Climatologie":
    st.markdown('<div class="section-head">☁️ CLIMATOLOGIE MENSUELLE</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=climatology.month_name, y=climatology["q75"], mode="lines",
        line=dict(width=0), showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=climatology.month_name, y=climatology["q25"], mode="lines",
        fill="tonexty", name="Intervalle 25–75 %", line=dict(width=0)
    ))
    fig.add_trace(go.Scatter(
        x=climatology.month_name, y=climatology["mean"],
        mode="lines+markers", name="Moyenne",
        line=dict(shape="spline" if smooth_mode else "linear",width=3)
    ))
    fig.update_layout(title="Cycle saisonnier moyen des précipitations")
    show_chart(fig, 480)

    clim_table = climatology[["month_name","mean","median","std","q25","q75","min","max"]].copy()
    clim_table.columns = ["Mois","Moyenne","Médiane","Écart-type","Q25","Q75","Min","Max"]
    st.dataframe(clim_table.style.format({c:"{:.1f}" for c in clim_table.columns if c!="Mois"}),use_container_width=True,hide_index=True)

    wet_months = climatology[climatology["mean"] >= seasonal_threshold]["month_name"].tolist()
    advice_box(
        "Interprétation climatologique",
        f"Avec un seuil exploratoire de {seasonal_threshold} mm/mois, les mois les plus régulièrement humides sont : {', '.join(wet_months) if wet_months else 'aucun mois au-dessus du seuil'}.",
        [
            "Utiliser la climatologie comme référence, mais vérifier la situation de l'année en cours.",
            "Planifier le drainage avant les mois historiquement les plus arrosés.",
            "Renforcer la gestion de l'irrigation lorsque les mois historiquement secs approchent.",
        ],
        "Un seuil mensuel ne suffit pas à déterminer seul une date agronomique de semis."
    )

# =============================================================================
# PAGE 3 — ANALYSE SPATIALE
# =============================================================================
elif page == "🗺️ Analyse spatiale":
    st.markdown('<div class="section-head">🗺️ ANALYSE SPATIALE</div>', unsafe_allow_html=True)

    c1,c2,c3 = st.columns([1,1,2])
    selected_year = c1.selectbox("Année",sorted(df.year.unique(),reverse=True))
    selected_month = c2.selectbox("Mois",range(1,13),format_func=lambda m:MONTHS_FR[m])
    animate_map = c3.toggle("▶️ Animer ce mois sur toutes les années",value=False)

    if animate_map:
        sub = df[df.month==selected_month].copy()
        fig = px.scatter_map(
            sub,lat="latitude",lon="longitude",
            size="precipitation_mm_month",color="precipitation_mm_month",
            animation_frame="year",zoom=5.5,map_style="open-street-map",
            size_max=32,title=f"Évolution spatiale — {MONTHS_FR[selected_month]}"
        )
    else:
        sub = df[(df.year==selected_year)&(df.month==selected_month)].copy()
        fig = px.scatter_map(
            sub,lat="latitude",lon="longitude",
            size="precipitation_mm_month",color="precipitation_mm_month",
            zoom=5.5,map_style="open-street-map",size_max=36,
            title=f"Précipitations — {MONTHS_FR[selected_month]} {selected_year}"
        )
    show_chart(fig, 600)

    if not sub.empty and "year" in sub.columns:
        current = sub.groupby(["latitude","longitude"],as_index=False)["precipitation_mm_month"].mean()
        spread = current.precipitation_mm_month.max() - current.precipitation_mm_month.min()
        advice_box(
            "Lecture spatiale pour la riziculture",
            f"L'écart entre les points les plus humides et les plus secs atteint environ {spread:.0f} mm pour la sélection affichée.",
            [
                "Éviter d'appliquer une seule décision d'irrigation à toute la région lorsque les écarts spatiaux sont importants.",
                "Renforcer le contrôle du drainage dans les zones les plus arrosées.",
                "Surveiller plus tôt les déficits hydriques dans les zones relativement sèches.",
            ],
            "Les points représentent une grille climatique et non les limites exactes de chaque parcelle."
        )

# =============================================================================
# PAGE 4 — MACHINE LEARNING
# =============================================================================
elif page == "✨ Machine Learning":
    st.markdown('<div class="section-head">✨ MACHINE LEARNING</div>', unsafe_allow_html=True)
    st.write("Entraînez et comparez les modèles directement à partir du CSV importé.")

    if st.button("🚀 Entraîner Random Forest + XGBoost",type="primary",use_container_width=True):
        with st.spinner("Préparation des variables temporelles et entraînement..."):
            try:
                st.session_state["ml_result"] = train_all_models(df)
            except Exception as exc:
                st.error(str(exc))

    ml_result = st.session_state.get("ml_result")

    if ml_result:
        comp = ml_result["comparison"]
        best = ml_result["best_name"]
        st.success(f"Meilleur modèle sur la validation : **{best}**")

        st.dataframe(
            comp.style.format({
                "valid_MAE":"{:.2f}","valid_RMSE":"{:.2f}","valid_R2":"{:.3f}",
                "test_MAE":"{:.2f}","test_RMSE":"{:.2f}","test_R2":"{:.3f}",
            }),
            use_container_width=True,hide_index=True
        )

        best_row = comp.iloc[0]
        cols = st.columns(4)
        gauges = [
            ("RMSE validation",best_row.valid_RMSE,0,max(best_row.valid_RMSE*2,1)),
            ("R² validation",best_row.valid_R2,-1,1),
            ("RMSE test",best_row.test_RMSE,0,max(best_row.test_RMSE*2,1)),
            ("R² test",best_row.test_R2,-1,1),
        ]
        for col,(label,val,vmin,vmax) in zip(cols,gauges):
            with col:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",value=float(val),
                    title={"text":label},
                    gauge={"axis":{"range":[vmin,vmax]},"bar":{"color":"#56e35e"}}
                ))
                show_chart(fig,250)

        best_model = ml_result["best_model"]
        if hasattr(best_model,"feature_importances_"):
            imp = pd.DataFrame({"variable":FEATURES,"importance":best_model.feature_importances_}).sort_values("importance",ascending=True)
            fig = px.bar(imp.tail(10),x="importance",y="variable",orientation="h",title="Variables les plus influentes")
            show_chart(fig,430)

        r2v = float(best_row.valid_R2)
        if r2v >= .6:
            interpretation = "Le modèle explique une part intéressante de la variabilité de validation, mais la pluie reste un phénomène difficile à prévoir."
        elif r2v >= .3:
            interpretation = "Le modèle capte une partie de la structure historique, avec une incertitude encore importante."
        else:
            interpretation = "La performance est faible : les prévisions doivent être considérées comme très exploratoires."

        advice_box(
            "Interprétation du modèle",
            interpretation,
            [
                "Utiliser la prévision comme signal complémentaire, jamais comme seule base de décision agricole.",
                "Comparer les sorties avec les observations locales récentes avant le semis ou l'irrigation.",
                "Réentraîner le modèle lorsque de nouvelles années de données deviennent disponibles.",
            ],
            "Une bonne performance statistique ne garantit pas une prévision exacte d'un événement local."
        )
    else:
        st.info("Cliquez sur le bouton d'entraînement pour générer les performances à partir des données importées.")

# =============================================================================
# PAGE 5 — PRÉVISION
# =============================================================================
elif page == "📅 Prévision":
    st.markdown('<div class="section-head">📅 PRÉVISION SAISONNIÈRE</div>', unsafe_allow_html=True)

    ml_result = st.session_state.get("ml_result")
    last_observed_year = int(df.year.max())
    default_forecast_year = last_observed_year + 1
    forecast_year = st.number_input(
        "Année à prévoir",
        min_value=default_forecast_year,
        max_value=default_forecast_year + 10,
        value=default_forecast_year,
        step=1,
    )

    if ml_result is None:
        st.warning("Entraînez d'abord les modèles dans l'onglet **Machine Learning**.")
    else:
        if st.button(f"🌧️ Générer la prévision {forecast_year}", type="primary", use_container_width=True):
            with st.spinner(f"Calcul récursif {last_observed_year + 1} → {forecast_year}..."):
                actual_fc = make_forecast(df, ml_result["best_model"], int(forecast_year))
                st.session_state["forecast_v6"] = actual_fc
                st.session_state["forecast_v6_target"] = int(forecast_year)

        fc_all = st.session_state.get("forecast_v6")
        generated_target = st.session_state.get("forecast_v6_target")

        if isinstance(fc_all, pd.DataFrame) and not fc_all.empty:
            if generated_target != int(forecast_year):
                st.info(f"L'affichage actuel correspond à {generated_target}. Cliquez sur **Générer la prévision {forecast_year}** pour recalculer.")
            else:
                fc = fc_all[fc_all.year == int(forecast_year)].copy()
                reg_fc = fc.groupby("month")["prediction_mm"].agg(["mean","min","max","std"]).reset_index()
                reg_fc["month_name"] = reg_fc["month"].map(MONTHS_FR)

                oct_dec = reg_fc[reg_fc.month.isin([10,11,12])]
                c1,c2,c3 = st.columns(3)
                for col,m in zip([c1,c2,c3],[10,11,12]):
                    row = reg_fc[reg_fc.month==m].iloc[0]
                    col.metric(
                        f"{MONTHS_FR[m]} {forecast_year}",
                        f"{row['mean']:.1f} mm",
                        f"± {row['std']:.1f} mm spatial"
                    )

                hist_mean = climatology[["month","mean"]].rename(columns={"mean":"hist_mean"})
                compare = reg_fc.merge(hist_mean,on="month",how="left")
                compare["ecart_pct"] = (compare["mean"]-compare["hist_mean"])/compare["hist_mean"].replace(0,np.nan)*100

                fig = go.Figure()
                fig.add_trace(go.Bar(x=compare.month_name,y=compare["mean"],name=f"Prévision {forecast_year}"))
                fig.add_trace(go.Scatter(x=compare.month_name,y=compare.hist_mean,mode="lines+markers",name="Moyenne historique"))
                fig.update_layout(title=f"Prévision mensuelle {forecast_year} vs climatologie")
                show_chart(fig,470)

                # Comparaison des années récursives calculées
                yearly_fc = (
                    fc_all.groupby(["year","month"],as_index=False)["prediction_mm"].mean()
                )
                yearly_fc["month_name"] = yearly_fc["month"].map(SHORT_MONTHS)
                if yearly_fc["year"].nunique() > 1:
                    fig2 = px.line(
                        yearly_fc,
                        x="month_name", y="prediction_mm", color="year",
                        markers=True,
                        title=f"Évolution des prévisions récursives {last_observed_year + 1}–{forecast_year}",
                        labels={"prediction_mm":"Précipitation (mm)","month_name":"Mois","year":"Année"},
                    )
                    show_chart(fig2,430)

                season_total = oct_dec["mean"].sum()
                historical_season = climatology[climatology.month.isin([10,11,12])]["mean"].sum()
                diff = (season_total-historical_season)/max(historical_season,1)*100
                horizon = int(forecast_year) - last_observed_year

                if diff > 15:
                    status = "plus humide que la référence historique"
                    tips = ["Préparer le drainage avant les épisodes prolongés.","Surveiller l'engorgement des parcelles.","Adapter les apports d'eau aux pluies réellement observées."]
                elif diff < -15:
                    status = "plus sèche que la référence historique"
                    tips = ["Sécuriser l'accès à l'irrigation.","Surveiller l'humidité des parcelles avant le repiquage.","Éviter de se fier uniquement au calendrier habituel."]
                else:
                    status = "proche de la référence historique"
                    tips = ["Maintenir la surveillance des pluies réelles.","Préparer à la fois drainage et irrigation.","Adapter les interventions au stade du riz."]

                if horizon == 1:
                    uncertainty = "Horizon court : c'est la prévision la plus proche des observations disponibles."
                elif horizon <= 3:
                    uncertainty = f"Horizon de {horizon} ans : l'incertitude augmente car les années intermédiaires sont elles-mêmes prédites."
                else:
                    uncertainty = f"Horizon de {horizon} ans : prévision très exploratoire ; l'accumulation d'erreurs récursives peut être importante."

                advice_box(
                    f"Lecture de la saison Octobre–Décembre {forecast_year}",
                    f"Le cumul moyen prévu sur Octobre–Décembre est d'environ {season_total:.0f} mm, soit {diff:+.1f}% par rapport à la climatologie : saison {status}.",
                    tips,
                    uncertainty
                )

                st.caption(
                    f"Méthode V6 : pour prévoir {forecast_year}, le système calcule successivement toutes les années depuis {last_observed_year + 1}. "
                    "Chaque année prédite alimente les retards et moyennes mobiles de l'année suivante."
                )

                st.download_button(
                    "⬇️ Exporter les prévisions jusqu'à l'année choisie",
                    data=fc_all.to_csv(index=False).encode("utf-8"),
                    file_name=f"previsions_recursives_{last_observed_year + 1}_{forecast_year}.csv",
                    mime="text/csv",
                )

# =============================================================================
# PAGE 6 — CONSEILS RIZICOLES
# =============================================================================
elif page == "🌾 Conseils rizicoles":
    st.markdown('<div class="section-head">🌾 CONSEILS RIZICOLES</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="advice-grid">
          <div class="advice-item"><b style="color:#25baff">💧 Début de saison</b>Observer la régularité des pluies, pas seulement un cumul élevé sur un seul mois.</div>
          <div class="advice-item"><b style="color:#65e35c">🌱 Excès d'eau</b>Maintenir les drains et surveiller l'engorgement lorsque les prévisions et observations indiquent une séquence humide.</div>
          <div class="advice-item"><b style="color:#ffb000">☀️ Déficit hydrique</b>Préserver l'eau, entretenir les canaux et prioriser les parcelles les plus sensibles en période sèche.</div>
          <div class="advice-item"><b style="color:#ce6df4">📅 Décisions</b>Combiner historique, prévision, état réel du sol et stade de la culture avant toute intervention.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    advice_box(
        "Règle d'utilisation",
        "Alaotra Rain transforme les données de pluie en indicateurs d'aide à la décision.",
        [
            "Vérifier les observations locales avant le semis, l'irrigation ou la fertilisation.",
            "Consulter les services techniques agricoles et les bulletins météorologiques officiels pour les décisions importantes.",
            "Documenter les observations de terrain afin d'améliorer progressivement le système.",
        ],
        "L'application n'est pas un substitut à l'expertise agronomique locale."
    )

# =============================================================================
# PAGE 7 — EXPORT & DONNÉES
# =============================================================================
elif page == "📤 Export & Données":
    st.markdown('<div class="section-head">📤 EXPORT & DONNÉES</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Observations",f"{len(df):,}")
    c2.metric("Points géographiques",df[["latitude","longitude"]].drop_duplicates().shape[0])
    c3.metric("Période",f"{int(df.year.min())}–{int(df.year.max())}")
    c4.metric("Valeurs manquantes",int(df.isna().sum().sum()))

    st.dataframe(df.head(300),use_container_width=True,hide_index=True)
    st.download_button(
        "⬇️ Exporter le dataset normalisé",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="alaotra_rain_dataset_normalise.csv",
        mime="text/csv",
    )

# =============================================================================
# PAGE 8 — À PROPOS
# =============================================================================
else:
    st.markdown('<div class="section-head">ℹ️ À PROPOS</div>', unsafe_allow_html=True)
    st.write(
        "Alaotra Rain est un prototype d'analyse et de prévision des précipitations "
        "orienté vers l'aide à la décision pour la riziculture dans la région Alaotra-Mangoro."
    )
    st.image(str(LOGO_PATH),width=280)
    st.markdown(
        '<div class="warn">Les prévisions sont expérimentales. Elles doivent être complétées par '
        'les observations terrain, les services techniques et les bulletins météorologiques officiels.</div>',
        unsafe_allow_html=True,
    )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    '<div class="tiny" style="text-align:center;margin-top:1.5rem;">Alaotra Rain V5 • Interface interactive • Données importées par l’utilisateur</div>',
    unsafe_allow_html=True,
)
