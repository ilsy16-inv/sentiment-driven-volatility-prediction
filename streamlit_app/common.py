"""Konfigurasi, gaya, dan fungsi bersama untuk dashboard Streamlit."""


from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# KONFIGURASI
# ============================================================
REPO = "ilsy16-inv/sentiment-driven-volatility-prediction"
BRANCH = "main"
EMITEN = ["ADRO", "ADMR", "ITMG", "PTBA", "MEDC", "AKRA", "PGAS"]
MODELS = ["Naive", "GARCH(1,1)", "XGBoost", "LSTM"]
NAMA = {"Naive": "Naive", "GARCH(1,1)": "GARCH(1,1)",
        "XGBoost": "GARCH–XGBoost", "LSTM": "GARCH–LSTM"}
WARNA = {"Naive": "#B4B8C0", "GARCH(1,1)": "#E8833A", "XGBoost": "#3B6FE0",
         "LSTM": "#1FA38A", "Aktual": "#1F2433"}
MERAH, HIJAU, UNGU = "#D9534F", "#1F9D6B", "#7A5AF8"
SENTIMEN = ["Sentiment_Mean", "Sentiment_Dispersion", "News_Count", "Ada_Berita"]
TARGET_MAPE = 20.0

FILES = {
    "pred_xgb": "06_dokumen_tesis/tabel/hasil_prediksi_xgboost.csv",
    "pred_lstm": "06_dokumen_tesis/tabel/hasil_prediksi_lstm.csv",
    "dm": "06_dokumen_tesis/tabel/tabel_uji_dm_robust_loss.csv",
    "garch": "06_dokumen_tesis/tabel/tabel_parameter_garch.csv",
    "shap": "06_dokumen_tesis/tabel/tabel_shap_per_emiten.csv",
    "abl": "06_dokumen_tesis/tabel/tabel_ablation_sentimen.csv",
    "kondisi": "06_dokumen_tesis/tabel/tabel_ablation_kondisi_pasar.csv",
    "indobert": "02_preprocessing/sentimen_indobert/output_sentimen_v2_manual/tabel_validasi_indobert.csv",
    "sent_harian": "02_preprocessing/sentimen_indobert/output_sentimen_v2_manual/skor_sentimen_harian.csv",
    "test": "03_dataset/final/dataset_test_{e}.csv",
    "scaler": "03_dataset/final/scaler_params_{e}.csv",
    # halaman investor
    "forecast_h1": "06_dokumen_tesis/tabel/tabel_forecast_h1.csv",
    "verifikasi_h1": "06_dokumen_tesis/tabel/verifikasi_H1_vs_aktual.csv",
    "gabungan": "03_dataset/final/dataset_gabungan_X1X2Y.csv",
    "artikel": "02_preprocessing/sentimen_indobert/output_sentimen_v2_manual/skor_sentimen_per_artikel.csv",
    "model_xgb": "04_model/xgboost/xgboost_{e}.pkl",
}

# ============================================================
# GAYA (CSS)
# ============================================================
CSS = f"""
<style>
  .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1350px; }}
  h1, h2, h3 {{ letter-spacing: -0.01em; }}
  .vp-hero {{ position: relative; overflow: hidden; border-radius: 22px; padding: 30px 34px 26px;
     background: linear-gradient(125deg, {WARNA['XGBoost']}22 0%, {WARNA['LSTM']}14 55%, {WARNA['GARCH(1,1)']}22 100%);
     border: 1px solid rgba(120,130,150,.22); margin-bottom: 18px; }}
  .vp-hero:after {{ content: ""; position: absolute; right: -90px; top: -90px; width: 300px; height: 300px; border-radius: 50%;
     background: radial-gradient(circle, {WARNA['XGBoost']}33 0%, transparent 70%); }}
  .vp-eyebrow {{ font-size: .74rem; letter-spacing: .12em; text-transform: uppercase; opacity: .65; margin-bottom: 6px; }}
  .vp-hero h1 {{ font-family: Georgia, 'Times New Roman', serif; font-weight: 400; font-size: 2.05rem; line-height: 1.15; margin: 0 0 10px; max-width: 900px; padding: 0; }}
  .vp-lead {{ opacity: .75; max-width: 820px; line-height: 1.55; margin: 0; }}
  .vp-chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; }}
  .vp-chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 3px 12px; border-radius: 999px; font-size: .78rem;
     background: rgba(255,255,255,.55); border: 1px solid rgba(120,130,150,.25); }}
  .vp-chip i {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
  @media (prefers-color-scheme: dark) {{ .vp-chip {{ background: rgba(20,22,30,.45); }} }}

  .vp-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 4px 0 14px; }}
  .vp-kpi {{ position: relative; overflow: hidden; border-radius: 16px; padding: 18px 18px 14px; border: 1px solid rgba(120,130,150,.22);
     background: var(--secondary-background-color); transition: transform .25s ease, box-shadow .25s ease; animation: vp-rise .6s ease both; }}
  .vp-kpi:hover {{ transform: translateY(-3px); box-shadow: 0 10px 24px rgba(20,30,60,.10); }}
  .vp-kpi:before {{ content: ""; position: absolute; left: 0; right: 0; top: 0; height: 4px; background: var(--c); }}
  .vp-kpi .lbl {{ font-size: .8rem; opacity: .7; display: flex; align-items: center; gap: 6px; }}
  .vp-kpi .lbl i {{ width: 8px; height: 8px; border-radius: 50%; background: var(--c); display: inline-block; }}
  .vp-kpi .big {{ font-family: Georgia, 'Times New Roman', serif; font-size: 2.3rem; line-height: 1.1; margin: 8px 0 4px; font-variant-numeric: tabular-nums; }}
  .vp-kpi .sub {{ font-size: .78rem; opacity: .7; }}
  .vp-kpi .good {{ color: {HIJAU}; font-weight: 650; opacity: 1; }}
  .vp-kpi .bad {{ color: {MERAH}; font-weight: 650; opacity: 1; }}
  .vp-gauge {{ height: 8px; border-radius: 999px; background: rgba(120,130,150,.18); margin: 10px 0 4px; overflow: hidden; display: flex; }}
  .vp-gauge span {{ height: 100%; border-radius: 999px; background: var(--c); animation: vp-grow 1.1s cubic-bezier(.2,.8,.2,1) both; transform-origin: left; }}
  .vp-gauge span + span {{ background: rgba(120,130,150,.35); border-radius: 0; }}
  .vp-scale {{ display: flex; justify-content: space-between; font-size: .72rem; opacity: .6; }}

  .vp-verdicts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 8px; }}
  .vp-verdict {{ display: flex; gap: 12px; align-items: flex-start; border-radius: 16px; padding: 14px 16px; line-height: 1.5; font-size: .92rem;
     background: color-mix(in srgb, var(--v) 9%, transparent); border: 1px solid color-mix(in srgb, var(--v) 32%, transparent); animation: vp-rise .7s ease both; }}
  .vp-verdict .ico {{ flex: none; width: 30px; height: 30px; border-radius: 50%; display: grid; place-items: center; font-weight: 800;
     color: var(--v); background: color-mix(in srgb, var(--v) 18%, transparent); }}
  .vp-verdict b.t {{ display: block; margin-bottom: 2px; }}

  .vp-sec {{ display: flex; align-items: center; gap: 12px; margin: 6px 0 2px; }}
  .vp-badge {{ width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center; font-weight: 800; color: {WARNA['XGBoost']};
     background: {WARNA['XGBoost']}1f; border: 1px solid {WARNA['XGBoost']}44; }}
  .vp-sec h3 {{ margin: 0; padding: 0; font-size: 1.15rem; }}
  .vp-sec p {{ margin: 0; font-size: .8rem; opacity: .65; }}
  .vp-pills {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 2px 0 6px; }}
  .vp-pill {{ display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px; border-radius: 999px; font-size: .78rem;
     background: color-mix(in srgb, var(--c) 10%, transparent); border: 1px solid color-mix(in srgb, var(--c) 35%, transparent); }}
  .vp-pill i {{ width: 8px; height: 8px; border-radius: 50%; background: var(--c); display: inline-block; }}

  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; flex-wrap: wrap; border-bottom: none; }}
  .stTabs [data-baseweb="tab"] {{ height: auto; padding: 7px 14px; border-radius: 999px; border: 1px solid rgba(120,130,150,.25); background: var(--secondary-background-color); }}
  .stTabs [aria-selected="true"] {{ background: {WARNA['XGBoost']}1c; border-color: {WARNA['XGBoost']}88; }}
  .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none; }}
  [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {{ border-radius: 14px; }}
  div[data-testid="stExpander"] {{ border-radius: 14px; }}
  .vp-foot {{ font-size: .75rem; opacity: .6; margin-top: 24px; text-align: center; }}

  .vp-radar {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin: 6px 0 10px; }}
  .vp-rc {{ position: relative; border-radius: 16px; padding: 16px 16px 12px; border: 1px solid color-mix(in srgb, var(--r) 35%, transparent);
     background: linear-gradient(160deg, color-mix(in srgb, var(--r) 12%, transparent), transparent 70%); animation: vp-rise .6s ease both;
     transition: transform .25s ease, box-shadow .25s ease; }}
  .vp-rc:hover {{ transform: translateY(-3px); box-shadow: 0 10px 24px rgba(20,30,60,.10); }}
  .vp-rc .top {{ display: flex; justify-content: space-between; align-items: center; }}
  .vp-rc .tk {{ font-weight: 800; font-size: 1.15rem; letter-spacing: .02em; }}
  .vp-rc .nm {{ font-size: .72rem; opacity: .6; }}
  .vp-tag {{ font-size: .72rem; font-weight: 700; padding: 3px 10px; border-radius: 999px; color: var(--r); background: color-mix(in srgb, var(--r) 16%, transparent); }}
  .vp-rc .v {{ font-family: Georgia, 'Times New Roman', serif; font-size: 1.9rem; margin: 8px 0 0; font-variant-numeric: tabular-nums; }}
  .vp-rc .v small {{ font-family: inherit; font-size: .8rem; opacity: .6; }}
  .vp-rc .ann {{ font-size: .78rem; opacity: .7; }}
  .vp-band {{ position: relative; height: 10px; border-radius: 999px; margin: 12px 0 4px; overflow: hidden; display: flex; }}
  .vp-band span {{ height: 100%; }}
  .vp-band .mk {{ position: absolute; top: -3px; width: 4px; height: 16px; border-radius: 2px; background: #1F2433; box-shadow: 0 0 0 2px #fff; }}
  .vp-rc .snt {{ font-size: .78rem; margin-top: 8px; padding-top: 8px; border-top: 1px dashed rgba(120,130,150,.3); opacity: .85; }}
  .vp-big2 {{ font-family: Georgia, 'Times New Roman', serif; font-size: 2.6rem; line-height: 1.05; font-variant-numeric: tabular-nums; }}
  .vp-disc {{ font-size: .76rem; opacity: .65; border-left: 3px solid rgba(120,130,150,.4); padding: 4px 10px; margin: 10px 0; }}

  @keyframes vp-rise {{ from {{ opacity: 0; transform: translateY(10px); }} }}
  @keyframes vp-grow {{ from {{ transform: scaleX(0); }} }}
  @media (prefers-reduced-motion: reduce) {{ .vp-kpi, .vp-verdict, .vp-gauge span {{ animation: none; }} }}
</style>
"""


PLOT_LAYOUT = dict(
    font=dict(family="Inter, Segoe UI, sans-serif", size=12),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=30, b=10, l=10, r=10),
    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#d0d4dc"),
    legend=dict(orientation="h", y=1.1, x=0, title=None),
    xaxis=dict(showgrid=False, linecolor="rgba(120,130,150,.35)"),
    yaxis=dict(gridcolor="rgba(120,130,150,.18)", griddash="dot", zeroline=False),
)


def gaya(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(**PLOT_LAYOUT, height=height)
    return fig


# ============================================================
# PEMBACAAN DATA DARI GITHUB
# ============================================================
def raw_url(path: str, repo: str, branch: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{branch}/{quote(path)}"


@st.cache_data(ttl=3600, show_spinner=False)
def baca(path: str, repo: str, branch: str, **kw) -> pd.DataFrame:
    return pd.read_csv(raw_url(path, repo, branch), encoding="utf-8-sig", **kw)


@st.cache_data(ttl=3600, show_spinner="Membaca data dari GitHub ...")
def muat_prediksi(repo: str, branch: str) -> pd.DataFrame:
    """Gabungkan aktual, Naive, GARCH(1,1), XGBoost, LSTM — sama seperti EVALUASI_baseline_DM_v3."""
    def inv(v, sc, k):
        lo, hi = float(sc.loc[k, "min"]), float(sc.loc[k, "max"])
        return np.asarray(v, float) * (hi - lo) + lo

    rows = []
    for e in EMITEN:
        te = baca(FILES["test"].format(e=e), repo, branch, parse_dates=["Date"])
        sc = baca(FILES["scaler"].format(e=e), repo, branch, index_col=0)
        rows.append(pd.DataFrame({
            "Emiten": e, "Date": te["Date"],
            "Aktual": inv(te["Volatilitas_Target"], sc, "Volatilitas_Target"),
            "Naive": inv(te["Vol_Lag1"], sc, "Vol_Lag1"),
            "GARCH(1,1)": inv(te["GARCH_Forecast"], sc, "GARCH_Forecast"),
        }))
    df = pd.concat(rows, ignore_index=True)
    for nama, key in [("XGBoost", "pred_xgb"), ("LSTM", "pred_lstm")]:
        p = baca(FILES[key], repo, branch, parse_dates=["Date"])
        p = p[["Emiten", "Date", "Y_Prediksi"]].rename(columns={"Y_Prediksi": nama})
        df = df.merge(p, on=["Emiten", "Date"], how="left")
    return df


def hitung_metrik(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m in MODELS:
        for e in EMITEN:
            g = df[(df.Emiten == e) & df[m].notna()]
            y, p = g["Aktual"].to_numpy(), g[m].to_numpy()
            out.append({
                "Model": m, "Emiten": e,
                "RMSE (×10⁻³)": np.sqrt(np.mean((y - p) ** 2)) * 1e3,
                "MAE (×10⁻³)": np.mean(np.abs(y - p)) * 1e3,
                "MAPE (%)": np.mean(np.abs((y - p) / y)) * 100,
                "R²": 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2),
            })
    m = pd.DataFrame(out)
    avg = m.groupby("Model", sort=False).mean(numeric_only=True).reset_index().assign(Emiten="Rata-rata")
    return pd.concat([m, avg], ignore_index=True)


def fmt_p(p: float) -> str:
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")


def angka(x: float, d: int = 2) -> str:
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def judul(huruf: str, teks: str, sub: str) -> None:
    st.markdown(f'<div class="vp-sec"><span class="vp-badge">{huruf}</span>'
                f'<div><h3>{teks}</h3><p>{sub}</p></div></div>', unsafe_allow_html=True)



def sumber() -> tuple[str, str]:
    """Repositori dan branch yang dipilih di sidebar."""
    return st.session_state.get("repo", REPO), st.session_state.get("branch", BRANCH)
