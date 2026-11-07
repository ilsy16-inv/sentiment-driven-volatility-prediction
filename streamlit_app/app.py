"""
Dashboard Hasil Penelitian — Prediksi Volatilitas Saham Energi LQ45
====================================================================
Tesis Magister: Ilda Nurida (6032251068), SIMT ITS, 2026.

Semua data dibaca LANGSUNG dari repositori GitHub:
    https://github.com/ilsy16-inv/sentiment-driven-volatility-prediction

Setiap kali hasil notebook terbaru di-push ke GitHub, dashboard ini ikut
memperbarui datanya (cache 1 jam, atau klik "Muat ulang data" di sidebar).

Menjalankan secara lokal:
    pip install -r requirements.txt
    streamlit run app.py
"""

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
}

st.set_page_config(page_title="Volatilitas Saham Energi LQ45", page_icon="📈",
                   layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# GAYA (CSS)
# ============================================================
st.markdown(f"""
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

  @keyframes vp-rise {{ from {{ opacity: 0; transform: translateY(10px); }} }}
  @keyframes vp-grow {{ from {{ transform: scaleX(0); }} }}
  @media (prefers-reduced-motion: reduce) {{ .vp-kpi, .vp-verdict, .vp-gauge span {{ animation: none; }} }}
</style>
""", unsafe_allow_html=True)

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


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("⚙️ Sumber data")
    repo = st.text_input("Repositori GitHub", REPO)
    branch = st.text_input("Branch", BRANCH)
    st.caption(f"[Buka repositori ↗](https://github.com/{repo}/tree/{branch})")
    if st.button("🔄 Muat ulang data", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Data di-cache 1 jam. Klik tombol di atas setelah push hasil notebook terbaru.")
    st.divider()
    st.caption("Tesis Magister — Ilda Nurida (6032251068)\nSIMT ITS · 2026")

try:
    df = muat_prediksi(repo, branch)
    metrik = hitung_metrik(df)
    dm = baca(FILES["dm"], repo, branch)
    garch = baca(FILES["garch"], repo, branch)
    shap = baca(FILES["shap"], repo, branch)
    abl = baca(FILES["abl"], repo, branch)
    kondisi = baca(FILES["kondisi"], repo, branch)
    indobert = baca(FILES["indobert"], repo, branch)
except Exception as err:  # noqa: BLE001
    st.error(f"Gagal membaca data dari GitHub ({repo}, branch {branch}).\n\n{err}")
    st.stop()

if df[["XGBoost", "LSTM"]].isna().any().any():
    st.warning("Sebagian tanggal uji belum punya prediksi XGBoost/LSTM. "
               "Jalankan ulang notebook model versi terbaru lalu push ke GitHub.")

avg = metrik[metrik.Emiten == "Rata-rata"].set_index("Model")
shap_sent = shap[shap.Fitur.isin(SENTIMEN)]["RATA-RATA"].sum() / shap["RATA-RATA"].sum()
dm_gab = dm[(dm.Loss == "MSE") & (dm.Emiten == "Gabungan")].set_index("Pembanding").loc["XGBoost vs GARCH(1,1)"]
n_sig = int((abl.p_value < 0.05).sum())

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="vp-hero">
  <div class="vp-eyebrow">Tesis Magister · SIMT ITS · 2026 · revisi ICoABCD</div>
  <h1>Prediksi volatilitas saham energi LQ45 dengan hybrid GARCH–machine learning dan sentimen berita</h1>
  <p class="vp-lead">GARCH–XGBoost dan GARCH–LSTM dibandingkan dengan Naive dan GARCH(1,1) pada periode uji
  {df.Date.min():%d %b %Y} – {df.Date.max():%d %b %Y} ({angka(len(df), 0)} hari-emiten), dengan fitur sentimen
  IndoBERT dari berita ekonomi.</p>
  <div class="vp-chips">
    <span class="vp-chip"><i style="background:{WARNA['XGBoost']}"></i>Hybrid GARCH–XGBoost</span>
    <span class="vp-chip"><i style="background:{WARNA['LSTM']}"></i>Hybrid GARCH–LSTM</span>
    <span class="vp-chip"><i style="background:{WARNA['GARCH(1,1)']}"></i>Baseline GARCH(1,1)-t</span>
    <span class="vp-chip">IndoBERT · indobert-base-p2</span>
    <span class="vp-chip">{' · '.join(EMITEN)}</span>
  </div>
</div>
""", unsafe_allow_html=True)


def kartu_mape(model: str, warna: str, catatan: str | None = None) -> str:
    v = avg.loc[model, "MAPE (%)"]
    lebar = min(100, v / TARGET_MAPE * 100)
    if catatan is None:
        d = v - avg.loc["GARCH(1,1)", "MAPE (%)"]
        kls = "good" if d < 0 else "bad"
        catatan = f'<span class="{kls}">{"−" if d < 0 else "+"}{angka(abs(d))} poin</span> vs GARCH(1,1)'
    return f"""<div class="vp-kpi" style="--c:{warna}">
      <div class="lbl"><i></i>MAPE {NAMA[model]}</div>
      <div class="big">{angka(v)}%</div><div class="sub">{catatan}</div>
      <div class="vp-gauge"><span style="width:{lebar:.1f}%"></span></div>
      <div class="vp-scale"><span>0%</span><span>target {TARGET_MAPE:.0f}%</span></div></div>"""


st.markdown(f"""<div class="vp-kpis">
  {kartu_mape("XGBoost", WARNA["XGBoost"])}
  {kartu_mape("LSTM", WARNA["LSTM"])}
  {kartu_mape("GARCH(1,1)", WARNA["GARCH(1,1)"], "Baseline ekonometrika")}
  <div class="vp-kpi" style="--c:{UNGU}">
    <div class="lbl"><i></i>Kontribusi SHAP sentimen</div>
    <div class="big">{angka(shap_sent * 100)}%</div><div class="sub">dari total |SHAP| GARCH–XGBoost</div>
    <div class="vp-gauge"><span style="width:{shap_sent * 100:.1f}%"></span><span style="flex:1"></span></div>
    <div class="vp-scale"><span>Sentimen (X2)</span><span>Teknikal &amp; GARCH (X1)</span></div>
  </div>
</div>""", unsafe_allow_html=True)

st.markdown(f"""<div class="vp-verdicts">
  <div class="vp-verdict" style="--v:{HIJAU}"><span class="ico">✓</span><div><b class="t">Target akurasi tercapai</b>
    MAPE rata-rata GARCH–XGBoost <b>{angka(avg.loc['XGBoost', 'MAPE (%)'])}%</b> dan GARCH–LSTM
    <b>{angka(avg.loc['LSTM', 'MAPE (%)'])}%</b>, jauh di bawah target {TARGET_MAPE:.0f}%.</div></div>
  <div class="vp-verdict" style="--v:#E0A100"><span class="ico">!</span><div><b class="t">GARCH(1,1) masih tangguh</b>
    Pada uji Diebold–Mariano gabungan (loss MSE), GARCH(1,1) lebih akurat daripada GARCH–XGBoost
    (DM = <b>{angka(dm_gab.DM)}</b>, p = <b>{fmt_p(dm_gab.p_value)}</b>).</div></div>
  <div class="vp-verdict" style="--v:{WARNA['XGBoost']}"><span class="ico">i</span><div><b class="t">Sentimen dipakai, tetapi belum menurunkan error</b>
    Fitur sentimen menyumbang <b>{angka(shap_sent * 100)}%</b> SHAP; pada uji ablasi <b>{n_sig} dari {len(abl)}</b>
    kombinasi model–emiten berubah signifikan.</div></div>
</div>""", unsafe_allow_html=True)

st.write("")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Performa model", "📈 Aktual vs prediksi", "⚖️ Uji Diebold–Mariano",
    "🧩 Kontribusi fitur (SHAP)", "📰 Pengaruh sentimen", "🔎 Karakteristik & validasi",
])

# ------------------------------------------------------------
# TAB 1 — PERFORMA
# ------------------------------------------------------------
with tab1:
    judul("C", "Performa prediksi volatilitas", "Naive, GARCH(1,1), GARCH–XGBoost, dan GARCH–LSTM pada data uji")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        met = st.segmented_control("Metrik", ["MAPE (%)", "RMSE (×10⁻³)", "MAE (×10⁻³)", "R²"],
                                   default="MAPE (%)", key="met") or "MAPE (%)"
        d = metrik[metrik.Emiten != "Rata-rata"].copy()
        d["Model "] = d.Model.map(NAMA)
        fig = px.bar(d, x="Emiten", y=met, color="Model ", barmode="group",
                     color_discrete_map={NAMA[k]: WARNA[k] for k in MODELS},
                     category_orders={"Emiten": EMITEN, "Model ": [NAMA[m] for m in MODELS]})
        fig.update_traces(marker_line_width=0, hovertemplate="%{x} · %{fullData.name}<br>" + met + ": %{y:.3f}<extra></extra>")
        fig.update_layout(bargap=0.22, bargroupgap=0.08, yaxis_title=None, xaxis_title=None)
        if met == "R²":
            fig.update_yaxes(range=[max(0, d[met].min() - 0.02), 1])
        st.plotly_chart(gaya(fig, 430), width="stretch")
    with c2:
        st.markdown("**Rata-rata 7 emiten** · Table VIII")
        t = avg.reset_index()
        t["Model"] = t.Model.map(NAMA)
        st.dataframe(
            t[["Model", "MAPE (%)", "RMSE (×10⁻³)", "MAE (×10⁻³)", "R²"]],
            hide_index=True, width="stretch",
            column_config={
                "MAPE (%)": st.column_config.ProgressColumn("MAPE (%)", format="%.2f", min_value=0,
                                                            max_value=float(t["MAPE (%)"].max()) * 1.1),
                "RMSE (×10⁻³)": st.column_config.NumberColumn(format="%.3f"),
                "MAE (×10⁻³)": st.column_config.NumberColumn(format="%.3f"),
                "R²": st.column_config.NumberColumn(format="%.4f")},
        )
        best = avg["MAPE (%)"].idxmin()
        st.success(f"MAPE terendah: **{NAMA[best]}** ({angka(avg.loc[best, 'MAPE (%)'])}%). "
                   "Dihitung ulang dari file prediksi; identik dengan EVALUASI_baseline_DM_v3.")
    with st.expander("Tabel lengkap per emiten"):
        t = metrik.copy()
        t["Model"] = t.Model.map(NAMA)
        st.dataframe(t.round(4), hide_index=True, width="stretch")
        st.download_button("⬇️ Unduh CSV", t.to_csv(index=False).encode("utf-8-sig"),
                           "metrik_model.csv", "text/csv")

# ------------------------------------------------------------
# TAB 2 — AKTUAL VS PREDIKSI
# ------------------------------------------------------------
with tab2:
    judul("C", "Volatilitas aktual vs prediksi", "Volatilitas harian (std. log return 20 hari) untuk satu hari ke depan")
    c1, c2 = st.columns([2, 3])
    with c1:
        em = st.segmented_control("Emiten", EMITEN, default="ITMG", key="em") or "ITMG"
    with c2:
        tampil = st.multiselect("Model", MODELS, default=["GARCH(1,1)", "XGBoost", "LSTM"],
                                format_func=lambda m: NAMA[m])
    g = df[df.Emiten == em]
    mm = metrik[metrik.Emiten == em].set_index("Model")
    st.markdown('<div class="vp-pills">' + "".join(
        f'<span class="vp-pill" style="--c:{WARNA[m]}"><i></i>{NAMA[m]} · MAPE {angka(mm.loc[m, "MAPE (%)"])}% · '
        f'R² {angka(mm.loc[m, "R²"], 3)}</span>' for m in tampil) + "</div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.Date, y=g.Aktual, name="Aktual", line=dict(color=WARNA["Aktual"], width=2.4),
                             fill="tozeroy", fillcolor="rgba(59,111,224,0.07)"))
    for m in tampil:
        fig.add_trace(go.Scatter(x=g.Date, y=g[m], name=NAMA[m], line=dict(color=WARNA[m], width=1.6)))
    fig.update_layout(hovermode="x unified", yaxis_title="Volatilitas harian")
    fig.update_yaxes(range=[g[["Aktual"] + tampil].min().min() * 0.9, g[["Aktual"] + tampil].max().max() * 1.05])
    st.plotly_chart(gaya(fig, 470), width="stretch")

# ------------------------------------------------------------
# TAB 3 — DIEBOLD–MARIANO
# ------------------------------------------------------------
with tab3:
    judul("D", "Uji Diebold–Mariano", "Apakah perbedaan akurasi antar-model signifikan secara statistik?")
    loss = st.segmented_control("Fungsi loss", ["MSE", "MAE", "QLIKE"], default="MSE", key="loss") or "MSE"
    s = dm[dm.Loss == loss]
    urut_e = EMITEN + ["Gabungan"]
    pairs = list(dict.fromkeys(s.Pembanding))
    z = s.pivot(index="Emiten", columns="Pembanding", values="DM").reindex(index=urut_e, columns=pairs)
    p = s.pivot(index="Emiten", columns="Pembanding", values="p_value").reindex(index=urut_e, columns=pairs)
    kode = np.where(p < 0.01, 2 * np.sign(z), np.where(p < 0.05, np.sign(z), 0))
    bintang = np.where(p < 0.01, "**", np.where(p < 0.05, "*", ""))
    teks = z.round(2).astype(str) + bintang
    label_x = [x.replace("XGBoost", "XGB").replace("GARCH(1,1)", "GARCH").replace(" vs ", "<br>vs ") for x in pairs]
    fig = go.Figure(go.Heatmap(
        z=kode, x=label_x, y=urut_e, text=teks.values, customdata=p.values, texttemplate="%{text}",
        textfont=dict(size=13),
        colorscale=[[0, WARNA["XGBoost"]], [0.25, "#9DB6F0"], [0.5, "#EEF0F4"], [0.75, "#F4C3A0"], [1, WARNA["GARCH(1,1)"]]],
        zmin=-2, zmax=2, showscale=False, xgap=4, ygap=4,
        hovertemplate="%{y} · %{x}<br>DM = %{text}<br>p = %{customdata:.4f}<extra></extra>"))
    fig.update_layout(yaxis_autorange="reversed", xaxis_side="top")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(gaya(fig, 500), width="stretch")
    st.caption("DM < 0: model pertama lebih akurat. **Biru** = model pertama signifikan lebih akurat, "
               "**oranye** = model kedua signifikan lebih akurat, abu-abu = tidak signifikan. "
               "\\* p < 0,05 · \\*\\* p < 0,01 · koreksi Harvey–Leybourne–Newbold.")

# ------------------------------------------------------------
# TAB 4 — SHAP
# ------------------------------------------------------------
with tab4:
    judul("E", "Kontribusi fitur (SHAP)", "Fitur apa yang paling dipakai model GARCH–XGBoost?")
    c1, c2 = st.columns([3, 1], gap="large")
    with c2:
        kol = st.selectbox("Tampilkan", ["RATA-RATA"] + EMITEN,
                           format_func=lambda k: "Rata-rata 7 emiten" if k == "RATA-RATA" else k)
        don = go.Figure(go.Pie(labels=["Sentimen (X2)", "Teknikal & GARCH (X1)"], values=[shap_sent, 1 - shap_sent],
                               hole=0.66, marker=dict(colors=[UNGU, "#C9CDD5"]), sort=False,
                               textinfo="none", hovertemplate="%{label}: %{percent}<extra></extra>"))
        don.update_layout(showlegend=False, annotations=[dict(text=f"<b>{angka(shap_sent * 100)}%</b><br>sentimen",
                                                              showarrow=False, font_size=16)])
        st.plotly_chart(gaya(don, 240), width="stretch")
        st.caption("Porsi total |SHAP| rata-rata 7 emiten menurut kelompok fitur.")
    with c1:
        d = shap[["Fitur", kol]].sort_values(kol)
        d["Kelompok"] = np.where(d.Fitur.isin(SENTIMEN), "Sentimen (X2)", "Teknikal & GARCH (X1)")
        fig = px.bar(d, x=kol, y="Fitur", color="Kelompok", orientation="h", text=kol,
                     color_discrete_map={"Sentimen (X2)": UNGU, "Teknikal & GARCH (X1)": "#C9CDD5"})
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False, marker_line_width=0)
        fig.update_layout(xaxis_title="Rata-rata |SHAP|", yaxis_title=None)
        fig.update_xaxes(showgrid=True, gridcolor="rgba(120,130,150,.18)", griddash="dot")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(gaya(fig, 580), width="stretch")

# ------------------------------------------------------------
# TAB 5 — ABLASI SENTIMEN
# ------------------------------------------------------------
with tab5:
    judul("F", "Pengaruh sentimen: uji ablasi", "Model dilatih ulang dengan X1 saja vs X1 + X2")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Perubahan RMSE setelah fitur sentimen ditambahkan**")
        d = abl.copy()
        d["Model "] = d.Model.map(NAMA)
        d["Arah"] = np.where(d["Perubahan_RMSE_%"] < 0, "Membaik", "Memburuk")
        fig = px.bar(d, x="Perubahan_RMSE_%", y="Emiten", color="Arah", facet_col="Model ",
                     orientation="h", category_orders={"Emiten": EMITEN},
                     color_discrete_map={"Membaik": WARNA["LSTM"], "Memburuk": MERAH},
                     hover_data={"p_value": ":.3f", "DM": ":.2f"})
        fig.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>"))
        fig.add_vline(x=0, line_color="rgba(120,130,150,.6)")
        fig.update_traces(marker_line_width=0)
        fig.update_xaxes(title="Δ RMSE (%)", showgrid=True, gridcolor="rgba(120,130,150,.18)", griddash="dot")
        fig.update_yaxes(title=None)
        st.plotly_chart(gaya(fig, 430), width="stretch")
        if n_sig == 0:
            st.info(f"Tidak ada perubahan yang signifikan (0 dari {len(abl)}, uji DM p < 0,05).")
        else:
            st.warning(f"Signifikan: {n_sig} dari {len(abl)} (uji DM p < 0,05).")
    with c2:
        st.markdown("**Per kondisi pasar** · RMSE ×10⁻³")
        t = kondisi.copy()
        t["Model"] = t.Model.map(NAMA)
        t["Signifikan"] = np.where(t.p_value < 0.05, "✓ ya", "–")
        st.dataframe(
            t.drop(columns=["Dimensi"]), hide_index=True, width="stretch",
            column_config={"RMSE_X1 (1e-3)": st.column_config.NumberColumn("X1", format="%.3f"),
                           "RMSE_X1+X2 (1e-3)": st.column_config.NumberColumn("X1+X2", format="%.3f"),
                           "Perubahan_RMSE_%": st.column_config.NumberColumn("Δ RMSE", format="%+.2f%%"),
                           "DM": st.column_config.NumberColumn(format="%.2f"),
                           "p_value": st.column_config.NumberColumn("p", format="%.4f"),
                           "n_hari": st.column_config.NumberColumn("Hari")})
        for _, rr in kondisi[kondisi.p_value < 0.05].iterrows():
            st.error(f"Signifikan: **{NAMA[rr.Model]}** memburuk pada kondisi *{rr.Kondisi}* "
                     f"({angka(rr['Perubahan_RMSE_%'])}%, p = {fmt_p(rr.p_value)}).")
    with st.expander("Tabel ablasi per emiten"):
        st.dataframe(abl.round(4), hide_index=True, width="stretch")

# ------------------------------------------------------------
# TAB 6 — KARAKTERISTIK & VALIDASI
# ------------------------------------------------------------
with tab6:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        judul("A", "Karakteristik volatilitas", "Parameter GARCH(1,1)-t per emiten (data latih)")
        st.dataframe(
            garch, hide_index=True, width="stretch",
            column_config={"alpha+beta": st.column_config.ProgressColumn("α + β", format="%.4f", min_value=0, max_value=1),
                           "alpha": st.column_config.NumberColumn("α", format="%.4f"),
                           "beta": st.column_config.NumberColumn("β", format="%.4f"),
                           "Half-life (hari)": st.column_config.NumberColumn(format="%.1f")})
        st.caption("α + β mendekati 1 = volatility clustering kuat; half-life = hari hingga guncangan berkurang setengahnya.")
    with c2:
        judul("B", "Validasi IndoBERT", "Kesesuaian dengan empat jenis label acuan")
        st.dataframe(
            indobert, hide_index=True, width="stretch",
            column_config={"Akurasi": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1),
                           "MacroF1": st.column_config.ProgressColumn("Macro-F1", format="%.3f", min_value=0, max_value=1)})
        d = indobert.melt(id_vars="Acuan", value_vars=["Akurasi", "MacroF1"], var_name="Metrik", value_name="Nilai")
        fig = px.bar(d, x="Acuan", y="Nilai", color="Metrik", barmode="group", text="Nilai",
                     color_discrete_sequence=[WARNA["XGBoost"], WARNA["LSTM"]])
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_width=0)
        fig.update_layout(yaxis_range=[0, 1.05], xaxis_title=None, yaxis_title=None)
        st.plotly_chart(gaya(fig, 300), width="stretch")

    st.markdown("**Skor sentimen harian (IndoBERT)**")
    try:
        sh = baca(FILES["sent_harian"], repo, branch, parse_dates=["Date"])
        em2 = st.segmented_control("Emiten ", EMITEN, default="ADRO", key="em_sent") or "ADRO"
        g = sh[sh.Emiten == em2].sort_values("Date")
        g["Rata-rata 20 hari"] = g.Sentiment_Mean.rolling(20, min_periods=5).mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=g.Date, y=g.Sentiment_Mean, name="Sentimen harian",
                             marker_color=np.where(g.Sentiment_Mean >= 0, WARNA["LSTM"], MERAH), opacity=0.4,
                             marker_line_width=0))
        fig.add_trace(go.Scatter(x=g.Date, y=g["Rata-rata 20 hari"], name="Rata-rata 20 hari",
                                 line=dict(color=WARNA["Aktual"], width=2)))
        st.plotly_chart(gaya(fig, 330), width="stretch")
    except Exception:  # noqa: BLE001
        st.caption("Skor sentimen harian tidak dapat dibaca.")

with st.expander("ℹ️ Definisi dan sumber data"):
    st.markdown(f"""
- **Model hybrid**: XGBoost/LSTM memprediksi rasio ln(σₜ₊₁ / σ_GARCH), lalu dikalikan kembali dengan prakiraan GARCH(1,1)-t.
- **Naive**: volatilitas besok = volatilitas hari ini.
- **X1**: fitur teknikal, lag volatilitas, dan prakiraan GARCH. **X2**: {", ".join(SENTIMEN)}.
- **Sumber** (dibaca langsung dari [GitHub](https://github.com/{repo})):
""" + "\n".join(f"  - `{v.format(e='<EMITEN>')}`" for v in FILES.values()))

st.markdown('<div class="vp-foot">Ilda Nurida · Magister SIMT ITS · 2026 · data: '
            f'<a href="https://github.com/{repo}">github.com/{repo}</a></div>', unsafe_allow_html=True)
