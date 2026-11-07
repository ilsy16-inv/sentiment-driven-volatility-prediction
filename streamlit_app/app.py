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
WARNA = {"Naive": "#B0B0B0", "GARCH(1,1)": "#E8833A",
         "XGBoost": "#2F6FDE", "LSTM": "#2BA37A", "Aktual": "#222222"}
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

st.set_page_config(page_title="Volatilitas Saham Energi LQ45", page_icon="📈", layout="wide")


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


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Sumber data")
    repo = st.text_input("Repositori GitHub", REPO)
    branch = st.text_input("Branch", BRANCH)
    st.caption(f"[Buka repositori](https://github.com/{repo}/tree/{branch})")
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
shap_total = shap["RATA-RATA"].sum()
shap_sent = shap[shap.Fitur.isin(SENTIMEN)]["RATA-RATA"].sum() / shap_total
dm_mse_gab = dm[(dm.Loss == "MSE") & (dm.Emiten == "Gabungan")].set_index("Pembanding")

# ============================================================
# HEADER
# ============================================================
st.title("Prediksi Volatilitas Return Saham Sektor Energi LQ45")
st.markdown(
    "**Hybrid GARCH–XGBoost dan GARCH–LSTM dengan fitur sentimen berita (IndoBERT)** · "
    f"periode uji {df.Date.min():%d %b %Y} – {df.Date.max():%d %b %Y} · "
    f"{df.Emiten.nunique()} emiten · {len(df):,} hari-emiten".replace(",", ".")
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("MAPE GARCH–XGBoost", f"{angka(avg.loc['XGBoost', 'MAPE (%)'])}%",
          f"{angka(avg.loc['XGBoost', 'MAPE (%)'] - avg.loc['GARCH(1,1)', 'MAPE (%)'])} poin vs GARCH",
          delta_color="inverse")
k2.metric("MAPE GARCH–LSTM", f"{angka(avg.loc['LSTM', 'MAPE (%)'])}%",
          f"{angka(avg.loc['LSTM', 'MAPE (%)'] - avg.loc['GARCH(1,1)', 'MAPE (%)'])} poin vs GARCH",
          delta_color="inverse")
k3.metric("MAPE GARCH(1,1)", f"{angka(avg.loc['GARCH(1,1)', 'MAPE (%)'])}%", "baseline", delta_color="off")
k4.metric("Kontribusi SHAP sentimen", f"{angka(shap_sent * 100)}%", "GARCH–XGBoost", delta_color="off")

r = dm_mse_gab.loc["XGBoost vs GARCH(1,1)"]
st.info(
    f"**Temuan utama.** Kedua model hybrid mencapai MAPE jauh di bawah target {TARGET_MAPE:.0f}%. "
    f"Namun uji Diebold–Mariano gabungan (loss MSE) menunjukkan GARCH(1,1) masih lebih akurat daripada "
    f"GARCH–XGBoost (DM = {angka(r.DM)}, p = {fmt_p(r.p_value)}). Fitur sentimen menyumbang "
    f"{angka(shap_sent * 100)}% kontribusi SHAP, tetapi pada uji ablasi "
    f"{int((abl.p_value < 0.05).sum())} dari {len(abl)} kombinasi model–emiten berubah signifikan."
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Performa model", "📈 Aktual vs prediksi", "⚖️ Uji Diebold–Mariano",
    "🧩 Kontribusi fitur (SHAP)", "📰 Pengaruh sentimen", "🔎 Karakteristik & validasi",
])

# ------------------------------------------------------------
# TAB 1 — PERFORMA
# ------------------------------------------------------------
with tab1:
    c1, c2 = st.columns([3, 2])
    with c1:
        met = st.selectbox("Metrik", ["MAPE (%)", "RMSE (×10⁻³)", "MAE (×10⁻³)", "R²"])
        d = metrik[metrik.Emiten != "Rata-rata"].copy()
        d["Model "] = d.Model.map(NAMA)
        fig = px.bar(d, x="Emiten", y=met, color="Model ", barmode="group",
                     color_discrete_map={NAMA[k]: v for k, v in WARNA.items() if k in NAMA},
                     category_orders={"Emiten": EMITEN, "Model ": [NAMA[m] for m in MODELS]})
        if met == "R²":
            fig.update_yaxes(range=[max(0, d[met].min() - 0.02), 1])
        fig.update_layout(height=420, legend_title=None, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Rata-rata 7 emiten** (Table VIII)")
        t = avg.reset_index()
        t["Model"] = t.Model.map(NAMA)
        st.dataframe(
            t[["Model", "RMSE (×10⁻³)", "MAE (×10⁻³)", "MAPE (%)", "R²"]],
            hide_index=True, width="stretch",
            column_config={"RMSE (×10⁻³)": st.column_config.NumberColumn(format="%.3f"),
                           "MAE (×10⁻³)": st.column_config.NumberColumn(format="%.3f"),
                           "MAPE (%)": st.column_config.NumberColumn(format="%.2f"),
                           "R²": st.column_config.NumberColumn(format="%.4f")},
        )
        st.caption("Dihitung ulang dari file prediksi; identik dengan EVALUASI_baseline_DM_v3.")
    with st.expander("Tabel lengkap per emiten"):
        t = metrik.copy()
        t["Model"] = t.Model.map(NAMA)
        st.dataframe(t.round(4), hide_index=True, width="stretch")
        st.download_button("Unduh CSV", t.to_csv(index=False).encode("utf-8-sig"),
                           "metrik_model.csv", "text/csv")

# ------------------------------------------------------------
# TAB 2 — AKTUAL VS PREDIKSI
# ------------------------------------------------------------
with tab2:
    c1, c2 = st.columns([1, 3])
    with c1:
        em = st.selectbox("Emiten", EMITEN, index=EMITEN.index("ITMG"))
        tampil = st.multiselect("Model", MODELS, default=["GARCH(1,1)", "XGBoost", "LSTM"],
                                format_func=lambda m: NAMA[m])
        g = df[df.Emiten == em]
        st.dataframe(metrik[metrik.Emiten == em].assign(Model=lambda x: x.Model.map(NAMA))
                     [["Model", "MAPE (%)", "R²"]].round(3), hide_index=True, width="stretch")
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g.Date, y=g.Aktual, name="Aktual",
                                 line=dict(color=WARNA["Aktual"], width=2.2)))
        for m in tampil:
            fig.add_trace(go.Scatter(x=g.Date, y=g[m], name=NAMA[m],
                                     line=dict(color=WARNA[m], width=1.4)))
        fig.update_layout(height=460, hovermode="x unified", yaxis_title="Volatilitas harian",
                          legend=dict(orientation="h", y=1.08), margin=dict(t=30, b=10))
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------
# TAB 3 — DIEBOLD–MARIANO
# ------------------------------------------------------------
with tab3:
    loss = st.radio("Fungsi loss", ["MSE", "MAE", "QLIKE"], horizontal=True)
    s = dm[dm.Loss == loss]
    urut_e = EMITEN + ["Gabungan"]
    pairs = list(dict.fromkeys(s.Pembanding))
    z = s.pivot(index="Emiten", columns="Pembanding", values="DM").reindex(index=urut_e, columns=pairs)
    p = s.pivot(index="Emiten", columns="Pembanding", values="p_value").reindex(index=urut_e, columns=pairs)
    warna = np.where(p < 0.05, np.sign(z), 0)  # -1: model pertama lebih akurat, +1: model kedua
    teks = z.round(2).astype(str) + "<br>p=" + p.round(3).astype(str)
    fig = go.Figure(go.Heatmap(
        z=warna, x=pairs, y=urut_e, text=teks.values, texttemplate="%{text}",
        colorscale=[[0, WARNA["XGBoost"]], [0.5, "#EEEEEE"], [1, WARNA["GARCH(1,1)"]]],
        zmin=-1, zmax=1, showscale=False, xgap=3, ygap=3,
        hovertemplate="%{y} · %{x}<br>DM = %{text}<extra></extra>"))
    fig.update_layout(height=480, yaxis_autorange="reversed", margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("DM < 0: model pertama lebih akurat. **Biru** = model pertama signifikan lebih akurat, "
               "**oranye** = model kedua signifikan lebih akurat, abu-abu = tidak berbeda signifikan (α = 5%). "
               "Uji DM dengan koreksi Harvey–Leybourne–Newbold.")

# ------------------------------------------------------------
# TAB 4 — SHAP
# ------------------------------------------------------------
with tab4:
    c1, c2 = st.columns([3, 1])
    with c2:
        kol = st.selectbox("Tampilkan", ["RATA-RATA"] + EMITEN,
                           format_func=lambda k: "Rata-rata 7 emiten" if k == "RATA-RATA" else k)
        st.metric("Porsi fitur sentimen (X2)", f"{angka(shap_sent * 100)}%")
        st.metric("Porsi teknikal & GARCH (X1)", f"{angka((1 - shap_sent) * 100)}%")
    with c1:
        d = shap[["Fitur", kol]].sort_values(kol)
        d["Kelompok"] = np.where(d.Fitur.isin(SENTIMEN), "Sentimen (X2)", "Teknikal & GARCH (X1)")
        fig = px.bar(d, x=kol, y="Fitur", color="Kelompok", orientation="h",
                     color_discrete_map={"Sentimen (X2)": WARNA["XGBoost"], "Teknikal & GARCH (X1)": "#B8B8B8"})
        fig.update_layout(height=560, xaxis_title="Rata-rata |SHAP|", yaxis_title=None,
                          legend_title=None, legend=dict(orientation="h", y=1.05), margin=dict(t=30, b=10))
        st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------
# TAB 5 — ABLASI SENTIMEN
# ------------------------------------------------------------
with tab5:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Perubahan RMSE setelah fitur sentimen ditambahkan** (X1 → X1+X2)")
        d = abl.copy()
        d["Model "] = d.Model.map(NAMA)
        d["Arah"] = np.where(d["Perubahan_RMSE_%"] < 0, "Membaik", "Memburuk")
        fig = px.bar(d, x="Perubahan_RMSE_%", y="Emiten", color="Arah", facet_col="Model ",
                     orientation="h", category_orders={"Emiten": EMITEN},
                     color_discrete_map={"Membaik": WARNA["XGBoost"], "Memburuk": "#D9534F"},
                     hover_data={"p_value": ":.3f", "DM": ":.2f"})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        fig.add_vline(x=0, line_color="#666")
        fig.update_layout(height=420, legend_title=None, xaxis_title="Δ RMSE (%)",
                          xaxis2_title="Δ RMSE (%)", margin=dict(t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Signifikan (p < 0,05): {int((abl.p_value < 0.05).sum())} dari {len(abl)}.")
    with c2:
        st.markdown("**Per kondisi pasar**")
        t = kondisi.copy()
        t["Model"] = t.Model.map(NAMA)
        st.dataframe(t.drop(columns=["Dimensi"]).round(4), hide_index=True, width="stretch",
                     column_config={"p_value": st.column_config.NumberColumn(format="%.4f"),
                                    "Perubahan_RMSE_%": st.column_config.NumberColumn(format="%+.2f%%")})
        sig = kondisi[kondisi.p_value < 0.05]
        for _, rr in sig.iterrows():
            st.caption(f"Signifikan: {NAMA[rr.Model]} · {rr.Kondisi} "
                       f"({angka(rr['Perubahan_RMSE_%'])}%, p = {fmt_p(rr.p_value)}).")
    with st.expander("Tabel ablasi per emiten"):
        st.dataframe(abl.round(4), hide_index=True, width="stretch")

# ------------------------------------------------------------
# TAB 6 — KARAKTERISTIK & VALIDASI
# ------------------------------------------------------------
with tab6:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Parameter GARCH(1,1)-t per emiten** (data latih)")
        st.dataframe(garch, hide_index=True, width="stretch")
        st.caption("α + β mendekati 1 = volatility clustering kuat; half-life = hari hingga guncangan "
                   "berkurang setengahnya.")
    with c2:
        st.markdown("**Validasi IndoBERT terhadap label acuan**")
        st.dataframe(indobert, hide_index=True, width="stretch")
        d = indobert.melt(id_vars="Acuan", value_vars=["Akurasi", "MacroF1"], var_name="Metrik", value_name="Nilai")
        fig = px.bar(d, x="Acuan", y="Nilai", color="Metrik", barmode="group",
                     color_discrete_sequence=[WARNA["XGBoost"], WARNA["GARCH(1,1)"]])
        fig.update_layout(height=300, yaxis_range=[0, 1], legend_title=None, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Skor sentimen harian (IndoBERT)**")
    try:
        sh = baca(FILES["sent_harian"], repo, branch, parse_dates=["Date"])
        em2 = st.selectbox("Emiten ", EMITEN, key="em_sent")
        g = sh[sh.Emiten == em2].sort_values("Date")
        g["Rata-rata 20 hari"] = g.Sentiment_Mean.rolling(20, min_periods=5).mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=g.Date, y=g.Sentiment_Mean, name="Sentimen harian",
                             marker_color=np.where(g.Sentiment_Mean >= 0, WARNA["LSTM"], "#D9534F"), opacity=0.45))
        fig.add_trace(go.Scatter(x=g.Date, y=g["Rata-rata 20 hari"], name="Rata-rata 20 hari",
                                 line=dict(color=WARNA["Aktual"], width=1.8)))
        fig.update_layout(height=320, legend=dict(orientation="h", y=1.1), margin=dict(t=30, b=10))
        st.plotly_chart(fig, width="stretch")
    except Exception:  # noqa: BLE001
        st.caption("Skor sentimen harian tidak dapat dibaca.")

with st.expander("ℹ️ Definisi dan sumber data"):
    st.markdown(f"""
- **Model hybrid**: XGBoost/LSTM memprediksi rasio ln(σₜ₊₁ / σ_GARCH), lalu dikalikan kembali dengan prakiraan GARCH(1,1)-t.
- **Naive**: volatilitas besok = volatilitas hari ini.
- **X1**: fitur teknikal, lag volatilitas, dan prakiraan GARCH. **X2**: {", ".join(SENTIMEN)}.
- **Sumber** (dibaca langsung dari [GitHub](https://github.com/{repo})):
""" + "\n".join(f"  - `{v.format(e='<EMITEN>')}`" for v in FILES.values()))
