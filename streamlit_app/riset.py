"""Halaman hasil penelitian (revisi konferensi)."""
from common import (BRANCH, FILES, MODELS, NAMA, REPO, SENTIMEN, TARGET_MAPE, WARNA, MERAH, HIJAU, UNGU,
                    EMITEN, angka, baca, fmt_p, gaya, hitung_metrik, judul, muat_prediksi, sumber)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

repo, branch = sumber()

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
