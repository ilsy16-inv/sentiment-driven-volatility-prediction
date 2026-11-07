"""Halaman Investor: radar risiko H+1, simulator skenario berita, dan analisis historis."""
import io
import math
import urllib.request

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import EMITEN, FILES, MERAH, HIJAU, WARNA, angka, baca, gaya, judul, raw_url, sumber

repo, branch = sumber()

RISIKO = {"Low Risk": ("Rendah", HIJAU), "Medium Risk": ("Sedang", "#E0A100"), "High Risk": ("Tinggi", MERAH)}
NAMA_PT = {"ADRO": "Alamtri Resources (Adaro)", "ADMR": "Adaro Minerals", "ITMG": "Indo Tambangraya Megah",
           "PTBA": "Bukit Asam", "MEDC": "Medco Energi", "AKRA": "AKR Corporindo", "PGAS": "Perusahaan Gas Negara"}
FITUR = ["Log_Return", "Vol_Lag1", "Vol_Lag2", "Vol_Lag3", "Vol_MA5", "Vol_STD5", "MA5_Ratio", "MA10_Ratio",
         "MA20_Ratio", "RSI14", "Stoch_K", "Stoch_D", "Volume_ZScore", "R_Exit", "GARCH_Forecast",
         "Sentiment_Mean", "Sentiment_Dispersion", "News_Count", "Ada_Berita"]
SKENARIO = {  # skor rata-rata, dispersi, jumlah berita, ada berita
    "🔴 Berita negatif": (-0.8, 0.3, 3, 1),
    "⚪ Berita netral": (0.0, 0.3, 3, 1),
    "🟢 Berita positif": (0.8, 0.3, 3, 1),
    "∅ Tanpa berita": (0.0, 0.0, 0, 0),
}


def kategori(v: float, q33: float, q66: float) -> str:
    return "Low Risk" if v < q33 else ("Medium Risk" if v < q66 else "High Risk")


def tahunan(v: float) -> float:
    return v * math.sqrt(252) * 100


@st.cache_resource(ttl=3600, show_spinner="Memuat model GARCH–XGBoost dari GitHub ...")
def muat_model(repo: str, branch: str, e: str):
    with urllib.request.urlopen(raw_url(FILES["model_xgb"].format(e=e), repo, branch), timeout=60) as r:
        return joblib.load(io.BytesIO(r.read()))


@st.cache_data(ttl=3600, show_spinner=False)
def muat_artikel(repo: str, branch: str) -> pd.DataFrame:
    a = baca(FILES["artikel"], repo, branch,
             usecols=["Sumber", "Kode_Emiten", "Headline", "Tanggal", "Link", "Label_Sentimen", "Skor_Sentimen"])
    a["Tanggal"] = pd.to_datetime(a["Tanggal"], errors="coerce")
    return a.dropna(subset=["Tanggal"])


@st.cache_data(ttl=3600, show_spinner=False)
def muat_historis(repo: str, branch: str) -> pd.DataFrame:
    d = baca(FILES["gabungan"], repo, branch, parse_dates=["Date"]).sort_values(["Emiten", "Date"])
    d["AbsReturn_Besok"] = d.groupby("Emiten")["Log_Return"].shift(-1).abs() * 100
    d["dVol"] = (d["Volatilitas_Target"] / d["Vol_Lag1"] - 1) * 100
    d["Kelas"] = np.select([d.Ada_Berita == 0, d.Sentiment_Mean <= -0.1, d.Sentiment_Mean >= 0.1],
                           ["Tanpa berita", "Negatif", "Positif"], "Netral")
    return d


try:
    fc = baca(FILES["forecast_h1"], repo, branch)
    ver = baca(FILES["verifikasi_h1"], repo, branch)
    sent = baca(FILES["sent_harian"], repo, branch, parse_dates=["Date"])
except Exception as err:  # noqa: BLE001
    st.error(f"Gagal membaca data dari GitHub ({repo}, branch {branch}).\n\n{err}")
    st.stop()

fc = fc.set_index("Emiten").reindex(EMITEN)
basis = pd.to_datetime(fc["Basis (hari terakhir data)"].iloc[0])
n_tinggi = int((fc["Kategori_Risiko_XGB"] == "High Risk").sum())

# ============================================================
# HEADER
# ============================================================
st.markdown(f"""
<div class="vp-hero">
  <div class="vp-eyebrow">Untuk investor &amp; manajemen risiko · model hybrid GARCH–XGBoost</div>
  <h1>Radar risiko volatilitas saham energi LQ45</h1>
  <p class="vp-lead">Perkiraan seberapa besar harga saham akan bergerak besok (volatilitas H+1), dibandingkan dengan
  kebiasaan masing-masing saham, serta seberapa jauh berita positif/negatif mengubah perkiraan itu.
  Data terakhir: <b>{basis:%d %B %Y}</b>.</p>
  <div class="vp-chips">
    <span class="vp-chip"><i style="background:{MERAH}"></i>{n_tinggi} saham berisiko tinggi</span>
    <span class="vp-chip"><i style="background:{WARNA['XGBoost']}"></i>Error prediksi H+1 rata-rata {angka(ver['Err_XGB_%'].mean())}%</span>
    <span class="vp-chip">Bukan rekomendasi beli/jual</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 1. RADAR RISIKO H+1
# ============================================================
judul("1", "Radar risiko besok (H+1)",
      f"Prediksi volatilitas harian untuk hari bursa setelah {basis:%d %b %Y}, dibanding riwayat masing-masing saham")

kartu = []
for e in EMITEN:
    r = fc.loc[e]
    v, q33, q66 = r["Vol_Prediksi_XGBoost"], r["Threshold_Q33"], r["Threshold_Q66"]
    label, warna = RISIKO[r["Kategori_Risiko_XGB"]]
    top = max(q66 * 1.5, v * 1.1)
    w1, w2 = q33 / top * 100, (q66 - q33) / top * 100
    pos = min(v / top * 100, 99)
    s5 = sent[(sent.Emiten == e) & (sent.Date <= basis)].tail(5)
    nb = int(s5.News_Count.sum())
    skor = (s5.Sentiment_Mean * s5.News_Count).sum() / nb if nb else 0.0
    ikon = "🔴" if skor <= -0.1 else ("🟢" if skor >= 0.1 else "⚪")
    teks_sent = (f"{ikon} {nb} berita dalam 5 hari bursa terakhir, skor rata-rata {angka(skor)}"
                 if nb else "∅ Tidak ada berita dalam 5 hari bursa terakhir")
    kartu.append(f"""<div class="vp-rc" style="--r:{warna}">
      <div class="top"><div><div class="tk">{e}</div><div class="nm">{NAMA_PT[e]}</div></div>
      <span class="vp-tag">Risiko {label}</span></div>
      <div class="v">{angka(v * 100)}% <small>/ hari</small></div>
      <div class="ann">≈ {angka(tahunan(v), 1)}% per tahun (disetahunkan)</div>
      <div class="vp-band"><span style="width:{w1:.1f}%;background:{HIJAU}44"></span><span style="width:{w2:.1f}%;background:#E0A10044"></span>
        <span style="flex:1;background:{MERAH}44"></span><span class="mk" style="left:calc({pos:.1f}% - 2px)"></span></div>
      <div class="vp-scale"><span>rendah</span><span>sedang</span><span>tinggi</span></div>
      <div class="snt">{teks_sent}</div></div>""")
st.markdown(f'<div class="vp-radar">{"".join(kartu)}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="vp-disc">Batas risiko per saham memakai kuantil 33% dan 66% volatilitas historisnya sendiri: '
    '“tinggi” berarti lebih bergejolak dari biasanya <i>untuk saham itu</i>, bukan dibanding saham lain. '
    'Volatilitas 2% per hari kira-kira berarti harga lazim bergerak ±2% dalam sehari.</div>', unsafe_allow_html=True)

with st.expander("Seberapa tepat prediksi H+1 sebelumnya? (verifikasi 1 Juli 2026)"):
    t = ver[["Emiten", "Aktual", "XGBoost", "GARCH(1,1)", "Naive", "Err_XGB_%", "Err_GARCH_%", "Err_Naive_%"]].copy()
    st.dataframe(t, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ["Aktual", "XGBoost", "GARCH(1,1)", "Naive"]}
                 | {c: st.column_config.NumberColumn(c.replace("Err_", "Error ").replace("_%", " (%)"), format="%.2f")
                    for c in ["Err_XGB_%", "Err_GARCH_%", "Err_Naive_%"]})
    st.caption(f"Rata-rata error absolut: GARCH–XGBoost {angka(ver['Err_XGB_%'].mean())}% · "
               f"GARCH {angka(ver['Err_GARCH_%'].mean())}% · Naive {angka(ver['Err_Naive_%'].mean())}%.")

# ============================================================
# 2. SIMULATOR SKENARIO BERITA
# ============================================================
st.write("")
judul("2", "Simulator: kalau beritanya negatif atau positif?",
      "Model GARCH–XGBoost dijalankan ulang dengan skenario sentimen berita yang Anda pilih")

c1, c2 = st.columns([1, 2], gap="large")
with c1:
    em = st.selectbox("Saham", EMITEN, format_func=lambda e: f"{e} · {NAMA_PT[e]}")
    te = baca(FILES["test"].format(e=em), repo, branch, parse_dates=["Date"])
    sc = baca(FILES["scaler"].format(e=em), repo, branch, index_col=0)
    tgl = st.select_slider("Tanggal acuan (hari berita)", options=list(te.Date.dt.date), value=te.Date.dt.date.iloc[-1],
                           format_func=lambda d: d.strftime("%d %b %Y"))
    pilihan = st.segmented_control("Skenario berita", list(SKENARIO) + ["🎛️ Atur sendiri"], default="🔴 Berita negatif") \
        or "🔴 Berita negatif"
    if pilihan == "🎛️ Atur sendiri":
        sm = st.slider("Skor sentimen rata-rata", -1.0, 1.0, -0.5, 0.05, help="−1 sangat negatif, +1 sangat positif")
        nc = st.slider("Jumlah berita hari itu", 0, int(sc.loc["News_Count", "max"]), 3)
        sd = st.slider("Perbedaan nada antar-berita (dispersi)", 0.0, float(sc.loc["Sentiment_Dispersion", "max"]), 0.3, 0.05)
        ada = 1 if nc > 0 else 0
    else:
        sm, sd, nc, ada = SKENARIO[pilihan]

try:
    model = muat_model(repo, branch, em)
except Exception as err:  # noqa: BLE001
    st.error(f"Model untuk {em} tidak dapat dimuat: {err}")
    st.stop()

kolom = list(getattr(model, "feature_names_in_", FITUR))
inv = lambda v, k: v * (sc.loc[k, "max"] - sc.loc[k, "min"]) + sc.loc[k, "min"]  # noqa: E731
fwd = lambda v, k: float(np.clip((v - sc.loc[k, "min"]) / (sc.loc[k, "max"] - sc.loc[k, "min"]), 0, 1))  # noqa: E731
baris = te[te.Date.dt.date == tgl].iloc[[0]]
garch_f = float(inv(baris["GARCH_Forecast"].iloc[0], "GARCH_Forecast"))
aktual = float(inv(baris["Volatilitas_Target"].iloc[0], "Volatilitas_Target"))
asli = {k: float(inv(baris[k].iloc[0], k)) for k in ["Sentiment_Mean", "Sentiment_Dispersion", "News_Count", "Ada_Berita"]}


def prediksi(skor, disp, jml, ada_):
    x = baris[kolom].copy()
    x["Sentiment_Mean"], x["Sentiment_Dispersion"] = fwd(skor, "Sentiment_Mean"), fwd(disp, "Sentiment_Dispersion")
    x["News_Count"], x["Ada_Berita"] = fwd(jml, "News_Count"), fwd(ada_, "Ada_Berita")
    return float(garch_f * np.exp(model.predict(x[kolom])[0] / 100))


v_sken = prediksi(sm, sd, nc, ada)
v_asli = prediksi(asli["Sentiment_Mean"], asli["Sentiment_Dispersion"], asli["News_Count"], asli["Ada_Berita"])
v_netral = prediksi(0.0, 0.3, max(nc, 3), 1)
q33, q66 = float(fc.loc[em, "Threshold_Q33"]), float(fc.loc[em, "Threshold_Q66"])
kat = kategori(v_sken, q33, q66)
lbl, wr = RISIKO[kat]
geser = (v_sken / v_asli - 1) * 100

with c2:
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"<div class='vp-kpi' style='--c:{wr}'><div class='lbl'><i></i>Prediksi skenario</div>"
                f"<div class='big'>{angka(v_sken * 100)}%</div><div class='sub'>per hari · risiko <b>{lbl}</b></div></div>",
                unsafe_allow_html=True)
    k2.markdown(f"<div class='vp-kpi' style='--c:{WARNA['XGBoost']}'><div class='lbl'><i></i>Dengan berita sebenarnya</div>"
                f"<div class='big'>{angka(v_asli * 100)}%</div><div class='sub'>skor {angka(asli['Sentiment_Mean'])} · "
                f"{int(round(asli['News_Count']))} berita</div></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='vp-kpi' style='--c:#1F2433'><div class='lbl'><i></i>Volatilitas aktual besoknya</div>"
                f"<div class='big'>{angka(aktual * 100)}%</div><div class='sub'>GARCH saja: {angka(garch_f * 100)}%</div></div>",
                unsafe_allow_html=True)

    # kurva sensitivitas: skor sentimen -1..+1
    xs = np.round(np.linspace(-1, 1, 41), 3)
    jml_k = max(nc, 1)
    ys = [prediksi(x, sd if sd > 0 else 0.3, jml_k, 1) * 100 for x in xs]
    y_lo, y_hi = min(ys + [v_sken * 100, q33 * 100]) * 0.97, max(ys + [v_sken * 100, q66 * 100]) * 1.03
    fig = go.Figure()
    for a, b, col, nm in [(y_lo, q33 * 100, HIJAU, "Risiko rendah"), (q33 * 100, q66 * 100, "#E0A100", "Risiko sedang"),
                          (q66 * 100, y_hi, MERAH, "Risiko tinggi")]:
        if b > a:
            fig.add_hrect(y0=a, y1=b, fillcolor=col, opacity=0.07, line_width=0,
                          annotation_text=nm, annotation_position="top left", annotation_font_size=10)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Prediksi ({jml_k} berita)",
                             line=dict(color=WARNA["XGBoost"], width=3, shape="hv"),
                             hovertemplate="Skor %{x:.2f} → %{y:.3f}% per hari<extra></extra>"))
    fig.add_hline(y=aktual * 100, line_dash="dot", line_color="#1F2433",
                  annotation_text="aktual besoknya", annotation_position="bottom right", annotation_font_size=10)
    if ada:
        fig.add_trace(go.Scatter(x=[sm], y=[v_sken * 100], mode="markers", name="Skenario Anda",
                                 marker=dict(size=14, color=wr, line=dict(color="white", width=2))))
    fig.update_layout(xaxis_title="Skor sentimen berita (−1 negatif … +1 positif)", yaxis_title="Volatilitas besok (% per hari)",
                      showlegend=True)
    fig.update_yaxes(range=[y_lo, y_hi])
    st.plotly_chart(gaya(fig, 360), width="stretch")

rentang = (max(ys) / min(ys) - 1) * 100
pesan = (f"Untuk {em} pada {tgl:%d %b %Y}, mengubah berita dari sangat negatif ke sangat positif hanya menggeser "
         f"prediksi volatilitas sekitar **{angka(rentang, 1)}%** (relatif). Skenario Anda mengubah prediksi "
         f"**{'+' if geser >= 0 else '−'}{angka(abs(geser), 2)}%** dibanding berita sebenarnya.")
st.info("💡 " + pesan + " Sinyal utama model datang dari pola volatilitas beberapa hari terakhir dan prakiraan GARCH, "
        "bukan dari nada berita. Ini sejalan dengan hasil uji ablasi penelitian (penambahan fitur sentimen tidak "
        "mengubah akurasi secara signifikan).")

# ============================================================
# 3. ANALISIS HISTORIS
# ============================================================
st.write("")
judul("3", "Apa yang biasanya terjadi setelah berita negatif?",
      "Pergerakan harga hari berikutnya menurut nada berita hari ini, data 2022–2026")
hist = muat_historis(repo, branch)
c1, c2 = st.columns([1, 3], gap="large")
with c1:
    em_h = st.selectbox("Saham ", ["Semua saham"] + EMITEN)
    ukuran = st.radio("Ukuran", ["Pergerakan harga besok (|return|)", "Perubahan volatilitas 20 hari"],
                      help="|return| = besar perubahan harga harian tanpa melihat arah naik/turun.")
d = hist if em_h == "Semua saham" else hist[hist.Emiten == em_h]
kol_u = "AbsReturn_Besok" if ukuran.startswith("Pergerakan") else "dVol"
urut = ["Negatif", "Netral", "Positif", "Tanpa berita"]
warna_k = {"Negatif": MERAH, "Netral": "#9AA0AA", "Positif": HIJAU, "Tanpa berita": "#C9CDD5"}
g = d.dropna(subset=[kol_u]).groupby("Kelas")[kol_u].agg(["mean", "std", "count"]).reindex(urut)
g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
with c2:
    fig = go.Figure(go.Bar(
        x=[f"{k}<br><span style='font-size:10px'>n = {int(n)}</span>" for k, n in zip(g.index, g["count"].fillna(0))],
        y=g["mean"], marker_color=[warna_k[k] for k in g.index], marker_line_width=0,
        error_y=dict(type="data", array=g["ci"], color="rgba(60,60,70,.6)", thickness=1.5, width=6),
        text=[f"{angka(v, 2)}%" if pd.notna(v) else "" for v in g["mean"]], textposition="outside",
        hovertemplate="%{x}<br>rata-rata %{y:.3f}%<extra></extra>"))
    fig.update_layout(yaxis_title=("Rata-rata |return| besok (%)" if kol_u == "AbsReturn_Besok"
                                   else "Rata-rata perubahan volatilitas (%)"), showlegend=False)
    st.plotly_chart(gaya(fig, 360), width="stretch")


def uji(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Uji Welch dua sisi (pendekatan normal)."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 5 or len(b) < 5:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / math.sqrt(a.var() / len(a) + b.var() / len(b))
    return t, math.erfc(abs(t) / math.sqrt(2))


t_stat, p_val = uji(d.loc[d.Kelas == "Negatif", kol_u], d.loc[d.Kelas == "Netral", kol_u])
if pd.notna(p_val):
    selisih = g.loc["Negatif", "mean"] - g.loc["Netral", "mean"]
    sig = "berbeda signifikan" if p_val < 0.05 else "belum berbeda signifikan"
    st.markdown(f"Setelah hari berita **negatif**, rata-rata nilainya **{'+' if selisih >= 0 else '−'}{angka(abs(selisih), 2)} poin** "
                f"dibanding hari berita netral ({sig}, p = {angka(p_val, 3)}).")
st.markdown('<div class="vp-disc">Ini korelasi, bukan sebab-akibat: berita negatif sering muncul saat pasar memang sedang '
            'bergejolak. Garis kecil di atas batang = selang kepercayaan 95%. Kelas: negatif ≤ −0,1; positif ≥ +0,1; '
            'di antaranya netral.</div>', unsafe_allow_html=True)

# ============================================================
# 4. BERITA TERBARU
# ============================================================
st.write("")
judul("4", "Berita terbaru dan sentimennya", "Judul berita yang dinilai IndoBERT, sampai tanggal data terakhir")
try:
    art = muat_artikel(repo, branch)
    c1, c2 = st.columns([1, 3])
    with c1:
        em_b = st.selectbox("Saham  ", EMITEN, index=EMITEN.index(em))
        n_b = st.slider("Jumlah berita", 5, 30, 10)
        a_em = art[art.Kode_Emiten == em_b]
        cnt = a_em[a_em.Tanggal > basis - pd.Timedelta(days=30)].Label_Sentimen.value_counts()
        st.markdown(f"**30 hari terakhir**  \n🟢 {cnt.get('Positif', 0)} positif · ⚪ {cnt.get('Netral', 0)} netral · "
                    f"🔴 {cnt.get('Negatif', 0)} negatif")
    with c2:
        t = a_em.sort_values("Tanggal", ascending=False).head(n_b).copy()
        t["Sentimen"] = t.Label_Sentimen.map({"Positif": "🟢 Positif", "Negatif": "🔴 Negatif", "Netral": "⚪ Netral"})
        st.dataframe(t[["Tanggal", "Sentimen", "Headline", "Sumber", "Link"]], hide_index=True, width="stretch",
                     column_config={"Tanggal": st.column_config.DateColumn(format="DD MMM YYYY"),
                                    "Headline": st.column_config.TextColumn("Judul berita", width="large"),
                                    "Link": st.column_config.LinkColumn("Tautan", display_text="buka ↗")})
except Exception:  # noqa: BLE001
    st.caption("Data berita tidak dapat dibaca.")

st.markdown('<div class="vp-foot">Dashboard ini adalah alat bantu analisis hasil penelitian, bukan rekomendasi investasi. '
            'Volatilitas mengukur besar pergerakan harga, bukan arah naik/turunnya.</div>', unsafe_allow_html=True)
