# ==============================================================================
# SIAPKAN DATA DASHBOARD — Tesis Ilda Nurida (6032251068)
# ------------------------------------------------------------------------------
# Mengumpulkan file hasil notebook yang tersebar di banyak folder, menyeragamkan
# nama kolomnya, lalu menyimpannya dengan nama bernomor ke SATU folder:
#
#     Dashboard Penelitian\dashboard_app\data_final\
#
# Aman dijalankan berulang kali: file sumber TIDAK diubah, hanya dibaca.
# Jalankan di notebook baru (mis. PREP_04_data_dashboard.ipynb) — tempel seluruh
# isi file ini ke satu sel, lalu Shift+Enter.
# ==============================================================================
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------
# 0. LOKASI — cukup ubah ROOT bila folder tesis dipindah
# ------------------------------------------------------------------------------
ROOT = Path(r'E:\DATA PENTING ILDA NURIDA\DOKUMEN\KULIAH\S2 2025\THESIS'
            r'\Penelitian\TESIS_PEMODELAN PREDIKSI VOLATILITAS RETURN SAHAM_2026')
OUT  = ROOT / 'Dashboard Penelitian' / 'dashboard_app' / 'data_final'
OUT.mkdir(parents=True, exist_ok=True)

TABEL = ROOT / '06_dokumen_tesis' / 'tabel'
URUTAN_EMITEN = ['ADRO', 'ADMR', 'ITMG', 'PTBA', 'MEDC', 'AKRA', 'PGAS']

# Semua file SUMBER dikumpulkan di sini, supaya mudah diganti bila ada versi baru
SUMBER = {
    'meta_teknikal'  : ROOT / '03_dataset' / 'processed' / 'technical_features_meta.json',
    'teknikal'       : ROOT / '03_dataset' / 'processed' / 'technical_features.csv',
    'sentimen_harian': ROOT / '03_dataset' / 'processed' / 'skor_sentimen_harian.csv',
    'berita_artikel' : ROOT / '02_preprocessing' / 'sentimen_indobert' / 'output_sentimen 1'
                            / 'skor_sentimen_per_artikel.csv',
    'dataset_model'  : ROOT / '03_dataset' / 'final' / 'dataset_gabungan_X1X2Y.csv',
    'split'          : TABEL / 'tabel_split_train_test.csv',
    'eval_xgb'       : TABEL / 'tabel_evaluasi_xgboost.csv',
    'eval_lstm'      : TABEL / 'tabel_evaluasi_lstm.csv',
    'ablasi'         : TABEL / 'tabel_ablation_sentimen.csv',
    'ablasi_mentah'  : TABEL / 'tabel_ablation_mentah.csv',
    'h1_verifikasi'  : TABEL / 'verifikasi_H1_vs_aktual.csv',
    'h1_forecast'    : TABEL / 'tabel_forecast_h1.csv',
    'shap'           : TABEL / 'tabel_shap_mean_abs.csv',
    'hp_xgb'         : TABEL / 'tabel_hyperparameter_xgboost.csv',
    'hp_lstm'        : TABEL / 'tabel_hyperparameter_lstm.csv',
    # --- pendukung (opsional) ---
    'dm_model'       : TABEL / 'tabel_uji_diebold_mariano.csv',
    'dm_ablasi'      : TABEL / 'tabel_diebold_mariano_ablation.csv',
    'korelasi_sent'  : TABEL / 'tabel_korelasi_sentimen_volatilitas.csv',
    'korelasi_lag'   : TABEL / 'tabel_multilag_korelasi_sentimen_volatilitas.csv',
    'event_study'    : TABEL / 'tabel_event_study_sentimen_volatilitas.csv',
    'eval_indobert'  : ROOT / '02_preprocessing' / 'sentimen_indobert' / 'output_sentimen'
                            / 'tabel_evaluasi_indobert.csv',
}

# Ada beberapa versi file prediksi -> dipilih otomatis yang MAPE-nya cocok dgn naskah
KANDIDAT_PREDIKSI = {
    'XGBoost': [TABEL / 'hasil_prediksi_xgboost.csv',
                TABEL / 'hasil_prediksi_harian_xgboost.csv',
                ROOT / '05_evaluasi' / 'hasil_prediksi_xgboost.csv'],
    'LSTM'   : [TABEL / 'hasil_prediksi_lstm.csv',
                ROOT / '05_evaluasi' / 'hasil_prediksi_lstm.csv'],
}

# MAPE (%) per emiten di naskah tesis (Tabel 4.15 & 4.17) — acuan pencocokan
MAPE_NASKAH = {
    'XGBoost': {'ADRO': 7.77, 'ADMR': 8.46, 'ITMG': 9.95, 'PTBA': 15.25,
                'MEDC': 6.55, 'AKRA': 6.81, 'PGAS': 8.66},
    'LSTM'   : {'ADRO': 12.83, 'ADMR': 15.38, 'ITMG': 18.46, 'PTBA': 27.34,
                'MEDC': 13.12, 'AKRA': 14.35, 'PGAS': 12.33},
}

FITUR_SENTIMEN = {'Sentiment_Mean', 'Sentiment_Dispersion', 'News_Count',
                  'News_Available', 'Ada_Berita'}
FITUR_VOLUME = {'Volume_ZScore'}
FITUR_HISTORIS = {'Vol_Lag1', 'Vol_Lag2', 'Vol_Lag3', 'Vol_MA5', 'Vol_STD5', 'Log_Return'}


# ------------------------------------------------------------------------------
# 1. FUNGSI BANTU
# ------------------------------------------------------------------------------
laporan = []   # rekap status setiap file keluaran

def catat(nomor, nama, status, pesan=''):
    laporan.append({'No': nomor, 'File': nama, 'Status': status, 'Keterangan': pesan})
    ikon = {'OK': '✅', 'PERIKSA': '⚠️', 'GAGAL': '❌', 'LEWAT': '➖'}[status]
    print(f'{ikon} {nomor} {nama:<34} {pesan}')

def baca(kunci_atau_path, **kw):
    """Baca CSV dari kamus SUMBER (atau path langsung). None bila tidak ada."""
    path = SUMBER.get(kunci_atau_path, kunci_atau_path) if isinstance(kunci_atau_path, str) \
        else kunci_atau_path
    if not Path(path).exists():
        return None
    return pd.read_csv(path, encoding='utf-8-sig', **kw)

def standar(df):
    """Seragamkan nama kolom kunci, format tanggal & kode emiten."""
    df = df.rename(columns={'Kode_Emiten': 'Emiten', 'Tanggal': 'Date',
                            'Unnamed: 0': '_idx'})
    df = df.drop(columns=[c for c in ['_idx', 'No'] if c in df.columns])
    if 'Emiten' in df.columns:
        df['Emiten'] = df['Emiten'].astype(str).str.upper().str.strip()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

def simpan(df, nama):
    """Urutkan per emiten & tanggal, lalu tulis dengan format seragam."""
    df = df.copy()
    kunci = [c for c in ['Model', 'Emiten', 'Date'] if c in df.columns]
    if kunci:
        # Urutan emiten mengikuti URUTAN_EMITEN; kode lain (jika ada) ditaruh di akhir
        posisi = {e: i for i, e in enumerate(URUTAN_EMITEN)}
        df = df.sort_values(kunci, key=lambda s: s.map(posisi).fillna(99)
                            if s.name == 'Emiten' else s, kind='stable')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    df.to_csv(OUT / nama, index=False, encoding='utf-8-sig')
    return len(df)

def cari_kolom(df, *kata):
    """Cari nama kolom yang mengandung salah satu kata (tidak peka huruf besar)."""
    for c in df.columns:
        if any(k.lower() in c.lower() for k in kata):
            return c
    return None

def kelompok_fitur(f):
    if f in FITUR_SENTIMEN:  return 'Sentimen'
    if f in FITUR_VOLUME:    return 'Volume'
    if f in FITUR_HISTORIS:  return 'Historis Volatilitas'
    return 'Teknikal'

def jalankan(nomor, nama, fungsi):
    """Bungkus tiap langkah agar satu kegagalan tidak menghentikan yang lain."""
    try:
        fungsi(nomor, nama)
    except Exception as e:
        catat(nomor, nama, 'GAGAL', f'{type(e).__name__}: {e}')


print('=' * 78)
print('  SIAPKAN DATA DASHBOARD')
print(f'  Tujuan: {OUT}')
print('=' * 78)

# ------------------------------------------------------------------------------
# 2. LANGKAH PER FILE
# ------------------------------------------------------------------------------
def f01(no, nama):
    df = baca('teknikal')
    if df is None: return catat(no, nama, 'GAGAL', 'technical_features.csv tidak ada')
    df = standar(df)
    n = simpan(df, nama)
    catat(no, nama, 'OK', f'{n:,} baris | {df["Date"].min():%d/%m/%Y}–{df["Date"].max():%d/%m/%Y}')

def f02(no, nama):
    df = baca('sentimen_harian')
    if df is None: return catat(no, nama, 'GAGAL', 'skor_sentimen_harian.csv tidak ada')
    df = standar(df).rename(columns={'Ada_Berita': 'News_Available'})
    n = simpan(df, nama)
    catat(no, nama, 'OK', f'{n:,} baris')

def f03(no, nama):
    path = SUMBER['berita_artikel']
    if not path.exists(): return catat(no, nama, 'LEWAT', 'skor_sentimen_per_artikel.csv tidak ada')
    # Isi berita lengkap TIDAK dibawa (26 MB) — cukup judul & tautan
    kolom = pd.read_csv(path, nrows=0, encoding='utf-8-sig').columns
    dipakai = [c for c in kolom if c not in ('Isi_Berita', 'Teks', 'Keyword')]
    df = standar(pd.read_csv(path, usecols=dipakai, encoding='utf-8-sig'))
    df = df.rename(columns={'Label_Auto': 'Label', 'Headline': 'Judul'})
    kol_skor = cari_kolom(df, 'Skor_Sentim')
    if kol_skor: df = df.rename(columns={kol_skor: 'Skor'})
    prob = [c for c in ['P_Negatif', 'P_Netral', 'P_Positif'] if c in df.columns]
    if prob: df['Confidence'] = df[prob].max(axis=1).round(4)
    n = simpan(df, nama)
    catat(no, nama, 'OK', f'{n:,} artikel | kolom: {list(df.columns)}')

def f04(no, nama):
    df = baca('dataset_model')
    if df is None: return catat(no, nama, 'GAGAL', 'dataset_gabungan_X1X2Y.csv tidak ada')
    df = standar(df).rename(columns={'Ada_Berita': 'News_Available'})
    split = baca('split')
    pesan = ''
    if split is not None:
        split = standar(split)
        mulai_uji = split.set_index('Emiten')['test_start'].pipe(pd.to_datetime, errors='coerce')
        df['Split'] = np.where(df['Date'] >= df['Emiten'].map(mulai_uji), 'Test', 'Train')
        pesan = '| ' + df.groupby('Split').size().to_string().replace('\n', ', ')
    n = simpan(df, nama)
    catat(no, nama, 'OK', f'{n:,} baris {pesan}')

def f05(no, nama):
    bagian = []
    for model, kunci in [('XGBoost', 'eval_xgb'), ('LSTM', 'eval_lstm')]:
        d = baca(kunci)
        if d is None: continue
        d = standar(d).rename(columns={'MAPE (%)': 'MAPE', 'sMAPE (%)': 'sMAPE'})
        d.insert(0, 'Model', model)
        bagian.append(d)
    if not bagian: return catat(no, nama, 'GAGAL', 'tabel_evaluasi_*.csv tidak ada')
    df = pd.concat(bagian, ignore_index=True)
    df = df[[c for c in ['Model', 'Emiten', 'RMSE', 'MAE', 'MAPE', 'sMAPE', 'R2', 'QLIKE']
             if c in df.columns]]
    # Baris ringkasan (mis. RATA-RATA) dibuang: dashboard menghitung agregatnya sendiri
    lain = sorted(set(df['Emiten']) - set(URUTAN_EMITEN))
    if lain:
        print(f'     Baris non-emiten dibuang dari tabel evaluasi: {lain}')
    df = df[df['Emiten'].isin(URUTAN_EMITEN)].copy()
    # Tambah N_test agar agregasi bisa ditimbang jumlah observasi
    split = baca('split')
    if split is not None:
        df['N_test'] = df['Emiten'].map(standar(split).set_index('Emiten')['n_test'])
    # Cocokkan MAPE per emiten dengan naskah dan tampilkan rinciannya
    df['MAPE_naskah'] = [MAPE_NASKAH[m].get(e) for m, e in zip(df['Model'], df['Emiten'])]
    df['Selisih'] = (df['MAPE'] - df['MAPE_naskah']).round(2)
    print('     MAPE file tabel_evaluasi_*.csv vs naskah (Tabel 4.15 & 4.17):')
    print(df.pivot(index='Emiten', columns='Model', values=['MAPE', 'MAPE_naskah', 'Selisih'])
            .reindex(URUTAN_EMITEN).round(2).to_string().replace('\n', '\n       '))
    maks = df['Selisih'].abs().max()
    beda = df.loc[df['Selisih'].abs() >= 0.05, ['Model', 'Emiten', 'Selisih']]
    n = simpan(df.drop(columns=['MAPE_naskah', 'Selisih']), nama)
    if beda.empty:
        catat(no, nama, 'OK', f'{n} baris | seluruh MAPE sama dengan naskah')
    else:
        daftar = ', '.join(f'{m} {e} ({s:+.2f})' for m, e, s in beda.itertuples(index=False))
        catat(no, nama, 'PERIKSA', f'{n} baris | beda dgn naskah: {daftar}')

def f06(no, nama):
    df = baca('ablasi')
    if df is None: return catat(no, nama, 'LEWAT', 'tabel_ablation_sentimen.csv tidak ada')
    df = standar(df).rename(columns={'Δ_RMSE_%': 'Delta_RMSE_pct', 'Δ_MAPE_pp': 'Delta_MAPE_pp',
                                     'Δ_R2': 'Delta_R2'})
    n = simpan(df, nama)
    catat(no, nama, 'OK', f'{n} baris | kolom: {list(df.columns)}')

def f07(no, nama):
    """Pilih file prediksi yang MAPE per emitennya paling cocok dgn naskah."""
    hasil = []
    for model, daftar in KANDIDAT_PREDIKSI.items():
        terbaik, skor_terbaik = None, np.inf
        print(f'     Kandidat prediksi {model}:')
        for path in daftar:
            if not path.exists():
                print(f'        - {path.relative_to(ROOT)}  (tidak ada)'); continue
            d = standar(pd.read_csv(path, encoding='utf-8-sig'))
            mape = (d.assign(e=(d['Y_Aktual'] - d['Y_Prediksi']).abs() / d['Y_Aktual'])
                      .groupby('Emiten')['e'].mean() * 100)
            acuan = pd.Series(MAPE_NASKAH[model])
            skor = (mape.reindex(acuan.index) - acuan).abs().mean()
            print(f'        - {path.relative_to(ROOT)}  -> selisih rata-rata {skor:.2f} poin')
            if skor < skor_terbaik:
                terbaik, skor_terbaik = d, skor
                nama_terbaik = path.relative_to(ROOT)
        if terbaik is None: continue
        print(f'       => DIPAKAI: {nama_terbaik}')
        mape_pilih = (terbaik.assign(e=(terbaik['Y_Aktual'] - terbaik['Y_Prediksi']).abs()
                                     / terbaik['Y_Aktual']).groupby('Emiten')['e'].mean() * 100)
        banding = pd.DataFrame({'MAPE_file': mape_pilih, 'MAPE_naskah': pd.Series(MAPE_NASKAH[model])})
        banding['Selisih'] = banding['MAPE_file'] - banding['MAPE_naskah']
        print('          ' + banding.reindex(URUTAN_EMITEN).round(2).to_string()
              .replace('\n', '\n          '))
        terbaik = terbaik.rename(columns={'Y_Aktual': 'Aktual', 'Y_Prediksi': 'Prediksi'})
        terbaik['Model'] = model
        terbaik['Residual'] = terbaik['Aktual'] - terbaik['Prediksi']
        hasil.append((terbaik[['Date', 'Emiten', 'Model', 'Aktual', 'Prediksi', 'Residual']],
                      skor_terbaik))
    if not hasil: return catat(no, nama, 'GAGAL', 'tidak ada file prediksi')
    df = pd.concat([h[0] for h in hasil], ignore_index=True)
    maks = max(h[1] for h in hasil)
    n = simpan(df, nama)
    catat(no, nama, 'OK' if maks < 0.5 else 'PERIKSA',
          f'{n:,} baris | selisih MAPE terbesar vs naskah {maks:.2f} poin')

def f08(no, nama):
    v = baca('h1_verifikasi')
    if v is None: return catat(no, nama, 'GAGAL', 'verifikasi_H1_vs_aktual.csv tidak ada')
    v = standar(v)
    f = baca('h1_forecast')
    f = standar(f) if f is not None else pd.DataFrame({'Emiten': v['Emiten']})
    kol_basis = cari_kolom(f, 'Basis')
    baris = []
    for model, kp, ke, kk in [('XGBoost', 'Pred_XGBoost', 'Err_XGB_%', 'Kategori_Risiko_XGB'),
                              ('LSTM', 'Pred_LSTM', 'Err_LSTM_%', 'Kategori_Risiko_LSTM')]:
        d = v[['Emiten']].copy()
        d['Model'] = model
        d['Prediksi'] = v[kp]
        d['Aktual'] = v.get('Vol_Aktual')
        d['Error_pct'] = v.get(ke)
        d = d.merge(f[[c for c in ['Emiten', kol_basis, kk, 'Threshold_Q33', 'Threshold_Q66']
                       if c and c in f.columns]], on='Emiten', how='left')
        d = d.rename(columns={kol_basis: 'Tanggal_Acuan', kk: 'Kategori_Risiko'})
        baris.append(d)
    df = pd.concat(baris, ignore_index=True)
    if 'Tanggal_Acuan' in df.columns:
        teks = df['Tanggal_Acuan'].astype(str)
        tgl = pd.to_datetime(teks, format='%Y-%m-%d', errors='coerce')          # 2026-06-30
        tgl = tgl.fillna(pd.to_datetime(teks, format='%d/%m/%Y', errors='coerce'))  # 30/06/2026
        df['Tanggal_Acuan'] = tgl
        df['Tanggal_Prediksi'] = df['Tanggal_Acuan'] + pd.offsets.BDay(1)
        for c in ['Tanggal_Acuan', 'Tanggal_Prediksi']:
            df[c] = df[c].dt.strftime('%Y-%m-%d')
    naskah = {'XGBoost': {'ADRO': 1.41, 'ADMR': 3.30, 'ITMG': 4.21, 'PTBA': 0.34,
                          'MEDC': 22.21, 'AKRA': 8.17, 'PGAS': 10.29},
              'LSTM'   : {'ADRO': 12.31, 'ADMR': 1.86, 'ITMG': 5.48, 'PTBA': 21.53,
                          'MEDC': 8.45, 'AKRA': 2.34, 'PGAS': 3.98}}   # Tabel 4.20
    df['Error_naskah'] = [naskah[m].get(e) for m, e in zip(df['Model'], df['Emiten'])]
    df['Selisih'] = (df['Error_pct'] - df['Error_naskah']).round(2)
    print('     Error H+1 (%) file vs naskah (Tabel 4.20):')
    print('       ' + df.pivot(index='Emiten', columns='Model',
                               values=['Error_pct', 'Error_naskah', 'Selisih'])
          .reindex(URUTAN_EMITEN).round(2).to_string().replace('\n', '\n       '))
    beda = df.loc[df['Selisih'].abs() >= 0.05]
    n = simpan(df.drop(columns=['Error_naskah', 'Selisih']), nama)
    rata = df.groupby('Model')['Error_pct'].mean().round(2).to_dict()
    catat(no, nama, 'OK' if beda.empty else 'PERIKSA',
          f'{n} baris | MAPE H+1 {rata} (naskah: XGBoost 7.13, LSTM 7.99)')

def f09(no, nama):
    df = baca('shap')
    if df is None: return catat(no, nama, 'GAGAL', 'tabel_shap_mean_abs.csv tidak ada')
    df = standar(df)
    df['Fitur'] = df['Fitur'].replace({'Ada_Berita': 'News_Available'})
    df['Kelompok'] = df['Fitur'].map(kelompok_fitur)
    df.insert(0, 'Model', 'XGBoost')
    n = simpan(df, nama)
    porsi = (df.groupby('Kelompok')['Mean_Abs_SHAP'].sum() / df['Mean_Abs_SHAP'].sum() * 100).round(2)
    # Dua cara menghitung porsi sentimen -> cocokkan dengan angka 2,49% di Kesimpulan
    per_emiten = (df[df['Kelompok'] == 'Sentimen'].groupby('Emiten')['Mean_Abs_SHAP'].sum()
                  / df.groupby('Emiten')['Mean_Abs_SHAP'].sum() * 100)
    print(f'     Porsi SHAP sentimen — cara A (total semua emiten) : {porsi.get("Sentimen", 0):.2f}%')
    print(f'     Porsi SHAP sentimen — cara B (rata-rata per emiten): {per_emiten.mean():.2f}%')
    print(f'     Per emiten: {per_emiten.reindex(URUTAN_EMITEN).round(2).to_dict()}')
    print('     Angka di Kesimpulan naskah: 2.49%')
    catat(no, nama, 'OK', f'{n} baris | porsi kelompok (%): {porsi.to_dict()}')

def f10(no, nama):
    bagian = []
    for model, kunci in [('XGBoost', 'hp_xgb'), ('LSTM', 'hp_lstm')]:
        d = baca(kunci)
        if d is None: continue
        d = standar(d).melt(id_vars='Emiten', var_name='Parameter', value_name='Nilai')
        d.insert(0, 'Model', model)
        bagian.append(d)
    if not bagian: return catat(no, nama, 'LEWAT', 'tabel hyperparameter tidak ada')
    n = simpan(pd.concat(bagian, ignore_index=True), nama)
    catat(no, nama, 'OK', f'{n} baris (format panjang: Model, Emiten, Parameter, Nilai)')

def salin_pendukung(kunci):
    def _f(no, nama):
        df = baca(kunci)
        if df is None: return catat(no, nama, 'LEWAT', f'{SUMBER[kunci].name} tidak ada')
        n = simpan(standar(df), nama)
        catat(no, nama, 'OK', f'{n} baris')
    return _f

def f00(no, nama):
    meta = {}
    if SUMBER['meta_teknikal'].exists():
        meta = json.loads(SUMBER['meta_teknikal'].read_text(encoding='utf-8'))
    meta['kolom_x2'] = ['Sentiment_Mean', 'Sentiment_Dispersion', 'News_Count', 'News_Available']
    meta['emiten'] = URUTAN_EMITEN
    split = baca('split')
    if split is not None:
        meta['split'] = standar(split).set_index('Emiten').astype(str).to_dict(orient='index')
    hp = baca('hp_lstm')
    if hp is not None and 'lookback' in hp.columns:
        meta['lookback_lstm'] = standar(hp).set_index('Emiten')['lookback'].astype(int).to_dict()
    meta['tanggal_update'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    meta['daftar_file'] = sorted(p.name for p in OUT.glob('*.csv'))
    (OUT / nama).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    catat(no, nama, 'OK', f'{len(meta["daftar_file"])} file terdaftar')


LANGKAH = [
    ('01', '01_technical_features.csv',          f01),
    ('02', '02_sentimen_harian.csv',             f02),
    ('03', '03_berita_berlabel.csv',             f03),
    ('04', '04_dataset_model.csv',               f04),
    ('05', '05_metrik_model.csv',                f05),
    ('06', '06_ablasi_sentimen.csv',             f06),
    ('07', '07_prediksi_test.csv',               f07),
    ('08', '08_prediksi_h1.csv',                 f08),
    ('09', '09_shap_importance.csv',             f09),
    ('10', '10_hyperparameter.csv',              f10),
    ('11', '11_uji_diebold_mariano.csv',         salin_pendukung('dm_model')),
    ('12', '12_dm_ablasi_sentimen.csv',          salin_pendukung('dm_ablasi')),
    ('13', '13_korelasi_sentimen_volatilitas.csv', salin_pendukung('korelasi_sent')),
    ('14', '14_korelasi_multilag.csv',           salin_pendukung('korelasi_lag')),
    ('15', '15_event_study_sentimen.csv',        salin_pendukung('event_study')),
    ('16', '16_evaluasi_indobert.csv',           salin_pendukung('eval_indobert')),
    ('00', '00_metadata.json',                   f00),   # terakhir: mendaftar semua file
]

for nomor, nama, fungsi in LANGKAH:
    jalankan(nomor, nama, fungsi)

# ------------------------------------------------------------------------------
# 3. REKAP
# ------------------------------------------------------------------------------
rekap = pd.DataFrame(laporan).sort_values('No')
rekap.to_csv(OUT / '_laporan_persiapan.csv', index=False, encoding='utf-8-sig')
print('=' * 78)
print(rekap['Status'].value_counts().to_string())
print(f'\nRekap tersimpan: {OUT / "_laporan_persiapan.csv"}')
if (rekap['Status'].isin(['GAGAL', 'PERIKSA'])).any():
    print('\n⚠️  Ada file berstatus GAGAL/PERIKSA — kirimkan output sel ini untuk ditelusuri.')
