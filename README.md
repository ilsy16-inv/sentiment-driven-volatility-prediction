# Sentiment-Driven Volatility Prediction

Repositori backup penelitian Tesis Magister:

**Pemodelan Prediksi Volatilitas Return Saham Sektor Energi (LQ45) Berdasarkan Analisis Sentimen Menggunakan Metode XGBoost dan LSTM**

- Penulis: Ilda Nurida
- Program: Magister, Sekolah Interdisipliner Manajemen dan Teknologi – ITS
- Tahun: 2026

## Ringkasan Penelitian

| Komponen | Keterangan |
|---|---|
| Objek | ADRO, ADMR, ITMG, PTBA, MEDC, AKRA (sektor energi LQ45) |
| Periode data | April 2023 – April/Mei 2026 |
| Input X1 (teknikal) | OHLCV (Yahoo Finance), RSI, Moving Average, Stochastic Oscillator |
| Input X2 (sentimen) | Berita ekonomi → IndoBERT (`indobert-base-p2`): +1 / 0 / −1 |
| Output Y | Volatilitas harian (standar deviasi log return) |
| Split data | 80% train / 20% test, kronologis (Time-Series Split) |
| Model | XGBoost (tabular) vs LSTM (sekuensial, lookback 10 hari); baseline GARCH |
| Evaluasi | RMSE, MAE, MAPE (target MAPE < 20%) |

## Struktur Folder

| Folder | Isi |
|---|---|
| `01_scraping_data/harga_saham/` | Notebook scraping harga saham (yfinance) |
| `01_scraping_data/berita_sentimen/` | Checkpoint hasil scraping berita per sumber (Antara, Bloomberg, CNBC, Detik, IDX Channel, Kontan) |
| `02_preprocessing/EDA/` | EDA: harga saham, log return, volatilitas, korelasi |
| `02_preprocessing/sentimen_indobert/` | Pelabelan & fine-tuning IndoBERT, skor sentimen per artikel (v1 leksikon, v2 manual, v3 LLM) |
| `02_preprocessing/teknikal/` | Indikator teknikal, integrasi dataset, dan pemodelan versi FIX |
| `03_dataset/` | Dataset `raw`, `processed`, `final` (X1 + X2 + Y) |
| `04_model/`, `04_modeling/` | Pemodelan XGBoost, LSTM, GARCH, ablation sentimen, forecast out-of-sample |
| `05_evaluasi/` | Hasil evaluasi |
| `06_dokumen_tesis/` | Draf tesis, gambar (EDA, latar belakang, model), dan tabel |
| `Dashboard Penelitian/` | Aplikasi dashboard dan data final |
| `Laporan Progress/` | Laporan progres & presentasi |

## Notebook Versi Terbaru (per 7 November 2026)

Versi terbaru berada di `02_preprocessing/teknikal/FIX/FIX_revisi konferensi/`:

| Tahap | Notebook |
|---|---|
| Indikator teknikal | `PREP_01_technical_indicators_FIX_v4.ipynb` |
| Sentimen IndoBERT | `PREP_02_indobert_sentimen_v5.ipynb`, `PREP_02_indobert_sentimen_TESIS_FIX_v4..ipynb` |
| Integrasi dataset | `PREP_03_integrasi_dataset_TESIS_FIX_v3.ipynb` |
| XGBoost | `MODEL_xgboost_TESIS_FIX_v4.ipynb` |
| LSTM | `MODEL_lstm_FIX_v5.ipynb` |
| Ablation sentimen | `ABLATION_sentimen_FIX_v2.ipynb` |
| Evaluasi + uji Diebold–Mariano | `EVALUASI_baseline_DM_v3.ipynb` |
| Forecast H+1 | `FORECAST_H+1_v2.ipynb` |
| Baseline GARCH | `04_model/FIX/Baseline_GARCH.ipynb` |

Notebook yang belum memiliki output (belum dijalankan): `EDA_03_volatilitas_return_saham_revisi1`, `PREP_01_technical_indicators_TESIS_FIX_v1`, `PREP_02a_buat_set_uji`, `PREP_02c_eksperimen_label_LLM`, `Laporan Progress/INDOBERT_SENTIMEN`.

## Yang Tidak Disimpan di Repositori Ini

- **Checkpoint & bobot model IndoBERT / XLM-R** (`ckpt/`, `checkpoint-*/`, `model_final/`, `tmp_*`): 475 MB–3 GB per file, melebihi batas GitHub (100 MB).
  - **Model final** yang dipakai `PREP_02_indobert_sentimen_v5.ipynb` (`02_preprocessing/sentimen_indobert/output_sentimen_v2_manual/model_final/`, ±475 MB) disimpan di [Google Drive](https://drive.google.com/drive/folders/1640To5b2ej4e19HuvrXHUOa2OC7TeYsC) (backup 7 November 2026). Untuk memakainya kembali, salin folder tersebut ke path aslinya; notebook akan memuat model tanpa fine-tune ulang.
- **`03_dataset_v1_leksikon/`**: salinan `03_dataset/` (ukuran file identik).
- Environment Python (`venv_tf_new`, `.venv`).
