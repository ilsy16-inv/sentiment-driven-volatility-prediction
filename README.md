# Sentiment-Driven Volatility Prediction

Repositori backup penelitian Tesis Magister:

**Pemodelan Prediksi Volatilitas Return Saham Sektor Energi (LQ45) Berdasarkan Analisis Sentimen Menggunakan Metode XGBoost dan LSTM**

- Penulis: Ilda Nurida (NRP 6032251068)
- Program: Magister, Sekolah Interdisipliner Manajemen dan Teknologi – ITS
- Dosen Pembimbing: Dr. Ir. Ahmad Saikhu, S.Si, MT.
- Tahun: 2026

## Ringkasan Penelitian

| Komponen | Keterangan |
|---|---|
| Objek | ADRO, ADMR, ITMG, PTBA, MEDC, AKRA (sektor energi LQ45) |
| Periode data | April 2023 – April/Mei 2026 |
| Input X1 (teknikal) | OHLCV (Yahoo Finance), RSI, Moving Average, Stochastic Oscillator |
| Input X2 (sentimen) | Berita Detik Finance, Kontan, Bisnis.com → IndoBERT (`indobert-base-p2`): +1 / 0 / −1 |
| Output Y | Volatilitas harian (standar deviasi log return) |
| Split data | 80% train / 20% test, kronologis (Time-Series Split) |
| Model | XGBoost (tabular) vs LSTM (sekuensial, lookback 10 hari) |
| Evaluasi | RMSE, MAE, MAPE (target MAPE < 20%) |

## Struktur Folder

```
data/        # data mentah & olahan (harga saham, berita, skor sentimen)
notebooks/   # notebook eksplorasi & pemodelan
src/         # skrip Python (scraping, preprocessing, sentimen, model)
results/     # hasil evaluasi, grafik, model tersimpan
docs/        # draf tesis & dokumen pendukung
```
