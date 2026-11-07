# Dashboard Streamlit — Hasil Penelitian

Dashboard interaktif hasil revisi konferensi (hybrid GARCH–XGBoost dan GARCH–LSTM).
Semua data dibaca **langsung dari repositori GitHub ini** (alamat `raw.githubusercontent.com`),
jadi setiap hasil notebook yang di-push otomatis tampil di dashboard (cache 1 jam, atau klik
**Muat ulang data** di sidebar).

## Isi

| Tab | Isi | File sumber |
|---|---|---|
| Ringkasan | MAPE rata-rata, kontribusi SHAP sentimen, temuan utama | (dihitung dari file di bawah) |
| Performa model | RMSE, MAE, MAPE, R² per emiten (Table VIII) | `03_dataset/final/dataset_test_*.csv`, `scaler_params_*.csv`, `06_dokumen_tesis/tabel/hasil_prediksi_xgboost.csv`, `hasil_prediksi_lstm.csv` |
| Aktual vs prediksi | Grafik harian per emiten | sama seperti di atas |
| Uji Diebold–Mariano | Heatmap DM untuk loss MSE, MAE, QLIKE | `06_dokumen_tesis/tabel/tabel_uji_dm_robust_loss.csv` |
| Kontribusi fitur (SHAP) | Rata-rata \|SHAP\| per fitur | `06_dokumen_tesis/tabel/tabel_shap_per_emiten.csv` |
| Pengaruh sentimen | Ablasi X1 vs X1+X2 per emiten dan kondisi pasar | `tabel_ablation_sentimen.csv`, `tabel_ablation_kondisi_pasar.csv` |
| Karakteristik & validasi | Parameter GARCH, validasi IndoBERT, sentimen harian | `tabel_parameter_garch.csv`, `output_sentimen_v2_manual/tabel_validasi_indobert.csv`, `skor_sentimen_harian.csv` |

## Menjalankan di laptop

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

## Online di Streamlit Community Cloud (gratis)

1. Buka https://share.streamlit.io dan login dengan akun GitHub **ilsy16-inv**.
2. Klik **Create app → Deploy a public app from GitHub**.
3. Isi: Repository `ilsy16-inv/sentiment-driven-volatility-prediction`, Branch `main`,
   Main file path `streamlit_app/app.py`.
4. Klik **Deploy**. Setelah 2–3 menit, dashboard punya alamat sendiri (`https://….streamlit.app`).

Catatan: aplikasi membaca file dari repo **public**. Jika repo diubah menjadi private,
alamat raw GitHub tidak bisa dibaca tanpa token.
