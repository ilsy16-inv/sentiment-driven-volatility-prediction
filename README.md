# Sentiment-Driven Volatility Prediction

This repository serves as a backup and documentation archive for a Master's thesis research project entitled:

**Sentiment-Based Volatility Prediction Modeling of Energy Sector Stock Returns (LQ45) Using XGBoost and Long Short-Term Memory (LSTM)**

- **Author:** Ilda Nurida
- **Program:** Master's Program, Interdisciplinary School of Management and Technology (SIMT), Institut Teknologi Sepuluh Nopember (ITS)
- **Year:** 2026

## Research Overview

| Component | Description |
|---|---|
| Research Objects | ADRO, ADMR, ITMG, PTBA, MEDC, AKRA (LQ45 energy sector stocks) |
| Data Period | April 2023 – April/May 2026 |
| Input X1 (Technical Features) | OHLCV data (Yahoo Finance), RSI, Moving Average, and Stochastic Oscillator |
| Input X2 (Sentiment Features) | Economic news articles processed using IndoBERT (`indobert-base-p2`): +1 / 0 / −1 |
| Output Y (Target Variable) | Daily volatility (standard deviation of log returns) |
| Data Splitting | 80% training / 20% testing, chronological order (time-series split) |
| Models | XGBoost (tabular), LSTM (sequential, 10-day lookback), and GARCH baseline |
| Evaluation Metrics | RMSE, MAE, and MAPE (target MAPE < 20%) |

## Repository Structure

| Directory | Description |
|---|---|
| `01_scraping_data/harga_saham/` | Stock price data scraping notebooks using yfinance |
| `01_scraping_data/berita_sentimen/` | News scraping checkpoints from various sources (Antara, Bloomberg, CNBC, Detik, IDX Channel, and Kontan) |
| `02_preprocessing/EDA/` | Exploratory Data Analysis (EDA): stock prices, log returns, volatility, and correlations |
| `02_preprocessing/sentimen_indobert/` | IndoBERT labeling and fine-tuning, including article-level sentiment scores (v1: lexicon-based, v2: manual annotation, v3: LLM-assisted) |
| `02_preprocessing/teknikal/` | Technical indicator calculation, dataset integration, and finalized modeling pipelines |
| `03_dataset/` | Raw, processed, and final datasets (X1 + X2 + Y) |
| `04_model/`, `04_modeling/` | XGBoost, LSTM, and GARCH modeling, sentiment ablation experiments, and out-of-sample forecasting |
| `05_evaluasi/` | Model evaluation results |
| `06_dokumen_tesis/` | Thesis drafts, figures (EDA, research background, and model diagrams), and tables |
| `Dashboard Penelitian/` | Research dashboard application and final datasets |
| `Laporan Progress/` | Research progress reports and presentation materials |

## Latest Notebook Versions (as of November 7, 2026)

The latest finalized notebooks are located in:

`02_preprocessing/teknikal/FIX/FIX_revisi konferensi/`

| Research Stage | Notebook |
|---|---|
| Technical Indicators | `PREP_01_technical_indicators_FIX_v4.ipynb` |
| IndoBERT Sentiment Analysis | `PREP_02_indobert_sentimen_v5.ipynb`, `PREP_02_indobert_sentimen_TESIS_FIX_v4..ipynb` |
| Dataset Integration | `PREP_03_integrasi_dataset_TESIS_FIX_v3.ipynb` |
| XGBoost Model | `MODEL_xgboost_TESIS_FIX_v4.ipynb` |
| LSTM Model | `MODEL_lstm_FIX_v5.ipynb` |
| Sentiment Ablation Study | `ABLATION_sentimen_FIX_v2.ipynb` |
| Model Evaluation and Diebold–Mariano Test | `EVALUASI_baseline_DM_v3.ipynb` |
| One-Step-Ahead Forecasting (H+1) | `FORECAST_H+1_v2.ipynb` |
| GARCH Baseline | `04_model/FIX/Baseline_GARCH.ipynb` |

### Notebooks Pending Execution

The following notebooks do not yet contain execution outputs and have not been run:

- `EDA_03_volatilitas_return_saham_revisi1`
- `PREP_01_technical_indicators_TESIS_FIX_v1`
- `PREP_02a_buat_set_uji`
- `PREP_02c_eksperimen_label_LLM`
- `Laporan Progress/INDOBERT_SENTIMEN`

## Files Excluded from This Repository

The following files and directories are intentionally excluded from this repository due to storage limitations and redundancy.

### 1. IndoBERT / XLM-R Model Checkpoints and Weights

Model checkpoints and weight directories, including `ckpt/`, `checkpoint-*/`, `model_final/`, and `tmp_*`, are excluded because their file sizes range from approximately 475 MB to 3 GB per file, exceeding GitHub's 100 MB per-file limit.

The final model used in `PREP_02_indobert_sentimen_v5.ipynb` is stored separately on Google Drive:

**[Access the Final IndoBERT Model – Google Drive](https://drive.google.com/drive/folders/1640To5b2ej4e19HuvrXHUOa2OC7TeYsC)**

- **Directory:** `02_preprocessing/sentimen_indobert/output_sentimen_v2_manual/model_final/`
- **Approximate Size:** 475 MB
- **Backup Date:** November 7, 2026

To reuse the trained model, copy the downloaded `model_final/` directory back to its original path. The notebook will load the existing model without requiring another fine-tuning process.

### 2. Duplicate Dataset Directory

`03_dataset_v1_leksikon/` is excluded because it contains a duplicate copy of `03_dataset/`, with identical file sizes.

### 3. Python Virtual Environments

Python virtual environment directories, including `venv_tf_new/` and `.venv/`, are excluded to avoid unnecessary repository storage usage.

---

*This repository is maintained for research documentation, code backup, reproducibility, and further development of sentiment-driven stock return volatility prediction models.*
