# ════════════════════════════════════════════════════════════════════
# PERBANDINGAN MODEL BAHASA UNTUK KLASIFIKASI SENTIMEN BERITA EKONOMI
#   Kandidat : IndoBERT-base-p1, IndoBERT-base-p2, XLM-R base
#   Data     : data anotasi manual (yang dipakai untuk fine-tuning)
#   Syarat   : split, konfigurasi, dan class weight SAMA untuk semua model
#   Output   : perbandingan_model_sentimen.csv  -> Tabel di Bab 4.4.1
#
# Jalankan di Google Colab dengan GPU (Runtime > Change runtime type > T4 GPU)
# Perkiraan waktu: ± 5–15 menit per model per seed untuk ± 1.000 data
# ════════════════════════════════════════════════════════════════════

# %% Cell 1 — Instalasi (cukup sekali)
# !pip install -q transformers datasets accelerate scikit-learn

# %% Cell 2 — Konfigurasi (SESUAIKAN BAGIAN INI)
FILE_ANOTASI = r"E:\DATA PENTING ILDA NURIDA\DOKUMEN\KULIAH\S2 2025\THESIS\Penelitian\TESIS_PEMODELAN PREDIKSI VOLATILITAS RETURN SAHAM_2026\02_preprocessing\sentimen_indobert\output_sentimen\anotasi_gabungan_20260722_2343_n1088.csv"
# Di Colab: cukup tulis nama filenya, mis. FILE_ANOTASI = 'anotasi_gabungan_20260722_2343_n1088.csv'
KOLOM_TEKS   = None     # None = deteksi otomatis; atau isi nama kolom, mis. 'Teks'
KOLOM_LABEL  = None     # None = deteksi otomatis; atau isi nama kolom label MANUAL, mis. 'Label_Manual'

KANDIDAT = {
    'IndoBERT-base-p1': 'indobenchmark/indobert-base-p1',
    'IndoBERT-base-p2': 'indobenchmark/indobert-base-p2',
    'XLM-R base'      : 'xlm-roberta-base',
}
SEEDS = [42]          # untuk hasil lebih kuat: [42, 7, 2026] lalu laporkan rata-rata ± std

# Konfigurasi sama dengan Tabel konfigurasi fine-tuning di Bab 3
LR, BATCH, EPOCHS, MAX_LEN = 2e-5, 16, 5, 128
WEIGHT_DECAY, WARMUP       = 0.01, 0.1
TEST_SIZE, SPLIT_SEED      = 0.20, 42           # split dibuat SEKALI, dipakai semua model

# %% Cell 3 — Muat data dan split stratified (satu split untuk semua model)
import time, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             cohen_kappa_score, confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, set_seed)

try:
    df = pd.read_csv(FILE_ANOTASI, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(FILE_ANOTASI, encoding='latin-1')
print(f'File dimuat: {len(df)} baris')
print('Kolom tersedia:', df.columns.tolist())

def cari(kandidat, kecuali=()):
    norm = {c.lower().replace(' ', '_'): c for c in df.columns}
    for k in kandidat:
        if k in norm and norm[k] not in kecuali:
            return norm[k]
    return None

if KOLOM_TEKS is None:
    KOLOM_TEKS = cari(['teks', 'text', 'teks_bersih', 'clean_text', 'headline', 'judul', 'title', 'kalimat'])
if KOLOM_LABEL is None:   # utamakan label manual/final, hindari label otomatis leksikon
    KOLOM_LABEL = cari(['label_manual', 'label_final', 'label_valid', 'manual_label', 'label_anotasi',
                        'anotasi', 'label', 'sentimen', 'sentiment', 'label_sentimen', 'kelas'])
if KOLOM_TEKS is None or KOLOM_LABEL is None:
    raise SystemExit(f'\n>> Kolom tidak terdeteksi otomatis. Isi KOLOM_TEKS dan KOLOM_LABEL di Cell 2 '
                     f'dengan salah satu nama kolom di atas.')
print(f'Dipakai -> teks: "{KOLOM_TEKS}" | label: "{KOLOM_LABEL}"')

df = df[[KOLOM_TEKS, KOLOM_LABEL]].dropna()
df[KOLOM_TEKS] = df[KOLOM_TEKS].astype(str).str.strip()
LABELS = ['Negatif', 'Netral', 'Positif']
PETA = {'negatif': 'Negatif', 'negative': 'Negatif', 'neg': 'Negatif', '-1': 'Negatif', '-1.0': 'Negatif',
        'netral': 'Netral', 'neutral': 'Netral', 'neu': 'Netral', '0': 'Netral', '0.0': 'Netral',
        'positif': 'Positif', 'positive': 'Positif', 'pos': 'Positif', '1': 'Positif', '1.0': 'Positif'}
asli = df[KOLOM_LABEL].astype(str).str.strip()
print('Nilai label asli:', asli.value_counts().to_dict())
df[KOLOM_LABEL] = asli.str.lower().map(PETA)
if df[KOLOM_LABEL].isna().any():
    raise SystemExit(f'\n>> Ada nilai label yang tidak dikenali: {sorted(asli[df[KOLOM_LABEL].isna()].unique())[:10]}. '
                     f'Jika label berupa 0/1/2, periksa urutan kelasnya lalu sesuaikan PETA.')
lab2id = {l: i for i, l in enumerate(LABELS)}
df['label'] = df[KOLOM_LABEL].map(lab2id)

tr_df, va_df = train_test_split(df, test_size=TEST_SIZE, stratify=df['label'], random_state=SPLIT_SEED)
print(f'Total {len(df)} | latih {len(tr_df)} | validasi {len(va_df)}')
print(pd.crosstab(df[KOLOM_LABEL], 'n').T)

# bobot kelas: w_c = N / (K × n_c)  — sama dengan Persamaan di Bab 3
w = compute_class_weight('balanced', classes=np.arange(3), y=tr_df['label'])
print('Class weight:', dict(zip(LABELS, w.round(3))))

# %% Cell 4 — Trainer dengan weighted cross-entropy
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        out = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            out.logits, labels, weight=torch.tensor(w, dtype=torch.float, device=out.logits.device))
        return (loss, out) if return_outputs else loss

def metrik(pred):
    y, p = pred.label_ids, pred.predictions.argmax(-1)
    return {'accuracy': accuracy_score(y, p), 'f1_macro': f1_score(y, p, average='macro')}

def latih_satu(nama, ckpt, seed):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(ckpt)
    enc = lambda b: tok(b[KOLOM_TEKS], truncation=True, padding='max_length', max_length=MAX_LEN)
    ds_tr = Dataset.from_pandas(tr_df[[KOLOM_TEKS, 'label']].reset_index(drop=True)).map(enc, batched=True)
    ds_va = Dataset.from_pandas(va_df[[KOLOM_TEKS, 'label']].reset_index(drop=True)).map(enc, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt, num_labels=3, id2label=dict(enumerate(LABELS)), label2id=lab2id)
    args = TrainingArguments(
        output_dir=f'tmp_{nama.replace(" ", "_")}_{seed}', learning_rate=LR,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=32,
        num_train_epochs=EPOCHS, weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP,
        eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
        metric_for_best_model='f1_macro', save_total_limit=1, seed=seed,
        report_to='none', fp16=torch.cuda.is_available())
    tr = WeightedTrainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                         compute_metrics=metrik)
    t0 = time.time(); tr.train(); menit = (time.time() - t0) / 60
    y = va_df['label'].values
    p = tr.predict(ds_va).predictions.argmax(-1)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[0, 1, 2], zero_division=0)
    hasil = {'Model': nama, 'Seed': seed,
             'Parameter (juta)': round(sum(x.numel() for x in model.parameters()) / 1e6, 1),
             'Accuracy': accuracy_score(y, p), 'F1 macro': f1_score(y, p, average='macro'),
             'F1 Negatif': f1[0], 'F1 Netral': f1[1], 'F1 Positif': f1[2],
             'Cohen Kappa': cohen_kappa_score(y, p), 'Waktu latih (menit)': round(menit, 1)}
    print(f'\n{nama} (seed {seed})\n', confusion_matrix(y, p, labels=[0, 1, 2]))
    del model, tr; torch.cuda.empty_cache()
    return hasil

# %% Cell 5 — Jalankan semua kandidat
rows = [latih_satu(n, c, s) for n, c in KANDIDAT.items() for s in SEEDS]
hasil = pd.DataFrame(rows)

# %% Cell 6 — Ringkasan untuk tabel Bab 4.4.1
num = ['Accuracy', 'F1 macro', 'F1 Negatif', 'F1 Netral', 'F1 Positif', 'Cohen Kappa', 'Waktu latih (menit)']
ring = hasil.groupby('Model', sort=False).agg({'Parameter (juta)': 'first', **{k: 'mean' for k in num}})
if len(SEEDS) > 1:
    ring['F1 macro (std)'] = hasil.groupby('Model', sort=False)['F1 macro'].std()
ring = ring.sort_values('F1 macro', ascending=False).round(4)
print('\n=== PERBANDINGAN MODEL (data validasi berita ekonomi) ===')
print(ring.to_string())
ring.to_csv('perbandingan_model_sentimen.csv')
hasil.to_csv('perbandingan_model_sentimen_per_seed.csv', index=False)

terbaik = ring.index[0]
selisih = ring['F1 macro'].iloc[0] - ring.loc['IndoBERT-base-p2', 'F1 macro']
print(f'\nF1 macro tertinggi: {terbaik}. Selisih terhadap IndoBERT-base-p2: {selisih * 100:.2f} poin.')
print('Catatan: jika selisih kecil (< ±1 poin), pemilihan p2 dapat dipertahankan dengan alasan efisiensi.')
