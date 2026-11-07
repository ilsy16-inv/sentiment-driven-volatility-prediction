"""
cek_file_tesis.py
=================
Memindai folder penelitian tesis dan membuat daftar semua file yang
berkaitan dengan penelitian, sebagai persiapan backup ke GitHub.

Tesis : Pemodelan Prediksi Volatilitas Return Saham Sektor Energi (LQ45)
        Berdasarkan Analisis Sentimen Menggunakan XGBoost dan LSTM
Penulis: Ilda Nurida (6032251068)

Cara pakai:
  1. Sesuaikan ROOT di bawah dengan lokasi folder tesis Anda.
  2. Jalankan di Jupyter (copy-paste ke satu sel) atau di terminal:
        python cek_file_tesis.py
  3. Hasil: ringkasan di layar + file 'daftar_file_tesis.csv' di folder ROOT.
  4. (Opsional) Setelah dicek, ubah BUAT_ZIP = True lalu jalankan lagi
     untuk membuat 'BACKUP_GITHUB_part01.zip', 'part02', dst. yang siap diunggah.
"""

import json
import os
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd

# ============================================================
# KONFIGURASI — SESUAIKAN
# ============================================================
ROOT = Path(r"E:\DATA PENTING ILDA NURIDA\DOKUMEN\KULIAH\S2 2025\THESIS"
            r"\Penelitian\TESIS_PEMODELAN PREDIKSI VOLATILITAS RETURN SAHAM_2026")

BUAT_ZIP = False          # True = buat BACKUP_GITHUB_partXX.zip (hanya file yang lolos)
UKURAN_PART_MB = 90       # zip dipecah per ~90 MB agar mudah diunggah
BATAS_GITHUB_MB = 100     # GitHub menolak file > 100 MB
BATAS_PERINGATAN_MB = 50  # GitHub memberi peringatan untuk file > 50 MB

# Folder yang dilewati (library, cache, environment)
FOLDER_DIABAIKAN = {
    ".venv", "venv", "env", ".env", "__pycache__", ".ipynb_checkpoints",
    ".git", "site-packages", "node_modules", ".idea", ".vscode", ".cache",
}

# Pola nama folder yang dilewati: checkpoint & bobot model IndoBERT/XLM-R
# (tiap checkpoint 475 MB - 3 GB, tidak bisa masuk GitHub -> simpan di Google Drive)
POLA_FOLDER_DIABAIKAN = (
    "ckpt", "checkpoint-", "indobert_sentimen_checkpoint", "tmp_",
    "indobert_finetuned", "indobert_sentimen_finetuned", "model_final",
)

# Folder spesifik yang tidak ikut di-backup (path relatif terhadap ROOT).
# 03_dataset_v1_leksikon: 41+13+22+9 file dengan ukuran identik dengan 03_dataset
# -> kemungkinan besar salinan. Hapus dari set ini jika ingin tetap ikut.
FOLDER_SPESIFIK_DIABAIKAN = {
    "03_dataset_v1_leksikon",
}

# Kategori berdasarkan ekstensi
KATEGORI = {
    "Notebook":        {".ipynb"},
    "Script Python":   {".py"},
    "Data":            {".csv", ".xlsx", ".xls", ".parquet", ".feather", ".txt", ".tsv"},
    "Konfigurasi/Meta": {".json", ".yaml", ".yml"},
    "Model":           {".pkl", ".joblib", ".h5", ".keras", ".pt", ".pth",
                        ".bin", ".safetensors", ".ubj", ".model", ".onnx"},
    "Gambar":          {".png", ".jpg", ".jpeg", ".svg", ".pdf_fig"},
    "Dokumen":         {".docx", ".doc", ".pdf", ".pptx", ".md"},
    "Arsip":           {".zip", ".rar", ".7z"},
}

# Kata kunci untuk menandai relevansi dengan penelitian
KATA_KUNCI = [
    "adro", "admr", "itmg", "ptba", "medc", "akra", "lq45", "saham",
    "return", "volatil", "sentimen", "sentiment", "indobert", "berita",
    "xgb", "xgboost", "lstm", "eda", "prep", "model", "evaluasi",
    "forecast", "teknikal", "technical", "rsi", "stoch", "anotasi",
    "tabel", "gambar", "fig", "tesis", "thesis", "scraping",
]


# ============================================================
# FUNGSI BANTU
# ============================================================
def kategori_file(path: Path) -> str:
    ext = path.suffix.lower()
    for nama, exts in KATEGORI.items():
        if ext in exts:
            return nama
    return "Lainnya"


def info_notebook(path: Path) -> str:
    """Hitung sel kode dan berapa yang sudah dijalankan (punya output)."""
    try:
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
        sel = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        dijalankan = sum(1 for c in sel if c.get("outputs"))
        catatan = f"{len(sel)} sel kode, {dijalankan} punya output"
        if sel and dijalankan == 0:
            catatan += " | BELUM DIJALANKAN / output kosong"
        return catatan
    except Exception as e:  # noqa: BLE001
        return f"Gagal dibaca: {type(e).__name__}"


def status_file(path: Path, ukuran_mb: float) -> str:
    nama = path.name
    if nama.startswith("~$") or nama.startswith(".~"):
        return "SAMPAH (file sementara Office)"
    if ukuran_mb > BATAS_GITHUB_MB:
        return "TERLALU BESAR (>100 MB, simpan di Google Drive)"
    if ukuran_mb > BATAS_PERINGATAN_MB:
        return "BESAR (50-100 MB, bisa tapi sebaiknya dihindari)"
    if ukuran_mb == 0:
        return "KOSONG (0 byte)"
    return "OK"


def relevan(path: Path) -> str:
    teks = str(path).lower()
    cocok = [k for k in KATA_KUNCI if k in teks]
    return ", ".join(sorted(set(cocok))[:5]) if cocok else "-"


# ============================================================
# PEMINDAIAN
# ============================================================
def pindai(root: Path) -> pd.DataFrame:
    baris = []
    for dirpath, dirnames, filenames in os.walk(root):
        # buang folder yang diabaikan agar os.walk tidak masuk ke dalamnya
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = [
            d for d in dirnames
            if d not in FOLDER_DIABAIKAN
            and not any(d.lower().startswith(pl) or d.lower() == pl
                        for pl in POLA_FOLDER_DIABAIKAN)
            and (rel_dir / d).as_posix() not in FOLDER_SPESIFIK_DIABAIKAN
        ]
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                st = p.stat()
            except OSError:
                continue
            ukuran_mb = st.st_size / 1_048_576
            kat = kategori_file(p)
            baris.append({
                "folder": str(p.parent.relative_to(root)) or ".",
                "nama_file": fn,
                "kategori": kat,
                "ukuran_MB": round(ukuran_mb, 3),
                "terakhir_diubah": datetime.fromtimestamp(st.st_mtime)
                                          .strftime("%Y-%m-%d %H:%M"),
                "status_github": status_file(p, ukuran_mb),
                "kata_kunci": relevan(p.relative_to(root)),
                "info_notebook": info_notebook(p) if kat == "Notebook" else "",
                "path_relatif": str(p.relative_to(root)),
            })
    return pd.DataFrame(baris)


def cetak_ringkasan(df: pd.DataFrame) -> None:
    garis = "=" * 78
    print(garis)
    print(f"  PEMINDAIAN FOLDER TESIS\n  {ROOT}")
    print(garis)
    print(f"  Total file ditemukan : {len(df)}")
    print(f"  Total ukuran         : {df['ukuran_MB'].sum():,.1f} MB\n")

    print("-- Per kategori " + "-" * 62)
    ring = (df.groupby("kategori")
              .agg(jumlah=("nama_file", "count"), total_MB=("ukuran_MB", "sum"))
              .sort_values("jumlah", ascending=False))
    ring["total_MB"] = ring["total_MB"].round(1)
    print(ring.to_string(), "\n")

    print("-- Per folder " + "-" * 64)
    for folder, g in df.sort_values(["folder", "nama_file"]).groupby("folder"):
        print(f"\n[{folder}]  ({len(g)} file, {g['ukuran_MB'].sum():.1f} MB)")
        for _, r in g.iterrows():
            tanda = "  " if r["status_github"] == "OK" else "! "
            ekstra = f"  <- {r['info_notebook']}" if r["info_notebook"] else ""
            print(f"  {tanda}{r['ukuran_MB']:9.2f} MB  {r['nama_file']}{ekstra}")

    masalah = df[df["status_github"] != "OK"]
    print("\n" + garis)
    if masalah.empty:
        print("  Semua file aman untuk GitHub.")
    else:
        print(f"  {len(masalah)} FILE PERLU PERHATIAN:")
        for _, r in masalah.iterrows():
            print(f"  - {r['path_relatif']}  ({r['ukuran_MB']:.1f} MB) -> {r['status_github']}")

    belum = df[df["info_notebook"].str.contains("BELUM DIJALANKAN", na=False)]
    if not belum.empty:
        print("\n  Notebook tanpa output (cek apakah ini versi yang benar):")
        for _, r in belum.iterrows():
            print(f"  - {r['path_relatif']}")
    print(garis)


def buat_zip(df: pd.DataFrame, root: Path) -> None:
    """Zip file yang lolos, dipecah per ~UKURAN_PART_MB (ukuran terkompresi)."""
    lolos = df[df["status_github"].isin(
        ["OK", "BESAR (50-100 MB, bisa tapi sebaiknya dihindari)"])]
    for lama in root.glob("BACKUP_GITHUB_part*.zip"):
        lama.unlink()
    batas = UKURAN_PART_MB * 1_048_576
    part, zf, hasil = 0, None, []
    for rel in sorted(lolos["path_relatif"]):
        if zf is None or zf.fp.tell() > batas:
            if zf:
                zf.close()
            part += 1
            tujuan = root / f"BACKUP_GITHUB_part{part:02d}.zip"
            zf = zipfile.ZipFile(tujuan, "w", zipfile.ZIP_DEFLATED)
            hasil.append(tujuan)
        zf.write(root / rel, arcname=rel)
    if zf:
        zf.close()
    print(f"\nZIP dibuat ({len(lolos)} file, {len(hasil)} bagian):")
    for z in hasil:
        print(f"  {z.name}  {z.stat().st_size / 1_048_576:6.1f} MB")


# ============================================================
# JALANKAN
# ============================================================
if __name__ == "__main__" or "get_ipython" in globals():
    if not ROOT.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan:\n{ROOT}\n"
                                "Periksa kembali variabel ROOT.")
    df = pindai(ROOT)
    df = df[~df["nama_file"].str.match(r"^(BACKUP_GITHUB.*\.zip|daftar_file_tesis\.csv)$")]
    cetak_ringkasan(df)

    out_csv = ROOT / "daftar_file_tesis.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")  # utf-8-sig: rapi di Excel
    print(f"\nDaftar lengkap disimpan: {out_csv}")

    if BUAT_ZIP:
        buat_zip(df, ROOT)