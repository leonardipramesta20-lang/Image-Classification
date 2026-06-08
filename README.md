# 🏗️ Deteksi Keretakan Beton — CNN Inspector

Aplikasi web berbasis **Streamlit** untuk mendeteksi dan mengklasifikasikan keretakan pada permukaan beton secara otomatis menggunakan model **Convolutional Neural Network (CNN)**.

---

## 👤 Identitas

| Keterangan | Detail |
|---|---|
| **Nama** | Jayaghu Leonardi Pramesta |
| **NIM** | 032400037 |
| **Program Studi** | Elektro Mekanika |

---

## 📌 Deskripsi Proyek

Sistem ini memanfaatkan deep learning (CNN) untuk menganalisis citra digital permukaan beton dan mengklasifikasikannya ke dalam dua kategori:

- **Retak** — terdeteksi adanya keretakan pada permukaan beton
- **Tidak Retak** — kondisi permukaan normal

Hasil analisis dapat diekspor sebagai laporan **PDF** (lengkap dengan grafik & kesimpulan otomatis) maupun file **CSV**.

---

## ✨ Fitur Utama

| Fitur | Keterangan |
|---|---|
| 🧠 Upload Model CNN | Load file `.h5` langsung dari browser |
| 🖼️ Multi-Citra | Analisis banyak gambar sekaligus |
| 🔃 Sorting Hasil | Urutkan berdasarkan confidence tertinggi/terendah |
| 📝 Catatan Inspeksi | Tambah catatan lapangan yang masuk ke laporan PDF |
| 📄 Laporan PDF Otomatis | Grafik donut + bar + kesimpulan otomatis |
| 📥 Export CSV | Download hasil dalam format spreadsheet |
| 📊 Dashboard Statistik | Visualisasi interaktif distribusi & confidence |
| 🗒️ Riwayat Sesi | Lihat & kelola riwayat inspeksi per batch |
| 🔄 Koreksi Label | Toggle untuk membalik urutan label jika prediksi terbalik |
| ⚙️ Threshold Adjustable | Atur batas confidence klasifikasi |

---

## 🛠️ Teknologi yang Digunakan

| Library | Versi | Kegunaan |
|---|---|---|
| [Streamlit](https://streamlit.io) | ≥ 1.30 | Framework UI |
| [TensorFlow / Keras](https://www.tensorflow.org) | ≥ 2.10 | Load & inferensi model CNN |
| [Plotly](https://plotly.com/python/) | ≥ 5.0 | Grafik interaktif |
| [Matplotlib](https://matplotlib.org) | ≥ 3.5 | Grafik untuk laporan PDF |
| [ReportLab](https://www.reportlab.com) | ≥ 3.6 | Generate laporan PDF |
| [Pillow](https://pillow.readthedocs.io) | ≥ 9.0 | Pemrosesan citra |
| [NumPy](https://numpy.org) | ≥ 1.21 | Komputasi array |
| [Pandas](https://pandas.pydata.org) | ≥ 1.3 | Manipulasi data |

---

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/username/cnn-inspector-beton.git
cd cnn-inspector-beton
```

### 2. Install Dependencies

Disarankan menggunakan virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 3. Jalankan Aplikasi

```bash
streamlit run streamlit_app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.

---

## 📦 Requirements

Buat file `requirements.txt` dengan isi berikut:

```
streamlit>=1.30.0
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
pillow>=9.0.0
plotly>=5.0.0
matplotlib>=3.5.0
reportlab>=3.6.0
```

---

## 📖 Panduan Penggunaan

1. **Upload Model** — Klik *Upload Model (.h5)* di sidebar, pilih file model CNN kamu
2. **Buka Menu Inspeksi** — Pilih menu `🔍 Inspeksi` dari navigasi
3. **Upload Gambar** — Upload satu atau beberapa gambar beton (JPG/PNG)
4. **Tambah Catatan** *(opsional)* — Isi kolom catatan inspeksi lapangan
5. **Lihat Hasil** — Hasil prediksi tampil dengan label dan progress bar confidence
6. **Download Laporan** — Klik tombol Download PDF atau CSV

> **Tips:** Jika hasil prediksi terbalik (retak terdeteksi sebagai normal), aktifkan toggle **🔄 Balik urutan label** di sidebar.

---

## 🗂️ Struktur File

```
cnn-inspector-beton/
│
├── streamlit_app.py       # File utama aplikasi
├── requirements.txt       # Daftar dependensi
├── README.md              # Dokumentasi ini
└── model/                 # (opsional) Letakkan file .h5 di sini
    └── model_cnn.h5
```

---

## ☁️ Deploy ke Streamlit Cloud

1. Push repository ini ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **New app** → pilih repository ini
4. Set **Main file path** ke `streamlit_app.py`
5. Klik **Deploy**

> **Catatan:** File model `.h5` tidak perlu di-push ke GitHub. Upload langsung via sidebar saat aplikasi berjalan.

---

## 📄 Contoh Output Laporan PDF

Laporan PDF yang dihasilkan mencakup:

- Identitas inspektor & tanggal laporan
- Ringkasan hasil inspeksi (total citra, jumlah retak, persentase)
- Catatan inspeksi lapangan
- Grafik donut distribusi & grafik bar confidence per citra
- Tabel detail setiap citra beserta status dan confidence
- Kesimpulan otomatis berdasarkan hasil analisis

---

## 📝 Lisensi

Proyek ini dibuat untuk keperluan akademik. Silakan digunakan dan dimodifikasi sesuai kebutuhan dengan tetap mencantumkan atribusi.

---

<div align="center">
  Dibuat oleh <strong>Jayaghu Leonardi Pramesta</strong> — 032400037 — Elektro Mekanika
</div>
