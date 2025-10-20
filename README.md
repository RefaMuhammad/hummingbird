# 🦉 Hummingbird NLP – Pencarian Semantik Plus

Proyek ini adalah sebuah aplikasi web interaktif yang mengimplementasikan algoritma pencarian semantik (semantic search), yang terinspirasi dari pembaruan **Google Hummingbird**. Aplikasi ini tidak hanya mencari berdasarkan kata kunci, tetapi juga memahami makna dan konteks di balik pertanyaan pengguna untuk menemukan jawaban yang paling relevan dari kumpulan data.

Aplikasi ini dibangun menggunakan Streamlit dan memanfaatkan model Transformer canggih untuk pemrosesan bahasa alami (NLP).

## 🚀 Fitur Utama

Berdasarkan `interface.py`, fitur-fitur utama dari proyek ini meliputi:

  * **Pencarian Semantik (Semantic Embedding):** Menggunakan model `SentenceTransformer` (`all-MiniLM-L6-v2`) untuk mengubah pertanyaan dan dokumen konteks menjadi vektor embedding. Ini memungkinkan pencarian berdasarkan kesamaan makna (cosine similarity), bukan hanya kata kunci yang cocok.
  * **Reranking Hasil:** Setelah mendapatkan kandidat dokumen teratas dari pencarian semantik, model `CrossEncoder` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) digunakan untuk menilai ulang (rerank) pasangan (pertanyaan, konteks) untuk mendapatkan hasil yang paling akurat di peringkat teratas.
  * **Peningkatan Query (Query Enhancement):** Menerapkan teknik NLP menggunakan **spaCy** untuk meningkatkan pertanyaan pengguna sebelum diproses:
      * **Lemmatization:** Mengubah kata-kata ke bentuk dasarnya (misalnya, "won" -\> "win").
      * **Ekspansi Sinonim:** Secara otomatis menambahkan sinonim statis dan semantik (menggunakan vektor kata spaCy) untuk memperluas cakupan pencarian.
  * **Pengenalan Entitas (Entity Recognition - NER):** Menggunakan **spaCy** untuk mengekstrak dan menampilkan entitas penting (seperti Orang, Tempat, Organisasi) dari pertanyaan pengguna, memberikan pemahaman yang lebih baik tentang apa yang dicari.
  * **Antarmuka Interaktif:** Dibangun dengan **Streamlit** untuk menyediakan antarmuka web yang ramah pengguna untuk mengajukan pertanyaan dan melihat hasil analisis.

## 💻 Teknologi yang Digunakan

Proyek ini memanfaatkan pustaka Python modern untuk NLP dan web:

  * **Streamlit:** Untuk membangun dan menjalankan aplikasi web interaktif.
  * **SentenceTransformers:** Untuk model embedding dan cross-encoder.
  * **spaCy:** Untuk tugas-tugas NLP canggih (Lemmatization, NER, Word Vectors).
  * **Transformers (Hugging Face):** Sebagai dependensi inti untuk `SentenceTransformers`.
  * **Torch:** Sebagai backend deep learning untuk model-model Transformer.

## 📂 Susunan Project

Berikut adalah struktur file utama dalam repositori ini:

```
.
├── interface.py       # Skrip utama aplikasi Streamlit
├── requirements.txt   # Daftar dependensi Python
└── dev-v1.1.json      # Dataset SQuAD (Stanford Question Answering Dataset) v1.1
```

  * `interface.py`: Titik masuk utama aplikasi. File ini menangani UI, pemuatan model, pemrosesan query, pencarian, dan reranking.
  * `requirements.txt`: Berisi semua paket Python yang diperlukan untuk menjalankan proyek.
  * `dev-v1.1.json`: File data yang berisi paragraf konteks dan pertanyaan (berbasis SQuAD) yang digunakan aplikasi sebagai basis pengetahuan untuk mencari jawaban.

## 🛠️ Instalasi & Prasyarat

Untuk menjalankan proyek ini di mesin lokal Anda, ikuti langkah-langkah berikut:

**1. Clone Repository**

```bash
git clone https://github.com/nama-anda/hummingbird.git
cd hummingbird
```

**2. Buat Virtual Environment (Direkomendasikan)**

```bash
python -m venv venv
source venv/bin/activate  # Di Windows: venv\Scripts\activate
```

**3. Instal Dependensi**
Pastikan Anda memiliki Python 3.8+ terinstal.

```bash
pip install -r requirements.txt
```

**4. Unduh Model spaCy**
Proyek ini menggunakan model `en_core_web_md` dari spaCy untuk vektor kata dan NER. Jika tidak ditemukan, ia akan mencoba `en_core_web_sm`.

Unduh model medium (direkomendasikan):

```bash
python -m spacy download en_core_web_md
```

Atau unduh model kecil (fallback):

```bash
python -m spacy download en_core_web_sm
```

## 🏃 Contoh Penggunaan

Setelah semua prasyarat terinstal, jalankan aplikasi Streamlit:

```bash
streamlit run interface.py
```

Aplikasi akan otomatis terbuka di browser Anda (biasanya di `http://localhost:8501`).

1.  Buka aplikasi di browser.
2.  Pastikan file `dev-v1.1.json` berada di direktori yang sama.
3.  Masukkan pertanyaan dalam bahasa Inggris di kotak input (contoh: *Who won Super Bowl 50?* atau *What does AFC stand for?*).
4.  Klik tombol "Cari Makna 🔍".
5.  Aplikasi akan menampilkan hasil analisis query (query yang ditingkatkan, entitas) dan konteks yang paling relevan dari dataset.

## 🤝 Kontribusi

Kontribusi untuk proyek ini sangat diharapkan\! Jika Anda memiliki ide untuk perbaikan atau menemukan bug, silakan buka *Issue* atau kirimkan *Pull Request*.

1.  *Fork* repositori ini.
2.  Buat *branch* fitur baru (`git checkout -b fitur/perbaikan-keren`).
3.  *Commit* perubahan Anda (`git commit -am 'Menambahkan fitur keren'`).
4.  *Push* ke *branch* (`git push origin fitur/perbaikan-keren`).
5.  Buat *Pull Request* baru.
