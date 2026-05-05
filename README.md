# 🈯 Comic Translator

Aplikasi penerjemah komik/manga otomatis ke **Bahasa Indonesia**, ditenagai YOLO + Gemini AI / Qwen2-VL.

## ✨ Fitur
- Deteksi bubble otomatis (YOLOv8, 2 model)
- OCR via Gemini AI (perlu API key) atau Qwen2-VL (lokal)
- Terjemahan: Google Translate, Gemini AI, Helsinki-NLP, DeepL
- Bahasa sumber: English, Japanese, Korean, Chinese, French, Spanish
- Format input: JPG/PNG, ZIP, **CBZ**, RAR, **CBR**, PDF, MangaDex link
- Output: PNG + PDF

## 🚀 Cara Pakai
```bash
pip install -r requirements.txt
python app.py
```

## 🔧 Changelog v2
- **Bug fix**: `qwen2_vl_ocr` output format `;`-delimited yang benar
- **Bug fix**: `process_bubble` tidak crash saat tidak ada kontur
- **Performance**: Model YOLO di-cache (tidak reload tiap gambar)
- **Performance**: HuggingFace pipeline di-cache
- **Fitur baru**: Format `.cbz` / `.cbr` didukung
- **Fitur baru**: Pilihan bahasa sumber (Japanese, Korean, dsb.)
- **Fitur baru**: Adaptive thresholding untuk bubble berwarna
- **Improvement**: Algoritma fitting teks pakai binary search
- **Improvement**: Error handling menyeluruh
- **Improvement**: UI dengan tab terpisah & status lebih jelas
