<div align="center">

# 🔓 Xiaomi Mi Account Lock OCR Extractor

<p>
  <img src="https://img.shields.io/badge/Python-3.12+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-4.9-green?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/EasyOCR-1.7-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

<p>
  <b>🎯 Automatic Xiaomi Mi Account Unlock Code Extraction from Photos</b>
</p>

<p>
  <a href="#-features">✨ Features</a> •
  <a href="#-usage">🚀 Usage</a> •
  <a href="#-pipeline">🔄 Pipeline</a> •
  <a href="#-models">🤖 Models</a> •
  <a href="#-installation">📦 Installation</a>
</p>

<p>
  <img src="https://img.shields.io/github/stars/saeidsaadatigero/Xiaomi-Mi-Account-Lock-OCR-Extractor?style=social" />
  <img src="https://img.shields.io/github/forks/saeidsaadatigero/Xiaomi-Mi-Account-Lock-OCR-Extractor?style=social" />
</p>

</div>

---

## 🎯 What is this?

A powerful CLI tool that **extracts Xiaomi Mi Account unlock codes** from phone photos.

> When a Xiaomi phone gets locked, it displays a 4 or 5 segment code like `42MS-ETCO-R5MM-SWG`.
> This tool extracts that code from a photo! 📸 → 🔓

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Smart OCR** | EasyOCR + Vision AI combination |
| 🌍 **Multi-language** | Supports Persian & English UI |
| 🔄 **Auto Fallback** | If one model fails, next one tries |
| 📊 **Full Logging** | Every step is logged |
| 🧪 **Tested** | 16 unit tests |
| ⚡ **Fast** | Model caching, no redundant downloads |

---

## 🔄 Processing Pipeline


┌─────────────┐ │ Input Image│ ← test_photo.jpg └──────┬──────┘ ▼ ┌─────────────────┐ │ Preprocessing │ ← CLAHE + Denoise + Threshold │ (OpenCV) │ └──────┬──────────┘ ▼ ┌─────────────────┐ │ EasyOCR │ ← First attempt (local) │ (CPU/GPU) │ └──────┬──────────┘ ▼ (if failed) ┌─────────────────┐ │ Gemma 4 26B │ ← Smart vision model │ (OpenRouter) │ └──────┬──────────┘ ▼ (if rate limited) ┌─────────────────┐ │ Nemotron 12B │ ← Reasoning model │ (OpenRouter) │ └──────┬──────────┘ ▼ ┌─────────────────┐ │ Code Extraction│ ← Regex + Validation └──────┬──────────┘ ▼ ┌─────────────────┐ │ JSON Output │ → {"unlock_code": "42MS-ETCO-R5MM-SWG"} └─────────────────┘

YAML
Copy

---

## 🤖 AI Models Used

| Model | Type | Purpose |
|-------|------|---------|
| **EasyOCR** | Local | Fast OCR on CPU |
| **Gemma 4 26B** | Vision API | Smart text recognition in images |
| **Nemotron 12B VL** | Reasoning API | Deep image analysis |

---

## 📦 Installation

```bash
# Clone the project
git clone https://github.com/saeidsaadatigero/Xiaomi-Mi-Account-Lock-OCR-Extractor.git
cd Xiaomi-Mi-Account-Lock-OCR-Extractor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Setup API key
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY

🚀 Usage
BASH
Copy
# Simple usage
python main.py --image photo.jpg

# With verbose logging
python main.py --image photo.jpg --verbose

# Force specific model
python main.py --image photo.jpg --model gemma4

Success Output:
JSON
Copy
{
  "unlock_code": "42MS-ETCO-R5MM-SWG",
  "model": "gemma4",
  "confidence": null,
  "duration_ms": 9204.51
}

Error Output:
JSON
Copy
{
  "error": "PatternNotFoundError",
  "message": "No unlock code pattern found...",
  "models_tried": ["easyocr", "gemma4", "nemotron"]
}

🧪 Tests
BASH
Copy
pytest tests/ -v

CPP
Copy
tests/test_extractor_service.py::TestExtractPatternFromText::test_single_valid_code PASSED
tests/test_extractor_service.py::TestExtractPatternFromText::test_five_segment_code PASSED
tests/test_extractor_service.py::TestExtractPatternFromText::test_no_pattern_found PASSED
...
================== 16 passed in 11.82s ==================

📁 Project Structure
BASH
Copy
unlock_code_extractor/
├── main.py                    # CLI entry point
├── config.py                  # Configuration
├── exceptions.py              # Custom exceptions
├── requirements.txt           # Dependencies
├── .env.example               # Environment template
├── services/
│   └── extractor_service.py   # Core business logic
├── utils/
│   ├── image_preprocessor.py  # Image preprocessing
│   ├── ocr_corrector.py      # OCR correction
│   └── logger.py              # Logging setup
└── tests/
    └── test_extractor_service.py  # Unit tests

⚙️ Configuration (.env)
ENV
Copy
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
PRIMARY_API_MODEL=google/gemma-4-26b-a4b-it:free
SECONDARY_API_MODEL=nvidia/nemotron-nano-12b-v2-vl:free
OCR_CONFIDENCE_THRESHOLD=0.70
EASYOCR_GPU=false

🤝 Contributing

Contributions are welcome!

Fork the project
Create your branch (git checkout -b feature/amazing)
Commit your changes (git commit -m 'feat: add amazing feature')
Push to the branch (git push origin feature/amazing)
Open a Pull Request
📜 License

This project is released under the MIT License.

Made with ❤️ by Saeid Saadatigero

⭐ Don't forget to star if you like it! ⭐

```
