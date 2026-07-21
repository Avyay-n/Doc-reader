# Hospital Bill & Invoice AI Extractor (`Doc reader`)

An end-to-end, privacy-preserving document extraction pipeline powered by local LLMs (`Ollama`) and OCR (`pdfplumber` + `PaddleOCR`/`EasyOCR`). It automatically parses complex hospital bills, pharmacy receipts, and itemized invoices into clean JSON, CSV, and Master Excel sheets without sending sensitive patient data to external clouds.

---

## 🛠 Prerequisites for Another Laptop

To run this project on a new PC or laptop (Windows, macOS, or Linux), you need two primary components installed:

1. **Python 3.10, 3.11, or 3.12+** (Fully supported).
   - Download from [python.org/downloads](https://www.python.org/downloads/).
   - *Note on Python 3.12.8:* For text-based PDFs, `pdfplumber`, `PyPDF2`, and `ollama` work 100% cleanly on Python 3.12+. If you process scanned image PDFs, `paddleocr` pip wheels may be skipped on Python 3.12, but our extractor automatically falls back to **`EasyOCR`** (`which fully supports Python 3.12`).
   - *Windows Users:* Ensure **"Add Python to PATH"** is checked during installation.
2. **Ollama** (Local AI Model Engine).
   - Download and install from [ollama.com](https://ollama.com/).

---

## 🚀 Step-by-Step Setup Guide

### Step 1: Download & Run Ollama (`qwen2.5:7b`)
1. Install Ollama and make sure the application is running in your system tray / background.
2. Open your terminal (`PowerShell` or `Command Prompt` on Windows, `Terminal` on Mac) and pull the required model:
   ```bash
   ollama pull qwen2.5:7b
   ```
   *(This downloads the `qwen2.5:7b` model used by the extractor. If you prefer another model, you can pull `llama3.1` or `mistral`).*

---

### Step 2: Copy the Project Folder
Copy the `Doc reader` folder onto the new laptop (via USB drive, Git clone, or shared drive).

---

### Step 3: Create a Virtual Environment & Install Dependencies
Open your terminal right inside the `Doc reader` project folder and run:

#### On Windows (PowerShell):
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install required Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

#### On macOS / Linux:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install required Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on Windows OCR:** If `paddleocr` requires C++ build dependencies on a fresh Windows PC, ensure the **Microsoft Visual C++ Redistributable** is installed from Microsoft's official website.

---

## ▶️ How to Run the Extractor

Place your PDF invoices or medical bills (e.g., `doc1.pdf`, `doc2.pdf`, `doc3.pdf`) directly inside the project folder.

### 1. Process a Single PDF Document
Run the python script with the target PDF filename:
```powershell
python doc_extractor.py doc1.pdf
```

### 2. Process All PDFs in the Folder
If you don't provide a filename, the script automatically finds and processes all `.pdf` files in the directory:
```powershell
python doc_extractor.py
```

### 3. Specify a Different Ollama Model
If you pulled a different model or want to test another version:
```powershell
python doc_extractor.py doc1.pdf --model llama3.1
```

---

## 📂 Output files

Once the script finishes processing, all extracted data is automatically saved inside the **`output_docs/`** directory:

1. **`output_docs/<filename>.json`**: Complete structured JSON output with page-by-page financial breakdown, line items, quantities, unit prices, net amounts, and grand total verification.
2. **`output_docs/<filename>.csv`**: Flat tabular spreadsheet format for quick filtering.
3. **`output_docs/Master_Hospital_Billing.xlsx`**: A consolidated Excel ledger that automatically appends every new bill processed across all documents into a single master sheet!
