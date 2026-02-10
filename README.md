# Receipt RAG Extraction System

A high-accuracy receipt data extraction system leveraging **Phi-3** via a multi-prompt strategy and **RAG (Retrieval-Augmented Generation)** to minimize hallucinations and maximize structured data quality.

## 🚀 Overview

This system processes receipt images or OCR text and converts them into structured JSON data. It uses a sophisticated pipeline that includes text normalization, vendor classification, and a two-stage LLM extraction technique.

### Key Features

- **Multi-Prompt Strategy:** Uses specialized prompts for header extraction and itemized list extraction to ensure high precision.
- **RAG Integration:** Retrieves context from a knowledge base of receipt patterns and rules to guide the LLM.
- **OCR Normalization:** Corrects common OCR errors and standardizes text before extraction.
- **Confidence Scoring:** Provides per-field confidence scores and overall validation flags.
- **Pattern Learning:** Automatically learns and stores new receipt layouts to improve future extractions.
- **FastAPI Interface:** Ready-to-use API for seamless integration.

## 🛠️ Tech Stack

- **Large Language Model:** Phi-3 (via Ollama or HuggingFace)
- **Vector Database:** For RAG context retrieval
- **Framework:** FastAPI
- **Language:** Python 3.9+
- **OCR:** Compatible with standard OCR engines (Tesseract, Google Vision, etc.)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mohammedjaasir/RAG_SYSTEM.git
   cd RAG_SYSTEM
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the API

Start the FastAPI server:

```bash
uvicorn receipt.rag.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. You can access the automatic documentation at `http://localhost:8000/docs`.

## 📂 Project Structure

- `receipt/rag/`: Core RAG and LLM extraction logic.
- `receipt/extraction/`: Phi-3 specific extraction modules.
- `receipt/ocr/`: OCR client and processing.
- `receipt/support/`: Normalization and scoring utilities.
- `receipt/pipeline_orchestrator.py`: The main entry point for the full processing flow.

## 🛡️ License

MIT License - See the [LICENSE](LICENSE) file for details.
