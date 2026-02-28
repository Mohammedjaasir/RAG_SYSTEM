<div align="center">

# 🧾 Receipt RAG Extraction System

**AI-powered receipt data extraction with zero hallucinations and structured JSON output**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.1.0-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-F66B2B?style=for-the-badge)](https://www.trychroma.com/)
[![Phi-3](https://img.shields.io/badge/LLM-Phi--3-FF5733?style=for-the-badge&logo=microsoft&logoColor=white)](https://azure.microsoft.com/en-us/products/phi-3)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br />

> Transform raw, messy OCR receipt text into clean, validated, structured JSON data — powered by Retrieval-Augmented Generation, multi-prompt LLM strategies, and adaptive pattern learning.

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **Multi-Prompt LLM Strategy** | Separate, specialized prompts for header fields (vendor, date, total) and itemized line extraction for maximum precision |
| 🔍 **RAG-Augmented Extraction** | Retrieves semantically similar receipt patterns from ChromaDB to guide the LLM with live-context examples |
| 🔤 **OCR Normalization** | Auto-corrects common OCR errors (scrambled layouts, garbled characters) before LLM processing |
| 🏷️ **Vendor Classification** | Identifies and classifies vendor/shop types (restaurant, hotel, supermarket, etc.) to apply domain-specific extraction rules |
| 🛡️ **Hallucination Guard** | Validates LLM output against raw OCR source — any field not supported by the original text is flagged or rejected |
| 📊 **Confidence Scoring** | Per-field confidence scores and an overall validation flag for downstream quality control |
| 🧠 **Adaptive Pattern Learning** | Automatically learns new receipt formats/layouts and stores them in the knowledge base to improve future extractions |
| ⚡ **FastAPI REST Interface** | Production-ready API with `/extract`, `/health`, and `/clean-db` endpoints with full CORS support |
| 🗃️ **Structured JSON Output** | Outputs a clean, validated JSON schema with vendor info, date, currency, line items, totals, VAT, and more |

---

## 🏗️ Architecture

```
Receipt Image / OCR Text
        │
        ▼
┌───────────────────┐
│   OCR Client      │  ← Tesseract / Google Vision / Custom OCR
└────────┬──────────┘
         │ raw text
         ▼
┌───────────────────┐
│  OCR Normalizer   │  ← Corrects scrambled layouts, encodes errors
└────────┬──────────┘
         │ cleaned text
         ▼
┌───────────────────┐     ┌────────────────────┐
│ Vendor Classifier │────▶│  RAG Retriever      │
│                   │     │  (ChromaDB +        │
│ (type, name)      │     │   LangChain)        │
└────────┬──────────┘     └────────┬───────────┘
         │                         │ top-k context examples
         └─────────┬───────────────┘
                   ▼
        ┌────────────────────┐
        │  Phi-3 LLM Engine  │  ← Multi-Prompt Strategy
        │  ┌──────────────┐  │     • Prompt 1: Header fields
        │  │ Header Prompt│  │     • Prompt 2: Line items
        │  └──────────────┘  │
        │  ┌──────────────┐  │
        │  │ Items Prompt │  │
        │  └──────────────┘  │
        └────────┬───────────┘
                 │ raw LLM response
                 ▼
        ┌────────────────────┐
        │ Hallucination      │  ← Validates every field vs. OCR
        │ Validator          │    Rejects unsupported data
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────┐
        │  Confidence Scorer │  ← Per-field scores + overall flag
        └────────┬───────────┘
                 │
                 ▼
             JSON Output
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **LLM** | Microsoft Phi-3 (via Ollama or HuggingFace) |
| **RAG Framework** | LangChain + LangChain-Community |
| **Vector Database** | ChromaDB |
| **Embeddings** | Sentence-Transformers |
| **API Framework** | FastAPI + Uvicorn |
| **OCR** | Tesseract, Google Vision, or custom client |
| **PDF Support** | PyMuPDF |
| **Validation** | Pydantic v2 |
| **JSON Repair** | json-repair |

---

## 📂 Project Structure

```
RAG_SYSTEM/
├── receipt/
│   ├── __init__.py
│   ├── pipeline_orchestrator.py      # Main entry point: full end-to-end pipeline
│   │
│   ├── rag/                          # Core RAG logic
│   │   ├── receipt_rag.py            #   RAG pipeline, LLM prompting, extraction
│   │   ├── rag_retriever.py          #   ChromaDB retrieval logic
│   │   ├── vector_store.py           #   Vector store management
│   │   ├── knowledge_base_loader.py  #   Loads receipt patterns into ChromaDB
│   │   ├── pattern_learner.py        #   Adaptive pattern learning
│   │   ├── vendor_classifier.py      #   Receipt type/vendor classification
│   │   ├── ocr_client.py             #   OCR integration (multi-engine)
│   │   ├── rag_refresh_manager.py    #   DB refresh/rebuild management
│   │   ├── logger_utils.py           #   Logging utilities
│   │   └── main.py                   #   FastAPI app & endpoints
│   │
│   ├── extraction/                   # LLM extraction modules
│   │   ├── phi_item_extractor.py     #   Phi-3 line item extraction
│   │   ├── phi_hf_loader.py          #   HuggingFace Phi-3 model loader
│   │   ├── comprehensive_receipt_extractor.py
│   │   ├── improved_vat_extractor.py
│   │   └── additional_fields_extractor.py
│   │
│   ├── classification/               # Vendor/receipt classification
│   ├── standardization/              # Field normalization & standardization
│   ├── reconstruction/               # Data reconstruction utilities
│   ├── orchestration/                # Orchestration helpers
│   └── pipeline/                     # Pipeline stage definitions
│
├── config/                           # App / model configuration
├── requirements.txt
└── README.md
```

---

## 📦 Installation

### Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.com/) installed locally (for Phi-3 via Ollama)
- *(Optional)* HuggingFace account for model downloads

### 1. Clone the Repository

```bash
git clone https://github.com/Mohammedjaasir/RAG_SYSTEM.git
cd RAG_SYSTEM
```

### 2. Create & Activate Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS / Linux)
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull the Phi-3 Model (Ollama)

```bash
ollama pull phi3
```

> **Note:** If using HuggingFace instead, set the appropriate model environment variables in your `.env` file.

### 5. Environment Configuration (Optional)

Create a `.env` file in the root directory:

```env
# LLM Backend: "ollama" or "huggingface"
LLM_BACKEND=ollama

# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3

# ChromaDB path
CHROMA_DB_PATH=./receipt/rag/chroma_db
```

---

## 🚀 Running the API

Start the FastAPI server:

```bash
uvicorn receipt.rag.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be live at **`http://localhost:8000`**

📖 Interactive API docs: **`http://localhost:8000/docs`**

---

## 📡 API Endpoints

### `POST /extract` — Extract Receipt Data

Send raw OCR text and receive structured JSON.

**Request Body:**
```json
{
  "ocr_text": "SOCIAL KITCHEN\n123 High Street\nDate: 12/05/2024\nCappuccino       3.50\nAvocado Toast    8.00\nOrange Juice     4.20\n-------------------\nSubTotal:       15.70\nVAT (20%):       3.14\nTOTAL:          18.84",
  "retrieve_k": 3
}
```

**Response:**
```json
{
  "vendor_name": "Social Kitchen",
  "vendor_type": "restaurant",
  "date": "2024-05-12",
  "currency": "GBP",
  "items": [
    { "name": "Cappuccino",     "quantity": 1, "unit_price": 3.50,  "total_price": 3.50  },
    { "name": "Avocado Toast",  "quantity": 1, "unit_price": 8.00,  "total_price": 8.00  },
    { "name": "Orange Juice",   "quantity": 1, "unit_price": 4.20,  "total_price": 4.20  }
  ],
  "subtotal": 15.70,
  "vat": 3.14,
  "total": 18.84,
  "validation_passed": true,
  "_metadata": {
    "overall_confidence": 0.97,
    "relevance_scores": [0.91, 0.87, 0.82]
  }
}
```

---

### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "service": "receipt-rag-api",
  "version": "1.1.0"
}
```

---

### `POST /clean-db` — Reset Vector Database

Forces the ChromaDB vector store to clear and rebuild on the next request. Useful when significantly updating the knowledge base.

```bash
curl -X POST http://localhost:8000/clean-db
```

---

## 🧪 Testing

Run individual test scripts from the `receipt/rag/` directory:

```bash
# Test RAG extraction end-to-end
python receipt/rag/test_rag_demo.py

# Test diverse receipt types (hotel, restaurant, supermarket)
python receipt/rag/test_diverse.py

# Test hallucination prevention
python receipt/rag/test_hallucination_fix.py

# Test shop/vendor name extraction accuracy
python receipt/rag/test_shop_name.py

# Test API endpoints directly
python receipt/rag/test_api.py
```

---

## 🔄 How It Works

### 1. 📥 Input & Normalization
Raw OCR text (from any OCR engine) is fed into the normalizer, which corrects common OCR artifacts: character substitutions, scrambled column layouts (numbers appearing before item names), encoding issues, and whitespace noise.

### 2. 🏷️ Vendor Classification
The system detects the vendor name and type (restaurant, hotel, supermarket, pharmacy, etc.) using pattern-matching and a classifier trained on receipt structures. This determination selects the appropriate extraction rules.

### 3. 🔍 RAG Context Retrieval
The normalized text is embedded with `sentence-transformers` and queried against ChromaDB. The top-k most semantically similar reference receipts are retrieved and injected as in-context examples to the LLM.

### 4. 🤖 Multi-Prompt LLM Extraction  
Two specialized LLM prompts run sequentially:
- **Header Prompt** – extracts vendor name, date, address, totals, VAT, and currency.
- **Item Prompt** – extracts each line item with quantity, unit price, and total.

This separation reduces cross-contamination errors and improves accuracy on complex receipts.

### 5. 🛡️ Hallucination Validation
Every extracted field is validated against the source OCR text. Fields containing values not grounded in the original text are flagged as low-confidence or rejected entirely.

### 6. 📊 Confidence Scoring & Output
Each field receives a confidence score. An `overall_confidence` and `validation_passed` flag are returned alongside the extracted data.

### 7. 🧠 Pattern Learning
Successfully extracted receipts are automatically stored back into the ChromaDB knowledge base, continuously improving retrieval quality for future receipts of similar formats.

---

## 🗺️ Roadmap

- [ ] 📸 Direct image input support (auto-OCR from image uploads)
- [ ] 🌐 Multi-language receipt support
- [ ] 🔐 API key authentication for production deployments
- [ ] 📈 Extraction analytics dashboard
- [ ] 🗄️ PostgreSQL / cloud database integration
- [ ] 🐳 Docker Compose deployment setup

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to your branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ by [Mohammedjaasir](https://github.com/Mohammedjaasir)

⭐ **Star this repo** if you find it useful!

</div>
