from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path to import receipt_rag
sys.path.insert(0, str(Path(__file__).parent))

from receipt_rag import get_rag_pipeline, RAGResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Receipt RAG Extraction API",
    description="API for extracting structured data from receipts using RAG and multi-prompt LLM strategy.",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ExtractionRequest(BaseModel):
    ocr_text: str = Field(..., description="The full OCR text content of the receipt")
    retrieve_k: int = Field(3, description="Number of context examples to retrieve")

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup."""
    try:
        get_rag_pipeline()
        logger.info("RAG Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {e}")

@app.post("/extract", response_model=Dict[str, Any])
async def extract_receipt(request: ExtractionRequest):
    """
    Extract structured data from receipt OCR text.
    Uses a two-prompt strategy for better accuracy.
    """
    try:
        pipeline = get_rag_pipeline()
        result = pipeline.extract_from_ocr(
            ocr_text=request.ocr_text, 
            retrieve_k=request.retrieve_k
        )
        
        # Format response
        response = result.extracted_data.copy()
        
        # Add metadata
        response['_metadata'] = {
            'overall_confidence': result.confidence,
            'relevance_scores': result.relevance_scores,
            'context_used': result.context_used[:500] + "..." if len(result.context_used) > 500 else result.context_used
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clean-db")
async def clean_database():
    """
    Clear the vector database and force a rebuild on next request.
    Use this when the knowledge base files change significantly.
    """
    try:
        pipeline = get_rag_pipeline()
        pipeline.clean_database()
        return {"status": "success", "message": "Vector database cleared and state reset."}
    except Exception as e:
        logger.error(f"Failed to clean database: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "receipt-rag-api", "version": "1.1.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
