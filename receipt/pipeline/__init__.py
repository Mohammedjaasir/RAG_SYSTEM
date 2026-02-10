"""
Receipt Pipeline Module - Full-Page Structured Receipt Processing

Handles end-to-end receipt processing:
- Complete receipt extraction from PDF/Image
- Structured data processing
- Result formatting

Exports:
- ReceiptPipelineService: Main pipeline service for full-page receipt processing
"""

from .receipt_pipeline_service import ReceiptPipelineService

__all__ = [
    'ReceiptPipelineService'
]
