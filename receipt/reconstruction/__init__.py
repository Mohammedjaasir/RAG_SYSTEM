"""
Receipt Reconstruction Module - Text Order Reconstruction

Reconstructs proper order and structure of OCR extracted text:
- Sort lines by position
- Handle multi-line items
- Reconstruct table structure

Exports:
- OCRTextBox: Text box representation
- OCRTextLine: Text line representation
"""

from .sort_ocr_text import OCRTextBox, OCRTextLine

__all__ = [
    'OCRTextBox',
    'OCRTextLine'
]
