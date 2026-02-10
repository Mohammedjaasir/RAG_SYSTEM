"""
Receipt Support Module - Utilities & Helpers

Provides utility functions and helpers:
- Supplier name extraction
- Receipt type detection
- Common utility functions

Exports:
- HybridSupplierExtractor: Supplier name extraction
- get_receipt_type: Receipt type detection utility
"""

from .supplier_extractor import HybridSupplierExtractor
from .get_receipt_type import get_receipt_type

__all__ = [
    'HybridSupplierExtractor',
    'get_receipt_type'
]
