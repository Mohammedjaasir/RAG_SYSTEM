"""
Receipt Standardization Module - Format & Normalize Data

Standardizes receipt data format:
- Schema formatting
- Data normalization
- Field mapping and adaptation
- Confidence score assignment

Exports:
- StandardizedReceiptSchema: Receipt data schema
- FieldStandardizationAdapter: Field mapping and standardization
- FieldMigrationMapping: Field migration utilities
- EXTRACTION_PRIORITY: Priority definitions
- REQUIRED_FIELDS: Required fields list
"""

from .standardized_schema import (
    StandardizedReceiptSchema,
    FieldMigrationMapping,
    EXTRACTION_PRIORITY,
    REQUIRED_FIELDS
)
from .field_adapter import FieldStandardizationAdapter

__all__ = [
    'StandardizedReceiptSchema',
    'FieldStandardizationAdapter',
    'FieldMigrationMapping',
    'EXTRACTION_PRIORITY',
    'REQUIRED_FIELDS'
]
