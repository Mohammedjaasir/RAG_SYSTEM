#!/usr/bin/env python3
"""
Standardized Schema for Receipt Extraction - Priority 1B Implementation
Defines consistent field names, data structures, and interfaces between extractors.

This schema ensures:
1. Consistent field naming across all extractors
2. Standardized data structures for each field type
3. Clear separation between business logic and UI presentation
4. Backward compatibility during migration
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime

# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class AmountField:
    """Standardized structure for all monetary amounts."""
    amount: Optional[float]
    raw_text: str
    confidence: float
    line_number: int
    extraction_method: str
    currency: Optional[str] = "GBP"
    field_type: Optional[str] = None
    pattern_used: Optional[str] = None
    
    def __format__(self, format_spec: str) -> str:
        """Support for f-string formatting."""
        if format_spec:
            # If a format spec is provided, apply it to the amount
            return format(self.amount if self.amount is not None else 0, format_spec)
        else:
            # Default string representation
            return str(self.amount if self.amount is not None else 'None')
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'amount': self.amount,
            'raw_text': self.raw_text,
            'confidence': self.confidence,
            'line_number': self.line_number,
            'extraction_method': self.extraction_method,
            'currency': self.currency,
            'field_type': self.field_type,
            'pattern_used': self.pattern_used
        }

@dataclass
class TextField:
    """Standardized structure for text fields."""
    value: str
    raw_text: str
    confidence: float
    line_number: int
    extraction_method: str
    field_type: Optional[str] = None
    pattern_used: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'value': self.value,
            'raw_text': self.raw_text,
            'confidence': self.confidence,
            'line_number': self.line_number,
            'extraction_method': self.extraction_method,
            'field_type': self.field_type,
            'pattern_used': self.pattern_used
        }

@dataclass
class DateField:
    """Standardized structure for date fields."""
    date: Optional[str]  # ISO format YYYY-MM-DD
    raw_text: str
    confidence: float
    line_number: int
    extraction_method: str
    original_format: Optional[str] = None
    parsed_components: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'date': self.date,
            'raw_text': self.raw_text,
            'confidence': self.confidence,
            'line_number': self.line_number,
            'extraction_method': self.extraction_method,
            'original_format': self.original_format,
            'parsed_components': self.parsed_components
        }

@dataclass
class ExtractionMetadata:
    """Standardized metadata for extraction operations."""
    extraction_timestamp: str
    extractor_version: str
    source_file: str
    total_lines_processed: int
    extraction_status: str  # 'success', 'partial', 'failed'
    processing_time_ms: Optional[float] = None
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None

# =============================================================================
# BUSINESS FIELD CATEGORIES
# =============================================================================

class StandardizedReceiptSchema:
    """
    Standardized schema for receipt extraction results.
    
    Field Naming Conventions:
    - Use snake_case for all field names
    - Prefix with category: totals_, payment_, supplier_, discount_, etc.
    - Suffix with _amount for monetary values, _date for dates
    - Be descriptive but concise: items_subtotal vs subtotal
    """
    
    # SUPPLIER INFORMATION
    SUPPLIER_FIELDS = {
        'supplier_name': TextField,
        'supplier_address': List[TextField],
        'supplier_phone': List[TextField], 
        'supplier_email': List[TextField],
        'supplier_website': List[TextField],
        'supplier_vat_number': TextField,
        'supplier_registration_number': TextField
    }
    
    # RECEIPT IDENTIFICATION
    RECEIPT_ID_FIELDS = {
        'receipt_number': TextField,
        'receipt_date': DateField,
        'receipt_time': TextField,
        'transaction_id': TextField,
        'invoice_number': TextField,
        'reference_number': TextField,
        'auth_code': TextField,
        'terminal_id': TextField
    }
    
    # FINANCIAL TOTALS (Most Important for Accounting)
    TOTALS_FIELDS = {
        # Core totals (standardized names)
        'items_subtotal': AmountField,      # Sum of all items before discounts/VAT
        'discounts_total': AmountField,     # Total discounts applied 
        'subtotal_after_discounts': AmountField,  # Subtotal after discounts, before VAT
        'vat_total': AmountField,          # Total VAT amount
        'final_total': AmountField,        # Final amount to pay
        
        # Deprecated fields (for backward compatibility)
        'subtotal': AmountField,           # Maps to items_subtotal  
        'net_after_discount': AmountField, # Maps to final_total
        'total': AmountField,              # Maps to final_total
        'total_amount': AmountField        # Maps to final_total
    }
    
    # DISCOUNT AND SAVINGS DETAILS
    DISCOUNT_FIELDS = {
        'discount_items': List[Dict[str, Any]],  # Individual discount line items
        'coupon_items': List[Dict[str, Any]],    # Coupon/voucher applications
        'loyalty_savings': List[Dict[str, Any]], # Loyalty program savings
        'promotional_offers': List[Dict[str, Any]], # Special offers
        'discount_context': Dict[str, Any]       # Context from comprehensive analysis
    }
    
    # VAT/TAX INFORMATION
    VAT_FIELDS = {
        'vat_items': List[Dict[str, Any]],       # Individual VAT line items
        'vat_rate_breakdown': Dict[str, Any],    # VAT by rate (20%, 5%, etc.)
        'vat_exempt_amount': AmountField,        # VAT-exempt items
        'vat_inclusive_total': AmountField,      # Total including VAT
        'vat_exclusive_total': AmountField       # Total excluding VAT
    }
    
    # PAYMENT INFORMATION
    PAYMENT_FIELDS = {
        'payment_methods': List[Dict[str, Any]], # Card, cash, etc.
        'payment_amounts': List[AmountField],    # Amount per payment method
        'change_given': AmountField,             # Change returned
        'card_details': List[Dict[str, Any]],    # Card-specific info
        'cash_details': List[Dict[str, Any]],    # Cash transaction info
        'tender_amounts': List[AmountField]      # Amount tendered per method
    }
    
    # ITEM DETAILS
    ITEM_FIELDS = {
        'item_list': List[Dict[str, Any]],       # Extracted items with details
        'item_count': int,                       # Number of items
        'category_breakdown': Dict[str, Any],    # Items by category
        'item_extraction_metadata': Dict[str, Any]  # Item extraction stats
    }

# =============================================================================
# FIELD MAPPING AND MIGRATION
# =============================================================================

class FieldMigrationMapping:
    """
    Mapping between old field names and new standardized names.
    Enables backward compatibility during migration.
    """
    
    # Old -> New field mappings
    FIELD_MAPPINGS = {
        # Totals mappings
        'subtotal': 'items_subtotal',
        'total': 'final_total', 
        'total_amount': 'final_total',
        'net_after_discount': 'final_total',
        'grand_total': 'final_total',
        'sub_total': 'items_subtotal',
        'sub-total': 'items_subtotal',
        
        # Discount mappings
        'discount': 'discount_items',
        'discounts': 'discount_items', 
        'savings': 'loyalty_savings',
        'coupons': 'coupon_items',
        'vouchers': 'coupon_items',
        
        # VAT mappings
        'vat': 'vat_total',
        'vat_amount': 'vat_total',
        'tax': 'vat_total',
        'tax_amount': 'vat_total',
        
        # Payment mappings
        'payment_method': 'payment_methods',
        'card_type': 'card_details',
        'card_amount': 'payment_amounts',
        'change': 'change_given',
        
        # Supplier mappings
        'supplier': 'supplier_name',
        'merchant': 'supplier_name',
        'store': 'supplier_name',
        'business_name': 'supplier_name',
        'phone': 'supplier_phone',
        'email': 'supplier_email',
        'address': 'supplier_address',
        
        # Date/ID mappings  
        'date': 'receipt_date',
        'transaction_date': 'receipt_date',
        'receipt_id': 'receipt_number',
        'invoice_id': 'invoice_number'
    }
    
    # UI Display Names (for presentation layer)
    UI_DISPLAY_NAMES = {
        'items_subtotal': 'Sub-Total',
        'discounts_total': 'Discounts',
        'subtotal_after_discounts': 'Subtotal After Discounts', 
        'vat_total': 'VAT',
        'final_total': 'Total',
        'supplier_name': 'Store/Business Name',
        'receipt_date': 'Date',
        'receipt_number': 'Receipt Number',
        'payment_methods': 'Payment Method',
        'change_given': 'Change Given',
        'discount_items': 'Discount Details',
        'vat_items': 'VAT Breakdown'
    }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def migrate_field_name(old_name: str) -> str:
    """Convert old field name to standardized name."""
    return FieldMigrationMapping.FIELD_MAPPINGS.get(old_name, old_name)

def get_ui_display_name(field_name: str) -> str:
    """Get user-friendly display name for field."""
    return FieldMigrationMapping.UI_DISPLAY_NAMES.get(field_name, field_name.replace('_', ' ').title())

def create_amount_field(amount: float, raw_text: str, confidence: float, 
                       line_number: int, extraction_method: str, **kwargs) -> AmountField:
    """Helper to create standardized AmountField."""
    return AmountField(
        amount=amount,
        raw_text=raw_text,
        confidence=confidence,
        line_number=line_number,
        extraction_method=extraction_method,
        **kwargs
    )

def create_text_field(value: str, raw_text: str, confidence: float,
                     line_number: int, extraction_method: str, **kwargs) -> TextField:
    """Helper to create standardized TextField.""" 
    return TextField(
        value=value,
        raw_text=raw_text,
        confidence=confidence,
        line_number=line_number,
        extraction_method=extraction_method,
        **kwargs
    )

def validate_standardized_output(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate extraction output against standardized schema.
    Returns validation results with errors and warnings.
    """
    validation_results = {
        'errors': [],
        'warnings': [],
        'missing_fields': [],
        'deprecated_fields': []
    }
    
    # Check for deprecated field names
    for field_name in data.keys():
        if field_name in FieldMigrationMapping.FIELD_MAPPINGS:
            validation_results['deprecated_fields'].append(field_name)
            validation_results['warnings'].append(
                f"Field '{field_name}' is deprecated, use '{FieldMigrationMapping.FIELD_MAPPINGS[field_name]}'"
            )
    
    # Check for required field structures
    for field_name, field_value in data.items():
        if field_name.endswith('_amount') and field_value is not None:
            if not isinstance(field_value, (dict, AmountField)):
                validation_results['errors'].append(
                    f"Amount field '{field_name}' should be AmountField or dict with amount/confidence"
                )
    
    return validation_results

# =============================================================================
# SCHEMA CONSTANTS
# =============================================================================

# Priority order for field extraction (most important first)
EXTRACTION_PRIORITY = [
    'final_total',           # Most critical for accounting
    'items_subtotal',        # Essential for verification  
    'vat_total',            # Required for tax compliance
    'supplier_name',        # Business identification
    'receipt_date',         # Transaction timing
    'discounts_total',      # Discount validation
    'payment_methods',      # Payment verification
    'receipt_number'        # Reference tracking
]

# Fields that should always be present (even if None)
REQUIRED_FIELDS = [
    'final_total', 'items_subtotal', 'vat_total', 'supplier_name', 'receipt_date'
]

# Fields that can be computed from others
COMPUTED_FIELDS = {
    'subtotal_after_discounts': lambda data: (
        data.get('items_subtotal', {}).get('amount', 0) - 
        data.get('discounts_total', {}).get('amount', 0)
    ) if data.get('items_subtotal') and data.get('discounts_total') else None
}
