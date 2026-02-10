"""
Extraction Models - Data classes for all extraction results
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class ExtractionResult:
    """Base result class for all extracted fields."""
    value: Any
    raw_text: str
    confidence: float
    line_number: int
    extraction_method: str
    pattern_used: Optional[str] = None
    pattern_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TotalExtractionResult:
    """Specialized result for totals."""
    amount: float
    raw_text: str
    confidence: float
    line_number: int
    extraction_method: str
    rule_priority: int
    pattern_used: str
    field_type: str
    validation_flags: List[str] = field(default_factory=list)


@dataclass
class SupplierExtractionResult:
    """Specialized result for supplier extraction."""
    supplier_name: str
    raw_text: str
    confidence: float
    line_number: int
    extraction_method: str
    matched_config_supplier: Optional[str] = None
    reasons: List[str] = field(default_factory=list)
    position: Optional[int] = None
    cleaned_from: Optional[str] = None


@dataclass
class ReceiptStructureAnalysis:
    """Structure analysis results."""
    receipt_type: str = field(default='unknown')
    tax_system: str = field(default='unknown')
    currency_detected: Optional[str] = field(default=None)
    locale_indicators: List[str] = field(default_factory=list)
    
    discount_analysis: Dict[str, Any] = field(default_factory=lambda: {
        'has_discounts': False,
        'discount_types': [],
        'discount_indicators': [],
        'confidence': 0.0
    })
    
    tax_structure: Dict[str, Any] = field(default_factory=lambda: {
        'has_vat': False,
        'has_sales_tax': False,
        'tax_rate_detected': None,
        'tax_inclusive': False
    })
    
    format_complexity: str = field(default='simple')
    structural_elements: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComprehensiveExtractionResult:
    """Complete extraction result container."""
    extraction_status: str
    file_info: Dict[str, Any]
    extracted_data: Dict[str, Any]
    extraction_stats: Dict[str, Any]
    confidence_scores: Dict[str, float]
    token_level_confidence: Dict[str, List[Dict[str, Any]]]
    structure_analysis: Optional[ReceiptStructureAnalysis] = field(default=None)
    totals: Optional[Dict[str, Any]] = field(default=None)
    additional_fields: Optional[Dict[str, Any]] = field(default=None)