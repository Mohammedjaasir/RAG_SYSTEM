"""
Receipt Orchestration Module - Main API Entry Points

Provides high-level interfaces for receipt extraction operations.

Exports:
- ComprehensiveIntegratedExtractor: Main orchestrator combining all extraction services
- get_comprehensive_integrated_extractor(): Get singleton instance
- ComprehensiveReceiptExtractorAPI: Quick API for basic extraction
- get_comprehensive_extractor_service(): Get singleton API instance
"""

from .comprehensive_integrated_extractor import (
    ComprehensiveIntegratedExtractor,
    get_comprehensive_integrated_extractor
)

from .comprehensive import (
    ComprehensiveReceiptExtractorAPI,
    get_comprehensive_extractor_service
)

__all__ = [
    'ComprehensiveIntegratedExtractor',
    'get_comprehensive_integrated_extractor',
    'ComprehensiveReceiptExtractorAPI',
    'get_comprehensive_extractor_service'
]
