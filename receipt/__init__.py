"""
Receipt Extraction Module - Complete Receipt Processing System

Organized into 7 functional components:

📂 orchestration/ - Main API entry points
   ├─ ComprehensiveIntegratedExtractor: Main orchestrator (18 files)
   ├─ ComprehensiveReceiptExtractorAPI: Quick API wrapper
   └─ Singleton accessors

📂 extraction/ - Extract specific data types
   ├─ ComprehensiveReceiptDataExtractor: Receipt fields (4,259 lines)
   ├─ ComprehensiveItemExtractor: Item data (1,823 lines)
   ├─ ImprovedVATDataExtractor: VAT information
   └─ AdditionalFieldsExtractor: Payment, discount, contact info

📂 classification/ - Line classification
   ├─ AdvancedFeatureExtractor: Feature extraction (298 lines)
   ├─ AdvancedRuleEngine: Rule-based classification (646 lines)
   ├─ ComprehensiveBatchProcessor: Batch processing (449 lines)
   └─ ComprehensiveCSVConverter: CSV conversion (255 lines)

📂 reconstruction/ - Text order reconstruction
   ├─ LineReconstruction: Main reconstruction (282 lines)
   └─ SortOCRText: Text sorting (232 lines)

📂 standardization/ - Data standardization
   ├─ StandardizedReceiptSchema: Data schema
   └─ FieldStandardizationAdapter: Field normalization

📂 support/ - Utilities & helpers
   ├─ SupplierExtractor: Supplier extraction
   └─ get_receipt_type: Receipt type detection

📂 pipeline/ - Full-page processing
   └─ ReceiptPipelineService: End-to-end pipeline

QUICK START:
    from app.services.extraction.receipt import (
        get_comprehensive_integrated_extractor,
        get_receipt_type
    )
    
    extractor = get_comprehensive_integrated_extractor()
    result = extractor.extract_comprehensive_data_from_csv("classified.csv")
"""

# Import main orchestrators
from .orchestration import (
    ComprehensiveIntegratedExtractor,
    get_comprehensive_integrated_extractor,
    ComprehensiveReceiptExtractorAPI,
    get_comprehensive_extractor_service
)

# Import core extractors
from .extraction import (
    ComprehensiveReceiptDataExtractor,
    PhiItemExtractor,
    ImprovedVATDataExtractor,
    AdditionalFieldsExtractor
)

# Import classifiers
from .classification import (
    AdvancedFeatureExtractor,
    AdvancedRuleEngine,
    ComprehensiveBatchProcessor,
    ComprehensiveCSVConverter
)

# Import reconstruction services
from .reconstruction import (
    OCRTextBox,
    OCRTextLine
)

# Import standardization
from .standardization import (
    StandardizedReceiptSchema,
    FieldStandardizationAdapter,
    FieldMigrationMapping,
    EXTRACTION_PRIORITY,
    REQUIRED_FIELDS
)

# Import support utilities
from .support import (
    HybridSupplierExtractor,
    get_receipt_type
)

# Import pipeline
from .pipeline import (
    ReceiptPipelineService
)

__all__ = [
    # Orchestration
    'ComprehensiveIntegratedExtractor',
    'get_comprehensive_integrated_extractor',
    'ComprehensiveReceiptExtractorAPI',
    'get_comprehensive_extractor_service',
    
    # Extraction
    'ComprehensiveReceiptDataExtractor',
    'ComprehensiveItemExtractor',
    'ImprovedVATDataExtractor',
    'AdditionalFieldsExtractor',
    
    # Classification
    'AdvancedFeatureExtractor',
    'AdvancedRuleEngine',
    'ComprehensiveBatchProcessor',
    'ComprehensiveCSVConverter',
    
    # Reconstruction
    'OCRTextBox',
    'OCRTextLine',
    
    # Standardization
    'StandardizedReceiptSchema',
    'FieldStandardizationAdapter',
    'FieldMigrationMapping',
    'EXTRACTION_PRIORITY',
    'REQUIRED_FIELDS',
    
    # Support
    'HybridSupplierExtractor',
    'get_receipt_type',
    
    # Pipeline
    'ReceiptPipelineService'
]
