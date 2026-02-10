"""
Receipt Pipeline Service for full-page receipt processing

Wraps around pipeline_receipt.py to process structured receipts (Ebay, Acto style)
and extract receipt data in a standardized format.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import shutil

logger = logging.getLogger(__name__)


class ReceiptPipelineService:
    """Service for processing full-page receipts using the receipt pipeline"""
    
    def __init__(self):
        """Initialize receipt pipeline service with required dependencies"""
        self.pipeline = None
        self.extractor = None
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize pipeline and extractor if available"""
        try:
            from pipeline_final_refactored import InvoicePipeline
            self.pipeline = InvoicePipeline
            logger.info("✅ InvoicePipeline available for receipt processing")
        except ImportError as e:
            logger.warning("⚠️ InvoicePipeline not available: %s", e)
            self.pipeline = None
        
        # Note: ReceiptDataExtractor not found in codebase
        # Using comprehensive extractor instead for data extraction
        try:
            from app.services.extraction.receipt.orchestration.comprehensive import ComprehensiveReceiptExtractorAPI
            self.extractor = ComprehensiveReceiptExtractorAPI
            logger.info("✅ ComprehensiveReceiptExtractorAPI available for data extraction")
        except ImportError as e:
            logger.warning("⚠️ ComprehensiveReceiptExtractorAPI not available: %s", e)
            self.extractor = None
    
    def is_available(self) -> bool:
        """Check if receipt pipeline is available"""
        return self.pipeline is not None
    
    def process_receipt(self, document_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Process a full-page receipt document
        
        Args:
            document_path: Path to the receipt document (PDF or image)
            output_dir: Directory to save output files
            
        Returns:
            Dictionary with processing results
        """
        try:
            if not self.pipeline:
                return {
                    'success': False,
                    'error': 'Pipeline not available',
                    'data': {}
                }
            
            logger.info("🔄 Processing receipt: %s", document_path)
            
            # Initialize and run pipeline
            pipeline_instance = self.pipeline(
                input_path=document_path,
                output_dir=output_dir
            )
            
            # Run processing
            pipeline_instance.process()
            
            # Extract data
            receipt_data = self._extract_receipt_data(pipeline_instance)
            
            logger.info("✅ Receipt pipeline processing completed")
            
            return {
                'success': True,
                'error': None,
                'data': receipt_data,
                'output_dir': str(pipeline_instance.processing_dir)
            }
            
        except (ImportError, AttributeError, IOError) as e:
            logger.error("❌ Receipt pipeline failed: %s", str(e))
            return {
                'success': False,
                'error': str(e),
                'data': {}
            }
    
    def _extract_receipt_data(self, pipeline_instance) -> Dict[str, Any]:
        """Extract receipt data from pipeline output"""
        try:
            processing_dir = pipeline_instance.processing_dir
            
            # Look for invoice info JSON
            invoice_json = None
            for file_path in Path(processing_dir).glob("*_invoice_info.json"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    invoice_json = json.load(f)
                break
            
            if not invoice_json:
                logger.warning("No invoice data found in pipeline output")
                return {}
            
            # Extract structured data
            receipt_data = {
                'items': invoice_json.get('items', []),
                'amounts': invoice_json.get('amounts', {}),
                'vendor': invoice_json.get('vendor', {}),
                'date': invoice_json.get('date', ''),
                'reference_number': invoice_json.get('reference_number', ''),
                'raw_data': invoice_json
            }
            
            return receipt_data
            
        except (IOError, json.JSONDecodeError) as e:
            logger.warning("Error extracting receipt data: %s", str(e))
            return {}
    
    def convert_pipeline_to_comprehensive_format(self, pipeline_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert receipt pipeline output to comprehensive format compatible with API
        
        Args:
            pipeline_output: Output from process_receipt()
            
        Returns:
            Dictionary in comprehensive format
        """
        data = pipeline_output.get('data', {})
        
        # Build comprehensive format
        comprehensive = {
            'document_type': 'receipt',
            'receipt_type': 'FULL_PAGE',
            'items': [],
            'totals': {},
            'vendor_info': {},
            'metadata': {
                'source': 'receipt_pipeline',
                'processing_dir': pipeline_output.get('output_dir', '')
            }
        }
        
        # Extract items
        if 'items' in data and isinstance(data['items'], list):
            comprehensive['items'] = data['items']
        
        # Extract amounts
        if 'amounts' in data and isinstance(data['amounts'], dict):
            comprehensive['totals'] = data['amounts']
        
        # Extract vendor info
        if 'vendor' in data and isinstance(data['vendor'], dict):
            comprehensive['vendor_info'] = data['vendor']
        
        # Add date if present
        if 'date' in data and data['date']:
            comprehensive['metadata']['date'] = data['date']
        
        # Add reference number if present
        if 'reference_number' in data and data['reference_number']:
            comprehensive['metadata']['reference_number'] = data['reference_number']
        
        return comprehensive
    
    def copy_output_files(self, source_dir: str, destination_dir: str) -> bool:
        """Copy pipeline output files to destination directory"""
        try:
            source = Path(source_dir)
            dest = Path(destination_dir)
            
            if not source.exists():
                logger.warning("Source directory not found: %s", source)
                return False
            
            dest.mkdir(parents=True, exist_ok=True)
            
            # Copy all files from source to destination
            for item in source.iterdir():
                if item.is_file():
                    shutil.copy2(item, dest / item.name)
                elif item.is_dir():
                    shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
            
            logger.info("✅ Copied output files from %s to %s", source, dest)
            return True
            
        except (IOError, OSError) as e:
            logger.error("❌ Error copying output files: %s", str(e))
            return False
