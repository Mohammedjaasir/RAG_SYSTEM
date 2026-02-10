#!/usr/bin/env python3
"""
Comprehensive Receipt Data Extractor - Enhanced API Version v3.2.0
Main Orchestrator - Coordinates all field extractors and modules
"""
import json
import re
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import warnings
from typing import List, Optional
warnings.filterwarnings('ignore')

# Import all modular components
from .comprehensive_fields.field_extractors.supplier_extractor import SupplierExtractor
from .comprehensive_fields.field_extractors.date_extractor import DateExtractor
from .comprehensive_fields.field_extractors.totals_extractor import TotalsExtractor
from .comprehensive_fields.field_extractors.number_extractor import NumberExtractor
from .comprehensive_fields.field_extractors.currency_extractor import CurrencyExtractor
from .comprehensive_fields.field_extractors.vat_extractor import VATExtractor
from .comprehensive_fields.support_modules.discount_detector import DiscountDetector
from .comprehensive_fields.support_modules.structure_analyzer import StructureAnalyzer
from .comprehensive_fields.support_modules.payable_detector import PayableDetector
from .comprehensive_fields.shared_utils.config_manager import ConfigManager
from .comprehensive_fields.shared_utils.text_cleaner import TextCleaner
from .comprehensive_fields.shared_utils.pattern_matcher import PatternMatcher
from .comprehensive_fields.shared_utils.confidence_scorer import ConfidenceScorer
from .comprehensive_fields.models.extraction_models import (
    ExtractionResult, TotalExtractionResult, SupplierExtractionResult,
    ComprehensiveExtractionResult, ReceiptStructureAnalysis
)


class ComprehensiveReceiptDataExtractor:
    """
    Enhanced Comprehensive Receipt Data Extractor v3.2.0
    Main orchestrator that coordinates all field extractors and modules
    """
    
    def __init__(self, config_path=None):
        """Initialize the extractor with all modular components."""
        print(f"✅ Initializing Enhanced Comprehensive Receipt Extractor API v3.2.0")
        
        # Initialize shared utilities
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.text_cleaner = TextCleaner()
        self.pattern_matcher = PatternMatcher()
        self.confidence_scorer = ConfidenceScorer(self.config)
        
        # Initialize field extractors
        self.field_extractors = {
            'supplier': SupplierExtractor(self.config_manager, self.text_cleaner, self.confidence_scorer),
            'date': DateExtractor(self.config_manager, self.pattern_matcher, self.confidence_scorer),
            'totals': TotalsExtractor(self.config_manager, self.confidence_scorer),
            'numbers': NumberExtractor(self.config_manager, self.pattern_matcher, self.confidence_scorer),
            'currency': CurrencyExtractor(self.config_manager, self.pattern_matcher, self.confidence_scorer),
            'vat': VATExtractor(self.config_manager, self.pattern_matcher)
        }
        
        # Initialize support modules
        self.support_modules = {
            'discount': DiscountDetector(),
            'structure': StructureAnalyzer(),
            'payable': PayableDetector()
        }
        
        # Initialize current structure analysis
        self.current_structure_analysis = None
        
        print(f"   ✅ Config loaded: {self.config_manager.config_path}")
        print(f"   ✅ Field extractors initialized: {len(self.field_extractors)}")
        print(f"   ✅ Support modules initialized: {len(self.support_modules)}")
    
    def extract_from_file(self, predictions_file):
        """Main entry point - extract data from a CSV file."""
        try:
            # Load predictions
            df = pd.read_csv(predictions_file)
            print(f"🔄 Processing {len(df)} rows from CSV for comprehensive extraction")
            
            # Run structure analysis
            print(f"🔍 Running structure analysis...")
            self.current_structure_analysis = self.support_modules['structure'].analyze_receipt_structure(df)
            
            # Extract all fields using field extractors
            extracted_data = {}
            
            # 1. Extract supplier name
            print(f"🏪 Extracting supplier name...")
            supplier_result = self.field_extractors['supplier'].extract_supplier_name(df)
            extracted_data['supplier_name'] = supplier_result
            
            # 2. Extract date
            print(f"📅 Extracting receipt date...")
            date_result = self.field_extractors['date'].extract_date(df)
            extracted_data['receipt_date'] = date_result
            
            # 3. Extract currency
            print(f"💰 Extracting currency...")
            currency_result = self.field_extractors['currency'].extract_currency(df)
            extracted_data['currency'] = currency_result
            
            # 4. Extract VAT number
            print(f"🏷️ Extracting VAT number...")
            vat_result = self.field_extractors['vat'].extract_vat_number(df)
            extracted_data['vat_number'] = vat_result
            
            # 5. Extract receipt number
            print(f"🎫 Extracting receipt number...")
            receipt_num_result = self.field_extractors['numbers'].extract_receipt_number(df)
            extracted_data['receipt_number'] = receipt_num_result
            
            # 6. Extract other numbers
            print(f"🔢 Extracting other numbers...")
            extracted_data['invoice_number'] = self.field_extractors['numbers'].extract_invoice_number(df)
            extracted_data['transaction_number'] = self.field_extractors['numbers'].extract_transaction_number(df)
            extracted_data['reference_number'] = self.field_extractors['numbers'].extract_reference_number(df)
            extracted_data['auth_code'] = self.field_extractors['numbers'].extract_auth_code(df)
            
            # 7. Extract totals (with discount context and enhanced payable fallback)
            print(f"🧮 Extracting totals...")
            discount_context = self.support_modules['discount'].get_discount_context(df)
            totals_result = self.field_extractors['totals'].extract_totals_with_enhanced_payable_fallback(df)
            extracted_data['totals'] = totals_result
            
            # Build comprehensive results
            extraction_results = self._build_comprehensive_results(
                df, predictions_file, extracted_data, self.current_structure_analysis
            )
            
            print(f"✅ Comprehensive extraction completed!")
            return extraction_results
            
        except Exception as e:
            print(f"❌ Error during extraction: {e}")
            import traceback
            traceback.print_exc()
            return {
                'extraction_status': 'failed',
                'error': str(e),
                'source_file': str(Path(predictions_file).stem),
                'extracted_data': None
            }
    
    def _build_comprehensive_results(self, df, predictions_file, extracted_data, structure_analysis):
        """Build comprehensive extraction results."""
        # Calculate token-level confidence
        token_confidence = self._extract_token_level_confidence(extracted_data, df)
        
        # Convert ReceiptStructureAnalysis to dictionary for JSON serialization
        structure_analysis_dict = {}
        if structure_analysis:
            if hasattr(structure_analysis, '__dataclass_fields__'):
                # Convert dataclass to dictionary
                structure_analysis_dict = {
                    'receipt_type': structure_analysis.receipt_type,
                    'tax_system': structure_analysis.tax_system,
                    'currency_detected': structure_analysis.currency_detected,
                    'locale_indicators': structure_analysis.locale_indicators,
                    'discount_analysis': structure_analysis.discount_analysis,
                    'tax_structure': structure_analysis.tax_structure,
                    'format_complexity': structure_analysis.format_complexity,
                    'structural_elements': structure_analysis.structural_elements,
                    'confidence_scores': structure_analysis.confidence_scores
                }
            else:
                structure_analysis_dict = structure_analysis
        
        # Build results
        results = {
            'extraction_status': 'success',
            'structure_analysis': structure_analysis_dict,
            'file_info': {
                'source_file': df['source_file'].iloc[0] if 'source_file' in df.columns else str(Path(predictions_file).stem),
                'total_lines': len(df),
                'extraction_timestamp': datetime.now().isoformat(),
                'config_version': self.config.get('metadata', {}).get('version', '3.2.0'),
                'extractor_version': 'Comprehensive v3.2.0 - Modular Orchestrator'
            },
            'extracted_data': extracted_data,
            'extraction_stats': self._calculate_extraction_stats(extracted_data),
            'token_level_confidence': token_confidence
        }
        
        return results
    
    def _extract_token_level_confidence(self, extracted_data, df):
        """Extract token-level confidence for all fields."""
        token_confidence = {}
        
        for field_name, field_data in extracted_data.items():
            if field_data and hasattr(field_data, 'raw_text'):
                text_value = field_data.raw_text if hasattr(field_data, 'raw_text') else str(field_data)
                token_confidence[field_name] = self.confidence_scorer.extract_token_level_confidence(text_value, df)
        
        return token_confidence
    
    def _calculate_extraction_stats(self, extracted_data):
        """Calculate extraction statistics."""
        stats = {
            'fields_extracted': 0,
            'confidence_scores': {},
            'extraction_methods': {},
            'patterns_used': {}
        }
        
        for field_name, field_data in extracted_data.items():
            if field_data:
                if field_name == 'totals':
                    # Handle totals separately - they're TotalExtractionResult objects
                    if hasattr(field_data, 'subtotal') and field_data.subtotal:
                        stats['fields_extracted'] += 1
                        stats['confidence_scores']['subtotal'] = field_data.subtotal.confidence
                        stats['extraction_methods']['subtotal'] = field_data.subtotal.extraction_method
                    if hasattr(field_data, 'final_total') and field_data.final_total:
                        stats['fields_extracted'] += 1
                        stats['confidence_scores']['final_total'] = field_data.final_total.confidence
                        stats['extraction_methods']['final_total'] = field_data.final_total.extraction_method
                    # Also check if totals is a dictionary (backward compatibility)
                    elif isinstance(field_data, dict):
                        if field_data.get('subtotal'):
                            subtotal = field_data['subtotal']
                            stats['fields_extracted'] += 1
                            if isinstance(subtotal, dict):
                                stats['confidence_scores']['subtotal'] = subtotal.get('confidence', 0)
                                stats['extraction_methods']['subtotal'] = subtotal.get('extraction_method', '')
                            elif hasattr(subtotal, 'confidence'):
                                stats['confidence_scores']['subtotal'] = subtotal.confidence
                                stats['extraction_methods']['subtotal'] = subtotal.extraction_method
                        
                        if field_data.get('final_total'):
                            final_total = field_data['final_total']
                            stats['fields_extracted'] += 1
                            if isinstance(final_total, dict):
                                stats['confidence_scores']['final_total'] = final_total.get('confidence', 0)
                                stats['extraction_methods']['final_total'] = final_total.get('extraction_method', '')
                            elif hasattr(final_total, 'confidence'):
                                stats['confidence_scores']['final_total'] = final_total.confidence
                                stats['extraction_methods']['final_total'] = final_total.extraction_method
                elif hasattr(field_data, 'confidence'):
                    stats['fields_extracted'] += 1
                    stats['confidence_scores'][field_name] = field_data.confidence
                    stats['extraction_methods'][field_name] = field_data.extraction_method
        
        return stats
    
    def extract_totals_old_broken_method(self, df):
        """OLD BROKEN METHOD - kept for backward compatibility."""
        print("⚠️ Using old broken method (deprecated)")
        # This would delegate to the totals extractor's legacy method
        return {'total': None, 'subtotal': None}
    
    def extract_token_level_confidence(self, text_value, df):
        """Public method for token-level confidence extraction."""
        return self.confidence_scorer.extract_token_level_confidence(text_value, df)