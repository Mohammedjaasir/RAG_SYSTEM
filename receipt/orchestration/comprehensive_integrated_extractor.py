#!/usr/bin/env python3
"""
Comprehensive Integrated Extractor - API Version v1.4.0
Combines enhanced item extraction + full receipt data + enhanced VAT extraction + additional fields
Optimized for API integration with direct CSV processing

Version 1.4.0: Enhanced Item Extraction with Sophisticated Pattern Recognition
- Improved item extraction using reference script patterns with 85-100% accuracy
- Fixed line_type vs predicted_class classification issue for proper item detection
- Enhanced non-item line filtering to avoid payment/tax/total lines
- Better quantity detection (integers only) with improved unit handling
- Comprehensive item name cleaning with multi-stage pattern removal
- Maintained VAT Information System with @ symbol detection from v1.3.0
Version 1.2.0: Enhanced Supplier-Context Aware Address Extraction
- Integrated supplier context passing to additional fields extractor
- Enhanced address extraction using supplier-address relationship patterns
- Improved field standardization for complex address structures
- Added contextual contact information extraction with intelligent boundaries
"""
import pandas as pd
import numpy as np
import json
import re
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all extractors from receipt/extraction folder
from ..extraction.comprehensive_receipt_extractor import ComprehensiveReceiptDataExtractor
from ..extraction.improved_vat_extractor import ImprovedVATDataExtractor
from ..extraction.additional_fields_extractor import AdditionalFieldsExtractor
from ..extraction.phi_item_extractor import get_phi_item_extractor

# Import standardization components from receipt/standardization folder
from ..standardization.field_adapter import FieldStandardizationAdapter
from ..standardization.standardized_schema import (
    StandardizedReceiptSchema, 
    FieldMigrationMapping,
    EXTRACTION_PRIORITY,
    REQUIRED_FIELDS
)

class ComprehensiveIntegratedExtractor:
    """
    Comprehensive Integrated Extractor v1.2.0
    
    Combines:
    1. Enhanced item extraction (with all fixes)
    2. Complete receipt data extraction (supplier, dates, totals, etc.)
    3. Full VAT data extraction (codes, rates, amounts)
    4. Enhanced contextual address and contact extraction
    
    Version 1.2.0: Enhanced Supplier-Context Aware Address Extraction
    - Integrated supplier context passing for enhanced address detection
    - Added contextual address extraction using supplier-address relationship patterns
    - Enhanced field standardization for complex contact information structures
    - Improved contact information extraction with intelligent boundary detection
    
    Version 1.0.1: Enhanced date extraction with 15 comprehensive patterns
    
    Optimized for API usage with direct CSV processing
    """
    
    def __init__(self, config_path=None, enable_field_standardization=True):
        """
        Initialize all extractors with optional field standardization.
        
        Args:
            config_path: Path to extraction configuration
            enable_field_standardization: Enable Priority 1B standardized field naming
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))),
                                     'config', 'receipt_extraction_config.json')
        
        # Initialize component extractors (item extraction now handled by Phi-3)
        self.receipt_extractor = ComprehensiveReceiptDataExtractor(config_path)
        self.vat_extractor = ImprovedVATDataExtractor()
        self.additional_extractor = AdditionalFieldsExtractor(config_path)
        
        # Initialize field standardization (Priority 1B)
        self.enable_standardization = enable_field_standardization
        if self.enable_standardization:
            self.field_adapter = FieldStandardizationAdapter()
            print("📋 Field standardization enabled (Priority 1B)")
        else:
            self.field_adapter = None
            
        print(f"✅ Initialized Comprehensive Integrated Extractor API v1.4.0")
        print("   • Enhanced Item Extraction: ✅ Ready")
        print("   • Receipt Data Extraction: ✅ Ready") 
        print("   • VAT Data Extraction: ✅ Ready")
        print("   • Enhanced Contextual Address Extraction: ✅ Ready")
    
    def extract_from_ocr_with_phi3_items(self, reconstructed_text: str, csv_file_path: str):
        """
        Extract complete receipt data with Phi-3 item extraction from full OCR text.
        
        This method combines:
        - Phi-3 LLM-based item extraction from full reconstructed OCR text
        - Traditional pattern-based extraction for other fields (from classified CSV)
        
        Args:
            reconstructed_text: Full reconstructed text from line reconstruction (unclassified)
            csv_file_path: Path to the classified line CSV file
            
        Returns:
            Complete receipt data in comprehensive format
        """
        
        if not os.path.exists(csv_file_path):
            return {
                'extraction_status': 'failed',
                'error': f'CSV file not found: {csv_file_path}',
                'extracted_data': {},
                'fields_extracted': 0
            }
        
        if not reconstructed_text or reconstructed_text.strip() == "":
            return {
                'extraction_status': 'failed',
                'error': 'Empty reconstructed text provided',
                'extracted_data': {},
                'fields_extracted': 0
            }
        
        print(f"🔍 Starting comprehensive extraction with Phi-3 item extraction (v1.5.0)...")
        
        try:
            # 1. PHI-3 ITEM EXTRACTION (from full reconstructed text)
            print(f"🤖 Extracting items using Phi-3 LLM...")
            phi_extractor = get_phi_item_extractor()
            phi_items = phi_extractor.extract_items(reconstructed_text)
            
            # Check if Phi-3 failed or returned no items
            if not phi_items:
                print(f"⚠️  Phi-3 extracted 0 items - checking if model is available...")
                if not phi_extractor.model_loaded:
                    print(f"❌ Phi-3 model not loaded! Falling back to CSV-based hybrid extraction...")
                    # Fall back to CSV-based item extraction
                    from ..extraction.comprehensive_item_extractor import ComprehensiveItemExtractor
                    item_extractor = ComprehensiveItemExtractor()
                    csv_extraction_result = item_extractor.extract_items_from_csv(csv_file_path)
                    phi_items = csv_extraction_result.get('item_extraction', {}).get('item_data', [])
                    if phi_items:
                        print(f"✅ CSV-based extraction extracted {len(phi_items)} items (using hybrid method as fallback)")
                        # Mark these as fallback extraction
                        for item in phi_items:
                            item['extraction_method'] = 'csv_hybrid_fallback'
                else:
                    print(f"⚠️  Phi-3 model is loaded but returned 0 items")
            else:
                print(f"✅ Phi-3 extracted {len(phi_items)} items")
            
            item_results = {
                'item_extraction': {
                    'item_headers': [],
                    'item_data': phi_items,
                    'summary': {
                        'total_items': len(phi_items),
                        'extraction_method': 'phi_3_llm',
                        'source': 'phi_item_extractor'
                    }
                }
            }
            
            # 2. COMPLETE RECEIPT DATA EXTRACTION (from classified CSV - unchanged)
            print(f"📄 Extracting receipt data...")
            receipt_results = self.receipt_extractor.extract_from_file(csv_file_path)
            
            # 3. VAT DATA EXTRACTION (from classified CSV - unchanged)
            print(f"🧾 Extracting VAT data...")
            vat_results = self.vat_extractor.extract_vat_data_from_csv(csv_file_path)
            
            # 4. GET DISCOUNT CONTEXT FROM RECEIPT EXTRACTOR
            print(f"🎯 Getting discount context...")
            discount_context = None
            try:
                discount_context = self.receipt_extractor.get_discount_context()
                if discount_context:
                    print(f"   - Has discounts: {discount_context.get('has_discounts', False)}")
            except Exception as e:
                print(f"⚠️ Warning: Could not get discount context: {e}")
            
            # 5. GET SUPPLIER CONTEXT FROM RECEIPT EXTRACTOR
            print(f"🏪 Getting supplier context...")
            supplier_context = None
            try:
                if receipt_results and receipt_results.get('extracted_data'):
                    supplier_data = receipt_results['extracted_data'].get('supplier_name')
                    if supplier_data and hasattr(supplier_data, 'text') and hasattr(supplier_data, 'line_number'):
                        supplier_context = {
                            'supplier_name': supplier_data.text,
                            'line_number': supplier_data.line_number,
                            'confidence': supplier_data.confidence
                        }
            except Exception as e:
                print(f"⚠️ Warning: Could not get supplier context: {e}")
            
            # 6. ADDITIONAL FIELDS EXTRACTION (from classified CSV - unchanged)
            print(f"💳 Extracting additional fields...")
            try:
                additional_results = self.additional_extractor.extract_additional_fields_from_csv(
                    csv_file_path, 
                    discount_context=discount_context,
                    supplier_context=supplier_context
                )
            except Exception as e:
                print(f"⚠️ Warning: Additional fields extraction failed: {e}")
                additional_results = {
                    'payment_details': {'payment_methods': [], 'payment_amounts': [], 'change_details': [], 'card_details': [], 'cash_details': []},
                    'discount_details': {'discounts': [], 'savings': [], 'coupons': [], 'net_after_discount': None},
                    'contact_details': {'addresses': [], 'phone_numbers': [], 'email_addresses': [], 'websites': [], 'fax_numbers': []},
                    'extraction_metadata': {'error': str(e), 'extraction_status': 'failed'}
                }
            
            # 7. BUILD COMPREHENSIVE FORMAT
            extracted_data = {}
            
            # Initialize default values
            extracted_data.update({
                'supplier_name': 'Unknown',
                'receipt_date': None,
                'currency': None,
                'vat_number': None,
                'receipt_number': None,
                'total_amount': None,
                'subtotal_amount': None,
                'supplier_confidence': 0.0,
                'date_confidence': 0.0,
                'currency_confidence': 0.0,
                'vat_confidence': 0.0
            })
            
            # Process receipt data extraction
            if receipt_results:
                basic_data = receipt_results.get('extracted_data', {})
                if basic_data:
                    extracted_data['receipt_data_detailed'] = basic_data
                    self._extract_simple_receipt_fields(basic_data, extracted_data)
                    if extracted_data.get('total_amount'):
                        try:
                            extracted_data['total_amount'] = float(extracted_data['total_amount'])
                        except (ValueError, TypeError):
                            extracted_data['total_amount'] = None
            
            # Add comprehensive field collections
            extracted_data['vat_details'] = {
                'vat_headers': vat_results.get('vat_headers', []),
                'vat_data_entries': vat_results.get('vat_data_entries', []),
                'vat_summary': vat_results.get('vat_summary', {}),
                'extraction_metadata': vat_results.get('extraction_metadata', {})
            }
            
            # Item details (from Phi-3)
            item_extraction = item_results.get('item_extraction', {})
            extracted_data['item_details'] = {
                'item_headers': item_extraction.get('item_headers', []),
                'item_data': item_extraction.get('item_data', []),
                'summary': item_extraction.get('summary', {})
            }
            
            # Additional fields
            extracted_data['payment_details'] = additional_results.get('payment_details', {}) if additional_results else {}
            extracted_data['discount_details'] = additional_results.get('discount_details', {}) if additional_results else {}
            extracted_data['contact_details'] = additional_results.get('contact_details', {}) if additional_results else {}
            
            # Ensure additional field dictionaries have required structure
            for category, lists in {
                'payment_details': ['payment_methods', 'payment_amounts', 'change_details', 'card_details', 'cash_details'],
                'discount_details': ['discounts', 'savings', 'coupons'],
                'contact_details': ['addresses', 'phone_numbers', 'email_addresses', 'websites', 'fax_numbers']
            }.items():
                for list_field in lists:
                    if extracted_data[category].get(list_field) is None:
                        extracted_data[category][list_field] = []
            
            # Calculate total fields extracted
            fields_extracted = self._count_extracted_fields(extracted_data)
            
            # Generate extraction statistics
            extraction_stats = {
                'item_extraction': {
                    'status': 'success' if phi_items else 'no_items',
                    'items_count': len(phi_items)
                },
                'receipt_extraction': {
                    'status': receipt_results.get('extraction_status', 'failed') if receipt_results else 'failed',
                    'fields_count': len(receipt_results.get('extracted_data') or {}) if receipt_results else 0
                },
                'vat_extraction': {
                    'status': vat_results.get('extraction_metadata', {}).get('extraction_status', 'failed') if vat_results else 'failed',
                    'entries_count': vat_results.get('vat_summary', {}).get('total_entries', 0) if vat_results else 0
                },
                'additional_extraction': {
                    'status': additional_results.get('extraction_metadata', {}).get('extraction_status', 'failed') if additional_results else 'failed'
                }
            }
            
            # Build comprehensive results
            comprehensive_results = {
                'extraction_status': 'success',
                'source_file': csv_file_path,
                'extraction_timestamp': datetime.now().isoformat(),
                'extractor_version': 'Comprehensive Integrated Extractor API v1.5.0 - Phi-3 Item Extraction',
                'item_extraction_method': 'phi_3_llm',
                'fields_extracted': fields_extracted,
                'extracted_data': extracted_data,
                'extraction_stats': extraction_stats,
                'confidence_scores': self._extract_confidence_scores(receipt_results, extracted_data),
                'token_level_confidence': receipt_results.get('token_level_confidence', {}) if receipt_results else {},
                'structure_analysis': receipt_results.get('structure_analysis', {}) if receipt_results else {},
                'totals': (receipt_results.get('extracted_data') or {}).get('totals', {}) if receipt_results else {},
                'additional_fields': additional_results if additional_results else {}
            }
            
            # Apply field standardization if enabled
            if self.enable_standardization and self.field_adapter:
                print("📋 Applying field standardization...")
                standardized_data = self.field_adapter.standardize_extractor_output(
                    comprehensive_results['extracted_data'], 
                    'comprehensive_integrated_extractor'
                )
                compatible_data = self.field_adapter.generate_backward_compatible_output(standardized_data)
                comprehensive_results['extracted_data_standardized'] = standardized_data
                comprehensive_results['extracted_data'] = compatible_data
                standardization_report = self.field_adapter.get_conversion_report()
                comprehensive_results['standardization_report'] = standardization_report
                print(f"   ✅ Field standardization completed")
            
            print(f"✅ Comprehensive extraction with Phi-3 completed! {fields_extracted} fields extracted")
            return comprehensive_results
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Error during comprehensive extraction: {e}")
            return {
                'extraction_status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'full_traceback': error_details,
                'source_file': csv_file_path,
                'extraction_timestamp': datetime.now().isoformat(),
                'extracted_data': {},
                'fields_extracted': 0
            }
    
    def extract_comprehensive_data_from_csv(self, csv_file_path):
        """
        Extract complete receipt data from a CSV file (optimized for API use).
        
        Args:
            csv_file_path: Path to the line classification CSV file
            
        Returns:
            Complete receipt data in comprehensive format
        """
        
        if not os.path.exists(csv_file_path):
            return {
                'extraction_status': 'failed',
                'error': f'CSV file not found: {csv_file_path}',
                'extracted_data': {},
                'fields_extracted': 0
            }
        
        print(f"🔍 Starting comprehensive extraction from CSV: {csv_file_path}")
        
        try:
            # NOTE: Item extraction is now handled by Phi-3 in extract_from_ocr_with_phi3_items()
            # This method (extract_comprehensive_data_from_csv) is deprecated for new workflows
            
            # 1. COMPLETE RECEIPT DATA EXTRACTION
            print(f"📄 Extracting receipt data...")
            receipt_results = self.receipt_extractor.extract_from_file(csv_file_path)
            
            # 3. VAT DATA EXTRACTION  
            print(f"🧾 Extracting VAT data...")
            vat_results = self.vat_extractor.extract_vat_data_from_csv(csv_file_path)
            
            # 4. GET DISCOUNT CONTEXT FROM RECEIPT EXTRACTOR
            print(f"🎯 Getting discount context from comprehensive extractor...")
            discount_context = None
            try:
                discount_context = self.receipt_extractor.get_discount_context()
                print(f"📋 Discount context available: {discount_context is not None}")
                if discount_context:
                    print(f"   - Has discounts: {discount_context.get('has_discounts', False)}")
                    print(f"   - Pattern matches: {len(discount_context.get('pattern_matches', []))}")
                    print(f"   - Exclusions found: {len(discount_context.get('exclusions_found', []))}")
            except Exception as e:
                print(f"⚠️ Warning: Could not get discount context: {e}")
                discount_context = None
            
            # 5. GET SUPPLIER CONTEXT FROM RECEIPT EXTRACTOR
            print(f"🏪 Getting supplier context for address extraction...")
            supplier_context = None
            try:
                if receipt_results and receipt_results.get('extracted_data'):
                    supplier_data = receipt_results['extracted_data'].get('supplier_name')
                    if supplier_data and hasattr(supplier_data, 'text') and hasattr(supplier_data, 'line_number'):
                        supplier_context = {
                            'supplier_name': supplier_data.text,
                            'line_number': supplier_data.line_number,
                            'confidence': supplier_data.confidence
                        }
                        print(f"📍 Supplier context: '{supplier_context['supplier_name']}' at line {supplier_context['line_number']}")
                    elif isinstance(supplier_data, dict):
                        # Handle dict format
                        supplier_context = {
                            'supplier_name': supplier_data.get('value', supplier_data.get('text', '')),
                            'line_number': supplier_data.get('line_number'),
                            'confidence': supplier_data.get('confidence', 0.0)
                        }
                        if supplier_context['line_number']:
                            print(f"📍 Supplier context: '{supplier_context['supplier_name']}' at line {supplier_context['line_number']}")
            except Exception as e:
                print(f"⚠️ Warning: Could not get supplier context: {e}")
                supplier_context = None
            
            # 6. ADDITIONAL FIELDS EXTRACTION (with discount and supplier context)
            print(f"💳 Extracting additional fields...")
            try:
                additional_results = self.additional_extractor.extract_additional_fields_from_csv(
                    csv_file_path, 
                    discount_context=discount_context,
                    supplier_context=supplier_context
                )
            except Exception as e:
                print(f"⚠️ Warning: Additional fields extraction failed: {e}")
                # Provide default structure to prevent NoneType errors
                additional_results = {
                    'payment_details': {
                        'payment_methods': [],
                        'payment_amounts': [],
                        'change_details': [],
                        'card_details': [],
                        'cash_details': []
                    },
                    'discount_details': {
                        'discounts': [],
                        'savings': [],
                        'coupons': [],
                        'net_after_discount': None
                    },
                    'contact_details': {
                        'addresses': [],
                        'phone_numbers': [],
                        'email_addresses': [],
                        'websites': [],
                        'fax_numbers': []
                    },
                    'extraction_metadata': {
                        'error': str(e),
                        'extraction_status': 'failed'
                    }
                }
            
            # 5. BUILD COMPREHENSIVE FORMAT
            extracted_data = {}
            
            # Add basic receipt fields from receipt extractor
            # Initialize default values
            extracted_data.update({
                'supplier_name': 'Unknown',
                'receipt_date': None,
                'currency': None,
                'vat_number': None,
                'receipt_number': None,
                'total_amount': None,
                'subtotal_amount': None,
                'supplier_confidence': 0.0,
                'date_confidence': 0.0,
                'currency_confidence': 0.0,
                'vat_confidence': 0.0
            })
            
            # Process receipt data extraction
            if receipt_results:
                basic_data = receipt_results.get('extracted_data', {})
                if basic_data:
                    # Store complex receipt data for reference
                    extracted_data['receipt_data_detailed'] = basic_data
                    
                    # Extract and flatten the main fields for direct access
                    self._extract_simple_receipt_fields(basic_data, extracted_data)
                    
                    # Ensure values are properly formatted
                    if extracted_data.get('total_amount'):
                        try:
                            extracted_data['total_amount'] = float(extracted_data['total_amount'])
                        except (ValueError, TypeError):
                            extracted_data['total_amount'] = None
            
            # Add comprehensive field collections
            
            # VAT details
            extracted_data['vat_details'] = {
                'vat_headers': vat_results.get('vat_headers', []),
                'vat_data_entries': vat_results.get('vat_data_entries', []),
                'vat_summary': vat_results.get('vat_summary', {}),
                'extraction_metadata': vat_results.get('extraction_metadata', {})
            }
            
            # Item details
            item_extraction = item_results.get('item_extraction', {})
            extracted_data['item_details'] = {
                'item_headers': item_extraction.get('item_headers', []),
                'item_data': item_extraction.get('item_data', []),
                'summary': item_extraction.get('summary', {})
            }
            
            # Additional fields - with defensive defaults to prevent None values
            extracted_data['payment_details'] = additional_results.get('payment_details', {}) if additional_results else {}
            extracted_data['discount_details'] = additional_results.get('discount_details', {}) if additional_results else {}
            extracted_data['contact_details'] = additional_results.get('contact_details', {}) if additional_results else {}
            
            # Ensure additional field dictionaries have the required structure
            if not isinstance(extracted_data['payment_details'], dict):
                extracted_data['payment_details'] = {}
            if not isinstance(extracted_data['discount_details'], dict):
                extracted_data['discount_details'] = {}
            if not isinstance(extracted_data['contact_details'], dict):
                extracted_data['contact_details'] = {}
                
            # Ensure nested lists exist to prevent len() on None errors
            for category, lists in {
                'payment_details': ['payment_methods', 'payment_amounts', 'change_details', 'card_details', 'cash_details'],
                'discount_details': ['discounts', 'savings', 'coupons'],
                'contact_details': ['addresses', 'phone_numbers', 'email_addresses', 'websites', 'fax_numbers']
            }.items():
                for list_field in lists:
                    if extracted_data[category].get(list_field) is None:
                        extracted_data[category][list_field] = []
            
            # Calculate total fields extracted
            fields_extracted = self._count_extracted_fields(extracted_data)
            
            # Generate extraction statistics
            extraction_stats = self._generate_extraction_stats(
                receipt_results, item_results, vat_results, additional_results
            )
            
            comprehensive_results = {
                'extraction_status': 'success',
                'source_file': csv_file_path,
                'extraction_timestamp': datetime.now().isoformat(),
                'extractor_version': 'Comprehensive Integrated Extractor API v1.4.0 - Enhanced Item Extraction with Adaptive Pattern Recognition',
                'fields_extracted': fields_extracted,
                'extracted_data': extracted_data,
                'extraction_stats': extraction_stats,
                'confidence_scores': self._extract_confidence_scores(receipt_results, extracted_data),
                
                # Token-level confidence from receipt extractor (same as bank statements/invoices)
                'token_level_confidence': receipt_results.get('token_level_confidence', {}) if receipt_results else {},
                
                # Priority 2B: Add structure analysis and totals from comprehensive extractor
                'structure_analysis': receipt_results.get('structure_analysis', {}) if receipt_results else {},
                'totals': (receipt_results.get('extracted_data') or {}).get('totals', {}) if receipt_results else {},
                
                # Priority 2B: Add additional fields from additional extractor 
                'additional_fields': additional_results if additional_results else {}
            }
            
            # Apply field standardization (Priority 1B)
            if self.enable_standardization and self.field_adapter:
                print("📋 Applying Priority 1B field standardization...")
                
                # Standardize the field names and structures
                standardized_data = self.field_adapter.standardize_extractor_output(
                    comprehensive_results['extracted_data'], 
                    'comprehensive_integrated_extractor'
                )
                
                # Generate backward-compatible version for existing APIs
                compatible_data = self.field_adapter.generate_backward_compatible_output(standardized_data)
                
                # Update results with both versions
                comprehensive_results['extracted_data_standardized'] = standardized_data
                comprehensive_results['extracted_data'] = compatible_data  # Maintain backward compatibility
                
                # Add standardization metadata
                standardization_report = self.field_adapter.get_conversion_report()
                comprehensive_results['standardization_report'] = standardization_report
                
                print(f"   ✅ Field standardization completed")
                if standardization_report['conversion_summary']['fields_migrated'] > 0:
                    print(f"   📊 Migrated {standardization_report['conversion_summary']['fields_migrated']} deprecated fields")
            
            print(f"✅ Comprehensive extraction completed! {fields_extracted} fields extracted")
            return comprehensive_results
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Error during comprehensive extraction: {e}")
            print(f"📋 FULL TRACEBACK:")
            print(error_details)
            return {
                'extraction_status': 'failed',
                'error': str(e),
                'error_type': type(e).__name__,
                'full_traceback': error_details,
                'source_file': csv_file_path,
                'extraction_timestamp': datetime.now().isoformat(),
                'extracted_data': {},
                'fields_extracted': 0
            }
    
    def _extract_simple_receipt_fields(self, basic_data, extracted_data):
        """Extract simple values from complex receipt data structure."""
        
        # Extract supplier name
        if basic_data.get('supplier_name'):
            if isinstance(basic_data['supplier_name'], dict):
                extracted_data['supplier_name'] = basic_data['supplier_name'].get('supplier_name', 'Unknown')
                extracted_data['supplier_confidence'] = basic_data['supplier_name'].get('confidence', 0.0)
            else:
                extracted_data['supplier_name'] = str(basic_data['supplier_name'])
                extracted_data['supplier_confidence'] = 1.0  # Direct value assignment
        
        # Extract receipt date
        if basic_data.get('receipt_date'):
            if isinstance(basic_data['receipt_date'], dict):
                # Extract from dict - key is 'value' not 'receipt_date'
                extracted_data['receipt_date'] = basic_data['receipt_date'].get('value', None)
                extracted_data['date_confidence'] = basic_data['receipt_date'].get('confidence', 0.0)
            else:
                extracted_data['receipt_date'] = basic_data['receipt_date']
        
        # Extract currency
        if basic_data.get('currency') and isinstance(basic_data['currency'], dict):
            extracted_data['currency'] = basic_data['currency'].get('value', 'GBP')
            extracted_data['currency_confidence'] = basic_data['currency'].get('confidence', 0.0)
        elif basic_data.get('currency'):
            extracted_data['currency'] = basic_data['currency']
        
        # Extract VAT number
        if basic_data.get('vat_number'):
            if isinstance(basic_data['vat_number'], dict):
                # Extract from dict - key is 'value' not 'vat_number'
                extracted_data['vat_number'] = basic_data['vat_number'].get('value', None)
                extracted_data['vat_confidence'] = basic_data['vat_number'].get('confidence', 0.0)
            else:
                extracted_data['vat_number'] = basic_data['vat_number']
        
        # Extract receipt number (if available)
        if basic_data.get('receipt_number'):
            if isinstance(basic_data['receipt_number'], dict):
                # Extract from dict - key is 'value' not 'receipt_number'
                extracted_data['receipt_number'] = basic_data['receipt_number'].get('value', None)
                extracted_data['receipt_number_confidence'] = basic_data['receipt_number'].get('confidence', 0.0)
            else:
                extracted_data['receipt_number'] = basic_data['receipt_number']
        
        # Extract transaction numbers
        for field in ['invoice_number', 'transaction_number', 'reference_number', 'auth_code']:
            if basic_data.get(field):
                extracted_data[field] = basic_data[field]
        
        # Extract total amounts (support both legacy and standardized field names - Priority 1B)
        totals = basic_data.get('totals', {})
        
        # Try standardized field names first, then fall back to legacy names
        total_field = totals.get('final_total') or totals.get('net_after_discount') or totals.get('total')
        subtotal_field = totals.get('items_subtotal') or totals.get('subtotal')
        
        if total_field:
            extracted_data['total_amount'] = total_field['amount']
            extracted_data['total_confidence'] = total_field['confidence']
            # Also set payable_amount (same as total_amount for most receipts)
            extracted_data['payable_amount'] = total_field['amount']
            extracted_data['payable_confidence'] = total_field['confidence']
            
        if subtotal_field:
            extracted_data['subtotal_amount'] = subtotal_field['amount']
            
        # Also add standardized fields to extracted_data
        if totals:
            extracted_data['totals'] = totals  # Include the full totals structure
            
    def _count_extracted_fields(self, extracted_data):
        """Count total extracted fields."""
        count = 0
        
        # Basic receipt fields
        basic_fields = ['supplier_name', 'receipt_date', 'currency', 'vat_number', 'receipt_number', 'total_amount', 'subtotal_amount']
        for field in basic_fields:
            if extracted_data.get(field):
                count += 1
        
        # VAT fields - defensive against None values
        vat_entries = extracted_data.get('vat_details', {}).get('vat_data_entries', [])
        if vat_entries is not None:
            count += len(vat_entries)
        
        # Item fields - defensive against None values
        item_data = extracted_data.get('item_details', {}).get('item_data', [])
        if item_data is not None:
            count += len(item_data)
        
        # Payment fields - defensive against None values
        payment_methods = extracted_data.get('payment_details', {}).get('payment_methods', [])
        if payment_methods is not None:
            count += len(payment_methods)
        
        # Contact fields - defensive against None values
        addresses = extracted_data.get('contact_details', {}).get('addresses', [])
        phones = extracted_data.get('contact_details', {}).get('phone_numbers', [])
        emails = extracted_data.get('contact_details', {}).get('email_addresses', [])
        
        # Safely count contact details
        if addresses is not None:
            count += len(addresses)
        if phones is not None:
            count += len(phones)
        if emails is not None:
            count += len(emails)
        
        return count
    
    def _generate_extraction_stats(self, receipt_results, item_results, vat_results, additional_results):
        """Generate comprehensive extraction statistics."""
        stats = {
            'receipt_extraction': {
                'status': receipt_results.get('extraction_status', 'failed') if receipt_results else 'failed',
                'fields_count': len(receipt_results.get('extracted_data') or {}) if receipt_results else 0
            },
            'item_extraction': {
                'status': 'success' if item_results and item_results.get('item_extraction') else 'failed',
                'items_count': len((item_results.get('item_extraction') or {}).get('item_data') or []) if item_results else 0
            },
            'vat_extraction': {
                'status': vat_results.get('extraction_metadata', {}).get('extraction_status', 'failed') if vat_results else 'failed',
                'entries_count': vat_results.get('vat_summary', {}).get('total_entries', 0) if vat_results else 0
            },
            'additional_extraction': {
                'status': additional_results.get('extraction_metadata', {}).get('extraction_status', 'failed') if additional_results else 'failed',
                'payment_methods': len((additional_results.get('payment_details') or {}).get('payment_methods') or []) if additional_results else 0,
                'contact_details': len((additional_results.get('contact_details') or {}).get('addresses') or []) if additional_results else 0
            }
        }
        return stats
    
    def _extract_confidence_scores(self, receipt_results, extracted_data):
        """Extract confidence scores from receipt results for web display."""
        confidence_scores = {}
        
        if receipt_results and receipt_results.get('extraction_stats'):
            confidence_scores = receipt_results['extraction_stats'].get('confidence_scores', {})
        
        # Add item confidence scores
        item_data = extracted_data.get('item_details', {}).get('item_data', [])
        if item_data:
            # Average confidence for items
            total_confidence = sum(item.get('confidence', 0.0) for item in item_data)
            avg_confidence = total_confidence / len(item_data) if item_data else 0.0
            confidence_scores['items'] = avg_confidence
        
        return confidence_scores


# Service instance for API integration
_comprehensive_integrated_extractor = None

def get_comprehensive_integrated_extractor():
    """Get or create the comprehensive integrated extractor service instance."""
    global _comprehensive_integrated_extractor
    if _comprehensive_integrated_extractor is None:
        _comprehensive_integrated_extractor = ComprehensiveIntegratedExtractor()
    
    return _comprehensive_integrated_extractor
