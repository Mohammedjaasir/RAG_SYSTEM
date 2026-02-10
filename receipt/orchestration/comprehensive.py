#!/usr/bin/env python3
"""
Comprehensive Receipt Data Extractor - API Integration Version

Extracts comprehensive receipt data including:
- Basic fields: supplier_name, receipt_date, receipt_number, currency, vat_number, total_amount
- VAT details: vat_code, vat_rate, vat_amount, net_amount  
- Item details: item_code, item_name, item_quantity, item_unit_price, item_amount, item_vat_code
- Additional fields: discount, savings, net_after_discount, change, payment_method, card_type, card_amount
- Contact info: address, email, phone, website

This is adapted from the comprehensive integrated extractor for API integration.
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

class ComprehensiveReceiptExtractorAPI:
    """
    API-integrated version of the comprehensive receipt extractor.
    Processes line classification CSV files and extracts all receipt fields.
    """
    
    def __init__(self, config_path=None):
        """Initialize the comprehensive extractor."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'config', 'receipt_extraction_config.json')
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Extract configuration components
        self.suppliers = self.config['suppliers']['all_suppliers']
        self.keyword_categories = self.config['keyword_categories']['categories']
        self.extraction_patterns = self.config['extraction_patterns']
        self.extraction_rules = self.config['extraction_rules']
        
        print(f"✅ Initialized Comprehensive Receipt Extractor API")
        print(f"   Config loaded: {self.config_path}")
        print(f"   Suppliers: {len(self.suppliers)}")
        
    def _load_config(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Provide default configuration if config file is not available."""
        return {
            'suppliers': {
                'all_suppliers': [
                    'Morrison', 'MORRISON', 'Morrisons', 'MORRISONS',
                    'TESCO', 'Tesco', 'tesco', 'Tesco Express',
                    'ASDA', 'Asda', 'asda',
                    'Sainsbury', 'Sainsburys', 'SAINSBURY', 'SAINSBURYS',
                    'ALDI', 'Aldi', 'aldi',
                    'LIDL', 'Lidl', 'lidl',
                    'WAITROSE', 'Waitrose', 'waitrose',
                    'M&S', 'MARKS & SPENCER', 'Marks & Spencer',
                    'Co-op', 'CO-OP', 'coop', 'COOP',
                    'SPAR', 'Spar', 'spar',
                    'Iceland', 'ICELAND', 'iceland',
                    'Boots', 'BOOTS', 'boots'
                ]
            },
            'extraction_patterns': {
                'date_patterns': [
                    {'pattern': r'\\b\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}\\b', 'examples': ['24/04/2022']},
                    {'pattern': r'\\b\\d{1,2}\\s+\\d{1,2}\\s+\\d{2,4}\\b', 'examples': ['24 04 2022']},
                    {'pattern': r'\\b\\d{2,4}[/-]\\d{1,2}[/-]\\d{1,2}\\b', 'examples': ['2022/04/24']},
                ]
            },
            'extraction_rules': {
                'supplier_extraction': {'target_classes': ['HEADER']},
                'date_extraction': {'target_classes': ['HEADER', 'FOOTER', 'IGNORE']},
                'currency_extraction': {'target_classes': ['ALL'], 'default_value': 'GBP'}
            }
        }
    
    def _safe_convert(self, value, target_type=float):
        """Safely convert values for JSON serialization."""
        try:
            if pd.isna(value) or value is None:
                return None
            if target_type == float:
                return float(value)
            elif target_type == int:
                return int(float(value))
            elif target_type == str:
                return str(value)
            else:
                return value
        except (ValueError, TypeError):
            return None
    
    def extract_supplier_name(self, df):
        """Extract supplier/vendor name from header lines."""
        header_lines = df[df['line_type'] == 'HEADER'].copy()
        
        best_match = None
        highest_confidence = 0.0
        
        # Look for known suppliers in header lines
        for _, row in header_lines.iterrows():
            text = str(row['cleaned_text']).strip()
            text_clean = re.sub(r'[^\\w\\s&-]', '', text)
            
            for supplier in self.suppliers:
                if supplier.lower() in text_clean.lower():
                    confidence = self._safe_convert(row['confidence'])
                    if confidence and confidence > highest_confidence:
                        best_match = {
                            'supplier_name': text_clean.strip(),
                            'raw_text': text,
                            'confidence': confidence,
                            'line_number': self._safe_convert(row['line_number'], int),
                            'extraction_method': 'pattern_match'
                        }
                        highest_confidence = confidence
        
        if best_match:
            return best_match
        
        # Fallback: return first header line
        if len(header_lines) > 0:
            first_header = header_lines.iloc[0]
            return {
                'supplier_name': str(first_header['cleaned_text']).strip(),
                'raw_text': str(first_header['cleaned_text']).strip(),
                'confidence': self._safe_convert(first_header['confidence']),
                'line_number': self._safe_convert(first_header['line_number'], int),
                'extraction_method': 'first_header_fallback'
            }
        
        return None
    
    def extract_date(self, df):
        """Extract receipt date from various locations."""
        target_classes = ['HEADER', 'FOOTER', 'IGNORE']
        candidate_lines = df[df['line_type'].isin(target_classes)].copy()
        
        date_patterns = [
            r'\\b\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}\\b',
            r'\\b\\d{1,2}\\s+\\d{1,2}\\s+\\d{2,4}\\b',
            r'\\b\\d{2,4}[/-]\\d{1,2}[/-]\\d{1,2}\\b',
            r'\\b\\d{1,2}[.]\\d{1,2}[.]\\d{2,4}\\b'
        ]
        
        for _, row in candidate_lines.iterrows():
            text = str(row['cleaned_text']).strip()
            
            for pattern in date_patterns:
                match = re.search(pattern, text)
                if match:
                    return {
                        'receipt_date': match.group(0),
                        'raw_text': text,
                        'confidence': self._safe_convert(row['confidence']),
                        'line_number': self._safe_convert(row['line_number'], int),
                        'extraction_method': 'pattern_match'
                    }
        
        return None
    
    def extract_currency(self, df):
        """Extract currency information."""
        currency_patterns = [r'GBP|gbp', r'£', r'€', r'\\bE\\b']
        
        for _, row in df.iterrows():
            text = str(row['cleaned_text']).strip()
            
            for pattern in currency_patterns:
                if re.search(pattern, text):
                    currency = 'GBP' if pattern in ['£', r'\\bE\\b'] else 'EUR' if pattern == '€' else 'GBP'
                    return {
                        'currency': currency,
                        'raw_text': text,
                        'confidence': self._safe_convert(row['confidence']),
                        'line_number': self._safe_convert(row['line_number'], int),
                        'extraction_method': 'pattern_match'
                    }
        
        # Default fallback
        return {
            'currency': 'GBP',
            'raw_text': 'Default (UK receipt)',
            'confidence': 0.5,
            'line_number': 0,
            'extraction_method': 'default_fallback'
        }
    
    def extract_vat_number(self, df):
        """Extract VAT registration number."""
        target_classes = ['HEADER', 'VAT_HEADER']
        candidate_lines = df[df['line_type'].isin(target_classes)].copy()
        
        vat_patterns = [
            r'VAT\\s*[Nn]o\\.?\\s*:?\\s*(\\d{9}|\\d{3}\\s*\\d{4}\\s*\\d{2})',
            r'VAT\\s*REG\\s*[Nn]o\\.?\\s*:?\\s*(\\d{9}|\\d{3}\\s*\\d{4}\\s*\\d{2})',
            r'VAT:?\\s*(\\d{9}|\\d{3}\\s*\\d{4}\\s*\\d{2})'
        ]
        
        for _, row in candidate_lines.iterrows():
            text = str(row['cleaned_text']).strip()
            
            for pattern in vat_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return {
                        'vat_number': match.group(1).replace(' ', ''),
                        'raw_text': text,
                        'confidence': self._safe_convert(row['confidence']),
                        'line_number': self._safe_convert(row['line_number'], int),
                        'extraction_method': 'pattern_match'
                    }
        
        return None
    
    def extract_total_amount(self, df):
        """Extract total amount from summary lines."""
        summary_lines = df[df['line_type'] == 'SUMMARY_KEY_VALUE'].copy()
        
        total_patterns = [
            r'TOTAL\\s*:?\\s*£?([0-9]+[.,]\\d{2})',
            r'Total\\s*:?\\s*£?([0-9]+[.,]\\d{2})',
            r'GRAND\\s*TOTAL\\s*:?\\s*£?([0-9]+[.,]\\d{2})',
            r'£([0-9]+[.,]\\d{2})\\s*TOTAL'
        ]
        
        for _, row in summary_lines.iterrows():
            text = str(row['cleaned_text']).strip()
            
            for pattern in total_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    amount_str = match.group(1).replace(',', '.')
                    try:
                        amount = float(amount_str)
                        return {
                            'total_amount': amount,
                            'raw_text': text,
                            'confidence': self._safe_convert(row['confidence']),
                            'line_number': self._safe_convert(row['line_number'], int),
                            'extraction_method': 'pattern_match'
                        }
                    except ValueError:
                        continue
        
        return None
    
    def extract_basic_items(self, df):
        """Extract basic item information."""
        item_lines = df[df['line_type'] == 'ITEM_DATA'].copy()
        
        items = []
        for _, row in item_lines.iterrows():
            text = str(row['cleaned_text']).strip()
            
            # Simple item extraction - extract item name and try to find price
            price_match = re.search(r'([0-9]+[.,]\\d{2})(?!\\d)', text)
            price = None
            if price_match:
                try:
                    price = float(price_match.group(1).replace(',', '.'))
                except ValueError:
                    pass
            
            # Extract item name (text before the price)
            item_name = text
            if price_match:
                item_name = text[:price_match.start()].strip()
            
            # Clean item name
            item_name = re.sub(r'^\\d+\\s+', '', item_name)  # Remove leading numbers
            item_name = item_name.strip()
            
            if item_name and len(item_name) > 2:  # Only meaningful names
                items.append({
                    'item_name': item_name,
                    'item_amount': price,
                    'raw_text': text,
                    'line_number': self._safe_convert(row['line_number'], int),
                    'confidence': self._safe_convert(row['confidence'])
                })
        
        return items
    
    def extract_comprehensive_data(self, csv_file_path):
        """
        Extract comprehensive receipt data from line classification CSV.
        
        Args:
            csv_file_path: Path to the line classification CSV file
            
        Returns:
            Dictionary with all extracted receipt fields
        """
        try:
            # Load CSV file
            df = pd.read_csv(csv_file_path)
            
            # Validate CSV structure
            required_columns = ['line_number', 'cleaned_text', 'line_type', 'confidence']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return {
                    'error': f'Missing required columns: {missing_columns}',
                    'extraction_status': 'failed'
                }
            
            # Extract all basic fields
            supplier_data = self.extract_supplier_name(df)
            date_data = self.extract_date(df)
            currency_data = self.extract_currency(df)
            vat_data = self.extract_vat_number(df)
            total_data = self.extract_total_amount(df)
            items_data = self.extract_basic_items(df)
            
            # Build comprehensive results
            extracted_data = {
                # Basic receipt fields
                'supplier_name': supplier_data['supplier_name'] if supplier_data else None,
                'receipt_date': date_data['receipt_date'] if date_data else None,
                'receipt_number': None,  # To be implemented with advanced patterns
                'currency': currency_data['currency'] if currency_data else 'GBP',
                'vat_number': vat_data['vat_number'] if vat_data else None,
                'total_amount': total_data['total_amount'] if total_data else None,
                
                # VAT details (basic extraction for now)
                'vat_code': None,
                'vat_rate': None,
                'vat_amount': None,
                'net_amount': None,
                
                # Item details (basic)
                'items': items_data,
                'item_count': len(items_data),
                
                # Additional fields (to be implemented)
                'discount': None,
                'savings': None,
                'net_after_discount': None,
                'change': None,
                'payment_method': None,
                'card_type': None,
                'card_amount': None,
                'address': None,
                'email': None,
                'phone': None,
                'website': None
            }
            
            # Calculate confidence scores
            confidence_scores = {}
            extraction_methods = {}
            fields_extracted = 0
            
            for field_name in ['supplier_name', 'receipt_date', 'currency', 'vat_number', 'total_amount']:
                field_data_map = {
                    'supplier_name': supplier_data,
                    'receipt_date': date_data,
                    'currency': currency_data,
                    'vat_number': vat_data,
                    'total_amount': total_data
                }
                
                field_data = field_data_map[field_name]
                if field_data and extracted_data[field_name] is not None:
                    fields_extracted += 1
                    confidence_scores[field_name] = field_data.get('confidence', 0.0)
                    extraction_methods[field_name] = field_data.get('extraction_method', 'unknown')
            
            # Add items to count
            if items_data:
                fields_extracted += len(items_data)
            
            return {
                'extraction_status': 'success',
                'source_file': csv_file_path,
                'extraction_timestamp': datetime.now().isoformat(),
                'extractor_version': 'Comprehensive API v1.0',
                'total_lines_processed': len(df),
                'fields_extracted': fields_extracted,
                'extracted_data': extracted_data,
                'confidence_scores': confidence_scores,
                'extraction_methods': extraction_methods,
                'raw_field_data': {
                    'supplier': supplier_data,
                    'date': date_data,
                    'currency': currency_data,
                    'vat': vat_data,
                    'total': total_data
                }
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'extraction_status': 'failed',
                'source_file': csv_file_path,
                'extraction_timestamp': datetime.now().isoformat()
            }


# Service instance for API integration
_comprehensive_extractor_service = None

def get_comprehensive_extractor_service():
    """Get or create the comprehensive extractor service instance."""
    global _comprehensive_extractor_service
    if _comprehensive_extractor_service is None:
        # Let the API class construct the correct path from its location
        _comprehensive_extractor_service = ComprehensiveReceiptExtractorAPI()
    
    return _comprehensive_extractor_service
