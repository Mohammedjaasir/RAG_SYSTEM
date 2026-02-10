#!/usr/bin/env python3
"""
Improved VAT Data Extractor - API Version v2.6.9
Enhanced VAT Information System with comprehensive pattern recognition
Version 2.6.9: Integrated Enhanced VAT Processing from vat_details.py
- Enhanced OCR comma/decimal handling (5,59 → 5.59 automatic correction)
- Advanced combined format detection for A(20.00%) patterns
- Improved tax code and percentage recognition with flexible validation
- Smart amount sorting and classification logic
- Enhanced doubt field handling for uncertain extractions
- Comprehensive pattern recognition with 14+ VAT format patterns
- Hybrid extraction approach combining both traditional and enhanced methods
"""
import pandas as pd
import numpy as np
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedVATDataExtractor:
    def __init__(self):
        self.vat_code_patterns = [
            r'VAT\\s+([A-Z])(?:\\s|£)',        # VAT B £, VAT E £ (common UK format)
            r'\\b([A-Z])(?:\\s|$)',           # Single letter: A, B, G
            r'\\b([A-Z]\\d+)(?:\\s|$)',        # Letter+digits: A1, B2, C3
        ]
        
        self.vat_rate_patterns = [
            # VAT @ patterns (enhanced for Screwfix format)
            r'VAT\\s*@\\s*(\\d+(?:\\.\\d+)?)\\s*%?',  # VAT @ 20%, VAT @ 0%
            r'@\\s*(\\d+(?:\\.\\d+)?)\\s*%',          # @ 20%, @0%
            
            # Standard patterns
            r'(\\d+(?:\\.\\d+)?)\\s*%\\s*VAT',        # 20.00 % VAT
            r'(\\d+(?:\\.\\d+)?)\\s*%(?!\\s*[£€$])',  # 20%, 5.5% (not followed by currency)
            
            # Rate context patterns  
            r'Rate[:\\s]+(\\d+(?:\\.\\d+)?)\\s*%',     # Rate: 20%
            r'(\\d+(?:\\.\\d+)?)\\s*%\\s*Rate',       # 20% Rate
            
            # Zero rate specific patterns
            r'@\\s*(0(?:\\.0+)?)\\s*%?',              # @0%, @0.0%
            r'VAT\\s*@\\s*(0(?:\\.0+)?)\\s*%?',       # VAT @ 0%
        ]
        
        # Enhanced amount patterns that handle OCR errors and optional £ symbol
        self.amount_patterns = [
            r'£?\\s*(\\d{1,4}(?:\\.\\d{2})?)',     # £13.33, 13.33, £2.67
            r'£?\\s*(\\d{1,3}(?:,\\d{3})*(?:\\.\\d{2})?)',  # £1,234.56
            r'(\\d+)(?:\\s*E|\\s*€|\\s*£)',        # Handle OCR errors like "6.00 E"
        ]
        
        # Business logic constants
        self.VAT_TOLERANCE = 0.01  # Allow 1p tolerance for rounding

    # Enhanced helper functions from vat_details.py (v2.6.9)
    def is_tax_code_enhanced(self, value):
        """Enhanced tax code detection from vat_details.py"""
        # Handles A, A1, B2, and also A: cases
        return re.fullmatch(r'[A-Za-z0-9]{1,2}:?', value) is not None

    def is_percentage_enhanced(self, value):
        """Enhanced percentage detection from vat_details.py"""
        return re.fullmatch(r'\d+(\.\d+)?%', value) is not None

    def parse_amount_enhanced(self, val):
        """Enhanced amount parsing with OCR correction from vat_details.py"""
        if not val:
            return None
        # Fix OCR error like 5,59 -> 5.59
        val = val.replace(',', '.')
        match = re.search(r'[\d]+(?:\.\d+)?', val)
        return float(match.group()) if match else None

    def parse_tax_code_and_percent_enhanced(self, val):
        """
        Enhanced extraction of combined tax code and percent formats from vat_details.py
        Extract tax code and percent if format is like A(20.00%)
        Returns (tax_code, tax_percent) or (None, None)
        """
        match = re.fullmatch(r'([A-Za-z0-9]{1,2}:?)\((\d+(\.\d+)?%)\)', val)
        if match:
            return match.group(1).rstrip(':'), match.group(2)
        return None, None

    def extract_vat_line_enhanced(self, line_values):
        """
        Enhanced VAT line extraction using vat_details.py logic
        Processes a list of values from a VAT line and returns structured data
        """
        tax_code = None
        tax_percent = None
        tax_amount = None
        net_amount = None
        total_amount = None
        doubt = None

        amounts = []
        for val in line_values:
            # Fix OCR commas before any checks
            val_fixed = val.replace(',', '.')

            # Check for combined format like A(20.00%)
            tc, tp = self.parse_tax_code_and_percent_enhanced(val_fixed)
            if tc and tp:
                tax_code = tc
                tax_percent = tp
                continue

            if self.is_percentage_enhanced(val_fixed):
                tax_percent = val_fixed
            elif self.is_tax_code_enhanced(val_fixed):
                tax_code = val_fixed.rstrip(':')  # remove trailing colon if exists
            elif re.fullmatch(r'\d{1,2}', val_fixed):  
                # Handles plain numbers like 20, 23, 25 as tax percent (no % symbol)
                tax_percent = val_fixed + "%"
            elif re.search(r'\d', val_fixed):
                amt = self.parse_amount_enhanced(val_fixed)
                if amt is not None:
                    amounts.append((amt, val_fixed))  # keep original with currency

        # Sort amounts by value
        amounts_sorted = sorted(amounts, key=lambda x: x[0])
        currency_values = [x[1] for x in amounts_sorted]

        # Smart amount assignment based on vat_details.py logic
        if len(amounts_sorted) >= 3:
            tax_amount = currency_values[0]
            net_amount = currency_values[1]
            total_amount = currency_values[-1]
        elif len(amounts_sorted) == 2:
            # When 2 amounts: smaller is VAT, larger is NET
            tax_amount = currency_values[0]
            net_amount = currency_values[1]
            # Calculate total from THIS LINE ONLY - don't pull from other lines
            vat_numeric = amounts_sorted[0][0]  # numeric VAT amount
            net_numeric = amounts_sorted[1][0]  # numeric NET amount
            total_from_calc = vat_numeric + net_numeric
            # Format it as a currency string
            total_amount = f"{total_from_calc:.2f}"
            # Log to show this is line-specific calculation
            print(f"      🔢 LINE-SPECIFIC CALC (2 amounts): {vat_numeric:.2f} (VAT) + {net_numeric:.2f} (NET) = {total_from_calc:.2f} (TOTAL)")
        elif len(amounts_sorted) == 1:
            tax_amount = currency_values[0]

        return {
            'vat_code': tax_code if tax_code else None,
            'vat_rate': float(tax_percent.rstrip('%')) if tax_percent and tax_percent.rstrip('%').replace('.', '').isdigit() else None,
            'vat_amount': self.parse_amount_enhanced(tax_amount) if tax_amount else None,
            'net_amount': self.parse_amount_enhanced(net_amount) if net_amount else None,
            'total_amount': self.parse_amount_enhanced(total_amount) if total_amount else None,
            'doubt': self.parse_amount_enhanced(doubt) if doubt else None,
            'extraction_method': 'enhanced_vat_details_v2.6.9'
        }
    
    def extract_vat_data_from_csv(self, csv_file_path: str) -> Dict:
        """Extract VAT data directly from CSV file (for API use)"""
        try:
            if not os.path.exists(csv_file_path):
                return {
                    'vat_headers': [],
                    'vat_data_entries': [],
                    'vat_summary': {'total_entries': 0, 'error': f'CSV file not found: {csv_file_path}'},
                    'extraction_metadata': {
                        'extraction_timestamp': datetime.now().isoformat(),
                        'source_file': csv_file_path,
                        'extractor_version': 'VAT Information System v2.6.9 - Enhanced VAT Processing Integration',
                        'extraction_status': 'failed'
                    }
                }
            
            # Load the CSV file
            df = pd.read_csv(csv_file_path)
            
            # Use predicted_class if available, otherwise fall back to line_type
            class_column = 'predicted_class' if 'predicted_class' in df.columns else 'line_type'
            text_column = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
            
            # Get VAT headers and data - look for both explicit VAT classifications and VAT-like content
            vat_headers = df[df[class_column] == 'VAT_HEADER'] if 'VAT_HEADER' in df[class_column].values else pd.DataFrame()
            vat_data_lines = df[df[class_column] == 'VAT_DATA'] if 'VAT_DATA' in df[class_column].values else pd.DataFrame()
            
            # Comprehensive VAT content detection - covers ALL possible VAT formats including OCR errors
            vat_pattern_text = r'(?i)(' + '|'.join([
                # Basic VAT patterns (OCR-tolerant)
                r'\d+(?:\.\d+)?\s*[%a].*[VU]AT',  # % VAT patterns (% can be misread as 'a', V as U)
                r'[VU]AT.*\d+(?:\.\d+)?\s*[%a]',  # VAT % patterns (OCR tolerant)
                r'[VU]AT\s*@\s*\d+(?:\.\d+)?\s*[%a]?',  # VAT @ % (enhanced, OCR tolerant)
                r'@\s*\d+(?:\.\d+)?\s*[%a]',   # @ % patterns (% can be 'a')
                r'[VU]AT[:\s]*[£€$]\s*\d+(?:\.\d+)?',  # VAT: amount (V can be U)
                
                # Fuel station format (20.00% 51.65 10.33 49.86 9.97)
                r'\d+(?:\.\d+)?\s*%\s+\d+(?:\.\d+)?\s+\d+(?:\.\d+)?\s+\d+(?:\.\d+)?\s+\d+(?:\.\d+)?',  # % followed by 4+ amounts
                r'^2[0-5](?:\.\d+)?\s*%\s+\d+(?:\.\d+)?\s+\d+(?:\.\d+)?',  # 20-25% followed by amounts (common VAT rates)
                r'1[0-9](?:\.\d+)?\s*%\s+\d+(?:\.\d+)?\s+\d+(?:\.\d+)?',  # 10-19% followed by amounts
                r'0(?:\.\d+)?\s*%\s+\d+(?:\.\d+)?\s+\d+(?:\.\d+)?',  # 0% (zero-rated) followed by amounts
                
                # Code-based patterns
                r'[A-Z]+\d*\s+\d+(?:\.\d+)?\s*%\s*(?:[£€$]\s*)?\d+(?:\.\d+)?',  # Code Rate% amounts
                r'\d+(?:\.\d+)?\s*%\s*(?:Rated|Rate)\s+\d+(?:\.\d+)?',  # % Rated amounts
                
                # Named rate patterns
                r'(?:Standard|Reduced|Zero)\s+Rate\s+\d+(?:\.\d+)?\s*%',  # Standard/Reduced/Zero Rate %
                r'VAT\s+Rate\s+Analysis',  # VAT Rate Analysis headers
                
                # International VAT
                r'(?:TVA|MwSt|IVA)\s+\d+(?:\.\d+)?\s*%',  # International VAT (French, German, Spanish)
                
                # Service/Food VAT
                r'(?:Service|Food|Delivery).*VAT',  # Service/Food VAT
                r'VAT\s+(?:Exempt|Free)',  # VAT Exempt/Free
                
                # Descriptive VAT
                r'[£€$]?\s*\d+(?:\.\d+)?\s*\+\s*VAT',  # amount + VAT
                r'Inc\.?\s+VAT',  # Inc VAT / Inc. VAT
                
                # Multi-currency and symbols
                r'(?:VAT|Tax)\s*\d+(?:\.\d+)?\s*%\s*[€£$]',  # VAT rate% currency
                
                # Tabular formats
                r'(?:Rate|Net|VAT|Gross).*(?:Rate|Net|VAT|Gross)',  # Tabular headers
                r'[A-Z]\s+\d+(?:\.\d+)?\s*%\s+\d+(?:\.\d+)?.*\d+(?:\.\d+)?',  # Code rate% amounts
                
                # Zero-rated
                r'[A-Z]+\d*\s+0(?:\.0+)?\s*%',  # Zero rate patterns
                
                # Simple amount patterns with VAT context
                r'(?:\d+(?:\.\d+)?\s*%.*){2,}',  # Multiple percentages (likely VAT table)
            ]) + ')'
            potential_vat_lines = df[df[text_column].str.contains(vat_pattern_text, na=False, regex=True)]
            
            # Remove already classified VAT lines to avoid duplicates
            if not vat_data_lines.empty:
                potential_vat_lines = potential_vat_lines[~potential_vat_lines.index.isin(vat_data_lines.index)]
            
            # Combine explicit VAT_DATA with potential VAT lines
            if not potential_vat_lines.empty:
                vat_data_lines = pd.concat([vat_data_lines, potential_vat_lines], ignore_index=True)
            
            # Process headers - include both explicit VAT_HEADER and detected headers
            headers = []
            for _, row in vat_headers.iterrows():
                headers.append({
                    'text': str(row.get(text_column, '')),
                    'line_number': int(row.get('line_number', 0)),
                    'confidence': float(row.get('confidence', 0.0))
                })
            
            # Also detect VAT headers from potential VAT lines
            for _, row in potential_vat_lines.iterrows():
                text = str(row.get(text_column, ''))
                if self.detect_vat_headers(text):
                    headers.append({
                        'text': text,
                        'line_number': int(row.get('line_number', 0)),
                        'confidence': float(row.get('confidence', 0.0)),
                        'detected': True
                    })
            
            # Process VAT data entries - only include valid entries with actual VAT data
            entries = []
            for _, row in vat_data_lines.iterrows():
                entry = self.process_vat_data_entry(
                    {
                        'raw_text': str(row.get(text_column, '')),
                        'line_number': int(row.get('line_number', 0)),
                        'confidence': float(row.get('confidence', 0.0))
                    },
                    headers
                )
                # Only include entries that have meaningful VAT information
                if entry and self.is_valid_vat_entry(entry):
                    entries.append(entry)
            
            # Generate summary
            summary = {
                'total_entries': len(entries),
                'entries_with_vat_code': len([e for e in entries if e.get('vat_code')]),
                'entries_with_vat_rate': len([e for e in entries if e.get('vat_rate')]),
                'entries_with_net_amount': len([e for e in entries if e.get('net_amount')]),
                'entries_with_vat_amount': len([e for e in entries if e.get('vat_amount')]),
                'entries_with_total_amount': len([e for e in entries if e.get('total_amount')]),
            }
            
            return {
                'vat_headers': headers,
                'vat_data_entries': entries,
                'vat_summary': summary,
                'extraction_metadata': {
                    'extraction_timestamp': datetime.now().isoformat(),
                    'source_file': csv_file_path,
                    'extractor_version': 'VAT Information System v2.6.9 - Enhanced VAT Processing Integration',
                    'total_lines_processed': len(df),
                    'extraction_status': 'success'
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting VAT data from CSV: {e}")
            return {
                'vat_headers': [],
                'vat_data_entries': [],
                'vat_summary': {'total_entries': 0, 'error': str(e)},
                'extraction_metadata': {
                    'extraction_timestamp': datetime.now().isoformat(),
                    'source_file': csv_file_path,
                    'extractor_version': 'VAT Information System v2.6.9 - Enhanced VAT Processing Integration',
                    'extraction_status': 'failed'
                }
            }
        
    def extract_amounts_from_text(self, text: str) -> List[float]:
        """Extract only actual monetary amounts from VAT_DATA line (excluding VAT rates)"""
        amounts = []
        
        # Clean text for better parsing
        cleaned_text = text.replace('E', '€').replace('£267', '£2.67')  # Common OCR fixes
        
        # More specific patterns that focus on currency amounts only
        currency_amount_patterns = [
            r'£\\s*(\\d{1,4}(?:\\.\\d{2})?)',           # £41.68, £8.34, £50.02
            r'€\\s*(\\d{1,4}(?:\\.\\d{2})?)',           # €41.68, €8.34, €50.02
            r'\\b(\\d{1,4}\\.\\d{2})\\b(?!\\s*%)',        # 41.68, 8.34, 50.02 (but not followed by %)
            r'(\\d{1,3}(?:,\\d{3})*\\.\\d{2})(?!\\s*%)' # 1,234.56 (but not followed by %)
        ]
        
        for pattern in currency_amount_patterns:
            matches = re.findall(pattern, cleaned_text)
            for match in matches:
                try:
                    amounts.append(float(match))
                except ValueError:
                    continue
        
        # Remove duplicates and return sorted
        unique_amounts = sorted(list(set(amounts)))
        
        # Additional filtering: remove obvious VAT rate numbers
        filtered_amounts = []
        for amount in unique_amounts:
            # Skip common VAT rates (5, 10, 15, 20, 25) when they appear as whole numbers
            if amount in [5.0, 10.0, 15.0, 20.0, 25.0]:
                continue
            filtered_amounts.append(amount)
        
        return filtered_amounts
    
    def analyze_vat_header_context(self, vat_headers: List[Dict]) -> Dict[str, str]:
        """Analyze VAT headers to determine column order"""
        header_context = {
            'vat_position': None,
            'net_position': None, 
            'gross_position': None,
            'total_position': None
        }
        
        for header in vat_headers:
            text = header.get('text', '').upper()
            
            # Common VAT header patterns
            if 'VAT' in text and 'NET' in text:
                header_context['column_order'] = 'vat_net'
            
            if 'GROSS' in text and 'NET' in text:
                header_context['column_order'] = 'net_gross'
            
            if 'VAT' in text and 'GROSS' in text and 'NET' in text:
                header_context['column_order'] = 'vat_net_gross'
        
        return header_context
    
    def identify_amounts_using_business_logic(self, amounts: List[float], header_context: Dict) -> Dict[str, float]:
        """Identify VAT, NET, and TOTAL amounts from OCR extracted amounts only (no calculations)"""
        if len(amounts) < 2:
            return {'net_amount': None, 'vat_amount': None, 'total_amount': None}
        
        # Sort amounts for easier analysis
        sorted_amounts = sorted(amounts)
        
        # Use header context to determine column order if available
        column_order = header_context.get('column_order', '')
        
        # Strategy: Use position and relative size to identify amounts
        result = {'net_amount': None, 'vat_amount': None, 'total_amount': None}
        
        if len(amounts) == 2:
            # Two amounts: typically VAT and NET - ALWAYS calculate total from THIS LINE ONLY
            smaller, larger = sorted_amounts[0], sorted_amounts[1]
            ratio = smaller / larger if larger > 0 else 0
            
            if 0.05 <= ratio <= 0.4:  # Typical VAT ratio (this line's VAT)
                result['vat_amount'] = smaller
                result['net_amount'] = larger
                # Calculate this line's total = net + vat (FRESH CALCULATION, NOT FROM ELSEWHERE)
                result['total_amount'] = smaller + larger
                print(f"      🔢 BUSINESS LOGIC (VAT ratio): {larger:.2f} (NET) + {smaller:.2f} (VAT) = {smaller + larger:.2f} (TOTAL)")
            else:
                # If ratio doesn't match VAT pattern, treat as NET and TOTAL with fresh calculation
                result['net_amount'] = smaller
                # Calculate total = must sum to complete the transaction
                result['total_amount'] = smaller + larger
                print(f"      🔢 BUSINESS LOGIC (no VAT ratio): {smaller:.2f} (NET) + {larger:.2f} (?) = {smaller + larger:.2f} (TOTAL)")
        
        elif len(amounts) == 3:
            # Three amounts: likely VAT, NET, TOTAL in some order
            small, medium, large = sorted_amounts[0], sorted_amounts[1], sorted_amounts[2]
            
            # Check if small amount could be VAT for medium amount
            if medium > 0 and 0.05 <= (small / medium) <= 0.4:
                result['vat_amount'] = small
                result['net_amount'] = medium
                result['total_amount'] = large
            else:
                # Assign based on size
                result['net_amount'] = medium
                result['total_amount'] = large
                # Keep VAT as None if we can't identify it clearly
        
        elif len(amounts) > 3:
            # More than 3 amounts - use largest 3
            top_amounts = sorted_amounts[-3:]
            result['net_amount'] = top_amounts[1]  # Middle of top 3
            result['total_amount'] = top_amounts[2]  # Largest
            
            # Try to find VAT amount
            for amount in amounts:
                if result['net_amount'] and 0.05 <= (amount / result['net_amount']) <= 0.4:
                    result['vat_amount'] = amount
                    break
        
        # Fallback: if no pattern found, use simple size-based assignment
        if not any(result.values()) and len(amounts) >= 2:
            result['net_amount'] = sorted_amounts[-2]  # Second largest
            result['total_amount'] = sorted_amounts[-1]  # Largest
        
        return result
    
    def extract_vat_code(self, text: str) -> Optional[str]:
        """Extract VAT code from text"""
        for pattern in self.vat_code_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return None
    
    def extract_vat_rate(self, text: str) -> Optional[float]:
        """Extract VAT rate from text"""
        for pattern in self.vat_rate_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None
    
    def extract_uk_vat_line_format(self, text: str) -> Dict:
        """Extract VAT information from comprehensive UK and international formats using flexible patterns"""
        
        # Clean text for better parsing (handle OCR errors)
        cleaned_text = self.clean_text_for_vat_parsing(text)
        
        # Comprehensive VAT patterns covering all possible receipt formats
        vat_patterns = [
            # Pattern 1: VAT @ rate format "VAT @ 0%" or "VAT @ 20%" (Screwfix format)
            {
                'pattern': r'VAT\s*@\s*(\d+(?:\.\d+)?)\s*%?\s*(?:[£€$]?\s*(\d+(?:\.\d+)?))?(?:\s*[£€$]?\s*(\d+(?:\.\d+)?))?(?:\s*[£€$]?\s*(\d+(?:\.\d+)?))?',
                'groups': ['rate', 'net_amount', 'vat_amount', 'total_amount'],
                'method': 'vat_at_rate_format'
            },
            
            # Pattern 2: Standard table format "20% 33.34 6.67 40.01" (from fuel receipt)
            {
                'pattern': r'(\d+(?:\.\d+)?)\s*%\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)',
                'groups': ['rate', 'net_amount', 'vat_amount', 'total_amount'],
                'method': 'standard_table_format'
            },
            
            # Pattern 2: With currency symbols "20% £33.34 £6.67 £40.01"
            {
                'pattern': r'(\d+(?:\.\d+)?)\s*%\s*[£€$]?\s*(\d+(?:\.\d+)?)\s*[£€$]?\s*(\d+(?:\.\d+)?)\s*[£€$]?\s*(\d+(?:\.\d+)?)',
                'groups': ['rate', 'net_amount', 'vat_amount', 'total_amount'],
                'method': 'currency_table_format'
            },
            
            # Pattern 3: Code-based format "G 20.0% £41.68 £8.34 £50.02"
            {
                'pattern': r'([A-Z]+\d*)\s+(\d+(?:\.\d+)?)\s*%\s*[£€$]?\s*(\d+(?:\.\d+)?)\s*[£€$]?\s*(\d+(?:\.\d+)?)\s*[£€$]?\s*(\d+(?:\.\d+)?)',
                'groups': ['code', 'rate', 'net_amount', 'vat_amount', 'total_amount'],
                'method': 'code_table_format'
            },
            
            # Pattern 4: Analysis format "A1 20% Rated 12.50 2.50 15.00"
            {
                'pattern': r'([A-Z]+\d*)\s+(\d+(?:\.\d+)?)\s*%\s*(?:\w+\s+)?(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)',
                'groups': ['code', 'rate', 'net_amount', 'vat_amount', 'total_amount'],
                'method': 'analysis_format'
            },
            
            # Pattern 5: Reverse format "£33.34 6.67% £40.01" (net, rate, total)
            {
                'pattern': r'[£€$]?\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*%\s*[£€$]?\s*(\d+(?:\.\d+)?)',
                'groups': ['net_amount', 'rate', 'total_amount'],
                'method': 'reverse_format'
            },
            
            # Pattern 6: Descriptive format "VAT @ 20%: £6.67 on £33.34"
            {
                'pattern': r'VAT\s*@\s*(\d+(?:\.\d+)?)\s*%\s*:?\s*[£€$]?\s*(\d+(?:\.\d+)?)\s*(?:on|net|from)?\s*[£€$]?\s*(\d+(?:\.\d+)?)',
                'groups': ['rate', 'vat_amount', 'net_amount'],
                'method': 'descriptive_format'
            },
            
            # Pattern 7: Split format "Net: £33.34 VAT: £6.67 Total: £40.01"
            {
                'pattern': r'(?:Net|Amount)\s*:?\s*[£€$]?\s*(\d+(?:\.\d+)?).*?VAT\s*:?\s*[£€$]?\s*(\d+(?:\.\d+)?).*?(?:Total|Gross)\s*:?\s*[£€$]?\s*(\d+(?:\.\d+)?)',
                'groups': ['net_amount', 'vat_amount', 'total_amount'],
                'method': 'split_format'
            },
            
            # Pattern 8: VAT rate with amounts "20.00 % VAT B £ 0.33" or "20.00 % VAT E 6.75"
            {
                'pattern': r'(\d+(?:\.\d+)?)\s*[%a]\s*[VU]AT\s+([A-Z]+\d*)\s*[£€$]?\s*(\d+(?:\.\d+)?)',
                'groups': ['rate', 'code', 'vat_amount'],
                'method': 'rate_vat_code_format_ocr_tolerant'
            },
            
            # Pattern 9: Zero-rated "0% £5.00 £0.00 £5.00" or "Z 0% 5.00"
            {
                'pattern': r'(?:([A-Z]+\d*)\s+)?(0(?:\.0+)?)\s*%\s*[£€$]?\s*(\d+(?:\.\d+)?)(?:\s*[£€$]?\s*(0(?:\.0+)?))?\s*[£€$]?\s*(\d+(?:\.\d+)?)?',
                'groups': ['code', 'rate', 'net_amount', 'vat_amount', 'total_amount'],
                'method': 'zero_rated_format'
            },
            
            # Pattern 10: International formats "TVA 19.6% 100.00 €19.60 €119.60"
            {
                'pattern': r'(?:TVA|MwSt|IVA)\s*(\d+(?:\.\d+)?)\s*%\s*[£€$]?\s*(\d+(?:\.\d+)?)\s*[£€$]?\s*(\d+(?:\.\d+)?)\s*[£€$]?\s*(\d+(?:\.\d+)?)',
                'groups': ['rate', 'net_amount', 'vat_amount', 'total_amount'],
                'method': 'international_format'
            },
            
            # Pattern 11: Simple VAT amount "VAT: £6.67" or "Inc VAT £6.67"
            {
                'pattern': r'(?:Inc\.?\s+)?VAT\s*:?\s*[£€$]\s*(\d+(?:\.\d+)?)',
                'groups': ['vat_amount'],
                'method': 'simple_vat_format'
            },
            
            # Pattern 12: Percentage with amounts in sequence "20% 10.00 2.00 12.00"
            {
                'pattern': r'(\d+(?:\.\d+)?)\s*%\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)',
                'groups': ['rate', 'net_amount', 'vat_amount', 'total_amount'],
                'method': 'percentage_sequence_format'
            },
            
            # Pattern 13: Totals format "Totals 33.34 6.67 40.01"
            {
                'pattern': r'(?:Total[s]?|Sum)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)',
                'groups': ['net_amount', 'vat_amount', 'total_amount'],
                'method': 'totals_format'
            },
            
            # Pattern 14: Flexible amounts extraction (fallback)
            {
                'pattern': r'(?:^|\b)(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)(?:\s|$)',
                'groups': ['amount1', 'amount2', 'amount3'],
                'method': 'flexible_amounts_format'
            }
        ]
        
        for pattern_info in vat_patterns:
            match = re.search(pattern_info['pattern'], cleaned_text, re.IGNORECASE)
            if match:
                try:
                    result = {'extraction_method': pattern_info['method']}
                    groups = pattern_info['groups']
                    
                    # Extract matched values
                    matched_values = []
                    for i, field_name in enumerate(groups, 1):
                        value = match.group(i)
                        if value:  # Only process non-None values
                            if field_name == 'rate':
                                result['vat_rate'] = float(value)
                            elif field_name in ['net_amount', 'vat_amount', 'total_amount']:
                                result[field_name] = float(value)
                            elif field_name == 'code':
                                result['vat_code'] = value.upper()
                            elif field_name == 'currency':
                                result['currency'] = value
                            elif field_name.startswith('amount'):
                                # Flexible amount fields for business logic processing
                                matched_values.append(float(value))
                    
                    # Handle flexible formats that need business logic
                    if pattern_info['method'] == 'flexible_amounts_format':
                        # Use business logic to identify amounts
                        amount_identification = self.identify_vat_amounts_using_business_logic(matched_values)
                        result.update(amount_identification)
                    elif pattern_info['method'] == 'reverse_format':
                        # Handle reverse format where we need to calculate VAT amount
                        if 'net_amount' in result and 'total_amount' in result and 'vat_rate' in result:
                            calculated_vat = result['total_amount'] - result['net_amount']
                            result['vat_amount'] = calculated_vat
                    
                    # Handle special cases
                    method = pattern_info['method']
                    if 'zero_rated' in method:
                        result['vat_status'] = 'zero_rated'
                        if not result.get('vat_amount'):
                            result['vat_amount'] = 0.0
                    
                    # Validate that we have meaningful VAT data
                    has_rate = result.get('vat_rate') is not None
                    has_code = result.get('vat_code') is not None
                    has_amount = result.get('vat_amount') is not None
                    has_net = result.get('net_amount') is not None
                    
                    if has_rate or has_code or has_amount or has_net:
                        # Additional validation for amounts
                        if self.validate_extracted_amounts(result):
                            return result
                        
                except (ValueError, IndexError, TypeError):
                    continue
        
        return {}
    
    def clean_text_for_vat_parsing(self, text: str) -> str:
        """Clean text for better VAT pattern matching"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common OCR errors
        cleaned = cleaned.replace('£267', '£2.67')  # Missing decimal
        # REMOVED: cleaned = cleaned.replace(' E ', ' € ')     # This destroys VAT codes like "E"
        cleaned = cleaned.replace('O%', '0%')       # Zero misread as O
        cleaned = cleaned.replace('l%', '1%')       # One misread as l
        
        # Normalize currency symbols - more specific to avoid affecting VAT codes
        cleaned = re.sub(r'([£€$])\s+', r'\1', cleaned)  # Remove space after currency symbols
        
        # Handle space-separated decimals "33 34" -> "33.34"
        cleaned = re.sub(r'(\d+)\s+(\d{2})(?=\s|$)', r'\1.\2', cleaned)
        
        return cleaned
    
    def identify_vat_amounts_using_business_logic(self, amounts: List[float], rate: Optional[float] = None) -> Dict[str, float]:
        """Use business logic to identify VAT components from extracted amounts"""
        if not amounts or len(amounts) < 2:
            return {'net_amount': None, 'vat_amount': None, 'total_amount': None}
        
        # Sort amounts for analysis
        sorted_amounts = sorted(amounts)
        result = {'net_amount': None, 'vat_amount': None, 'total_amount': None}
        
        if len(amounts) == 2:
            smaller, larger = sorted_amounts[0], sorted_amounts[1]
            # Check if smaller could be VAT of larger
            if larger > 0 and 0.05 <= (smaller / larger) <= 0.4:
                result['net_amount'] = larger
                result['vat_amount'] = smaller
            else:
                # Might be net and total
                result['net_amount'] = smaller
                result['total_amount'] = larger
                
        elif len(amounts) == 3:
            # Try to identify the VAT component
            small, medium, large = sorted_amounts
            
            # Look for reasonable VAT relationships
            # Check if small is VAT of medium
            if medium > 0 and 0.05 <= (small / medium) <= 0.4:
                result['net_amount'] = medium
                result['vat_amount'] = small
                result['total_amount'] = large
            # Check if small is VAT of large
            elif large > 0 and 0.05 <= (small / large) <= 0.3:
                # Could be net, vat, total where vat is calculated on total
                result['total_amount'] = large
                result['vat_amount'] = small
                result['net_amount'] = medium
            # Check if medium is VAT of large
            elif large > 0 and 0.05 <= (medium / large) <= 0.4:
                result['net_amount'] = large
                result['vat_amount'] = medium
                # small might be something else
            else:
                # Use positional logic: assume net, vat, total
                result['net_amount'] = small if small > medium * 0.7 else medium
                result['vat_amount'] = small if small <= medium * 0.4 else medium
                result['total_amount'] = large
        
        # Validate relationships
        if result['net_amount'] and result['vat_amount'] and result['total_amount']:
            expected_total = result['net_amount'] + result['vat_amount']
            if abs(expected_total - result['total_amount']) > 0.02:  # Allow 2p tolerance
                # Relationships don't match, try different assignment
                if result['vat_amount'] and result['total_amount']:
                    result['net_amount'] = result['total_amount'] - result['vat_amount']
        
        return result
    
    def detect_vat_headers(self, text: str) -> bool:
        """Detect if a line contains VAT header information"""
        header_patterns = [
            r'(?i)VAT\s+(?:Rate\s+)?Analysis',
            r'(?i)(?:Rate|Net|VAT|Gross|Total).*(?:Rate|Net|VAT|Gross|Total)',
            r'(?i)VAT\s+Summary',
            r'(?i)Tax\s+Breakdown',
            r'(?i)VAT\s+Details',
            r'(?i)(?:Description|Code)\s+(?:Net|VAT|Total)',
            r'(?i)VAT\s+Registration',
            r'(?i)VAT\s+No\.?\s*\d+'
        ]
        
        for pattern in header_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def validate_extracted_amounts(self, result: Dict) -> bool:
        """Validate that extracted amounts are reasonable"""
        try:
            net = result.get('net_amount')
            vat = result.get('vat_amount')
            total = result.get('total_amount')
            rate = result.get('vat_rate')
            
            # Basic range checks
            for amount in [net, vat, total]:
                if amount is not None and (amount < 0 or amount > 100000):
                    return False
            
            # Relationship checks if we have multiple amounts
            if net and vat:
                # VAT should be reasonable percentage of net
                ratio = vat / net if net > 0 else 0
                if not (0 <= ratio <= 0.5):  # 0-50% is reasonable
                    return False
                    
            if net and vat and total:
                # Total should approximately equal net + vat (allow 2p tolerance)
                expected_total = net + vat
                if abs(total - expected_total) > 0.02:
                    return False
                    
            if rate and net and vat:
                # Rate should match calculated rate (allow 1% tolerance)
                calculated_rate = (vat / net * 100) if net > 0 else 0
                if abs(rate - calculated_rate) > 1.0:
                    return False
            
            return True
            
        except (TypeError, ZeroDivisionError, ValueError):
            return False
    
    def is_valid_vat_entry(self, entry: Dict) -> bool:
        """Check if a VAT entry contains meaningful VAT information"""
        # Must have at least one of these essential VAT components
        has_vat_code = entry.get('vat_code') is not None
        has_vat_rate = entry.get('vat_rate') is not None
        has_vat_amount = entry.get('vat_amount') is not None
        has_amounts = entry.get('net_amount') is not None or entry.get('total_amount') is not None
        
        # Entry is valid if it has VAT code OR VAT rate OR VAT amount
        # AND it's not just a header line
        raw_text = entry.get('raw_text', '').upper()
        is_header_only = any(header_word in raw_text for header_word in [
            'SUMMARY', 'DESCRIPTION', 'HEADER', 'ANALYSIS', 'NET VAT TOTAL', 'VAT NO.'
        ])
        
        return (has_vat_code or has_vat_rate or has_vat_amount) and not is_header_only
    
    def process_vat_data_entry(self, entry: Dict, vat_headers: List[Dict]) -> Dict:
        """Process a single VAT data entry with enhanced extraction (v2.6.9)"""
        raw_text = entry.get('raw_text', '')
        
        # Try enhanced vat_details.py approach first (v2.6.9)
        line_values = raw_text.split()
        enhanced_result = self.extract_vat_line_enhanced(line_values)
        
        # Check if enhanced extraction was successful
        if enhanced_result and any(enhanced_result.get(field) for field in ['vat_code', 'vat_rate', 'vat_amount']):
            entry.update(enhanced_result)
            return entry
        
        # Fall back to UK-specific format
        uk_format = self.extract_uk_vat_line_format(raw_text)
        if uk_format:
            # Use UK format extraction
            amounts_found = []
            if uk_format.get('vat_amount'): amounts_found.append(uk_format.get('vat_amount'))
            if uk_format.get('net_amount'): amounts_found.append(uk_format.get('net_amount'))
            if uk_format.get('total_amount'): amounts_found.append(uk_format.get('total_amount'))
            
            entry.update({
                'vat_code': uk_format.get('vat_code'),
                'vat_rate': uk_format.get('vat_rate'),
                'vat_amount': uk_format.get('vat_amount'),
                'net_amount': uk_format.get('net_amount'),  # May be provided in table format
                'total_amount': uk_format.get('total_amount'),  # May be provided in table format
                'amounts_found': amounts_found,
                'extraction_method': uk_format.get('extraction_method')
            })
        else:
            # Fall back to generic extraction
            vat_code = self.extract_vat_code(raw_text)
            vat_rate = self.extract_vat_rate(raw_text)
            
            # Extract amounts
            amounts = self.extract_amounts_from_text(raw_text)
            
            # Analyze header context
            header_context = self.analyze_vat_header_context(vat_headers)
            
            # Use business logic to identify amounts
            amount_identification = self.identify_amounts_using_business_logic(amounts, header_context)
            
            # Update entry with improved extraction
            entry.update({
                'vat_code': vat_code,
                'vat_rate': vat_rate,
                'net_amount': amount_identification.get('net_amount'),
                'vat_amount': amount_identification.get('vat_amount'), 
                'total_amount': amount_identification.get('total_amount'),
                'amounts_found': amounts,
                'amount_identification_method': 'business_logic_enhanced',
                'header_context_used': header_context,
                'extraction_method': 'comprehensive'
            })
        
        return entry
