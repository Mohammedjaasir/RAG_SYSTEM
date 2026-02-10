"""
Advanced Feature Extractor for Receipt Line Classification

This module extracts comprehensive features using:
1. Keyword analysis data (company_info, tax_and_fees, itemization, etc.)
2. Token patterns (prices, dates, phone numbers, etc.)
3. Structural features (position, length, surrounding context)
4. OCR confidence integration
"""

import json
import re
from typing import Dict, List, Optional, Set
from pathlib import Path


class AdvancedFeatureExtractor:
    """Extract comprehensive features from receipt lines."""
    
    def __init__(self, analysis_root: str = None):
        """Initialize with analysis data."""
        # Build category keyword sets
        self.category_keywords = self._build_category_keywords()
        
        # Define pattern matchers
        self.patterns = self._build_pattern_matchers()
        
        # Position thresholds
        self.header_threshold = 0.25  # First 25% is header area
        self.footer_threshold = 0.75  # Last 25% is footer area
    

    
    def _build_category_keywords(self) -> Dict[str, Set[str]]:
        """Build keyword sets for each category."""
        
        # Known categories from analysis
        categories = {
            'company_info': {
                'ltd', 'limited', 'plc', 'inc', 'corp', 'store', 'shop', 'market',
                'pharmacy', 'supermarket', 'express', 'tesco', 'sainsbury', 'asda',
                'morrisons', 'aldi', 'lidl', 'co-op', 'waitrose'
            },
            'transaction_summary': {
                'total', 'subtotal', 'sub-total', 'sub total', 'balance', 'due',
                'amount', 'sum', 'net', 'gross', 'final', 'grand total'
            },
            'tax_and_fees': {
                'vat', 'tax', 'rate', 'duty', 'levy', '@', '%', 'percent',
                'inclusive', 'exclusive', 'exempt', 'zero rated'
            },
            'itemization': {
                'qty', 'quantity', 'description', 'item', 'product', 'price',
                'each', 'unit', 'per', 'x', 'amount', 'line', 'no.'
            },
            'payment': {
                'card', 'debit', 'credit', 'visa', 'mastercard', 'amex', 'cash',
                'contactless', 'chip', 'pin', 'payment', 'paid', 'tender',
                'change', 'auth code', 'terminal', 'aid', 'pan'
            },
            'date_and_time': {
                'date', 'time', 'am', 'pm', 'monday', 'tuesday', 'wednesday',
                'thursday', 'friday', 'saturday', 'sunday', 'jan', 'feb', 'mar',
                'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
            },
            'identification': {
                'receipt', 'transaction', 'sale', 'purchase', 'order', 'ref',
                'reference', 'number', 'id', 'seq', 'till', 'register', 'operator'
            },
            'miscellaneous': {
                'thank you', 'thanks', 'welcome', 'come again', 'visit',
                'customer service', 'opening hours', 'phone', 'tel', 'www',
                'website', 'email', 'address', 'postcode'
            }
        }
        
        return categories
    
    def _build_pattern_matchers(self) -> Dict:
        """Build regex patterns for token detection."""
        
        patterns = {
            # Price patterns
            'price': re.compile(r'[£$€¥]?\s*\d+[.,]\d{2}|[£$€¥]\s*\d+|\d+[.,]\d{2}\s*[£$€¥]', re.IGNORECASE),
            
            # Percentage patterns
            'percentage': re.compile(r'\d+[.,]?\d*\s*%|@\s*\d+[.,]?\d*', re.IGNORECASE),
            
            # Date patterns
            'date': re.compile(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{2,4}[/-]\d{1,2}[/-]\d{1,2}', re.IGNORECASE),
            
            # Time patterns
            'time': re.compile(r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]m)?', re.IGNORECASE),
            
            # Phone patterns
            'phone': re.compile(r'(?:\+44\s?)?0\d{3,4}\s?\d{3,4}\s?\d{3,4}|\d{11}', re.IGNORECASE),
            
            # Email patterns
            'email': re.compile(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', re.IGNORECASE),
            
            # Website patterns
            'website': re.compile(r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|[a-zA-Z0-9.-]+\.com', re.IGNORECASE),
            
            # Numeric patterns
            'number': re.compile(r'\b\d+\b'),
            
            # Card number patterns (masked)
            'card_number': re.compile(r'\*+\d{4}|x+\d{4}', re.IGNORECASE),
            
            # Postcode patterns (UK)
            'postcode': re.compile(r'[a-z]{1,2}\d[a-z\d]?\s?\d[a-z]{2}', re.IGNORECASE),
            
            # VAT number patterns
            'vat_number': re.compile(r'(?:vat\s*(?:no\.?|number))?\s*:?\s*\d{9,12}', re.IGNORECASE),
        }
        
        return patterns
    
    def extract_features(self, text: str, line_position: int = 0, 
                        total_lines: int = 1, ocr_confidence: float = None) -> Dict:
        """Extract comprehensive features from a text line."""
        
        # Normalize text
        cleaned_text = text.strip().lower()
        
        # Basic line features
        features = {
            'line_length': len(text),
            'word_count': len(text.split()),
            'char_count': len(text.strip()),
            'is_empty': len(text.strip()) == 0,
            'has_punctuation': bool(re.search(r'[.,!?;:]', text)),
            'has_digits': bool(re.search(r'\d', text)),
            'has_upper': bool(re.search(r'[A-Z]', text)),
            'position_ratio': line_position / max(1, total_lines - 1),
        }
        
        # Position-based features
        features.update(self._extract_position_features(line_position, total_lines))
        
        # Keyword-based features
        features.update(self._extract_keyword_features(cleaned_text))
        
        # Pattern-based features
        features.update(self._extract_pattern_features(text))
        
        # Structural features
        features.update(self._extract_structural_features(text))
        
        # OCR confidence
        if ocr_confidence is not None:
            features['ocr_confidence'] = ocr_confidence
            features['low_ocr_confidence'] = ocr_confidence < 0.8
        
        return features
    
    def _extract_position_features(self, position: int, total_lines: int) -> Dict:
        """Extract position-based features."""
        
        position_ratio = position / max(1, total_lines - 1)
        
        return {
            'is_first_line': position == 0,
            'is_last_line': position == total_lines - 1,
            'is_header_area': position_ratio <= self.header_threshold,
            'is_body_area': self.header_threshold < position_ratio < self.footer_threshold,
            'is_footer_area': position_ratio >= self.footer_threshold,
            'line_position': position,
            'total_lines': total_lines,
            'position_ratio': position_ratio
        }
    
    def _extract_keyword_features(self, text: str) -> Dict:
        """Extract keyword-based features using category analysis."""
        
        features = {}
        
        # Count matches for each category
        for category, keywords in self.category_keywords.items():
            count = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword in text:
                    count += 1
                    matched_keywords.append(keyword)
            
            features[f'{category}_count'] = count
            features[f'has_{category}'] = count > 0
            features[f'{category}_keywords'] = matched_keywords
        
        # Special combinations
        features['has_qty_description'] = ('qty' in text or 'quantity' in text) and 'description' in text
        features['has_total_keywords'] = any(word in text for word in ['total', 'subtotal', 'balance', 'due'])
        features['has_vat_keywords'] = any(word in text for word in ['vat', 'tax', '@'])
        features['has_payment_card'] = any(word in text for word in ['card', 'visa', 'mastercard', 'debit'])
        
        # Overall keyword density
        total_keyword_matches = sum(features[f'{cat}_count'] for cat in self.category_keywords.keys())
        features['total_keyword_matches'] = total_keyword_matches
        features['keyword_density'] = total_keyword_matches / max(1, len(text.split()))
        
        return features
    
    def _extract_pattern_features(self, text: str) -> Dict:
        """Extract pattern-based features."""
        
        features = {}
        
        # Apply all patterns
        for pattern_name, pattern in self.patterns.items():
            matches = pattern.findall(text)
            features[f'has_{pattern_name}'] = len(matches) > 0
            features[f'{pattern_name}_count'] = len(matches)
            features[f'{pattern_name}_matches'] = matches
        
        # Special pattern combinations
        features['has_price_and_quantity'] = features['has_price'] and features['has_number']
        features['has_date_time'] = features['has_date'] or features['has_time']
        features['has_contact_info'] = features['has_phone'] or features['has_email'] or features['has_website']
        
        return features
    
    def _extract_structural_features(self, text: str) -> Dict:
        """Extract structural features from text formatting."""
        
        features = {
            'starts_with_number': text.strip() and text.strip()[0].isdigit(),
            'starts_with_letter': text.strip() and text.strip()[0].isalpha(),
            'ends_with_punctuation': text.strip() and text.strip()[-1] in '.,!?;:',
            'all_caps': text.strip().isupper() and len(text.strip()) > 2,
            'has_colon': ':' in text,
            'has_equals': '=' in text,
            'has_dash': '-' in text,
            'has_parentheses': '(' in text or ')' in text,
            'has_asterisk': '*' in text,
            'has_hash': '#' in text,
            'multiple_spaces': '  ' in text,
            'tabular_format': '\t' in text or text.count(' ') > 5,
        }
        
        # Calculate spacing patterns
        words = text.split()
        if len(words) >= 2:
            features['word_spacing_regular'] = len(set(len(w) for w in words)) <= 2
            features['has_alignment_spacing'] = any(len(gap) > 2 for gap in re.findall(r' {2,}', text))
        else:
            features['word_spacing_regular'] = False
            features['has_alignment_spacing'] = False
        
        return features


def test_advanced_feature_extractor():
    """Test the advanced feature extractor."""
    print("🧪 Testing Advanced Feature Extractor")
    print("=" * 50)
    
    extractor = AdvancedFeatureExtractor("/mnt/data/Projects/ML/solution")
    
    # Test different types of lines
    test_lines = [
        ("TESCO Express", "store header"),
        ("QTY  DESCRIPTION      PRICE", "item header"),
        ("2    Coca Cola 330ml  £2.50", "item data"),
        ("SUBTOTAL           £7.30", "subtotal"),
        ("VAT @ 20%          £1.46", "vat line"),
        ("TOTAL              £8.76", "total"),
        ("Card Payment", "payment method"),
        ("Thank you for shopping!", "footer message"),
        ("", "empty line")
    ]
    
    print("Feature extraction results:")
    print("-" * 60)
    
    for i, (text, description) in enumerate(test_lines):
        features = extractor.extract_features(text, i, len(test_lines))
        
        print(f"\nLine {i+1}: '{text}' ({description})")
        
        # Show key features
        key_features = [
            'position_ratio', 'is_header_area', 'is_body_area', 'is_footer_area',
            'transaction_summary_count', 'itemization_count', 'tax_and_fees_count',
            'has_price', 'has_percentage', 'total_keyword_matches'
        ]
        
        for feature in key_features:
            if feature in features:
                print(f"  {feature}: {features[feature]}")
    
    print("\n✅ Advanced feature extraction test completed!")


if __name__ == "__main__":
    test_advanced_feature_extractor()
