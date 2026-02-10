"""
Discount Detector - Detects discount context in receipts
"""
import re
import pandas as pd
from typing import Dict, Any, List


class DiscountDetector:
    """Detects discount context in receipts."""
    
    def __init__(self):
        self.discount_context = {
            'has_discounts': False,
            'discount_types': [],
            'discount_indicators': [],
            'exclusions_found': [],
            'source': 'fallback_method'
        }
    
    def detect_discounts(self, df) -> Dict[str, Any]:
        """Detect discounts in receipt data."""
        self.discount_context = self._has_actual_discounts(df)
        return self.discount_context
    
    def get_discount_context(self, df=None) -> Dict[str, Any]:
        """Get discount context (with optional detection)."""
        if df is not None:
            return self.detect_discounts(df)
        return self.discount_context
    
    def _has_actual_discounts(self, df) -> Dict[str, Any]:
        """
        AUTHORITATIVE discount detection method - single source of truth for discount existence.
        """
        text_column = 'cleaned_text' if 'cleaned_text' in df.columns else ('text' if 'text' in df.columns else 'original_text')
        
        # Combine all text for comprehensive analysis
        all_text = ' '.join(df[text_column].fillna('').astype(str)).lower()
        
        # Exclusions - text that should NOT be considered discount-related
        exclusions = [
            'no loyalty card', 'loyalty card presented', 'register for loyalty',
            'rewards on', 'visit www', 'download the', 'use the website',
            'for more information', 'member benefits', 'enjoy', 'fuelsave',
            'total net', 'net total', 'vat', 'tax'
        ]
        
        # Check exclusions first
        exclusions_found = []
        for exclusion in exclusions:
            if exclusion in all_text:
                exclusions_found.append(exclusion)
        
        # Look for explicit discount indicators with word boundaries
        discount_keywords = [
            r'\bdiscount\b', r'\bcoupon\s+(?:used|applied|redeemed)\b', 
            r'\bvoucher\s+(?:used|applied|redeemed)\b', r'\bpromo\s+code\b.*applied',
            r'\boffer\b', r'\bdeal\b', r'\breduction\b', r'\bmarkdown\b', 
            r'\bclearance\b', r'\bmember\s+discount\b', r'\bloyalty\s+discount\b', 
            r'\bstaff\s+discount\b', r'\bemployee\s+discount\b'
        ]
        
        # Specific discount amount/percentage patterns
        discount_amount_patterns = [
            r'\b\d+%\s*off\b',
            r'\boff\s+check\b',
            r'\boff\s+total\b',
            r'\b\d+\.\d{2}\s*off\b',
            r'£\d+\.\d{2}\s*off\b',
            r'\bdiscount\s+total\s*[-:]\s*£?\d+\.?\d*\b',
        ]
        
        # Context-aware "save" patterns (must be in discount context)
        save_discount_patterns = [
            r'\bsave\s*£\d+\.?\d*\b',
            r'\bsaved\s*£\d+\.?\d*\b',
            r'\byou\s*save\b',
            r'\btotal\s*saved\b',
            r'\bsave\s*\d+%',
            r'\bsaving\s*£\d+\.?\d*\b',
        ]
        
        discount_found = False
        discount_types = []
        discount_indicators = []
        raw_patterns_found = []
        
        # Check for explicit discount keywords
        for keyword_pattern in discount_keywords:
            match = re.search(keyword_pattern, all_text, re.IGNORECASE)
            if match:
                discount_found = True
                discount_indicators.append(match.group())
                raw_patterns_found.append(keyword_pattern)
        
        # Check for discount amount patterns
        for amount_pattern in discount_amount_patterns:
            match = re.search(amount_pattern, all_text, re.IGNORECASE)
            if match:
                discount_found = True
                discount_types.append('amount_discount')
                discount_indicators.append(match.group())
                raw_patterns_found.append(amount_pattern)
        
        # Check for "save" patterns in discount context
        for save_pattern in save_discount_patterns:
            matches = re.finditer(save_pattern, all_text, re.IGNORECASE)
            for match in matches:
                # Additional context check
                match_context = all_text[max(0, match.start()-10):match.end()+10]
                if 'fuelsave' not in match_context and 'diesel' not in match_context:
                    discount_found = True
                    discount_types.append('savings_discount')
                    discount_indicators.append(match.group())
                    raw_patterns_found.append(save_pattern)
                    break
        
        # Calculate discount confidence
        discount_confidence = 0.0
        if discount_indicators:
            discount_confidence = min(len(discount_indicators) * 0.4, 1.0)
            if exclusions_found:
                discount_confidence = max(discount_confidence * 0.6, 0.3)
        
        discount_context = {
            'has_discounts': discount_found and discount_confidence > 0.3,
            'discount_types': list(set(discount_types)),
            'discount_indicators': discount_indicators,
            'exclusions_found': exclusions_found,
            'raw_patterns_found': raw_patterns_found,
            'confidence': discount_confidence,
            'source': 'discount_detector_module'
        }
        
        if not discount_context['has_discounts']:
            print("ℹ️  No discount indicators found - treating as regular total")
        else:
            print(f"✅ Discounts detected: {discount_context['discount_types']}")
        
        return discount_context