#!/usr/bin/env python3
"""
Vendor Classifier - Upfront Vendor Identification v1.0.0

Classifies vendor from OCR text before extraction:
- Vendor name detection (Tesco, Walmart, Reliance Fresh, etc.)
- Vendor type classification (grocery, petrol, pharmacy, restaurant)
- Country detection (via currency, phone, address patterns)
- Confidence scoring for classification accuracy

Benefits:
- Targeted RAG queries (vendor-specific patterns)
- Better Phi-3 context
- Faster extraction with pre-filtered knowledge

No user data stored - only patterns matched.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VendorClassification:
    """Result of vendor classification."""
    vendor_name: Optional[str] = None
    vendor_name_normalized: Optional[str] = None
    vendor_type: Optional[str] = None  # grocery, petrol, pharmacy, restaurant, ecommerce, other
    country: Optional[str] = None
    confidence: float = 0.0
    matched_patterns: List[str] = None
    
    def __post_init__(self):
        if self.matched_patterns is None:
            self.matched_patterns = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'vendor_name': self.vendor_name,
            'vendor_name_normalized': self.vendor_name_normalized,
            'vendor_type': self.vendor_type,
            'country': self.country,
            'confidence': self.confidence,
            'matched_patterns': self.matched_patterns
        }


class VendorClassifier:
    """
    Classifies vendor from receipt OCR text.
    
    Uses pattern matching + heuristics for fast, accurate classification.
    Runs before RAG retrieval to enable targeted queries.
    """
    
    def __init__(self, vendor_patterns: Optional[Dict[str, Any]] = None):
        """
        Initialize vendor classifier.
        
        Args:
            vendor_patterns: Optional custom vendor patterns (loaded from knowledge base)
        """
        self.vendor_patterns = vendor_patterns or self._load_default_patterns()
        logger.info(f"✅ VendorClassifier initialized with {len(self.vendor_patterns)} vendor patterns")
    
    def classify(self, ocr_text: str, country_hint: Optional[str] = None) -> VendorClassification:
        """
        Classify vendor from OCR text.
        
        Args:
            ocr_text: Raw or normalized OCR text
            country_hint: Optional country hint from previous detection
            
        Returns:
            VendorClassification with detected vendor, type, country
        """
        if not ocr_text or not ocr_text.strip():
            logger.warning("Empty OCR text for vendor classification")
            return VendorClassification(confidence=0.0)
        
        # Normalize text for matching
        text_upper = ocr_text.upper()
        text_lower = ocr_text.lower()
        
        # Step 1: Detect country (if not provided)
        country = country_hint or self._detect_country(ocr_text)
        
        # Step 2: Match vendor patterns
        vendor_match = self._match_vendor(text_upper, text_lower, country)
        
        if vendor_match:
            vendor_name, vendor_type, confidence, patterns = vendor_match
            
            # Normalize vendor name (title case)
            vendor_name_normalized = self._normalize_vendor_name(vendor_name)
            
            logger.info(f"✅ Classified vendor: {vendor_name_normalized} ({vendor_type}, {country}) confidence={confidence:.2f}")
            
            return VendorClassification(
                vendor_name=vendor_name,
                vendor_name_normalized=vendor_name_normalized,
                vendor_type=vendor_type,
                country=country,
                confidence=confidence,
                matched_patterns=patterns
            )
        else:
            # No specific vendor matched, try to detect generic type
            vendor_type = self._detect_vendor_type_generic(text_upper)
            
            logger.info(f"No specific vendor matched, generic type: {vendor_type}, country: {country}")
            
            return VendorClassification(
                vendor_type=vendor_type,
                country=country,
                confidence=0.3
            )
    
    def _detect_country(self, text: str) -> Optional[str]:
        """Detect country from currency and phone patterns."""
        # Currency detection (highest priority)
        if re.search(r'[£]', text):
            return 'UK'
        elif re.search(r'[€]', text):
            return 'EU'
        elif re.search(r'[$]', text) and not re.search(r'[A-Z]{2}\$', text):  # Not AU$, NZ$, etc.
            return 'US'
        elif re.search(r'[₹]|Rs\.?|INR', text, re.IGNORECASE):
            return 'IN'
        
        # Phone pattern detection
        if re.search(r'\+44|0\d{10}', text):
            return 'UK'
        elif re.search(r'\+91|0\d{10}', text):
            return 'IN'
        elif re.search(r'\+1[\s-]?\d{3}', text):
            return 'US'
        
        # VAT/GST detection
        if re.search(r'VAT\s*(?:Reg\.?|Registration|No\.?|Number)', text, re.IGNORECASE):
            return 'UK'
        elif re.search(r'GSTIN|GST\s*No', text, re.IGNORECASE):
            return 'IN'
        
        return None
    
    def _match_vendor(
        self,
        text_upper: str,
        text_lower: str,
        country: Optional[str]
    ) -> Optional[Tuple[str, str, float, List[str]]]:
        """
        Match vendor from patterns.
        
        Returns:
            (vendor_name, vendor_type, confidence, matched_patterns) or None
        """
        best_match = None
        best_confidence = 0.0
        
        for vendor_key, vendor_info in self.vendor_patterns.items():
            # Filter by country if specified
            vendor_country = vendor_info.get('country')
            if country and vendor_country and vendor_country != country:
                continue
            
            # Check name patterns
            name_patterns = vendor_info.get('name_patterns', [])
            matched_patterns = []
            match_count = 0
            
            for pattern in name_patterns:
                pattern_upper = pattern.upper()
                # Exact match (word boundary)
                if re.search(rf'\b{re.escape(pattern_upper)}\b', text_upper):
                    matched_patterns.append(pattern)
                    match_count += 2  # Exact match is strong
                # Partial match
                elif pattern_upper in text_upper:
                    matched_patterns.append(f"{pattern}*")
                    match_count += 1
            
            if match_count > 0:
                # Check additional patterns (address keywords, phone, website)
                address_keywords = vendor_info.get('address_keywords', [])
                for keyword in address_keywords:
                    if keyword.upper() in text_upper:
                        match_count += 0.5
                        matched_patterns.append(f"addr:{keyword}")
                
                phone_pattern = vendor_info.get('phone_pattern')
                if phone_pattern and re.search(phone_pattern, text_upper):
                    match_count += 0.5
                    matched_patterns.append("phone")
                
                website = vendor_info.get('website')
                if website and website.lower() in text_lower:
                    match_count += 0.5
                    matched_patterns.append("website")
                
                # Calculate confidence (normalize by max possible score)
                confidence = min(1.0, match_count / (len(name_patterns) * 2 + 2))
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = (
                        vendor_info.get('name', vendor_key),
                        vendor_info.get('type', 'other'),
                        confidence,
                        matched_patterns
                    )
        
        # Only return if confidence is reasonable
        if best_match and best_confidence >= 0.4:
            return best_match
        
        return None
    
    def _detect_vendor_type_generic(self, text_upper: str) -> Optional[str]:
        """Detect generic vendor type from keywords."""
        # Grocery keywords
        if any(kw in text_upper for kw in ['SUPERMARKET', 'GROCERY', 'FRESH', 'MART', 'STORE', 'SHOP']):
            return 'grocery'
        
        # Petrol/Gas station keywords
        if any(kw in text_upper for kw in ['PETROL', 'FUEL', 'GAS STATION', 'PUMP', 'SHELL', 'BP', 'EXXON']):
            return 'petrol'
        
        # Pharmacy keywords
        if any(kw in text_upper for kw in ['PHARMACY', 'CHEMIST', 'DRUG STORE', 'MEDICAL']):
            return 'pharmacy'
        
        # Restaurant keywords
        if any(kw in text_upper for kw in ['RESTAURANT', 'CAFE', 'COFFEE', 'DINER', 'BISTRO', 'EATERY']):
            return 'restaurant'
        
        # Ecommerce keywords
        if any(kw in text_upper for kw in ['AMAZON', 'EBAY', 'ORDER', 'INVOICE', 'DISPATCH']):
            return 'ecommerce'
        
        return 'other'
    
    def _normalize_vendor_name(self, vendor_name: str) -> str:
        """Normalize vendor name to consistent format."""
        # Title case, but preserve known acronyms
        acronyms = ['BP', 'M&S', 'ASDA', 'UK', 'US', 'PLC', 'LTD', 'LLC', 'CO']
        
        words = vendor_name.split()
        normalized = []
        
        for word in words:
            if word.upper() in acronyms:
                normalized.append(word.upper())
            else:
                normalized.append(word.title())
        
        return ' '.join(normalized)
    
    def _load_default_patterns(self) -> Dict[str, Any]:
        """Load default vendor patterns (expandable via JSON later)."""
        return {
            # UK Vendors
            'tesco_uk': {
                'name': 'Tesco',
                'name_patterns': ['Tesco', 'TESCO PLC', 'Tesco Stores'],
                'type': 'grocery',
                'country': 'UK',
                'address_keywords': ['Welwyn Garden', 'Cheshunt', 'Hertfordshire'],
                'website': 'tesco.com',
                'phone_pattern': r'0800.*345.*677'
            },
            'sainsburys_uk': {
                'name': "Sainsbury's",
                'name_patterns': ["Sainsbury's", 'SAINSBURYS', 'J SAINSBURY'],
                'type': 'grocery',
                'country': 'UK',
                'address_keywords': ['Holborn', 'London'],
                'website': 'sainsburys.co.uk'
            },
            'asda_uk': {
                'name': 'ASDA',
                'name_patterns': ['ASDA', 'ASDA STORES'],
                'type': 'grocery',
                'country': 'UK',
                'address_keywords': ['Leeds', 'Yorkshire'],
                'website': 'asda.com'
            },
            'morrisons_uk': {
                'name': 'Morrisons',
                'name_patterns': ['Morrisons', 'WM MORRISON'],
                'type': 'grocery',
                'country': 'UK',
                'address_keywords': ['Bradford', 'Yorkshire'],
                'website': 'morrisons.com'
            },
            'waitrose_uk': {
                'name': 'Waitrose',
                'name_patterns': ['Waitrose', 'WAITROSE & PARTNERS'],
                'type': 'grocery',
                'country': 'UK',
                'website': 'waitrose.com'
            },
            'marks_spencer_uk': {
                'name': 'Marks & Spencer',
                'name_patterns': ['M&S', 'MARKS & SPENCER', 'MARKS AND SPENCER'],
                'type': 'grocery',
                'country': 'UK',
                'address_keywords': ['Baker Street', 'London'],
                'website': 'marksandspencer.com'
            },
            'lidl_uk': {
                'name': 'Lidl',
                'name_patterns': ['LIDL', 'LIDL UK'],
                'type': 'grocery',
                'country': 'UK',
                'website': 'lidl.co.uk'
            },
            'aldi_uk': {
                'name': 'Aldi',
                'name_patterns': ['ALDI', 'ALDI STORES'],
                'type': 'grocery',
                'country': 'UK',
                'website': 'aldi.co.uk'
            },
            'bp_uk': {
                'name': 'BP',
                'name_patterns': ['BP', 'BP EXPRESS', 'BP CONNECT'],
                'type': 'petrol',
                'country': 'UK',
                'website': 'bp.com',
                'phone_pattern': r'0800.*40.*24.*02'
            },
            'shell_uk': {
                'name': 'Shell',
                'name_patterns': ['SHELL', 'SHELL UK'],
                'type': 'petrol',
                'country': 'UK',
                'website': 'shell.co.uk'
            },
            
            # India Vendors
            'reliance_fresh_in': {
                'name': 'Reliance Fresh',
                'name_patterns': ['Reliance Fresh', 'RELIANCE RETAIL', 'R-FRESH'],
                'type': 'grocery',
                'country': 'IN',
                'address_keywords': ['Mumbai', 'Maharashtra'],
                'website': 'relianceretail.com'
            },
            'big_bazaar_in': {
                'name': 'Big Bazaar',
                'name_patterns': ['Big Bazaar', 'BIG BAZAAR', 'Future Retail'],
                'type': 'grocery',
                'country': 'IN',
                'address_keywords': ['Mumbai'],
                'website': 'bigbazaar.com'
            },
            'dmart_in': {
                'name': 'DMart',
                'name_patterns': ['DMart', 'D-MART', 'AVENUE SUPERMARTS'],
                'type': 'grocery',
                'country': 'IN',
                'address_keywords': ['Mumbai', 'Maharashtra'],
                'website': 'dmart.in'
            },
            'more_in': {
                'name': 'More Supermarket',
                'name_patterns': ['MORE', 'More Supermarket', 'MORE MEGASTORE'],
                'type': 'grocery',
                'country': 'IN',
                'website': 'morestore.com'
            },
            'spencers_in': {
                'name': "Spencer's",
                'name_patterns': ["Spencer's", 'SPENCERS RETAIL'],
                'type': 'grocery',
                'country': 'IN',
                'address_keywords': ['Kolkata'],
                'website': 'spencersretail.com'
            },
            
            # US Vendors
            'walmart_us': {
                'name': 'Walmart',
                'name_patterns': ['Walmart', 'WAL-MART', 'WALMART STORES'],
                'type': 'grocery',
                'country': 'US',
                'address_keywords': ['Bentonville', 'Arkansas'],
                'website': 'walmart.com'
            },
            'target_us': {
                'name': 'Target',
                'name_patterns': ['TARGET', 'TARGET STORES'],
                'type': 'grocery',
                'country': 'US',
                'address_keywords': ['Minneapolis'],
                'website': 'target.com'
            },
            'kroger_us': {
                'name': 'Kroger',
                'name_patterns': ['KROGER', 'THE KROGER CO'],
                'type': 'grocery',
                'country': 'US',
                'address_keywords': ['Cincinnati', 'Ohio'],
                'website': 'kroger.com'
            },
            'costco_us': {
                'name': 'Costco',
                'name_patterns': ['COSTCO', 'COSTCO WHOLESALE'],
                'type': 'grocery',
                'country': 'US',
                'address_keywords': ['Issaquah', 'Washington'],
                'website': 'costco.com'
            },
            'whole_foods_us': {
                'name': 'Whole Foods',
                'name_patterns': ['WHOLE FOODS', 'WHOLE FOODS MARKET'],
                'type': 'grocery',
                'country': 'US',
                'address_keywords': ['Austin', 'Texas'],
                'website': 'wholefoodsmarket.com'
            },
            'amazon_us': {
                'name': 'Amazon',
                'name_patterns': ['AMAZON', 'AMAZON.COM', 'AMAZON LLC'],
                'type': 'ecommerce',
                'country': 'US',
                'address_keywords': ['Seattle', 'Washington'],
                'website': 'amazon.com'
            }
        }
    
    def update_patterns(self, new_patterns: Dict[str, Any]):
        """
        Update vendor patterns (for dynamic learning).
        
        Args:
            new_patterns: New vendor patterns to merge
        """
        self.vendor_patterns.update(new_patterns)
        logger.info(f"Updated vendor patterns, now {len(self.vendor_patterns)} vendors")


# Singleton instance
_vendor_classifier_instance = None


def get_vendor_classifier() -> VendorClassifier:
    """Get singleton VendorClassifier instance."""
    global _vendor_classifier_instance
    if _vendor_classifier_instance is None:
        _vendor_classifier_instance = VendorClassifier()
    return _vendor_classifier_instance
