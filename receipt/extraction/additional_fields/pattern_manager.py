#!/usr/bin/env python3
"""
Pattern Manager for Additional Fields Extractor v2.7.1
Centralizes all regex patterns and configuration for multi-country extraction
"""

import json
import os
import re

class PatternManager:
    """
    Manages all regex patterns for the extraction system.
    Provides centralized pattern access and configuration.
    """
    
    def __init__(self, config_path=None):
        """Initialize pattern manager with configuration."""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', 'config', 'receipt_extraction_config.json')
        
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize patterns
        self._init_payment_patterns()
        self._init_address_patterns()
        self._init_contact_patterns()
        self._init_discount_patterns()
        
        print(f"✅ Initialized Pattern Manager v2.7.1")
        print(f"   Multi-country patterns: ✅ Ready (UK, US, CA, AU, IN)")
    
    def _load_config(self):
        """Load extraction configuration."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            return {}
    
    def _init_payment_patterns(self):
        """Initialize payment-related patterns."""
        # Payment keywords from config
        self.payment_keywords = self.config.get('keyword_categories', {}).get('categories', {}).get('payment', {}).get('keywords', [])
        
        # Contact/address keywords from config
        self.contact_keywords = self.config.get('keyword_categories', {}).get('categories', {}).get('company_info', {}).get('keywords', [])
        
        # Card keywords for multi-language support
        self.card_keywords = {
            'english': [
                'card', 'credit', 'debit', 'visa', 'mastercard', 'amex',
                'credit card', 'debit card', 'payment by card', 'card payment',
                'paid by card', 'tender card', 'american express'
            ],
            'spanish': ['tarjeta', 'crédito', 'débito'],
            'french': ['carte', 'crédit'],
            'german': ['karte', 'kredit', 'kreditkarte'],
            'italian': ['carta', 'credito'],
            'portuguese': ['cartão']
        }
        
        # Change keywords for multi-language support
        self.change_keywords = {
            'english': [
                'change', 'balance', 'balance due', 'cash returned', 'refund',
                'return amount', 'change given', 'change due', 'your change',
                'cash back', 'returned'
            ],
            'spanish': ['cambio', 'devuelto', 'saldo', 'reembolso', 'vuelta'],
            'french': ['monnaie', 'rendu', 'solde', 'remboursement', 'retour'],
            'german': ['rückgeld', 'wechselgeld', 'zurück', 'rückerstattung'],
            'italian': ['resto', 'cambio', 'rimborso', 'saldo'],
            'portuguese': ['troco', 'devolução', 'saldo', 'reembolso'],
            'dutch': ['wisselgeld', 'terug', 'retour'],
            'chinese': ['找零', '找钱', '余额', '退款'],
            'japanese': ['お釣り', '釣り銭', '返金'],
            'hindi': ['बाकी', 'शेष', 'वापसी']
        }
        
        # Payment line patterns
        self.payment_line_patterns = [
            r'paid\s+by\s*:',           # "Paid By:"
            r'payment\s*:',             # "Payment:"
            r'tender\s*:',              # "Tender:"
            r'cash\s+tendered',         # "Cash Tendered"
            r'card\s+payment',          # "Card Payment"
            r'debit\s+card\s+\d+',      # "Debit Card 12.34"
            r'credit\s+card\s+\d+',     # "Credit Card 12.34"
            r'visa\s+\d+',              # "Visa 12.34"
            r'mastercard\s+\d+',        # "Mastercard 12.34"
            r'processed\s+by',          # "Processed by"
            r'payment\s+method',        # "Payment Method"
            r'payment\s+by\s+card',     # "Payment by card"
            r'paid\s+by\s+card',        # "Paid by card"
            r'tender\s+card',           # "Tender card"
        ]
        
        # Card number patterns
        self.card_number_patterns = [
            r'X{12,}\d{4}',          # XXXXXXXXXXXX1094
            r'\*{12,}\d{4}',         # ************1094  
            r'X{4,}\d{4}',           # XXXX1094
            r'\*{4,}\d{4}',          # ****1094
            r'\d{4}\*{4,}',          # 1094****
            r'\d{4}X{4,}',           # 1094XXXX
            r'\b\d{4}\b',            # Just 4 digits (when in card context)
        ]
        
        # Strict card patterns (from detect_card_number.py)
        self.strict_card_patterns = [
            r'(?:\*|X|x){16}(\d{4})',     # 16 masks + 4 digits
            r'(?:\*|X|x){15}(\d{4})',     # 15 masks + 4 digits
            r'(?:\*|X|x){14}(\d{4})',     # 14 masks + 4 digits
            r'(?:\*|X|x){13}(\d{4})',     # 13 masks + 4 digits
            r'(?:\*|X|x){12}(\d{4})',     # 12 masks + 4 digits
            r'(?:\*|X|x){11}(\d{4})',     # 11 masks + 4 digits
            r'(?:\*|X|x){16}',            # Full 16-character mask
        ]
        
        # Medium confidence card patterns
        self.medium_card_patterns = [
            (r'(\*{10,11}\d{4})', 'high'),
            (r'(X{10,11}\d{4})', 'high'),
            (r'(\*{8,9}\d{4})', 'medium'),
            (r'(X{8,9}\d{4})', 'medium'),
            (r'(x{8,11}\d{4})', 'medium'),
            (r'(\*{6,7}\d{4})', 'medium'),
            (r'(X{6,7}\d{4})', 'medium'),
            (r'(\*{4,5}\d{4})', 'medium'),
            (r'(X{4,5}\d{4})', 'medium'),
            (r'([\*X]+\d{4})', 'medium'),
        ]
        
        # Change patterns
        self.change_patterns = [
            r'change\s*(?:due|given|back)', r'your\s*change', r'change\s*[£$€¥₹][\d.]+',
            r'cash\s*back', r'returned', r'balance\s*due', r'no\s*change\s*due'
        ]
        
        # Product with card name exclusions
        self.product_with_card_patterns = [
            r'greeting\s+card',
            r'gift\s+card(?!\s+payment)',
            r'playing\s+card',
            r'birthday\s+card',
            r'christmas\s+card',
            r'post\s+card',
            r'business\s+card',
            r'memory\s+card',
            r'sd\s+card',
            r'sim\s+card',
            r'loyalty\s+card(?!\s+payment)',
            r'\d+\s+card',
        ]
    
    def _init_address_patterns(self):
        """Initialize address-related patterns with multi-country support."""
        # Multi-country postal/ZIP code patterns (v2.7.0)
        self.postcode_patterns = {
            'UK': r'\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d[A-Z]{2})\b',          # M1 1AA, W1A 0AX
            'US': r'\b\d{5}(?:-\d{4})?\b',                                    # 12345, 12345-6789
            'CA': r'\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b',                          # K1A 0A6, H0H 0H0
            'AU': r'\b\d{4}\b',                                              # 2000, 3000
            'IN': r'\b\d{6}\b',                                              # 110001, 400001
        }
        
        # Combined multi-country postcode pattern for auto-detection
        all_postcode_patterns = '|'.join(f'({pattern})' for pattern in self.postcode_patterns.values())
        self.postcode_pattern = re.compile(all_postcode_patterns, re.IGNORECASE)
        
        # Enhanced global address keywords pattern (v2.7.0)
        self.address_keywords_pattern = re.compile(
            r'\b(street|st|road|rd|avenue|ave|lane|ln|drive|dr|boulevard|blvd|'
            r'way|place|pl|court|ct|circle|terrace|square|park|'
            r'building|floor|suite|unit|apt|apartment|house|tower|block|'
            r'estate|nagar|marg|colony|sector|phase|plot|gali|chowk|'
            r'centre|center|plaza|mall|close|crescent|grove|'
            r'business\s+park|industrial\s+estate|retail\s+park|'
            r'town\s+centre|city\s+centre|shopping\s+center)\b',
            re.IGNORECASE
        )
        
        # Enhanced global address noise pattern (v2.7.0)
        self.address_noise_pattern = re.compile(
            r'\b(thank|total|amount|subtotal|payment|sale|cash|change|'
            r'visa|mastercard|amex|card\s*name|auth\s*code|approved|'
            r'terminal|merchant|transaction|receipt|stan|till|pump|'
            r'date|time|price|quantity|vat|tax|rate|excl|incl|'
            r'membership|balance|points|pin\s*verified|served|'
            r'service\s*charge|charity|donation|www\.|\.com|\.co\.|'
            r'tel|telephone|phone|mobile|fax|email|web|site|'
            r'barcode|code|id|ref|batch|sequence|application)\b',
            re.IGNORECASE
        )
        
        # Date patterns to exclude from addresses (v2.7.0 - enhanced)
        self.date_pattern = re.compile(
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|'                          # DD/MM/YYYY, DD-MM-YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b|'                            # YYYY/MM/DD, YYYY-MM-DD
            r'\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{2,4}\b|'  # DD MMM YYYY
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2},?\s+\d{2,4}\b|' # MMM DD, YYYY
            r'\b\d{2}:\d{2}(?::\d{2})?\b',                                 # Time formats
            re.IGNORECASE
        )
        
        # VAT/Tax number patterns (v2.7.0 - enhanced)
        self.vat_pattern = re.compile(
            r'\b(vat|tax|registration|gst|tin|ein|ssn)\s*(no|number|reg|id)[\s:.#-]*[a-zA-Z0-9]+\b|'
            r'\bvat\s*:\s*[a-zA-Z0-9\s]+\b|'
            r'\breg\s*no[\s:.#-]*[a-zA-Z0-9]+\b|'
            r'\btax\s*id[\s:.#-]*[a-zA-Z0-9]+\b|'
            r'\b(abn|acn|tfn)\s*[\s:.#-]*[a-zA-Z0-9]+\b',                  # Australian business numbers
            re.IGNORECASE
        )
        
        # Enhanced menu/product pattern (v2.7.0 - global)
        self.menu_items_pattern = re.compile(
            r'\b(water|cola|coke|pepsi|sprite|fanta|juice|tea|coffee|'
            r'burger|pizza|sandwich|chips|fries|salad|soup|pasta|'
            r'chicken|beef|pork|lamb|fish|paneer|tofu|rice|'
            r'starter|main|dessert|side|drink|beverage|appetizer|'
            r'bottle|can|glass|pint|wine|beer|ale|lager|'
            r'small|medium|large|regular|extra|super|'
            r'roti|paratha|puri|naan|biryani|curry|dosa|idli|'
            r'ice\s*cream|pancake|breakfast|toast|energy|monster|'
            r'stella|artois|diesel|fuel|unleaded|petrol|gasoline|'
            r'pack|box|bag|each|piece|item|qty|quantity)\b',
            re.IGNORECASE
        )
        
        # Enhanced technical system patterns (v2.7.0)
        self.technical_pattern = re.compile(
            r'\b(merchant|terminal|stan|receipt\s*no|application|transaction|'
            r'auth\s*code|approval|reference|batch|trace|sequence|'
            r'pos|eft|eftpos|pin\s*pad|card\s*reader|'
            r'barcode|sku|upc|ean|isbn|asin)\b', 
            re.IGNORECASE
        )
        
        # Address terminator patterns
        self.address_terminator_patterns = [
            # Contact information
            r'(?i)^(tel|telephone|phone|ph|mob|mobile|fax)[\s:\-]',
            r'(?i)^(email|e-mail|web|website)[\s:\-@]',
            
            # Business information  
            r'(?i)^(vat|tax)\s*(no|number|reg|#)[\s:\-]',
            r'(?i)^(store|site|branch)\s*(no|number|#)[\s:\-]',
            
            # Empty or very short lines
            r'^.{1,2}$',
            
            # Document references
            r'(?i)^(receipt|invoice|transaction)\s*(no|number|#)[\s:\-]',
        ]
        
        # Address indicators
        self.address_indicators = [
            # Street/road patterns
            r'\b\d+\s+\w+\s+(street|st|road|rd|avenue|ave|lane|ln|drive|dr|close|cl|way|place|pl)\b',
            
            # Building/location names
            r'\b(house|building|court|centre|center|park|gardens?|estate|plaza|mall)\b',
            
            # Area/city names
            r'^[A-Z][a-zA-Z\s\-\']{2,25}$',
            
            # Business areas/districts
            r'\b(business\s+park|industrial\s+estate|retail\s+park|town\s+centre|city\s+centre)\b',
        ]
    
    def _init_contact_patterns(self):
        """Initialize contact information patterns."""
        # Telephone patterns (v2.7.0 - multi-country)
        self.telephone_pattern = re.compile(
            r'\b(tel|telephone|phone|mobile|fax|call)[\s:.#-]*[\d\s\(\)\+\-\.]+\b|'
            r'\b\d{10,11}\b|'                                              # 10-11 digit sequences
            r'\b0\d{3}\s*\d{3}\s*\d{4}\b|'                                # UK landline
            r'\b\+44\s*\d+\b|'                                            # UK international
            r'\b\+1[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{4}\b|'             # US/CA international
            r'\b\+61\s*\d+\b|'                                            # AU international
            r'\b\+91\s*\d+\b|'                                            # IN international
            r'\b\(\d{3,5}\)\s*\d+\b|'                                     # Area code in brackets
            r'\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b',                         # US format XXX-XXX-XXXX
            re.IGNORECASE
        )
        
        # Phone label patterns
        self.phone_label_patterns = [
            r'(?i)(?:tel|telephone|phone|ph|mob|mobile|fax)[\s:\-\.\,_]*([0-9\s\(\)\+\-\.]+)',
            r'(?i)(?:te[1lt]|tet|tei|tel|ph[0o]ne|phone|m[0o]b|mob)[\s:\-\.\,_]*([0-9\s\(\)\+\-\.]+)',
            r'(?i)(?:t[e3][1l]|t[e3]t|phon[e3]|mob[i1l])[\s:\-\.\,_]*([0-9\s\(\)\+\-\.]+)',
            r'(?i)(?:tel|telephone|phone|ph|mob|mobile)[\s:\-\.\,_]*[\(\[\{]?([0-9\s\)\]\}\(\)\+\-\.]+)',
            r'(?i)(?:tel|telephone|phone|ph)[\s:\-\.\,_]*([0-9\s\(\)\+\-\.]+)[\s]*(?:ext|x|#|extension)[\s]*([0-9]+)?',
            r'(?i)(?:tel|telephone|phone|mob|mobile)([0-9]{10,11})',
        ]
        
        # General phone patterns
        self.general_phone_patterns = [
            r'\b(\+44[\s\-]?\d{2}[\s\-]?\d{4}[\s\-]?\d{4})\b',  # +44 format
            r'\b(\+44[\s\-]?\d{3}[\s\-]?\d{3}[\s\-]?\d{4})\b',
            r'\b(0\d{2}[\s\-]?\d{4}[\s\-]?\d{4})\b',  # 020 format (London)
            r'\b(0\d{3}[\s\-]?\d{3}[\s\-]?\d{4})\b',  # 0123 format
            r'\b(0\d{4}[\s\-]?\d{6})\b',              # 01234 format
            r'\b(07\d{3}[\s\-]?\d{6})\b',             # Mobile 07xxx
            r'[\(\[\{]?(0\d{3})[\)\]\}]?[\s\-]?(\d{3})[\s\-]?(\d{4})',
            r'[\(\[\{]?(0\d{4})[\)\]\}]?[\s\-]?(\d{6})',
        ]
        
        # Email patterns
        self.email_patterns = [
            r'(?i)(?:email|e-mail|mail)[\s:]*([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:com|co\.uk|org|net|gov|edu|info)\b',
        ]
        
        # Website patterns
        self.website_patterns = [
            r'www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
            r'https?://[A-Za-z0-9.-]+\.[A-Za-z]{2,}[/\w.-]*',
            r'(?:website|web|visit|url|site)[\s:]*([A-Za-z0-9.-]+\.(?:com|co\.uk|org|net|gov|edu))\b',
            r'(?:www\.)?([A-Za-z0-9.-]+\.(?:com|co\.uk|org|net|gov|edu))(?:\s|$)',
        ]
        
        # Barcode/ID detection patterns
        self.barcode_patterns = [
            r'\d{12,}',  # Very long number sequences (12+ digits)
            r'[A-Z]{2,}\d+',  # Letter-number combinations
            r'\d+[A-Z]+\d+',  # Number-letter-number patterns
            r'(?:barcode|code|id|ref|transaction|receipt|batch)',
        ]
    
    def _init_discount_patterns(self):
        """Initialize discount-related patterns."""
        # Discount patterns
        self.discount_patterns = [
            r'\bdiscount\b',
            r'\bsaving\b', 
            r'\bsaved\b',
            r'\boff\b',
            r'\breduction\b',
            r'\bcoupon\b',
            r'\bvoucher\b',
            r'\bpromo\b',
            r'\brebate\b',
            r'\b\d+%\s*(off|discount|saving)\b',
            r'\b(off|discount|saving)\s*\d+%\b',
            r'\bminus\b',
            r'\bless\b.*\d',
            r'\b-\s*£?\d+\.\d{2}\b'  # Negative amounts
        ]
        
        # Discount total patterns
        self.discount_total_patterns = [
            r'discount\s*:\s*([£$]?[\d,]+\.?\d*)',
            r'total\s+discount\s*:\s*([£$]?[\d,]+\.?\d*)',
            r'discount\s+total\s*:\s*([£$]?[\d,]+\.?\d*)',
            r'total\s+savings?\s*:\s*([£$]?[\d,]+\.?\d*)',
            r'you\s+saved?\s*:\s*([£$]?[\d,]+\.?\d*)'
        ]
        
        # Discount exclusions
        self.discount_exclusions = [
            'no loyalty card', 'loyalty card presented', 'register for loyalty',
            'rewards on', 'visit www', 'download the', 'use the website',
            'for more information', 'member benefits', 'enjoy'
        ]
        
        # VAT exclusions
        self.vat_exclusions = ['vat', 'tax', 'total net', 'net total', 'net vat', 'vat net']
        
        # Discount coupon patterns
        self.coupon_patterns = [
            r'\bcoupon\s+(used|applied|redeemed)\b',
            r'\bvoucher\s+(used|applied|redeemed)\b',
            r'\bpromo\s+code\b.*applied',
            r'\bgift\s+card\s+(used|applied)\b',
            r'\bpoints\s+redeemed\b',
            r'\bcashback\s+applied\b',
            r'\bloyalty\s+discount\b',
            r'\bmember\s+discount\b'
        ]
    
    def get_all_card_keywords(self):
        """Get all card keywords from all languages."""
        all_keywords = []
        for lang_keywords in self.card_keywords.values():
            all_keywords.extend(lang_keywords)
        return all_keywords
    
    def get_all_change_keywords(self):
        """Get all change keywords from all languages."""
        all_keywords = []
        for lang_keywords in self.change_keywords.values():
            all_keywords.extend(lang_keywords)
        return all_keywords
    
    def contains_any_postcode_pattern(self, text):
        """Check if text contains any multi-country postal/ZIP code pattern."""
        return bool(self.postcode_pattern.search(text))
    
    def is_address_noise_line(self, line):
        """Check if line contains transaction/system noise."""
        return bool(self.address_noise_pattern.search(line))
    
    def is_menu_item_line(self, line):
        """Check if line is a menu item."""
        if self.menu_items_pattern.search(line):
            return True
        if re.search(r'\d+\.\d{2}\s*$', line):  # Ends with price
            return True
        if re.match(r'^\d+\s+[A-Za-z]', line):  # Starts with quantity + item
            return True
        if re.search(r'\b(x\d+|qty\s*\d+|\d+\s*(pcs|pieces|units))\b', line, re.IGNORECASE):
            return True
        return False
    
    def has_address_keywords(self, line):
        """Check if line has address keywords."""
        return bool(self.address_keywords_pattern.search(line))
    
    def is_address_terminator(self, text):
        """Check if line indicates end of address block."""
        return any(re.search(pattern, text) for pattern in self.address_terminator_patterns)