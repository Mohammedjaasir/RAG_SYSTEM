#!/usr/bin/env python3
"""
Card Details Extractor for Payment Extraction v2.7.1
Specializes in card type, brand, and number extraction
"""

import re


class CardDetailsExtractor:
    """
    Extracts card-specific information including type, brand, and card numbers.
    Supports multiple card types with multi-language detection.
    """
    
    def __init__(self, pattern_manager=None, payment_method_extractor=None):
        """Initialize card details extractor with dependencies."""
        self.pattern_manager = pattern_manager
        self.payment_method_extractor = payment_method_extractor
        print(f"✅ Initialized Card Details Extractor v2.7.1")
    
    def extract_card_type_and_details(self, text):
        """
        Enhanced card type detection with multi-language support and hybrid approach.
        Extracts card type, brand, and card numbers with confidence scoring.
        
        Args:
            text: Text string to analyze
            
        Returns:
            Dictionary with card details (type, brand, number, masked)
        """
        if not text:
            return {
                'card_type': None,
                'card_brand': None,
                'card_number': None,
                'card_number_masked': None
            }
        
        # Initialize return structure
        card_details = {
            'card_type': None,
            'card_brand': None,
            'card_number': None,
            'card_number_masked': None
        }
        
        # Step 1: Try hybrid card number detection first
        hybrid_result = self.extract_card_number_hybrid(text)
        if hybrid_result.get('card_number'):
            card_details['card_number'] = hybrid_result['card_number']
            card_details['card_number_masked'] = hybrid_result['card_number_masked']
            card_details['detection_method'] = hybrid_result.get('detection_method', 'unknown')
            card_details['detection_confidence'] = hybrid_result.get('detection_confidence', 'unknown')
            found_card_number = True
        else:
            found_card_number = False
        
        # Step 2: Quick detection using simple approach
        if self.payment_method_extractor:
            simple_card_detected = self.payment_method_extractor.detect_card_payment_simple(text)
        else:
            simple_card_detected = False
        
        # If we found neither card number nor simple card detection, return empty
        if not found_card_number and not simple_card_detected:
            return card_details
        
        # Step 3: Detailed analysis (existing logic enhanced with multi-language)
        text_lower = text.lower()
        
        # Enhanced card type patterns with multi-language support
        card_type_patterns = {
            'debit_mastercard': [
                'debit mastercard', 'mastercard debit',
                'débito mastercard', 'mastercard débito',  # Spanish
                'carte débit mastercard', 'mastercard débit',  # French
                'karte debit mastercard', 'mastercard debit karte'  # German
            ],
            'credit_mastercard': [
                'credit mastercard', 'mastercard credit',
                'crédito mastercard', 'mastercard crédito',  # Spanish
                'carte crédit mastercard', 'mastercard crédit',  # French
                'kredit mastercard', 'kreditkarte mastercard'  # German
            ],
            'debit_visa': [
                'debit visa', 'visa debit', 'visa electron', 'visa delta',
                'débito visa', 'visa débito',  # Spanish
                'carte débit visa', 'visa débit',  # French
                'karte debit visa', 'visa debit karte'  # German
            ],
            'credit_visa': [
                'credit visa', 'visa credit', 'visa classic', 'visa gold', 'visa platinum',
                'crédito visa', 'visa crédito',  # Spanish
                'carte crédit visa', 'visa crédit',  # French
                'kredit visa', 'kreditkarte visa'  # German
            ],
            'debit': [
                'debit card', 'debit', 'débito', 'carte débit', 'karte debit',
                'carta di debito', 'cartão de débito'  # Italian and Portuguese
            ],
            'credit': [
                'credit card', 'credit', 'crédito', 'carte crédit', 'kredit', 'kreditkarte',
                'carte de crédit'  # French: "carte de crédit"
            ],
        }
        
        # Enhanced card brand patterns with multi-language support
        card_brand_patterns = {
            'visa': [
                'visa', 'visa card', 'carte visa', 'karte visa'
            ],
            'mastercard': [
                'mastercard', 'master card', r'\bmc\b',
                'carte mastercard', 'karte mastercard'
            ],
            'amex': [
                'amex', 'american express', 'carte amex', 'karte amex'
            ],
            'maestro': [
                'maestro', 'carte maestro', 'karte maestro'
            ],
            'discover': [
                'discover', 'carte discover', 'karte discover'
            ],
            'jcb': [
                'jcb', 'carte jcb', 'karte jcb'
            ],
            'diners': [
                'diners', 'diners club', 'carte diners', 'karte diners'
            ]
        }
        
        # Extract card type
        for card_type, patterns in card_type_patterns.items():
            for pattern in patterns:
                if pattern.startswith(r'\b') and pattern.endswith(r'\b'):
                    if re.search(pattern, text_lower):
                        card_details['card_type'] = card_type
                        break
                else:
                    if pattern in text_lower:
                        card_details['card_type'] = card_type
                        break
            if card_details['card_type']:
                break
        
        # Extract card brand (if not already determined)
        if not card_details['card_type'] or 'debit' in card_details['card_type'] or 'credit' in card_details['card_type']:
            for brand, patterns in card_brand_patterns.items():
                for pattern in patterns:
                    if pattern.startswith(r'\b') and pattern.endswith(r'\b'):
                        if re.search(pattern, text_lower):
                            card_details['card_brand'] = brand
                            break
                    else:
                        if pattern in text_lower:
                            card_details['card_brand'] = brand
                            break
                if card_details['card_brand']:
                    break
        
        # Legacy fallback - only use if hybrid detection didn't find anything
        if not found_card_number and self.pattern_manager and hasattr(self.pattern_manager, 'card_number_patterns'):
            for pattern in self.pattern_manager.card_number_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    card_details['card_number_masked'] = match.group(0)
                    # Extract just the visible digits
                    digits_match = re.search(r'\d{4}', match.group(0))
                    if digits_match:
                        card_details['card_number'] = digits_match.group(0)
                    card_details['detection_method'] = 'legacy_pattern'
                    card_details['detection_confidence'] = 'medium'
                    break
        
        return card_details
    
    def enhance_payment_methods_with_card_details(self, payment_details):
        """
        Enhance payment methods with detailed card information from separately extracted card_details.
        Updates card type, brand, and number information in payment methods.
        
        Args:
            payment_details: Dictionary with payment_methods and card_details lists
        """
        payment_methods = payment_details.get('payment_methods', [])
        card_details = payment_details.get('card_details', [])
        
        for payment_method in payment_methods:
            if payment_method.get('method') == 'card':
                # Find the best matching card details for this payment method
                best_card_number = None
                best_card_type = None
                best_card_brand = payment_method.get('card_brand')
                
                # Look for card number in card details
                for card_detail in card_details:
                    if card_detail.get('card_number_masked'):
                        best_card_number = card_detail['card_number_masked']
                        print(f"      💳 Enhanced payment method with card number: {best_card_number}")
                    
                    # Get more specific card type if available
                    if card_detail.get('card_type') and not best_card_type:
                        best_card_type = card_detail['card_type']
                    
                    # Get card brand if not already set
                    if card_detail.get('card_brand') and not best_card_brand:
                        best_card_brand = card_detail['card_brand']
                
                # Update payment method with enhanced details
                if best_card_number:
                    payment_method['card_number_masked'] = best_card_number
                    # Extract visible digits
                    digits_match = re.search(r'\d{4}', best_card_number)
                    if digits_match:
                        payment_method['card_number'] = digits_match.group(0)
                
                if best_card_type:
                    payment_method['card_type'] = best_card_type
                    
                if best_card_brand:
                    payment_method['card_brand'] = best_card_brand
    
    def classify_card_detail(self, text):
        """
        Enhanced classification of card processing details with modern payment patterns.
        Identifies auth codes, PAN, terminal info, and other card-related details.
        
        Args:
            text: Text string containing card detail
            
        Returns:
            String classification of card detail type
        """
        text_lower = text.lower()
        text_normalized = re.sub(r'[^\w\s*#:]', ' ', text_lower)
        
        # Authorization codes and reference numbers
        auth_patterns = [
            r'auth\s*code', r'authorization\s*code', r'auth\s*#', r'approval\s*code',
            r'reference\s*#', r'ref\s*#', r'transaction\s*#', r'txn\s*#',
            r'approval\s*#', r'trace\s*#'
        ]
        
        for pattern in auth_patterns:
            if re.search(pattern, text_lower):
                return 'authorization_code'
        
        # Primary Account Number (PAN) - masked card numbers
        if self.pattern_manager and hasattr(self.pattern_manager, 'strict_card_patterns'):
            for pattern in self.pattern_manager.strict_card_patterns[:5]:  # First 5 patterns for PAN
                if re.search(pattern, text):
                    return 'primary_account_number'
        
        # Application Identifier (chip card data)
        if re.search(r'aid\s*[:=]', text_lower):
            return 'application_identifier'
        
        # Terminal and merchant data
        terminal_patterns = [
            r'terminal\s*[:=]', r'tid\s*[:=]', r'merchant\s*[:=]', r'mid\s*[:=]',
            r'location\s*[:=]', r'store\s*[:=]'
        ]
        
        for pattern in terminal_patterns:
            if re.search(pattern, text_lower):
                return 'terminal_info'
        
        # Card brand identification
        card_brands = {
            'mastercard': [
                'mastercard', 'master card', 'mc card', 'debit mastercard',
                'credit mastercard', r'\bmc\b'
            ],
            'visa': [
                'visa', 'visa card', 'debit visa', 'credit visa',
                'visa electron', 'visa delta', 'visa classic'
            ],
            'amex': [
                'amex', 'american express', 'am ex', 'americanexpress'
            ],
            'discover': [
                'discover', 'discover card'
            ],
            'maestro': [
                'maestro', 'uk maestro'
            ],
            'jcb': [
                'jcb', 'jcb card'
            ],
            'diners': [
                'diners', 'diners club'
            ]
        }
        
        for brand, patterns in card_brands.items():
            for pattern in patterns:
                if pattern.startswith(r'\b') and pattern.endswith(r'\b'):
                    if re.search(pattern, text_normalized):
                        return 'card_brand'
                else:
                    if pattern in text_lower:
                        return 'card_brand'
        
        # Payment method types
        payment_types = [
            'contactless', 'chip and pin', 'chip & pin', 'magnetic stripe',
            'swipe', 'insert', 'tap', 'wave', 'nfc', 'emv'
        ]
        
        for payment_type in payment_types:
            if payment_type in text_lower:
                return 'payment_method_type'
        
        # Digital wallet identifiers
        digital_wallets = [
            'apple pay', 'google pay', 'samsung pay', 'android pay',
            'paypal', 'apple wallet'
        ]
        
        for wallet in digital_wallets:
            if wallet in text_lower:
                return 'digital_wallet'
        
        # Transaction status and verification
        if any(term in text_lower for term in ['approved', 'declined', 'verified', 'pin ok', 'signature']):
            return 'transaction_status'
        
        # Timestamps and dates related to card processing
        if re.search(r'\d{2}[:/]\d{2}[:/]\d{2,4}|\d{2}[:/]\d{2}|\b\d{2}:\d{2}\b', text):
            return 'transaction_timestamp'
        
        # Amounts and fees
        if re.search(r'(fee|charge|surcharge)', text_lower) and re.search(r'[£$€]\d+', text):
            return 'transaction_fee'
        
        # Receipt/batch numbers
        if re.search(r'(receipt|batch|sequence)\s*#?\s*\d+', text_lower):
            return 'receipt_reference'
        
        # CVV/Security code (usually wouldn't appear on receipt, but just in case)
        if re.search(r'cv[v2]|security\s*code', text_lower):
            return 'security_info'
        
        # Default classification for unrecognized card-related text
        if any(card_term in text_lower for card_term in ['card', 'payment', 'transaction', 'processed']):
            return 'card_processing_info'
        
        return 'other_card_detail'
    
    def extract_masked_card_digits_strict(self, text):
        """
        Strict card number detection inspired by detect_card_number.py.
        Only matches cards with 12+ character masks for high confidence.
        
        Args:
            text: Text string to analyze
            
        Returns:
            Dictionary with card detection results or None
        """
        if not self.pattern_manager or not hasattr(self.pattern_manager, 'strict_card_patterns'):
            return None
        
        # Strict patterns from detect_card_number.py - 12+ masks + digits
        for pattern in self.pattern_manager.strict_card_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Return first match with full details
                original_masked = match.group(0)  # Full matched string
                if match.groups():  # Has captured digits
                    last_digits = match.group(1)
                    return {
                        'last_digits': last_digits,
                        'original_masked': original_masked,
                        'mask_pattern': pattern,
                        'confidence': 'high',
                        'detection_method': 'strict_12plus_mask'
                    }
                else:
                    # Full mask with no visible digits
                    return {
                        'last_digits': None,
                        'original_masked': original_masked,
                        'mask_pattern': pattern,
                        'confidence': 'high',
                        'detection_method': 'strict_full_mask'
                    }
        
        return None
    
    def extract_card_number_hybrid(self, text):
        """
        Hybrid card number detection: strict detection + flexible fallback.
        Three-tier approach: strict > medium > context-based.
        
        Args:
            text: Text string to analyze
            
        Returns:
            Dictionary with card number extraction results
        """
        # Step 1: Try strict detection first (highest confidence)
        strict_result = self.extract_masked_card_digits_strict(text)
        if strict_result:
            return {
                'card_number': strict_result['last_digits'],
                'card_number_masked': strict_result['original_masked'],
                'detection_method': strict_result['detection_method'],
                'detection_confidence': strict_result['confidence']
            }
        
        # Step 2: Medium confidence patterns (existing logic enhanced)
        if self.pattern_manager and hasattr(self.pattern_manager, 'medium_card_patterns'):
            for pattern, confidence in self.pattern_manager.medium_card_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    masked_number = match.group(1)
                    digits_match = re.search(r'\d{4}', masked_number)
                    if digits_match:
                        digits = digits_match.group(0)
                        return {
                            'card_number': digits,
                            'card_number_masked': masked_number,
                            'detection_method': f'flexible_pattern_{confidence}',
                            'detection_confidence': confidence
                        }
        
        # Step 3: Context-based detection (lowest confidence)
        if self.payment_method_extractor:
            simple_card_detected = self.payment_method_extractor.detect_card_payment_simple(text)
        else:
            simple_card_detected = False
        
        if simple_card_detected:
            context_patterns = [
                (r'ending in\s*(\d{4})', 'low'),
                (r'ends in\s*(\d{4})', 'low'),
                (r'last\s*(\d{4})', 'low'),
                (r'\b(\d{4})\b(?=.*(?:card|visa|master|amex))', 'low'),
                (r'(?:card|visa|master|amex).*?(\d{4})', 'low'),
            ]
            
            for pattern, confidence in context_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    digits = match.group(1)
                    return {
                        'card_number': digits,
                        'card_number_masked': f"****{digits}",
                        'detection_method': 'context_based',
                        'detection_confidence': confidence
                    }
        
        # Also try patterns without strict card context but with high likelihood
        general_context_patterns = [
            (r'payment.*?ending in\s*(\d{4})', 'medium'),
            (r'transaction.*?(\d{4})', 'low'),
        ]
        
        for pattern, confidence in general_context_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                digits = match.group(1)
                return {
                    'card_number': digits, 
                    'card_number_masked': f"****{digits}",
                    'detection_method': 'general_context',
                    'detection_confidence': confidence
                }
        
        return {'card_number': None, 'card_number_masked': None}
