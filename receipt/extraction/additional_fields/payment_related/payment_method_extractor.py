#!/usr/bin/env python3
"""
Payment Method Extractor for Payment Extraction v2.7.1
Specializes in payment method identification and validation
"""

import re


class PaymentMethodExtractor:
    """
    Identifies and validates payment methods from receipt text.
    Supports card, cash, digital, online, and voucher payment types.
    """
    
    def __init__(self, pattern_manager=None, amount_extractor=None):
        """Initialize payment method extractor with dependencies."""
        self.pattern_manager = pattern_manager
        self.amount_extractor = amount_extractor
        print(f"✅ Initialized Payment Method Extractor v2.7.1")
    
    def identify_payment_method(self, text):
        """
        Enhanced payment method identification returning broader categories.
        Returns: card/cash/digital/online/voucher/other for payment method field.
        
        Args:
            text: Text string to analyze
            
        Returns:
            String payment method type or None
        """
        text_lower = text.lower()
        text_normalized = re.sub(r'[^\w\s]', ' ', text_lower)
        
        # Digital wallets and mobile payments
        digital_patterns = [
            'apple pay', 'applepay', 'google pay', 'googlepay', 'samsung pay', 'samsungpay',
            'android pay', 'paypal', 'pay pal', 'contactless', 'tap to pay', 'nfc payment'
        ]
        
        # Online and electronic transfers
        online_patterns = [
            'online payment', 'web payment', 'internet banking', 'bank transfer',
            'direct transfer', 'wire transfer', 'bacs payment', 'faster payment'
        ]
        
        # Cash payments
        cash_patterns = [
            'cash', 'cash payment', 'cash tender', 'notes and coins',
            'exact change', 'change given', 'paid by: cash'
        ]
        
        # Card payments (any type of card)
        card_patterns = [
            'card', 'debit', 'credit', 'visa', 'mastercard', 'amex', 'american express',
            'maestro', 'discover', 'chip', 'pin', 'swipe', 'insert card'
        ]
        
        # Check patterns in priority order
        # 1. Digital payments first
        if any(pattern in text_lower for pattern in digital_patterns):
            return 'digital'
        
        # 2. Cash payments
        if any(pattern in text_lower for pattern in cash_patterns):
            return 'cash'
        
        # 3. Online/transfer payments
        if any(pattern in text_lower for pattern in online_patterns):
            return 'online'
        
        # 4. Card payments (broad category)
        if any(pattern in text_lower for pattern in card_patterns):
            return 'card'
        
        # 5. Gift cards and vouchers
        if any(pattern in text_lower for pattern in ['gift card', 'voucher', 'store credit']):
            return 'voucher'
        
        # Additional heuristics for edge cases
        if re.search(r'\*{4,}\d{4}|\d{4}\*{4,}|\*{12,}\d{4}', text):
            return 'card'
        
        if re.search(r'(£|$|€)\s*\d+\.\d{2}', text) and any(word in text_lower for word in ['paid', 'payment', 'total', 'charge']):
            return 'other'
        
        return None
    
    def detect_card_payment_simple(self, text):
        """
        Simple card payment detection with multi-language support.
        Excludes product names containing "card".
        
        Args:
            text: Text string to analyze
            
        Returns:
            Boolean indicating if this is a card payment
        """
        if not text:
            return False
            
        text_lower = text.lower()
        
        # First check for product exclusions
        product_exclusions = [
            'greeting card', 'gift card', 'playing card', 'birthday card',
            'christmas card', 'post card', 'business card', 'memory card',
            'sd card', 'sim card', 'loyalty card',
            'card printing', 'card set', 'card game'
        ]
        
        # If it matches a product pattern, not a payment
        for exclusion in product_exclusions:
            if exclusion in text_lower:
                # Exception: if it explicitly mentions payment, it might be a gift card payment
                if not any(payment_word in text_lower for payment_word in ['payment', 'paid by', 'tender']):
                    return False
        
        # Get all card keywords from all languages
        all_keywords = self.pattern_manager.get_all_card_keywords() if self.pattern_manager else []
        
        # Simple keyword matching (fast and effective)
        return any(keyword in text_lower for keyword in all_keywords)
    
    def is_actual_payment_line(self, text, row):
        """
        Enhanced validation for actual payment lines vs product names.
        Uses multiple heuristics to distinguish payment info from product listings.
        
        Args:
            text: Text string to validate
            row: DataFrame row with line metadata
            
        Returns:
            Boolean indicating if this is an actual payment line
        """
        text_lower = text.lower()
        line_type = str(row.get('predicted_class', '')).upper()
        
        # Enhanced card keywords detection
        card_keywords = self.pattern_manager.get_all_card_keywords() if self.pattern_manager else []
        has_card_keyword = any(keyword in text_lower for keyword in card_keywords)
        
        # Check for strong payment indicators
        if self.pattern_manager and hasattr(self.pattern_manager, 'payment_line_patterns'):
            for pattern in self.pattern_manager.payment_line_patterns:
                if re.search(pattern, text_lower):
                    return True
        
        # Enhanced: If line has card keyword + amount pattern, likely a payment
        if has_card_keyword:
            if self.amount_extractor:
                amount_pattern = self.amount_extractor.build_simple_amount_pattern()
            else:
                amount_pattern = r'[\$£€¥₹]\s*\d+|\d+[.,]\d{2}|\d+\s*[\$£€¥₹]'
            
            has_amount = re.search(amount_pattern, text)
            if has_amount:
                # Additional check: ensure it's not a product with "card" in the name
                if not self._is_product_with_card_name(text_lower):
                    return True
        
        # Exclude obvious product lines
        if self.pattern_manager and hasattr(self.pattern_manager, 'product_with_card_patterns'):
            for exclusion in self.pattern_manager.product_with_card_patterns:
                if re.search(exclusion, text_lower):
                    return False
        
        # Line type validation
        if line_type in ['HEADER']:  # Unlikely to have payment info in header
            return False
        
        # Additional context validation
        if 'card' in text_lower:
            product_terms = [
                'pack', 'set', 'box', 'each', 'piece', 'item',
                'color', 'colour', 'size', 'large', 'small', 'medium',
                'design', 'pattern', 'style', 'type',
                '£', '$', '€'  # Currency at the end suggests product pricing
            ]
            
            # Count product-like terms
            product_term_count = sum(1 for term in product_terms if term in text_lower)
            
            # If we found the word "card" but also multiple product terms, likely a product
            if product_term_count >= 1:
                # But check if it has explicit payment context
                payment_context = any(ctx in text_lower for ctx in [
                    'paid', 'payment', 'tender', 'by:', 'method', 'processed', 'transaction'
                ])
                if not payment_context:
                    return False
        
        # If the line contains monetary amounts and payment-related words, it's likely a payment
        has_amount = bool(re.search(r'[£$€]\s*\d+|\d+\.\d{2}', text))
        payment_words = ['cash', 'debit', 'credit', 'visa', 'mastercard', 'amex']
        has_payment_word = any(word in text_lower for word in payment_words)
        
        if has_amount and has_payment_word:
            # Additional check: ensure it's not just a product with these words
            payment_structure_patterns = [
                r'(cash|debit|credit|visa|mastercard)\s+[\d£$€]',  # "CASH 12.34"
                r'[\d£$€]+\s+(cash|debit|credit|visa|mastercard)',  # "12.34 CASH"
                r'by\s*:\s*(cash|card|debit|credit)',              # "By: CASH"
            ]
            
            for pattern in payment_structure_patterns:
                if re.search(pattern, text_lower):
                    return True
        
        # Default: if we can't determine clearly, be conservative
        explicit_indicators = ['paid', 'payment', 'tender', 'by:', 'processed', 'transaction']
        return any(indicator in text_lower for indicator in explicit_indicators)
    
    def _is_product_with_card_name(self, text_lower):
        """
        Check if text contains 'card' as part of a product name rather than payment method.
        
        Args:
            text_lower: Lowercased text to check
            
        Returns:
            Boolean indicating if this is a product with "card" in the name
        """
        if not self.pattern_manager or not hasattr(self.pattern_manager, 'product_with_card_patterns'):
            return False
        
        return any(re.search(pattern, text_lower) for pattern in self.pattern_manager.product_with_card_patterns)
