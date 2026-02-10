#!/usr/bin/env python3
"""
Payment Extractor for Additional Fields Extractor v2.7.1
Orchestrates payment information extraction using specialized extractors
"""

import re
from .amount_extractor import AmountExtractor
from .payment_method_extractor import PaymentMethodExtractor
from .card_details_extractor import CardDetailsExtractor
from .change_extractor import ChangeExtractor


class PaymentExtractor:
    """
    Orchestrator class that coordinates payment information extraction.
    Delegates to specialized extractors for amounts, methods, card details, and change.
    """
    
    def __init__(self, pattern_manager=None):
        """
        Initialize payment extractor with all specialized extractors.
        
        Args:
            pattern_manager: PatternManager instance for centralized patterns
        """
        self.pattern_manager = pattern_manager
        
        # Initialize specialized extractors
        self.amount_extractor = AmountExtractor(pattern_manager)
        self.payment_method_extractor = PaymentMethodExtractor(pattern_manager, self.amount_extractor)
        self.card_details_extractor = CardDetailsExtractor(pattern_manager, self.payment_method_extractor)
        self.change_extractor = ChangeExtractor(pattern_manager, self.amount_extractor)
        
        print(f"✅ Initialized Payment Extractor v2.7.1")
    
    def extract_payment_details(self, df):
        """
        Enhanced payment information extraction with comprehensive method detection.
        Orchestrates all specialized extractors to extract payment methods, amounts, 
        card details, and change information.
        
        Args:
            df: DataFrame with receipt text and predictions
            
        Returns:
            Dictionary with complete payment details including methods, amounts, 
            card info, and change information
        """
        
        payment_details = {
            'payment_methods': [],
            'payment_amounts': [],
            'change_details': [],
            'card_details': [],
            'cash_details': [],
            'payment_summary': {}
        }
        
        # Priority 2B: Look in multiple line types for payment information
        payment_lines = df[df['predicted_class'].isin(['ITEM_DATA', 'SUMMARY_KEY_VALUE', 'FOOTER', 'HEADER'])]
        
        total_payment_amount = 0
        payment_methods_found = set()
        
        for _, row in payment_lines.iterrows():
            text = str(row['text']).strip()
            text_lower = text.lower()
            
            if len(text) < 2:  # Skip very short lines
                continue
            
            # Extract payment methods with enhanced detection
            payment_method = self.payment_method_extractor.identify_payment_method(text)
            if payment_method:
                print(f"      🔍 PAYMENT METHOD DEBUG: '{text}' -> payment_method: '{payment_method}'")
                # Enhanced validation to exclude product lines
                if self.payment_method_extractor.is_actual_payment_line(text, row):
                    # Use enhanced card-specific amount extraction for card payments
                    if payment_method == 'card':
                        amount = self.amount_extractor.extract_card_amount_from_line(text)
                        if not amount:  # Fallback to general amount extraction
                            amount = self.amount_extractor.extract_amount_from_text(text)
                    else:
                        amount = self.amount_extractor.extract_amount_from_text(text)
                    
                    # Extract card-specific details for card payments
                    card_details = None
                    if payment_method == 'card':
                        card_details = self.card_details_extractor.extract_card_type_and_details(text)
                        print(f"      🏧 DEBUG: Card details for '{text}': {card_details}")
                    
                    # Additional validation - ensure this looks like a payment line
                    payment_indicators = [
                        'paid', 'payment', 'charge', 'total', 'amount', 'tender',
                        'tendered', 'by:', 'method', 'transaction', 'processed'
                    ]
                    
                    has_payment_indicator = any(indicator in text_lower for indicator in payment_indicators)
                    
                    if has_payment_indicator or amount:
                        confidence = float(row.get('confidence_score', 0.7))
                        
                        # Boost confidence for clear payment indicators
                        if has_payment_indicator:
                            confidence = min(confidence + 0.1, 1.0)
                        
                        # Extra boost for explicit payment patterns
                        if any(pattern in text_lower for pattern in ['paid by:', 'payment:', 'tendered']):
                            confidence = min(confidence + 0.2, 1.0)
                        
                        payment_entry = {
                            'method': payment_method,
                            'raw_text': text,
                            'amount': amount,
                            'line_number': int(row.get('line_number', 0)),
                            'confidence': confidence,
                            'payment_indicators': [ind for ind in payment_indicators if ind in text_lower]
                        }
                        
                        # Add card-specific details if this is a card payment
                        if card_details:
                            payment_entry.update({
                                'card_type': card_details.get('card_type'),
                                'card_brand': card_details.get('card_brand'), 
                                'card_number': card_details.get('card_number'),
                                'card_number_masked': card_details.get('card_number_masked')
                            })
                        
                        payment_details['payment_methods'].append(payment_entry)
                        payment_methods_found.add(payment_method)
                        
                        if amount:
                            total_payment_amount += amount
                        
                        print(f"      💳 Found payment method: {payment_method} - {text} (amount: {amount})")
                        if card_details and card_details.get('card_type'):
                            print(f"         🏧 Card details: {card_details.get('card_type')} {card_details.get('card_brand')} {card_details.get('card_number_masked')}")
                else:
                    print(f"      ❌ Rejected as product line: {text} (detected method: {payment_method})")
            
            # Enhanced change information extraction with multi-language support
            enhanced_is_change = self.change_extractor.detect_change_simple(text)
            
            # Traditional patterns (for backward compatibility)
            traditional_is_change = False
            if not enhanced_is_change:
                if self.pattern_manager and hasattr(self.pattern_manager, 'change_patterns'):
                    pattern_match = any(re.search(pattern, text_lower) for pattern in self.pattern_manager.change_patterns)
                    if pattern_match:
                        # Additional validation for traditional patterns
                        has_amount = bool(re.search(r'[£$€¥₹]\s*\d+|\d+[.,]\d{2}|\d+\s*[£$€¥₹]', text))
                        has_context = any(term in text_lower for term in ['due', 'given', 'tendered', ':', 'total'])
                        
                        # Exclude obvious non-monetary contexts
                        exclusions = ['spare change', 'loose change', 'exchange rate', 'oil change']
                        has_exclusion = any(excl in text_lower for excl in exclusions)
                        
                        traditional_is_change = (has_amount or has_context) and not has_exclusion
            
            # Use either detection method
            is_change_line = traditional_is_change or enhanced_is_change
            
            if is_change_line:
                amount = self.amount_extractor.extract_amount_from_text(text)
                
                # Handle "No Change Due" as amount = 0
                if re.search(r'no\s*change\s*due', text_lower):
                    amount = 0.0
                    change_type = 'no_change_due'
                elif any(term in text_lower for term in ['due', 'owed', 'owing']):
                    change_type = 'change_due'
                elif any(term in text_lower for term in ['given', 'returned', 'back', 'rendu', 'devuelto', 'zurück']):
                    change_type = 'change_given'
                elif any(term in text_lower for term in ['refund', 'reembolso', 'remboursement', 'rückerstattung', 'rimborso']):
                    change_type = 'refund'
                else:
                    change_type = 'change_given'
                
                change_entry = {
                    'raw_text': text,
                    'amount': amount,
                    'line_number': int(row.get('line_number', 0)),
                    'confidence': float(row.get('confidence_score', 0.7)),
                    'change_type': change_type,
                    'detection_method': 'enhanced_traditional' if traditional_is_change else 'multi_language_enhanced'
                }
                payment_details['change_details'].append(change_entry)
                print(f"      💰 Found change info: {text} (amount: {amount}, type: {change_type})")
            
            # Extract specific cash payment details
            if 'cash' in text_lower and not is_change_line:
                amount = self.amount_extractor.extract_amount_from_text(text)
                if amount:
                    cash_entry = {
                        'raw_text': text,
                        'amount': amount,
                        'line_number': int(row.get('line_number', 0)),
                        'confidence': float(row.get('confidence_score', 0.7)),
                        'cash_type': 'cash_tendered'
                    }
                    payment_details['cash_details'].append(cash_entry)
                    print(f"      💵 Found cash payment: {text} (amount: {amount})")
        
        # Also check ALL lines for detailed payment processing information
        all_lines = df[df['predicted_class'].isin(['IGNORE', 'FOOTER', 'HEADER'])]
        for _, row in all_lines.iterrows():
            text = str(row['text']).strip()
            text_lower = text.lower()
            
            # Enhanced card processing detail detection
            card_indicators = [
                'auth', 'pan', 'aid', 'mastercard', 'visa', 'contactless',
                'chip', 'pin', 'terminal', 'merchant', 'approval', 'ref',
                'transaction', 'batch', 'trace', 'sequence', 'card name',
                'expires', 'start date', 'seq. number'
            ]
            
            has_card_indicator = any(indicator in text_lower for indicator in card_indicators)
            
            # Enhanced card number patterns  
            has_card_number = False
            if self.pattern_manager and hasattr(self.pattern_manager, 'card_number_patterns'):
                has_card_number = any(re.search(pattern, text) for pattern in self.pattern_manager.card_number_patterns)
            
            if has_card_indicator or has_card_number:
                detail_type = self.card_details_extractor.classify_card_detail(text)
                
                # Extract card details for this line
                card_info = self.card_details_extractor.extract_card_type_and_details(text)
                
                card_detail_entry = {
                    'raw_text': text,
                    'line_number': int(row.get('line_number', 0)),
                    'detail_type': detail_type,
                    'confidence': float(row.get('confidence_score', 0.6)),
                    'indicators_found': [ind for ind in card_indicators if ind in text_lower]
                }
                
                # Add extracted card information
                if card_info.get('card_type'):
                    card_detail_entry['card_type'] = card_info['card_type']
                if card_info.get('card_brand'):
                    card_detail_entry['card_brand'] = card_info['card_brand']
                if card_info.get('card_number'):
                    card_detail_entry['card_number'] = card_info['card_number']
                if card_info.get('card_number_masked'):
                    card_detail_entry['card_number_masked'] = card_info['card_number_masked']
                
                payment_details['card_details'].append(card_detail_entry)
                print(f"      🏧 Found card detail ({detail_type}): {text}")
                if card_info.get('card_number_masked'):
                    print(f"         💳 Card number: {card_info['card_number_masked']}")
        
        # Enhance payment methods with detailed card information from card_details
        self.card_details_extractor.enhance_payment_methods_with_card_details(payment_details)
        
        # Enhanced change detection using proximity search (post-processing)
        if len(payment_details['change_details']) == 0:
            print("   🔍 No change details found with traditional methods, trying proximity search...")
            proximity_change_details = self.change_extractor.extract_change_amount_proximity(df)
            if proximity_change_details:
                payment_details['change_details'].extend(proximity_change_details)
                print(f"   ✅ Found {len(proximity_change_details)} change details using proximity search")
        
        # Generate payment summary
        payment_details['payment_summary'] = {
            'total_payment_methods': len(payment_details['payment_methods']),
            'unique_methods_count': len(payment_methods_found),
            'methods_found': list(payment_methods_found),
            'total_payment_amount': total_payment_amount if total_payment_amount > 0 else None,
            'has_change': len(payment_details['change_details']) > 0,
            'has_card_details': len(payment_details['card_details']) > 0,
            'has_cash_details': len(payment_details['cash_details']) > 0
        }
        
        print(f"   ✅ Payment extraction summary: {len(payment_details['payment_methods'])} methods, {len(payment_details['card_details'])} card details")
        
        # DEDUPLICATION: Keep only the best payment method (highest confidence)
        if len(payment_details['payment_methods']) > 1:
            sorted_methods = sorted(payment_details['payment_methods'], 
                                  key=lambda x: x.get('confidence', 0), 
                                  reverse=True)
            best_payment = sorted_methods[0]
            removed_count = len(payment_details['payment_methods']) - 1
            print(f"   🧹 Deduplication: Kept best payment method (confidence: {best_payment.get('confidence', 0):.0%}), removed {removed_count} duplicate(s)")
            payment_details['payment_methods'] = [best_payment]
        
        return payment_details
