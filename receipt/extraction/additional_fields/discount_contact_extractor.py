#!/usr/bin/env python3
"""
Discount and Contact Extractor for Additional Fields Extractor v2.7.1
Specializes in discount details and contact information extraction
"""

import re
import pandas as pd
from .pattern_manager import PatternManager
from .address_extractor import AddressExtractor

class DiscountContactExtractor:
    """
    Extracts discount information and contact details including addresses, phones, emails.
    """
    
    def __init__(self, pattern_manager=None, address_extractor=None):
        """Initialize discount and contact extractor."""
        self.pattern_manager = pattern_manager or PatternManager()
        self.address_extractor = address_extractor or AddressExtractor(self.pattern_manager)
        print(f"✅ Initialized Discount & Contact Extractor v2.7.1")
    
    def extract_discount_details(self, df, discount_context=None):
        """
        Priority 2B: Extract detailed discount information using context from comprehensive extractor.
        
        RESPONSIBILITY BOUNDARY: 
        - Comprehensive Extractor: Determines IF discounts exist and provides context
        - Additional Extractor: Extracts SPECIFIC discount details and amounts
        """
        
        discount_details = {
            'discounts': [],
            'savings': [], 
            'coupons': [],
            'promotional_offers': [],
            'loyalty_discounts': [],
            'staff_discounts': []
        }
        
        # Priority 2B: Rely entirely on comprehensive extractor's discount context
        if not discount_context or not discount_context.get('has_discounts', False):
            print("ℹ️  No discount context from comprehensive extractor - returning empty discount details")
            return discount_details
        
        print(f"🎯 Priority 2B: Processing detailed discount extraction with context: {discount_context.get('source', 'unknown')}")
        print(f"   Discount types detected by comprehensive extractor: {discount_context.get('discount_types', [])}")
        
        # Only look for detailed discount information, not existence determination
        discount_lines = df[df['predicted_class'].isin(['ITEM_DATA', 'SUMMARY_KEY_VALUE', 'FOOTER'])]
        
        # Priority 2B: Extract detailed discount information based on context
        for _, row in discount_lines.iterrows():
            text = str(row['text']).strip()
            text_lower = text.lower()
            
            # Skip empty lines
            if not text or len(text.strip()) < 3:
                continue
            
            # Extract specific discount details based on comprehensive extractor's findings
            discount_type_found = None
            amount = self._extract_amount_from_text(text)
            percentage = self._extract_percentage_from_text(text)
            
            # EXCLUDE subtotal lines from discount extraction
            if any(subtotal_keyword in text_lower for subtotal_keyword in ['subtotal', 'sub total', 'sub-total']):
                continue
            
            # Categorize discounts based on patterns
            if any(keyword in text_lower for keyword in ['member', 'loyalty', 'club']):
                discount_type_found = 'loyalty_discounts'
            elif any(keyword in text_lower for keyword in ['staff', 'employee']):
                discount_type_found = 'staff_discounts'  
            elif any(keyword in text_lower for keyword in ['coupon', 'voucher', 'promo']):
                discount_type_found = 'coupons'
            elif any(keyword in text_lower for keyword in ['offer', 'deal', 'promotion']):
                discount_type_found = 'promotional_offers'
            elif any(keyword in text_lower for keyword in ['discount', 'off', 'save', 'reduction']):
                discount_type_found = 'discounts'
            elif any(keyword in text_lower for keyword in ['saving']):
                discount_type_found = 'savings'
            
            # Only add if we found a specific discount type
            if (discount_type_found and 
                (amount or percentage or any(ind in text_lower for ind in ['-', 'off', 'discount', 'save'])) and
                'total' not in text_lower):
                
                discount_entry = {
                    'raw_text': text,
                    'amount': amount,
                    'percentage': percentage,
                    'line_number': int(row.get('line_number', 0)),
                    'confidence': float(row.get('confidence_score', 0.0)),
                    'discount_category': discount_type_found.replace('_', ' ').title()
                }
                
                discount_details[discount_type_found].append(discount_entry)
                print(f"      📄 Found {discount_type_found}: {text} (amount: {amount}, %: {percentage})")
        
        # Look for total discount summary lines first
        total_discount_amount = None
        for _, row in df.iterrows():
            text = str(row.get('text', '')).strip()
            text_lower = text.lower()
            
            # Look for summary discount lines
            for pattern in self.pattern_manager.discount_total_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    try:
                        amount_str = match.group(1).replace('£', '').replace('$', '').replace(',', '')
                        total_discount_amount = float(amount_str)
                        print(f"      🎯 Found total discount summary: {text} -> £{total_discount_amount}")
                        break
                    except (ValueError, TypeError):
                        continue
            
            if total_discount_amount:
                break
        
        # If no summary line found, calculate from individual discount amounts
        if total_discount_amount is None:
            total_discount_amount = 0.0
            discount_categories = ['discounts', 'savings', 'coupons', 'promotional_offers', 'loyalty_discounts', 'staff_discounts']
            for category in discount_categories:
                if category in discount_details and isinstance(discount_details[category], list):
                    for discount in discount_details[category]:
                        if isinstance(discount, dict) and discount.get('amount'):
                            try:
                                total_discount_amount += float(discount['amount'])
                            except (ValueError, TypeError):
                                continue
        
        # Add total discount amount to the discount_details structure
        discount_details['total_discount_amount'] = total_discount_amount if total_discount_amount and total_discount_amount > 0 else None
        
        # Summarize findings
        discount_categories = ['discounts', 'savings', 'coupons', 'promotional_offers', 'loyalty_discounts', 'staff_discounts']
        total_discounts = sum(len(discount_details[category]) for category in discount_categories if category in discount_details and isinstance(discount_details[category], list))
        print(f"   ✅ Priority 2B: Extracted {total_discounts} detailed discount entries")
        if total_discount_amount > 0:
            print(f"   💰 Total discount amount calculated: ${total_discount_amount}")
        
        return discount_details
    
    def extract_contact_details(self, df, supplier_context=None):
        """
        Extract address and contact information using supplier-address context relationship.
        
        Args:
            df: DataFrame with receipt text and predictions
            supplier_context: Optional supplier context for enhanced address detection
            
        Returns:
            Dictionary with contact details
        """
        
        contact_details = {
            'addresses': [],
            'phone_numbers': [],
            'email_addresses': [],
            'websites': [],
            'fax_numbers': []
        }
        
        # Look in HEADER and FOOTER for contact information
        contact_lines = df[df['predicted_class'].isin(['HEADER', 'FOOTER'])]
        
        # Enhanced address extraction using supplier context
        address_lines = self.address_extractor.extract_contextual_addresses(contact_lines, supplier_context)
        
        # Process individual contact fields
        for _, row in contact_lines.iterrows():
            text = str(row['text']).strip()
            
            # Extract specific contact details
            phone = self._extract_phone_number(text)
            if phone:
                contact_details['phone_numbers'].append({
                    'phone_number': phone,
                    'raw_text': text,
                    'line_number': row['line_number'],
                    'confidence': row['confidence_score']
                })
            
            email = self._extract_email(text)
            if email:
                contact_details['email_addresses'].append({
                    'email': email,
                    'raw_text': text,
                    'line_number': row['line_number'],
                    'confidence': row['confidence_score']
                })
            
            website = self._extract_website(text)
            if website:
                contact_details['websites'].append({
                    'website': website,
                    'raw_text': text,
                    'line_number': row['line_number'],
                    'confidence': row['confidence_score']
                })
            
            fax = self._extract_fax_number(text)
            if fax:
                contact_details['fax_numbers'].append({
                    'fax_number': fax,
                    'raw_text': text,
                    'line_number': row['line_number'],
                    'confidence': row['confidence_score']
                })
        
        # Process address blocks from contextual address extraction
        for address_block in address_lines:
            if address_block and len(address_block) > 0:
                # Combine address lines into full address
                full_address = ', '.join([component['text'] for component in address_block])
                
                # Calculate average confidence
                total_confidence = sum([component['confidence'] for component in address_block])
                average_confidence = total_confidence / len(address_block)
                
                # Format address data
                contact_details['addresses'].append({
                    'address': full_address,
                    'full_address': full_address,
                    'components': address_block,
                    'line_numbers': [component['line_number'] for component in address_block],
                    'confidence': average_confidence,
                    'extraction_method': 'supplier_context_aware'
                })
        
        return contact_details
    
    def _extract_phone_number(self, text):
        """
        Enhanced phone number extraction with OCR error handling and label prioritization.
        """
        
        # Try labeled patterns first (highest priority)
        for pattern in self.pattern_manager.phone_label_patterns:
            match = re.search(pattern, text)
            if match:
                phone_candidate = match.group(1).strip()
                validated_phone = self._validate_and_clean_phone(phone_candidate, text)
                if validated_phone:
                    return validated_phone
        
        # Only if no labeled phone found, try general patterns
        if not self._is_likely_barcode_or_id(text):
            for pattern in self.pattern_manager.general_phone_patterns:
                match = re.search(pattern, text)
                if match:
                    if len(match.groups()) == 1:
                        phone_candidate = match.group(1)
                    else:
                        phone_candidate = ''.join(match.groups())
                    
                    validated_phone = self._validate_and_clean_phone(phone_candidate, text)
                    if validated_phone:
                        return validated_phone
        
        return None
    
    def _validate_and_clean_phone(self, phone_candidate, original_text):
        """
        Validate and clean phone number candidate with OCR error correction.
        """
        if not phone_candidate:
            return None
        
        # Clean and normalize
        cleaned = re.sub(r'[^\d\+]', '', phone_candidate)
        
        # Apply OCR corrections for common digit/letter confusion
        ocr_corrections = {
            'O': '0', 'o': '0',
            'l': '1', 'I': '1',
            'S': '5', 's': '5',
            'B': '8', 'b': '8',
            'Z': '2', 'z': '2',
        }
        
        # Apply corrections to original candidate before cleaning
        corrected_candidate = phone_candidate
        for wrong, right in ocr_corrections.items():
            corrected_candidate = corrected_candidate.replace(wrong, right)
        
        # Re-clean after corrections
        cleaned = re.sub(r'[^\d\+]', '', corrected_candidate)
        
        # Validate length and format
        if cleaned.startswith('+44'):
            if len(cleaned) == 13:
                return self._format_phone_number(cleaned, 'international')
        elif cleaned.startswith('44') and len(cleaned) == 12:
            return self._format_phone_number('+' + cleaned, 'international')
        elif cleaned.startswith('0') and len(cleaned) == 11:
            if not self._is_likely_barcode_sequence(cleaned):
                return self._format_phone_number(cleaned, 'domestic')
        elif cleaned.startswith('07') and len(cleaned) == 11:
            if not self._is_likely_barcode_sequence(cleaned):
                return self._format_phone_number(cleaned, 'mobile')
        elif len(cleaned) >= 7 and len(cleaned) <= 10:
            if self._has_phone_label_context(original_text):
                if not self._is_likely_barcode_sequence(cleaned):
                    return self._format_phone_number(cleaned, 'short')
        
        return None
    
    def _is_likely_barcode_or_id(self, text):
        """
        Determine if text is likely a barcode or ID number rather than phone context.
        """
        text_lower = text.lower()
        for indicator in self.pattern_manager.barcode_patterns:
            if re.search(indicator, text, re.IGNORECASE):
                return True
        
        # Check if surrounded by obvious non-phone context
        if any(keyword in text_lower for keyword in ['transaction', 'receipt', 'barcode', 'ref', 'id', 'code']):
            return True
            
        return False
    
    def _is_likely_barcode_sequence(self, number_sequence):
        """
        Check if a number sequence is likely a barcode based on patterns.
        """
        if len(number_sequence) > 13:
            return True
        
        # Check for patterns typical of barcodes/IDs
        if re.search(r'(\d)\1{4,}', number_sequence):
            return True
        
        # All same digit
        if len(set(number_sequence)) == 1 and len(number_sequence) > 8:
            return True
            
        return False
    
    def _has_phone_label_context(self, text):
        """
        Check if the text has clear phone label context indicating this is likely a phone number.
        """
        phone_labels = [
            r'(?i)(?:tel|telephone|phone|ph|mob|mobile|fax)[\s:\-\.\,_]*',
            r'(?i)contact[\s:\-\.\,_]*',
            r'(?i)call[\s:\-\.\,_]*',
        ]
        
        for pattern in phone_labels:
            if re.search(pattern, text):
                return True
        return False
    
    def _format_phone_number(self, cleaned_number, phone_type):
        """
        Format phone number according to type for consistent display.
        """
        if phone_type == 'international':
            if len(cleaned_number) == 13:
                if cleaned_number[3:5] == '20':
                    return f"{cleaned_number[:3]} {cleaned_number[3:5]} {cleaned_number[5:9]} {cleaned_number[9:]}"
                else:
                    return f"{cleaned_number[:3]} {cleaned_number[3:6]} {cleaned_number[6:9]} {cleaned_number[9:]}"
        
        elif phone_type == 'domestic':
            if cleaned_number[1:3] == '20':
                return f"{cleaned_number[:3]} {cleaned_number[3:7]} {cleaned_number[7:]}"
            elif cleaned_number[1:4] in ['161', '121', '113', '114', '115', '116', '117', '118', '151', '191']:
                return f"{cleaned_number[:4]} {cleaned_number[4:7]} {cleaned_number[7:]}"
            elif len(cleaned_number) == 11 and cleaned_number[1:5].isdigit():
                return f"{cleaned_number[:5]} {cleaned_number[5:]}"
            else:
                return f"{cleaned_number[:4]} {cleaned_number[4:7]} {cleaned_number[7:]}"
                
        elif phone_type == 'short':
            if len(cleaned_number) == 9:
                return f"{cleaned_number[:3]} {cleaned_number[3:6]} {cleaned_number[6:]}"
            elif len(cleaned_number) == 8:
                return f"{cleaned_number[:4]} {cleaned_number[4:]}"
            elif len(cleaned_number) == 7:
                return f"{cleaned_number[:3]} {cleaned_number[3:]}"
            else:
                return f"{cleaned_number[:3]} {cleaned_number[3:6]} {cleaned_number[6:]}"
                
        elif phone_type == 'mobile':
            if len(cleaned_number) == 11:
                return f"{cleaned_number[:5]} {cleaned_number[5:8]} {cleaned_number[8:]}"
        
        return cleaned_number
    
    def _extract_email(self, text):
        """
        Extract email address from text with enhanced pattern recognition.
        """
        for pattern in self.pattern_manager.email_patterns:
            match = re.search(pattern, text)
            if match:
                email = match.group(1) if match.groups() else match.group(0)
                if '@' in email and '.' in email.split('@')[1]:
                    return email
        
        return None
    
    def _extract_website(self, text):
        """
        Extract website URL from text - only explicit websites, NOT from email addresses.
        """
        # First check if this line contains an email - if so, skip website extraction
        if '@' in text:
            return None
        
        # Explicit website patterns
        for pattern in self.pattern_manager.website_patterns[:2]:  # First 2 patterns are explicit
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        # Only check for bare domains if they appear in clear website context
        for pattern in self.pattern_manager.website_patterns[2:]:  # Remaining patterns
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                domain = match.group(1) if match.groups() else match.group(0)
                # Additional validation - exclude if it looks like part of an email context
                if not any(email_indicator in text.lower() for email_indicator in ['email', 'e-mail', '@']):
                    return domain
        
        return None
    
    def _extract_fax_number(self, text):
        """Extract fax number from text."""
        text_lower = text.lower()
        if 'fax' in text_lower:
            # Look for phone number patterns after 'fax'
            fax_part = text_lower[text_lower.find('fax'):]
            phone = self._extract_phone_number(fax_part)
            return phone
        return None
    
    def _extract_amount_from_text(self, text):
        """
        Extract monetary amount from text.
        Reuses the same logic from PaymentExtractor.
        """
        # This is a simplified version - in practice, you might want to import the full method
        if not text or len(text.strip()) < 1:
            return None
        
        amount_patterns = [
            r'[£$€¥₹₽¢]\s*(\d+\.\d{2})',
            r'(\d+\.\d{2})\s*[£$€¥₹₽¢]',
            r'(\d+\.\d{2})(?!\d)',
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    amount_str = match.group(1) if match.groups() else match.group(0)
                    amount_str = re.sub(r'[^\d.]', '', amount_str)
                    return float(amount_str)
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _extract_percentage_from_text(self, text):
        """Extract percentage from text."""
        percentage_pattern = r'(\d+(?:\.\d+)?)\s?%'
        match = re.search(percentage_pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None