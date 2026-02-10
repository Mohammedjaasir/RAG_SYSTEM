#!/usr/bin/env python3
"""
Amount Extractor for Payment Extraction v2.7.1
Specializes in monetary amount extraction with multi-currency support
"""

import re


class AmountExtractor:
    """
    Extracts monetary amounts from receipt text with multi-currency support.
    Handles various decimal separators, thousands separators, and OCR errors.
    """
    
    def __init__(self, pattern_manager=None):
        """Initialize amount extractor with pattern manager."""
        self.pattern_manager = pattern_manager
        print(f"✅ Initialized Amount Extractor v2.7.1")
    
    def extract_amount_from_text(self, text):
        """
        Enhanced monetary amount extraction with multi-currency support.
        Handles 14+ distinct patterns with OCR error correction.
        
        Args:
            text: Text string to extract amount from
            
        Returns:
            Float amount or None if not found
        """
        if not text or len(text.strip()) < 1:
            return None
        
        # Enhanced currency patterns with OCR error handling
        amount_patterns = [
            # Currency symbol before amount (most common)
            r'[£$€¥₹₽¢]\s*(\d{1,3}(?:,\d{3})*\.\d{2})',  # £1,234.56
            r'[£$€¥₹₽¢]\s*(\d+\.\d{2})',  # £50.02, $10.99
            r'[£$€¥₹₽¢]\s*(\d+)',  # £50 (whole amounts)
            
            # Currency symbol after amount
            r'(\d{1,3}(?:,\d{3})*\.\d{2})\s*[£$€¥₹₽¢]',  # 1,234.56£
            r'(\d+\.\d{2})\s*[£$€¥₹₽¢]',  # 50.02£
            r'(\d+)\s*[£$€¥₹₽¢]',  # 50£
            
            # Currency codes (ISO format)
            r'(GBP|USD|EUR|JPY|INR|CAD|AUD)\s*(\d{1,3}(?:,\d{3})*\.\d{2})',
            r'(\d{1,3}(?:,\d{3})*\.\d{2})\s*(GBP|USD|EUR|JPY|INR|CAD|AUD)',
            
            # Amounts with thousands separators
            r'(\d{1,3}(?:,\d{3})+\.\d{2})(?!\d)',
            r'(\d{1,3}(?:\.\d{3})+,\d{2})(?!\d)',
            r'(\d{1,3}(?:\s\d{3})+\.\d{2})(?!\d)',
            
            # Standard decimal amounts
            r'(\d+\.\d{2})(?!\d)',
            r'(\d+\.\d{1})(?!\d)',
            
            # Whole number amounts in payment contexts
            r'(\d{2,})(?=\s*(?:paid|total|amount|charge|fee|payment))',
            
            # OCR error corrections
            r'[£$€¥₹₽¢]\s*([O0]\d*\.\d{2})',
            r'[£$€¥₹₽¢]\s*(\d*[O0]\d*\.\d{2})',
            
            # Negative amounts
            r'-\s*[£$€¥₹₽¢]\s*(\d+\.\d{2})',
            r'[£$€¥₹₽¢]\s*(\d+\.\d{2})\s*-',
        ]
        
        best_amount = None
        best_confidence = 0
        
        for i, pattern in enumerate(amount_patterns):
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in matches:
                    try:
                        # Handle tuple matches
                        if isinstance(match, tuple):
                            amount_str = None
                            for part in match:
                                if re.match(r'\d', part):
                                    amount_str = part
                                    break
                            if not amount_str:
                                continue
                        else:
                            amount_str = match
                        
                        # Clean up the amount string
                        amount_str = amount_str.replace('O', '0').replace('o', '0')
                        
                        # Handle different decimal separators
                        if ',' in amount_str and '.' in amount_str:
                            last_comma = amount_str.rfind(',')
                            last_dot = amount_str.rfind('.')
                            if last_dot > last_comma:
                                amount_str = amount_str.replace(',', '')
                            else:
                                amount_str = amount_str.replace('.', '').replace(',', '.')
                        elif ',' in amount_str:
                            parts = amount_str.split(',')
                            if len(parts) == 2 and len(parts[1]) == 2:
                                amount_str = amount_str.replace(',', '.')
                            else:
                                amount_str = amount_str.replace(',', '')
                        
                        # Remove any remaining non-numeric characters except decimal point
                        amount_str = re.sub(r'[^\d.]', '', amount_str)
                        
                        if amount_str and '.' in amount_str:
                            amount = float(amount_str)
                        elif amount_str:
                            amount = float(amount_str)
                        else:
                            continue
                        
                        # Validate amount (reasonable range for most transactions)
                        if 0.01 <= amount <= 999999.99:
                            # Calculate confidence based on pattern priority and format
                            confidence = 1.0 - (i * 0.05)
                            
                            # Boost confidence for amounts with currency symbols
                            if i < 6:
                                confidence += 0.1
                            
                            # Boost confidence for proper decimal formatting
                            if '.' in amount_str and len(amount_str.split('.')[1]) == 2:
                                confidence += 0.05
                            
                            if confidence > best_confidence:
                                best_amount = amount
                                best_confidence = confidence
                        
                    except (ValueError, TypeError, IndexError):
                        continue
        
        return best_amount
    
    def extract_percentage_from_text(self, text):
        """
        Extract percentage value from text.
        
        Args:
            text: Text string to extract percentage from
            
        Returns:
            Float percentage or None if not found
        """
        percentage_pattern = r'(\d+(?:\.\d+)?)\s?%'
        match = re.search(percentage_pattern, text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None
    
    def clean_text_line(self, line):
        """
        Normalize OCR artifacts in a text line.
        
        Args:
            line: Text line to clean
            
        Returns:
            Cleaned text line
        """
        line = re.sub(r"[\u2010-\u2015\u2212‐‑‒–—−]", "-", line)
        line = re.sub(r"\s+", " ", line)
        return line.strip()
    
    def build_simple_amount_pattern(self):
        """
        Build regex pattern for amounts with various currencies.
        
        Returns:
            Regex pattern string for amount matching
        """
        currencies = r'[\$£€¥₹]|USD|EUR|GBP|Rs'
        number = r'\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d{1,2})?'
        return rf'(?:{currencies})?\s*{number}'
    
    def normalize_simple_amount(self, amount_str):
        """
        Convert extracted amount to standard format.
        Handles both US and European decimal formats.
        
        Args:
            amount_str: Amount string to normalize
            
        Returns:
            Normalized amount string
        """
        # Remove currency symbols
        amount_str = re.sub(r'[\$£€¥₹]|USD|EUR|GBP|Rs', '', amount_str)
        amount_str = amount_str.strip()
        
        # Handle different formats
        if re.search(r',\d{2}$', amount_str):
            # European format: 1.234,56 -> 1234.56
            amount_str = amount_str.replace('.', '').replace(' ', '').replace(',', '.')
        else:
            # US format: 1,234.56 -> 1234.56
            amount_str = amount_str.replace(',', '').replace(' ', '')
        
        return amount_str
    
    def extract_card_amount_enhanced(self, text_lines):
        """
        Extract card payment amount using simplified approach.
        Uses same-line and next-line search strategies.
        
        Args:
            text_lines: List of text lines or single string with newlines
            
        Returns:
            Float amount or None if not found
        """
        if isinstance(text_lines, str):
            text_lines = text_lines.splitlines()
            
        lines = [self.clean_text_line(ln) for ln in text_lines if ln.strip()]
        card_keywords = self.pattern_manager.get_all_card_keywords() if self.pattern_manager else []
        amount_pattern = self.build_simple_amount_pattern()
        
        # Strategy 1: Search line by line (same line → next line)
        for i, line in enumerate(lines):
            lower_line = line.lower()
            
            if any(keyword in lower_line for keyword in card_keywords):
                # Check same line first
                same_line = re.findall(amount_pattern, line)
                if same_line:
                    try:
                        normalized = self.normalize_simple_amount(same_line[-1])
                        return float(normalized) if normalized else None
                    except (ValueError, TypeError):
                        pass
                
                # Check next line
                if i + 1 < len(lines):
                    next_line = re.findall(amount_pattern, lines[i + 1])
                    if next_line:
                        try:
                            normalized = self.normalize_simple_amount(next_line[-1])
                            return float(normalized) if normalized else None
                        except (ValueError, TypeError):
                            pass
        
        # Strategy 2: Fallback merged layout search
        full_text = ' '.join(lines)
        card_pattern = '|'.join(re.escape(k) for k in card_keywords) if card_keywords else ''
        if card_pattern:
            combined = rf'(?:{card_pattern})[^\d]*({amount_pattern})'
            found = re.search(combined, full_text, re.IGNORECASE | re.DOTALL)
            
            if found:
                amt = re.findall(amount_pattern, found.group())
                if amt:
                    try:
                        normalized = self.normalize_simple_amount(amt[-1])
                        return float(normalized) if normalized else None
                    except (ValueError, TypeError):
                        pass
        
        return None
    
    def extract_card_amount_from_line(self, text):
        """
        Extract card amount from a single line using enhanced detection.
        Falls back to general amount extraction if card-specific fails.
        
        Args:
            text: Single text line
            
        Returns:
            Float amount or None if not found
        """
        if not text or len(text.strip()) < 3:
            return None
            
        # First try the simple card-specific approach
        card_amount = self.extract_card_amount_enhanced([text])
        if card_amount:
            return card_amount
            
        # Fallback to existing complex amount extraction
        return self.extract_amount_from_text(text)
