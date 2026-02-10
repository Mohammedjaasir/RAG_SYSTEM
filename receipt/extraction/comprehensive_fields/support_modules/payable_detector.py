"""
Payable Detector - Detects payable amounts with enhanced multi-currency support
"""
import re
from typing import List, Optional, Dict, Any


class PayableDetector:
    """Detects payable amounts with enhanced multi-currency support."""
    
    def __init__(self):
        pass
    
    def _normalize_amount_enhanced(self, amount_str: str) -> Optional[float]:
        """
        Enhanced amount normalization with multi-format support.
        Handles: 1,234.56 (US), 1.234,56 (EU), 1 234,56 (FR), and mixed formats.
        """
        if not amount_str:
            return None
            
        amount_str = str(amount_str).strip()
        
        # Remove currency symbols first
        currency_symbols = [
            r'₹', r'Rs\.?', r'INR',
            r'£', r'GBP',
            r'\$', r'USD',
            r'€', r'EUR',
            r'¥', r'JPY', r'CNY',
            r'CA\$', r'CAD',
            r'A\$', r'AUD',
        ]
        
        # Remove currency symbols and extra spaces
        for symbol in currency_symbols:
            amount_str = re.sub(symbol, '', amount_str, flags=re.IGNORECASE)
        
        amount_str = amount_str.strip()
        
        if not amount_str:
            return None
        
        # Count dots and commas
        dot_count = amount_str.count('.')
        comma_count = amount_str.count(',')
        space_count = amount_str.count(' ')
        
        # Remove spaces (used as thousands separator in some locales)
        amount_str = amount_str.replace(' ', '')
        
        try:
            # Determine format and normalize
            if comma_count == 0 and dot_count <= 1:
                return float(amount_str)
            elif dot_count > 1 and comma_count == 1:
                amount_str = amount_str.replace('.', '').replace(',', '.')
            elif comma_count > 1 and dot_count == 1:
                amount_str = amount_str.replace(',', '')
            elif dot_count == 1 and comma_count == 1:
                dot_pos = amount_str.rindex('.')
                comma_pos = amount_str.rindex(',')
                if dot_pos > comma_pos:
                    amount_str = amount_str.replace(',', '')
                else:
                    amount_str = amount_str.replace('.', '').replace(',', '.')
            elif comma_count > 0 and dot_count == 0:
                if comma_count == 1 and len(amount_str.split(',')[1]) <= 2:
                    amount_str = amount_str.replace(',', '.')
                else:
                    amount_str = amount_str.replace(',', '')
            
            return float(amount_str)
        except (ValueError, IndexError):
            return None
    
    def _extract_amounts_from_line_enhanced(self, line: str) -> List[float]:
        """
        Extract all valid amounts from a line using enhanced multi-currency patterns.
        """
        # Enhanced currency symbols
        currency_symbols = [
            r'₹', r'Rs\.?', r'INR',
            r'£', r'GBP',
            r'\$', r'USD',
            r'€', r'EUR',
            r'¥', r'JPY', r'CNY',
            r'CA\$', r'CAD',
            r'A\$', r'AUD',
        ]
        
        # Build comprehensive amount regex with multi-currency support
        currency_pattern = '|'.join(currency_symbols)
        amount_regex = re.compile(
            rf'(?:(?:{currency_pattern})\s*)?'
            r'(\d{1,3}(?:[,.\s]\d{3})*(?:[.,]\d{1,2})?)'
        )
        
        amounts = amount_regex.findall(line)
        amounts_float = []
        
        for amt in amounts:
            amt_val = self._normalize_amount_enhanced(amt)
            if amt_val is not None and amt_val > 0:
                amounts_float.append(amt_val)
        
        return amounts_float
    
    def _extract_highest_amount_from_line_enhanced(self, line: str) -> Optional[float]:
        """
        Extract the highest amount from a line using enhanced detection.
        """
        amounts = self._extract_amounts_from_line_enhanced(line)
        return max(amounts) if amounts else None
    
    def find_payable_amount_enhanced(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced payable amount detection with multi-currency support.
        """
        # Enhanced payable keywords
        payable_keywords = [
            r'grand\s*total',
            r'net\s*total', 
            r'amount\s*payable',
            r'amount\s*due',
            r'balance\s*due',
            r'net\s*payable',
            r'total\s*amount',
            r'\btotal\b',
            r'sum\s*total',
            r'final\s*amount',
            r'total\s*to\s*pay',
            r'amount\s*owing'
        ]
        
        # Enhanced exclusion patterns
        exclude_keywords = [
            r'change',
            r'balance\s*returned',
            r'amount\s*returned', 
            r'refund',
            r'cash\s*back',
            r'change\s*due',
            r'balance\s*due\s*back',
            r'change\s*given',
            r'your\s*change'
        ]
        
        # Build exclusion pattern
        exclude_pattern = re.compile('|'.join(exclude_keywords), re.IGNORECASE)
        
        lines = text.splitlines() if isinstance(text, str) else text
        candidate_amounts = []
        matched_lines = []
        
        for line in lines:
            line_clean = line.strip() if isinstance(line, str) else str(line).strip()
            
            # Skip lines with exclusion keywords
            if exclude_pattern.search(line_clean):
                continue
            
            # Check for payable keywords
            for kw in payable_keywords:
                if re.search(kw, line_clean, re.IGNORECASE):
                    highest_amt = self._extract_highest_amount_from_line_enhanced(line_clean)
                    if highest_amt is not None:
                        candidate_amounts.append(highest_amt)
                        matched_lines.append({
                            'line': line_clean,
                            'amount': highest_amt,
                            'keyword': kw
                        })
                    break
        
        if candidate_amounts:
            # Return the highest matched value with metadata
            max_amount = max(candidate_amounts)
            best_match = next(match for match in matched_lines if match['amount'] == max_amount)
            
            return {
                'amount': max_amount,
                'raw_text': best_match['line'],
                'matched_keyword': best_match['keyword'],
                'confidence': 0.8,
                'extraction_method': 'enhanced_multi_currency_payable_detector',
                'currency_support': 'multi_currency',
                'total_candidates': len(candidate_amounts),
                'all_candidates': candidate_amounts
            }
        
        return None
    
    def _clean_amount_simple(self, amount_str: str) -> Optional[float]:
        """
        Clean amount string by removing currency symbols and thousands separators.
        """
        if not amount_str:
            return None
        
        # Remove common currency symbols and leading/trailing whitespace
        cleaned = re.sub(r'[€$£]', '', amount_str).strip()
        
        # Handle potential European/other separators:
        # Remove all commas/spaces used as thousands separators
        cleaned = re.sub(r'[, ]', '', cleaned)
        
        try:
            # Convert to float
            return float(cleaned)
        except ValueError:
            return None
    
    def detect_subtotal_simple_priority(self, receipt_text: str) -> Optional[Dict[str, Any]]:
        """
        Detect subtotal using simple priority-based approach.
        """
        if not receipt_text:
            return None
        
        # Normalize the text to handle case-insensitivity
        text = receipt_text.upper().strip()
        
        # Enhanced amount pattern with flexible currency and separators
        amount_pattern = r'[\s]*([€$£]?[0-9]{1,3}(?:[,\s]?[0-9]{3})*(?:\.[0-9]{1,2})?)'
        
        # Prioritized patterns for subtotal detection
        subtotal_patterns = {
            # 1. Exact Subtotal Match (Highest Priority)
            "SUBTOTAL": {
                'pattern': r"SUB\s?TOTAL[^\n]*?" + amount_pattern,
                'confidence': 0.95,
                'description': 'Exact subtotal match'
            },
            
            # 2. Pre-Tax/Net Match (Slightly lower priority)  
            "PRE-TOTAL / NET TOTAL": {
                'pattern': r"(?:PRE\s?TOTAL|NET\s?TOTAL|TOTAL\s?BEFORE\s?TAX)[^\n]*?" + amount_pattern,
                'confidence': 0.85,
                'description': 'Pre-tax or net total'
            },
            
            # 3. Final Total Match (Only used if no subtotal found)
            "TOTAL": {
                'pattern': r"(?<!GRAND\s)(?<!FINAL\s)TOTAL[^\n]*?" + amount_pattern,
                'confidence': 0.65,
                'description': 'Generic total (may be subtotal in simple receipts)'
            }
        }
        
        # Try patterns in priority order
        for source, pattern_info in subtotal_patterns.items():
            pattern = pattern_info['pattern']
            confidence = pattern_info['confidence']
            description = pattern_info['description']
            
            # Use re.DOTALL to allow search across lines if necessary
            match = re.search(pattern, text, re.DOTALL)
            if match:
                # The captured group is the amount string
                amount_str = match.group(1).strip()
                subtotal_value = self._clean_amount_simple(amount_str)
                
                if subtotal_value is not None and subtotal_value > 0:
                    print(f"   🎯 Simple priority match: {source} -> £{subtotal_value}")
                    
                    return {
                        'amount': subtotal_value,
                        'source': source,
                        'pattern_description': description,
                        'raw_text': match.group(0).strip(),
                        'confidence': confidence,
                        'line_number': None,
                        'amount_string': amount_str
                    }
        
        return None
    
    def extract_amounts_from_line(self, line: str, pattern: str) -> List[float]:
        """
        Extract currency amounts from a line using the given regex pattern.
        """
        matches = re.findall(pattern, line)
        amounts = []
        for amt in matches:
            try:
                val = float(amt.replace(',', '').strip())
                if val > 0:
                    amounts.append(val)
            except ValueError:
                continue
        return amounts
    
    def extract_net_after_discount_standalone(self, text: str, payment_keywords: List[str], 
                                            net_total_keywords: List[str], pattern: str) -> Optional[float]:
        """
        Extract the net after discount amount from the receipt text using simple priority logic.
        """
        lines = text.splitlines()

        # First attempt: payment keywords
        for line in lines:
            lower_line = line.lower()
            if any(kw in lower_line for kw in payment_keywords):
                amounts = self.extract_amounts_from_line(line, pattern)
                if amounts:
                    return round(max(amounts), 2)

        # Fallback: net total keywords
        for line in lines:
            lower_line = line.lower()
            if any(kw in lower_line for kw in net_total_keywords):
                amounts = self.extract_amounts_from_line(line, pattern)
                if amounts:
                    return round(max(amounts), 2)

        # Not found
        return None
    
    def extract_net_after_discount_simple_priority(self, df) -> Optional[Dict[str, Any]]:
        """
        Extract net after discount using simple priority-based approach.
        """
        print("🎯 Starting simple priority-based net after discount detection...")
        
        # Convert DataFrame to text
        text_column = 'cleaned_text' if 'cleaned_text' in df.columns else ('text' if 'text' in df.columns else 'original_text')
        if text_column not in df.columns:
            print("❌ No suitable text column found for simple priority detection")
            return None
            
        all_text = '\n'.join(df[text_column].fillna('').astype(str))
        
        # Enhanced multi-currency pattern
        pattern = r'(?:₹|Rs\.?|INR|GBP|GB|£|\$|€|¥|CA\$|A\$)?\s*(\d{1,3}(?:[,\s]\d{3})*(?:\.\d{1,2})?)'
        
        lines = all_text.splitlines()
        
        print(f"   🔍 Scanning {len(lines)} lines with priority keywords...")
        
        # Priority keywords
        priority_keywords = [
            'net after discount', 'total after discount', 'amount after discount',
            'final amount', 'amount due', 'balance due', 'amount payable'
        ]
        secondary_keywords = ['payment', 'visa', 'mastercard', 'card sale', 'total due']
        
        # Priority 1: Context-aware primary keywords
        for line_idx, line in enumerate(lines):
            lower_line = line.lower()
            for keyword_idx, keyword in enumerate(priority_keywords):
                if keyword in lower_line:
                    print(f"      🎯 Found priority keyword '{keyword}' in line {line_idx + 1}: {line.strip()}")
                    amounts = self.extract_amounts_from_line(line, pattern)
                    if amounts:
                        max_amount = max(amounts)
                        print(f"         💰 Extracted amount: £{max_amount}")
                        
                        return {
                            'amount': round(max_amount, 2),
                            'raw_text': line.strip(),
                            'confidence': 0.95 - (keyword_idx * 0.05),
                            'extraction_method': 'simple_priority_context_aware',
                            'matched_keywords': [keyword],
                            'priority_level': 1,
                            'line_number': line_idx + 1
                        }
        
        print("   ⚠️ No priority keyword matches found, trying secondary keywords...")
        
        # Priority 2: Secondary keywords
        for line_idx, line in enumerate(lines):
            lower_line = line.lower()
            for keyword_idx, keyword in enumerate(secondary_keywords):
                if keyword in lower_line:
                    print(f"      🎯 Found secondary keyword '{keyword}' in line {line_idx + 1}: {line.strip()}")
                    amounts = self.extract_amounts_from_line(line, pattern)
                    if amounts:
                        max_amount = max(amounts)
                        print(f"         💰 Extracted amount: £{max_amount}")
                        
                        return {
                            'amount': round(max_amount, 2),
                            'raw_text': line.strip(),
                            'confidence': 0.8 - (keyword_idx * 0.05),
                            'extraction_method': 'simple_priority_secondary',
                            'matched_keywords': [keyword],
                            'priority_level': 2,
                            'line_number': line_idx + 1
                        }
        
        print("   ❌ Simple priority detection found no matching keywords")
        return None