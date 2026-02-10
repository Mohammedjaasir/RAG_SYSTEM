#!/usr/bin/env python3
"""
Change Extractor for Payment Extraction v2.7.1
Specializes in change and refund amount extraction
"""

import re


class ChangeExtractor:
    """
    Detects and extracts change, refund, and related payment adjustments.
    Supports multi-language detection with proximity-based search strategies.
    """
    
    def __init__(self, pattern_manager=None, amount_extractor=None):
        """Initialize change extractor with dependencies."""
        self.pattern_manager = pattern_manager
        self.amount_extractor = amount_extractor
        print(f"✅ Initialized Change Extractor v2.7.1")
    
    def detect_change_simple(self, text):
        """
        Simple change detection with multi-language support.
        Excludes non-monetary contexts like "oil change" or "change room".
        
        Args:
            text: Text string to analyze
            
        Returns:
            Boolean indicating if this is change-related text
        """
        if not text:
            return False
            
        text_lower = text.lower()
        
        # Exclude non-monetary change contexts
        exclusion_patterns = [
            'spare change', 'loose change', 'small change',
            'change jar', 'change purse', 'change wallet',
            'exchange rate', 'currency exchange', 'rate change',
            'oil change', 'tire change', 'address change',
            'climate change', 'change room', 'changing room'
        ]
        
        # If it matches an exclusion pattern, not a payment change
        for exclusion in exclusion_patterns:
            if exclusion in text_lower:
                return False
        
        # Get all change keywords from all languages
        all_keywords = self.pattern_manager.get_all_change_keywords() if self.pattern_manager else []
        
        # Check for keyword match
        has_keyword = any(keyword in text_lower for keyword in all_keywords)
        
        if has_keyword:
            # Additional validation: require either amount context or payment context
            has_amount_context = bool(re.search(r'[£$€¥₹]\s*\d+|\d+[.,]\d{2}|\d+\s*[£$€¥₹]', text))
            
            payment_context_terms = ['due', 'given', 'tendered', 'paid', 'received', ':', 'total']
            has_payment_context = any(term in text_lower for term in payment_context_terms)
            
            return has_amount_context or has_payment_context
        
        return False
    
    def extract_change_amount_proximity(self, df):
        """
        Enhanced change amount extraction using proximity search.
        Uses same-line/next-line strategy and fallback pattern search.
        
        Args:
            df: DataFrame with receipt lines
            
        Returns:
            List of extracted change entries (max 1)
        """
        change_details = []
        
        # Convert DataFrame to line list for proximity search
        lines = []
        for _, row in df.iterrows():
            lines.append({
                'text': str(row['text']).strip(),
                'line_number': row.get('line_number', 0),
                'confidence': row.get('confidence_score', 0.7),
                'predicted_class': row.get('predicted_class', ''),
                'row_index': row.name
            })
        
        # Enhanced keywords (multi-language)
        all_change_keywords = self.pattern_manager.get_all_change_keywords() if self.pattern_manager else []
        
        matches = []
        
        # Strategy 1: Same line → Next line search
        for i, line_data in enumerate(lines):
            text = line_data['text']
            text_lower = text.lower()
            
            for keyword in all_change_keywords:
                if keyword in text_lower:
                    # First check for product exclusions
                    exclusion_patterns = [
                        'spare change', 'loose change', 'small change',
                        'change jar', 'change purse', 'change wallet',
                        'exchange rate', 'currency exchange'
                    ]
                    
                    has_exclusion = any(exclusion in text_lower for exclusion in exclusion_patterns)
                    if has_exclusion:
                        continue
                    
                    # Check same line first
                    if self.amount_extractor:
                        amount = self.amount_extractor.extract_amount_from_text(text)
                    else:
                        amount = None
                    
                    if amount:
                        change_entry = self._create_change_entry_enhanced(line_data, amount, keyword, 'same_line')
                        matches.append(change_entry)
                        break
                    
                    # Check next line
                    if i + 1 < len(lines):
                        next_text = lines[i + 1]['text']
                        if self.amount_extractor:
                            amount = self.amount_extractor.extract_amount_from_text(next_text)
                        else:
                            amount = None
                        
                        if amount:
                            change_entry = self._create_change_entry_enhanced(lines[i + 1], amount, keyword, 'next_line')
                            matches.append(change_entry)
                            break
        
        # Strategy 2: Fallback pattern search for complex layouts
        if not matches:
            full_text = '\n'.join(line['text'] for line in lines)
            for keyword in all_change_keywords:
                # Check for exclusions before pattern search
                exclusion_patterns = [
                    'spare change', 'loose change', 'small change',
                    'change jar', 'change purse', 'change wallet',
                    'exchange rate', 'currency exchange'
                ]
                
                text_lower = full_text.lower()
                has_exclusion = any(exclusion in text_lower for exclusion in exclusion_patterns)
                if has_exclusion:
                    continue
                
                if self.amount_extractor:
                    amount_pattern = self.amount_extractor.build_simple_amount_pattern()
                else:
                    amount_pattern = r'[\$£€¥₹]\s*\d+|\d+[.,]\d{2}'
                
                pattern = rf'{re.escape(keyword)}[:\s]*.*?({amount_pattern})'
                found = re.search(pattern, full_text, re.IGNORECASE)
                if found:
                    amount_str = re.findall(amount_pattern, found.group())
                    if amount_str:
                        if self.amount_extractor:
                            amount = self.amount_extractor.normalize_simple_amount(amount_str[-1])
                            amount = float(amount) if amount else None
                        else:
                            amount = None
                        
                        if amount:
                            # Find the line containing this match
                            for line_data in lines:
                                if keyword in line_data['text'].lower() or amount_str[-1] in line_data['text']:
                                    change_entry = self._create_change_entry_enhanced(line_data, amount, keyword, 'fallback_pattern')
                                    matches.append(change_entry)
                                    break
                            break
        
        return matches[-1:] if matches else []
    
    def _create_change_entry_enhanced(self, line_data, amount, keyword, detection_method):
        """
        Create enhanced change entry with detection metadata.
        Classifies change type and includes detection information.
        
        Args:
            line_data: Line data dictionary
            amount: Extracted amount value
            keyword: Detected keyword that triggered change detection
            detection_method: Method used to detect change (same_line, next_line, etc)
            
        Returns:
            Dictionary with change entry information
        """
        text = line_data['text']
        text_lower = text.lower()
        
        # Determine change type based on text content
        if re.search(r'no\s*change\s*due', text_lower):
            amount = 0.0
            change_type = 'no_change_due'
        elif any(term in text_lower for term in ['due', 'owed', 'owing']):
            change_type = 'change_due'
        elif any(term in text_lower for term in ['given', 'returned', 'back']):
            change_type = 'change_given'
        elif any(term in text_lower for term in ['refund', 'reembolso', 'remboursement', 'rückerstattung', 'rimborso']):
            change_type = 'refund'
        else:
            change_type = 'change_given'
        
        return {
            'raw_text': text,
            'amount': amount,
            'line_number': int(line_data['line_number']),
            'confidence': float(line_data['confidence']),
            'change_type': change_type,
            'detection_keyword': keyword,
            'detection_method': detection_method,
            'multi_language_enhanced': True
        }
