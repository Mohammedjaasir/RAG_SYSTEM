"""
Pattern Matcher - Handles regex pattern matching for extraction
"""
import re
from typing import List, Optional, Match, Pattern, Any


class PatternMatcher:
    """Handles regex pattern matching operations for receipt extraction."""
    
    def __init__(self):
        self.cache = {}  # Cache compiled patterns for performance
    
    def compile_pattern(self, pattern_str: str, flags: int = re.IGNORECASE) -> Pattern:
        """Compile and cache regex pattern."""
        cache_key = f"{pattern_str}_{flags}"
        if cache_key not in self.cache:
            self.cache[cache_key] = re.compile(pattern_str, flags)
        return self.cache[cache_key]
    
    def find_all_matches(self, text: str, pattern_str: str, flags: int = re.IGNORECASE) -> List[Match]:
        """Find all matches for a pattern in text."""
        pattern = self.compile_pattern(pattern_str, flags)
        return list(pattern.finditer(text))
    
    def search_pattern(self, text: str, pattern_str: str, flags: int = re.IGNORECASE) -> Optional[Match]:
        """Search for pattern in text (first match)."""
        pattern = self.compile_pattern(pattern_str, flags)
        return pattern.search(text)
    
    def match_pattern(self, text: str, pattern_str: str, flags: int = re.IGNORECASE) -> Optional[Match]:
        """Match pattern at beginning of text."""
        pattern = self.compile_pattern(pattern_str, flags)
        return pattern.match(text)
    
    def findall(self, text: str, pattern_str: str, flags: int = re.IGNORECASE) -> List[str]:
        """Find all non-overlapping matches."""
        pattern = self.compile_pattern(pattern_str, flags)
        return pattern.findall(text)
    
    def extract_amounts_from_line(self, line: str, pattern_str: str) -> List[float]:
        """
        Extract currency amounts from a line using the given regex pattern.
        From original function 53.
        """
        pattern = self.compile_pattern(pattern_str)
        matches = self.findall(line, pattern_str)
        
        amounts = []
        for amt in matches:
            try:
                # Clean the amount string
                cleaned_amt = amt.replace(',', '').strip()
                val = float(cleaned_amt)
                if val > 0:
                    amounts.append(val)
            except (ValueError, TypeError):
                continue
        return amounts
    
    def extract_with_groups(self, text: str, pattern_str: str, group_names: List[str] = None, 
                           flags: int = re.IGNORECASE) -> Optional[dict]:
        """
        Extract pattern with named groups or indexed groups.
        
        Args:
            text: Text to search
            pattern_str: Regex pattern string
            group_names: Optional list of group names
            flags: Regex flags
            
        Returns:
            Dictionary with group names/indices as keys and matched values as values
        """
        match = self.search_pattern(text, pattern_str, flags)
        if not match:
            return None
        
        result = {}
        if group_names and match.groups():
            # Use provided group names
            for i, group_name in enumerate(group_names):
                if i < len(match.groups()):
                    result[group_name] = match.group(i + 1)
        elif match.groupdict():
            # Use named groups from pattern
            result = match.groupdict()
        else:
            # Use indexed groups
            for i in range(len(match.groups())):
                result[f'group_{i+1}'] = match.group(i + 1)
        
        return result
    
    def multi_pattern_search(self, text: str, pattern_list: List[str], flags: int = re.IGNORECASE) -> List[dict]:
        """
        Search for multiple patterns and return all matches.
        
        Args:
            text: Text to search
            pattern_list: List of regex pattern strings
            flags: Regex flags
            
        Returns:
            List of match dictionaries with pattern and matches
        """
        results = []
        for pattern_str in pattern_list:
            matches = self.find_all_matches(text, pattern_str, flags)
            if matches:
                results.append({
                    'pattern': pattern_str,
                    'matches': [m.group() for m in matches],
                    'full_matches': matches
                })
        return results
    
    def validate_pattern_match(self, text: str, pattern_str: str, min_length: int = 1, 
                              max_length: int = 100, flags: int = re.IGNORECASE) -> Optional[str]:
        """
        Validate pattern match meets length criteria.
        
        Args:
            text: Text to search
            pattern_str: Regex pattern string
            min_length: Minimum match length
            max_length: Maximum match length
            flags: Regex flags
            
        Returns:
            Validated match string or None
        """
        match = self.search_pattern(text, pattern_str, flags)
        if not match:
            return None
        
        matched_text = match.group()
        if min_length <= len(matched_text) <= max_length:
            return matched_text
        return None
    
    def extract_subtotal_amounts(self, line: str) -> List[float]:
        """
        Specialized method for extracting subtotal amounts.
        Handles various subtotal formats and OCR artifacts.
        """
        # Enhanced amount pattern with flexible currency and separators
        amount_pattern = r'[\s]*([€$£]?[0-9]{1,3}(?:[,\s]?[0-9]{3})*(?:\.[0-9]{1,2})?)'
        
        # Subtotal-specific patterns
        subtotal_patterns = [
            r"SUB\s?TOTAL[^\n]*?" + amount_pattern,
            r"(?:PRE\s?TOTAL|NET\s?TOTAL|TOTAL\s?BEFORE\s?TAX)[^\n]*?" + amount_pattern,
            r"(?<!GRAND\s)(?<!FINAL\s)TOTAL[^\n]*?" + amount_pattern
        ]
        
        amounts = []
        for pattern in subtotal_patterns:
            match = self.search_pattern(line, pattern, re.IGNORECASE | re.DOTALL)
            if match and match.groups():
                amount_str = match.group(1).strip()
                try:
                    # Clean and convert amount
                    cleaned = re.sub(r'[€$£\s,]', '', amount_str)
                    amount = float(cleaned)
                    if amount > 0:
                        amounts.append(amount)
                except (ValueError, TypeError):
                    continue
        
        return amounts
    
    def extract_total_amounts(self, line: str) -> List[float]:
        """
        Specialized method for extracting total amounts.
        Handles various total formats.
        """
        # Enhanced amount pattern
        amount_pattern = r'[\s]*([€$£]?[0-9]{1,3}(?:[,\s]?[0-9]{3})*(?:\.[0-9]{1,2})?)'
        
        # Total-specific patterns
        total_patterns = [
            r"GRAND\s?TOTAL[^\n]*?" + amount_pattern,
            r"FINAL\s?TOTAL[^\n]*?" + amount_pattern,
            r"AMOUNT\s?DUE[^\n]*?" + amount_pattern,
            r"BALANCE\s?DUE[^\n]*?" + amount_pattern,
            r"TOTAL\s?PAYABLE[^\n]*?" + amount_pattern
        ]
        
        amounts = []
        for pattern in total_patterns:
            match = self.search_pattern(line, pattern, re.IGNORECASE | re.DOTALL)
            if match and match.groups():
                amount_str = match.group(1).strip()
                try:
                    # Clean and convert amount
                    cleaned = re.sub(r'[€$£\s,]', '', amount_str)
                    amount = float(cleaned)
                    if amount > 0:
                        amounts.append(amount)
                except (ValueError, TypeError):
                    continue
        
        return amounts