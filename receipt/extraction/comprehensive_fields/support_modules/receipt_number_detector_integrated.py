"""
Integrated Receipt Number Detector - Dynamic receipt number detection with configurable keywords
"""
import re
from typing import List


class IntegratedReceiptNumberDetector:
    """Integrated receipt number detector with configurable keywords and patterns."""
    
    def __init__(self, keywords: List[str], number_pattern: str):
        if not keywords or not number_pattern:
            raise ValueError("Both keywords and number_pattern must be provided and non-empty.")
        
        self.keywords = keywords
        self.number_pattern = number_pattern
        
        # Build dynamic pattern from configurable keywords
        keywords_regex = "|".join([re.escape(k) for k in self.keywords])
        pattern = rf"(?P<keyword>{keywords_regex})[:\s#]*?(?P<number>{self.number_pattern})"
        self.compiled_pattern = re.compile(pattern, re.IGNORECASE)
        
        # Enhanced fallback pattern for alphanumeric receipt numbers
        self.fallback_pattern = re.compile(r"\b[A-Z0-9\-_#]{4,20}\b")

    def detect_receipt_number(self, text: str) -> str:
        """
        Enhanced two-tier receipt number detection with keyword prioritization.
        Tier 1: Keyword + number pattern matching
        Tier 2: Fallback - alphanumeric sequences on keyword lines or adjacent lines
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        # Tier 1: Direct keyword + number pattern matching
        for line in lines:
            match = self.compiled_pattern.search(line)
            if match:
                number = match.group("number").strip()
                # Validate the extracted number is reasonable
                if len(number) >= 3 and not number.isspace():
                    return number
        
        # Tier 2: Enhanced fallback - find alphanumeric sequences on keyword lines
        for i, line in enumerate(lines):
            if any(keyword.lower() in line.lower() for keyword in self.keywords):
                # Check current line for fallback pattern
                fallback_match = self.fallback_pattern.search(line)
                if fallback_match:
                    number = fallback_match.group(0).strip()
                    # Additional validation for fallback results
                    if len(number) >= 4 and not number.isspace():
                        return number
                
                # Check next line if keyword line doesn't have a number
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    next_match = self.fallback_pattern.search(next_line)
                    if next_match:
                        number = next_match.group(0).strip()
                        # Validation for next line results
                        if len(number) >= 4 and not number.isspace():
                            return number
                
                # Check previous line if keyword line doesn't have a number
                if i - 1 >= 0:
                    prev_line = lines[i - 1]
                    prev_match = self.fallback_pattern.search(prev_line)
                    if prev_match:
                        number = prev_match.group(0).strip()
                        # Validation for previous line results
                        if len(number) >= 4 and not number.isspace():
                            return number
        
        return "Not found"