"""
Text Cleaner - Handles OCR text cleaning and normalization
"""
import re


class TextCleaner:
    """Cleans and normalizes text for extraction."""
    
    def clean_text_line_for_ocr(self, line):
        """
        Enhanced OCR text cleaning with unicode normalization.
        Normalize OCR artifacts: replace unicode spaces/dashes and lowercase.
        """
        if not line:
            return ""
        
        # Replace all dash variants with standard dash
        line = re.sub(r"[\u2010-\u2015\u2212‐‑‒–—−]", "-", line)
        
        # Normalize spacing
        line = re.sub(r"\s+", " ", line)
        
        return line.strip()
    
    def clean_supplier_name(self, supplier_name):
        """
        Clean supplier name by removing date/time patterns and address contamination.
        """
        if not supplier_name or not isinstance(supplier_name, str):
            return supplier_name
        
        original = supplier_name.strip()
        cleaned = original
        
        # Remove common date/time contamination patterns
        datetime_patterns = [
            r'\b(MON|TUE|WED|THU|FRI|SAT|SUN)\s+\d{1,2}.*$',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}.*$',
            r'\b\d{2}:\d{2}.*$',
            r'\b\d{1,2}\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC).*$',
        ]
        
        for pattern in datetime_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        
        # Remove address contamination from end
        address_suffixes = [
            r'\s+\d+\s+[A-Z][a-z]+\s+(Road|Street|Lane|Avenue|Drive|Way|Close).*$',
            r'\s+[A-Z]{2}\d+\s*\d*[A-Z]*$',
            r'\s+(Nr\.?|Near)\s+[A-Z].*$',
            r'\s+Tel:?.*$',
            r'\s+\d{5}.*$',
        ]
        
        for pattern in address_suffixes:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        
        # Remove transaction/system info
        system_patterns = [
            r'\s+Trans\s+\d+.*$',
            r'\s+Store\s+\d+.*$',
            r'\s+POS\s+\d+.*$',
            r'\s+Op\s+Name.*$',
        ]
        
        for pattern in system_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
        
        # Clean up multiple spaces and trailing punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = re.sub(r'[,.\-\s]+$', '', cleaned).strip()
        
        # Only return cleaned version if it's substantially the same
        if len(cleaned) >= len(original) * 0.5 and cleaned.strip():
            return cleaned
        else:
            return original
    
    def extract_supplier_name_from_line(self, line: str) -> str:
        """
        Extract only supplier name from a line using enhanced text processing.
        """
        words = re.findall(r"[A-Za-z'&]+", line)
        supplier_words = []

        for word in words:
            if word.islower() and supplier_words:
                break
            if word[0].isupper() or "'" in word or "&" in word:
                supplier_words.append(word)
            else:
                if supplier_words:
                    break

        return " ".join(supplier_words) if supplier_words else line