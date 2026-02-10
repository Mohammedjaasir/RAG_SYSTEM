"""
VAT Extractor - Extracts VAT number from receipts
"""
import re
import pandas as pd
from typing import Optional
from ..shared_utils.config_manager import ConfigManager
from ..shared_utils.pattern_matcher import PatternMatcher
from ..models.extraction_models import ExtractionResult


class VATExtractor:
    """Extracts VAT number using enhanced patterns and ultra-dynamic detection."""
    
    def __init__(self, config_manager: ConfigManager, pattern_matcher: PatternMatcher):
        self.config_manager = config_manager
        self.pattern_matcher = pattern_matcher
        self.extraction_rules = config_manager.config.get('extraction_rules', {})
        self.vat_patterns = config_manager.get_patterns('vat_patterns')
    
    def extract_vat_number(self, df) -> Optional[ExtractionResult]:
        """
        Extract VAT number using enhanced patterns and ultra-dynamic detection.
        """
        # First try the enhanced ultra-dynamic detector on full text
        text_column = 'cleaned_text' if 'cleaned_text' in df.columns else ('text' if 'text' in df.columns else 'original_text')
        all_text = ' '.join(df[text_column].fillna('').astype(str))
        
        print(f"🏷️ Trying enhanced VAT detection on combined text...")
        enhanced_vat = self.extract_vat_number_from_text(all_text)
        
        if enhanced_vat:
            # Find the source line for better context
            best_line = None
            best_confidence = 0.0
            
            for idx, row in df.iterrows():
                line_text = str(row[text_column]).strip()
                if enhanced_vat.lower() in line_text.lower() or any(part in line_text.lower() for part in enhanced_vat.lower().split('-')):
                    confidence_column = 'confidence' if 'confidence' in row else 'confidence_score'
                    line_confidence = float(row[confidence_column]) if confidence_column in row and pd.notna(row[confidence_column]) else 0.5
                    
                    # Boost confidence if line contains VAT keywords
                    if re.search(r'\bvat\b|\bv\.a\.t\b', line_text, re.IGNORECASE):
                        line_confidence += 0.3
                    
                    if line_confidence > best_confidence:
                        best_confidence = line_confidence
                        best_line = line_text
            
            if best_line:
                print(f"✅ Enhanced VAT detection found: '{enhanced_vat}' in line: '{best_line}'")
                return {
                    'value': enhanced_vat,
                    'raw_text': best_line,
                    'confidence': min(best_confidence, 0.95),
                    'line_number': 1,
                    'extraction_method': 'enhanced_ultra_dynamic_detector',
                    'pattern_used': 'Ultra-dynamic VAT pattern with OCR noise filtering'
                }
            else:
                print(f"✅ Enhanced VAT detection found: '{enhanced_vat}' (no specific line matched)")
                return {
                    'value': enhanced_vat,
                    'raw_text': enhanced_vat,
                    'confidence': 0.8,
                    'line_number': 1,
                    'extraction_method': 'enhanced_ultra_dynamic_detector',
                    'pattern_used': 'Ultra-dynamic VAT pattern (full text scan)'
                }
        
        print(f"🔄 Enhanced detection failed, falling back to legacy pattern matching...")
        # Fallback to generic field extraction
        return self._extract_vat_number_generic(df)
    
    def extract_vat_number_from_text(self, text: str) -> Optional[str]:
        """
        Ultra-dynamic VAT number extractor.
        Detects patterns like:
        - VAT : 427-6572-31
        - VAT No: GB 220 4302 31
        - VAT Reg No. 243 L 105 93
        - VAT Registration no: GB 135 0685 22
        - V.A.T Registration Number: FR123456789
        """
        # Normalize spacing
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s{2,}", " ", text)

        # Flexible regex: covers all 'VAT' + 'Reg/Registration' + 'No/Number' forms
        vat_pattern = re.compile(
            r"(?i)\bV\.?A\.?T\.?(?:\s*(?:Reg(?:istration)?\.?)?\s*(?:No\.?|Number|num|n[ou])?)?\s*[:\-]?\s*([A-Z]{0,3}\s*[\dA-Z\- ]{5,25})"
        )

        matches = vat_pattern.findall(text)
        if not matches:
            return None

        valid_vats = []
        for vat_raw in matches:
            vat_clean = re.sub(r"[^A-Z0-9\-]", "", vat_raw.upper()).strip("-")

            # Remove trailing OCR noise letters
            vat_clean = re.sub(r"[A-Z]+$", "", vat_clean)

            # Trim after the last numeric digit
            last_digit_match = re.search(r"\d(?=[^0-9]*$)", vat_clean)
            if last_digit_match:
                vat_clean = vat_clean[: last_digit_match.end()]

            vat_clean = vat_clean.strip()

            # Validate typical VAT number length (7–12)
            if 7 <= len(vat_clean) <= 12:
                valid_vats.append(vat_clean)

        # Return first valid match
        return valid_vats[0] if valid_vats else None
    
    def _extract_vat_number_generic(self, df) -> Optional[ExtractionResult]:
        """Generic VAT number extraction using pattern matching."""
        rules = self.extraction_rules.get('vat_extraction', {})
        target_classes = rules.get('target_classes', ['HEADER', 'IGNORE', 'FOOTER'])
        
        class_column = 'line_type' if 'line_type' in df.columns else 'predicted_class'
        candidate_lines = df[df[class_column].isin(target_classes)].copy()
        
        best_match = None
        highest_confidence = 0.0
        
        # Search using configured patterns
        for _, row in candidate_lines.iterrows():
            text_column = 'cleaned_text' if 'cleaned_text' in row else ('text' if 'text' in row else 'original_text')
            text = str(row[text_column]).strip()
            
            for pattern_config in self.vat_patterns:
                pattern = pattern_config['pattern']
                match = self.pattern_matcher.search_pattern(text, pattern, re.IGNORECASE)
                
                if match:
                    # Extract the field value (last group)
                    field_value = match.groups()[-1] if match.groups() else match.group(0)
                    
                    # Clean VAT number
                    field_value = re.sub(r"[^A-Z0-9\-]", "", field_value.upper()).strip("-")
                    field_value = re.sub(r"[A-Z]+$", "", field_value)
                    last_digit_match = re.search(r"\d(?=[^0-9]*$)", field_value)
                    if last_digit_match:
                        field_value = field_value[: last_digit_match.end()]
                    field_value = field_value.strip()
                    
                    # Calculate confidence
                    confidence_column = 'confidence' if 'confidence' in row else 'confidence_score'
                    ml_confidence = float(row[confidence_column]) if confidence_column in row and pd.notna(row[confidence_column]) else 0.5
                    keyword_confidence = 0.4 if re.search(r'\bvat\b', text, re.IGNORECASE) else 0.0
                    combined_confidence = (ml_confidence + keyword_confidence + 0.4) / 2.4
                    
                    if combined_confidence > highest_confidence:
                        highest_confidence = combined_confidence
                        best_match = {
                            'value': field_value,
                            'raw_text': text,
                            'confidence': combined_confidence,
                            'line_number': int(row.get('line_number', 0)),
                            'extraction_method': 'config_regex_pattern',
                            'pattern_used': pattern_config['description']
                        }
        
        return best_match   