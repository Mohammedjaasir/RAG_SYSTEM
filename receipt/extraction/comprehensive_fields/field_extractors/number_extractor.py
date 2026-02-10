"""
Number Extractor - Extracts various identification numbers from receipts
"""
import re
import pandas as pd
from typing import Optional, List
from ..shared_utils.config_manager import ConfigManager
from ..shared_utils.pattern_matcher import PatternMatcher
from ..shared_utils.confidence_scorer import ConfidenceScorer
from ..support_modules.receipt_number_detector_integrated import IntegratedReceiptNumberDetector
from ..models.extraction_models import ExtractionResult


class NumberExtractor:
    """Extracts various identification numbers from receipts."""
    
    def __init__(self, config_manager: ConfigManager, pattern_matcher: PatternMatcher, confidence_scorer: ConfidenceScorer):
        self.config_manager = config_manager
        self.pattern_matcher = pattern_matcher
        self.confidence_scorer = confidence_scorer
        self.extraction_rules = config_manager.config.get('extraction_rules', {})
        
        # Get patterns for different number types
        self.receipt_number_patterns = config_manager.get_patterns('receipt_number_patterns')
        self.invoice_number_patterns = config_manager.get_patterns('invoice_number_patterns')
        self.transaction_number_patterns = config_manager.get_patterns('transaction_number_patterns')
        self.reference_number_patterns = config_manager.get_patterns('reference_number_patterns')
        self.auth_code_patterns = config_manager.get_patterns('auth_code_patterns')
    
    def extract_receipt_number(self, df) -> Optional[ExtractionResult]:
        """Enhanced receipt number extraction with dynamic detection as primary method."""
        print("🎫 Starting enhanced receipt number extraction...")
        
        # PRIORITY 1: Dynamic Receipt Number Detection (Primary Method)
        print("🎯 Priority 1: Applying dynamic receipt number detection as PRIMARY method...")
        
        dynamic_result = self._extract_receipt_number_dynamic_priority(df)
        if dynamic_result:
            print(f"✅ Dynamic receipt number detection found: {dynamic_result['value']}")
            print(f"   📋 Matched text: {dynamic_result['raw_text'][:100]}...")
            print(f"   🎯 Method: {dynamic_result['extraction_method']}")
            print(f"   📊 Confidence: {dynamic_result['confidence']}")
            return dynamic_result
        else:
            print("⚠️ Dynamic receipt number detection found no results - falling back to ML classification method")
            
        # PRIORITY 2: ML Classification-Based Detection (Fallback)
        print("🔄 Priority 2: Falling back to ML classification-based receipt number extraction...")
        
        ml_result = self._extract_receipt_number_ml_classification_fallback(df)
        if ml_result:
            print(f"✅ ML classification fallback found: {ml_result['value']}")
            print(f"   Method: {ml_result['extraction_method']}")
            print(f"   Confidence: {ml_result['confidence']}")
            return ml_result
        else:
            print("❌ Both dynamic and ML classification methods found no receipt numbers")
            return None
    
    def _extract_receipt_number_dynamic_priority(self, df) -> Optional[ExtractionResult]:
        """
        Extract receipt number using dynamic detection logic.
        Primary method with configurable keywords and enhanced pattern support.
        """
        print("🎯 Starting dynamic receipt number detection with configurable patterns...")
        
        # Convert DataFrame to text for dynamic detection
        text_column = 'cleaned_text' if 'cleaned_text' in df.columns else ('text' if 'text' in df.columns else 'original_text')
        if text_column not in df.columns:
            print("❌ No suitable text column found for dynamic receipt number detection")
            return None
            
        all_text = '\n'.join(df[text_column].fillna('').astype(str))
        
        # Enhanced keywords with priority order
        high_priority_keywords = [
            "receipt no", "receipt number", "receipt id", "receipt #", "receipt"
        ]
        
        medium_priority_keywords = [
            "invoice id", "invoice no", "invoice number", "invoice #", 
            "order number", "order no", "order #", "order id",
            "ticket number", "ticket no", "ticket #", "ticket id",
            "bill number", "bill no", "bill #", "bill id",
            "reference", "ref no", "ref #", "reference number",
            "invoice", "order", "ticket", "bill"
        ]
        
        low_priority_keywords = [
            "transaction no", "transaction number", "transaction id", "trans no", "transaction"
        ]
        
        # Enhanced pattern for various receipt number formats
        enhanced_pattern = r"[A-Z0-9\-_#]{3,20}"
        
        # Try high priority keywords first
        for priority_name, keywords in [
            ("high", high_priority_keywords),
            ("medium", medium_priority_keywords), 
            ("low", low_priority_keywords)
        ]:
            print(f"   🔍 Trying {priority_name} priority keywords: {keywords[:3]}...")
            
            # Initialize dynamic detector with current priority keywords
            detector = IntegratedReceiptNumberDetector(keywords, enhanced_pattern)
            detected_number = detector.detect_receipt_number(all_text)
            
            if detected_number != "Not found":
                print(f"   ✅ Found with {priority_name} priority: {detected_number}")
                
                # Find source line for metadata
                best_match_line = self._find_receipt_number_source_line(df, detected_number)
                
                confidence = 0.95 if priority_name == "high" else (0.9 if priority_name == "medium" else 0.85)
                
                return {
                    'value': detected_number.strip(),
                    'raw_text': best_match_line['text'] if best_match_line else all_text[:100],
                    'confidence': confidence,
                    'line_number': best_match_line['line_number'] if best_match_line else 0,
                    'extraction_method': 'dynamic_receipt_number_detector_primary',
                    'pattern_used': f'Dynamic pattern with {priority_name} priority keywords and enhanced alphanumeric support',
                    'pattern_type': 'dynamic_configurable'
                }
        
        print("   ❌ No receipt number found with any priority level")
        return None
    
    def _extract_receipt_number_ml_classification_fallback(self, df) -> Optional[ExtractionResult]:
        """
        Fallback method using original ML classification-based extraction.
        """
        print("🔄 Starting ML classification-based receipt number extraction fallback...")
        
        # Use generic field extraction method
        ml_result = self.extract_generic_field(
            df, 'receipt_number', 'receipt_number_patterns',
            'receipt_number_extraction', 'identification'
        )
        
        if ml_result:
            ml_result['extraction_method'] = 'ml_classification_config_patterns_fallback'
            ml_result['pattern_type'] = 'configuration_based'
            print(f"   📋 ML method found: {ml_result['value']}")
            print(f"   📊 Confidence: {ml_result['confidence']}")
        
        return ml_result
    
    def extract_invoice_number(self, df) -> Optional[ExtractionResult]:
        """Extract invoice number using configured patterns."""
        return self.extract_generic_field(
            df, 'invoice_number', 'invoice_number_patterns',
            'invoice_number_extraction', 'identification'
        )
    
    def extract_transaction_number(self, df) -> Optional[ExtractionResult]:
        """Extract transaction number using configured patterns."""
        return self.extract_generic_field(
            df, 'transaction_number', 'transaction_number_patterns',
            'transaction_number_extraction', 'identification'
        )
    
    def extract_reference_number(self, df) -> Optional[ExtractionResult]:
        """Extract reference number using configured patterns."""
        return self.extract_generic_field(
            df, 'reference_number', 'reference_number_patterns',
            'reference_number_extraction', 'identification'
        )
    
    def extract_auth_code(self, df) -> Optional[ExtractionResult]:
        """Extract authorization code using configured patterns."""
        return self.extract_generic_field(
            df, 'auth_code', 'auth_code_patterns',
            'auth_code_extraction', 'payment'
        )
    
    def extract_generic_field(self, df, field_name, patterns_key, rules_key, keyword_category) -> Optional[ExtractionResult]:
        """Generic extraction method for fields with similar pattern logic."""
        rules = self.extraction_rules.get(rules_key, {})
        target_classes = rules.get('target_classes', ['HEADER', 'IGNORE', 'FOOTER'])
        patterns = getattr(self, f'{patterns_key}', [])
        
        class_column = 'line_type' if 'line_type' in df.columns else 'predicted_class'
        candidate_lines = df[df[class_column].isin(target_classes)].copy()
        
        best_match = None
        highest_confidence = 0.0
        
        # Search using configured patterns
        for _, row in candidate_lines.iterrows():
            text_column = 'cleaned_text' if 'cleaned_text' in row else ('text' if 'text' in row else 'original_text')
            text = str(row[text_column]).strip()
            
            for pattern_config in patterns:
                pattern = pattern_config['pattern']
                match = self.pattern_matcher.search_pattern(text, pattern, re.IGNORECASE)
                
                if match:
                    # Extract the field value (last group)
                    field_value = match.groups()[-1] if match.groups() else match.group(0)
                    
                    # Clean field value if needed
                    if field_name in ['vat_number']:
                        field_value = re.sub(r"[^A-Z0-9\-]", "", field_value.upper()).strip("-")
                        field_value = re.sub(r"[A-Z]+$", "", field_value)
                        last_digit_match = re.search(r"\d(?=[^0-9]*$)", field_value)
                        if last_digit_match:
                            field_value = field_value[: last_digit_match.end()]
                        field_value = field_value.strip()
                    
                    # Calculate confidence
                    confidence_column = 'confidence' if 'confidence' in row else 'confidence_score'
                    ml_confidence = self.confidence_scorer.safe_convert(row[confidence_column])
                    keyword_confidence, _ = self.confidence_scorer.calculate_keyword_confidence(text, keyword_category)
                    combined_confidence = (ml_confidence + keyword_confidence + 0.4) / 2.4
                    
                    if combined_confidence > highest_confidence:
                        highest_confidence = combined_confidence
                        best_match = ExtractionResult(
                            value=field_value,
                            raw_text=text,
                            confidence=combined_confidence,
                            line_number=self.confidence_scorer.safe_convert(row.get('line_number', 0), int),
                            extraction_method='config_regex_pattern',
                            pattern_used=pattern_config['description']
                        )
        
        return best_match
    
    def _find_receipt_number_source_line(self, df, detected_number):
        """Find the source line that contains the detected receipt number for metadata."""
        text_column = 'cleaned_text' if 'cleaned_text' in df.columns else ('text' if 'text' in df.columns else 'original_text')
        
        # Clean the detected number for searching
        clean_number = detected_number.strip()
        
        # Search for the receipt number in the text
        for idx, row in df.iterrows():
            text = str(row[text_column]).strip()
            
            # Direct match for the receipt number
            if clean_number in text:
                return {
                    'text': text,
                    'line_number': self.confidence_scorer.safe_convert(row.get('line_number', idx + 1), int)
                }
            
            # Case-insensitive search for partial matches
            if clean_number.lower() in text.lower():
                return {
                    'text': text,
                    'line_number': self.confidence_scorer.safe_convert(row.get('line_number', idx + 1), int)
                }
        
        return None