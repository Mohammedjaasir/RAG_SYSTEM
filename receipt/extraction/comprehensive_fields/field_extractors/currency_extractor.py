"""
Currency Extractor - Extracts currency information from receipts
"""
import re
import pandas as pd
from typing import Optional
from ..shared_utils.config_manager import ConfigManager
from ..shared_utils.pattern_matcher import PatternMatcher
from ..shared_utils.confidence_scorer import ConfidenceScorer
from ..models.extraction_models import ExtractionResult


class CurrencyExtractor:
    """Extracts currency using configured patterns."""
    
    def __init__(self, config_manager: ConfigManager, pattern_matcher: PatternMatcher, confidence_scorer: ConfidenceScorer):
        self.config_manager = config_manager
        self.pattern_matcher = pattern_matcher
        self.confidence_scorer = confidence_scorer
        self.currency_patterns = config_manager.get_patterns('currency_patterns')
    
    def extract_currency(self, df) -> ExtractionResult:
        """Extract currency using configured patterns."""
        best_match = None
        highest_confidence = 0.0
        
        # Look through all lines for currency indicators
        for _, row in df.iterrows():
            text_column = 'cleaned_text' if 'cleaned_text' in row else ('text' if 'text' in row else 'original_text')
            text = str(row[text_column]).strip()
            
            for pattern_config in self.currency_patterns:
                pattern = pattern_config['pattern']
                if self.pattern_matcher.search_pattern(text, pattern, re.IGNORECASE):
                    # Determine currency
                    currency = 'GBP'  # Default for UK receipts
                    if '£' in text:
                        currency = 'GBP'
                    elif '€' in text or re.search(r'\\bE\\b', text):
                        currency = 'GBP'  # Assume OCR error
                    elif 'GBP' in text.upper():
                        currency = 'GBP'
                    
                    # Calculate confidence
                    confidence_column = 'confidence' if 'confidence' in row else 'confidence_score'
                    ml_confidence = self.confidence_scorer.safe_convert(row[confidence_column])
                    pattern_confidence = 0.8 if '£' in text else 0.6
                    combined_confidence = (ml_confidence + pattern_confidence) / 2
                    
                    if combined_confidence > highest_confidence:
                        highest_confidence = combined_confidence
                        best_match = {
                            'value': currency,
                            'raw_text': text,
                            'confidence': combined_confidence,
                            'line_number': self.confidence_scorer.safe_convert(row.get('line_number', 0), int),
                            'extraction_method': 'config_pattern_detection',
                            'pattern_used': pattern_config['description']
                        }
        
        # Default fallback as per rules
        if not best_match:
            return {
                'value': 'GBP',
                'raw_text': 'Default (UK receipt)',
                'confidence': 0.5,
                'line_number': 0,
                'extraction_method': 'default_fallback'
            }
        
        return best_match