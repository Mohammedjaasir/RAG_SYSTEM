"""
Confidence Scorer - Calculates confidence scores for extracted data
"""
import re
import pandas as pd
from typing import List, Tuple


class ConfidenceScorer:
    """Calculates confidence scores for extracted data."""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_keyword_confidence(self, text, category_name):
        """Calculate confidence based on keyword presence in text."""
        if not self.config or 'keyword_categories' not in self.config:
            return 0.0, []
        
        keyword_categories = self.config.get('keyword_categories', {}).get('categories', {})
        if category_name not in keyword_categories:
            return 0.0, []
        
        category_data = keyword_categories[category_name]
        keywords = category_data.get('keywords', [])
        high_value_keywords = category_data.get('high_value_keywords', {})
        base_weight = category_data.get('scoring_weight', 1.0)
        
        text_lower = text.lower()
        confidence = 0.0
        matches = []
        
        # Check for keyword matches
        for keyword in keywords:
            if keyword.lower() in text_lower:
                keyword_score = high_value_keywords.get(keyword, 1.0)
                confidence += keyword_score * 0.1
                matches.append(keyword)
        
        # Apply base weight
        if matches:
            confidence = min(confidence * base_weight * 0.1, 1.0)
        
        return confidence, matches
    
    def should_use_actual_class(self, row):
        """Determine if we should use actual_class instead of predicted_class."""
        class_column = 'line_type' if 'line_type' in row else 'predicted_class'
        predicted = row[class_column]
        
        # Check if actual_class column exists
        if 'actual_class' in row:
            actual = row['actual_class'] 
            confidence_column = 'confidence' if 'confidence' in row else 'confidence_score'
            confidence = row[confidence_column]
            
            # Use actual class if prediction differs and confidence is low (<0.7)
            if predicted != actual and confidence < 0.7:
                return True, actual
        
        return False, predicted
    
    def calculate_date_pattern_quality(self, date_str, pattern_config):
        """Calculate quality score based on date format and completeness."""
        pattern_quality = 0.0
        
        # Perfect formats get highest scores
        if re.match(r'^\d{2}/\d{2}/\d{4}$', date_str):
            pattern_quality = 0.35
        elif re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', date_str):
            pattern_quality = 0.32
        elif re.match(r'^\d{2}\.\d{2}\.\d{4}$', date_str):
            pattern_quality = 0.34
        elif re.match(r'^\d{1,2}\.\d{1,2}\.\d{4}$', date_str):
            pattern_quality = 0.31
        elif re.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$', date_str):
            pattern_quality = 0.33
        elif re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', date_str, re.IGNORECASE):
            pattern_quality = 0.30
        elif re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', date_str, re.IGNORECASE):
            pattern_quality = 0.28
        elif re.search(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', date_str, re.IGNORECASE):
            pattern_quality = 0.25
        elif re.search(r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b', date_str, re.IGNORECASE):
            pattern_quality = 0.22
        elif len([c for c in date_str if c.isdigit()]) >= 8:
            pattern_quality = 0.20
        elif len([c for c in date_str if c.isdigit()]) >= 6:
            pattern_quality = 0.15
        else:
            pattern_quality = 0.10
        
        # Additional bonuses for specific pattern types
        if 'ordinal' in pattern_config.get('description', '').lower():
            pattern_quality += 0.05
        if 'full' in pattern_config.get('description', '').lower():
            pattern_quality += 0.03
            
        return min(pattern_quality, 0.4)
    
    def extract_token_level_confidence(self, text_value, df):
        """
        Extract token-level confidence for a given text value.
        """
        if not text_value or pd.isna(text_value):
            return []
        
        # Convert to string and tokenize
        text_str = str(text_value).strip()
        tokens = text_str.split()
        
        confidence_tokens = []
        text_column = 'cleaned_text' if 'cleaned_text' in df.columns else 'text'
        confidence_column = 'confidence' if 'confidence' in df.columns else 'confidence_score'
        
        # Search for matching text in DataFrame rows
        for token in tokens:
            token_lower = token.lower()
            found = False
            
            for _, row in df.iterrows():
                row_text = str(row.get(text_column, '')).lower()
                if token_lower in row_text:
                    # Found the token in this row, use its confidence
                    conf = row.get(confidence_column, 1.0)
                    confidence_tokens.append({
                        'token': token,
                        'confidence': float(conf) if not pd.isna(conf) else 1.0
                    })
                    found = True
                    break
            
            # If not found, add with default confidence
            if not found:
                confidence_tokens.append({
                    'token': token,
                    'confidence': 1.0
                })
        
        return confidence_tokens
    
    def safe_convert(self, value, target_type=float):
        """Safely convert pandas/numpy types to native Python types."""
        try:
            if target_type == float:
                return float(value)
            elif target_type == int:
                return int(value)
            elif target_type == str:
                return str(value)
            else:
                return value
        except (ValueError, TypeError):
            return 0 if target_type in [int, float] else str(value)