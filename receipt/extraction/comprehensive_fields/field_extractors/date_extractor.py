"""
Date Extractor - Extracts receipt date using comprehensive detection methods
"""
import re
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any
from ..shared_utils.config_manager import ConfigManager
from ..shared_utils.pattern_matcher import PatternMatcher
from ..shared_utils.confidence_scorer import ConfidenceScorer
from ..support_modules.date_detector_integrated import IntegratedReceiptDateDetector
from ..models.extraction_models import ExtractionResult


class DateExtractor:
    """Extracts receipt date using comprehensive date detection as primary method."""
    
    def __init__(self, config_manager: ConfigManager, pattern_matcher: PatternMatcher, confidence_scorer: ConfidenceScorer):
        self.config_manager = config_manager
        self.pattern_matcher = pattern_matcher
        self.confidence_scorer = confidence_scorer
        self.extraction_rules = config_manager.config.get('extraction_rules', {})
        self.date_patterns = config_manager.get_patterns('date_patterns')
        self.integrated_date_detector = IntegratedReceiptDateDetector()
    
    def extract_date(self, df) -> Optional[ExtractionResult]:
        """Enhanced receipt date extraction with comprehensive date detection as primary method."""
        print("📅 Starting enhanced receipt date extraction...")
        
        # PRIORITY 1: Comprehensive Date Detection (Primary Method)
        print("🎯 Priority 1: Applying comprehensive date detection as PRIMARY method...")
        
        comprehensive_result = self._extract_date_comprehensive_priority(df)
        if comprehensive_result:
            print(f"✅ Comprehensive date detection found: {comprehensive_result['value']}")
            print(f"   📋 Matched text: {comprehensive_result['raw_text'][:100]}...")
            print(f"   🎯 Method: {comprehensive_result['extraction_method']}")
            print(f"   📊 Confidence: {comprehensive_result['confidence']}")
            return comprehensive_result
        else:
            print("⚠️ Comprehensive date detection found no results - falling back to ML classification method")
            
        # PRIORITY 2: ML Classification-Based Detection (Fallback)
        print("🔄 Priority 2: Falling back to ML classification-based date extraction...")
        
        ml_result = self._extract_date_ml_classification_fallback(df)
        if ml_result:
            print(f"✅ ML classification fallback found: {ml_result['value']}")
            print(f"   Method: {ml_result['extraction_method']}")
            print(f"   Confidence: {ml_result['confidence']}")
            return ml_result
        else:
            print("❌ Both comprehensive and ML classification methods found no dates")
            return None
    
    def _extract_date_comprehensive_priority(self, df) -> Optional[ExtractionResult]:
        """
        Extract date using comprehensive detection logic from receipt_date_detector.py.
        Primary method with smart search strategy and extensive format support.
        """
        print("🎯 Starting comprehensive date detection with smart search strategy...")
        
        # Convert DataFrame to text for comprehensive detection
        text_column = 'cleaned_text' if 'cleaned_text' in df.columns else ('text' if 'text' in df.columns else 'original_text')
        if text_column not in df.columns:
            print("❌ No suitable text column found for comprehensive date detection")
            return None
            
        all_text = '\n'.join(df[text_column].fillna('').astype(str))
        
        detected_date = self.integrated_date_detector.detect_date(all_text)
        
        if detected_date != "No date found":
            # Find source line for metadata
            best_match_line = self._find_date_source_line(df, detected_date)
            
            return {
                'value': detected_date,
                'raw_text': best_match_line['text'] if best_match_line else all_text[:100],
                'confidence': 0.92,
                'line_number': best_match_line['line_number'] if best_match_line else 0,
                'extraction_method': 'comprehensive_date_detector_primary',
                'pattern_used': 'Smart search with 10+ patterns and 15+ formats',
                'pattern_type': 'comprehensive_format'
            }
        
        return None
    
    def _extract_date_ml_classification_fallback(self, df) -> Optional[ExtractionResult]:
        """
        Fallback method using original ML classification-based extraction.
        Enhanced version of the original extract_date method.
        """
        print("🔄 Starting ML classification-based date extraction fallback...")
        
        if 'date_extraction' not in self.extraction_rules:
            print("⚠️  No date extraction rules found in fallback method")
            return None
            
        rules = self.extraction_rules['date_extraction']
        target_classes = rules.get('target_classes', ['HEADER', 'FOOTER', 'IGNORE'])
        
        class_column = 'line_type' if 'line_type' in df.columns else 'predicted_class'
        candidate_lines = df[df[class_column].isin(target_classes)].copy()
        
        best_match = None
        highest_confidence = 0.0
        
        print(f"   📋 Looking for dates in {len(candidate_lines)} candidate lines using {len(self.date_patterns)} config patterns")
        
        # Search using configured patterns
        for _, row in candidate_lines.iterrows():
            text_column = 'cleaned_text' if 'cleaned_text' in row else ('text' if 'text' in row else 'original_text')
            text = str(row[text_column]).strip()
            
            for pattern_config in self.date_patterns:
                pattern = pattern_config['pattern']
                matches = self.pattern_matcher.find_pattern_matches(text, pattern, re.IGNORECASE)
                
                for match in matches:
                    # For patterns with groups, extract the actual date part
                    date_str = None
                    if len(match.groups()) >= 1:
                        date_str = match.group(1)
                    else:
                        date_str = match.group(0)
                    
                    if not date_str or not date_str.strip():
                        continue
                    
                    date_str = date_str.strip()
                    print(f"   ✅ Found date candidate: '{date_str}' using pattern: {pattern_config['description']}")
                    
                    # Calculate confidence
                    confidence_column = 'confidence' if 'confidence' in row else 'confidence_score'
                    ml_confidence = self.confidence_scorer.safe_convert(row[confidence_column])
                    keyword_confidence, _ = self.confidence_scorer.calculate_keyword_confidence(text, 'date_and_time')
                    
                    # Enhanced pattern quality scoring
                    pattern_quality = self.confidence_scorer.calculate_date_pattern_quality(date_str, pattern_config)
                    
                    # Bonus for date/time prefix patterns
                    prefix_bonus = 0.0
                    if re.search(r'(?i)(date|time|date/time|datetime)[\s:\-]', text):
                        prefix_bonus = 0.15
                    
                    combined_confidence = min((ml_confidence + keyword_confidence + pattern_quality + prefix_bonus + 0.1) / 2.25, 1.0)
                    
                    if combined_confidence > highest_confidence:
                        highest_confidence = combined_confidence
                        best_match = {
                            'value': date_str,
                            'raw_text': text,
                            'confidence': combined_confidence,
                            'line_number': self.confidence_scorer.safe_convert(row.get('line_number', 0), int),
                            'extraction_method': 'ml_classification_fallback',
                            'pattern_used': pattern_config['description'],
                            'pattern_type': self._get_date_pattern_type(date_str)
                        }
        
        return best_match
    
    def _get_date_pattern_type(self, date_str):
        """Determine the type/format of the extracted date."""
        if re.match(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$', date_str):
            return 'numeric_dmy'
        elif re.match(r'^\d{1,2}\.\d{1,2}\.\d{2,4}$', date_str):
            return 'numeric_dmy_dots'
        elif re.match(r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$', date_str):
            return 'iso_ymd'
        elif re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', date_str, re.IGNORECASE):
            return 'text_full_month'
        elif re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', date_str, re.IGNORECASE):
            return 'text_abbrev_month'
        elif re.search(r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b', date_str, re.IGNORECASE):
            return 'text_with_full_day'
        elif re.search(r'\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b', date_str, re.IGNORECASE):
            return 'text_with_abbrev_day'
        else:
            return 'other_format'
    
    def _find_date_source_line(self, df, detected_date):
        """Find the source line that contains the detected date for metadata."""
        text_column = 'cleaned_text' if 'cleaned_text' in df.columns else ('text' if 'text' in df.columns else 'original_text')
        
        # Convert detected date to various formats to search for
        try:
            dt = datetime.strptime(detected_date, '%Y-%m-%d')
            search_formats = [
                dt.strftime('%d/%m/%Y'), dt.strftime('%d-%m-%Y'), dt.strftime('%d.%m.%Y'),
                dt.strftime('%d/%m/%y'), dt.strftime('%d-%m-%y'), dt.strftime('%d.%m.%y'),
                dt.strftime('%d %b %Y'), dt.strftime('%d %B %Y'),
                dt.strftime('%b %d %Y'), dt.strftime('%B %d, %Y'),
                detected_date
            ]
        except:
            search_formats = [detected_date]
        
        for idx, row in df.iterrows():
            text = str(row[text_column]).strip()
            for date_format in search_formats:
                if date_format in text:
                    return {
                        'text': text,
                        'line_number': self.confidence_scorer.safe_convert(row.get('line_number', idx + 1), int)
                    }
        
        return None