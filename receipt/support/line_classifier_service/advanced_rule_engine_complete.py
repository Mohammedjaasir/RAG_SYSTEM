"""
Advanced Rule-Based Line Classifier - Complete Implementation

This classifier uses sophisticated rule-based logic with:
1. Multi-level priority rules
2. Feature-weighted confidence scoring
3. Context-aware classification
4. Fuzzy logic for borderline cases
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from .advanced_feature_extractor import AdvancedFeatureExtractor


class LineType(Enum):
    """Enhanced line classification types."""
    ITEM_HEADER = "ITEM_HEADER"
    ITEM_DATA = "ITEM_DATA"
    SUMMARY_KEY_VALUE = "SUMMARY_KEY_VALUE"
    VAT_HEADER = "VAT_HEADER"
    VAT_DATA = "VAT_DATA"
    HEADER = "HEADER"
    FOOTER = "FOOTER"
    IGNORE = "IGNORE"


@dataclass
class ClassificationResult:
    """Enhanced classification result with detailed metadata."""
    line_type: LineType
    confidence: float
    primary_confidence: float
    secondary_confidence: float
    reasons: List[str]
    features_used: List[str]
    rule_triggered: str
    rule_priority: int
    evidence_score: float
    context_bonus: float


class AdvancedRuleEngine:
    """Sophisticated rule-based classifier with multi-level logic."""
    
    def __init__(self, analysis_root: str = None):
        """Initialize the advanced classifier."""
        self.feature_extractor = AdvancedFeatureExtractor(analysis_root)
        
        # Classification history for context
        self.document_history = []
        self.recent_classifications = []
        self.max_history = 10
        
        # Rule weights and thresholds
        self.rule_weights = self._define_rule_weights()
        self.confidence_thresholds = self._define_confidence_thresholds()
    
    def _define_rule_weights(self) -> Dict[str, float]:
        """Define weights for different types of evidence."""
        return {
            'keyword_exact_match': 1.0,
            'keyword_partial_match': 0.6,
            'pattern_strong_match': 0.9,
            'pattern_weak_match': 0.5,
            'position_primary': 0.7,
            'position_secondary': 0.4,
            'structure_match': 0.6,
            'context_boost': 0.3,
            'transaction_summary_keywords': 1.2,
            'itemization_keywords': 1.1,
            'tax_keywords': 1.0,
            'payment_keywords': 0.9,
            'company_keywords': 0.8,
            'price_pattern': 1.0,
            'percentage_pattern': 0.9,
            'date_time_pattern': 0.7,
            'contact_pattern': 0.6,
            'header_position_bonus': 0.3,
            'footer_position_bonus': 0.2,
            'body_position_neutral': 0.0,
        }
    
    def _define_confidence_thresholds(self) -> Dict[str, float]:
        """Define confidence thresholds for decision making."""
        return {
            'very_high': 0.9,
            'high': 0.75,
            'medium': 0.6,
            'low': 0.4,
            'reject': 0.3
        }
    
    def classify_line(self, text: str, line_position: int = 0, 
                     total_lines: int = 1, ocr_confidence: float = None) -> ClassificationResult:
        """Classify a line using advanced rule-based logic."""
        
        # Extract comprehensive features
        features = self.feature_extractor.extract_features(
            text, line_position, total_lines, ocr_confidence
        )
        
        # Apply rule cascade
        result = self._apply_rule_cascade(features, text)
        
        # Apply context adjustments
        result = self._apply_context_adjustments(result, features)
        
        # Update history
        self._update_history(result)
        
        return result
    
    def _apply_rule_cascade(self, features: Dict, text: str) -> ClassificationResult:
        """Apply rules in priority order with confidence scoring."""
        
        # Priority 1: Handle empty/ignore lines
        if features.get('is_empty', False) or features.get('char_count', 0) <= 2:
            return ClassificationResult(
                line_type=LineType.IGNORE,
                confidence=0.95,
                primary_confidence=0.95,
                secondary_confidence=0.0,
                evidence_score=0.95,
                context_bonus=0.0,
                rule_triggered="ignore_empty",
                rule_priority=1,
                reasons=["Very short or empty line"],
                features_used=["char_count"]
            )
        
        # Priority 2: Strong itemization headers
        header_score = self._calculate_item_header_score(features)
        if header_score >= self.confidence_thresholds['high']:
            return ClassificationResult(
                line_type=LineType.ITEM_HEADER,
                confidence=header_score,
                primary_confidence=header_score,
                secondary_confidence=0.0,
                evidence_score=header_score,
                context_bonus=0.0,
                rule_triggered="item_header_strong",
                rule_priority=2,
                reasons=self._get_item_header_reasons(features),
                features_used=["has_qty_description"]
            )
        
        # Priority 3: Strong summary lines
        summary_score = self._calculate_summary_score(features)
        if summary_score >= self.confidence_thresholds['high']:
            return ClassificationResult(
                line_type=LineType.SUMMARY_KEY_VALUE,
                confidence=summary_score,
                primary_confidence=summary_score,
                secondary_confidence=0.0,
                evidence_score=summary_score,
                context_bonus=0.0,
                rule_triggered="summary_strong",
                rule_priority=3,
                reasons=self._get_summary_reasons(features),
                features_used=["has_price"]
            )
        
        # Priority 4: VAT data with percentage
        vat_data_score = self._calculate_vat_data_score(features)
        if vat_data_score >= self.confidence_thresholds['high']:
            return ClassificationResult(
                line_type=LineType.VAT_DATA,
                confidence=vat_data_score,
                primary_confidence=vat_data_score,
                secondary_confidence=0.0,
                evidence_score=vat_data_score,
                context_bonus=0.0,
                rule_triggered="vat_data_strong",
                rule_priority=4,
                reasons=self._get_vat_data_reasons(features),
                features_used=["has_percentage", "has_vat_keywords"]
            )
        
        # Priority 5: Header area classification
        header_area_score = self._calculate_header_area_score(features)
        if features.get('is_header_area', False) and header_area_score >= self.confidence_thresholds['medium']:
            return ClassificationResult(
                line_type=LineType.HEADER,
                confidence=header_area_score,
                primary_confidence=header_area_score,
                secondary_confidence=0.0,
                evidence_score=header_area_score,
                context_bonus=0.0,
                rule_triggered="header_area",
                rule_priority=5,
                reasons=self._get_header_reasons(features),
                features_used=["company_info_count"]
            )
        
        # Priority 6: Footer area classification  
        footer_area_score = self._calculate_footer_area_score(features)
        if features.get('is_footer_area', False) and footer_area_score >= self.confidence_thresholds['medium']:
            return ClassificationResult(
                line_type=LineType.FOOTER,
                confidence=footer_area_score,
                primary_confidence=footer_area_score,
                secondary_confidence=0.0,
                evidence_score=footer_area_score,
                context_bonus=0.0,
                rule_triggered="footer_area",
                rule_priority=6,
                reasons=self._get_footer_reasons(features),
                features_used=["miscellaneous_count"]
            )
        
        # Priority 7: Item data with price
        item_data_score = self._calculate_item_data_score(features)
        if item_data_score >= self.confidence_thresholds['medium']:
            return ClassificationResult(
                line_type=LineType.ITEM_DATA,
                confidence=item_data_score,
                primary_confidence=item_data_score,
                secondary_confidence=0.0,
                evidence_score=item_data_score,
                context_bonus=0.0,
                rule_triggered="item_data_price",
                rule_priority=7,
                reasons=["Contains price and quantity"],
                features_used=["is_body_area"]
            )
        
        # Priority 8: Weak itemization headers
        if header_score >= self.confidence_thresholds['medium']:
            return ClassificationResult(
                line_type=LineType.ITEM_HEADER,
                confidence=header_score,
                primary_confidence=header_score,
                secondary_confidence=0.0,
                evidence_score=header_score,
                context_bonus=0.0,
                rule_triggered="item_header_weak",
                rule_priority=8,
                reasons=self._get_item_header_reasons(features),
                features_used=["itemization_count"]
            )
        
        # Priority 9: Weak summary lines
        if summary_score >= self.confidence_thresholds['medium']:
            return ClassificationResult(
                line_type=LineType.SUMMARY_KEY_VALUE,
                confidence=summary_score,
                primary_confidence=summary_score,
                secondary_confidence=0.0,
                evidence_score=summary_score,
                context_bonus=0.0,
                rule_triggered="summary_weak",
                rule_priority=9,
                reasons=self._get_summary_reasons(features),
                features_used=["transaction_summary_count"]
            )
        
        # Priority 10: VAT header (without percentage)
        vat_header_score = self._calculate_vat_header_score(features)
        if vat_header_score >= self.confidence_thresholds['medium']:
            return ClassificationResult(
                line_type=LineType.VAT_HEADER,
                confidence=vat_header_score,
                primary_confidence=vat_header_score,
                secondary_confidence=0.0,
                evidence_score=vat_header_score,
                context_bonus=0.0,
                rule_triggered="vat_header",
                rule_priority=10,
                reasons=["VAT keywords without percentage"],
                features_used=["tax_and_fees_count"]
            )
        
        # Fallback: Low confidence classifications
        fallback_result = self._apply_fallback_rules(features)
        return fallback_result
    
    def _calculate_item_header_score(self, features: Dict) -> float:
        """Calculate confidence score for item header classification."""
        score = 0.0
        
        # Strong indicators
        if features.get('has_qty_description', False):
            score += 0.6
        
        itemization_count = features.get('itemization_count', 0)
        if itemization_count >= 3:
            score += 0.5
        elif itemization_count >= 2:
            score += 0.3
        elif itemization_count >= 1:
            score += 0.1
        
        # Position bonus
        if features.get('is_header_area', False):
            score += 0.2
        
        # Structural indicators
        if features.get('has_alignment_spacing', False):
            score += 0.15
        
        if features.get('all_caps', False):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_summary_score(self, features: Dict) -> float:
        """Calculate confidence score for summary line classification."""
        score = 0.0
        
        # Keywords
        if features.get('has_total_keywords', False):
            score += 0.5
        
        summary_count = features.get('transaction_summary_count', 0)
        score += min(summary_count * 0.2, 0.4)
        
        # Price pattern
        if features.get('has_price', False):
            score += 0.3
        
        # Position (body or footer area)
        if features.get('is_body_area', False) or features.get('is_footer_area', False):
            score += 0.1
        
        # Structural indicators
        if features.get('has_colon', False) or features.get('has_alignment_spacing', False):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_vat_data_score(self, features: Dict) -> float:
        """Calculate confidence score for VAT data classification."""
        score = 0.0
        
        # VAT keywords
        if features.get('has_vat_keywords', False):
            score += 0.4
        
        tax_count = features.get('tax_and_fees_count', 0)
        score += min(tax_count * 0.15, 0.3)
        
        # Percentage pattern (strong indicator)
        if features.get('has_percentage', False):
            score += 0.4
        
        # Price pattern
        if features.get('has_price', False):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_header_area_score(self, features: Dict) -> float:
        """Calculate confidence score for header area classification."""
        score = 0.0
        
        # Position weight
        if features.get('is_header_area', False):
            score += 0.3
        
        # Company info
        company_count = features.get('company_info_count', 0)
        score += min(company_count * 0.2, 0.4)
        
        # First line bonus
        if features.get('is_first_line', False):
            score += 0.2
        
        # Structural indicators
        if features.get('all_caps', False):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_footer_area_score(self, features: Dict) -> float:
        """Calculate confidence score for footer area classification."""
        score = 0.0
        
        # Position weight
        if features.get('is_footer_area', False):
            score += 0.3
        
        # Miscellaneous content
        misc_count = features.get('miscellaneous_count', 0)
        score += min(misc_count * 0.2, 0.4)
        
        # Contact info
        if features.get('has_contact_info', False):
            score += 0.2
        
        # Payment info
        payment_count = features.get('payment_count', 0)
        score += min(payment_count * 0.1, 0.2)
        
        return min(score, 1.0)
    
    def _calculate_item_data_score(self, features: Dict) -> float:
        """Calculate confidence score for item data classification."""
        score = 0.0
        
        # Price pattern (strong indicator for items)
        if features.get('has_price', False):
            score += 0.4
        
        # Body area position
        if features.get('is_body_area', False):
            score += 0.2
        
        # Numeric content
        if features.get('has_number', False):
            score += 0.1
        
        # Price and quantity combination
        if features.get('has_price_and_quantity', False):
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_vat_header_score(self, features: Dict) -> float:
        """Calculate confidence score for VAT header classification."""
        score = 0.0
        
        # VAT keywords without percentage
        if features.get('has_vat_keywords', False) and not features.get('has_percentage', False):
            score += 0.5
        
        tax_count = features.get('tax_and_fees_count', 0)
        score += min(tax_count * 0.2, 0.4)
        
        return min(score, 1.0)
    
    def _apply_fallback_rules(self, features: Dict) -> ClassificationResult:
        """Apply fallback rules for low-confidence cases."""
        
        # Default based on position
        if features.get('is_header_area', False):
            return ClassificationResult(
                line_type=LineType.HEADER,
                confidence=0.3,
                primary_confidence=0.3,
                secondary_confidence=0.0,
                evidence_score=0.3,
                context_bonus=0.0,
                rule_triggered="fallback_header",
                rule_priority=11,
                reasons=["Header area fallback"],
                features_used=["position"]
            )
        elif features.get('is_footer_area', False):
            return ClassificationResult(
                line_type=LineType.FOOTER,
                confidence=0.3,
                primary_confidence=0.3,
                secondary_confidence=0.0,
                evidence_score=0.3,
                context_bonus=0.0,
                rule_triggered="fallback_footer",
                rule_priority=12,
                reasons=["Footer area fallback"],
                features_used=["position"]
            )
        else:
            return ClassificationResult(
                line_type=LineType.IGNORE,
                confidence=0.2,
                primary_confidence=0.2,
                secondary_confidence=0.0,
                evidence_score=0.2,
                context_bonus=0.0,
                rule_triggered="fallback_ignore",
                rule_priority=13,
                reasons=["No classification rules matched"],
                features_used=["fallback"]
            )
    
    def _apply_context_adjustments(self, result: ClassificationResult, features: Dict) -> ClassificationResult:
        """Apply context-based adjustments to classification."""
        
        context_bonus = 0.0
        context_reasons = []
        
        # Look at recent classifications for context
        if self.recent_classifications:
            last_result = self.recent_classifications[-1]
            
            # Sequential item detection
            if (result.line_type == LineType.ITEM_DATA and 
                last_result.line_type in [LineType.ITEM_DATA, LineType.ITEM_HEADER]):
                context_bonus += 0.05
                context_reasons.append("Sequential item")
            
            # Summary after items
            if (result.line_type == LineType.SUMMARY_KEY_VALUE and 
                any(r.line_type == LineType.ITEM_DATA for r in self.recent_classifications[-3:])):
                context_bonus += 0.1
                context_reasons.append("Summary after items")
        
        # Apply context adjustments
        if context_bonus > 0:
            result.confidence = min(result.confidence + context_bonus, 1.0)
            result.context_bonus = context_bonus
            result.reasons.extend(context_reasons)
        
        return result
    
    def _update_history(self, result: ClassificationResult):
        """Update classification history for context awareness."""
        self.recent_classifications.append(result)
        if len(self.recent_classifications) > self.max_history:
            self.recent_classifications.pop(0)
    
    def _get_item_header_reasons(self, features: Dict) -> List[str]:
        """Get detailed reasons for item header classification."""
        reasons = []
        if features.get('has_qty_description', False):
            reasons.append("Contains QTY and DESCRIPTION keywords")
        if features.get('itemization_count', 0) >= 2:
            reasons.append(f"Multiple itemization keywords ({features.get('itemization_count', 0)})")
        if features.get('has_alignment_spacing', False):
            reasons.append("Aligned spacing pattern")
        return reasons
    
    def _get_summary_reasons(self, features: Dict) -> List[str]:
        """Get detailed reasons for summary classification."""
        reasons = []
        if features.get('has_total_keywords', False):
            reasons.append("Contains total/subtotal keywords")
        if features.get('has_price', False):
            reasons.append("Contains price pattern")
        if features.get('transaction_summary_count', 0) > 0:
            reasons.append(f"Transaction summary keywords ({features.get('transaction_summary_count', 0)})")
        return reasons
    
    def _get_vat_data_reasons(self, features: Dict) -> List[str]:
        """Get detailed reasons for VAT data classification."""
        reasons = []
        if features.get('has_vat_keywords', False):
            reasons.append("Contains VAT keywords")
        if features.get('has_percentage', False):
            reasons.append("Contains percentage pattern")
        if features.get('has_price', False):
            reasons.append("Contains price amount")
        return reasons
    
    def _get_header_reasons(self, features: Dict) -> List[str]:
        """Get detailed reasons for header classification."""
        reasons = []
        if features.get('company_info_count', 0) > 0:
            reasons.append(f"Company info keywords ({features.get('company_info_count', 0)})")
        if features.get('is_first_line', False):
            reasons.append("First line of document")
        if features.get('all_caps', False):
            reasons.append("All caps formatting")
        return reasons
    
    def _get_footer_reasons(self, features: Dict) -> List[str]:
        """Get detailed reasons for footer classification."""
        reasons = []
        if features.get('miscellaneous_count', 0) > 0:
            reasons.append(f"Footer keywords ({features.get('miscellaneous_count', 0)})")
        if features.get('has_contact_info', False):
            reasons.append("Contains contact information")
        if features.get('payment_count', 0) > 0:
            reasons.append("Payment-related content")
        return reasons
    
    def _get_item_data_reasons(self, features: Dict) -> List[str]:
        """Get detailed reasons for item data classification."""
        reasons = []
        if features.get('has_price', False):
            reasons.append("Contains price pattern")
        if features.get('has_price_and_quantity', False):
            reasons.append("Price and quantity combination")
        if features.get('is_body_area', False):
            reasons.append("Located in body area")
        return reasons
    
    def classify_document(self, lines: List[str], filename: str = None) -> List[ClassificationResult]:
        """Classify all lines in a document."""
        # Clear history for new document
        self.recent_classifications = []
        
        results = []
        for i, line in enumerate(lines):
            result = self.classify_line(line, i, len(lines))
            results.append(result)
        
        return results


def test_advanced_classifier():
    """Test the advanced classifier."""
    print("🚀 Testing Advanced Rule-Based Classifier")
    print("=" * 60)
    
    classifier = AdvancedRuleEngine()
    
    # Test with the same document as simple classifier
    test_lines = [
        "TESCO Express",
        "123 High Street, London",
        "",
        "QTY  DESCRIPTION      PRICE",
        "2    Coca Cola 330ml  £2.50",
        "1    Bread Loaf       £1.20", 
        "3    Milk 1L          £3.60",
        "",
        "SUB-TOTAL           £7.30",
        "VAT @ 20%           £1.46",
        "TOTAL               £8.76",
        "",
        "Thank you for shopping!",
        "Visit: www.tesco.com"
    ]
    
    results = classifier.classify_document(test_lines, "test_receipt.txt")
    
    print(f"\nAdvanced Classification Results:")
    print("-" * 60)
    
    for i, (line, result) in enumerate(zip(test_lines, results)):
        print(f"Line {i+1:2d}: {result.line_type.value:15s} ({result.confidence:.2f}) | {line[:40]}")
    
    # Show summary
    type_counts = {}
    for result in results:
        line_type = result.line_type.value
        type_counts[line_type] = type_counts.get(line_type, 0) + 1
    
    print(f"\n📊 Advanced Classification Summary:")
    for line_type, count in sorted(type_counts.items()):
        print(f"  {line_type:20s}: {count:3d} lines")
    
    print(f"\n✅ Advanced classifier test completed!")
    return results


if __name__ == "__main__":
    test_advanced_classifier()
