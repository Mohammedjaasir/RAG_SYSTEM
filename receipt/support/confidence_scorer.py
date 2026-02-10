#!/usr/bin/env python3
"""
Confidence Scorer - Weighted Confidence Calculation v1.0.0

Implements the production confidence scoring formula:
    Final Confidence = (OCR × 0.4) + (Pattern Match × 0.3) + (Model × 0.3)

Features:
- Per-field confidence breakdown
- Validation checks (total = subtotal + tax)
- Review flagging based on thresholds
- Audit trail for confidence decisions
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceResult:
    """Structured confidence result with full breakdown."""
    overall_confidence: float
    ocr_confidence: float
    pattern_match_confidence: float
    model_confidence: float
    
    # Review flags
    needs_review: bool
    auto_approved: bool
    
    # Per-field breakdown
    field_confidences: Dict[str, float]
    
    # Validation results
    validation_passed: bool
    validation_errors: List[str]
    
    # Weights used
    weights: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConfidenceScorer:
    """
    Production-grade confidence scorer for receipt extraction.
    
    Implements weighted scoring:
    - OCR confidence (40%): Raw OCR output quality
    - Pattern match (30%): RAG knowledge base match quality
    - Model confidence (30%): Phi-3 extraction confidence
    
    Thresholds:
    - < 0.7: Manual review required
    - 0.7 - 0.85: Standard output
    - >= 0.85: Auto-approve
    """
    
    # Configurable weights
    DEFAULT_WEIGHTS = {
        'ocr': 0.4,
        'pattern': 0.3,
        'model': 0.3
    }
    
    # Review thresholds
    REVIEW_THRESHOLD = 0.7
    AUTO_APPROVE_THRESHOLD = 0.85
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        review_threshold: float = 0.7,
        auto_approve_threshold: float = 0.85
    ):
        """
        Initialize confidence scorer.
        
        Args:
            weights: Custom weights for confidence components
            review_threshold: Below this triggers manual review
            auto_approve_threshold: Above this enables auto-approval
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.review_threshold = review_threshold
        self.auto_approve_threshold = auto_approve_threshold
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Weights sum to {weight_sum}, normalizing to 1.0")
            for key in self.weights:
                self.weights[key] /= weight_sum
        
        logger.info(f"✅ ConfidenceScorer initialized with weights: {self.weights}")
    
    def calculate(
        self,
        ocr_confidence: float,
        pattern_match_confidence: float,
        model_confidence: float,
        extracted_data: Optional[Dict[str, Any]] = None,
        field_confidences: Optional[Dict[str, float]] = None
    ) -> ConfidenceResult:
        """
        Calculate weighted confidence score.
        
        Args:
            ocr_confidence: OCR quality score (0-1)
            pattern_match_confidence: RAG pattern match score (0-1)
            model_confidence: Phi-3 model confidence (0-1)
            extracted_data: Optional extracted data for validation
            field_confidences: Optional per-field confidence scores
            
        Returns:
            ConfidenceResult with full breakdown
        """
        # Clamp inputs to [0, 1]
        ocr_conf = max(0.0, min(1.0, ocr_confidence))
        pattern_conf = max(0.0, min(1.0, pattern_match_confidence))
        model_conf = max(0.0, min(1.0, model_confidence))
        
        # Calculate weighted overall confidence
        overall = (
            ocr_conf * self.weights['ocr'] +
            pattern_conf * self.weights['pattern'] +
            model_conf * self.weights['model']
        )
        
        # Determine review flags
        needs_review = overall < self.review_threshold
        auto_approved = overall >= self.auto_approve_threshold
        
        # Run validation if data provided
        validation_passed = True
        validation_errors = []
        
        if extracted_data:
            validation_passed, validation_errors = self._validate_extraction(extracted_data)
            
            # If validation fails, force review
            if not validation_passed:
                needs_review = True
                auto_approved = False
        
        # Use provided field confidences or calculate defaults
        final_field_confidences = field_confidences or {}
        
        result = ConfidenceResult(
            overall_confidence=round(overall, 4),
            ocr_confidence=round(ocr_conf, 4),
            pattern_match_confidence=round(pattern_conf, 4),
            model_confidence=round(model_conf, 4),
            needs_review=needs_review,
            auto_approved=auto_approved,
            field_confidences=final_field_confidences,
            validation_passed=validation_passed,
            validation_errors=validation_errors,
            weights=self.weights.copy()
        )
        
        logger.debug(f"Confidence: {overall:.3f} (OCR:{ocr_conf:.2f}, Pattern:{pattern_conf:.2f}, Model:{model_conf:.2f})")
        
        return result
    
    def calculate_from_items(
        self,
        items: List[Dict[str, Any]],
        ocr_confidence: float = 0.8,
        rag_context_available: bool = False,
        pattern_match_confidence: Optional[float] = None,
        per_field_confidences: Optional[Dict[str, float]] = None
    ) -> ConfidenceResult:
        """
        Calculate confidence from extracted items list.
        
        Args:
            items: List of extracted items with confidence scores
            ocr_confidence: Base OCR confidence
            rag_context_available: Whether RAG context was used (fallback)
            pattern_match_confidence: Direct pattern match confidence from RAG (if available)
            per_field_confidences: Per-field confidence from full extraction (vendor, totals, etc.)
            
        Returns:
            ConfidenceResult with aggregated scores
        """
        # Calculate average model confidence from items
        if items:
            item_confidences = [
                item.get('extraction_confidence', item.get('confidence', 0.5))
                for item in items
            ]
            model_confidence = sum(item_confidences) / len(item_confidences)
        else:
            model_confidence = 0.5
        
        # Use provided pattern_match_confidence, or fallback to boolean-based
        if pattern_match_confidence is not None:
            pattern_confidence = pattern_match_confidence
        else:
            pattern_confidence = 0.7 if rag_context_available else 0.5
        
        # Aggregate field confidences from items
        field_confidences = self._aggregate_field_confidences(items)
        
        # Merge per-field confidences from full extraction (vendor, totals, etc.)
        if per_field_confidences:
            field_confidences.update(per_field_confidences)
        
        # Build extracted data for validation
        extracted_data = {
            'items': items,
            'item_count': len(items)
        }
        
        return self.calculate(
            ocr_confidence=ocr_confidence,
            pattern_match_confidence=pattern_confidence,
            model_confidence=model_confidence,
            extracted_data=extracted_data,
            field_confidences=field_confidences
        )
    
    def _aggregate_field_confidences(
        self,
        items: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Aggregate field confidences from items."""
        field_scores = {
            'item_name': [],
            'quantity': [],
            'price': [],
            'vat_code': []
        }
        
        for item in items:
            breakdown = item.get('confidence_breakdown', {})
            for field, scores in field_scores.items():
                if field in breakdown:
                    scores.append(breakdown[field])
        
        # Calculate averages
        aggregated = {}
        for field, scores in field_scores.items():
            if scores:
                aggregated[field] = round(sum(scores) / len(scores), 4)
        
        return aggregated
    
    def _validate_extraction(
        self,
        extracted_data: Dict[str, Any]
    ) -> tuple:
        """
        Validate extracted data for consistency.
        
        Returns:
            Tuple of (passed: bool, errors: List[str])
        """
        errors = []
        
        # Check for required fields
        items = extracted_data.get('items', [])
        
        if not items:
            errors.append("No items extracted")
        
        # Validate individual items
        for i, item in enumerate(items):
            item_name = item.get('item_name', '')
            price = item.get('item_price') or item.get('item_amount')
            
            if not item_name:
                errors.append(f"Item {i+1}: Missing item name")
            
            if price is not None:
                try:
                    float_price = float(price)
                    # Check for unreasonable prices
                    if float_price < 0 and abs(float_price) > 10000:
                        errors.append(f"Item {i+1}: Suspicious negative amount")
                    if float_price > 100000:
                        errors.append(f"Item {i+1}: Unusually high price")
                except (ValueError, TypeError):
                    errors.append(f"Item {i+1}: Invalid price format")
        
        # Check total validation if available
        subtotal = extracted_data.get('subtotal')
        tax = extracted_data.get('tax_amount')
        total = extracted_data.get('total_amount')
        
        if subtotal is not None and tax is not None and total is not None:
            try:
                calculated_total = float(subtotal) + float(tax)
                actual_total = float(total)
                
                # Allow 1% tolerance for rounding
                tolerance = actual_total * 0.01
                if abs(calculated_total - actual_total) > max(tolerance, 0.02):
                    errors.append(
                        f"Total mismatch: {subtotal} + {tax} != {total}"
                    )
            except (ValueError, TypeError):
                pass  # Can't validate non-numeric values
        
        passed = len(errors) == 0
        return passed, errors


# Singleton instance
_confidence_scorer_instance = None


def get_confidence_scorer() -> ConfidenceScorer:
    """Get or create singleton ConfidenceScorer instance."""
    global _confidence_scorer_instance
    
    if _confidence_scorer_instance is None:
        _confidence_scorer_instance = ConfidenceScorer()
    
    return _confidence_scorer_instance
