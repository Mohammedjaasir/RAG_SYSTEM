#!/usr/bin/env python3
"""
Payment Extractor for Additional Fields Extractor v2.7.1
Main entry point - delegates to specialized extractors in payment_related module
"""

from .pattern_manager import PatternManager
from .payment_related import PaymentExtractor as PaymentExtractorImpl


class PaymentExtractor:
    """
    Main Payment Extractor class that delegates to specialized extractors.
    Provides backward compatibility with existing code.
    """
    
    def __init__(self, pattern_manager=None):
        """
        Initialize payment extractor with pattern manager.
        Delegates to the actual implementation in payment_related module.
        
        Args:
            pattern_manager: PatternManager instance
        """
        self.pattern_manager = pattern_manager or PatternManager()
        self._extractor = PaymentExtractorImpl(self.pattern_manager)
        print(f"✅ Initialized Payment Extractor v2.7.1")
    
    def extract_payment_details(self, df):
        """
        Enhanced payment information extraction with comprehensive method detection.
        Extracts payment methods, amounts, card details, and change information.
        Delegates to the PaymentExtractor implementation in payment_related module.
        
        Args:
            df: DataFrame with receipt text and predictions
            
        Returns:
            Dictionary with payment details
        """
        return self._extractor.extract_payment_details(df)

