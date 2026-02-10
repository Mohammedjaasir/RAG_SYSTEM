#!/usr/bin/env python3
"""
Additional Fields Extractor v2.7.1 - Refactored Version
Orchestrates specialized extractors for payment, discount, address, and contact details
"""

import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .additional_fields.pattern_manager import PatternManager
from .additional_fields.address_extractor import AddressExtractor
from .additional_fields.payment_extractor import PaymentExtractor
from .additional_fields.discount_contact_extractor import DiscountContactExtractor

class AdditionalFieldsExtractor:
    """
    Additional Fields Extractor v2.7.1 - Refactored
    
    Orchestrates specialized extractors:
    1. PaymentExtractor: Payment details (card/cash/change/authorization codes)
    2. AddressExtractor: Multi-country address extraction with postal code termination
    3. DiscountContactExtractor: Discount parsing and contact information
    4. PatternManager: Centralized pattern management
    
    CRITICAL FIX (v2.7.1):
    - FIXED: Supplier-based address extraction now stops immediately after postal/ZIP code
    """
    
    def __init__(self, config_path=None):
        """Initialize extractor with specialized components."""
        # Initialize core components
        self.pattern_manager = PatternManager(config_path)
        self.address_extractor = AddressExtractor(self.pattern_manager)
        self.payment_extractor = PaymentExtractor(self.pattern_manager)
        self.discount_contact_extractor = DiscountContactExtractor(self.pattern_manager, self.address_extractor)
        
        print(f"✅ Initialized Additional Fields Extractor v2.7.1 (Refactored)")
        print(f"   Specialized extractors: ✅ Pattern, Payment, Address, Discount/Contact")
        print(f"   Multi-country support: ✅ Ready (UK, US, CA, AU, IN)")
    
    def extract_additional_fields_from_csv(self, csv_file_path, discount_context=None, supplier_context=None):
        """
        Extract additional fields from CSV file (API integration method).
        
        Args:
            csv_file_path: Path to line classification CSV file
            discount_context: Optional discount context from comprehensive extractor
            supplier_context: Optional supplier context for address extraction
            
        Returns:
            Dictionary with payment, discount, and contact details
        """
        
        if not Path(csv_file_path).exists():
            return self._create_error_response(csv_file_path, f'CSV file not found: {csv_file_path}')
        
        try:
            # Read CSV and normalize column names
            df = pd.read_csv(csv_file_path)
            df = self._normalize_dataframe_columns(df)
            
            # Extract different field categories using specialized extractors
            payment_details = self.payment_extractor.extract_payment_details(df)
            discount_details = self.discount_contact_extractor.extract_discount_details(df, discount_context)
            contact_details = self.discount_contact_extractor.extract_contact_details(df, supplier_context)
            
            return self._create_success_response(
                csv_file_path, 
                len(df), 
                payment_details, 
                discount_details, 
                contact_details
            )
            
        except Exception as e:
            return self._create_error_response(csv_file_path, f'Error processing CSV: {str(e)}')
    
    def extract_additional_fields(self, ml_predictions_path):
        """
        Extract additional fields from ML predictions CSV file.
        
        Args:
            ml_predictions_path: Path to detailed_predictions.csv file
            
        Returns:
            Dictionary with payment, discount, and contact details
        """
        
        if not Path(ml_predictions_path).exists():
            return self._create_error_response(ml_predictions_path, f'ML predictions file not found: {ml_predictions_path}')
        
        try:
            # Read ML predictions
            df = pd.read_csv(ml_predictions_path)
            df = self._normalize_dataframe_columns(df)
            
            # Extract different field categories
            payment_details = self.payment_extractor.extract_payment_details(df)
            discount_details = self.discount_contact_extractor.extract_discount_details(df)
            contact_details = self.discount_contact_extractor.extract_contact_details(df, supplier_context=None)
            
            return self._create_success_response(
                ml_predictions_path, 
                len(df), 
                payment_details, 
                discount_details, 
                contact_details
            )
            
        except Exception as e:
            return self._create_error_response(ml_predictions_path, f'Error processing ML predictions: {str(e)}')
    
    def _normalize_dataframe_columns(self, df):
        """Normalize DataFrame column names for compatibility."""
        # Handle different column name formats for API compatibility
        if 'line_type' in df.columns and 'predicted_class' not in df.columns:
            df['predicted_class'] = df['line_type']
        if 'confidence' in df.columns and 'confidence_score' not in df.columns:
            df['confidence_score'] = df['confidence']
        if 'cleaned_text' in df.columns and 'text' not in df.columns:
            df['text'] = df['cleaned_text']
        
        # Ensure required columns exist
        if 'line_number' not in df.columns:
            df['line_number'] = range(len(df))
        
        return df
    
    def _create_success_response(self, file_path, total_lines, payment_details, discount_details, contact_details):
        """Create success response structure."""
        return {
            'payment_details': payment_details,
            'discount_details': discount_details,
            'contact_details': contact_details,
            'extraction_metadata': {
                'extraction_timestamp': datetime.now().isoformat(),
                'source_file': file_path,
                'extractor_version': "Additional Fields Extractor v2.7.1 (Refactored)",
                'total_lines_processed': total_lines,
                'extraction_status': 'success'
            }
        }
    
    def _create_error_response(self, file_path, error_message):
        """Create error response structure."""
        return {
            'payment_details': {
                'payment_methods': [],
                'payment_amounts': [],
                'change_details': [],
                'card_details': [],
                'cash_details': []
            },
            'discount_details': {
                'discounts': [],
                'savings': [],
                'coupons': [],
                'net_after_discount': None
            },
            'contact_details': {
                'addresses': [],
                'phone_numbers': [],
                'email_addresses': [],
                'fax_numbers': [],
                'websites': []
            },
            'extraction_metadata': {
                'extraction_timestamp': datetime.now().isoformat(),
                'source_file': file_path,
                'extractor_version': "Additional Fields Extractor v2.7.1 (Refactored)",
                'total_lines_processed': 0,
                'error': error_message,
                'extraction_status': 'failed'
            }
        }


# Test function for the refactored extractor
def test_refactored_extractor():
    """Test the refactored additional fields extractor on a sample file."""
    
    # Test on file 000 (has payment info)
    test_file = "/mnt/data/Projects/ML/solution/src/ml_training/full_dataset_inference_results/000_sorted_ml_training/detailed_predictions.csv"
    
    extractor = AdditionalFieldsExtractorRefactored()
    results = extractor.extract_additional_fields(test_file)
    
    print("\n" + "="*60)
    print("REFACTORED ADDITIONAL FIELDS EXTRACTION TEST RESULTS")
    print("="*60)
    
    # Payment Details
    print(f"\n📱 PAYMENT DETAILS:")
    payment = results['payment_details']
    print(f"   Payment Methods: {len(payment['payment_methods'])}")
    for method in payment['payment_methods']:
        print(f"     • {method['method']}: {method['raw_text']} (£{method['amount']})")
    
    print(f"   Change Details: {len(payment['change_details'])}")
    for change in payment['change_details']:
        print(f"     • {change['raw_text']} (Amount: {change['amount']})")
    
    # Discount Details
    print(f"\n💰 DISCOUNT DETAILS:")
    discount = results['discount_details']
    print(f"   Discounts: {len(discount['discounts'])}")
    for disc in discount['discounts']:
        print(f"     • {disc['raw_text']} (Amount: {disc['amount']}, %: {disc['percentage']})")
    
    # Contact Details
    print(f"\n📞 CONTACT DETAILS:")
    contact = results['contact_details']
    
    print(f"   Addresses: {len(contact['addresses'])}")
    for addr in contact['addresses']:
        print(f"     • {addr['full_address']}")
    
    print(f"   Phone Numbers: {len(contact['phone_numbers'])}")
    for phone in contact['phone_numbers']:
        print(f"     • {phone['phone_number']}: {phone['raw_text']}")
    
    return results


if __name__ == "__main__":
    # Run test
    test_results = test_refactored_extractor()