#!/usr/bin/env python3
"""
Test RAG Pipeline with Social Kitchen Receipt
"""

import os
import sys
import json
import logging
from pathlib import Path
from logger_utils import setup_output_capture

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

def test_social_kitchen_extraction():
    """Test the RAG pipeline with Social Kitchen receipt OCR text."""
    
    print("\n" + "="*70)
    print("TESTING RAG PIPELINE WITH SOCIAL KITCHEN RECEIPT")
    print("="*70 + "\n")
    
    # Import here to catch errors
    try:
        from receipt_rag import get_rag_pipeline
    except Exception as e:
        print(f"❌ Failed to import RAG pipeline: {e}")
        return
    
    # Social Kitchen receipt OCR text (transcribed from the image)
    social_kitchen_ocr = """
    Social Kitchen
    149, West Sambandam Road
    Coimbatore
    RS Puram-641002
    GST: 33GJGPS5840M1ZM
    
    25/01/2026 12:43 PM
    ORDER REF: 202601258YSBW54070
    BILL NO: 92464
    
    ITEM            QTY  PRICE  AMOUNT
    
    Udapi Pudi Masala Dosa
                    2   59.00  118.00
    Ragi masala Dosa
                    1   69.00   69.00
    Avil Milk Shakes
                    1   69.00   69.00
                    
    TOTAL           4          256.00
    
    CGST            2.5 %     6.10
    SGST            2.5 %     6.10
    
    GRAND TOTAL     256.00
    
    TOKEN: SS 44
    
    THANK YOU VISIT AGAIN
    """
    
    print("INPUT RECEIPT:")
    print("-" * 70)
    print(social_kitchen_ocr)
    print("-" * 70)
    
    with setup_output_capture(__file__):
        try:
            # Initialize RAG pipeline
            print("\n[*] Initializing RAG pipeline...")
            pipeline = get_rag_pipeline()
            
            # Run extraction
            print("\n[*] Running extraction on Social Kitchen receipt...")
            result = pipeline.extract_from_ocr(social_kitchen_ocr, retrieve_k=3)
            
            # Display results
            print("\n" + "="*70)
            print("EXTRACTION RESULTS")
            print("="*70)
            
            print(f"\nConfidence: {result.confidence:.2%}")
            
            print("\nExtracted Data:")
            print(json.dumps(result.extracted_data, indent=2))
            
            # Save results using helper
            pipeline.save_result_to_file(result, "social_kitchen")
            
            print("\n" + "="*70)
            print("[+] Test completed successfully!")
            print("="*70)
            
        except Exception as e:
            print(f"\n[X] Unexpected error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_social_kitchen_extraction()
