#!/usr/bin/env python3
"""
Test RAG Pipeline with Saska's Receipt
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

def test_saskas_extraction():
    """Test the RAG pipeline with Saska's receipt OCR text."""
    
    print("\n" + "="*70)
    print("TESTING RAG PIPELINE WITH SASKA'S RECEIPT")
    print("="*70 + "\n")
    
    # Import RAG pipeline
    from receipt_rag import get_rag_pipeline
    
    # Saska's receipt OCR text (manual transcription)
    saskas_ocr = """
    SASKA'S
    3768 Mission Blvd
    San Diego CA 92109
    (858) 488-7311
    
    Server: Sue
    Table 103/1
    Guests: 2
    Reprint #: 2
    08/15/2017
    5:51 PM
    10005
    
    Draft Blackhouse            7.00
    "Cowboy" 16oz Ribeye       45.00
    Make It Blue                5.00
    Add Lobster                21.00
    CK Manhattan  (2 @5.00)    10.00
    Whistle Pig Rye 10yr (2 @20.00) 40.00
    (2)BMOD Up
    "Duke" 16oz Top Sirloin    34.00
    Make It Blue                5.00
    
    Subtotal                  167.00
    Tax                        12.94
    
    Total                     179.94
    
    Balance Due               179.94
    
    SUGGESTED TIP
    BEFORE DISCOUNTS
    18% =$30.06
    20% =$33.40
    22% =$36.74
    """
    
    print("INPUT RECEIPT:")
    print("-" * 70)
    print(saskas_ocr)
    print("-" * 70)
    
    try:
        # Initialize RAG pipeline
        print("\n[*] Initializing RAG pipeline...")
        pipeline = get_rag_pipeline()
        
        # Run extraction
        print("\n[*] Running extraction on Saska's receipt...")
        result = pipeline.extract_from_ocr(saskas_ocr, retrieve_k=3)
        
        # Display results
        print("\n" + "="*70)
        print("EXTRACTION RESULTS")
        print("="*70)
        
        print(f"\nConfidence: {result.confidence:.2%}")
        
        # Save to file to avoid encoding issues
        output_file = Path("saskas_extraction.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result.extracted_data, f, indent=2)
            
        print(f"\nExtraction saved to {output_file.absolute()}")
        
        # Simple console output (ASCII only)
        extracted_data = result.extracted_data
        print(f"\nSupplier: {extracted_data.get('supplier_name', 'N/A')}")
        print(f"Total: {extracted_data.get('total_amount', 'N/A')}")

        
    except Exception as e:
        print(f"\n[X] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_saskas_extraction()
