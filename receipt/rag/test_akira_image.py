#!/usr/bin/env python3
"""
Test RAG Pipeline with OCR text transcribed from the user's image.
"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from receipt_rag import get_rag_pipeline
from logger_utils import setup_output_capture

def test_akira_image_extraction():
    # OCR text transcribed from the user's image
    akira_ocr = """
    Akira
    385 Rt 9 W
    Glenmont , NY 12077
    518-434-8880
    www.akirasushigroup.com
    2018-11-18 18:58:34
    #37 PICKUP
    Michelle
    
    QTY ITEM Amt
    1 Hi Chk Steak(K) $22.95
      Med
      Noodle $2.00
    1 Sushi Deluxe $20.75
    1 Glenmont Roll $15.95
    1 Philly R $6.50
    2 Sp Tuna R $13.00
    1 Salmon R $5.50
    
    Subtotal $86.65
    Tax $6.93
    Total $93.58
    *** Unpaid ***
    """
    
    with setup_output_capture(__file__):
        print("\n" + "="*70)
        print("PROCESSING AKIRA RECEIPT (TRANSCRIPTION FROM IMAGE)")
        print("="*70 + "\n")
        
        pipeline = get_rag_pipeline()
        
        print("\n[*] Running extraction...")
        result = pipeline.extract_from_ocr(akira_ocr, retrieve_k=3)
        
        # Display results
        print("\n" + "="*70)
        print("EXTRACTION RESULTS")
        print("="*70)
        
        print(f"\nConfidence: {result.confidence:.2%}")
        print("\nExtracted Data:")
        print(json.dumps(result.extracted_data, indent=2))
        
        # Save results using helper
        pipeline.save_result_to_file(result, "akira_from_image")
        
        print("\n" + "="*70)
        print("[+] Process completed successfully!")
        print("="*70)

if __name__ == "__main__":
    test_akira_image_extraction()
