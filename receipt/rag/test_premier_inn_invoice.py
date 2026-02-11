#!/usr/bin/env python3
"""
Test RAG Pipeline with OCR text transcribed from the user's Premier Inn image.
"""

import sys
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from receipt_rag import get_rag_pipeline
from logger_utils import setup_output_capture

def test_premier_inn_invoice_extraction():
    # OCR text transcribed from the user's image
    invoice_ocr = """
    Premier Inn
    Premier Inn Ashford North
    Maidstone Road, Hothfield Common, Ashford
    Kent
    TN26 1AP
    
    Invoice
    
    Ben Ten
    28 Derby Road
    Wirksworth
    MATLOCK
    DE4 4BG
    
    Invoice Date: 01.12.2025
    Guest Name: Ben Ten
    Room No:
    Arrival Date: 01.12.2025
    Departure Date: 02.12.2025
    Invoice No: AAFI29713
    Confirmation No: AAF5003454
    Customer Reference:
    PO No:
    Date Issued: 01.12.2025
    
    Date Description Net Amount in VAT in % VAT Amount in Gross Amount in Amount Received
    01.12.2025 Digital Visa 0.00 0 54
    01.12.2025 Prepayment (20% VAT) Accommodation 45.00 20% 9.00 54.00
    
    Total 54.00 54.00
    Balance 0.00
    """
    
    with setup_output_capture(__file__):
        print("\n" + "="*70)
        print("PROCESSING PREMIER INN INVOICE (TRANSCRIPTION FROM IMAGE)")
        print("="*70 + "\n")
        
        pipeline = get_rag_pipeline()
        
        print("\n[*] Running extraction...")
        result = pipeline.extract_from_ocr(invoice_ocr, retrieve_k=3)
        
        # Display results
        print("\n" + "="*70)
        print("EXTRACTION RESULTS")
        print("="*70)
        
        print(f"\nConfidence: {result.confidence:.2%}")
        print("\nExtracted Data:")
        print(json.dumps(result.extracted_data, indent=2))
        
        # Save results using helper
        pipeline.save_result_to_file(result, "premier_inn_invoice")
        
        print("\n" + "="*70)
        print("[+] Process completed successfully!")
        print("="*70)

if __name__ == "__main__":
    test_premier_inn_invoice_extraction()
