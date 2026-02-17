#!/usr/bin/env python3
"""
Reproduction test for Hallucination Fix.
Tests if the updated RAG pipeline correctly identifies semantic fields
like Supplier and Total from noisy OCR text.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from receipt_rag import get_rag_pipeline

def test_hallucination_prevention():
    print("\n" + "="*70)
    print("TESTING HALLUCINATION PREVENTION (RECEIPT 1006 REPRODUCTION)")
    print("="*70 + "\n")
    
    # Using the FULL OCR text reported in the process logs
    full_ocr = "Akira 386 RtOW Gienmont,NY 12077 618-434-8880 www.akiras 2018-11-18 185834 #37 PICKUP roup.com - Le9, Phil QTY ITEM HiChk SteakK) Amt $22.95 Med Noodle Sushi Deluxe Glenmont Roll 3200 $20.75 $15.95 $6.50 $13.00 $5.50 $86.65 $6.93 Philly R Sp Tuna R SalmonR Subtotal Tax: Total:  Unpaid : $93.58"
    
    print("INPUT RAW OCR:")
    print("-" * 70)
    print(full_ocr)
    print("-" * 70)
    
    pipeline = get_rag_pipeline()
    
    print("\n[*] Running extraction...")
    result = pipeline.extract_from_ocr(full_ocr, retrieve_k=1)
    
    print("\n" + "="*70)
    print("EXTRACTION RESULTS")
    print("="*70)
    
    data = result.extracted_data
    print(json.dumps(data, indent=2))
    
    print(f"\nOverall Confidence: {result.confidence:.2%}")
    
    print("\nVERIFICATION:")
    
    # Check for Supplier
    supplier = data.get('supplier_name', '')
    if 'AKIRA' in str(supplier).upper():
        print("SUCCESS: Supplier name 'Akira' correctly extracted.")
    else:
        print(f"FAIL: Supplier name '{supplier}' is incorrect (Expected 'Akira').")

    # Check for Total Amount
    total = data.get('total_amount')
    if total == 93.58:
        print("SUCCESS: Total amount 93.58 correctly extracted.")
    elif total == 6.93:
        print("FAIL: Total amount is 6.93 (extracted the Tax instead of the Total).")
    else:
        print(f"FAIL: Total amount '{total}' is incorrect (Expected 93.58).")

    # Check for reasoning
    reasoning = data.get('extraction_reasoning')
    if reasoning:
        print(f"\nExtraction Reasoning: {reasoning}")
    else:
        print("\n[!] No extraction reasoning provided.")

    # Check address for hallucinations (should be clean of item info)
    address = str(data.get('address', ''))
    hallucination_report = result.hallucination_report or []
    
    hallucination_found = False
    if '$' in address or 'QTY' in address.upper() or 'AMT' in address.upper():
        print("FAIL: Address still contains leaked item data!")
        hallucination_found = True
    else:
        print("SUCCESS: Address field appears clean of item data.")
        
    if hallucination_report:
        print("\n[!] Hallucination Warnings Found:")
        for warning in hallucination_report:
            print(f"  - {warning}")
        
    if any("address" in w.lower() for w in hallucination_report):
        print("SUCCESS: Validation logic correctly flagged the address hallucination.")
    elif hallucination_found:
        print("FAIL: Validation logic failed to flag the address hallucination.")

if __name__ == "__main__":
    test_hallucination_prevention()
