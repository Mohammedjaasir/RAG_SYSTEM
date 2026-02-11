#!/usr/bin/env python3
"""
Verification script for Generalized Receipt Extraction
Tests the pipeline with a diverse, unseen receipt (Hotel Bill)
"""

import os
import sys
import json
import logging
from pathlib import Path
from logger_utils import setup_output_capture

# Add project root to path (parent of 'receipt' package)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

def test_diverse_receipt():
    """Test the RAG pipeline with a diverse hotel receipt sample."""
    
    print("\n" + "="*70)
    print("VERIFYING GENERALIZED EXTRACTION: HOTEL RECEIPT")
    print("="*70 + "\n")
    
    try:
        from receipt.rag.receipt_rag import get_rag_pipeline
    except Exception as e:
        print(f"❌ Failed to import RAG pipeline: {e}")
        return
    
    # New diverse OCR sample: A Hotel Bill
    hotel_ocr = """
    GRAND PLAZA HOTEL
    123 Luxury Way, London, W1 1AA
    Tel: +44 20 7123 4567
    VAT NO: GB 987 6543 21
    
    GUEST INVOICE
    Invoice No: 887654
    Date: 2024-02-05
    Guest: Mr. John Doe
    Room: 402
    
    Description             Qty    Rate       Total
    Accommodation           2      £150.00    £300.00
    Breakfast               2      £15.00     £30.00
    Mini Bar                1      £12.50     £12.50
    VAT (20%)                                 £68.50
    
    TOTAL PAYABLE                             £411.00
    
    Payment Type: AMEX **** 1234
    Status: PAID
    """
    
    print("INPUT RECEIPT:")
    print("-" * 70)
    print(hotel_ocr.strip())
    print("-" * 70)
    
    with setup_output_capture(__file__):
        try:
            pipeline = get_rag_pipeline()
            
            print("\n[*] Running extraction...")
            result = pipeline.extract_from_ocr(hotel_ocr, retrieve_k=2)
            
            print("\n" + "="*70)
            print("EXTRACTION RESULTS")
            print("="*70)
            
            data = result.extracted_data
            print(f"Supplier: {data.get('supplier_name')}")
            print(f"Date:     {data.get('date') or data.get('receipt_date')}")
            print(f"Total:    {data.get('total_amount')}")
            print(f"VAT #:    {data.get('vat_number')}")
            
            print("\nITEMS:")
            items = data.get('items', [])
            if items:
                for item in items:
                    name = item.get('name', item.get('description', 'N/A'))
                    price = item.get('total_price', item.get('item_amount', 'N/A'))
                    print(f"  - {name}: {price}")
            else:
                print("  No items found.")
                
            print("\nFull JSON:")
            print(json.dumps(data, indent=2))
            
            # Save results using helper
            pipeline.save_result_to_file(result, "diverse_hotel")
            
            print("\n" + "="*70)
            print("[+] Verification complete!")
            print("="*70)
            
        except Exception:
            import traceback
            print("\n" + "="*70)
            print("CRITICAL ERROR DURING INITIALIZATION")
            print("="*70)
            traceback.print_exc()
            print("="*70)

if __name__ == "__main__":
    test_diverse_receipt()
