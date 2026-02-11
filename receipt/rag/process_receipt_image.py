#!/usr/bin/env python3
"""
Process a receipt image using OCR and RAG extraction.
Saves terminal output and extraction results to files.
"""

import sys
import json
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ocr_client import get_ocr_client
from receipt_rag import get_rag_pipeline
from logger_utils import setup_output_capture

def process_image(image_path):
    """Full pipeline: Image -> OCR -> RAG -> Files"""
    
    p = Path(image_path)
    if not p.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return

    # Use TerminalTee to save everything to a log file
    with setup_output_capture(__file__):
        print("\n" + "="*70)
        print(f"PROCESSING RECEIPT IMAGE: {p.name}")
        print("="*70 + "\n")
        
        try:
            # 1. OCR Step
            print("[*] Initializing OCR client...")
            ocr_client = get_ocr_client()
            
            print(f"[*] Processing image via OCR service ({ocr_client.base_url})...")
            ocr_result = ocr_client.process_image(p)
            
            print(f"\n[✓] OCR complete! Extracted {len(ocr_result.text)} characters.")
            print("-" * 70)
            print("OCR TEXT PREVIEW:")
            print(ocr_result.text[:500] + "..." if len(ocr_result.text) > 500 else ocr_result.text)
            print("-" * 70)
            
            # 2. RAG Step
            print("\n[*] Initializing RAG pipeline...")
            rag_pipeline = get_rag_pipeline()
            
            print("[*] Running field extraction...")
            rag_result = rag_pipeline.extract_from_ocr(ocr_result.text)
            
            # 3. Save results and Display Details
            print("\n" + "="*70)
            print("RECEIPT RAG EXTRACTION")
            print("="*70)
            
            data = rag_result.extracted_data
            
            # Helper to safely extract values from production format (nested dicts)
            def get_v(k):
                val = data.get(k, 'N/A')
                if isinstance(val, dict) and 'value' in val:
                    return val['value']
                return val

            # 1. Header Information
            print(f"\n[✓] Confidence Score: {rag_result.confidence:.2%}")
            print(f"    Supplier:         {get_v('supplier_name')}")
            print(f"    Address:          {get_v('address')}")
            print(f"    Invoice #:        {get_v('receipt_number')}")
            
            # Handle Date (which can be a list in production format)
            date_raw = get_v('date')
            date_str = date_raw[0].get('value') if isinstance(date_raw, list) and date_raw and isinstance(date_raw[0], dict) else date_raw
            print(f"    Date:             {date_str}")
            
            # 2. Line Items
            print("\n[✓] Extracted Items:")
            items = data.get('items', [])
            if items:
                print(f"    {'QTY':<5} {'ITEM DESCRIPTION':<40} {'PRICE':>10}")
                print(f"    {'-'*5} {'-'*40} {'-'*10}")
                for item in items:
                    if isinstance(item, dict):
                        qty = item.get('quantity', [])
                        qty = qty[0] if isinstance(qty, list) and qty else '1'
                        name = item.get('name', ['N/A'])
                        name = name[0] if isinstance(name, list) and name else 'N/A'
                        price = item.get('total_price', {})
                        price_val = price.get('value', 'N/A') if isinstance(price, dict) else price
                        print(f"    {str(qty):<5} {str(name)[:40]:<40} {str(price_val):>10}")
            else:
                print("    No items extracted.")

            # 3. Financial Totals
            print("\n[✓] Financial Totals:")
            print(f"    Subtotal:         {get_v('net_amount') or get_v('subtotal')}")
            print(f"    Tax/VAT:          {get_v('vat_amount') or get_v('total_tax_amount')}")
            print(f"    TOTAL AMOUNT:     {get_v('total_amount')}")
            
            print("\n" + "-"*70)
            
            # Use our new helper to save JSON and Summary
            base_name = p.stem
            rag_pipeline.save_result_to_file(rag_result, base_name)
            
            print("\n" + "="*70)
            print(f"[✓] Process completed successfully!")
            print(f"[✓] Terminal log: {p.stem}_output.txt")
            print(f"[✓] Data results: {p.stem}_result.json")
            print(f"[✓] Data summary: {p.stem}_summary.txt")
            print("="*70)

        except Exception as e:
            print(f"\n[X] Unexpected error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_receipt_image.py <path_to_image>")
        sys.exit(1)
    
    process_image(sys.argv[1])
