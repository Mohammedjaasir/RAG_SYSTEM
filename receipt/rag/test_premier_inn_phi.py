#!/usr/bin/env python3
"""
Test RAG Pipeline with OCR text transcribed from the user's Premier Inn image.
Uses phi3.5 model.
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
        print("PROCESSING PREMIER INN INVOICE (TRANSCRIPTION FROM IMAGE) - USING PHI3.5")
        print("="*70 + "\n")
        
        # Use a temporary database path
        temp_db = Path(__file__).parent / "temp_chroma_db_phi"
        if temp_db.exists():
            import shutil
            shutil.rmtree(temp_db)
            
        # Explicitly passing model_name="phi3.5" (redundant since it's default now)
        pipeline = get_rag_pipeline(persist_directory=str(temp_db), model_name="phi3.5")
        
        print(f"\n[*] Running extraction using model: phi3.5...")
        result = pipeline.extract_from_ocr(invoice_ocr, retrieve_k=3)
        
        # Display results with the requested header and all details
        print("\n" + "="*70)
        print("RECEIPT RAG EXTRACTION RESULTS (PHI3.5)")
        print("="*70)
        
        data = result.extracted_data
        
        # Helper to safely extract values from production format (nested dicts)
        def get_v(k):
            val = data.get(k, 'N/A')
            if isinstance(val, dict) and 'value' in val:
                return val['value']
            return val

        # 1. Header Information
        print(f"\n[✓] Confidence Score: {result.confidence:.2%}")
        
        # Hallucination Check - Terminal Display
        if result.hallucination_report:
            print("\n" + "!"*70)
            print("  ⚠️  HALLUCINATION WARNING")
            for warning in result.hallucination_report:
                print(f"  [X] {warning}")
            print("!"*70)
        else:
            print("\n[✓] Hallucination Check: PASSED (Data is grounded in OCR text)")
            
        print(f"\n    Supplier:         {get_v('supplier_name')}")
        print(f"    Address:          {get_v('address')}")
        print(f"    Invoice #:        {get_v('receipt_number')}")
        
        # Handle Date
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
        
        # Save results
        pipeline.save_result_to_file(result, "premier_inn_phi")
        
        print("\n" + "="*70)
        print("[+] Process completed successfully!")
        print("="*70)

if __name__ == "__main__":
    test_premier_inn_invoice_extraction()
