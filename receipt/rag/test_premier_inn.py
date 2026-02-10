#!/usr/bin/env python3
"""
Test RAG Pipeline with Premier Inn Invoice
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

def test_premier_inn_extraction():
    """Test the RAG pipeline with Premier Inn invoice OCR text."""
    
    print("\n" + "="*70)
    print("TESTING RAG PIPELINE WITH PREMIER INN INVOICE")
    print("="*70 + "\n")
    
    # Import here to catch errors
    try:
        from receipt_rag import get_rag_pipeline
    except Exception as e:
        print(f"❌ Failed to import RAG pipeline: {e}")
        return
    
    # Premier Inn invoice OCR text (from the image)
    premier_inn_ocr = """
    Premier Inn Ashford North
    Maidstone Road Hothfield Common, Ashford
    Kent
    TN26 1AP
    
    Invoice
    
    Blan Ten                                Invoice Date:       01 12 2025
    28 Derby Road                           Guest Name:         Blan Ten
    Whitwords                               Room No:            513
    Nottingham                              Departure Date:     02 12 2025
    DE4 4BG                                 Invoice No:         AAFI29713
                                           Folio Number:       3454
                                           Customer Reference:
                                           PO No:              AAF5003454
                                           Arrival Date:       01 12 2025
    
    Date        Description                 Net Amount in   VAT in %   VAT Amount in   Gross Amount in   Amount Received
                                           GBP                         GBP              GBP               in GBP
    01 12 2025  Digital Visit              0.00            0.00        0.00             0.00              54
    01 12 2025  Prepayment (20%            46.00           20%         0.00             54.00
                VAT)
                Accommodation
                
                                           Total                       54.00            54.00
                                           Balance                     0.00
    
    Rate of VAT              Net Amount GBP    VAT Amount GBP    Gross Amount GBP
    Postponed VAT            46.00              0.00              54.00
    Exempt
    Total                    46.00              0.00              54.00
    
    Whitbread Group PLC, Whitbread Court, Houghton Hall Business Park, Porz Avenue, Dunstable LU5 5XE. Registered in
    England number 29423. VAT registration number 204 282 864.
    """
    
    print("INPUT INVOICE:")
    print("-" * 70)
    print(premier_inn_ocr)
    print("-" * 70)
    
    try:
        # Initialize RAG pipeline
        print("\n[*] Initializing RAG pipeline...")
        pipeline = get_rag_pipeline()
        
        # Run extraction
        print("\n[*] Running extraction on Premier Inn invoice...")
        result = pipeline.extract_from_ocr(premier_inn_ocr, retrieve_k=3)
        
        # Display results
        print("\n" + "="*70)
        print("EXTRACTION RESULTS")
        print("="*70)
        
        print(f"\nConfidence: {result.confidence:.2%}")
        
        print("\nExtracted Data:")
        print(json.dumps(result.extracted_data, indent=2))
        
        print(f"\nRelevance Scores: {[f'{s:.3f}' for s in result.relevance_scores]}")
        
        print("\nContext Used (first 500 chars):")
        print("-" * 70)
        context_preview = result.context_used[:500] + "..." if len(result.context_used) > 500 else result.context_used
        print(context_preview)
        print("-" * 70)
        
        # Display detailed extraction
        print("\n" + "="*70)
        print("DETAILED EXTRACTION")
        print("="*70)
        
        extracted_data = result.extracted_data
        
        print(f"\n🏨 Supplier: {extracted_data.get('supplier_name', 'N/A')}")
        print(f"📍 Address: {extracted_data.get('address', 'N/A')}")
        print(f"📄 Invoice #: {extracted_data.get('receipt_number', 'N/A')}")
        print(f"📅 Date: {extracted_data.get('date', 'N/A')}")
        print(f"👤 Guest: {extracted_data.get('customer_name', 'N/A')}")
        print(f"🚪 Room: {extracted_data.get('room_number', 'N/A')}")
        
        print("\n" + "-" * 70)
        print("ITEMS:")
        if 'items' in extracted_data and isinstance(extracted_data['items'], list):
            print(f"Found {len(extracted_data['items'])} items:")
            for item in extracted_data['items']:
                if isinstance(item, dict):
                    print(f"  • {item.get('name', 'N/A')} - {extracted_data.get('currency', 'GBP')} {item.get('total_price', item.get('price', 'N/A'))}")
        else:
            print("No items found")
        
        print("-" * 70)
        print(f"\n💰 Net Total: {extracted_data.get('currency', 'GBP')} {extracted_data.get('subtotal', 'N/A')}")
        print(f"💷 VAT: {extracted_data.get('currency', 'GBP')} {extracted_data.get('total_tax_amount', 'N/A')}")
        print(f"💳 Total: {extracted_data.get('currency', 'GBP')} {extracted_data.get('total_amount', 'N/A')}")
        
        # Hallucination check
        print("\n" + "="*70)
        print("HALLUCINATION CHECK")
        print("="*70)
        
        # Items that should NOT appear (from other receipts in knowledge base)
        hallucination_keywords = [
            'WAGAMAMA', 'TESCO', 'FRESH MART',
            'DUCK', 'RAMEN', 'GYOZA', 'NOODLE',
            'APPLES', 'MILK', 'BREAD'
        ]
        
        hallucinated = False
        full_text = json.dumps(extracted_data).upper()
        
        for keyword in hallucination_keywords:
            if keyword in full_text:
                print(f"[!] POTENTIAL HALLUCINATION: Found '{keyword}' - this should not be in a Premier Inn invoice!")
                hallucinated = True
        
        if not hallucinated:
            print("[✓] No hallucinations detected!")
            print("[✓] All extracted data appears to be from the actual invoice.")
        
        print("\n" + "="*70)
        print("[+] Test completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n[X] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_premier_inn_extraction()
