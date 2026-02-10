#!/usr/bin/env python3
"""
Test RAG Pipeline with 'Shop Name' Generic Receipt
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

def test_shop_name_extraction():
    """Test the RAG pipeline with the generic 'Shop Name' receipt OCR text."""
    
    print("\n" + "="*70)
    print("TESTING RAG PIPELINE WITH 'SHOP NAME' RECEIPT")
    print("="*70 + "\n")
    
    # Import here to catch errors
    try:
        from receipt_rag import get_rag_pipeline
    except Exception as e:
        print(f"❌ Failed to import RAG pipeline: {e}")
        return
    
    # OCR text manually transcribed from the provided image
    receipt_ocr = """
    SHOP NAME
    Address: Lorem Ipsum, 23-10
    Telp. 11223344
    * * * * * * * * * * * * * * * * * * * *
    CASH RECEIPT
    * * * * * * * * * * * * * * * * * * * *
    Description           Price
    Lorem                   1.1
    Ipsum                   2.2
    Dolor sit amet          3.3
    Consectetur             4.4
    Adipiscing elit         5.5
    * * * * * * * * * * * * * * * * * * * *
    Total                  16.5
    Cash                   20.0
    Change                  3.5
    * * * * * * * * * * * * * * * * * * * *
    Bank card       ... - - - 234
    Approval Code        #123456
    * * * * * * * * * * * * * * * * * * * *
    THANK YOU!
    ||||||| | || ||||| |||| || |||
    """
    
    print("INPUT RECEIPT:")
    print("-" * 70)
    print(receipt_ocr)
    print("-" * 70)
    
    try:
        # Initialize RAG pipeline
        print("\n[*] Initializing RAG pipeline...")
        pipeline = get_rag_pipeline()
        
        # Run extraction
        print("\n[*] Running extraction on Shop Name receipt...")
        result = pipeline.extract_from_ocr(receipt_ocr, retrieve_k=3)
        
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
        
        print("\n" + "-" * 70)
        print("ITEMS:")
        if 'items' in extracted_data and isinstance(extracted_data['items'], list):
            print(f"Found {len(extracted_data['items'])} items:")
            for item in extracted_data['items']:
                if isinstance(item, dict):
                    print(f"  • {item.get('description', item.get('name', 'N/A'))} - {extracted_data.get('currency', 'GBP')} {item.get('total_price', item.get('price', 'N/A'))}")
        else:
            print("No items found")
        
        print("-" * 70)
        print(f"\n💰 Net Total: {extracted_data.get('currency', 'GBP')} {extracted_data.get('subtotal', 'N/A')}")
        print(f"💷 VAT: {extracted_data.get('currency', 'GBP')} {extracted_data.get('total_tax_amount', 'N/A')}")
        print(f"💳 Total: {extracted_data.get('currency', 'GBP')} {extracted_data.get('total_amount', 'N/A')}")
        
        print("\n" + "="*70)
        print("[+] Test completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n[X] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_shop_name_extraction()
