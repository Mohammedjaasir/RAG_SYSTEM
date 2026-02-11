#!/usr/bin/env python3
"""
Test RAG Pipeline - Check for Hallucinations
"""

import os
import sys
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

def test_rag_extraction():
    """Test the RAG pipeline with a simple receipt."""
    
    print("\n" + "="*70)
    print("TESTING RAG PIPELINE FOR HALLUCINATIONS")
    print("="*70 + "\n")
    
    # Import here to catch errors
    try:
        from receipt_rag import get_rag_pipeline
    except Exception as e:
        print(f"❌ Failed to import RAG pipeline: {e}")
        return
    
    # Create a test receipt that is DIFFERENT from the examples
    # This receipt is deliberately simple to test if the model hallucinates
    # items from the context examples
    test_receipt = """
    FRESH MART GROCERY
    456 OAK STREET
    CHICAGO, IL 60601
    
    Date: 02/09/2026
    Time: 14:25
    Receipt: FM-2026-00789
    
    Apples 1kg          $3.99
    Milk 2L             $4.50
    Bread               $2.25
    
    SUBTOTAL:          $10.74
    TAX (7%):           $0.75
    TOTAL:             $11.49
    
    VISA ending 5678
    """
    
    print("INPUT RECEIPT:")
    print("-" * 70)
    print(test_receipt)
    print("-" * 70)
    
    with setup_output_capture(__file__):
        try:
            # Initialize RAG pipeline
            print("\n[*] Initializing RAG pipeline...")
            pipeline = get_rag_pipeline()
            
            # Run extraction
            print("\n[*] Running extraction...")
            result = pipeline.extract_from_ocr(test_receipt, retrieve_k=3)
            
            # Display results
            print("\n" + "="*70)
            print("EXTRACTION RESULTS")
            print("="*70)
            
            print(f"\nConfidence: {result.confidence:.2%}")
            
            print("\nExtracted Data:")
            import json
            print(json.dumps(result.extracted_data, indent=2))
            
            print(f"\nRelevance Scores: {[f'{s:.3f}' for s in result.relevance_scores]}")
            
            print("\nContext Used (first 500 chars):")
            print("-" * 70)
            context_preview = result.context_used[:500] + "..." if len(result.context_used) > 500 else result.context_used
            print(context_preview)
            print("-" * 70)
            
            # Save results using helper
            pipeline.save_result_to_file(result, "rag_demo")
            
            # Check for hallucinations
            print("\nHALLUCINATION CHECK:")
            print("-" * 70)
            
            extracted_data = result.extracted_data
            
            # Display detailed extraction
            print("\nDetailed Extraction:")
            print(f"Vendor: {extracted_data.get('supplier_name')}")
            print(f"Address: {extracted_data.get('address')}")
            print(f"Receipt #: {extracted_data.get('receipt_number')}")
            print(f"Date/Time: {extracted_data.get('date')} {extracted_data.get('time')}")
            print("-" * 30)
            
            if 'items' in extracted_data and isinstance(extracted_data['items'], list):
                print(f"Items found: {len(extracted_data['items'])}")
                for item in extracted_data['items']:
                    if isinstance(item, dict):
                        print(f"  - {item.get('name', 'N/A')} (Qty: {item.get('quantity', 1)}) : {extracted_data.get('currency', '')}{item.get('total_price', item.get('price', 'N/A'))}")
            
            print("-" * 30)
            print(f"Subtotal: {extracted_data.get('subtotal')}")
            print(f"Tax: {extracted_data.get('total_tax_amount')}")
            print(f"Total: {extracted_data.get('total_amount')}")
            print(f"Payment: {extracted_data.get('payment_info')}")
            
            # Check if extracted items match input
            if 'items' in extracted_data and isinstance(extracted_data['items'], list):
                print(f"\n[+] Found {len(extracted_data['items'])} items in extraction")
                
                # Expected items
                expected_items = ['Apples', 'Milk', 'Bread']
                hallucination_items = ['WAGAMAMA', 'TESCO', 'Chicken', 'Noodle', 'Duck', 'Gyoza']
                
                # Check for hallucinations from context examples
                hallucinated = False
                for item in extracted_data['items']:
                    item_name = str(item.get('name', '')).upper()
                    for bad_item in hallucination_items:
                        if bad_item.upper() in item_name:
                            print(f"\n[!] HALLUCINATION DETECTED: Found '{item.get('name')}' - this is from context examples!")
                            hallucinated = True
                
                if not hallucinated:
                    print("\n[+] No hallucinations detected! Items appear to be from the actual receipt.")
            else:
                print("[!] No items found in extraction")
            
            print("-" * 70)
            
            print("\n[+] Test completed successfully!")
            
        except FileNotFoundError as e:
            print(f"\n[X] Error: {e}")
            print("\nTo fix this:")
            print("   1. Download Phi-3 model (.gguf file)")
            print("   2. Update model_path in receipt_rag.py or set it explicitly")
            print("   3. Or use a different LLM (OpenAI, etc.)")
            
        except Exception as e:
            print(f"\n[X] Unexpected error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_rag_extraction()
