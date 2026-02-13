import requests
import json

def test_api_extraction():
    url = "http://localhost:8000/extract"
    
    # Diverse OCR sample: A Hotel Bill
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
    
    payload = {
        "ocr_text": hotel_ocr,
        "retrieve_k": 2
    }
    
    print(f"[*] Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("\n" + "="*50)
        print("API EXTRACTION RESULTS")
        print("="*50)
        print(f"Supplier: {result.get('supplier_name')}")
        print(f"Date:     {result.get('date') or result.get('receipt_date')}")
        print(f"Total:    {result.get('total_amount')}")
        print(f"VAT #:    {result.get('vat_number')}")
        
        print("\nItems:")
        # Check different possible item keys
        items = result.get('items', result.get('item_list', []))
        if items:
            for item in items:
                name = item.get('name', item.get('description', 'N/A'))
                print(f"  - {name}")
        else:
            print("  No items found.")
            
        print("\nMetadata:")
        print(json.dumps(result.get('_metadata', {}), indent=2))
        
        print("\nFull Result:")
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"❌ Error during API request: {e}")

if __name__ == "__main__":
    test_api_extraction()
