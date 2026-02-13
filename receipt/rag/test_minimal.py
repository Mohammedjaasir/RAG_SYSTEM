import requests
import json

def extract_direct():
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

    prompt = f"""You are a professional receipt data extraction expert.
Extract the following fields from the receipt OCR text provided below:
- supplier_name
- address
- date (YYYY-MM-DD)
- total_amount
- currency
- items (list of name, quantity, total_price)

RECEIPT OCR TEXT:
{invoice_ocr}

Output the result as a raw JSON object. No markdown, no explanations.
"""

    print("[*] Sending request to Ollama (llama3)...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=300
        )
        response.raise_for_status()
        result = response.json().get('response', '')
        print("\n=== EXTRACTION RESULT ===")
        print(result)
        print("==========================")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    extract_direct()
