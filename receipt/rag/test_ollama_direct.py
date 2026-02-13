import requests
import json

def test_extraction():
    url = "http://localhost:11434/api/generate"
    model = "phi3.5"
    
    ocr_text = """Certsure LLP Warwick House Houghton Hall Park Houghton Regis DUNSTABLE LU55 5ZX Page 1of1 Invoice 5Certsure E Bill-To-Party VAT REG No. GB155441713 Information Invoice Number 88041459 Date of Issue Reference Order 2907959/ 15.07.2024 Account No. Currency Payment Terms 28th of this month by DD Description AN Moor Electrical Ltd 5K Kevlyn Cresent Old Netley Southampton SO31 8EX 12.07.2024 3036273 GBP Submissions madel between 09/06/2024 and 08/07/2024 Ship-To-Party Al Moor Electrical Ltd 5K Kevlyn Cresent Old Netley Southampton SO31 8EX Item Material/Description 10 CNOCEICR18.2C Quantity 1PC Unit Price 1.50 Value 1.50 AC-Electrical Install Condition 18.2 Sub-Total VAT-Total 1.50 0.30 1.80 Invoice Amount VAT Summary VAT Rate Goods 1.50 VAT 0.30 Standard rated output' VAT: 20% Installment Plan (Direct Debits On Or Shortly After) DD No. 1 Date 28.07.2024 Amount Method 1.80 Direct Debit This invoice is for information purposes only. The amount due willl be collected by Direct Debit on or immediately after the dates detailed above. NICEICi isa at tradingb brand of Certsurel LLP, al Limited Liability Partnership. Registered in Englanda andV Wales withr registered number OC379918. Registered office andp principal place ofb businessi is: Warwick House, Houghton HallF Park, Houghton Regis, Dunstable, LU5 5ZX."""
    
    prompt = f"""You are a professional receipt data extraction expert. 
Your task is to extract structured information from the <target_receipt> provided below.

<target_receipt>
ACTUAL OCR TEXT TO EXTRACT FROM:
{ocr_text}
</target_receipt>

INSTRUCTIONS:
1. **Missing Data**: If a field is not present, return `null`.
2. **Format**: Output your results as a SINGLE valid JSON object.
3. **Fields**:
   - supplier_name
   - address
   - date (YYYY-MM-DD)
   - total_amount (float)
   - vat_amount (float)
   - items (list of: name, quantity, total_price)

Output ONLY the JSON object. No preamble, no markdown formatting.
"""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    
    print(f"Sending request to Ollama ({model})...")
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        print("\nSUCCESS!")
        print("Response:")
        print(result.get("response", ""))
    except Exception as e:
        print(f"\nFAILED: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            print(f"Error Body: {e.response.text}")

if __name__ == "__main__":
    test_extraction()
