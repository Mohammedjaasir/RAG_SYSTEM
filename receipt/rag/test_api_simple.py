import requests
import json

def test_api_simple():
    url = "http://localhost:8000/extract"
    simple_ocr = "TESCO\n2023-12-01\nTOTAL £10.00"
    
    payload = {"ocr_text": simple_ocr, "retrieve_k": 1}
    
    print(f"[*] Sending simple request to {url}...")
    try:
        response = requests.post(url, json=payload, timeout=120)
        print(f"Response: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api_simple()
