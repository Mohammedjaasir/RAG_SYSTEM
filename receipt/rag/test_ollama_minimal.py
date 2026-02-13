import requests
import json

def test_minimal():
    url = "http://localhost:11434/api/generate"
    model = "phi3.5"
    
    ocr_text = "Certsure LLP Invoice 123 Total 1.80"
    
    prompt = f"Extract supplier and total from: {ocr_text}. Return JSON."

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    
    print(f"Sending minimal request to Ollama ({model})...")
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        print("SUCCESS: ", response.json().get("response"))
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_minimal()
