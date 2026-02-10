
import sys

# Write bytes directly to avoid encoding issues
try:
    with open(r'e:\Downloads\receipt\receipt\rag\test_shop_name_result.txt', 'rb') as f:
        content = f.read()
        # Decode manually and replace errors
        text = content.decode('utf-16', errors='replace')
        # Encode to utf-8 for console output
        sys.stdout.buffer.write(text.encode('utf-8'))
except Exception as e:
    print(f"Error: {e}")
