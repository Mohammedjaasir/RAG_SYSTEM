import sys
import pathlib

def read_test_output():
    path = pathlib.Path('e:/Downloads/receipt/receipt/rag/akira_output.txt')
    if not path.exists():
        print(f"File {path} not found")
        return
        
    try:
        # PowerShell redirection often creates UTF-16LE files
        content = path.read_text(encoding='utf-16')
        
        # Write to a new file safely
        output_path = pathlib.Path('e:/Downloads/receipt/receipt/rag/akira_cleaned.txt')
        with output_path.open('w', encoding='ascii', errors='ignore') as f:
            f.write(content)
        print(f"Cleaned output written to {output_path}")
        
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    read_test_output()
