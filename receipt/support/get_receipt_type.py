import fitz  # PyMuPDF
from PIL import Image
import os

def get_receipt_type(file_path):
    """
    Determines if a file should be processed as a 'Vertical' receipt 
    or a 'Full Page/Table' receipt (like Ebay).
    Returns: 'VERTICAL' or 'FULL_PAGE'
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        with fitz.open(file_path) as doc:
            # 1. Multi-page Check (Ebay is often multi-page)
            if len(doc) > 1:
                print("Detected multi-page PDF, treating as FULL_PAGE receipt.")
                return 'FULL_PAGE'

            # 2. Analyze First Page Dimensions & Content
            page = doc[0]
            width, height = page.rect.width, page.rect.height
            text = page.get_text().lower()

            # 3. Dimension Check: Is it a narrow "tape" receipt?
            # Standard A4/Letter width is ~600 points. 
            # Thermal receipts are usually < 300 points (unless scanned full A4).
            if width < 400: 
                print("Detected narrow width, treating as VERTICAL receipt.")
                return 'VERTICAL'

            # 4. Keyword Check: Does it look like an Invoice/Digital Receipt?
            # Ebay/Amazon receipts almost always have these distinct column headers.
            table_keywords = ['item title', 'description', 'quantity', 'qty', 'unit price']
            if any(keyword in text for keyword in table_keywords):
                print("Detected table keywords, treating as FULL_PAGE receipt.")
                return 'FULL_PAGE'
            
            # Fallback: If it's A4 but has no invoice keywords, treat as vertical/image scan
            print("Fallback: Treating as VERTICAL receipt.")
            return 'VERTICAL'

    elif ext in ['.png', '.jpg', '.jpeg', '.tiff']:
        # Images are almost always vertical tape receipts in your workflow
        # You can add the aspect ratio check here if needed
        print("Image file detected, treating as VERTICAL receipt.")
        return 'VERTICAL'

    print("Unknown file type, treating as VERTICAL receipt.")
    return 'VERTICAL' # Default safe fallback

if __name__ == "__main__":
    test_file = "./data/5095.png"
    receipt_type = get_receipt_type(test_file)
    print(f"Receipt Type: {receipt_type}")