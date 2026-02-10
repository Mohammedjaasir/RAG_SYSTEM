import sys
import os
from pathlib import Path

# Try to replicate the environment of the test script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
print(f"sys.path: {sys.path}")

try:
    print("\nAttempting to import receipt.rag.receipt_rag...")
    from receipt.rag.receipt_rag import get_rag_pipeline
    print("Import success!")
    
    print("\nAttempting to initialize RAG pipeline (this triggers lazy imports)...")
    pipeline = get_rag_pipeline()
    pipeline._initialize()
    print("Initialization success!")
except ImportError as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()

print("\nListing all modules that contain 'config':")
for name in sys.modules:
    if 'config' in name.lower():
        try:
            print(f"  {name}: {sys.modules[name].__file__}")
        except:
            print(f"  {name}: <built-in or no __file__>")
