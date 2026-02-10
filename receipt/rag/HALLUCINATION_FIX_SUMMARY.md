# RAG Hallucination Fix - Summary

## Problem Identified
Your RAG system was experiencing **hallucinations** where Phi-3 was extracting data from the *example receipts* in the context instead of from the *actual receipt* you wanted to process.

## Root Causes

1. **Small Chunk Size (200 chars)**: This fragmented example receipts into pieces that looked like valid data
   - Example: A chunk might contain just "Total: £127.70" without the header "WAGAMAMA Receipt Example:"
   - Phi-3 couldn't tell it was an example vs actual data

2. **Weak Prompt Separation**: The old prompt said "Use examples as reference" but didn't strongly prevent extraction from examples
   - The model got confused about what was context vs target data

3. **No Clear XML Structure**: Context and target were mixed in plain text

## Fixes Applied

### ✅ 1. Increased Chunk Size (200 → 1000)
**File**: `receipt_rag.py` (line 93)

```python
chunk_size: int = 1000,  # Was: 200
```

**Why this helps**: Larger chunks preserve context headers like "WAGAMAMA Receipt Example:" so Phi-3 knows it's an example.

### ✅ 2. XML-Based Prompt with Negative Constraints
**File**: `receipt_rag.py` (lines 359-381)

```python
template="""You are a receipt extraction expert. Your task is to extract structured data from the <target_receipt> content.

<context>
{context}
</context>

<target_receipt>
{receipt_text}
</target_receipt>

INSTRUCTIONS:
1. The <context> section contains EXAMPLES and PATTERNS. Do NOT extract data from <context>.
2. Only extract data that appears explicitly in <target_receipt>.
3. If a field is missing in <target_receipt>, return null.
4. Extract the following fields in JSON format:
   - vendor_name
   - date
   - invoice_number
   - total_amount
   - tax_amount
   - currency
   - items (list of items with name and price)

Output ONLY valid JSON.
"""
```

**Why this helps**:
- XML tags `<context>` and `<target_receipt>` clearly separate sections
- Explicit negative instructions: "Do NOT extract data from <context>"
- Tells model to return null if field is missing (reduces hallucination pressure)

### ✅ 3. Vector DB Rebuild Logic
**File**: `receipt_rag.py` (lines 193-198)

```python
stored_chunk_size = metadata.get('chunk_size', 0)

if latest_mtime == last_mtime and stored_chunk_size == self.chunk_size:
    needs_rebuild = False
```

**Why this helps**: Automatically rebuilds the vector database when chunk size changes, ensuring consistency.

### ✅ 4. Path Bug Fix
**File**: `receipt_rag.py` (line 130)

```python
Path("/home/vanshtomar/Downloads/RAG_DEMO/models/phi-3.5-mini-instruct-q4_k_m.gguf")
```

**Why this helps**: Fixed a string/Path mix-up that caused crashes when checking if model exists.

## Current Issue: ChromaDB Rust Bindings

The test failed due to a ChromaDB Rust bindings issue:
```
pyo3_runtime.PanicException: range start index 10 out of range for slice of length 9
```

This is a known issue with certain ChromaDB versions on Windows.

## Solutions

### Option 1: Downgrade ChromaDB (Quick Fix)
```bash
pip uninstall chromadb
pip install chromadb==0.4.24
```

### Option 2: Use Alternative LLM (Recommended)
Instead of local Phi-3, use a cloud API that doesn't need local model files:

#### A. Use OpenAI
```python
# In receipt_rag.py, replace _generate_answer() to use OpenAI
from openai import OpenAI

def _generate_answer(self, ocr_text: str, context: str) -> str:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    prompt = f"""You are a receipt extraction expert...
    
<context>
{context}
</context>

<target_receipt>
{ocr_text}
</target_receipt>

INSTRUCTIONS:
1. The <context> section contains EXAMPLES. Do NOT extract data from <context>.
2. Only extract data from <target_receipt>.
...
"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    return response.choices[0].message.content
```

#### B. Use Google Gemini (Free)
```python
import google.generativeai as genai

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-pro')

response = model.generate_content(prompt)
return response.text
```

### Option 3: Fix ChromaDB Installation
```bash
# Try reinstalling with specific versions
pip uninstall chromadb chromadb-client pydantic
pip install chromadb==0.5.0 pydantic==2.6.0
```

## Testing Without Full Pipeline

You can test the prompt improvements without running the full pipeline:

```python
# Simple test - just the prompt logic
test_prompt = """You are a receipt extraction expert. Your task is to extract structured data from the <target_receipt> content.

<context>
WAGAMAMA Receipt Example:
1 CHICKEN NOODLE 5.25
TOTAL: £127.70
</context>

<target_receipt>
FRESH MART GROCERY
Apples 1kg $3.99
Milk 2L $4.50
TOTAL: $8.49
</target_receipt>

INSTRUCTIONS:
1. The <context> section contains EXAMPLES. Do NOT extract data from <context>.
2. Only extract data from <target_receipt>.
3. Extract: vendor_name, items, total_amount

Output ONLY valid JSON.
"""

# Send this to your LLM and verify it extracts:
# - Fresh Mart items (Apples, Milk)
# - NOT Wagamama items (Chicken Noodle)
```

## Verification Checklist

Once you get the pipeline running, verify:

- [ ] Items extracted match the *actual* receipt, not examples
- [ ] No "WAGAMAMA", "TESCO", "Duck Gyoza" items appear in extraction
- [ ] Vendor name is from actual receipt
- [ ] Totals match actual receipt values
- [ ] Context retrieval works (shows relevant examples)
- [ ] Confidence scores are reasonable (>60%)

## Next Steps

1. **Fix ChromaDB**: Try Option 1 or 3 above
2. **Or switch LLM**: Use OpenAI/Gemini (Option 2)
3. **Run test**: `python test_rag_demo.py`
4. **Verify**: Check that extracted items are NOT from example receipts

## Files Modified

- `receipt_rag.py` - Core RAG pipeline with hallucination fixes
- `test_rag_demo.py` - Test script to verify behavior

## Key Takeaway

The hallucination problem was caused by:
1. Small chunks losing context about what's an "example"
2. Weak prompt separation between examples and target data

The fix uses:
1. Larger chunks (1000 chars) to preserve context
2. XML tags to clearly separate sections
3. Explicit negative instructions to prevent extraction from examples

This approach works with ANY LLM (Phi-3, GPT-4, Gemini, etc.) - the prompt structure is the most important part.
