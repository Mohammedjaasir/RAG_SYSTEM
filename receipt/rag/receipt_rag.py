#!/usr/bin/env python3
"""
Receipt RAG Pipeline - Simplified Demo-Based Implementation

Based on the RAG demo implementation, this module provides:
- Receipt knowledge base loading from text files
- Document chunking and embedding
- Similarity search for relevant receipt patterns
- Phi-3 LLM generation for receipt field extraction

This replaces the complex enterprise RAG system with a clean, maintainable approach.
"""

import os
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_langchain_imports = None


def _get_langchain():
    """Lazy import LangChain components."""
    global _langchain_imports
    if _langchain_imports is None:
        try:
            from langchain_community.document_loaders import TextLoader, DirectoryLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma
            from langchain_core.prompts import PromptTemplate
            
            _langchain_imports = {
                'TextLoader': TextLoader,
                'DirectoryLoader': DirectoryLoader,
                'RecursiveCharacterTextSplitter': RecursiveCharacterTextSplitter,
                'HuggingFaceEmbeddings': HuggingFaceEmbeddings,
                'Chroma': Chroma,
                'PromptTemplate': PromptTemplate
            }
            logger.info("LangChain components imported successfully")
        except ImportError as e:
            logger.error(f"LangChain not installed: {e}")
            logger.error("Run: pip install langchain-community langchain-core langchain-text-splitters")
            raise
    return _langchain_imports


@dataclass
class RAGResult:
    """Result from RAG pipeline."""
    extracted_data: Dict[str, Any]
    confidence: float
    context_used: str
    raw_llm_response: str
    relevance_scores: List[float]
    hallucination_report: List[str] = None


class ReceiptRAGPipeline:
    """
    Simplified RAG pipeline for receipt extraction.
    
    Based on RAG demo: Load  Split  Embed  Retrieve  Generate
    """
    
    def __init__(
        self,
        knowledge_base_dir: Optional[str] = None,
        persist_directory: Optional[str] = None,
        ollama_url: str = "http://localhost:11434",
        model_name: str = "phi3.5",
        chunk_size: int = 1000,
        chunk_overlap: int = 40,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            knowledge_base_dir: Directory containing receipt example text files
            persist_directory: Custom directory for ChromaDB persistence
            ollama_url: URL of Ollama service (default: http://localhost:11434)
            model_name: Ollama model name (default: phi3.5)
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            embedding_model: HuggingFace embedding model name
        """
        # Setup paths
        if knowledge_base_dir is None:
            knowledge_base_dir = Path(__file__).parent / "knowledge_base" / "receipts"
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        
        # ChromaDB persistence
        if persist_directory:
            self.persist_dir = Path(persist_directory)
        else:
            self.persist_dir = Path(__file__).parent / "chroma_db"
            
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.persist_dir / "metadata.json"
        
        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model
        
        # Ollama configuration
        self.ollama_url = ollama_url
        self.model_name = model_name
        
        # Initialize components (lazy)
        self.vector_db = None
        self.embeddings = None
        self.initialized = False
        
        logger.info(f" Receipt RAG Pipeline initialized")
        logger.info(f"   Knowledge base: {self.knowledge_base_dir}")
        logger.info(f"   Vector DB: {self.persist_dir}")
        logger.info(f"   Ollama: {self.ollama_url}, Model: {self.model_name}")
    
    def _initialize(self):
        """Initialize the RAG components (called on first use)."""
        if self.initialized:
            return
        
        logger.info(" Initializing RAG components...")
        
        # Import LangChain
        lc = _get_langchain()
        
        # Initialize embeddings
        logger.info(f" Loading embedding model: {self.embedding_model_name}")
        self.embeddings = lc['HuggingFaceEmbeddings'](model_name=self.embedding_model_name)
        logger.info(" Embeddings loaded")
        
        # Initialize or load vector database
        self._setup_vector_db(lc)
        
        # Initialize LLM (lazy - only when needed for generation)
        # self.llm will be initialized in _generate_answer()
        
        self.initialized = True
        logger.info("RAG pipeline ready")
    
    def _setup_vector_db(self, lc):
        """Setup vector database with smart caching (like demo)."""
        # Check if knowledge base files exist
        txt_files = list(self.knowledge_base_dir.glob("*.txt"))
        if not txt_files:
            logger.warning(f" No .txt files found in {self.knowledge_base_dir}")
            logger.warning("   Creating empty vector database")
            self.vector_db = lc['Chroma'](
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings
            )
            return
        
        # Get latest modification time of all source files
        latest_mtime = max(f.stat().st_mtime for f in txt_files)
        
        # Check if rebuild is needed
        needs_rebuild = True
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                last_mtime = metadata.get('last_modified', 0)
                
                stored_chunk_size = metadata.get('chunk_size', 0)
                
                if latest_mtime == last_mtime and stored_chunk_size == self.chunk_size:
                    needs_rebuild = False
                    logger.info(" Knowledge base unchanged, loading existing database")
                else:
                    logger.info(" Knowledge base changed, rebuilding database")
                    if self.persist_dir.exists():
                        # Clean up old database
                        for item in self.persist_dir.iterdir():
                            if item.is_file() and item.name != 'metadata.json':
                                item.unlink()
                            elif item.is_dir():
                                shutil.rmtree(item)
        
        if needs_rebuild:
            logger.info(" Loading receipt knowledge base...")
            
            # Load all text files
            loader = lc['DirectoryLoader'](
                str(self.knowledge_base_dir),
                glob="*.txt",
                loader_cls=lc['TextLoader']
            )
            docs = loader.load()
            logger.info(f" Loaded {len(docs)} document(s)")
            
            # Split documents
            logger.info(" Splitting documents into chunks...")
            splitter = lc['RecursiveCharacterTextSplitter'](
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = splitter.split_documents(docs)
            logger.info(f" Created {len(chunks)} chunks")
            
            # Create vector database
            logger.info(" Creating embeddings (this may take a moment)...")
            try:
                self.vector_db = lc['Chroma'].from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    persist_directory=str(self.persist_dir)
                )
            except Exception as e:
                logger.error(f" Failed to create vector database: {e}")
                logger.warning(" Continuing without knowledge base context")
                self.vector_db = None
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump({
                    'last_modified': latest_mtime,
                    'knowledge_base_dir': str(self.knowledge_base_dir),
                    'embedded_at': datetime.now().isoformat(),
                    'chunk_count': len(chunks),
                    'document_count': len(docs),
                    'chunk_size': self.chunk_size,
                    'chunk_overlap': self.chunk_overlap
                }, f, indent=2)
            
            logger.info(f" Vector DB ready with {self.vector_db._collection.count()} chunks")
            # Load existing database
            try:
                self.vector_db = lc['Chroma'](
                    persist_directory=str(self.persist_dir),
                    embedding_function=self.embeddings
                )
                logger.info(f" Loaded existing vector DB with {self.vector_db._collection.count()} chunks")
            except Exception as e:
                logger.error(f" Failed to load vector database: {e}")
                logger.warning(" Continuing without knowledge base context")
                self.vector_db = None
    
    def clean_database(self):
        """Clear the vector database and metadata to force a rebuild."""
        logger.info(" Cleaning vector database...")
        if self.persist_dir.exists():
            for item in self.persist_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            logger.info(" Database directory cleared")
        
        self.initialized = False
        self.vector_db = None
        logger.info(" Database state reset. It will be rebuilt on next use.")
    
    def retrieve_context(
        self,
        ocr_text: str,
        k: int = 3
    ) -> Dict[str, Any]:
        """
        Retrieve relevant context from knowledge base.
        
        Args:
            ocr_text: OCR text from receipt
            k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary with retrieved documents and metadata
        """
        self._initialize()
        
        if self.vector_db is None or self.vector_db._collection.count() == 0:
            logger.warning(" Vector database is empty, no context available")
            return {
                'documents': [],
                'context': '',
                'relevance_scores': []
            }
        
        logger.info(f" Retrieving relevant context (k={k})...")
        logger.info(f"   Query text: {ocr_text[:200]}..." if len(ocr_text) > 200 else f"   Query text: {ocr_text}")
        
        # Similarity search
        retrieved_docs = self.vector_db.similarity_search_with_score(ocr_text, k=k)
        
        logger.info(f" Retrieved {len(retrieved_docs)} relevant chunks")
        
        documents = []
        context_parts = []
        relevance_scores = []
        
        for i, (doc, score) in enumerate(retrieved_docs):
            distance = score
            relevance = 1.0 - distance  # Convert distance to relevance
            
            logger.info(f"    Chunk #{i+1}:")
            logger.info(f"       Distance: {distance:.4f}")
            logger.info(f"       Relevance: {relevance:.4f}")
            logger.info(f"       Content: {doc.page_content[:100]}...")
            
            documents.append({
                'content': doc.page_content,
                'metadata': doc.metadata,
                'relevance': relevance
            })
            context_parts.append(doc.page_content)
            relevance_scores.append(relevance)
        
        # Combine context
        context = "\n\n".join(context_parts)
        
        return {
            'documents': documents,
            'context': context,
            'relevance_scores': relevance_scores
        }
    
    def _clean_json_output(self, text: str) -> str:
        """Clean LLM output to ensure valid JSON."""
        import re
        
        # Remove markdown code blocks
        text = text.strip()
        if "```" in text:
            # Extract content between first and last ``` calls
            # Try to find a block starting with ```json first
            json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if json_match:
                text = json_match.group(1)
            else:
                # Fallback to any code block
                parts = text.split("```")
                for part in parts:
                    if "{" in part and "}" in part:
                        text = part
                        break
        
        # Remove any leading/trailing text outside the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            text = text[start:end+1]
        
        # Remove inline comments (// ...) and block comments (/* ... */)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        text = re.sub(r'(?<!:)\/\/.*', '', text)
        
        # Remove trailing commas before closing braces/brackets
        text = re.sub(r',(\s*[]}])', r'\1', text)
        
        # Fix missing quotes on keys (basic case)
        text = re.sub(r'(\s*)(\w+)(\s*):', r'\1"\2"\3:', text)
        
        # Ensure single quotes are replaced by double quotes in JSON
        # Be careful not to replace quotes inside strings
        # This is a very simple attempt, json_repair handles this better
        
        return text.strip()

    def _normalize_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize extracted fields (dates, amounts, etc.)."""
        import re

        # Fields that are ONLY valid inside the items list, never at top level
        _item_only_keys = {'quantity', 'qty', 'unit_price'}

        normalized = {}
        for key, value in data.items():
            # Drop stray item-column fields that leaked to top-level
            if key in _item_only_keys:
                continue

            # Already normalized or confidence field
            if key.endswith("_confidence"):
                normalized[key] = value
                continue

            # Normalize dates
            if key in ['date', 'receipt_date']:
                normalized[key] = self._normalize_date(value)
            # Normalize monetary amounts (but NOT generic 'total_price' at top-level — item field)
            elif key == 'total_price':
                # top-level total_price is a leaked item field — skip it
                continue
            elif any(k in key for k in ['amount', 'subtotal', 'tax', 'vat']) or key == 'total_amount':
                normalized[key] = self._normalize_amount(value)
            elif key == 'receipt_number':
                # Sanitize: receipt_number should be a short alphanumeric ID.
                # Wipe only if clearly contaminated with item/table data.
                val_str = str(value) if value is not None else ''
                dirty = (
                    len(val_str) > 120   # too long for any real receipt/invoice number
                    or any(tok in val_str.upper() for tok in ['ITEM', 'QTY', 'PRICE', 'AMUUNT', 'AMOUNT'])
                    or (re.search(r'[\[\{]', val_str) and len(val_str) > 30)  # list/dict AND long
                )
                normalized[key] = None if dirty else value
            elif key == 'items':
                normalized[key] = self._normalize_items(value)
            else:
                normalized[key] = value
        return normalized

    def _normalize_items(self, items_raw) -> list:
        """
        Normalize the 'items' field into a consistent list of dicts:
          [{"name": str, "quantity": str|int, "total_price": float|str}, ...]

        Handles every shape the LLM produces:
          - list of dicts  : [{"name":..., "quantity":..., "total_price":...}]
          - list of lists  : [[name, qty, price], [name, price], ...]
          - list of strings: ["Item 1 $5.00", ...]
          - None / non-list: returns []
        """
        import re
        if not items_raw or not isinstance(items_raw, list):
            return []

        normalised = []
        for item in items_raw:
            if isinstance(item, dict):
                # Standard dict format — standardise key names
                name = item.get('name') or item.get('description') or item.get('item') or 'N/A'
                if isinstance(name, list):
                    name = name[0] if name else 'N/A'
                qty = item.get('quantity') or item.get('qty') or '1'
                price = (
                    item.get('total_price')
                    or item.get('price')
                    or item.get('amount')
                    or item.get('unit_price')
                )
                if isinstance(price, dict):
                    price = price.get('value', price)
                normalised.append({'name': str(name), 'quantity': qty, 'total_price': price})

            elif isinstance(item, (list, tuple)):
                # [name, qty, price] or [name, price] or [name, {dict}]
                parts = list(item)
                if not parts:
                    continue
                name = str(parts[0]) if parts else 'N/A'
                # Skip items whose "name" is actually a dict (malformed LLM output)
                if parts[0] and isinstance(parts[0], dict):
                    continue
                # Find price: last numeric-looking entry
                price = None
                qty = '1'
                for part in reversed(parts[1:]):
                    if isinstance(part, (int, float)):
                        price = part
                        break
                    if isinstance(part, str):
                        cleaned = re.sub(r'[^\d.]', '', part)
                        if cleaned:
                            try:
                                price = float(cleaned)
                                break
                            except ValueError:
                                pass
                    if isinstance(part, dict):
                        price = part.get('total_price') or part.get('price') or part.get('value')
                        qty = part.get('quantity') or part.get('qty') or '1'
                        break
                if len(parts) >= 3 and not isinstance(parts[1], dict):
                    qty = parts[1]
                normalised.append({'name': name, 'quantity': qty, 'total_price': price})

            elif isinstance(item, str) and item.strip():
                # Plain string like "Masala Dosa 2 x 59.00"
                normalised.append({'name': item.strip(), 'quantity': '1', 'total_price': None})

        # Filter out entries that are clearly metadata rows (GST, dates, ref numbers)
        skip_names = {'gst', 'cgst', 'sgst', 'vat', 'tax', 'total', 'subtotal',
                      'grand total', 'discount', 'service charge'}
        normalised = [
            e for e in normalised
            if str(e.get('name', '')).strip().lower() not in skip_names
            and len(str(e.get('name', '')).strip()) > 1
        ]
        return normalised

    def _ocr_fallback_items(self, ocr_text: str) -> list:
        """
        Deterministic (regex-only) item extractor for use when the LLM
        fails to produce an items list.

        This handles two common OCR scrambling layouts:
          A) Interleaved:  "Masala Dosa  2  59.00  118.00  Vada  1  25.00  25.00"
          B) Separated:    "2 59.00 118.00  1 69.00 69.00  Masala Dosa  Vada"

        Strategy:
          1. Isolate the items block (between ITEM/QTY/PRICE header and TOTAL footer).
          2. Separate the block into a numeric zone and a name zone.
          3. In the numeric zone, extract qty+unit+total triplets.
          4. In the name zone, split on capitalised-word boundaries to recover names.
          5. Zip triplets with names and validate prices against the raw OCR.
        """
        import re

        # ── 1. Isolate the items block ──────────────────────────────────────────
        text = ' '.join(ocr_text.split())   # flatten

        # Header: the ITEM/QTY/PRICE/AMUUNT header row
        header_pat = re.compile(r'\b(?:ITEM|QTY|QUANTITY|PRICE|AMUUNT|AMOUNT)\b')
        # Footer: first TOTAL / CGST / SGST / TAX / GRAND after items
        footer_pat = re.compile(r'\b(?:SUBTOTAL|GRAND TOTAL|TOTAL|CGST|SGST|GST|VAT|TAX|TOKEN)\b',
                                re.I)

        section = text
        hm = header_pat.search(text)
        if hm:
            # skip all consecutive header words
            rest = text[hm.start():]
            # skip every consecutive QTY/ITEM/PRICE/AMUUNT word
            head_skip = re.compile(r'^(?:ITEM|QTY|QUANTITY|PRICE|AMUUNT|AMOUNT)\s*', re.I)
            while head_skip.match(rest):
                rest = head_skip.sub('', rest, count=1)
            section = rest

        fm = footer_pat.search(section)
        if fm:
            section = section[:fm.start()]

        section = section.strip()
        if not section:
            return []

        # ── 2. Determine layout: split section into numeric zone + name zone ────
        # A number token — integer or decimal
        num_tok = re.compile(r'^\d+(?:[.,]\d+)?$')
        tokens = section.split()

        # Locate the boundary where tokens switch from mostly-numeric to mostly-alpha
        # Find the last consecutive run of numeric tokens
        last_num_idx = -1
        for idx, tok in enumerate(tokens):
            if num_tok.match(tok):
                last_num_idx = idx

        if last_num_idx == -1:
            # No numbers found — cannot extract
            return []

        num_tokens  = [t for t in tokens[:last_num_idx + 1] if num_tok.match(t)]
        name_tokens = [t for t in tokens[last_num_idx + 1:] if not num_tok.match(t)]

        # ── 3. Extract numeric triplets [qty, unit_price, total_price] ──────────
        nums = []
        for t in num_tokens:
            try:
                nums.append(float(t.replace(',', '.')))
            except ValueError:
                pass

        if not nums:
            return []

        # ── 3. Extract triplets using qty-marker strategy ────────────────────────
        # A "qty marker" is a number that is a small whole integer (1-20) that
        # precedes two price values. We scan left-to-right building triplets.
        triplets = []

        def is_qty(n: float) -> bool:
            """Small whole integer most likely to be a quantity."""
            return n == int(n) and 1 <= int(n) <= 20

        i = 0
        while i < len(nums):
            if is_qty(nums[i]) and i + 2 < len(nums):
                q  = int(nums[i])
                u  = nums[i + 1]
                tp = nums[i + 2]
                # Sanity: unit_price and total_price should be positive decimals > quantity
                if u > 0 and tp > 0:
                    triplets.append({'quantity': q, 'unit_price': u, 'total_price': tp})
                    i += 3
                    continue
            # Fallback: treat as (unit_price, total_price) pair, qty=1
            if i + 1 < len(nums):
                triplets.append({'quantity': 1, 'unit_price': nums[i], 'total_price': nums[i+1]})
                i += 2
            else:
                # Lone number — single-item with no unit price
                triplets.append({'quantity': 1, 'unit_price': None, 'total_price': nums[i]})
                i += 1

        # ── 4. Split name blob into individual names ─────────────────────────────
        # Split on Capital-letter word boundaries: "MasalaDosa VadaPav" → items
        # Also try splitting every N words based on triplet count
        full_name_str = ' '.join(name_tokens)
        n_items = len(triplets)

        if n_items > 0 and len(name_tokens) >= n_items:
            words_per_name = len(name_tokens) // n_items
            names = []
            for i in range(n_items):
                start = i * words_per_name
                end = start + words_per_name if i < n_items - 1 else len(name_tokens)
                names.append(' '.join(name_tokens[start:end]))
        else:
            names = [full_name_str] * n_items

        # ── 5. Zip and ground-check ───────────────────────────────────────────────
        items = []
        for i, triplet in enumerate(triplets):
            name = names[i] if i < len(names) else 'Unknown'
            price_str = f"{triplet['total_price']:g}"
            # Only include if the price is actually in the raw OCR text
            if price_str in ocr_text or price_str.replace('.', ',') in ocr_text:
                items.append({
                    'name': name.strip(),
                    'quantity': triplet['quantity'],
                    'unit_price': triplet['unit_price'],
                    'total_price': triplet['total_price'],
                })
            else:
                logger.warning(
                    f" OCR fallback: skipping ungrounded price {price_str} for '{name}'"
                )

        return items


    def _normalize_date(self, date_str: Any) -> Optional[str]:
        """Normalize date strings to YYYY-MM-DD."""
        if not isinstance(date_str, str) or not date_str:
            return date_str

        import re
        from dateutil import parser

        # Remove common receipt artifacts
        date_str = re.sub(r'[^\w\s\-/.]', '', date_str).strip()

        try:
            # Try parsing with dateutil - prioritize year first for robustness
            dt = parser.parse(date_str, fuzzy=True, yearfirst=True)
            return dt.strftime('%Y-%m-%d')
        except:
            return date_str

    def _normalize_amount(self, amount: Any) -> Any:
        """Normalize amount to float."""
        if amount is None or isinstance(amount, (int, float)):
            return amount
            
        if isinstance(amount, list):
            return [self._normalize_amount(item) for item in amount]
            
        if isinstance(amount, dict):
            return {k: self._normalize_amount(v) for k, v in amount.items()}
            
        if not isinstance(amount, str):
            return amount
            
        # Strip currency symbols and whitespace
        import re
        cleaned = re.sub(r'[^\d.]', '', amount.replace(',', '.'))
        try:
            return float(cleaned)
        except:
            return amount

    def _generate_extraction(
        self,
        ocr_text: str,
        context: str
    ) -> str:
        """
        Generate extraction results for all fields in a single pass.
        
        Args:
            ocr_text: OCR text from receipt
            context: Retrieved context from knowledge base
        """
        import requests
        
        lc = _get_langchain()
        
        fields_desc = """
        - supplier_name
        - address
        - date (YYYY-MM-DD)
        - vat_amount (float)
        - net_amount (float, total excluding VAT)
        - total_amount (float)
        - vat_details (list of: rate, amount)
        - receipt_number
        - vat_number
        - payment_method (cash/card/etc.)
        - items (list of: name, quantity, unit_price, total_price)
        """

        # Build context block only if context exists
        context_block = ""
        if context.strip():
            context_block = f"""
<guidelines_and_examples>
{context}
</guidelines_and_examples>
"""

        template = f"""You are a professional receipt data extraction expert. 
Extract data ONLY from the <target_receipt> into the JSON format below.{context_block}

<target_receipt>
{ocr_text}
</target_receipt>

SCHEMA:
- supplier_name: string (The brand/shop name. This is almost always the VERY FIRST word or identifying entity. DO NOT include addresses or line items here.)
- address: string (Full physical address if found. DO NOT include line items or totals here.)
- date: YYYY-MM-DD
- total_amount: float (The FINAL BALANCE to be paid. Look for "Total", "Balance Due", "Grand Total". Ignore suggested tips.)
- vat_amount: float (Tax amount only. Look for "Tax", "VAT", "GST". DO NOT confuse with unit prices or quantity.)
- receipt_number: string (Unique identifier like Invoice #, Check #, or Receipt ID.)
- items: list of [name, quantity, total_price] (Extract EVERY SINGLE line item from the receipt. DO NOT skip any. DO NOT stop until you reach the 'Subtotal' or 'Total' section.)
- extraction_reasoning: string (Briefly cite the exact OCR text used for supplier and total.

RULES:
1. GROUNDING: ONLY use data explicitly found in <target_receipt>. Every value you return must come from a specific word or number in <target_receipt>.
2. NO HALLUCINATION: If a field is missing or ambiguous, set it to null. NEVER copy or adapt values from <guidelines_and_examples>.
3. DO NOT INFER: Do not guess dates, names, or amounts that are not clearly stated. Use null rather than an approximation.
4. NO ARITHMETIC: Do not calculate or derive the total from line items — read it directly from the receipt text (e.g. the line labelled "Total" or "Balance Due").
5. CONTEXT IS REFERENCE ONLY: <guidelines_and_examples> shows JSON structure only. NEVER extract supplier names, amounts, or item names from it.
6. ITEM SEPARATION: Each line item must be a separate entry in 'items'. Stop extracting items when you reach 'Subtotal' or 'Total'.
7. TAX IDENTIFICATION: Distinguish clearly between 'Tax'/'VAT'/'GST' (→ vat_amount) and line item prices.
8. CLEANLINESS: No currency symbols or commas in numeric fields.
9. FORMAT: Output VALID JSON ONLY. No preamble. No afterword.
"""
        
        final_prompt = template
        
        model_to_use = self.model_name
        if model_to_use == "llama3":
            model_to_use = "llama3:latest"
            
        logger.info(f" Generating extraction with Ollama ({model_to_use})...")
        logger.info(f" Prompt length: {len(final_prompt)} characters")
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_to_use,
                    "prompt": final_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "seed": 42,
                        "num_predict": 2048
                    }
                },
                timeout=600
            )
            response.raise_for_status()
            llm_output = response.json().get('response', '')
            return self._clean_json_output(llm_output)
        except Exception as e:
            logger.error(f" Extraction generation failed: {e}")
            raise

    def extract_from_ocr(
        self,
        ocr_text: str,
        retrieve_k: int = 3
    ) -> RAGResult:
        """
        Main extraction method: Retrieve context + Generate extraction.
        
        Args:
            ocr_text: OCR text from receipt
            retrieve_k: Number of context chunks to retrieve
            
        Returns:
            RAGResult with extracted data and metadata
        """
        self._initialize()
        
        logger.info("=" * 70)
        logger.info(" RECEIPT RAG EXTRACTION STARTED")
        logger.info("=" * 70)
        
        # Step 1: Retrieve context
        retrieval = self.retrieve_context(ocr_text, k=retrieve_k)
        context = retrieval['context']
        relevance_scores = retrieval['relevance_scores']
        
        # Step 2: Generate extraction (Single Prompt Strategy)
        raw_output = self._generate_extraction(ocr_text, context)
        
        # Step 3: Parsing & Normalization
        data = self._parse_llm_response(raw_output)
        data = self._normalize_extracted_data(data)

        # Step 3b: OCR fallback for items when LLM returned none
        if not data.get('items'):
            fallback_items = self._ocr_fallback_items(ocr_text)
            if fallback_items:
                data['items'] = fallback_items
                logger.info(f" OCR fallback extracted {len(fallback_items)} item(s)")

        
        # Merge results, but keep track of failures
        merged_data = data
        parsing_failures = []
        
        if "_parsing_error" in data:
            parsing_failures.append("LLM response failed to parse as valid JSON")
        
        # Step 4: Fallback Strategy & Overall Confidence
        # Check field-wise confidence
        fallbacks = []
        for key, value in list(merged_data.items()):
            if key.endswith("_confidence"):
                field_name = key.replace("_confidence", "")
                
                # Skip non-numeric confidence values (lists, dicts, etc.)
                if isinstance(value, (list, dict)):
                    continue
                    
                try:
                    confidence_score = float(value) if value is not None else 0.0
                except (ValueError, TypeError):
                    confidence_score = 0.0
                    
                if confidence_score < 0.90:
                    fallbacks.append({
                        "field": field_name,
                        "score": confidence_score,
                        "action": "Manual review required"
                    })
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
        overall_confidence = 0.6 + (0.3 * avg_relevance)
        
        logger.info("=" * 70)
        logger.info("EXTRACTION COMPLETE")
        logger.info(f"   Overall Confidence: {overall_confidence:.2%}")
        if fallbacks:
            logger.warning(f"   Low confidence fields detected: {len(fallbacks)}")
        logger.info("=" * 70)
        
        result = RAGResult(
            extracted_data=merged_data,
            confidence=overall_confidence,
            context_used=context,
            raw_llm_response=raw_output,
            relevance_scores=relevance_scores
        )
        
        # Add metadata and warnings to extracted data
        if parsing_failures:
            merged_data["_parsing_failures"] = parsing_failures
        if fallbacks:
            merged_data["_low_confidence_warnings"] = fallbacks
            
        # Step 5: Hallucination Validation
        hallucination_report = self._validate_hallucinations(merged_data, ocr_text, context)
        if hallucination_report:
            merged_data["_hallucination_report"] = hallucination_report
            # Adjust overall confidence if hallucinations suspected
            overall_confidence *= (1.0 - (0.2 * len(hallucination_report)))
            overall_confidence = max(0.0, overall_confidence)
        
        result = RAGResult(
            extracted_data=merged_data,
            confidence=overall_confidence,
            context_used=context,
            raw_llm_response=raw_output,
            relevance_scores=relevance_scores,
            hallucination_report=hallucination_report
        )
        
        return result

    def _validate_hallucinations(self, data: Dict[str, Any], ocr_text: str, context: str) -> List[str]:
        """
        Check for common hallucination patterns by cross-referencing with OCR source.
        """
        report = []
        import re

        ocr_lower = ocr_text.lower()

        def _normalise_numeric_str(s: str) -> str:
            """Strip currency symbols and normalise comma/dot so 1,80 == 1.80."""
            s = re.sub(r'[^\d.,]', '', s)
            # European format: 1.234,56 -> replace . then ,
            if re.search(r'\d\.\d{3},\d', s):
                s = s.replace('.', '').replace(',', '.')
            else:
                s = s.replace(',', '.')
            return s

        # Helper to check if a value exists in OCR text
        def is_in_ocr(value):
            if value is None or value == 'N/A' or value == '':
                return True

            # Numeric grounding — normalise both sides before comparing
            if isinstance(value, (int, float)):
                if value == 0 or value == 0.0:
                    return True
                # Build a pattern that matches the number with optional surrounding currency/comma variation
                # e.g. 1.80 matches "1.80", "1,80", "£1.80"
                num_str = f"{value:g}"  # e.g. "1.8" or "93.58"
                # Also produce comma variant
                comma_variant = num_str.replace('.', ',')
                # Regex: optional non-digit prefix, then the number
                for variant in [num_str, comma_variant]:
                    if re.search(re.escape(variant), ocr_text):
                        return True
                return False

            v_str = str(value).strip().lower()
            if not v_str:
                return True

            # For multiline addresses, check each line
            if '\n' in v_str:
                lines = [line.strip() for line in v_str.split('\n') if line.strip()]
                return all(line in ocr_lower for line in lines)

            # For string values that look like numbers (e.g. "1.80"), normalise
            normalised = _normalise_numeric_str(v_str)
            if normalised and re.match(r'^[\d.]+$', normalised):
                try:
                    num_val = float(normalised)
                    return is_in_ocr(num_val)
                except ValueError:
                    pass

            return v_str in ocr_lower

        # 1. Fact Check: Key Fields
        for field in ['total_amount', 'vat_amount', 'receipt_number', 'supplier_name']:
            val = data.get(field)
            if isinstance(val, dict) and 'value' in val:
                val = val['value']
            if val and not is_in_ocr(val):
                report.append(f"Potential hallucination: '{field}' value '{val}' not found in raw OCR text.")

        # 1.1 Leaked item data in address/supplier fields
        for field in ['address', 'supplier_name']:
            val = str(data.get(field, '')).lower()
            if '$' in val or 'amt' in val or 'qty' in val or any(f"${i}" in val for i in range(10)):
                report.append(f"Potential hallucination: '{field}' seems to contain line item data (prices/amounts).")

        # 2. Fact Check: Items (Names and Prices)
        items = data.get('items', [])
        calculated_subtotal = 0.0

        for item in items:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                name = str(item[0])
                price = item[-1]
            elif isinstance(item, dict):
                name = item.get('name', 'N/A')
                if isinstance(name, list) and name:
                    name = name[0]
                price = item.get('total_price', item.get('price', 0.0))
                if isinstance(price, dict) and 'value' in price:
                    price = price['value']
            else:
                continue

            # Grounding for name: only flag if no word from the name appears in OCR
            if name != 'N/A' and not is_in_ocr(name):
                words = [w for w in str(name).split() if len(w) > 3]
                if words and not any(w.lower() in ocr_lower for w in words):
                    report.append(f"Potential hallucination: Item name '{name}' not grounded in OCR text.")

            # Grounding for price
            if price and not is_in_ocr(price):
                report.append(f"Potential hallucination: Item price '{price}' for '{name}' not found in OCR text.")

            try:
                calculated_subtotal += float(price) if price else 0.0
            except (ValueError, TypeError):
                pass

        # 3. Sanity Check: Sum of Items + Tax ≈ Total
        total_amt = data.get('total_amount')
        if isinstance(total_amt, dict):
            total_amt = total_amt.get('value')
        tax_amt = data.get('vat_amount', 0.0)
        if isinstance(tax_amt, dict):
            tax_amt = tax_amt.get('value')

        try:
            total_float = float(total_amt) if total_amt else 0.0
            tax_float = float(tax_amt) if tax_amt else 0.0
            if items and total_float > 0:
                diff = abs((calculated_subtotal + tax_float) - total_float)
                if diff > 1.0 and diff > (total_float * 0.05):
                    report.append(
                        f"Potential hallucination/imprecision: Sum of items ({calculated_subtotal:.2f}) "
                        f"+ tax ({tax_float:.2f}) does not match Total ({total_float:.2f})."
                    )
        except (ValueError, TypeError):
            pass

        # 4. Context-Leak Check
        # Any non-trivial token that appears in KB context but NOT in OCR text is suspicious
        if context.strip():
            context_tokens = set(re.findall(r'\b[a-zA-Z]{5,}\b', context.lower()))
            # Remove tokens that also appear in OCR (legitimate overlap)
            ocr_tokens = set(re.findall(r'\b[a-zA-Z]{5,}\b', ocr_lower))
            context_only_tokens = context_tokens - ocr_tokens

            data_str_lower = str(data).lower()
            # Only flag if a context-only token appears in the *values* we extracted
            for tok in context_only_tokens:
                if tok in data_str_lower:
                    # Avoid noisy single-word matches for very common words
                    if len(tok) >= 6:
                        report.append(
                            f"Potential context leak: '{tok}' appears in KB examples but not in OCR — "
                            "may have been copied from a context example."
                        )
                        break  # Report once per extraction to avoid flooding

        # 5. Hard-coded placeholder blacklist (fast path)
        placeholder_blacklist = [
            'freshmart', 'fresh mart', 'anytown', 'main street', '123 luxury way',
            'organic apples', 'whole wheat bread', 'grande latte', 'wagamama',
            'tesco', 'sainsbury', 'starbucks', 'mcdonald',
        ]
        data_str_lower = str(data).lower()
        for placeholder in placeholder_blacklist:
            if placeholder in data_str_lower and placeholder not in ocr_lower:
                report.append(
                    f"Hallucination Detected: Found common placeholder/example value '{placeholder}' not present in OCR."
                )

        return report
    
    def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """
        Parse LLM response into structured JSON with aggressive repair logic.
        """
        import re
        import json
        from json_repair import repair_json
        
        cleaned_response = llm_response.strip()
        
        # 1. Try to find JSON block(s)
        # We look for something that starts with { and ends with }
        # Try to find the largest outer block first
        match = re.search(r'(\{.*\})', cleaned_response, re.DOTALL)
        if match:
            candidate = match.group(1)
            try:
                repaired = repair_json(candidate, return_objects=True)
                if isinstance(repaired, dict):
                    return repaired
                # Sometimes it returns a list of one dict
                if isinstance(repaired, list) and len(repaired) > 0 and isinstance(repaired[0], dict):
                    return repaired[0]
            except Exception:
                pass

        # 2. If that fails, try finding multiple objects (models sometimes output separate {field:val} blocks)
        # Non-recursive but handles most flat structures
        blocks = re.findall(r'(\{[^{}]+\})', cleaned_response, re.DOTALL)
        merged_result = {}
        success = False
        
        for block in blocks:
            try:
                repaired = repair_json(block, return_objects=True)
                if isinstance(repaired, dict):
                    merged_result.update(repaired)
                    success = True
            except Exception:
                continue
                
        if success:
            return merged_result

        # 3. Last ditch effort: try the whole response through json_repair
        try:
            repaired = repair_json(cleaned_response, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception:
            pass
            
        logger.warning("All JSON parsing attempts failed")
        return {
            "_original_response": llm_response,
            "_parsing_error": "Failed to parse LLM response into valid JSON"
        }
    
    def format_as_ocr_result(self, rag_result: RAGResult, original_ocr: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format RAG result to match OCR client output structure.
        
        Args:
            rag_result: RAG extraction result
            original_ocr: Original OCR client output
            
        Returns:
            Enhanced OCR result with RAG extractions
        """
        # Merge RAG extractions with original OCR structure
        output = original_ocr.copy() if original_ocr else {}
        
        # Add RAG extractions
        output['rag_extractions'] = rag_result.extracted_data
        output['rag_confidence'] = rag_result.confidence
        output['rag_context_relevance'] = rag_result.relevance_scores
        
        logger.info(" Formatted RAG result as OCR output")
        
        return output

    def save_result_to_file(self, result: RAGResult, base_name: str, target_dir: Optional[Path] = None):
        """
        Save RAG result to JSON and formatted text files.
        """
        if target_dir is None:
            target_dir = Path(__file__).parent
        
        # Save JSON
        json_path = target_dir / f"{base_name}_result.json"
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(result.extracted_data, f, indent=2)
        print(f" [OK] Saved JSON results to: {json_path}")
        
        # Save formatted text summary
        txt_path = target_dir / f"{base_name}_summary.txt"
        with open(txt_path, "w", encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write(f"EXTRACTION SUMMARY: {base_name.upper()}\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Confidence: {result.confidence:.2%}\n")
            f.write(f"Relevance Scores: {result.relevance_scores}\n")
            
            if result.hallucination_report:
                f.write("\n" + "!"*70 + "\n")
                f.write("HALLUCINATION WARNINGS:\n")
                for warning in result.hallucination_report:
                    f.write(f"  [X] {warning}\n")
                f.write("!"*70 + "\n\n")
            else:
                f.write("Hallucination check: PASSED (All data grounded in OCR)\n\n")
            
            data = result.extracted_data
            
            # Helper to extract value from production-style dict
            def get_val(key):
                val = data.get(key, 'N/A')
                if isinstance(val, dict) and 'value' in val:
                    return val['value']
                return val

            f.write(f"Supplier: {get_val('supplier_name')}\n")
            f.write(f"Address: {get_val('address')}\n")
            
            # Handle list-style dates/amounts
            date_val = get_val('date')
            if isinstance(date_val, list) and date_val:
                f.write(f"Date: {date_val[0].get('value') if isinstance(date_val[0], dict) else date_val[0]}\n")
            else:
                f.write(f"Date: {date_val}\n")
                
            f.write(f"Total Amount: {get_val('total_amount')}\n")
            tax_val = get_val('vat_amount')
            if tax_val == 'N/A': tax_val = get_val('tax_amount')
            f.write(f"Tax Amount: {tax_val}\n\n")
            
            f.write("Items:\n")
            for item in data.get('items', []):
                if isinstance(item, dict):
                    name = item.get('name', ['N/A'])[0] if isinstance(item.get('name'), list) else item.get('name', 'N/A')
                    price = item.get('total_price', {}).get('value', 'N/A') if isinstance(item.get('total_price'), dict) else item.get('total_price', 'N/A')
                    f.write(f"  - {name}: {price}\n")
            
            f.write("\n" + "="*70 + "\n")
            
        print(f" [✓] Saved text summary to: {txt_path}")


# Singleton instance
_rag_pipeline = None


def get_rag_pipeline(**kwargs) -> ReceiptRAGPipeline:
    """Get or create singleton RAG pipeline instance."""
    global _rag_pipeline
    
    if _rag_pipeline is None:
        _rag_pipeline = ReceiptRAGPipeline(**kwargs)
    
    return _rag_pipeline
