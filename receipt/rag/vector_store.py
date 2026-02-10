#!/usr/bin/env python3
"""
Vector Store - ChromaDB-based Receipt Knowledge Base v1.0.0

Provides persistent vector storage for receipt intelligence patterns:
- Receipt templates (vendor-agnostic patterns)
- Field definitions (Total, Invoice Number, etc.)
- Country/tax rules (GST, VAT patterns)
- Historical validated patterns

Features:
- Sentence-transformer embeddings (auto-downloads from HuggingFace)
- Versioned document schema with metadata
- Incremental refresh (no full re-embedding)
- Clean delete and version control
"""

import os
import json
import logging
import threading
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import hashlib

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_chromadb = None
_sentence_transformer = None


def _get_chromadb():
    """Lazy import ChromaDB."""
    global _chromadb
    if _chromadb is None:
        try:
            import chromadb
            _chromadb = chromadb
            logger.info("✅ ChromaDB imported successfully")
        except ImportError:
            logger.error("❌ ChromaDB not installed. Run: pip install chromadb")
            raise
    return _chromadb


def _get_sentence_transformer():
    """Lazy import sentence-transformers."""
    global _sentence_transformer
    if _sentence_transformer is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer = SentenceTransformer
            logger.info("✅ Sentence-transformers imported successfully")
        except ImportError:
            logger.error("❌ sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
    return _sentence_transformer


@dataclass
class KnowledgeDocument:
    """Document structure for the knowledge base."""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReceiptVectorStore:
    """
    ChromaDB-based vector store for receipt intelligence.
    
    Schema:
    {
        "doc_id": "template_indian_grocery_v1",
        "content": "Indian grocery receipts usually...",
        "metadata": {
            "vendor_type": "grocery",
            "country": "IN",
            "fields": ["invoice_no", "date", "total", "cgst", "sgst"],
            "version": 1,
            "doc_type": "template|field_definition|tax_rule|pattern"
        }
    }
    """
    
    _instance = None
    _lock = threading.Lock()
    
    # Default collection name
    COLLECTION_NAME = "receipt_knowledge"
    
    # Embedding model (lightweight, auto-downloads from HuggingFace)
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the vector store (singleton pattern)."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            # Determine persistence directory
            self.persist_dir = Path(__file__).parent / "chroma_db"
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            chromadb = _get_chromadb()
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            
            # Initialize embedding model
            SentenceTransformer = _get_sentence_transformer()
            logger.info(f"Loading embedding model: {self.EMBEDDING_MODEL}")
            self.embedding_model = SentenceTransformer(self.EMBEDDING_MODEL)
            logger.info("✅ Embedding model loaded")
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "Receipt intelligence knowledge base"}
            )
            
            self._initialized = True
            logger.info(f"✅ ReceiptVectorStore initialized at {self.persist_dir}")
            logger.info(f"   Collection: {self.COLLECTION_NAME}, Documents: {self.collection.count()}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using sentence-transformers."""
        return self.embedding_model.encode(text).tolist()
    
    def _generate_doc_id(self, doc_type: str, content: str, metadata: Dict) -> str:
        """Generate unique document ID based on content hash."""
        hash_input = f"{doc_type}_{json.dumps(metadata, sort_keys=True)}_{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def add_document(
        self,
        content: str,
        doc_type: str,
        vendor_type: Optional[str] = None,
        country: Optional[str] = None,
        fields: Optional[List[str]] = None,
        version: int = 1,
        custom_id: Optional[str] = None,
        **extra_metadata
    ) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            content: Document text content
            doc_type: Type of document (template, field_definition, tax_rule, pattern)
            vendor_type: Type of vendor (grocery, petrol, hotel, etc.)
            country: Country code (IN, UK, US, EU, MY)
            fields: List of fields this document describes
            version: Version number for incremental updates
            custom_id: Optional custom document ID
            **extra_metadata: Additional metadata fields
            
        Returns:
            Document ID
        """
        # Build metadata
        metadata = {
            "doc_type": doc_type,
            "version": version,
        }
        
        if vendor_type:
            metadata["vendor_type"] = vendor_type
        if country:
            metadata["country"] = country
        if fields:
            metadata["fields"] = ",".join(fields)  # ChromaDB doesn't support lists in metadata
        
        metadata.update(extra_metadata)
        
        # Generate or use provided ID
        doc_id = custom_id or self._generate_doc_id(doc_type, content, metadata)
        
        # Generate embedding
        embedding = self._generate_embedding(content)
        
        # Check if document exists (for updates)
        try:
            existing = self.collection.get(ids=[doc_id])
            if existing and existing['ids']:
                # Update existing document
                self.collection.update(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[metadata]
                )
                logger.debug(f"Updated document: {doc_id}")
            else:
                # Add new document
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[metadata]
                )
                logger.debug(f"Added document: {doc_id}")
        except Exception:
            # Document doesn't exist, add new
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata]
            )
            logger.debug(f"Added document: {doc_id}")
        
        return doc_id
    
    def query(
        self,
        query_text: str,
        n_results: int = 5,
        doc_type: Optional[str] = None,
        vendor_type: Optional[str] = None,
        country: Optional[str] = None,
        **filter_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for relevant documents.
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            doc_type: Filter by document type
            vendor_type: Filter by vendor type
            country: Filter by country code
            **filter_kwargs: Additional metadata filters
            
        Returns:
            List of matching documents with scores
        """
        if self.collection.count() == 0:
            logger.warning("Knowledge base is empty")
            return []
        
        # Build where filter
        where_filters = []
        
        if doc_type:
            where_filters.append({"doc_type": doc_type})
        if vendor_type:
            where_filters.append({"vendor_type": vendor_type})
        if country:
            where_filters.append({"country": country})
        
        for key, value in filter_kwargs.items():
            where_filters.append({key: value})
        
        # Combine filters with $and
        where = None
        if len(where_filters) == 1:
            where = where_filters[0]
        elif len(where_filters) > 1:
            where = {"$and": where_filters}
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query_text)
        
        # Query collection
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),
                where=where if where else None,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.warning(f"Query failed with filters, retrying without: {e}")
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
        
        # Format results
        formatted_results = []
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    'doc_id': doc_id,
                    'content': results['documents'][0][i] if results['documents'] else "",
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0.0,
                    'relevance_score': 1.0 - (results['distances'][0][i] if results['distances'] else 0.0)
                })
        
        logger.debug(f"Query returned {len(formatted_results)} results")
        return formatted_results
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document by ID."""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def delete_by_type(self, doc_type: str) -> int:
        """Delete all documents of a specific type."""
        try:
            # Get all documents of this type
            results = self.collection.get(
                where={"doc_type": doc_type},
                include=["metadatas"]
            )
            
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} documents of type: {doc_type}")
                return len(results['ids'])
            return 0
        except Exception as e:
            logger.error(f"Failed to delete documents of type {doc_type}: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        total_docs = self.collection.count()
        
        # Get document type distribution
        type_counts = {}
        try:
            all_docs = self.collection.get(include=["metadatas"])
            if all_docs and all_docs['metadatas']:
                for metadata in all_docs['metadatas']:
                    doc_type = metadata.get('doc_type', 'unknown')
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        except Exception as e:
            logger.warning(f"Failed to get document type distribution: {e}")
        
        return {
            "total_documents": total_docs,
            "collection_name": self.COLLECTION_NAME,
            "persist_dir": str(self.persist_dir),
            "embedding_model": self.EMBEDDING_MODEL,
            "document_types": type_counts
        }
    
    def clear_all(self) -> int:
        """Clear all documents from the knowledge base (HARD RESET)."""
        count = self.collection.count()
        if count > 0:
            # Get all IDs and delete
            all_docs = self.collection.get()
            if all_docs and all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
        logger.warning(f"⚠️ Cleared all {count} documents from knowledge base")
        return count


# Singleton accessor
_vector_store_instance = None


def get_vector_store() -> ReceiptVectorStore:
    """Get or create singleton ReceiptVectorStore instance."""
    global _vector_store_instance
    
    if _vector_store_instance is None:
        _vector_store_instance = ReceiptVectorStore()
    
    return _vector_store_instance
