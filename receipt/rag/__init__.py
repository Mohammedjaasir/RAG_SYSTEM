#!/usr/bin/env python3
"""
RAG Module - Receipt Intelligence Knowledge Base v1.0.0

Exports:
- ReceiptVectorStore: ChromaDB-based vector storage
- KnowledgeBaseLoader: Loads patterns into vector store
- RAGRetriever: Retrieves relevant knowledge for extraction
- VendorClassifier: Upfront vendor identification
- PatternLearner: Automatic pattern learning from successful extractions
"""

from .vector_store import ReceiptVectorStore, get_vector_store
from .knowledge_base_loader import (
    KnowledgeBaseLoader,
    get_knowledge_base_loader,
    initialize_knowledge_base
)
from .rag_retriever import RAGRetriever, get_rag_retriever
from .vendor_classifier import VendorClassifier, VendorClassification, get_vendor_classifier
from .pattern_learner import PatternLearner, LearnedPattern, get_pattern_learner
from .rag_refresh_manager import RAGRefreshManager, get_rag_refresh_manager, RefreshType

__all__ = [
    'ReceiptVectorStore',
    'get_vector_store',
    'KnowledgeBaseLoader',
    'get_knowledge_base_loader',
    'initialize_knowledge_base',
    'RAGRetriever',
    'get_rag_retriever',
    'VendorClassifier',
    'VendorClassification',
    'get_vendor_classifier',
    'PatternLearner',
    'LearnedPattern',
    'get_pattern_learner',
    'RAGRefreshManager',
    'get_rag_refresh_manager',
    'RefreshType'
]
