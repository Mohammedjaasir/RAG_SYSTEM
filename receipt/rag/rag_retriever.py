#!/usr/bin/env python3
"""
RAG Retriever - Retrieves Relevant Receipt Intelligence v1.0.0

Queries the vector store for receipt intelligence patterns relevant to:
- The current receipt being processed
- Country-specific tax rules
- Field extraction guidance
"""

import logging
import re
from typing import Dict, List, Any, Optional

from .vector_store import get_vector_store

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Retrieves relevant receipt intelligence from the vector store.
    
    Features:
    - Query by text content
    - Filter by country, vendor type
    - Format context for Phi-3 prompts
    """
    
    def __init__(self):
        """Initialize the RAG retriever."""
        self.vector_store = get_vector_store()
        logger.info("✅ RAGRetriever initialized")
    
    def retrieve(
        self,
        query_text: str,
        country: Optional[str] = None,
        vendor_type: Optional[str] = None,
        n_results: int = 5,
        min_relevance: float = 0.3
    ) -> Dict[str, Any]:
        """
        Retrieve relevant knowledge for extraction.
        
        Args:
            query_text: Receipt text or query
            country: Optional country code (IN, UK, US, etc.)
            vendor_type: Optional vendor type (grocery, petrol, etc.)
            n_results: Maximum results to return
            min_relevance: Minimum relevance score (0-1)
            
        Returns:
            Dictionary with retrieved context and metadata
        """
        results = {
            "field_definitions": [],
            "tax_rules": [],
            "layout_patterns": [],
            "templates": [],
            "total_results": 0,
            "query_country": country,
            "query_vendor_type": vendor_type
        }
        
        # Retrieve field definitions (always useful)
        field_results = self.vector_store.query(
            query_text=query_text,
            n_results=n_results,
            doc_type="field_definition"
        )
        for r in field_results:
            if r.get('relevance_score', 0) >= min_relevance:
                results["field_definitions"].append(r)
        
        # Retrieve tax rules (filter by country if provided)
        tax_results = self.vector_store.query(
            query_text=query_text,
            n_results=3,
            doc_type="tax_rule",
            country=country
        )
        for r in tax_results:
            if r.get('relevance_score', 0) >= min_relevance:
                results["tax_rules"].append(r)
        
        # Retrieve layout patterns
        layout_results = self.vector_store.query(
            query_text=query_text,
            n_results=3,
            doc_type="layout_pattern",
            vendor_type=vendor_type if vendor_type else "universal"
        )
        for r in layout_results:
            if r.get('relevance_score', 0) >= min_relevance:
                results["layout_patterns"].append(r)
        
        # Retrieve receipt templates (vendor-specific examples)
        template_results = self.vector_store.query(
            query_text=query_text,
            n_results=2,
            doc_type="template",
            country=country
        )
        for r in template_results:
            if r.get('relevance_score', 0) >= min_relevance:
                results["templates"].append(r)
        
        results["total_results"] = (
            len(results["field_definitions"]) +
            len(results["tax_rules"]) +
            len(results["layout_patterns"]) +
            len(results["templates"])
        )
        
        logger.debug(f"Retrieved {results['total_results']} knowledge documents")
        return results
    
    def format_for_prompt(
        self,
        retrieved_knowledge: Dict[str, Any],
        max_length: int = 1500
    ) -> str:
        """
        Format retrieved knowledge as context for Phi-3 prompt.
        
        IMPORTANT: This context is for GUIDANCE ONLY. Phi-3 should NOT extract
        data from these patterns - only use them to understand field types.
        
        Args:
            retrieved_knowledge: Output from retrieve()
            max_length: Maximum characters for context
            
        Returns:
            Formatted context string (sanitized to prevent extraction confusion)
        """
        sections = []
        
        # Add field guidance (rules only, no examples)
        if retrieved_knowledge.get("field_definitions"):
            field_context = "Field Types to Extract:\n"
            for field in retrieved_knowledge["field_definitions"][:3]:
                content = field.get('content', '')[:150]
                # Remove specific values/examples that might confuse the model
                content = self._sanitize_pattern(content)
                field_context += f"• {content}\n"
            sections.append(field_context)
        
        # Add tax rules (country-specific guidance)
        if retrieved_knowledge.get("tax_rules"):
            tax_context = "Tax/VAT Rules:\n"
            for rule in retrieved_knowledge["tax_rules"][:2]:
                content = rule.get('content', '')[:200]
                content = self._sanitize_pattern(content)
                tax_context += f"• {content}\n"
            sections.append(tax_context)
        
        # Add layout hints (structure only)
        if retrieved_knowledge.get("layout_patterns"):
            layout_context = "Receipt Structure Hints:\n"
            for pattern in retrieved_knowledge["layout_patterns"][:2]:
                content = pattern.get('content', '')[:150]
                content = self._sanitize_pattern(content)
                layout_context += f"• {content}\n"
            sections.append(layout_context)
        
        # SKIP templates - they contain specific examples that confuse extraction
        # Templates are too specific and cause Phi-3 to copy example data
        
        # Combine and truncate
        context = "\n".join(sections)
        if len(context) > max_length:
            context = context[:max_length] + "..."
        
        return context
    
    def _sanitize_pattern(self, text: str) -> str:
        """
        Sanitize pattern text to remove specific values that might be extracted.
        Keep only structural/type information.
        """
        # Remove common example patterns that cause confusion
        text = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '[DATE]', text)  # Dates
        text = re.sub(r'£\d+\.\d{2}', '[AMOUNT]', text)  # Prices
        text = re.sub(r'\$\d+\.\d{2}', '[AMOUNT]', text)
        text = re.sub(r'€\d+\.\d{2}', '[AMOUNT]', text)
        text = re.sub(r'₹\d+\.\d{2}', '[AMOUNT]', text)
        text = re.sub(r'\d{4}\s?\d{4}\s?\d{4}\s?\d{4}', '[CARD]', text)  # Card numbers
        text = re.sub(r'\*+\d{4}', '[CARD_LAST4]', text)  # Masked cards
        text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', '[NAME]', text)  # Names
        
        # Remove lines that start with specific product names (learned patterns)
        lines = text.split('\n')
        filtered = []
        for line in lines:
            # Skip lines that look like learned item patterns
            if re.match(r'^(Fresh|Loose|Organic|Premium|Product|Item)[\s:]', line):
                continue
            # Skip lines with "Field:" prefix that contain actual values
            if 'Field:' in line and any(char.isdigit() for char in line):
                continue
            filtered.append(line)
        
        return '\n'.join(filtered).strip()
    
    def retrieve_for_extraction(
        self,
        ocr_text: str,
        country: Optional[str] = None,
        vendor_type: Optional[str] = None
    ) -> str:
        """
        Convenience method: Retrieve and format knowledge for extraction.
        
        Args:
            ocr_text: Raw or normalized OCR text
            country: Country code
            vendor_type: Vendor type
            
        Returns:
            Formatted context ready for Phi-3 prompt
        """
        # Use first 500 chars of OCR as query (most relevant info usually at top)
        query = ocr_text[:500] if len(ocr_text) > 500 else ocr_text
        
        knowledge = self.retrieve(
            query_text=query,
            country=country,
            vendor_type=vendor_type,
            n_results=5,
            min_relevance=0.3
        )
        
        return self.format_for_prompt(knowledge)
    
    def retrieve_with_confidence(
        self,
        ocr_text: str,
        country: Optional[str] = None,
        vendor_type: Optional[str] = None
    ) -> tuple:
        """
        Retrieve knowledge and calculate pattern match confidence.
        
        Args:
            ocr_text: Raw or normalized OCR text
            country: Country code
            vendor_type: Vendor type
            
        Returns:
            Tuple of (context_string, pattern_match_confidence)
            Confidence is derived from relevance scores: 0.5 + 0.3 * avg_relevance
        """
        query = ocr_text[:500] if len(ocr_text) > 500 else ocr_text
        
        # Get knowledge with relevance scores
        knowledge = self.retrieve(
            query_text=query,
            country=country,
            vendor_type=vendor_type,
            n_results=5,
            min_relevance=0.3
        )
        
        # Calculate pattern match confidence from relevance scores
        all_scores = []
        for doc_type in ["field_definitions", "tax_rules", "layout_patterns", "templates"]:
            for doc in knowledge.get(doc_type, []):
                if isinstance(doc, dict) and "relevance" in doc:
                    all_scores.append(doc["relevance"])
        
        if all_scores:
            avg_relevance = sum(all_scores) / len(all_scores)
            # Scale: 0.5 base + 0.3 * relevance (max 0.8 for perfect matches)
            pattern_match_confidence = 0.5 + 0.3 * min(1.0, avg_relevance)
        else:
            # No matches = low confidence
            pattern_match_confidence = 0.5 if knowledge else 0.3
        
        context = self.format_for_prompt(knowledge)
        return context, round(pattern_match_confidence, 3)


# Singleton instance
_rag_retriever_instance = None


def get_rag_retriever() -> RAGRetriever:
    """Get or create singleton RAGRetriever instance."""
    global _rag_retriever_instance
    
    if _rag_retriever_instance is None:
        _rag_retriever_instance = RAGRetriever()
    
    return _rag_retriever_instance
