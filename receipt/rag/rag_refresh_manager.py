#!/usr/bin/env python3
"""
RAG Refresh Manager - Incremental Knowledge Base Updates v1.0.0

Handles soft and hard refresh of the RAG knowledge base:
- Soft refresh: New vendors, new formats (incremental, no full re-embedding)
- Hard refresh: New embedding model, chunking logic change (full re-embedding)

No user receipt data or PII is ever stored - only patterns.
"""

import logging
from typing import Dict, Any, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class RefreshType(Enum):
    """Types of RAG refresh operations."""
    SOFT = "soft"  # New vendors, new formats
    HARD = "hard"  # New embedding model, chunking change


class RAGRefreshManager:
    """
    Manages incremental and full refresh of the RAG knowledge base.
    
    Soft Refresh (incremental):
    - Triggered when new receipt pattern is learned
    - Adds single document to vector store
    - No re-embedding of existing documents
    
    Hard Refresh (full):
    - Triggered when embedding model changes
    - Triggered when chunking strategy changes
    - Re-embeds all documents
    """
    
    def __init__(self):
        """Initialize the refresh manager."""
        self._vector_store = None
        self._kb_loader = None
        self._refresh_history = []
        logger.info("RAGRefreshManager initialized")
    
    def _get_vector_store(self):
        """Lazy load vector store."""
        if self._vector_store is None:
            from .vector_store import get_vector_store
            self._vector_store = get_vector_store()
        return self._vector_store
    
    def _get_kb_loader(self):
        """Lazy load knowledge base loader."""
        if self._kb_loader is None:
            from .knowledge_base_loader import get_knowledge_base_loader
            self._kb_loader = get_knowledge_base_loader()
        return self._kb_loader
    
    def soft_refresh(
        self,
        pattern_content: str,
        pattern_type: str,
        vendor_type: Optional[str] = None,
        country: Optional[str] = None,
        **metadata
    ) -> Dict[str, Any]:
        """
        Perform soft refresh - add new pattern without full re-embedding.
        
        Use cases:
        - New vendor format detected
        - User corrected extraction → new pattern learned
        - New field pattern discovered
        
        Args:
            pattern_content: Description of the new pattern
            pattern_type: Type (field_definition, tax_rule, layout_pattern)
            vendor_type: Optional vendor type
            country: Optional country code
            **metadata: Additional metadata
            
        Returns:
            Refresh result with doc_id
        """
        try:
            vector_store = self._get_vector_store()
            
            # Add single document
            doc_id = vector_store.add_document(
                content=pattern_content,
                doc_type=pattern_type,
                vendor_type=vendor_type,
                country=country,
                learned=True,
                refresh_type="soft",
                **metadata
            )
            
            self._refresh_history.append({
                "type": RefreshType.SOFT.value,
                "doc_id": doc_id,
                "pattern_type": pattern_type
            })
            
            logger.info(f"Soft refresh: added pattern {doc_id}")
            
            return {
                "success": True,
                "refresh_type": RefreshType.SOFT.value,
                "doc_id": doc_id,
                "message": "Pattern added without full re-embedding"
            }
            
        except Exception as e:
            logger.error(f"Soft refresh failed: {e}")
            return {
                "success": False,
                "refresh_type": RefreshType.SOFT.value,
                "error": str(e)
            }
    
    def hard_refresh(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform hard refresh - full re-embedding of all documents.
        
        Use cases:
        - Embedding model upgraded
        - Chunking strategy changed
        - Schema migration
        
        Args:
            force: Skip confirmation (use with caution)
            
        Returns:
            Refresh result with counts
        """
        try:
            vector_store = self._get_vector_store()
            kb_loader = self._get_kb_loader()
            
            # Get current stats before clearing
            stats_before = vector_store.get_stats()
            
            # Clear and reload
            cleared_count = vector_store.clear_all()
            
            # Reload from source files
            counts = kb_loader.load_all(force_reload=False)  # Files already exist
            
            self._refresh_history.append({
                "type": RefreshType.HARD.value,
                "cleared": cleared_count,
                "reloaded": counts
            })
            
            logger.warning(f"Hard refresh: cleared {cleared_count}, reloaded {sum(counts.values())}")
            
            return {
                "success": True,
                "refresh_type": RefreshType.HARD.value,
                "cleared_count": cleared_count,
                "reloaded_count": counts,
                "message": "Full re-embedding completed"
            }
            
        except Exception as e:
            logger.error(f"Hard refresh failed: {e}")
            return {
                "success": False,
                "refresh_type": RefreshType.HARD.value,
                "error": str(e)
            }
    
    def learn_from_correction(
        self,
        original_extraction: Dict[str, Any],
        corrected_values: Dict[str, Any],
        vendor_name: Optional[str] = None,
        country: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Learn new pattern from user correction (soft refresh).
        
        This is the main entry point for the feedback loop.
        Analyzes the correction and creates a new pattern.
        
        Args:
            original_extraction: Original extracted values
            corrected_values: User-corrected values
            vendor_name: Vendor if known
            country: Country code
            
        Returns:
            Learning result
        """
        try:
            # Identify what was corrected
            corrections = []
            for field, corrected_value in corrected_values.items():
                original_value = original_extraction.get(field)
                if original_value != corrected_value:
                    corrections.append({
                        "field": field,
                        "original": original_value,
                        "corrected": corrected_value
                    })
            
            if not corrections:
                return {
                    "success": True,
                    "learned": False,
                    "message": "No corrections detected"
                }
            
            # Build pattern description
            pattern_parts = []
            
            if vendor_name:
                pattern_parts.append(f"Vendor: {vendor_name}")
            
            for correction in corrections:
                field = correction["field"]
                corrected = correction["corrected"]
                pattern_parts.append(
                    f"Field '{field}' should be extracted as '{corrected}'"
                )
            
            pattern_content = "\n".join(pattern_parts)
            
            # Perform soft refresh
            result = self.soft_refresh(
                pattern_content=pattern_content,
                pattern_type="layout_pattern",
                vendor_type=vendor_name.lower().replace(" ", "_") if vendor_name else None,
                country=country,
                learned_from_correction=True,
                corrections_count=len(corrections)
            )
            
            result["learned"] = True
            result["corrections_applied"] = len(corrections)
            
            return result
            
        except Exception as e:
            logger.error(f"Learning from correction failed: {e}")
            return {
                "success": False,
                "learned": False,
                "error": str(e)
            }
    
    def get_refresh_history(self) -> List[Dict[str, Any]]:
        """Get history of refresh operations."""
        return self._refresh_history.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get refresh manager status."""
        try:
            vector_store = self._get_vector_store()
            stats = vector_store.get_stats()
            return {
                "initialized": True,
                "knowledge_base_stats": stats,
                "refresh_count": len(self._refresh_history),
                "last_refresh": self._refresh_history[-1] if self._refresh_history else None
            }
        except Exception as e:
            return {
                "initialized": False,
                "error": str(e)
            }


# Singleton instance
_refresh_manager_instance = None


def get_rag_refresh_manager() -> RAGRefreshManager:
    """Get or create singleton RAGRefreshManager instance."""
    global _refresh_manager_instance
    
    if _refresh_manager_instance is None:
        _refresh_manager_instance = RAGRefreshManager()
    
    return _refresh_manager_instance
