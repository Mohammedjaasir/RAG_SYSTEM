#!/usr/bin/env python3
"""
Knowledge Base Loader - Initializes RAG with Receipt Intelligence v1.0.0

Loads receipt intelligence patterns into the vector store:
- Field definitions (Total, Invoice Number, etc.)
- Country/tax rules (GST, VAT patterns)
- Receipt layout patterns (vendor-agnostic)

No user receipt data or PII is stored - only extraction patterns.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from .vector_store import get_vector_store, ReceiptVectorStore

logger = logging.getLogger(__name__)


class KnowledgeBaseLoader:
    """
    Loads receipt intelligence into the vector store.
    
    Knowledge types:
    - field_definition: How to extract specific fields
    - tax_rule: Country-specific tax patterns
    - layout_pattern: Receipt structure patterns
    """
    
    def __init__(self):
        """Initialize the knowledge base loader."""
        self.vector_store = get_vector_store()
        self.knowledge_dir = Path(__file__).parent / "knowledge_base"
        logger.info(f"✅ KnowledgeBaseLoader initialized, knowledge_dir: {self.knowledge_dir}")
    
    def load_all(self, force_reload: bool = False) -> Dict[str, int]:
        """
        Load all knowledge into the vector store.
        
        Args:
            force_reload: If True, clear existing knowledge first
            
        Returns:
            Dictionary with counts of loaded documents by type
        """
        if force_reload:
            logger.warning("⚠️ Force reload requested, clearing existing knowledge")
            self.vector_store.clear_all()
        
        counts = {
            "field_definitions": 0,
            "tax_rules": 0,
            "layout_patterns": 0,
            "receipt_templates": 0
        }
        
        # Load field definitions
        counts["field_definitions"] = self._load_field_definitions()
        
        # Load tax rules
        counts["tax_rules"] = self._load_tax_rules()
        
        # Load layout patterns
        counts["layout_patterns"] = self._load_layout_patterns()
        
        # Load receipt templates (optional vendor-specific examples)
        counts["receipt_templates"] = self._load_receipt_templates()
        
        total = sum(counts.values())
        logger.info(f"✅ Loaded {total} knowledge documents: {counts}")
        
        return counts
    
    def _load_field_definitions(self) -> int:
        """Load field definition patterns."""
        file_path = self.knowledge_dir / "field_definitions.json"
        
        if not file_path.exists():
            logger.warning(f"Field definitions file not found: {file_path}")
            return 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            count = 0
            fields = data.get('fields', {})
            
            for field_name, field_info in fields.items():
                # Build content for embedding
                aliases = field_info.get('aliases', [])
                patterns = field_info.get('patterns', [])
                hints = field_info.get('extraction_hints', [])
                
                content = f"""
Field: {field_name}
Aliases: {', '.join(aliases)}
Patterns: {' '.join(patterns)}
Extraction hints: {' '.join(hints)}
                """.strip()
                
                # Add to vector store
                self.vector_store.add_document(
                    content=content,
                    doc_type="field_definition",
                    custom_id=f"field_{field_name}",
                    field_name=field_name,
                    confidence_boost=str(field_info.get('confidence_boost', 0))
                )
                count += 1
            
            logger.info(f"Loaded {count} field definitions")
            return count
            
        except Exception as e:
            logger.error(f"Error loading field definitions: {e}")
            return 0
    
    def _load_tax_rules(self) -> int:
        """Load country-specific tax rules."""
        file_path = self.knowledge_dir / "country_tax_rules.json"
        
        if not file_path.exists():
            logger.warning(f"Tax rules file not found: {file_path}")
            return 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            count = 0
            countries = data.get('countries', {})
            
            for country_code, country_info in countries.items():
                tax_rules = country_info.get('tax_rules', {})
                
                # Build content for embedding
                content = f"""
Country: {country_info.get('name', country_code)} ({country_code})
Currency: {country_info.get('currency', '')} ({', '.join(country_info.get('currency_codes', []))})
Tax System: {country_info.get('tax_system', '')}
Description: {tax_rules.get('description', '')}
Patterns: {' '.join(tax_rules.get('patterns', []))}
Calculation hints: {' '.join(tax_rules.get('calculation_hints', []))}
Date formats: {', '.join(country_info.get('date_formats', []))}
                """.strip()
                
                # Add to vector store
                self.vector_store.add_document(
                    content=content,
                    doc_type="tax_rule",
                    country=country_code,
                    custom_id=f"tax_{country_code}",
                    tax_system=country_info.get('tax_system', '')
                )
                count += 1
            
            logger.info(f"Loaded {count} tax rules")
            return count
            
        except Exception as e:
            logger.error(f"Error loading tax rules: {e}")
            return 0
    
    def _load_layout_patterns(self) -> int:
        """Load receipt layout patterns (hardcoded universal patterns)."""
        # Universal layout patterns - no country-specific, no vendor-specific
        patterns = [
            {
                "id": "layout_header",
                "content": """
Receipt Header Pattern:
- Vendor/store name typically at the very top
- May include logo area (recognized as image placeholder in OCR)
- Address and contact info follow vendor name
- Store/branch number often in header
- Date and time usually near top, after header
                """,
                "vendor_type": "universal"
            },
            {
                "id": "layout_items",
                "content": """
Receipt Items Pattern:
- Items listed in middle section of receipt
- Each line typically: Item name, Quantity (optional), Unit Price (optional), Total Price
- Item codes may appear before or after item name
- Quantities may use 'x' or '@' notation (e.g., '2 x 5.00' or '2 @ 5.00')
- Discounts often shown as negative amounts or with 'DISC', 'OFF', '-' prefix
- VAT/tax codes may appear after price (single letter or alphanumeric)
                """,
                "vendor_type": "universal"
            },
            {
                "id": "layout_totals",
                "content": """
Receipt Totals Pattern:
- Totals section at bottom of items list
- Order typically: Subtotal → Tax(es) → Total → Payment → Change
- TOTAL is usually the largest font or bold
- Multiple tax lines may appear (CGST/SGST for India, VAT for UK)
- Payment section shows method and amount tendered
- For cash: shows tendered amount and change
                """,
                "vendor_type": "universal"
            },
            {
                "id": "layout_footer",
                "content": """
Receipt Footer Pattern:
- Footer contains thank you message, return policy
- May include website, social media handles
- Loyalty points or rewards information
- QR codes or barcodes (recognized as image in OCR)
- Timestamp or transaction reference may repeat
                """,
                "vendor_type": "universal"
            }
        ]
        
        count = 0
        for pattern in patterns:
            self.vector_store.add_document(
                content=pattern["content"].strip(),
                doc_type="layout_pattern",
                custom_id=pattern["id"],
                vendor_type=pattern["vendor_type"]
            )
            count += 1
        
        logger.info(f"Loaded {count} layout patterns")
        return count
    
    def _load_receipt_templates(self) -> int:
        """
        Load receipt templates (vendor-specific examples).
        
        Templates provide examples of common receipt formats from specific vendors.
        This helps RAG provide better context for extraction.
        """
        file_path = self.knowledge_dir / "receipt_templates.json"
        
        # If file doesn't exist, use hardcoded examples
        if not file_path.exists():
            logger.info("receipt_templates.json not found, using hardcoded examples")
            templates = self._get_default_receipt_templates()
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                templates = data.get('templates', [])
            except Exception as e:
                logger.warning(f"Error loading receipt templates from file: {e}, using defaults")
                templates = self._get_default_receipt_templates()
        
        count = 0
        for template in templates:
            if not isinstance(template, dict):
                continue
            
            vendor = template.get('vendor_name', 'Unknown')
            country = template.get('country', 'universal')
            vendor_type = template.get('vendor_type', 'retail')
            
            # Build content for embedding
            content = f"""
Receipt Template Example: {vendor} ({country})
Vendor Type: {vendor_type}
Typical Fields: {', '.join(template.get('typical_fields', []))}
Layout Notes: {template.get('layout_notes', '')}
Field Locations: {template.get('field_locations', '')}
Special Characteristics: {template.get('special_characteristics', '')}
            """.strip()
            
            self.vector_store.add_document(
                content=content,
                doc_type="template",
                custom_id=f"template_{vendor.lower().replace(' ', '_')}_{country}",
                vendor_type=vendor_type,
                country=country,
                vendor_name=vendor
            )
            count += 1
        
        logger.info(f"Loaded {count} receipt templates")
        return count
    
    def _get_default_receipt_templates(self) -> List[Dict[str, Any]]:
        """Get default receipt templates (hardcoded examples)."""
        return [
            {
                "vendor_name": "Tesco UK",
                "country": "UK",
                "vendor_type": "grocery",
                "typical_fields": ["Store Name", "Store Number", "Date", "Time", "Items with VAT codes", "Subtotal", "VAT breakdown", "Total", "Payment method", "Clubcard number"],
                "layout_notes": "Tesco receipts typically have store name at top, items in middle with single-letter VAT codes (A, B, C), VAT summary before total",
                "field_locations": "Date/time near top after store info. VAT summary shows: VAT A @ 20%, VAT B @ 5%, etc. Total in bold or larger font.",
                "special_characteristics": "Uses Clubcard points system. Often includes fuel offers. VAT codes appear after each item price."
            },
            {
                "vendor_name": "Walmart US",
                "country": "US",
                "vendor_type": "grocery",
                "typical_fields": ["Store Name", "Store Number", "Date", "Time", "Items with codes", "Subtotal", "Tax", "Total", "Payment method"],
                "layout_notes": "Walmart receipts have store info at top, items with department codes, tax calculation at bottom",
                "field_locations": "Tax calculation: SUBTOTAL → TAX → TOTAL. Payment method shows: VISA/MASTERCARD/CASH with last 4 digits",
                "special_characteristics": "Department codes like 'D001' appear before items. Savings/discounts shown separately. Barcode at bottom."
            },
            {
                "vendor_name": "Big Bazaar India",
                "country": "IN",
                "vendor_type": "grocery",
                "typical_fields": ["Store Name", "GSTIN", "Date", "Time", "Items", "Subtotal", "CGST", "SGST", "Total", "Payment mode"],
                "layout_notes": "Indian receipts typically show GSTIN number, separate CGST and SGST lines, rounded total",
                "field_locations": "GSTIN near top. GST breakdown: Taxable Amount, CGST @X%, SGST @X%, then Total. Rounding adjustment may appear.",
                "special_characteristics": "Uses CGST (Central GST) and SGST (State GST) for domestic transactions. IGST for inter-state. HSN codes may appear."
            },
            {
                "vendor_name": "Amazon UK",
                "country": "UK",
                "vendor_type": "ecommerce",
                "typical_fields": ["Order Number", "Invoice Number", "Date", "Delivery Address", "Items with VAT", "Subtotal", "Delivery", "VAT", "Total", "Payment method"],
                "layout_notes": "E-commerce receipts have order/invoice numbers, delivery info, may span multiple pages",
                "field_locations": "Order ID at top. Items with quantity and unit price. Delivery charge separate line. VAT summary before grand total.",
                "special_characteristics": "Digital receipt format. Includes delivery tracking info. Returns policy section. Email/PDF format."
            },
            {
                "vendor_name": "Shell Petrol UK",
                "country": "UK",
                "vendor_type": "petrol",
                "typical_fields": ["Station Name", "Pump Number", "Date", "Time", "Fuel Type", "Volume", "Price per litre", "Total", "VAT", "Payment method"],
                "layout_notes": "Petrol receipts focus on pump number, fuel volume, price per unit",
                "field_locations": "Pump number prominent. Volume in litres. Price shown as pence/litre and total. VAT usually at standard rate.",
                "special_characteristics": "Fuel grade (Unleaded, Diesel, Premium). Volume × price = total shown clearly. Loyalty card info if used."
            }
        ]
    
    def add_learned_pattern(
        self,
        pattern_content: str,
        pattern_type: str,
        vendor_type: Optional[str] = None,
        country: Optional[str] = None,
        **metadata
    ) -> str:
        """
        Add a new learned pattern from user feedback (for RAG improvement).
        
        Args:
            pattern_content: Description of the pattern
            pattern_type: Type of pattern (field_definition, tax_rule, layout_pattern)
            vendor_type: Optional vendor type
            country: Optional country code
            **metadata: Additional metadata
            
        Returns:
            Document ID of the added pattern
        """
        doc_id = self.vector_store.add_document(
            content=pattern_content,
            doc_type=pattern_type,
            vendor_type=vendor_type,
            country=country,
            learned=True,  # Mark as learned (not pre-loaded)
            **metadata
        )
        
        logger.info(f"Added learned pattern: {doc_id}")
        return doc_id


# Singleton instance
_kb_loader_instance = None


def get_knowledge_base_loader() -> KnowledgeBaseLoader:
    """Get or create singleton KnowledgeBaseLoader instance."""
    global _kb_loader_instance
    
    if _kb_loader_instance is None:
        _kb_loader_instance = KnowledgeBaseLoader()
    
    return _kb_loader_instance


def initialize_knowledge_base(force_reload: bool = False) -> Dict[str, int]:
    """
    Initialize the knowledge base with all patterns.
    
    Args:
        force_reload: If True, clear and reload all knowledge
        
    Returns:
        Dictionary with counts of loaded documents
    """
    loader = get_knowledge_base_loader()
    return loader.load_all(force_reload=force_reload)
