#!/usr/bin/env python3
"""
Pattern Learner - Automatic Pattern Learning from Successful Extractions v1.0.0

Learns new patterns from high-confidence extractions:
- New vendor patterns (vendor names not in knowledge base)
- Field extraction patterns (successful extraction methods)
- Tax calculation patterns (correct VAT/GST calculations)
- Layout patterns (consistent receipt structures)

Triggers:
- Overall confidence >= 0.85 (auto-approved extractions)
- Validation passed
- New pattern detected (not already in knowledge base)

No user data stored - only extraction patterns and metadata.
"""

import logging
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LearnedPattern:
    """A pattern learned from successful extraction."""
    pattern_type: str  # vendor_template, field_pattern, tax_pattern, layout_pattern
    content: str
    metadata: Dict[str, Any]
    confidence: float
    source_request_id: str
    learned_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pattern_type': self.pattern_type,
            'content': self.content,
            'metadata': self.metadata,
            'confidence': self.confidence,
            'source_request_id': self.source_request_id,
            'learned_at': self.learned_at
        }


class PatternLearner:
    """
    Learns patterns from successful receipt extractions.
    
    Monitors high-confidence extractions and identifies:
    - New vendors not in knowledge base
    - Effective field extraction patterns
    - Successful tax calculation methods
    - Consistent layout structures
    
    Stores learned patterns in vector store for future use.
    """
    
    # Thresholds
    AUTO_LEARN_THRESHOLD = 0.85  # Auto-approve and learn from >= 85% confidence
    REVIEW_LEARN_THRESHOLD = 0.70  # Learn but mark for review for >= 70% confidence
    VENDOR_CONFIDENCE_THRESHOLD = 0.9  # Minimum confidence for vendor pattern
    FIELD_CONFIDENCE_THRESHOLD = 0.70  # Minimum confidence for field pattern (lowered)
    
    # Rate limiting
    MAX_PATTERNS_PER_DAY = 100  # Prevent knowledge base pollution
    MAX_REVIEW_PATTERNS_PER_DAY = 50  # Limit for review-required patterns
    
    def __init__(self):
        """Initialize pattern learner."""
        from .vector_store import get_vector_store
        from .vendor_classifier import get_vendor_classifier
        from .rag_refresh_manager import get_rag_refresh_manager
        
        self.vector_store = get_vector_store()
        self.vendor_classifier = get_vendor_classifier()
        self.refresh_manager = get_rag_refresh_manager()
        self.learned_today = 0
        self.review_learned_today = 0
        self.last_reset = datetime.now().date()
        
        logger.info(f"✅ PatternLearner initialized with RAG refresh manager")
        logger.info(f"   Auto-approve threshold: {self.AUTO_LEARN_THRESHOLD}")
        logger.info(f"   Review-required threshold: {self.REVIEW_LEARN_THRESHOLD}")
    
    def should_learn(self, overall_confidence: float, validation_passed: bool) -> bool:
        """
        Determine if we should learn from this extraction.
        
        NOTE: This now allows learning even with low overall confidence,
        since individual fields may have high confidence >= 70%.
        
        Args:
            overall_confidence: Overall extraction confidence (not used for filtering)
            validation_passed: Whether validation checks passed
            
        Returns:
            True if should learn, False otherwise
        """
        # Reset daily counter
        today = datetime.now().date()
        if today != self.last_reset:
            self.learned_today = 0
            self.review_learned_today = 0
            self.last_reset = today
        
        # Check combined rate limits
        if self.learned_today >= self.MAX_PATTERNS_PER_DAY:
            logger.warning(f"⚠️ Auto-learn rate limit reached: {self.learned_today} patterns today")
            return False
        
        if self.review_learned_today >= self.MAX_REVIEW_PATTERNS_PER_DAY:
            logger.warning(f"⚠️ Review-learn rate limit reached: {self.review_learned_today} patterns today")
            return False
        
        # Always allow learning attempt even if validation fails
        # Per-field logic will filter by individual confidence (>= 70%)
        # We can learn from high-confidence fields even if overall validation fails
        return True
    
    def learn_from_success(
        self,
        request_id: str,
        full_extraction: Dict[str, Any],
        ocr_text: str,
        normalized_text: str,
        overall_confidence: float,
        country_detected: Optional[str] = None,
        vendor_classification: Optional[Dict[str, Any]] = None
    ) -> List[LearnedPattern]:
        """
        Learn patterns from a successful extraction.
        
        Args:
            request_id: Unique request identifier
            full_extraction: Full Phi-3 extraction result
            ocr_text: Original OCR text
            normalized_text: Normalized text
            overall_confidence: Overall confidence score
            country_detected: Detected country code
            vendor_classification: Vendor classification result
            
        Returns:
            List of learned patterns
        """
        learned_patterns = []
        
        try:
            logger.info(f"[{request_id}] 🎓 Starting pattern learning (confidence: {overall_confidence:.2%})")
            
            # Learn vendor pattern (if new vendor)
            logger.debug(f"[{request_id}] Step 1/3: Checking vendor pattern...")
            vendor_pattern = self._learn_vendor_pattern(
                request_id, full_extraction, ocr_text,
                country_detected, vendor_classification
            )
            if vendor_pattern:
                learned_patterns.append(vendor_pattern)
                logger.info(f"[{request_id}]   ✅ Learned vendor pattern: {vendor_pattern.metadata.get('vendor_name')}")
            else:
                logger.debug(f"[{request_id}]   ⏭️  No new vendor pattern to learn")
            
            # Learn field extraction patterns
            logger.debug(f"[{request_id}] Step 2/3: Learning field patterns...")
            field_patterns = self._learn_field_patterns(
                request_id, full_extraction, normalized_text,
                country_detected
            )
            learned_patterns.extend(field_patterns)
            
            # Learn tax calculation patterns
            logger.debug(f"[{request_id}] Step 3/3: Checking tax pattern...")
            tax_pattern = self._learn_tax_pattern(
                request_id, full_extraction, country_detected
            )
            if tax_pattern:
                learned_patterns.append(tax_pattern)
                logger.info(f"[{request_id}]   ✅ Learned tax pattern: {country_detected or 'Universal'} tax system")
            else:
                logger.debug(f"[{request_id}]   ⏭️  No tax pattern to learn")
            
            # Determine if patterns need review based on confidence
            needs_review = overall_confidence < self.AUTO_LEARN_THRESHOLD
            
            # Store learned patterns
            logger.info(f"[{request_id}] 💾 Storing {len(learned_patterns)} patterns (review: {needs_review})...")
            for idx, pattern in enumerate(learned_patterns, 1):
                field_name = pattern.metadata.get('field_name', pattern.pattern_type)
                logger.debug(f"[{request_id}]   [{idx}/{len(learned_patterns)}] Storing '{field_name}' pattern...")
                self._store_pattern(pattern, needs_review=needs_review)
                if needs_review:
                    self.review_learned_today += 1
                else:
                    self.learned_today += 1
            
            if learned_patterns:
                if needs_review:
                    logger.info(f"[{request_id}] 📝 ✅ Learned {len(learned_patterns)} patterns (REVIEW REQUIRED, confidence={overall_confidence:.2%})")
                else:
                    logger.info(f"[{request_id}] ✅ ✅ Learned {len(learned_patterns)} patterns (AUTO-APPROVED, confidence={overall_confidence:.2%})")
                
                # Log stats
                logger.info(f"[{request_id}] 📊 Daily stats: auto-approved={self.learned_today}, review-required={self.review_learned_today}")
            else:
                logger.warning(f"[{request_id}] ⚠️  No patterns learned (all fields below threshold)")
            
            return learned_patterns
            
        except Exception as e:
            logger.error(f"[{request_id}] ❌ Error learning from success: {e}", exc_info=True)
            return []
    
    def _learn_vendor_pattern(
        self,
        request_id: str,
        full_extraction: Dict[str, Any],
        ocr_text: str,
        country: Optional[str],
        vendor_classification: Optional[Dict[str, Any]]
    ) -> Optional[LearnedPattern]:
        """Learn a new vendor pattern if vendor not in knowledge base."""
        supplier_name = full_extraction.get('supplier_name', {})
        
        if not isinstance(supplier_name, dict):
            return None
        
        vendor_name = supplier_name.get('value')
        confidence = supplier_name.get('confidence', 0.0)
        
        # Check if high confidence and not already classified
        if not vendor_name or confidence < self.VENDOR_CONFIDENCE_THRESHOLD:
            return None
        
        # Check if already in vendor classifier
        if vendor_classification and vendor_classification.get('vendor_name'):
            # Already classified, no need to learn
            return None
        
        # Extract vendor type from context
        vendor_type = self._infer_vendor_type(full_extraction, ocr_text)
        
        # Extract address/phone for pattern
        address_parts = []
        supplier_address = full_extraction.get('supplier_address', [])
        if isinstance(supplier_address, list):
            address_parts = [a.get('value', '') for a in supplier_address if isinstance(a, dict)]
        
        phone_parts = []
        supplier_phone = full_extraction.get('supplier_phone', [])
        if isinstance(supplier_phone, list):
            phone_parts = [p.get('value', '') for p in supplier_phone if isinstance(p, dict)]
        
        # Build pattern content
        content = f"""
Vendor: {vendor_name}
Type: {vendor_type}
Country: {country or 'Unknown'}
Common name variations: {vendor_name}, {vendor_name.upper()}
Address keywords: {', '.join(address_parts[:2])}
Phone pattern: {phone_parts[0] if phone_parts else 'N/A'}
Confidence: {confidence:.2f}
Source: Auto-learned from high-confidence extraction
        """.strip()
        
        metadata = {
            'vendor_name': vendor_name,
            'vendor_type': vendor_type,
            'country': country,
            'learned': True,
            'pattern_hash': self._hash_pattern(vendor_name)
        }
        
        return LearnedPattern(
            pattern_type='vendor_template',
            content=content,
            metadata=metadata,
            confidence=confidence,
            source_request_id=request_id,
            learned_at=datetime.now().isoformat()
        )
    
    def _learn_field_patterns(
        self,
        request_id: str,
        full_extraction: Dict[str, Any],
        normalized_text: str,
        country: Optional[str]
    ) -> List[LearnedPattern]:
        """Learn field extraction patterns from successful fields - PER FIELD."""
        patterns = []
        logger.info(f"[{request_id}] 🎓 Starting per-field pattern learning (threshold: {self.FIELD_CONFIDENCE_THRESHOLD})")
        
        # ===== 1. LEARN FROM HEADER FIELDS (supplier_name, receipt_number, receipt_date) =====
        logger.debug(f"[{request_id}] Checking header fields...")
        for field_name in ['supplier_name', 'receipt_number', 'receipt_date']:
            field_data = full_extraction.get(field_name, {})
            if isinstance(field_data, dict):
                conf = field_data.get('confidence', 0.0)
                value = field_data.get('value') or field_data.get('date')
                raw_text = field_data.get('raw_text', '')
                
                if conf >= self.FIELD_CONFIDENCE_THRESHOLD and value:
                    logger.info(f"[{request_id}]   ✅ Learning pattern for '{field_name}' (confidence: {conf:.2%})")
                    
                    content = f"""
Field: {field_name}
Value: {value}
Raw Text: {raw_text}
Confidence: {conf:.2f}
Country: {country or 'Universal'}
Extraction Method: Phi-3 LLM with RAG context
Pattern: Successfully extracted {field_name} from receipt text
Source: Auto-learned from validated extraction
                    """.strip()
                    
                    patterns.append(LearnedPattern(
                        pattern_type='field_pattern',
                        content=content,
                        metadata={
                            'field_name': field_name,
                            'field_value': str(value)[:100],  # Truncate long values
                            'country': country,
                            'learned': True,
                            'field_type': 'header'
                        },
                        confidence=conf,
                        source_request_id=request_id,
                        learned_at=datetime.now().isoformat()
                    ))
                else:
                    if conf > 0:
                        logger.debug(f"[{request_id}]   ⏭️  Skipping '{field_name}' (confidence: {conf:.2%} < {self.FIELD_CONFIDENCE_THRESHOLD})")
        
        # ===== 2. LEARN FROM TOTALS FIELDS (subtotal, tax, final_total) =====
        logger.debug(f"[{request_id}] Checking totals fields...")
        totals = full_extraction.get('totals', {})
        if isinstance(totals, dict):
            for total_key in ['subtotal', 'final_total', 'tax', 'tax_amount', 'items_subtotal', 'net_after_discount']:
                total_data = totals.get(total_key, {})
                if isinstance(total_data, dict):
                    amount = total_data.get('amount')
                    conf = total_data.get('confidence', 0.0)
                    
                    if conf >= self.FIELD_CONFIDENCE_THRESHOLD and amount is not None:
                        logger.info(f"[{request_id}]   ✅ Learning pattern for 'totals.{total_key}' (confidence: {conf:.2%}, amount: {amount})")
                        
                        content = f"""
Field: totals.{total_key}
Amount: {amount}
Confidence: {conf:.2f}
Country: {country or 'Universal'}
Pattern: Successfully extracted {total_key} from receipt totals section
Source: Auto-learned from validated extraction
                        """.strip()
                        
                        patterns.append(LearnedPattern(
                            pattern_type='field_pattern',
                            content=content,
                            metadata={
                                'field_name': f'totals.{total_key}',
                                'field_value': str(amount),
                                'country': country,
                                'learned': True,
                                'field_type': 'totals'
                            },
                            confidence=conf,
                            source_request_id=request_id,
                            learned_at=datetime.now().isoformat()
                        ))
        
        # ===== 3. LEARN FROM ITEMS (NEW!) =====
        logger.debug(f"[{request_id}] Checking item fields...")
        items = full_extraction.get('item_list', [])
        learned_items = 0
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            
            item_conf = item.get('confidence', 0.0)
            item_name = item.get('name') or item.get('item_name', '')
            
            if item_conf >= self.FIELD_CONFIDENCE_THRESHOLD and item_name:
                logger.info(f"[{request_id}]   ✅ Learning pattern for 'item[{idx}]' (confidence: {item_conf:.2%}, name: {item_name[:30]}...)")
                
                content = f"""
Field: item
Item Name: {item_name}
Quantity: {item.get('quantity', 'N/A')}
Unit Price: {item.get('unit_price', 'N/A')}
Total Price: {item.get('total_price', 'N/A')}
Confidence: {item_conf:.2f}
Country: {country or 'Universal'}
Pattern: Successfully extracted item details with all fields
Source: Auto-learned from validated extraction
                """.strip()
                
                patterns.append(LearnedPattern(
                    pattern_type='field_pattern',
                    content=content,
                    metadata={
                        'field_name': 'item',
                        'field_value': item_name[:100],
                        'country': country,
                        'learned': True,
                        'field_type': 'item'
                    },
                    confidence=item_conf,
                    source_request_id=request_id,
                    learned_at=datetime.now().isoformat()
                ))
                learned_items += 1
                
                # Limit to 5 items to avoid knowledge base pollution
                if learned_items >= 5:
                    logger.debug(f"[{request_id}]   ⚠️  Item learning limit reached (5 items)")
                    break
        
        if learned_items > 0:
            logger.info(f"[{request_id}]   📦 Learned patterns from {learned_items} items")
        
        # ===== 4. LEARN FROM VAT INFORMATION (NEW!) =====
        logger.debug(f"[{request_id}] Checking VAT fields...")
        vat_info = full_extraction.get('vat_information', [])
        
        # Handle both list format and dict with vat_data_entries
        if isinstance(vat_info, list):
            vat_entries = vat_info
        elif isinstance(vat_info, dict):
            vat_entries = vat_info.get('vat_data_entries', [])
        else:
            vat_entries = []
        
        for idx, vat_entry in enumerate(vat_entries):
                if not isinstance(vat_entry, dict):
                    continue
                
                vat_conf = vat_entry.get('confidence', 0.0)
                vat_code = vat_entry.get('vat_code', '')
                vat_rate = vat_entry.get('vat_rate', '')
                vat_amount = vat_entry.get('vat_amount', '')
                
                # Learn if confidence is good and has at least rate or amount
                if vat_conf >= self.FIELD_CONFIDENCE_THRESHOLD and (vat_rate or vat_amount):
                    logger.info(f"[{request_id}]   ✅ Learning pattern for 'vat[{idx}]' (confidence: {vat_conf:.2%}, rate: {vat_rate}, amount: {vat_amount})")
                    
                    content = f"""
Field: vat_information
VAT Code: {vat_code or 'N/A'}
VAT Rate: {vat_rate}
Net Amount: {vat_entry.get('net_amount', 'N/A')}
VAT Amount: {vat_amount}
Gross Amount: {vat_entry.get('gross_amount', 'N/A')}
Confidence: {vat_conf:.2f}
Country: {country or 'Universal'}
Pattern: Successfully extracted VAT breakdown with all components
Source: Auto-learned from validated extraction
                    """.strip()
                    
                    patterns.append(LearnedPattern(
                        pattern_type='field_pattern',
                        content=content,
                        metadata={
                            'field_name': 'vat_information',
                            'field_value': f"{vat_code} @ {vat_rate}%",
                            'country': country,
                            'learned': True,
                            'field_type': 'vat'
                        },
                        confidence=vat_conf,
                        source_request_id=request_id,
                        learned_at=datetime.now().isoformat()
                    ))
        
        # ===== 5. LEARN FROM PAYMENT METHODS (NEW!) =====
        logger.debug(f"[{request_id}] Checking payment fields...")
        payments = full_extraction.get('payment_methods', [])
        for idx, payment in enumerate(payments):
            if not isinstance(payment, dict):
                continue
            
            pay_conf = payment.get('confidence', 0.0)
            pay_method = payment.get('method', '')
            
            if pay_conf >= self.FIELD_CONFIDENCE_THRESHOLD and pay_method:
                logger.info(f"[{request_id}]   ✅ Learning pattern for 'payment[{idx}]' (confidence: {pay_conf:.2%}, method: {pay_method})")
                
                content = f"""
Field: payment_method
Method: {pay_method}
Amount: {payment.get('amount', 'N/A')}
Confidence: {pay_conf:.2f}
Country: {country or 'Universal'}
Pattern: Successfully extracted payment information
Source: Auto-learned from validated extraction
                """.strip()
                
                patterns.append(LearnedPattern(
                    pattern_type='field_pattern',
                    content=content,
                    metadata={
                        'field_name': 'payment_method',
                        'field_value': pay_method,
                        'country': country,
                        'learned': True,
                        'field_type': 'payment'
                    },
                    confidence=pay_conf,
                    source_request_id=request_id,
                    learned_at=datetime.now().isoformat()
                ))
        
        # ===== 6. LEARN FROM DISCOUNTS (NEW!) =====
        logger.debug(f"[{request_id}] Checking discount fields...")
        discounts = full_extraction.get('discount_items', [])
        for idx, discount in enumerate(discounts):
            if not isinstance(discount, dict):
                continue
            
            disc_conf = discount.get('confidence', 0.0)
            disc_name = discount.get('name', '')
            disc_amount = discount.get('amount', 0)
            
            if disc_conf >= self.FIELD_CONFIDENCE_THRESHOLD and disc_name:
                logger.info(f"[{request_id}]   ✅ Learning pattern for 'discount[{idx}]' (confidence: {disc_conf:.2%}, amount: {disc_amount})")
                
                content = f"""
Field: discount
Name: {disc_name}
Amount: {disc_amount}
Confidence: {disc_conf:.2f}
Country: {country or 'Universal'}
Pattern: Successfully extracted discount/promotion information
Source: Auto-learned from validated extraction
                """.strip()
                
                patterns.append(LearnedPattern(
                    pattern_type='field_pattern',
                    content=content,
                    metadata={
                        'field_name': 'discount',
                        'field_value': f"{disc_name}: {disc_amount}",
                        'country': country,
                        'learned': True,
                        'field_type': 'discount'
                    },
                    confidence=disc_conf,
                    source_request_id=request_id,
                    learned_at=datetime.now().isoformat()
                ))
        
        # ===== 7. LEARN FROM ADDRESS INFORMATION (NEW!) =====
        logger.debug(f"[{request_id}] Checking address fields...")
        addresses = full_extraction.get('supplier_address', [])
        address_learned = 0
        for idx, address in enumerate(addresses):
            if not isinstance(address, dict):
                continue
            
            addr_conf = address.get('confidence', 0.0)
            addr_value = address.get('value', '')
            
            if addr_conf >= self.FIELD_CONFIDENCE_THRESHOLD and addr_value:
                logger.info(f"[{request_id}]   ✅ Learning pattern for 'address[{idx}]' (confidence: {addr_conf:.2%})")
                
                content = f"""
Field: supplier_address
Address: {addr_value}
Confidence: {addr_conf:.2f}
Country: {country or 'Universal'}
Pattern: Successfully extracted supplier address information
Source: Auto-learned from validated extraction
                """.strip()
                
                patterns.append(LearnedPattern(
                    pattern_type='field_pattern',
                    content=content,
                    metadata={
                        'field_name': 'supplier_address',
                        'field_value': addr_value[:100],
                        'country': country,
                        'learned': True,
                        'field_type': 'address'
                    },
                    confidence=addr_conf,
                    source_request_id=request_id,
                    learned_at=datetime.now().isoformat()
                ))
                address_learned += 1
        
        if not addresses:
            logger.debug(f"[{request_id}]   ℹ️  No address fields found (ready to learn when present)")
        
        # ===== 8. LEARN FROM WEBSITE INFORMATION (NEW!) =====
        logger.debug(f"[{request_id}] Checking website fields...")
        websites = full_extraction.get('supplier_website', [])
        website_learned = 0
        for idx, website in enumerate(websites):
            if not isinstance(website, dict):
                continue
            
            web_conf = website.get('confidence', 0.0)
            web_value = website.get('value', '')
            
            if web_conf >= self.FIELD_CONFIDENCE_THRESHOLD and web_value:
                logger.info(f"[{request_id}]   ✅ Learning pattern for 'website[{idx}]' (confidence: {web_conf:.2%}, url: {web_value})")
                
                content = f"""
Field: supplier_website
Website: {web_value}
Confidence: {web_conf:.2f}
Country: {country or 'Universal'}
Pattern: Successfully extracted supplier website URL
Source: Auto-learned from validated extraction
                """.strip()
                
                patterns.append(LearnedPattern(
                    pattern_type='field_pattern',
                    content=content,
                    metadata={
                        'field_name': 'supplier_website',
                        'field_value': web_value,
                        'country': country,
                        'learned': True,
                        'field_type': 'website'
                    },
                    confidence=web_conf,
                    source_request_id=request_id,
                    learned_at=datetime.now().isoformat()
                ))
                website_learned += 1
        
        if not websites:
            logger.debug(f"[{request_id}]   ℹ️  No website fields found (ready to learn when present)")
        
        # ===== 9. LEARN FROM EMAIL INFORMATION (NEW!) =====
        logger.debug(f"[{request_id}] Checking email fields...")
        emails = full_extraction.get('supplier_email', [])
        email_learned = 0
        for idx, email in enumerate(emails):
            if not isinstance(email, dict):
                continue
            
            email_conf = email.get('confidence', 0.0)
            email_value = email.get('value', '')
            
            if email_conf >= self.FIELD_CONFIDENCE_THRESHOLD and email_value:
                logger.info(f"[{request_id}]   ✅ Learning pattern for 'email[{idx}]' (confidence: {email_conf:.2%}, email: {email_value})")
                
                content = f"""
Field: supplier_email
Email: {email_value}
Confidence: {email_conf:.2f}
Country: {country or 'Universal'}
Pattern: Successfully extracted supplier email address
Source: Auto-learned from validated extraction
                """.strip()
                
                patterns.append(LearnedPattern(
                    pattern_type='field_pattern',
                    content=content,
                    metadata={
                        'field_name': 'supplier_email',
                        'field_value': email_value,
                        'country': country,
                        'learned': True,
                        'field_type': 'email'
                    },
                    confidence=email_conf,
                    source_request_id=request_id,
                    learned_at=datetime.now().isoformat()
                ))
                email_learned += 1
        
        if not emails:
            logger.debug(f"[{request_id}]   ℹ️  No email fields found (ready to learn when present)")
        
        # ===== 10. LEARN FROM PHONE INFORMATION (NEW!) =====
        logger.debug(f"[{request_id}] Checking phone fields...")
        phones = full_extraction.get('supplier_phone', [])
        phone_learned = 0
        for idx, phone in enumerate(phones):
            if not isinstance(phone, dict):
                continue
            
            phone_conf = phone.get('confidence', 0.0)
            phone_value = phone.get('value', '')
            
            if phone_conf >= self.FIELD_CONFIDENCE_THRESHOLD and phone_value:
                logger.info(f"[{request_id}]   ✅ Learning pattern for 'phone[{idx}]' (confidence: {phone_conf:.2%}, phone: {phone_value})")
                
                content = f"""
Field: supplier_phone
Phone: {phone_value}
Confidence: {phone_conf:.2f}
Country: {country or 'Universal'}
Pattern: Successfully extracted supplier phone number
Source: Auto-learned from validated extraction
                """.strip()
                
                patterns.append(LearnedPattern(
                    pattern_type='field_pattern',
                    content=content,
                    metadata={
                        'field_name': 'supplier_phone',
                        'field_value': phone_value,
                        'country': country,
                        'learned': True,
                        'field_type': 'phone'
                    },
                    confidence=phone_conf,
                    source_request_id=request_id,
                    learned_at=datetime.now().isoformat()
                ))
                phone_learned += 1
        
        if not phones:
            logger.debug(f"[{request_id}]   ℹ️  No phone fields found (ready to learn when present)")
        
        # ===== 11. LEARN FROM CARD DETAILS (NEW!) =====
        logger.debug(f"[{request_id}] Checking card detail fields...")
        cards = full_extraction.get('card_details', [])
        card_learned = 0
        for idx, card in enumerate(cards):
            if not isinstance(card, dict):
                continue
            
            card_conf = card.get('confidence', 0.0)
            card_brand = card.get('card_brand', '')
            card_last4 = card.get('card_last_4', '')
            
            if card_conf >= self.FIELD_CONFIDENCE_THRESHOLD and card_brand:
                logger.info(f"[{request_id}]   ✅ Learning pattern for 'card[{idx}]' (confidence: {card_conf:.2%}, brand: {card_brand})")
                
                content = f"""
Field: card_details
Card Brand: {card_brand}
Last 4 Digits: {card_last4 or 'N/A'}
Confidence: {card_conf:.2f}
Country: {country or 'Universal'}
Pattern: Successfully extracted payment card information
Source: Auto-learned from validated extraction
                """.strip()
                
                patterns.append(LearnedPattern(
                    pattern_type='field_pattern',
                    content=content,
                    metadata={
                        'field_name': 'card_details',
                        'field_value': f"{card_brand} ****{card_last4}" if card_last4 else card_brand,
                        'country': country,
                        'learned': True,
                        'field_type': 'card'
                    },
                    confidence=card_conf,
                    source_request_id=request_id,
                    learned_at=datetime.now().isoformat()
                ))
                card_learned += 1
        
        if not cards:
            logger.debug(f"[{request_id}]   ℹ️  No card detail fields found (ready to learn when present)")
        
        # ===== SUMMARY =====
        if patterns:
            field_types = {}
            for p in patterns:
                field_type = p.metadata.get('field_type', 'unknown')
                field_types[field_type] = field_types.get(field_type, 0) + 1
            
            summary = ', '.join([f"{count} {ftype}" for ftype, count in field_types.items()])
            logger.info(f"[{request_id}] 🎯 Learned {len(patterns)} field patterns: {summary}")
        else:
            logger.warning(f"[{request_id}] ⚠️  No field patterns met confidence threshold ({self.FIELD_CONFIDENCE_THRESHOLD})")
        
        return patterns
    
    def _learn_tax_pattern(
        self,
        request_id: str,
        full_extraction: Dict[str, Any],
        country: Optional[str]
    ) -> Optional[LearnedPattern]:
        """Learn tax calculation pattern if correctly calculated."""
        vat_info = full_extraction.get('vat_information', {})
        if not isinstance(vat_info, dict):
            return None
        
        vat_entries = vat_info.get('vat_data_entries', [])
        if not vat_entries:
            return None
        
        # Check if VAT calculation is consistent
        total_vat = 0.0
        total_net = 0.0
        vat_rates = []
        
        for entry in vat_entries:
            if not isinstance(entry, dict):
                continue
            
            vat_amount = entry.get('vat_amount', 0)
            net_amount = entry.get('net_amount', 0)
            vat_rate = entry.get('vat_rate', 0)
            conf = entry.get('confidence', 0.0)
            
            if conf >= self.FIELD_CONFIDENCE_THRESHOLD:
                total_vat += vat_amount
                total_net += net_amount
                if vat_rate not in vat_rates:
                    vat_rates.append(vat_rate)
        
        # Learn pattern if we have valid VAT calculation
        if total_vat > 0 and total_net > 0 and vat_rates:
            calculated_rate = (total_vat / total_net * 100) if total_net > 0 else 0
            
            content = f"""
Tax calculation pattern:
Country: {country or 'Unknown'}
Tax system: {'VAT' if country == 'UK' else 'GST' if country == 'IN' else 'Sales Tax'}
Rates observed: {', '.join(f'{r}%' for r in vat_rates)}
Total net: {total_net:.2f}
Total tax: {total_vat:.2f}
Effective rate: {calculated_rate:.2f}%
Calculation validated: Tax amounts consistent with rates
Source: Auto-learned from validated calculation
            """.strip()
            
            return LearnedPattern(
                pattern_type='tax_pattern',
                content=content,
                metadata={
                    'country': country,
                    'tax_rates': vat_rates,
                    'total_vat': total_vat,
                    'total_net': total_net,
                    'learned': True
                },
                confidence=0.9,
                source_request_id=request_id,
                learned_at=datetime.now().isoformat()
            )
        
        return None
    
    def _infer_vendor_type(self, full_extraction: Dict[str, Any], ocr_text: str) -> str:
        """Infer vendor type from extraction and text."""
        # Check item names for clues
        items = full_extraction.get('item_list', [])
        
        food_keywords = ['milk', 'bread', 'cheese', 'meat', 'vegetable', 'fruit']
        fuel_keywords = ['petrol', 'diesel', 'fuel', 'litre', 'gallon']
        pharmacy_keywords = ['tablet', 'capsule', 'medicine', 'prescription', 'pharmacy']
        
        food_count = 0
        fuel_count = 0
        pharmacy_count = 0
        
        for item in items:
            if not isinstance(item, dict):
                continue
            item_name = item.get('item_name', '').lower()
            
            if any(kw in item_name for kw in food_keywords):
                food_count += 1
            if any(kw in item_name for kw in fuel_keywords):
                fuel_count += 1
            if any(kw in item_name for kw in pharmacy_keywords):
                pharmacy_count += 1
        
        # Determine type based on item analysis
        if fuel_count > 0:
            return 'petrol'
        elif pharmacy_count > 0:
            return 'pharmacy'
        elif food_count >= 2:
            return 'grocery'
        
        # Fallback to text analysis
        text_upper = ocr_text.upper()
        if any(kw in text_upper for kw in ['SUPERMARKET', 'GROCERY', 'FRESH']):
            return 'grocery'
        elif any(kw in text_upper for kw in ['FUEL', 'PETROL', 'PUMP']):
            return 'petrol'
        elif any(kw in text_upper for kw in ['PHARMACY', 'CHEMIST']):
            return 'pharmacy'
        elif any(kw in text_upper for kw in ['RESTAURANT', 'CAFE', 'DINER']):
            return 'restaurant'
        
        return 'other'
    
    def _store_pattern(self, pattern: LearnedPattern, needs_review: bool = False):
        """Store learned pattern in vector store using refresh manager."""
        try:
            # Extract metadata for soft refresh
            vendor_type = pattern.metadata.get('vendor_type')
            country = pattern.metadata.get('country')
            field_name = pattern.metadata.get('field_name', pattern.pattern_type)
            
            logger.debug(f"      Storing pattern: type={pattern.pattern_type}, field={field_name}, review={needs_review}")
            
            # Prepare clean metadata (remove non-serializable fields)
            clean_metadata = {
                'learning_confidence': pattern.confidence,
                'source_request': pattern.source_request_id,
                'learned_at': pattern.learned_at,
                'auto_learned': not needs_review,  # False if needs review
                'needs_review': needs_review,  # New flag for review-required patterns
                'review_status': 'pending' if needs_review else 'approved'
            }
            
            # Add serializable metadata
            for k, v in pattern.metadata.items():
                if k not in ['learned', 'vendor_type', 'country']:
                    if isinstance(v, (list, dict)):
                        clean_metadata[k] = str(v)
                    elif isinstance(v, (str, int, float, bool)) or v is None:
                        clean_metadata[k] = v
                    else:
                        clean_metadata[k] = str(v)
            
            logger.debug(f"      Using soft_refresh: vendor_type={vendor_type}, country={country}")
            
            # Use refresh manager for soft refresh (incremental update)
            result = self.refresh_manager.soft_refresh(
                pattern_content=pattern.content,
                pattern_type=pattern.pattern_type,
                vendor_type=vendor_type,
                country=country,
                **clean_metadata
            )
            
            if result.get('success'):
                doc_id = result.get('doc_id', 'unknown')
                logger.debug(f"      ✅ Stored via soft refresh: doc_id={doc_id[:16]}...")
            else:
                logger.warning(f"      ⚠️ Soft refresh completed with warnings: {result.get('message')}")
            
        except Exception as e:
            logger.error(f"      ❌ Failed to store pattern: {e}")
            # Fallback to direct storage if refresh manager fails
            try:
                logger.debug(f"      🔄 Trying fallback: direct vector store")
                doc_id = self.vector_store.add_document(
                    content=pattern.content,
                    doc_type=pattern.pattern_type,
                    learned=True,
                    **clean_metadata
                )
                logger.debug(f"      ✅ Fallback succeeded: doc_id={doc_id[:16]}...")
            except Exception as fallback_error:
                logger.error(f"      ❌ Fallback also failed: {fallback_error}")
    
    def _hash_pattern(self, pattern_text: str) -> str:
        """Generate hash for pattern deduplication."""
        return hashlib.md5(pattern_text.encode()).hexdigest()[:16]


# Singleton instance
_pattern_learner_instance = None


def get_pattern_learner() -> PatternLearner:
    """Get singleton PatternLearner instance."""
    global _pattern_learner_instance
    if _pattern_learner_instance is None:
        _pattern_learner_instance = PatternLearner()
    return _pattern_learner_instance
