"""
Hybrid Supplier Extraction Module v1.0.0

Combines geometric intelligence (coordinate-based analysis) with business pattern recognition
for superior invoice supplier name extraction.

Priority Chain:
1. BUSINESS PATTERNS with geometric scoring (NEW - most reliable)
2. LIMITED/LTD with overlap logic (existing - good for UK businesses)
3. EMAIL domain (validated against patterns - fallback)
4. WWW domain (fallback)
5. Height-based (last resort)

Features:
- 30+ business entity patterns (Limited, Ltd, Solutions, Services, etc.)
- Position-aware scoring (top of page = higher confidence)
- Address context detection (supplier followed by address = bonus)
- Enhanced filtering (dates, addresses, contact info)
- Word count validation (ideal length 2-5 words)
- Multi-factor scoring (patterns + position + OCR confidence + context)
"""

import re
from typing import Dict, List, Tuple, Optional


def _normalize_text(s):
    """Remove non-alphanumeric characters for comparison"""
    return re.sub(r'[^a-z0-9]', '', s.lower())


def _has_min_contiguous_prefix_match(top_text, email_local, min_len=2):
    """Check if email local part starts with contiguous prefix from top text"""
    top_clean = _normalize_text(top_text)
    email_clean = _normalize_text(email_local)
    
    if len(top_clean) < min_len:
        return False
    
    return email_clean.startswith(top_clean[:min_len])


class HybridSupplierExtractor:
    """Hybrid supplier extraction using geometric + pattern recognition"""
    
    def __init__(self):
        # Invalid words to filter out
        self.invalid_keywords = [
            "invoice", "document", "date", "invno", "vat", "subtotal",
            "ref", "ph:", "customer", "payment", "details", "postcode",
            "street", "avenue", "road", "suite", "house", "bank",
            "days", "months", "weeks", "hours", "terms", "duration",
            "project", "order", "description", "amount", "rate", "total",
            "evaluation", "warning", "created", "spire", "pdf",
            "bill to", "ship to", "deliver to", "remit to"
        ]
        
        # Address field keywords for customer name detection (prioritized)
        self.address_field_keywords = {
            'billing': [
                "bill to", "billed to", "billing to", "billing address",
                "bill address", "billing information", "invoice to",
                "bill-to", "billto", "account name", "account holder",
                "invoice address", "invoiced to", "billing details"
            ],
            'shipping': [
                "ship to", "shipped to", "shipping to", "shipping address",
                "delivery to", "deliver to", "delivery address",
                "ship-to", "shipto", "consignee", "recipient",
                "deliver address", "deliver-to", "consignee address",
                "recipient address", "despatch to", "dispatch to",
                "destination", "destination address"
            ],
            'customer': [
                "customer", "customer name", "client", "client name",
                "customer details", "purchaser", "buyer",
                "customer information", "client information",
                "customer account", "client account", "sold to"
            ],
            'attention': [
                "attention", "attn", "attn:", "care of", "c/o",
                "contact person", "attention to", "for the attention of",
                "attention of"
            ]
        }
        
        # Customer name exclusions (don't extract these as names)
        self.customer_name_exclusions = [
            "attention", "attn", "customer", "client", "ship to",
            "bill to", "address", "company", "contact", "dear",
            "re:", "ref:", "invoice", "account", "billing", "shipping",
            "delivery", "consignee", "recipient", "buyer", "purchaser",
            "sold to", "information", "details"
        ]
        
        # Business entity patterns (hierarchical scoring)
        self.business_patterns = [
            # HIGHEST PRIORITY: Complete business names (0.7-0.8)
            (r'\b\w+\s+\w+\s+(LIMITED|LTD|SOLUTIONS|SERVICES|SYSTEMS)\b', 0.8, "named_business_entity"),
            (r'\b\w+\s+(SOLUTIONS|SERVICES|SYSTEMS|TECHNOLOGIES)\b', 0.7, "tech_business"),
            
            # VERY HIGH: Legal entities (0.5-0.6)
            (r'\b(LIMITED|LTD|PLC|INC|CORPORATION|CORP|LLC|GMBH)\b', 0.6, "legal_entity"),
            (r'\b(INCORPORATED|COMPANY)\b', 0.5, "legal_entity_full"),
            
            # HIGH: Professional services (0.4-0.5)
            (r'\b(CONSULTING|PARTNERS|ASSOCIATES)\b', 0.5, "professional_services"),
            (r'\b(CONSULTING\s+GROUP|CONSULTING\s+SERVICES)\b', 0.5, "consulting_business"),
            
            # HIGH: Industry types (0.4-0.5)
            (r'\b(MANUFACTURING|INDUSTRIES|INDUSTRIAL)\b', 0.5, "manufacturing"),
            (r'\b(LOGISTICS|TRANSPORT|SHIPPING|FREIGHT)\b', 0.5, "logistics"),
            (r'\b(ENGINEERING|CONSTRUCTION|CONTRACTORS)\b', 0.5, "engineering"),
            
            # MEDIUM-HIGH: Business groups (0.35-0.45)
            (r'\b(ENTERPRISES|HOLDINGS|GROUP|INTERNATIONAL)\b', 0.45, "business_group"),
            (r'\b(GLOBAL|WORLDWIDE|NATIONAL)\b', 0.35, "scope_indicator"),
            
            # MEDIUM: Business indicators (0.25-0.35)
            (r'\b(&|\bAND\b)\b', 0.3, "partnership"),  # "Smith & Sons"
            (r'\b(CO\.|COMPANY)\b', 0.3, "company_indicator"),
            
            # MEDIUM: Supply/Trade (0.3-0.4)
            (r'\b(SUPPLIES|WHOLESALE|DISTRIBUTION|TRADING)\b', 0.4, "supply_business"),
            (r'\b(RETAIL|SHOP|STORE|MARKET)\b', 0.3, "retail"),
            
            # LOWER: Format indicators (0.1-0.2)
            (r'^[A-Z][a-z]+(\s+[A-Z][a-z]+){1,3}$', 0.2, "proper_name_format"),  # Title case
            (r'^[A-Z]{2,}\s+[A-Z][a-z]+', 0.15, "acronym_name"),  # "ABC Solutions"
            (r'[A-Z]{2,}', 0.1, "has_uppercase"),
        ]
        
        # Skip patterns for filtering
        self.skip_patterns = [
            r'^\d+$',  # Just numbers
            r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',  # Date patterns
            r'^\d{2}:\d{2}',  # Time patterns
            r'^TEL:?|^PHONE:?|^EMAIL:?|^WEB:?|^FAX:?',  # Contact prefixes
            r'^\d+\s+\w+\s*(STREET|ROAD|LANE|AVENUE|DRIVE)',  # Address lines
            r'^[A-Z]{1,3}\d+\s*\d*[A-Z]{0,3}$',  # Postcodes
            r'@.*\.',  # Email addresses
            r'^(BILL\s+TO|SHIP\s+TO|DELIVER\s+TO)',  # Delivery headers
            r'^\d{2}\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)',  # Date with month
        ]
        
        # Address patterns for context detection
        self.address_patterns = [
            r'\b\d+\s+\w+\s*(STREET|ROAD|LANE|AVENUE|DRIVE|ST|RD)\b',
            r'^[A-Z]{1,3}\d+\s*\d*[A-Z]{0,3}$',  # UK postcodes
            r'\b(TEL|PHONE|EMAIL|FAX):',
            r'^\d{5}(-\d{4})?$',  # US ZIP
            r'^\w+,?\s*[A-Z]{2,}$',  # City, State
        ]
    
    def is_invalid_supplier_text(self, text: str) -> bool:
        """Check if text should be filtered out"""
        text_upper = text.upper().strip()
        
        # Check keywords
        if any(kw in text_upper for kw in self.invalid_keywords):
            return True
        
        # Check patterns
        if any(re.search(pattern, text_upper) for pattern in self.skip_patterns):
            return True
        
        return False
    
    def score_supplier_by_business_patterns(self, text: str, y_position: float, 
                                           page_height: float) -> Tuple[float, List[str]]:
        """
        Score potential supplier text using business pattern recognition.
        
        Returns: (score, reasons_list)
        """
        if not text or len(text.strip()) < 2:
            return 0.0, []
        
        text_clean = text.upper().strip()
        score = 0.0
        reasons = []
        
        # Position bonus (top of page = more likely supplier)
        position_ratio = y_position / page_height if page_height > 0 else 0
        if position_ratio < 0.15:  # Top 15%
            score += 0.4
            reasons.append("top_position")
        elif position_ratio < 0.3:  # Top 30%
            score += 0.2
            reasons.append("upper_position")
        
        # Apply business patterns
        for pattern, bonus, reason in self.business_patterns:
            if re.search(pattern, text_clean):
                score += bonus
                reasons.append(reason)
        
        # Word count validation
        word_count = len(text.split())
        if 2 <= word_count <= 5:  # Ideal length
            score += 0.15
            reasons.append("ideal_length")
        elif word_count == 1:
            score -= 0.1
        elif word_count > 8:
            score -= 0.2
            reasons.append("too_long")
        
        return score, reasons
    
    def has_address_context_below(self, word: Dict, all_words: List[Dict], 
                                  max_distance: int = 100) -> Tuple[bool, float]:
        """
        Check if word is followed by address-like patterns.
        Returns: (has_address, address_score)
        """
        word_bottom = word['y1']
        address_indicators = 0
        
        # Look for words below within max_distance
        words_below = [
            w for w in all_words 
            if w['y0'] > word_bottom and w['y0'] < word_bottom + max_distance
        ]
        
        for w in words_below:
            text_upper = w['text'].upper()
            if any(re.search(pattern, text_upper) for pattern in self.address_patterns):
                address_indicators += 1
        
        address_score = min(address_indicators * 0.15, 0.3)
        return address_indicators > 0, address_score
    
    def is_customer_name_exclusion(self, text: str) -> bool:
        """Check if text should be excluded as customer name"""
        text_lower = text.lower().strip()
        
        # Check exclusion keywords
        for exclusion in self.customer_name_exclusions:
            if exclusion in text_lower:
                return True
        
        # Check if it's just punctuation or very short
        if len(text_lower) < 2:
            return True
        
        # Check if it's only special characters/punctuation
        if re.match(r'^[^a-zA-Z0-9]+$', text):
            return True
        
        return False
    
    def is_address_line(self, text: str) -> bool:
        """Check if text is an address line (not a name)"""
        text_upper = text.upper().strip()
        
        # Street patterns
        street_patterns = [
            r'\b\d+\s+\w+\s*(STREET|ROAD|LANE|AVENUE|DRIVE|WAY|CLOSE|ST|RD|AVE|LN|DR)\b',
            r'\b(SUITE|UNIT|FLOOR|LEVEL|BUILDING|BLOCK|FLAT|APT|APARTMENT)\s+\d+',
            r'\bP\.?O\.?\s+BOX\s+\d+',
        ]
        
        # Postcode patterns (UK and US)
        postcode_patterns = [
            r'^[A-Z]{1,3}\d+\s*\d*[A-Z]{0,3}$',  # UK postcode
            r'^\d{5}(-\d{4})?$',  # US ZIP
            r'^\d{4,6}$',  # Generic numeric postcode
        ]
        
        # Contact patterns
        contact_patterns = [
            r'^(TEL|PHONE|FAX|EMAIL|WEB|WWW|MOBILE|MOB):',
            r'^\+?\d[\d\s\-\(\)]{7,}$',  # Phone number
            r'@.*\.',  # Email
            r'^www\.',  # Website
        ]
        
        # City/region patterns (common suffixes)
        location_patterns = [
            r',\s*[A-Z]{2}\s*\d{5}',  # City, STATE ZIP
            r'\b(LONDON|MANCHESTER|BIRMINGHAM|LEEDS|GLASGOW|LIVERPOOL)\b',  # UK cities
            r'\b(NEW YORK|LOS ANGELES|CHICAGO|HOUSTON|SAN FRANCISCO)\b',  # US cities
        ]
        
        all_patterns = street_patterns + postcode_patterns + contact_patterns + location_patterns
        
        return any(re.search(pattern, text_upper) for pattern in all_patterns)
    
    def detect_customer_from_address_fields(self, words: List[Dict], 
                                           page_height: float) -> Tuple[Optional[str], float, str]:
        """
        Detect customer name from labeled address fields (Bill To, Ship To, etc.)
        
        Strategy:
        1. Find address field keywords (bill to, ship to, customer, etc.)
        2. Get the first line of text BELOW the keyword
        3. Validate it's a name (not "Address:", not postcode, etc.)
        4. Return with high confidence
        
        Returns: (customer_name, confidence, method)
        """
        print(f"[CUSTOMER_DETECTION] Searching for labeled address fields...")
        
        # Search top 60% of page (address fields usually in upper section)
        search_threshold = page_height * 0.6
        search_words = [w for w in words if w['y0'] < search_threshold]
        
        candidates = []
        
        # Iterate through all address field categories (prioritized order)
        for category, keywords in self.address_field_keywords.items():
            for word in search_words:
                text_lower = word['text'].lower().strip()
                
                # Check if this word contains an address keyword
                keyword_found = None
                for keyword in keywords:
                    if keyword in text_lower:
                        keyword_found = keyword
                        break
                
                if not keyword_found:
                    continue
                
                print(f"   Found address keyword: '{keyword_found}' in '{word['text']}'")
                
                # Get words BELOW this keyword (within 150px)
                keyword_bottom = word['y1']
                words_below = [
                    w for w in words
                    if w['y0'] > keyword_bottom and w['y0'] < keyword_bottom + 150
                ]
                
                if not words_below:
                    continue
                
                # Sort by y position (top to bottom)
                words_below.sort(key=lambda w: w['y0'])
                
                # Find the first valid customer name line
                for i, candidate_word in enumerate(words_below):
                    candidate_text = candidate_word['text'].strip()
                    
                    # Skip if it's an exclusion pattern
                    if self.is_customer_name_exclusion(candidate_text):
                        print(f"      Skipped (exclusion): '{candidate_text}'")
                        continue
                    
                    # Skip if it's an address line (street, postcode, etc.)
                    if self.is_address_line(candidate_text):
                        print(f"      Skipped (address line): '{candidate_text}'")
                        continue
                    
                    # Skip if too short or just numbers
                    if len(candidate_text) < 3 or candidate_text.isdigit():
                        print(f"      Skipped (too short/numeric): '{candidate_text}'")
                        continue
                    
                    # Calculate confidence based on category
                    base_confidence = {
                        'billing': 0.95,     # Highest - bill to is most reliable
                        'customer': 0.90,    # Very high
                        'shipping': 0.85,    # High
                        'attention': 0.80    # Good
                    }.get(category, 0.75)
                    
                    # Position bonus (earlier = better, first line gets bonus)
                    position_bonus = max(0, 0.05 - (i * 0.01))
                    
                    final_confidence = min(base_confidence + position_bonus, 1.0)
                    
                    candidates.append({
                        'name': candidate_text,
                        'confidence': final_confidence,
                        'category': category,
                        'keyword': keyword_found,
                        'y_position': candidate_word['y0']
                    })
                    
                    print(f"      ✅ Customer candidate: '{candidate_text}' "
                          f"(category: {category}, conf: {final_confidence:.2f})")
                    
                    # Stop after first valid name below this keyword
                    break
        
        if not candidates:
            print(f"[CUSTOMER_DETECTION] No address-based customer names found")
            return None, 0.0, "no_address_fields"
        
        # Select best candidate (highest confidence, then topmost position)
        best = max(candidates, key=lambda x: (x['confidence'], -x['y_position']))
        
        print(f"[CUSTOMER_DETECTION] ✅ BEST: '{best['name']}' "
              f"(category: {best['category']}, keyword: '{best['keyword']}', "
              f"conf: {best['confidence']:.2f})")
        
        return best['name'], best['confidence'], f"address_field_{best['category']}"
    
    def detect_email(self, words: List[Dict], stored_steps: List[str], 
                    highest_text: str) -> Tuple[Optional[str], str]:
        """Detect email and extract domain"""
        emails = []
        
        # Extract emails
        for w in words:
            if "@" in w["text"]:
                emails.append(w["text"])
        
        # Check for email: prefix
        wlist = [w["text"] for w in words]
        for i, t in enumerate(wlist):
            low = t.lower().replace(" ", "")
            if low.startswith("email"):
                if i + 1 < len(wlist):
                    nxt = wlist[i + 1].strip(":;, ")
                    if nxt:
                        emails.append(nxt)
                for sep in [":", ";", ",", "."]:
                    if sep in t:
                        part = t.split(sep, 1)[1].strip()
                        if part:
                            emails.append(part)
        
        if not emails:
            return None, "No email found"
        
        forbidden_domains = {"gmail", "credit", "debit"}
        
        # Build top keywords
        top_keywords = set()
        for s in stored_steps:
            top_keywords.add(s.lower())
            for word in re.findall(r"[A-Za-z]+", s.lower()):
                if len(word) >= 2:
                    top_keywords.add(word)
        
        if highest_text:
            top_keywords.add(highest_text.lower())
            for word in re.findall(r"[A-Za-z]+", highest_text.lower()):
                if len(word) >= 2:
                    top_keywords.add(word)
        
        best_domain = None
        best_reason = None
        
        for email in emails:
            if "@" not in email:
                continue
            
            parts = email.split("@")
            before_at = parts[0].lower()
            
            m = re.search(r"@([^.\s]+)", email)
            if not m:
                continue
            
            after_at = m.group(1).lower()
            
            if after_at in forbidden_domains:
                continue
            if re.search(r"[£$€₹]", after_at):
                continue
            if re.match(r"^-?\d+(\.\d+)?$", after_at):
                continue
            if re.match(r"^[0-9]", after_at):
                continue
            
            # Check contiguous prefix match
            for keyword in top_keywords:
                if _has_min_contiguous_prefix_match(keyword, before_at):
                    best_domain = before_at
                    best_reason = f"Prefix match: '{keyword}' vs '{before_at}'"
                    return best_domain, best_reason
            
            if not best_domain:
                best_domain = after_at
                best_reason = "Using domain after @"
        
        if best_domain:
            return best_domain, best_reason
        
        return None, "No valid email domain"
    
    def validate_email_domain_against_patterns(self, email_domain: str, 
                                              all_words: List[Dict]) -> Tuple[str, float]:
        """
        Validate email domain against visible business names.
        Returns: (name_to_use, confidence)
        """
        for word in all_words:
            text = word['text'].upper()
            
            if self.is_invalid_supplier_text(text):
                continue
            
            # Check if email domain appears in business name
            if email_domain.lower() in text.lower():
                # Found match - use full business name
                return text, 0.9
        
        # Email domain doesn't match visible names
        return email_domain, 0.4
    
    def detect_www(self, words: List[Dict], stored_steps: List[str], 
                  highest_text: str) -> Tuple[Optional[str], str]:
        """Detect website and extract domain"""
        key_words = []
        for s in stored_steps:
            key_words.extend(re.findall(r"[A-Za-z]+", s.lower()))
        key_words.extend(re.findall(r"[A-Za-z]+", highest_text.lower()))
        key_words = [k for k in key_words if len(k) >= 2]
        
        for w in words:
            t = w["text"].lower()
            if "www." in t:
                m = re.search(r"www\.([A-Za-z0-9-]+)", t)
                if not m:
                    continue
                dom = m.group(1)
                for kw in key_words:
                    if kw in dom:
                        return dom, f"WWW matched keyword '{kw}'"
                return dom, f"WWW fallback domain '{dom}'"
        
        return None, "No www found"
    
    def detect_limited_with_overlap(self, words: List[Dict]) -> Optional[str]:
        """Detect LIMITED/LTD with overlap logic"""
        def overlaps_horizontal(a0, a1, b0, b1):
            return not (a1 < b0 or b1 < a0)
        
        limited_candidates = [
            w for w in words 
            if re.search(r"\b(limited|ltd)\b", w["text"], re.I)
        ]
        
        if not limited_candidates:
            return None
        
        page_h = max(w["y1"] for w in words)
        limited_top_half = [w for w in limited_candidates if w["y0"] < page_h / 2]
        
        if not limited_top_half:
            return None
        
        limited_text = sorted(limited_top_half, key=lambda w: w["y0"])[0]
        limited_x0, limited_x1 = limited_text["x0"], limited_text["x1"]
        limited_y0, limited_y1 = limited_text["y0"], limited_text["y1"]
        limited_word = limited_text["text"].strip()
        
        if limited_word.upper() == "LIMITED":
            # Look for overlapped text above
            above_and_overlapped = []
            for w in words:
                if w == limited_text:
                    continue
                if w["y1"] < limited_y0:
                    if overlaps_horizontal(w["x0"], w["x1"], limited_x0, limited_x1):
                        above_and_overlapped.append(w)
            
            if not above_and_overlapped:
                return None
            
            closest_above = max(above_and_overlapped, key=lambda w: w["y1"])
            
            if self.is_invalid_supplier_text(closest_above["text"]):
                return None
            
            overlap_height = closest_above["y1"] - closest_above["y0"]
            threshold = overlap_height + 50
            
            tall_text_candidates = [
                w for w in words
                if w["y0"] < limited_y0 and not self.is_invalid_supplier_text(w["text"])
                and w != limited_text and w != closest_above
            ]
            
            for w in tall_text_candidates:
                text_height = w["y1"] - w["y0"]
                if text_height > threshold:
                    return w["text"]
            
            return f"{closest_above['text']} {limited_text['text']}"
        else:
            # Text has other words - use directly
            if self.is_invalid_supplier_text(limited_word):
                return None
            
            limited_height = limited_y1 - limited_y0
            threshold = limited_height + 50
            
            tall_text_candidates = [
                w for w in words
                if w["y0"] < limited_y0 and not self.is_invalid_supplier_text(w["text"])
                and w != limited_text
            ]
            
            for w in tall_text_candidates:
                text_height = w["y1"] - w["y0"]
                if text_height > threshold:
                    return w["text"]
            
            return limited_word
    
    def get_top_steps(self, words: List[Dict]) -> List[Dict]:
        """Get top text region using step algorithm"""
        def overlaps(a0, a1, b0, b1):
            return not (a1 < b0 or b1 < a0)
        
        words_sorted = sorted(words, key=lambda w: (w["x0"], w["y0"]))
        steps = []
        region_left = 0
        
        while True:
            candidates = [
                w for w in words_sorted
                if w["x0"] >= region_left and not self.is_invalid_supplier_text(w["text"])
            ]
            if not candidates:
                break
            
            candidates = sorted(candidates, key=lambda w: (w["x0"], w["y0"]))
            selected = None
            
            for w in candidates:
                L, R = w["x0"], w["x1"]
                blocker = False
                for other in words_sorted:
                    if other["y0"] < w["y0"]:
                        if overlaps(other["x0"], other["x1"], L, R):
                            blocker = True
                            break
                if not blocker:
                    selected = w
                    break
            
            if not selected:
                break
            
            steps.append(selected)
            region_left = selected["x1"]
            if len(steps) >= 6:
                break
        
        return steps
    
    def highest_height(self, words: List[Dict]) -> str:
        """Get highest/tallest text"""
        best = None
        max_h = -1
        for w in words:
            if self.is_invalid_supplier_text(w["text"]):
                continue
            h = w["y1"] - w["y0"]
            if h > max_h:
                max_h = h
                best = w["text"]
        return best if best else ""
    
    def extract(self, words: List[Dict], page_height: Optional[float] = None,
               extract_customer: bool = True) -> Tuple[Optional[str], float, str]:
        """
        Main extraction method - HYBRID APPROACH
        
        Args:
            words: List of word dicts with keys: text, x0, y0, x1, y1, confidence
            page_height: Optional page height for position scoring
            extract_customer: If True (default), prioritize customer name from address fields
        
        Returns:
            (name, confidence, method)
        """
        if not words:
            return None, 0.0, "no_words"
        
        # Calculate page height if not provided
        if page_height is None:
            page_height = max(w['y1'] for w in words) if words else 1000
        
        print(f"[HYBRID_SUPPLIER] Extraction mode: {'CUSTOMER (invoice)' if extract_customer else 'SUPPLIER (receipt)'}")
        
        # PRIORITY 0: CUSTOMER NAME FROM ADDRESS FIELDS (for invoices)
        if extract_customer:
            print(f"[HYBRID_SUPPLIER] P0 CUSTOMER ADDRESS FIELDS: Checking...")
            customer_name, confidence, method = self.detect_customer_from_address_fields(
                words, page_height
            )
            if customer_name and confidence > 0.75:
                print(f"[HYBRID_SUPPLIER] ✅ P0 CUSTOMER: '{customer_name}' "
                      f"(conf: {confidence:.2f}, method: {method})")
                return customer_name, confidence, method
            print(f"[HYBRID_SUPPLIER] P0 CUSTOMER: Not found or low confidence")
        
        # Get top steps for email/www logic
        steps = self.get_top_steps(words)
        step_texts = [w["text"] for w in steps]
        highest_txt = self.highest_height(words)
        
        print(f"[HYBRID_SUPPLIER] Extraction priority: PATTERNS → LIMITED → EMAIL → WWW → HEIGHT")
        
        # PRIORITY 1: BUSINESS PATTERN RECOGNITION (most reliable)
        print(f"[HYBRID_SUPPLIER] P1 BUSINESS PATTERNS: Analyzing top region...")
        
        top_region_threshold = page_height * 0.3
        top_words = [w for w in words if w['y0'] < top_region_threshold]
        
        business_candidates = []
        
        for word in top_words:
            text = word['text'].strip()
            
            if self.is_invalid_supplier_text(text):
                continue
            
            pattern_score, reasons = self.score_supplier_by_business_patterns(
                text, word['y0'], page_height
            )
            
            if pattern_score < 0.3:
                continue
            
            # Check address context
            has_address, address_bonus = self.has_address_context_below(word, words)
            if has_address:
                pattern_score += address_bonus
                reasons.append(f"address_context_+{address_bonus:.2f}")
            
            # Add OCR confidence
            ocr_confidence = word.get('confidence', 1.0)
            combined_score = (pattern_score * 0.8) + (ocr_confidence * 0.2)
            
            business_candidates.append({
                'text': text,
                'score': combined_score,
                'reasons': reasons,
                'y_position': word['y0']
            })
            
            print(f"   Candidate: '{text}' (score: {combined_score:.3f}, reasons: {reasons})")
        
        if business_candidates:
            best = max(business_candidates, key=lambda x: (x['score'], -x['y_position']))
            print(f"[HYBRID_SUPPLIER] ✅ P1 PATTERN: '{best['text']}' (score: {best['score']:.3f})")
            return best['text'], best['score'], 'business_pattern'
        
        print(f"[HYBRID_SUPPLIER] P1 PATTERN: No strong candidates")
        
        # PRIORITY 2: LIMITED/LTD
        limited_name = self.detect_limited_with_overlap(words)
        if limited_name:
            print(f"[HYBRID_SUPPLIER] ✅ P2 LIMITED: '{limited_name}'")
            return limited_name, 0.75, 'limited_overlap'
        print(f"[HYBRID_SUPPLIER] P2 LIMITED: Not found")
        
        # PRIORITY 3: EMAIL DOMAIN (validated)
        email_domain, email_reason = self.detect_email(words, step_texts, highest_txt)
        if email_domain:
            validated_name, confidence = self.validate_email_domain_against_patterns(
                email_domain, words
            )
            print(f"[HYBRID_SUPPLIER] ✅ P3 EMAIL: '{validated_name}' ({email_reason}, conf: {confidence:.2f})")
            return validated_name, confidence, 'email_validated'
        print(f"[HYBRID_SUPPLIER] P3 EMAIL: {email_reason}")
        
        # PRIORITY 4: WWW
        www_domain, www_reason = self.detect_www(words, step_texts, highest_txt)
        if www_domain:
            print(f"[HYBRID_SUPPLIER] ✅ P4 WWW: '{www_domain}' ({www_reason})")
            return www_domain, 0.6, 'www_domain'
        print(f"[HYBRID_SUPPLIER] P4 WWW: {www_reason}")
        
        # PRIORITY 5: HEIGHT FALLBACK
        print(f"[HYBRID_SUPPLIER] ✅ P5 FALLBACK HEIGHT: '{highest_txt}'")
        return highest_txt if highest_txt else None, 0.4, 'height_fallback'
