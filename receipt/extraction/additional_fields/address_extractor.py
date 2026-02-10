#!/usr/bin/env python3
"""
Address Extractor for Additional Fields Extractor v2.7.1
Specializes in multi-country address detection and extraction
"""

import re
from .pattern_manager import PatternManager

class AddressExtractor:
    """
    Extracts address information with multi-country support.
    Focuses on postal code detection and contextual address extraction.
    """
    
    def __init__(self, pattern_manager=None):
        """Initialize address extractor with pattern manager."""
        self.pattern_manager = pattern_manager or PatternManager()
        print(f"✅ Initialized Address Extractor v2.7.1")
        print(f"   Multi-country address detection: ✅ Ready")
    
    def detect_address_from_text(self, text_lines, supplier_context=None):
        """
        Enhanced multi-country address detection (v2.7.1 - Fixed Postal Code Stopping).
        
        Args:
            text_lines: List of text lines from receipt or single text string
            supplier_context: Optional supplier context for enhanced extraction
            
        Returns:
            Detected address string or None
        """
        
        if not text_lines:
            return None
        
        # Convert to list if string
        if isinstance(text_lines, str):
            text_lines = [line.strip() for line in text_lines.split("\n") if line.strip()]
        
        lines = [line.strip() for line in text_lines if line.strip()]
        if not lines:
            return None

        print(f"🌍 Multi-country address detection starting on {len(lines)} lines...")

        # Step 1: Enhanced line preprocessing for comma-separated content (v2.7.1)
        processed_lines = []
        for line in lines:
            # If line contains postal code and trailing data, try to split intelligently
            if self.pattern_manager.contains_any_postcode_pattern(line) and (',' in line):
                parts = self._split_line_at_postal_code(line)
                processed_lines.extend(parts)
            else:
                processed_lines.append(line)
        
        print(f"📋 Processed into {len(processed_lines)} lines after intelligent splitting")

        # Step 2: Multi-country postal/ZIP code detection (v2.7.1)
        postcode = None
        postcode_line_idx = -1
        detected_country = None

        for i, line in enumerate(processed_lines):
            # Try each country pattern
            for country, pattern in self.pattern_manager.postcode_patterns.items():
                country_pattern = re.compile(pattern, re.IGNORECASE)
                pc_match = country_pattern.search(line)
                if pc_match:
                    postcode = pc_match.group(0).strip().upper()
                    postcode_line_idx = i
                    detected_country = country
                    print(f"🏠 Found {country} postal code: '{postcode}' at line {i}")
                    break
            
            if postcode:
                break

        # Step 3: Enhanced address line collection (v2.7.1)
        address_lines = []

        if postcode_line_idx != -1:
            # Postal/ZIP code-based address collection with enhanced filtering
            print(f"📮 Using {detected_country} postal code approach from line {postcode_line_idx}")
            start_idx = max(0, postcode_line_idx - 5)
            
            for i in range(start_idx, postcode_line_idx + 1):
                line = processed_lines[i]
                
                # Enhanced multi-country noise filtering (v2.7.1)
                if self._is_address_noise_line_enhanced(line):
                    print(f"   ❌ Skipping noise line {i}: '{line}'")
                    continue
                if self._is_menu_item_line_enhanced(line):
                    print(f"   ❌ Skipping menu item line {i}: '{line}'")
                    continue
                    
                # More inclusive for postal/ZIP-based collection
                if (self.pattern_manager.has_address_keywords(line) or 
                    i >= postcode_line_idx - 2 or 
                    self.pattern_manager.contains_any_postcode_pattern(line)):
                    cleaned = self._clean_address_line_enhanced(line)
                    if cleaned and self._is_valid_address_line_enhanced(cleaned):
                        address_lines.append(cleaned)
                        print(f"   ✅ Added address line {i}: '{cleaned}'")
                        
                        # STOP immediately after finding postcode (v2.7.1)
                        if self.pattern_manager.contains_any_postcode_pattern(cleaned):
                            print(f"   🛑 Stopping after postcode found")
                            break
        
        else:
            # Enhanced keyword-based fallback with global patterns (v2.7.1)
            print("🔍 No postal code found, using enhanced keyword-based detection...")
            
            for i, line in enumerate(processed_lines):
                if i < 3:  # Skip very early lines (likely headers)
                    continue
                if i > 20:  # Don't go too far down
                    break
                    
                if self._is_address_noise_line_enhanced(line):
                    continue
                if self._is_menu_item_line_enhanced(line):
                    continue
                    
                if self.pattern_manager.has_address_keywords(line):
                    print(f"🏠 Found address keywords in line {i}: '{line}'")
                    # Found address keyword, collect subsequent lines
                    for j in range(i, min(i + 6, len(processed_lines))):
                        addr_line = processed_lines[j]
                        if (not self._is_address_noise_line_enhanced(addr_line) and 
                            not self._is_menu_item_line_enhanced(addr_line)):
                            cleaned = self._clean_address_line_enhanced(addr_line)
                            if cleaned and self._is_valid_address_line_enhanced(cleaned):
                                address_lines.append(cleaned)
                                print(f"   ✅ Added keyword-based line {j}: '{cleaned}'")
                                
                                # STOP after finding any postcode pattern (v2.7.1)
                                if self.pattern_manager.contains_any_postcode_pattern(cleaned):
                                    print(f"   🛑 Stopping after postcode pattern found")
                                    break
                        else:
                            break
                    break

        if not address_lines:
            print("❌ No address lines found")
            return None

        # Step 4: Enhanced duplicate removal and validation (v2.7.1)
        seen = set()
        unique_lines = []
        for line in address_lines:
            line_lower = line.lower()
            if line_lower not in seen:
                # Final strict validation to ensure no unwanted content
                if not self._is_address_noise_line_enhanced(line):
                    seen.add(line_lower)
                    unique_lines.append(line)
                    print(f"   📝 Unique address component: '{line}'")

        if not unique_lines:
            print("❌ No unique address lines after filtering")
            return None

        final_address = ", ".join(unique_lines)
        
        # Step 5: Multi-country address validation (v2.7.1)
        has_postcode = bool(self.pattern_manager.contains_any_postcode_pattern(final_address))
        has_keyword = bool(self.pattern_manager.address_keywords_pattern.search(final_address))
        
        if not has_postcode and not has_keyword:
            print(f"❌ Address validation failed: no postcode or keywords in '{final_address}'")
            return None
        
        address_words = final_address.replace(",", "").strip().split()
        if len(address_words) < 2:
            print(f"❌ Address too short: {len(address_words)} words in '{final_address}'")
            return None
        
        has_alpha_word = any(word for word in address_words if any(c.isalpha() for c in word))
        if not has_alpha_word:
            print(f"❌ Address has no alphabetic content: '{final_address}'")
            return None

        # Step 6: Enhanced address cleanup (v2.7.1)
        address = re.sub(r"\s+", " ", final_address)
        address = re.sub(r",\s*,+", ",", address)
        address = address.strip(", ")
        
        country_info = f" ({detected_country})" if detected_country else ""
        print(f"✅ Multi-country address detected{country_info}: '{address}'")

        return address if address else None
    
    def extract_contextual_addresses(self, contact_lines, supplier_context=None):
        """
        Extract addresses using supplier-address context relationship.
        
        Args:
            contact_lines: DataFrame with contact information lines
            supplier_context: Optional supplier context for enhanced extraction
            
        Returns:
            List of address blocks
        """
        
        address_lines = []
        
        # Get supplier line number if available from supplier context
        supplier_line_number = None
        if supplier_context and isinstance(supplier_context, dict):
            supplier_line_number = supplier_context.get('line_number')
        
        # If no supplier context, try to find supplier line
        if supplier_line_number is None:
            supplier_line_number = self._find_likely_supplier_line(contact_lines)
        
        if supplier_line_number is not None:
            print(f"🏪 Using supplier line {supplier_line_number} as address reference point")
            # Extract address block following supplier
            address_block = self._extract_address_block_after_supplier(contact_lines, supplier_line_number)
            if address_block:
                address_lines.append(address_block)
        
        # Fallback: Use traditional pattern-based detection
        remaining_lines = contact_lines[~contact_lines['line_number'].isin(
            [line['line_number'] for block in address_lines for line in block] if address_lines else []
        )]
        
        traditional_addresses = self._extract_traditional_address_blocks(remaining_lines)
        address_lines.extend(traditional_addresses)
        
        return address_lines
    
    def _split_line_at_postal_code(self, line):
        """
        Split a line at postal/ZIP code to separate address from trailing content (v2.7.1).
        
        Args:
            line: Line containing postal code and potentially trailing data
            
        Returns:
            List of processed line components
        """
        parts = []
        
        # Find postal code position
        postcode_match = None
        postcode_end = 0
        
        for country, pattern in self.pattern_manager.postcode_patterns.items():
            country_pattern = re.compile(pattern, re.IGNORECASE)
            match = country_pattern.search(line)
            if match:
                postcode_match = match
                postcode_end = match.end()
                print(f"📍 Found {country} postal code at position {match.start()}-{match.end()}: '{match.group(0)}'")
                break
        
        if postcode_match:
            # Split the line: before + postal code + after
            before_postcode = line[:postcode_match.start()].strip()
            postcode_part = line[postcode_match.start():postcode_end].strip()
            after_postcode = line[postcode_end:].strip()
            
            # Process before postal code (split on commas)
            if before_postcode:
                before_parts = [part.strip() for part in before_postcode.split(',') if part.strip()]
                parts.extend(before_parts)
            
            # Add postal code as separate component
            if postcode_part:
                parts.append(postcode_part)
            
            # Analyze after postal code - split and filter out obvious noise
            if after_postcode:
                after_parts = [part.strip() for part in after_postcode.split(',') if part.strip()]
                for part in after_parts:
                    # Skip obvious trailing noise (phone, VAT, dates, etc.)
                    if not self._is_obvious_trailing_noise(part):
                        parts.append(part)
                    else:
                        print(f"   🚮 Filtered trailing noise: '{part}'")
                        break  # Stop at first noise - everything after is likely noise
        
        else:
            # No postal code found, return original line
            parts = [line]
        
        print(f"📋 Split line into {len(parts)} parts: {parts}")
        return parts
    
    def _is_obvious_trailing_noise(self, text):
        """
        Check if text is obvious trailing noise that should be filtered out (v2.7.1).
        
        Args:
            text: Text to check
            
        Returns:
            True if text is obvious trailing noise
        """
        text_lower = text.lower().strip()
        
        # Obvious noise patterns
        noise_patterns = [
            r'^(tel?|telephone|phone|ph|mob|mobile|fax)[\s:]*',               # Phone labels
            r'^(vat|tax|registration)[\s:]*',                                 # VAT/Tax labels
            r'^(email|e-mail|web|website)[\s:@]*',                           # Contact labels
            r'^\d{4,5}\s*\d{6,}',                                            # Phone numbers
            r'^(mon|tue|wed|thu|fri|sat|sun)\s*\d',                         # Day dates
            r'^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d',     # Month dates
            r'^\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', # Date patterns
            r'^\d{2}:\d{2}',                                                 # Time patterns
            r'^20\d{2}',                                                     # Years
            r'^[a-z]+@[a-z]+\.',                                            # Email addresses
            r'^www\.',                                                       # Websites
            r'^receipt\s*(no|number|#)',                                     # Receipt numbers
            r'^transaction\s*(no|number|#)',                                 # Transaction numbers
        ]
        
        for pattern in noise_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # Check for obvious phone number patterns
        if re.search(r'\b\d{10,11}\b', text):
            return True
            
        # Check for VAT number patterns
        if re.search(r'\d{3,}\s*\d{3,}\s*\d{2,}', text):  # VAT-like number patterns
            return True
            
        return False
    
    def _is_address_noise_line_enhanced(self, line):
        """
        Enhanced address noise detection (v2.6.7) - excludes dates, VAT numbers, telephone.
        """
        if not line or len(line.strip()) < 2:
            return True
            
        # Check for standard noise patterns
        if self.pattern_manager.is_address_noise_line(line):
            return True
            
        # Check for date patterns
        if self.pattern_manager.date_pattern.search(line):
            return True
            
        # Check for VAT number patterns
        if self.pattern_manager.vat_pattern.search(line):
            return True
            
        # Check for telephone patterns
        if self.pattern_manager.telephone_pattern.search(line):
            return True
            
        # Check for pure numeric lines (likely IDs, timestamps, etc.)
        if re.match(r'^[\d\s\-\(\)\+\.:#]+$', line) and len(line.strip()) > 6:
            return True
            
        # Check for email patterns
        if '@' in line and '.' in line:
            return True
            
        # Check for website patterns
        if re.search(r'www\.|\.com|\.co\.uk|\.org|\.net', line, re.IGNORECASE):
            return True
            
        return False
    
    def _is_menu_item_line_enhanced(self, line):
        """Enhanced menu/product line detection with global patterns."""
        return self.pattern_manager.is_menu_item_line(line)
    
    def _clean_address_line_enhanced(self, line):
        """
        Enhanced address line cleaning with improved character handling (v2.7.0).
        """
        # Remove special characters but keep essential punctuation for international addresses
        cleaned = re.sub(r"[^\w\s,.\-&'()#/]", " ", line)
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
    
    def _is_valid_address_line_enhanced(self, line):
        """
        Enhanced address line validation (v2.7.0) with international address support.
        """
        if not line or len(line.strip()) < 2:
            return False
            
        # Check basic validation first
        if not self._is_valid_address_line(line):
            return False
            
        # Enhanced checks for v2.7.0 (international support)
        line_clean = line.strip()
        
        # Enhanced noise filtering for international addresses
        if self._is_address_noise_line_enhanced(line_clean):
            return False
            
        # Must contain at least some alphabetic characters
        if not re.search(r'[A-Za-z]', line_clean):
            return False
            
        # Exclude very long sequences of numbers (barcodes, IDs, etc.)
        if re.search(r'\d{12,}', line_clean):
            return False
            
        # Enhanced product code exclusion (international patterns)
        if re.match(r'^[A-Z0-9\-]{8,}$', line_clean, re.IGNORECASE):
            return False
            
        # Allow reasonable international character counts
        special_char_count = len(re.findall(r'[^\w\s\-\.,&\'()#/]', line_clean))
        if special_char_count > 4:  # Increased for international addresses
            return False
            
        # Reasonable word structure for international addresses
        words = line_clean.split()
        if len(words) > 10:  # Increased limit for international addresses
            return False
            
        # Additional international address validation
        if len(words) == 1:
            # Single word should be reasonable length and not all numbers
            if len(line_clean) > 25 or line_clean.isdigit():
                return False
        
        # Check for obvious technical/system codes
        if re.match(r'^[A-Z0-9]{6,}$', line_clean):
            return False
            
        return True
    
    def _is_valid_address_line(self, line):
        """Validate if line looks like a valid address component."""
        if len(line.strip()) < 3:
            return False
        if re.match(r'^[\d\s\-()]+$', line):  # Only digits and punctuation
            return False
        if re.match(r'^\d{6,}', line):  # Starts with long number
            return False
        if self.pattern_manager.technical_pattern.search(line):
            return False
        return True
    
    def _find_likely_supplier_line(self, contact_lines):
        """Find the most likely supplier line number from header lines."""
        # Look for first non-excluded header line
        exclude_patterns = [
            r'(?i)\b(receipt|invoice|bill|thank\s*you|welcome)\b',
            r'(?i)^(tel|phone|email|web|fax):',
            r'^\d+$',  # Just numbers
            r'^[A-Z]{1,3}\d+\s*[A-Z]{0,3}$',  # Postcodes
            r'@.*\.',  # Email addresses
        ]
        
        for _, row in contact_lines.iterrows():
            if row['predicted_class'] == 'HEADER':
                text = str(row['text']).strip()
                
                # Skip obvious non-supplier lines
                is_excluded = any(re.search(pattern, text) for pattern in exclude_patterns)
                if not is_excluded and len(text.strip()) > 2:
                    return row['line_number']
        
        return None
    
    def _extract_address_block_after_supplier(self, contact_lines, supplier_line_number):
        """
        Extract address lines immediately following the supplier name.
        
        Enhanced v2.7.1: Now properly stops after finding postal/ZIP code.
        """
        address_block = []
        
        # Get lines after supplier line, ordered by line number
        following_lines = contact_lines[
            (contact_lines['line_number'] > supplier_line_number) & 
            (contact_lines['predicted_class'] == 'HEADER')
        ].sort_values('line_number')
        
        print(f"📍 Analyzing {len(following_lines)} lines after supplier for address extraction...")
        
        for _, row in following_lines.iterrows():
            text = str(row['text']).strip()
            line_num = row['line_number']
            
            print(f"  Line {line_num}: '{text}'")
            
            # PRIORITY CHECK: Stop immediately if this line contains a postal/ZIP code (v2.7.1)
            if self.pattern_manager.contains_any_postcode_pattern(text):
                # Add the postal code line to address block
                if self._is_address_component(text):
                    address_block.append({
                        'text': text,
                        'line_number': line_num,
                        'confidence': row['confidence_score'],
                        'address_type': self._classify_address_component(text)
                    })
                    print(f"    ✅ Added postal code line: {self._classify_address_component(text)}")
                
                print(f"    🛑 STOPPING immediately after postal/ZIP code found (v2.7.1)")
                break
            
            # Check other termination conditions
            if self.pattern_manager.is_address_terminator(text):
                print(f"    ⛔ Address terminator detected: {self._get_termination_reason(text)}")
                break
            
            # Check if line looks like address component
            if self._is_address_component(text):
                address_block.append({
                    'text': text,
                    'line_number': line_num,
                    'confidence': row['confidence_score'],
                    'address_type': self._classify_address_component(text)
                })
                print(f"    ✅ Added as address component: {self._classify_address_component(text)}")
                
                # DOUBLE CHECK: If we just added a line that contains postal code, stop immediately
                if self.pattern_manager.contains_any_postcode_pattern(text):
                    print(f"    🛑 STOPPING - postal code detected in added component (v2.7.1)")
                    break
                    
            else:
                # Non-address line encountered - consider this end of address
                print(f"    🔄 Non-address line encountered, stopping address extraction")
                break
        
        if address_block:
            print(f"🏠 Extracted address block with {len(address_block)} components")
            return address_block
        
        return None
    
    def _get_termination_reason(self, text):
        """Get reason why line terminates address extraction."""
        if re.search(r'(?i)^(tel|telephone|phone|ph|mob|mobile|fax)[\s:\-]', text):
            return "phone number"
        elif re.search(r'(?i)^(email|e-mail|web|website)[\s:\-@]', text):
            return "contact info"
        elif re.search(r'(?i)^(vat|tax)\s*(no|number|reg|#)[\s:\-]', text):
            return "VAT number"
        elif re.search(r'^.{1,2}$', text):
            return "empty/short line"
        else:
            return "business info"
    
    def _is_address_component(self, text):
        """Check if text looks like a valid address component."""
        # Skip empty or very short lines
        if len(text.strip()) <= 2:
            return False
        
        # PRIORITY: Multi-country postal/ZIP code detection (v2.7.1)
        if self.pattern_manager.contains_any_postcode_pattern(text):
            return True
        
        # Check for obvious address patterns
        address_indicators = [
            # Street/road patterns
            r'\b\d+\s+\w+\s+(street|st|road|rd|avenue|ave|lane|ln|drive|dr|close|cl|way|place|pl)\b',
            
            # Building/location names
            r'\b(house|building|court|centre|center|park|gardens?|estate|plaza|mall)\b',
            
            # Area/city names
            r'^[A-Z][a-zA-Z\s\-\']{2,25}$',
            
            # Business areas/districts
            r'\b(business\s+park|industrial\s+estate|retail\s+park|town\s+centre|city\s+centre)\b',
        ]
        
        # Strong address indicators
        for pattern in address_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for reasonable address component characteristics
        word_count = len(text.split())
        
        # Single word location names (2-20 chars, not all numbers)
        if word_count == 1 and 2 <= len(text) <= 20 and not text.isdigit():
            # Check if it's likely a place name (proper case or uppercase)
            if text[0].isupper():
                return True
        
        # Multi-word components (2-5 words, reasonable length)
        elif 2 <= word_count <= 5 and len(text) <= 50:
            # Check if it has proper name characteristics
            if any(word[0].isupper() for word in text.split() if word):
                return True
        
        return False
    
    def _classify_address_component(self, text):
        """Classify the type of address component."""
        text_lower = text.lower()
        
        # Enhanced multi-country postal/ZIP code detection (v2.7.1)
        if self.pattern_manager.contains_any_postcode_pattern(text):
            return "postcode"
        
        # Street address detection  
        if re.search(r'\b\d+\s+\w+\s+(street|st|road|rd|avenue|ave|lane|ln|drive|dr|close|cl|way|place|pl)\b', text, re.IGNORECASE):
            return "street_address"
        
        # Building name detection
        if re.search(r'\b(house|building|court|centre|center|park|gardens?|estate|plaza|mall)\b', text, re.IGNORECASE):
            return "building_name"
        
        # Area/city detection (heuristic based on length and format)
        word_count = len(text.split())
        if word_count == 1 and 3 <= len(text) <= 20:
            return "area_or_city"
        elif 2 <= word_count <= 3:
            return "area_or_district"
        
        return "address_component"
    
    def _extract_traditional_address_blocks(self, remaining_lines):
        """Extract addresses using traditional pattern-based approach for fallback."""
        address_lines = []
        current_address = []
        
        for _, row in remaining_lines.iterrows():
            text = str(row['text']).strip()
            
            if row['predicted_class'] == 'HEADER':
                # Check if this line contains address-like information (traditional method)
                if self._is_address_line(text):
                    current_address.append({
                        'text': text,
                        'line_number': row['line_number'],
                        'confidence': row['confidence_score'],
                        'address_type': 'traditional_pattern'
                    })
                elif current_address:
                    # End of address block
                    address_lines.append(current_address)
                    current_address = []
        
        # Add any remaining address
        if current_address:
            address_lines.append(current_address)
        
        return address_lines
    
    def _is_address_line(self, text):
        """Check if a line contains address information."""
        text_lower = text.lower()
        
        # Address indicators
        address_patterns = [
            # UK postcode patterns
            r'[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}',
            # Street indicators
            r'\b(street|st|road|rd|avenue|ave|lane|ln|drive|dr|close|cl|way|place|pl|square|sq)\b',
            # Area/location indicators  
            r'\b(london|manchester|birmingham|leeds|glasgow|liverpool|bristol|edinburgh)\b',
        ]
        
        # Check for postcode or street indicators
        for pattern in address_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # Check for address keywords from config
        address_keywords = ['address', 'store', 'location', 'road', 'street', 'avenue', 'lane']
        if any(keyword in text_lower for keyword in address_keywords):
            return True
        
        return False