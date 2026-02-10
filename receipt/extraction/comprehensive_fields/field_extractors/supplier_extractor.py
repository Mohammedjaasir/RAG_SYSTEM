"""
Supplier Extractor - Extracts supplier/vendor name from receipt data
"""
import re
import pandas as pd
from typing import Dict, Optional, List
from ..shared_utils.config_manager import ConfigManager
from ..shared_utils.text_cleaner import TextCleaner
from ..shared_utils.confidence_scorer import ConfidenceScorer
from ..models.extraction_models import SupplierExtractionResult


class SupplierExtractor:
    """Extracts supplier name using intelligent supplier detection with address context."""
    
    def __init__(self, config_manager: ConfigManager, text_cleaner: TextCleaner, confidence_scorer: ConfidenceScorer):
        self.config_manager = config_manager
        self.text_cleaner = text_cleaner
        self.confidence_scorer = confidence_scorer
        self.suppliers = config_manager.get_suppliers()
        self.extraction_rules = config_manager.config.get('extraction_rules', {})
    
    def extract_supplier_name(self, df) -> Optional[SupplierExtractionResult]:
        """Extract supplier/vendor name using intelligent supplier detection with address context."""
        if 'supplier_extraction' not in self.extraction_rules:
            print("⚠️  No supplier extraction rules found")
            return None
        
        rules = self.extraction_rules['supplier_extraction']
        target_classes = rules.get('target_classes', ['HEADER'])
        
        # Handle different column name formats
        class_column = 'line_type' if 'line_type' in df.columns else 'predicted_class'
        header_lines = df[df[class_column].isin(target_classes)].copy().reset_index(drop=True)
        
        best_match = None
        highest_confidence = 0.0
        
        print(f"🏪 Analyzing {len(header_lines)} header lines for intelligent supplier extraction")
        
        # Step 1: Look for known suppliers from config list
        for idx, row in header_lines.iterrows():
            text_column = 'cleaned_text' if 'cleaned_text' in row else ('text' if 'text' in row else 'original_text')
            text = str(row[text_column]).strip()
            text_clean = re.sub(r'[^\w\s&-]', '', text)
            
            # Check against supplier list
            for supplier in self.suppliers:
                if supplier.lower() in text_clean.lower():
                    print(f"✅ Found known supplier: '{supplier}' in '{text_clean}'")
                    
                    # Calculate confidence
                    confidence_column = 'confidence' if 'confidence' in row else 'confidence_score'
                    ml_confidence = self.confidence_scorer.safe_convert(row[confidence_column])
                    
                    # Position bonus
                    position_bonus = max(0.3 - (idx * 0.05), 0.1)
                    
                    # Match quality bonus
                    match_bonus = 0.0
                    if supplier.lower() == text_clean.lower().strip():
                        match_bonus = 0.4
                    elif text_clean.lower().startswith(supplier.lower()):
                        match_bonus = 0.3
                    elif text_clean.lower().endswith(supplier.lower()):
                        match_bonus = 0.25
                    else:
                        match_bonus = 0.15
                    
                    combined_confidence = min((ml_confidence + position_bonus + match_bonus) / 1.8, 1.0)
                    
                    if combined_confidence > highest_confidence:
                        highest_confidence = combined_confidence
                        
                        # Clean and use the full business name
                        cleaned_text = self.text_cleaner.clean_supplier_name(text.strip())
                        
                        # Use cleaned text if it looks better
                        if len(cleaned_text.split()) <= 6 and len(cleaned_text) <= 50 and cleaned_text.strip():
                            supplier_name_to_use = cleaned_text
                            print(f"📝 Using cleaned receipt text '{cleaned_text}'")
                        elif len(text.strip()) <= 50 and len(text.split()) <= 6:
                            supplier_name_to_use = text.strip()
                            print(f"📝 Using full receipt text '{text}'")
                        else:
                            supplier_name_to_use = supplier
                            print(f"📝 Using config name '{supplier}'")
                        
                        best_match = {
                            'supplier_name': supplier_name_to_use,
                            'raw_text': text,
                            'confidence': combined_confidence,
                            'line_number': self.confidence_scorer.safe_convert(row.get('line_number', 0), int),
                            'extraction_method': 'config_pattern_match_full_text',
                            'matched_config_supplier': supplier
                        }
                        print(f"🎯 New best match: '{supplier_name_to_use}' (confidence: {combined_confidence:.3f})")
        
        if best_match:
            return best_match
        
        # Step 2: Enhanced supplier detection
        print("🔍 No known suppliers found, trying enhanced supplier detection...")
        enhanced_candidates = self.get_enhanced_supplier_candidates(df)
        
        if enhanced_candidates:
            best_enhanced = max(enhanced_candidates, key=lambda x: x['confidence'])
            print(f"✨ Enhanced detection found: '{best_enhanced['supplier_name']}' (confidence: {best_enhanced['confidence']:.3f})")
            
            if best_enhanced['confidence'] >= 0.7:
                return {
                    'supplier_name': best_enhanced['supplier_name'],
                    'raw_text': best_enhanced['raw_text'],
                    'confidence': best_enhanced['confidence'],
                    'line_number': best_enhanced['line_number'],
                    'extraction_method': 'enhanced_supplier_detector'
                }
        
        # Step 3: Intelligent heuristic analysis
        print("🧠 Applying intelligent heuristics for comparison...")
        supplier_candidates = []
        
        for idx, row in header_lines.iterrows():
            text_column = 'cleaned_text' if 'cleaned_text' in row else ('text' if 'text' in row else 'original_text')
            text = str(row[text_column]).strip()
            text_clean = text.upper()
            
            # Skip patterns to avoid date/time and address contamination
            skip_patterns = [
                r'\b(RECEIPT|INVOICE|BILL|TAX|VAT|THANK\s*YOU|WELCOME|HELLO)\b',
                r'^\d+$',
                r'^[A-Z]{1,3}\d+\s*[A-Z]{1,3}$',
                r'^\d{2}[A-Z]{3}\'\d{2}$',
                r'^TEL:?|^PHONE:?|^EMAIL:?|^WEB:?|^FAX:?',
                r'^\d+\s*\w+\s*(STREET|ROAD|LANE|AVENUE|DRIVE|WAY|CLOSE)',
                r'@.*\.',
                r'^STORE\s*#?\s*\d+',
                r'^\d{1,5}\s+\w+',
                r'\b(MON|TUE|WED|THU|FRI|SAT|SUN)\b.*\d{2}:\d{2}',
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{2}:\d{2}(:\d{2})?\b',
                r'\b\d{2}\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*\d{2,4}\b',
                r'^[A-Z]{2}\d+\s*\d*[A-Z]{0,3}$',
                r'^\d+\s+(LONDON\s+ROAD|HIGH\s+STREET|MAIN\s+STREET|CHURCH\s+LANE)',
                r'^NR\.?\s*[A-Z]',
                r'^\*+\s*(FUEL\s+RECEIP?T?|RECEIP?T?|REPRINT)\**$',
                r'^TILL\s+ID\s+\d+',
                r'^POS\s+\d+',
                r'^\*+.*\*+$',
            ]
            
            is_skip = any(re.search(pattern, text_clean) for pattern in skip_patterns)
            if is_skip:
                continue
            
            # Calculate supplier likelihood score
            likelihood_score = 0.0
            reasons = []
            
            # Bonus for early positions
            if idx == 0:
                likelihood_score += 0.4
                reasons.append("first_line")
            elif idx == 1:
                likelihood_score += 0.3
                reasons.append("second_line")
            elif idx < 3:
                likelihood_score += 0.2
                reasons.append("early_position")
            
            # Enhanced business name patterns
            business_patterns = [
                (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(SERVICE\s+STATION|GARAGE|PETROL\s+STATION)\b', 0.9, "named_fuel_business"),
                (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(LIMITED|LTD|SOLUTIONS|SERVICES)\b', 0.8, "named_business_entity"),
                (r'\b(SERVICE\s+STATION|GARAGE|PETROL\s+STATION|FILLING\s+STATION)\b', 0.7, "fuel_service_station"),
                (r'\b(FORECOURT|MOTORWAY\s+SERVICES|SERVICES\s+STATION)\b', 0.6, "fuel_service_provider"),
                (r'\b(SOLUTIONS|SERVICES|ENTERPRISES)\b', 0.5, "high_priority_business_type"),
                (r'\b(LIMITED|LTD|PLC|INC|CORPORATION|CORP|LLC)\b', 0.45, "legal_entity"),
                (r'\b(SHOPPING\s+CENT(ER|RE)|SUPERMARKET|EXPRESS)\b', 0.4, "retail_identifier"),
                (r'\b(RESTAURANT|CAFE|BAR|PUB|HOTEL|PHARMACY)\b', 0.35, "hospitality_business"),
                (r'\b(SHOP|STORE|MARKET|RETAIL)\b', 0.3, "retail_business"),
                (r'\b(OFFICE|CENTER|CENTRE|CLINIC|STUDIO)\b', 0.25, "service_business"),
                (r'\b(&|\bAND\b)\b', 0.2, "partnership_connector"),
                (r'\b(COMPANY|CO\.|GROUP|HOLDINGS)\b', 0.2, "business_entity"),
                (r'^[A-Z][a-z]+(\s+[A-Z][a-z]+)*$', 0.15, "proper_name_format"),
                (r'[A-Z]{2,}', 0.1, "has_uppercase"),
                (r'\b(BP|SHELL|ESSO|TEXACO|TOTAL|GULF|MURCO|JET)\b', 0.5, "fuel_brand"),
                (r'\b(SAINSBURY|TESCO|ASDA|MORRISONS|WAITROSE)\b', 0.3, "retail_brand"),
            ]
            
            for pattern, bonus, reason in business_patterns:
                if re.search(pattern, text_clean):
                    likelihood_score += bonus
                    reasons.append(reason)
            
            # ML confidence from line classification
            confidence_column = 'confidence' if 'confidence' in row else 'confidence_score'
            ml_confidence = self.confidence_scorer.safe_convert(row[confidence_column])
            likelihood_score += ml_confidence * 0.2
            
            if likelihood_score > 0.3:
                cleaned_supplier = self.text_cleaner.clean_supplier_name(text)
                display_name = cleaned_supplier.title() if cleaned_supplier else text.title()
                
                supplier_candidates.append(SupplierExtractionResult(
                    supplier_name=display_name,
                    raw_text=text,
                    confidence=min(likelihood_score, 1.0),
                    line_number=self.confidence_scorer.safe_convert(row.get('line_number', 0), int),
                    extraction_method='intelligent_heuristic_cleaned',
                    reasons=reasons,
                    position=idx,
                    cleaned_from=text if cleaned_supplier != text else None
                ))
        
        # Select best candidate
        if supplier_candidates:
            best_heuristic = max(supplier_candidates, key=lambda x: x.confidence)
            print(f"🏆 Best heuristic match: '{best_heuristic.supplier_name}' (confidence: {best_heuristic.confidence:.3f})")
            return best_heuristic
        
        # Step 4: Basic fallback
        print("🔄 Applying final fallback logic...")
        for idx, row in header_lines.iterrows():
            text_column = 'cleaned_text' if 'cleaned_text' in row else ('text' if 'text' in row else 'original_text')
            text = str(row[text_column]).strip()
            
            if not re.search(r'^(RECEIPT|INVOICE|THANK|WELCOME|VAT\s*NO|TEL:|EMAIL:|STORE\s*#)', text.upper()):
                confidence_column = 'confidence' if 'confidence' in row else 'confidence_score'
                
                enhanced_from_line = self.text_cleaner.extract_supplier_name_from_line(text)
                cleaned_text = self.text_cleaner.clean_supplier_name(text)
                
                if enhanced_from_line and enhanced_from_line != text and len(enhanced_from_line) > 2:
                    display_name = enhanced_from_line.title()
                    method = 'fallback_enhanced_line_extraction'
                elif cleaned_text and cleaned_text != text:
                    display_name = cleaned_text.title()
                    method = 'fallback_cleaned'
                else:
                    display_name = text.title()
                    method = 'fallback_first_valid'
                
                return {
                    'supplier_name': display_name,
                    'raw_text': text,
                    'confidence': self.confidence_scorer.safe_convert(row[confidence_column]) * 0.3,
                    'line_number': self.confidence_scorer.safe_convert(row.get('line_number', 0), int),
                    'extraction_method': method
                }
        
        print("❌ No valid supplier name found")
        return None
    
    def detect_supplier_name_from_text(self, text: str, max_lines: int = 10, min_line_length: int = 2) -> str:
        """
        Dynamically detect supplier name from OCR text using enhanced pattern recognition.
        """
        ignore_keywords = [
            "revolut", "visa", "mastercard", "amex", "card", "contactless",
            "auth code", "pan sequence", "receipt", "transaction",
            "authorisation code", "terminal id", "merchant number", 
            "approved", "customer copy", "please keep"
        ]
        
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return "Unknown"

        candidate_lines = []
        for line in lines[:max_lines]:
            lower_line = line.lower()
            if any(keyword in lower_line for keyword in ignore_keywords):
                continue
            if len(re.sub(r'\d', '', line)) <= min_line_length:
                continue
            candidate_lines.append(line)

        best_line = "Unknown"
        for line in candidate_lines:
            if re.search(r"(www\.|\.com|\.co|\.uk|\.org|\.net|@)", line.lower()):
                continue
            if re.search(r"[A-Za-z]", line):
                best_line = self.text_cleaner.extract_supplier_name_from_line(line)
                if best_line:
                    break

        if best_line == "Unknown" and candidate_lines:
            best_line = self.text_cleaner.extract_supplier_name_from_line(candidate_lines[0])

        return best_line
    
    def get_enhanced_supplier_candidates(self, df):
        """
        Enhanced supplier candidate detection using contextual patterns.
        """
        text_column = 'cleaned_text' if 'cleaned_text' in df.columns else ('text' if 'text' in df.columns else 'original_text')
        all_text = ' '.join(df[text_column].fillna('').astype(str))
        lines = [line.strip() for line in all_text.split("\n") if line.strip()]
        
        supplier_candidates = []
        
        # Contextual patterns for supplier detection
        contextual_patterns = [
            (r'thank\s+you\s+for\s+(?:shopping|eating|visiting|dining)\s+(?:at|with)\s+(.+?)(?:\s|$|[.!])', 0.9),
            (r'thank\s+you\s+for\s+choosing\s+(.+?)(?:\s|$|[.!])', 0.9),
            (r'welcome\s+to\s+(.+?)(?:\s|$|[.!])', 0.85),
            (r'(.+?)\s+everyday\s+amazing', 0.85),
            (r'(.+?)\s+every\s+little\s+helps', 0.9),
            (r'(.+?)\s+good\s+food\s+costs\s+less', 0.9),
        ]
        
        for pattern, confidence_boost in contextual_patterns:
            for line_idx, line in enumerate(lines):
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    extracted_name = match.group(1).strip()
                    cleaned_name = self.text_cleaner.extract_supplier_name_from_line(extracted_name)
                    
                    if cleaned_name and len(cleaned_name) > 2 and cleaned_name.lower() != "unknown":
                        position_score = 0.5
                        if line_idx < len(lines) * 0.3:
                            position_score = 0.9
                        elif line_idx < len(lines) * 0.7:
                            position_score = 0.7
                        elif 'thank you' in line.lower() or 'welcome' in line.lower():
                            position_score = 0.8
                        
                        final_confidence = confidence_boost * position_score
                        
                        supplier_candidates.append({
                            'supplier_name': cleaned_name,
                            'confidence': final_confidence,
                            'raw_text': line,
                            'line_number': line_idx + 1,
                            'extraction_method': 'contextual_pattern'
                        })
        
        return supplier_candidates