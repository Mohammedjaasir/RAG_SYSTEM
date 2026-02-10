#!/usr/bin/env python3
"""
Phi-3 Item Extractor - LLM-Powered Item Extraction v1.0.0
Extracts items from full reconstructed receipt text using Phi-3.5 mini LLM
Designed for direct integration with line reconstruction output (unclassified text)

Features:
- Full context awareness from reconstructed OCR text
- Per-item confidence scoring based on LLM output
- Handles various receipt formats without pre-classification
- Field-level confidence breakdown
- Raw LLM response tracking for debugging
- Graceful error handling with fallback structures
"""

import json
import re
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging
import threading
from functools import lru_cache

# Optional llama_cpp import - fallback to HuggingFace if not available
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

logger = logging.getLogger(__name__)

# Global lock for thread-safe model access
_model_lock = threading.Lock()


@lru_cache(maxsize=1)
def load_phi3_prompts_config() -> Dict[str, Any]:
    """Load Phi-3 prompts configuration from JSON file (cached)."""
    # Try multiple config locations
    possible_paths = [
        Path(__file__).parent.parent.parent.parent.parent / "config" / "phi3_prompts_config.json",
        Path("/app/config/phi3_prompts_config.json"),
        Path("./config/phi3_prompts_config.json"),
        Path("config/phi3_prompts_config.json"),
    ]
    
    for config_path in possible_paths:
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    logger.info(f"✅ Loaded Phi-3 prompts config from: {config_path}")
                    return config
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
    
    logger.warning("⚠️ Could not load phi3_prompts_config.json, using default prompts")
    return None


class PhiItemExtractor:
    """
    Phi-3 Item Extractor - LLM-powered item extraction from full receipt text
    
    Features:
    - Accepts full reconstructed OCR text (no pre-classification needed)
    - Returns items in Document API format
    - Calculates per-item and per-field confidence scores
    - Includes rich metadata for debugging and transparency
    """
    
    def __init__(self, model_path: Optional[str] = None, temperature: float = 0.0):
        """
        Initialize Phi-3 Item Extractor.
        
        Args:
            model_path: Path to Phi-3.5 GGUF model file (optional if using HF backend)
            temperature: LLM temperature (0.0 for deterministic extraction)
        """
        # Load prompts configuration
        self.prompts_config = load_phi3_prompts_config()
        
        # Override temperature from config if available
        if self.prompts_config and 'model_settings' in self.prompts_config:
            config_temp = self.prompts_config['model_settings'].get('temperature')
            if config_temp is not None:
                self.temperature = config_temp
            else:
                self.temperature = temperature
        else:
            self.temperature = temperature
            
        self.llm = None
        self.hf_loader = None
        self.model_loaded = False
        self.backend = None  # 'llama_cpp' or 'huggingface'
        
        # Check backend preference
        backend_pref = os.getenv("PHI_BACKEND", "auto").lower()
        
        # Try HuggingFace backend first if llama_cpp not available or if preferred
        if (not LLAMA_CPP_AVAILABLE or backend_pref == "hf" or backend_pref == "huggingface"):
            try:
                from .phi_hf_loader import get_phi3_hf_loader
                self.hf_loader = get_phi3_hf_loader()
                self.backend = "huggingface"
                logger.info("Using HuggingFace backend for Phi-3")
                self.model_loaded = True
                return
            except Exception as e:
                logger.warning(f"HuggingFace backend not available: {e}")
        
        # Fall back to llama_cpp if available
        if LLAMA_CPP_AVAILABLE and backend_pref != "hf":
            self.model_path = model_path
            self._load_model_llama_cpp()
        else:
            logger.warning("No Phi-3 backend available (llama_cpp or HuggingFace)")
            self.model_path = None
    
    def _load_model_llama_cpp(self):
        """Load Phi-3.5 model using llama-cpp backend."""
        if not self.model_path:
            # Try common locations for GGUF model
            # Current file: app/services/extraction/receipt/extraction/phi_item_extractor.py
            # Need to find: app/models/receipts/phi-3.5-mini-instruct-q4_k_m.gguf
            
            # Method 1: Use environment variable (best for Docker)
            model_from_env = os.getenv("PHI_MODEL_PATH")
            if model_from_env:
                possible_paths = [Path(model_from_env)]
            else:
                # Method 2: Calculate from current file location
                # Go up from: app/services/extraction/receipt/extraction/
                # To reach: app/
                # That's 5 levels up, then down to models/receipts/
                current_file = Path(__file__).resolve()
                app_dir = current_file.parent.parent.parent.parent.parent  # Go up to 'app' dir
                
                possible_paths = [
                    # Primary: From app directory
                    app_dir / "models" / "receipts" / "phi-3.5-mini-instruct-q4_k_m.gguf",
                    # Docker absolute path
                    Path("/app/models/receipts/phi-3.5-mini-instruct-q4_k_m.gguf"),
                    # Local development absolute path
                    Path("./app/models/receipts/phi-3.5-mini-instruct-q4_k_m.gguf").resolve(),
                    # Relative from working directory
                    Path("app/models/receipts/phi-3.5-mini-instruct-q4_k_m.gguf"),
                ]
            
            for path in possible_paths:
                abs_path = Path(path).resolve()
                logger.debug(f"Checking for Phi-3 model at: {abs_path}")
                if abs_path.exists():
                    self.model_path = str(abs_path)
                    logger.info(f"✅ Found Phi-3 model at: {self.model_path}")
                    break
            
            if not self.model_path:
                logger.warning(f"❌ Phi-3 model not found in any of the checked locations:")
                for path in possible_paths:
                    logger.warning(f"   - {Path(path).resolve()}")
                return
        
        if not Path(self.model_path).exists():
            logger.warning(f"Phi-3 model file not found at: {self.model_path}")
            return
        
        try:
            logger.info(f"Loading Phi-3.5 model from: {self.model_path}")
            # Thread-safe model loading with increased context
            with _model_lock:
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=8192,  # Increased from 4096 to handle longer receipts
                    n_threads=2,  # Reduced from 4 to prevent resource contention
                    n_batch=512,  # Add batch size for stability
                    verbose=False,
                    use_mlock=True,  # Lock model in memory to prevent swapping
                    n_gpu_layers=0  # Force CPU mode (no GPU)
                )
            self.model_loaded = True
            self.backend = "llama_cpp"
            logger.info("✅ Phi-3.5 model loaded successfully via llama-cpp (thread-safe)")
        except Exception as e:
            logger.error(f"❌ Failed to load Phi-3 model: {e}", exc_info=True)
            self.model_loaded = False
    
    def extract_items(self, reconstructed_text: str) -> List[Dict[str, Any]]:
        """
        Extract items from full reconstructed OCR text using Phi-3.
        
        Args:
            reconstructed_text: Full reconstructed text from line reconstruction
            
        Returns:
            List of item dictionaries in Document API format
        """
        if not self.model_loaded:
            logger.error("Phi-3 model not loaded, returning empty items list")
            return []
        
        if not reconstructed_text or reconstructed_text.strip() == "":
            logger.warning("Empty reconstructed text provided")
            return []
        
        try:
            logger.info("Starting Phi-3 item extraction from full receipt text...")
            
            # Call Phi-3 to extract items
            phi_response = self._call_phi3(reconstructed_text)
            
            if not phi_response:
                logger.warning("Phi-3 returned no response")
                return []
            
            # Parse and adapt response to Document API format
            items = self._adapt_phi_response_to_api_format(phi_response)
            
            logger.info(f"✅ Phi-3 extracted {len(items)} items from receipt")
            return items
            
        except Exception as e:
            logger.error(f"❌ Error during Phi-3 item extraction: {e}")
            return []
    
    def _call_phi3(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Call Phi-3 LLM to extract items from text.
        
        Args:
            text: Reconstructed receipt text
            
        Returns:
            Parsed JSON response from Phi-3
        """
        # Load system prompt from config or use default
        if self.prompts_config and 'basic_extraction' in self.prompts_config:
            system_prompt = self.prompts_config['basic_extraction']['system_prompt']
            output_schema = "\n\n" + self.prompts_config['basic_extraction']['output_schema']
            system_prompt += output_schema
        else:
            # Fallback to hardcoded prompt if config not available
            system_prompt = (
                        "You are a world-class receipt understanding and information extraction engine. "
                        "Your task is to extract ALL possible structured data from a receipt image or OCR text. "
                        "You must NEVER guess or hallucinate values. "
                        "Only extract what is explicitly present in the receipt. "
                        "Return ONLY valid JSON. No explanations."
                    )

        
        # Build prompt using template from config
        if self.prompts_config and 'prompt_templates' in self.prompts_config:
            template = self.prompts_config['prompt_templates']['phi3_chat_format']
            prompt = template.format(system_prompt=system_prompt, user_prompt=text)
        else:
            # Fallback to default template
            prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{text}<|end|>\n<|assistant|>\n{{"
        
        try:
            logger.debug(f"Calling Phi-3 ({self.backend}) with prompt length: {len(prompt)}")
            
            # Use appropriate backend with thread safety
            if self.backend == "huggingface" and self.hf_loader:
                raw_response = self.hf_loader.generate(prompt, max_tokens=1500, temperature=self.temperature)
            elif self.backend == "llama_cpp" and self.llm:
                # Thread-safe model inference - CRITICAL for multi-worker environments
                with _model_lock:
                    logger.debug("Acquired model lock for inference")
                    try:
                        # Get stop sequences from config
                        if self.prompts_config and 'model_settings' in self.prompts_config:
                            stop_seqs = self.prompts_config['model_settings'].get('stop_sequences', ["<|end|>", "</s>"])
                            max_tokens = self.prompts_config['model_settings'].get('max_tokens', 1500)
                        else:
                            stop_seqs = ["<|end|>", "</s>"]
                            max_tokens = 1500
                            
                        output = self.llm(
                            prompt,
                            max_tokens=max_tokens,
                            temperature=self.temperature,
                            stop=stop_seqs,
                            echo=False
                        )
                        raw_response = output["choices"][0]["text"].strip()
                        logger.debug("Released model lock after inference")
                    except Exception as inference_error:
                        logger.error(f"Segfault prevention: Model inference failed: {inference_error}", exc_info=True)
                        return None
            else:
                logger.error("No Phi-3 backend available")
                return None
            
            # Log full response for debugging
            logger.info(f"Phi-3 raw response length: {len(raw_response)} chars")
            logger.info(f"Phi-3 full raw response:\n{raw_response}")
            
            # Parse JSON response
            parsed_response = self._parse_phi3_response(raw_response)
            if parsed_response is None:
                logger.error(f"Failed to parse Phi-3 response. Full response:\n{raw_response}")
            else:
                logger.info(f"Successfully parsed Phi-3 response: {json.dumps(parsed_response, indent=2)}")
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error calling Phi-3: {e}", exc_info=True)
            return None
    
    def _preprocess_receipt_text(self, text: str) -> str:
        """
        Preprocess receipt text to aggressively reconstruct receipt structure.
        OCR often scrambles: "Name Price Total" into a single line with mixed data.
        """
        import re
        
        # First pass: Identify and separate sections
        lines = text.split('\n')
        item_section = []
        header_lines = []
        footer_lines = []
        total_lines = []
        
        in_items = False
        past_items = False
        
        for line in lines:
            line_clean = line.strip()
            if not line_clean:
                continue
            
            # Header detection (merchant, date, VAT reg, etc.)
            if any(keyword in line_clean.lower() for keyword in ['thank you', 'shopping at', 'tel :', 'site id', 'vat reg']):
                header_lines.append(line_clean)
                continue
            
            # Item section start detection
            if any(keyword in line_clean.lower() for keyword in ['qty name', 'name price', 'item']):
                in_items = True
                continue
            
            # Total/footer section detection
            if any(keyword in line_clean.lower() for keyword in ['subtotal', 'total savings', 'promotion', 'vat rate', 'visa', 'mastercard', 'purchase', 'approved']):
                past_items = True
                in_items = False
                total_lines.append(line_clean)
                continue
            
            # Classify line based on context
            if past_items:
                footer_lines.append(line_clean)
            elif in_items or (not header_lines and not past_items):
                # Potential item line - look for prices
                if re.search(r'\d+\.\d{2}', line_clean):
                    item_section.append(line_clean)
                else:
                    item_section.append(line_clean)
            else:
                header_lines.append(line_clean)
        
        # Second pass: Parse item lines aggressively
        # Pattern: "ITEMNAME  PRICE  PRICE  CODE" where CODE is B/S/E/Z
        structured_items = []
        price_pattern = r'(\d+\.?\d*)\s+(\d+\.?\d*)\s*([BSEZR])?'
        
        for item_line in item_section:
            # Find all prices in line
            matches = list(re.finditer(r'\d+\.\d{2}', item_line))
            
            if len(matches) >= 2:
                # Extract item name (everything before first price)
                first_price_pos = matches[0].start()
                item_name = item_line[:first_price_pos].strip()
                
                # Clean item name (remove table header words)
                item_name = re.sub(r'\b(qty|name|price|total)\b', '', item_name, flags=re.IGNORECASE).strip()
                item_name = re.sub(r'\s{2,}', ' ', item_name)  # Collapse whitespace
                
                if item_name and len(item_name) >= 2:
                    # Get prices
                    unit_price = matches[0].group()
                    total_price = matches[1].group() if len(matches) > 1 else unit_price
                    
                    # Get VAT code (B/S/E/Z at end of line)
                    vat_code = ''
                    vat_match = re.search(r'\s+([BSEZR])\s*$', item_line)
                    if vat_match:
                        vat_code = vat_match.group(1)
                    
                    structured_items.append(f"ITEM: {item_name} | Price: £{unit_price} | Total: £{total_price} | VAT: {vat_code}")
        
        # Reconstruct text with clear structure
        reconstructed = []
        
        if header_lines:
            reconstructed.append("=== HEADER ===")
            reconstructed.extend(header_lines)
        
        if structured_items:
            reconstructed.append("\n=== ITEMS ===")
            reconstructed.extend(structured_items)
        
        if total_lines:
            reconstructed.append("\n=== TOTALS ===")
            reconstructed.extend(total_lines)
        
        if footer_lines:
            reconstructed.append("\n=== FOOTER ===")
            reconstructed.extend(footer_lines)
        
        return '\n'.join(reconstructed)
    
    def _parse_phi3_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse Phi-3 JSON response with repair attempts.
        
        Args:
            response_text: Raw text response from Phi-3
            
        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        try:
            # Clean up common issues
            response_text = response_text.strip()
            
            # Hallucination detection is now done BEFORE parsing in _call_phi3_full_receipt
            # This function focuses on JSON parsing and repair only
            
            # Remove markdown code blocks
            response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            # The prompt ends with "{", so we need to add it if missing
            if not response_text.startswith("{"):
                response_text = "{" + response_text
            
            # Try direct parsing first (with decoder that stops at first object)
            try:
                # Use JSONDecoder to parse only the first object
                decoder = json.JSONDecoder()
                parsed, idx = decoder.raw_decode(response_text)
                if idx < len(response_text):
                    logger.debug(f"Ignored extra data after JSON (position {idx})")
                
                # Check if we got a valid receipt object (should have multiple keys)
                if len(parsed.keys()) < 3:
                    logger.warning(f"Parsed object has too few keys ({list(parsed.keys())}), trying to extract full JSON")
                    raise json.JSONDecodeError("Incomplete object", response_text, 0)
                
                return parsed
            except json.JSONDecodeError as e:
                logger.debug(f"Direct JSON parse failed: {e}, attempting repairs...")
                
                # Attempt 0: Fix common Phi-3 typos
                repaired = response_text
                # Fix "confidence": 0-1 → "confidence": 0.9
                repaired = re.sub(r'"confidence":\s*0-1\b', '"confidence": 0.9', repaired)
                # Fix "confidence": 1-0 → "confidence": 1.0
                repaired = re.sub(r'"confidence":\s*1-0\b', '"confidence": 1.0', repaired)
                # Fix malformed decimals like 0.0-1 → 0.9
                repaired = re.sub(r'\b0\.0-1\b', '0.9', repaired)
                # Fix missing decimal point: 09 → 0.9
                repaired = re.sub(r'"confidence":\s*09\b', '"confidence": 0.9', repaired)
                
                try:
                    parsed = json.loads(repaired)
                    if isinstance(parsed, dict) and len(parsed.keys()) >= 3:
                        logger.info("Successfully repaired JSON by fixing confidence typos")
                        return parsed
                except json.JSONDecodeError:
                    pass
                
                # Attempt 1: Extract only the first complete JSON object using brace counting
                brace_count = 0
                json_end = -1
                for i, char in enumerate(response_text):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > 0:
                    first_object = response_text[:json_end]
                    # Also try fixing typos in the extracted object
                    first_object = re.sub(r'"confidence":\s*0-1\b', '"confidence": 0.9', first_object)
                    first_object = re.sub(r'"confidence":\s*1-0\b', '"confidence": 1.0', first_object)
                    
                    try:
                        parsed = json.loads(first_object)
                        if len(parsed.keys()) >= 3:  # Valid receipt has multiple fields
                            logger.info("Successfully extracted first JSON object")
                            return parsed
                    except json.JSONDecodeError:
                        pass
                
                # Attempt 2: Fix incomplete JSON by closing braces
                repaired = response_text
                # Apply typo fixes first
                repaired = re.sub(r'"confidence":\s*0-1\b', '"confidence": 0.9', repaired)
                repaired = re.sub(r'"confidence":\s*1-0\b', '"confidence": 1.0', repaired)
                
                open_braces = repaired.count('{') - repaired.count('}')
                open_brackets = repaired.count('[') - repaired.count(']')
                
                if open_brackets > 0:
                    repaired += ']' * open_brackets
                if open_braces > 0:
                    repaired += '}' * open_braces
                
                try:
                    decoder = json.JSONDecoder()
                    parsed, _ = decoder.raw_decode(repaired)
                    logger.info("Successfully repaired JSON by closing brackets/braces")
                    return parsed
                except json.JSONDecodeError:
                    pass
                
                # Attempt 3: Fix trailing commas
                repaired = re.sub(r',(\s*[}\]])', r'\1', response_text)
                # Also try adding closing braces
                open_braces = repaired.count('{') - repaired.count('}')
                open_brackets = repaired.count('[') - repaired.count(']')
                if open_brackets > 0:
                    repaired += ']' * open_brackets
                if open_braces > 0:
                    repaired += '}' * open_braces
                    
                try:
                    decoder = json.JSONDecoder()
                    parsed, _ = decoder.raw_decode(repaired)
                    logger.info("Successfully repaired JSON by removing trailing commas and closing braces")
                    return parsed
                except json.JSONDecodeError:
                    pass
                
                # Attempt 4: Fix missing quotes around keys
                repaired = re.sub(r'(\w+):', r'"\1":', response_text)
                try:
                    decoder = json.JSONDecoder()
                    parsed, _ = decoder.raw_decode(repaired)
                    logger.info("Successfully repaired JSON by adding quotes to keys")
                    return parsed
                except json.JSONDecodeError:
                    pass
                
                logger.warning(f"All JSON repair attempts failed. Error: {e}")
                logger.error(f"Failed to parse Phi-3 response (first 500 chars):\n{response_text[:500]}")
                return None
            
        except Exception as e:
            logger.error(f"Unexpected error parsing Phi-3 response: {e}")
            logger.error(f"Response text (first 500 chars):\n{response_text[:500]}")
            return None
    
    def _adapt_phi_response_to_api_format(self, phi_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Adapt Phi-3 response to Document API item format.
        
        Phi-3 format:
        {
            'items': [
                {
                    'item': str,
                    'quantity': int|float,
                    'unit_price': float,
                    'price': float,
                    'vat_code': str
                }
            ]
        }
        
        Document API format:
        {
            'item_name': str,
            'item_quantity': str|null,
            'item_unit_price': str|null,
            'item_price': float|null,
            'item_amount': float|null,
            'item_vat_code': str|null,
            'item_code': null,
            'line_number': int,
            'extraction_confidence': float,
            'confidence_breakdown': {...},
            'extraction_details': {...}
        }
        
        Args:
            phi_response: Parsed Phi-3 response
            
        Returns:
            List of items in Document API format
        """
        items = []
        phi_items = phi_response.get('items', [])
        
        if not phi_items:
            logger.warning("No items found in Phi-3 response")
            return []
        
        for line_number, phi_item in enumerate(phi_items):
            try:
                # Extract fields from Phi-3 response
                item_name = phi_item.get('item', '').strip()
                quantity = phi_item.get('quantity')
                unit_price = phi_item.get('unit_price')
                price = phi_item.get('price')
                vat_code = phi_item.get('vat_code')
                
                # Skip if no item name
                if not item_name:
                    continue
                
                # Calculate per-field confidence scores
                confidence_breakdown = self._calculate_field_confidence(
                    item_name, quantity, unit_price, price, vat_code
                )
                
                # Overall confidence (average of non-zero field confidences)
                field_confidences = [v for v in confidence_breakdown.values() if v > 0]
                overall_confidence = (
                    sum(field_confidences) / len(field_confidences) 
                    if field_confidences else 0.0
                )
                
                # Format unit price as string (e.g., "£1.50/unit")
                unit_price_str = None
                if unit_price is not None:
                    try:
                        unit_price_float = float(unit_price)
                        unit_price_str = f"£{unit_price_float:.2f}/unit"
                    except (ValueError, TypeError):
                        pass
                
                # Format price as float
                price_float = None
                if price is not None:
                    try:
                        price_float = float(price)
                    except (ValueError, TypeError):
                        pass
                
                # Convert quantity to string
                quantity_str = None
                if quantity is not None:
                    quantity_str = str(quantity)
                
                # Build Document API item format
                api_item = {
                    'raw_text': item_name,  # Store original for reference
                    'item_code': None,  # Phi-3 doesn't extract codes
                    'item_name': item_name,
                    'item_quantity': quantity_str,
                    'item_unit_price': unit_price_str,
                    'item_price': price_float,
                    'item_amount': price_float,  # Same as item_price
                    'item_vat_code': vat_code,
                    'line_number': line_number,
                    'confidence': 0.95,  # Phi-3 confidence (fixed high)
                    'extraction_confidence': overall_confidence,
                    'confidence_breakdown': confidence_breakdown,
                    'extraction_details': {
                        'extraction_method': 'phi_3_llm',
                        'extraction_source': 'phi_item_extractor',
                        'model_version': 'phi-3.5-mini-instruct-q4_k_m',
                        'llm_raw_response': phi_item,
                        'parsing_errors': [],
                        'temperature': self.temperature
                    }
                }
                
                items.append(api_item)
                logger.debug(f"Item {line_number}: {item_name} (confidence: {overall_confidence:.2f})")
                
            except Exception as e:
                logger.warning(f"Error processing item {line_number}: {e}")
                continue
        
        logger.info(f"Adapted {len(items)} items from Phi-3 response to API format")
        return items
    
    def _calculate_field_confidence(
        self,
        item_name: str,
        quantity: Any,
        unit_price: Any,
        price: Any,
        vat_code: str
    ) -> Dict[str, float]:
        """
        Calculate confidence scores for each field based on Phi-3 output.
        
        Args:
            item_name: Item name from Phi-3
            quantity: Quantity value
            unit_price: Unit price value
            price: Item price
            vat_code: VAT code
            
        Returns:
            Dictionary of field confidence scores
        """
        confidence = {}
        
        # Item name confidence
        if item_name and len(item_name) >= 3:
            # Valid item name
            confidence['item_name'] = 0.95
        elif item_name:
            confidence['item_name'] = 0.50
        else:
            confidence['item_name'] = 0.0
        
        # Quantity confidence
        if quantity is not None:
            try:
                float_val = float(quantity)
                if float_val > 0:
                    confidence['quantity'] = 0.90
                else:
                    confidence['quantity'] = 0.30
            except (ValueError, TypeError):
                confidence['quantity'] = 0.20
        else:
            confidence['quantity'] = 0.0
        
        # Unit price confidence
        if unit_price is not None:
            try:
                float_val = float(unit_price)
                if float_val > 0:
                    confidence['unit_price'] = 0.90
                else:
                    confidence['unit_price'] = 0.30
            except (ValueError, TypeError):
                confidence['unit_price'] = 0.20
        else:
            confidence['unit_price'] = 0.0
        
        # Price confidence
        if price is not None:
            try:
                float_val = float(price)
                if float_val > 0:
                    confidence['price'] = 0.95  # Most important field
                else:
                    confidence['price'] = 0.30
            except (ValueError, TypeError):
                confidence['price'] = 0.20
        else:
            confidence['price'] = 0.0
        
        # VAT code confidence
        if vat_code and len(vat_code) <= 3:
            confidence['vat_code'] = 0.85
        elif vat_code:
            confidence['vat_code'] = 0.40
        else:
            confidence['vat_code'] = 0.0
        
        return confidence
    
    def get_stats(self) -> Dict[str, Any]:
        """Get extractor statistics."""
        return {
            'model_loaded': self.model_loaded,
            'backend': self.backend,
            'model_path': str(self.model_path) if self.model_path else None,
            'temperature': self.temperature,
            'extraction_method': 'phi_3_llm'
        }
    
    def extract_full_receipt(
        self,
        normalized_text: str,
        layout_json: Optional[str] = None,
        rag_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract complete receipt with all fields and per-field confidence."""
        if not self.model_loaded:
            return self._empty_receipt_result()
        if not normalized_text or not normalized_text.strip():
            return self._empty_receipt_result()
        
        try:
            logger.info("Starting Phi-3 full receipt extraction...")
            response = self._call_phi3_full_receipt(normalized_text, layout_json, rag_context)
            if not response:
                return self._empty_receipt_result()
            return self._normalize_full_response(response)
        except Exception as e:
            logger.error(f"Full receipt extraction error: {e}")
            return self._empty_receipt_result()
    
    def _call_phi3_full_receipt(self, text: str, layout_json: Optional[str], rag_context: Optional[str]) -> Optional[Dict]:
        """Call Phi-3 for comprehensive receipt extraction."""
        
        # Load prompts from config or use defaults
        if self.prompts_config and 'full_receipt_extraction' in self.prompts_config:
            config = self.prompts_config['full_receipt_extraction']
            base_sys_prompt = config['base_system_prompt']
            
            # Add RAG context as KNOWLEDGE GUIDANCE (not reference data)
            if rag_context:
                sys_prompt = base_sys_prompt + "\n\n" + config['rag_context_template'].format(rag_context=rag_context)
            else:
                sys_prompt = base_sys_prompt
            
            sys_prompt += "\n\n" + config['receipt_format_guide']
        else:
            # Fallback to hardcoded prompt
            base_sys_prompt = ("You are a specialized receipt data extraction system. "
                              "Your ONLY task is to extract structured JSON data from preprocessed receipt text.")
            sys_prompt = base_sys_prompt
        
        # Preprocess text to restructure scrambled OCR
        preprocessed_text = self._preprocess_receipt_text(text)
        
        # Build user prompt - ONLY include actual receipt data
        parts = [f"=== RECEIPT TEXT TO EXTRACT FROM ===\n{preprocessed_text}"]
        if layout_json:
            parts.append(f"\n=== LAYOUT COORDINATES (SUPPLEMENTARY) ===\n{layout_json}")
        
        # Get extraction task from config
        if self.prompts_config and 'full_receipt_extraction' in self.prompts_config:
            config = self.prompts_config['full_receipt_extraction']
            task = "\n\n" + config['extraction_task'] + "\n\n" + config['mandatory_extractions']
        else:
            # Fallback to minimal extraction task
            task = "\n\nEXTRACT TO JSON. Include items, totals, and payments."
        
        parts.append(task)
        user_prompt = "\n".join(parts)
        
        # Build Phi-3 prompt with special tokens
        sys_tag, end_tag, user_tag, asst_tag = "<" + "|system|>", "<" + "|end|>", "<" + "|user|>", "<" + "|assistant|>"
        full_prompt = f"{sys_tag}\n{sys_prompt}{end_tag}\n{user_tag}\n{user_prompt}{end_tag}\n{asst_tag}\n{{"
        
        logger.info(f"Phi-3 prompt length: {len(full_prompt)} chars")
        
        try:
            # Get stop sequences from config
            if self.prompts_config and 'model_settings' in self.prompts_config:
                stop_sequences = self.prompts_config['model_settings'].get('stop_sequences', [
                    end_tag, "</s>", "\n\n\n", "<|end|>", "<|user|>", "<|system|>"
                ])
                max_tokens = self.prompts_config['model_settings'].get('max_tokens', 3000)
            else:
                # Fallback stop sequences
                stop_sequences = [
                    end_tag, "</s>", "\n\n\n", "<|end|>", "<|user|>", "<|system|>",
                    "\nWritten by:",
                    "\nThe article",  
                    "\nThis article",
                ]
                max_tokens = 3000
            
            if self.backend == "huggingface" and self.hf_loader:
                raw_response = self.hf_loader.generate(full_prompt, max_tokens=max_tokens, temperature=self.temperature)
            elif self.backend == "llama_cpp" and self.llm:
                # Thread-safe model inference - CRITICAL for multi-worker environments
                with _model_lock:
                    logger.debug("Acquired model lock for full receipt extraction")
                    try:
                        output = self.llm(
                            full_prompt,
                            max_tokens=max_tokens,
                            temperature=self.temperature,
                            stop=stop_sequences,
                            repeat_penalty=1.05,
                            echo=False
                        )
                        raw_response = output["choices"][0]["text"].strip()
                        logger.debug("Released model lock after full receipt extraction")
                    except Exception as inference_error:
                        logger.error(f"Segfault prevention: Full receipt inference failed: {inference_error}", exc_info=True)
                        return None
            else:
                logger.error("No Phi-3 backend available")
                return None
            
            logger.info(f"✅ Phi-3 responded with {len(raw_response)} chars")
            
            # Check for major hallucinations (articles, long narrative text)
            # Allow minor contamination in JSON field values (will be cleaned later)
            response_lower = raw_response.lower()
            
            # Major hallucination patterns that indicate complete failure
            major_hallucination_patterns = [
                ('dr.', 'psychologist'),  # Academic bio (needs both)
                ('dr.', 'professor'),
                ('article', 'furthermore,'),  # Academic article
                ('this study', 'participants'),
                ('cognitive behavioral', 'therapy'),
                'translate the following',  # Translation exercise
                'quick brown fox',
                'le vif renard',
            ]
            
            # Check for major hallucinations requiring BOTH indicators
            major_hallucination = False
            for pattern in major_hallucination_patterns:
                if isinstance(pattern, tuple):
                    if pattern[0] in response_lower and pattern[1] in response_lower:
                        logger.error(f"🚫 MAJOR HALLUCINATION: Found both '{pattern[0]}' and '{pattern[1]}'")
                        major_hallucination = True
                        break
                elif pattern in response_lower:
                    logger.error(f"🚫 MAJOR HALLUCINATION: Found '{pattern}'")
                    major_hallucination = True
                    break
            
            if major_hallucination:
                logger.error(f"Response preview: {raw_response[:500]}")
                logger.error("Rejecting response due to severe hallucination")
                return None
            
            # Warn about minor contamination but continue
            minor_indicators = ['written by:', 'written in english', 'translate']
            for indicator in minor_indicators:
                if indicator in response_lower:
                    logger.warning(f"⚠️ Minor contamination detected: '{indicator}' - will attempt cleanup")
            
            # Truncate response if it's suspiciously long (likely hallucination)
            if len(raw_response) > 5000:
                logger.warning(f"⚠️ Response too long ({len(raw_response)} chars), truncating to 5000")
                raw_response = raw_response[:5000]
            
            # Log first 1000 chars of response for debugging
            logger.info(f"Phi-3 RAW RESPONSE (first 1000 chars):\n{raw_response[:1000]}\n{'='*80}")
            
            parsed = self._parse_phi3_response(raw_response)
            if parsed:
                logger.info(f"✅ Parsed response with {len(parsed)} top-level keys")
            else:
                logger.error(f"❌ Failed to parse Phi-3 response")
            return parsed
        except Exception as e:
            logger.error(f"Phi-3 call error: {e}")
            return None
            return None
    
    def _empty_receipt_result(self) -> Dict[str, Any]:
        """Return empty receipt result structure."""
        return {
            "supplier_name": {"value": None, "confidence": 0.0},
            "supplier_address": [],
            "supplier_phone": [],
            "supplier_email": [],
            "supplier_website": [],
            "receipt_number": {"value": None, "confidence": 0.0},
            "receipt_date": {"date": None, "raw_text": "", "confidence": 0.0},
            "totals": {
                "subtotal": {"amount": None, "confidence": 0.0},
                "tax_amount": {"amount": None, "confidence": 0.0},
                "final_total": {"amount": None, "confidence": 0.0},
                "items_subtotal": {"amount": None, "confidence": 0.0},
                "net_after_discount": {"amount": None, "confidence": 0.0}
            },
            "item_list": [],
            "vat_information": {"vat_data_entries": []},
            "payment_methods": [],
            "card_details": [],
            "discount_items": [],
            "coupon_items": [],
            "loyalty_savings": []
        }
    
    def _normalize_full_response(self, response: Dict) -> Dict[str, Any]:
        """Normalize and validate full receipt response."""
        logger.info(f"Normalizing full response with keys: {list(response.keys())}")
        
        result = self._empty_receipt_result()
        
        # Extract supplier data from nested structure
        supplier = response.get("supplier", {})
        if isinstance(supplier, dict):
            # Supplier name
            if supplier.get("name"):
                result["supplier_name"] = {"value": supplier["name"], "confidence": 0.95}
                logger.info(f"✅ Extracted supplier name: {supplier['name']}")
            
            # Supplier address
            addr = supplier.get("address", {})
            if isinstance(addr, dict) and any(addr.values()):
                addr_parts = []
                for key in ["street", "city", "state", "postal_code", "country"]:
                    if addr.get(key):
                        addr_parts.append({"value": addr[key], "confidence": 0.9})
                if addr_parts:
                    result["supplier_address"] = addr_parts
            
            # Phone, email, website (already arrays)
            if supplier.get("phone"):
                result["supplier_phone"] = [{"value": p, "confidence": 0.9} for p in supplier["phone"] if p]
            if supplier.get("email"):
                result["supplier_email"] = [{"value": e, "confidence": 0.95} for e in supplier["email"] if e]
            if supplier.get("website"):
                result["supplier_website"] = [{"value": w, "confidence": 0.9} for w in supplier["website"] if w]
        
        # Extract receipt data
        receipt = response.get("receipt", {})
        if isinstance(receipt, dict):
            # Receipt number
            if receipt.get("receipt_number"):
                rn = receipt["receipt_number"]
                result["receipt_number"] = {"value": rn, "confidence": 0.9}
                logger.info(f"✅ Extracted receipt number: {rn}")
            
            # Receipt date - clean up contamination
            if receipt.get("date"):
                date_str = receipt["date"]
                
                # Clean contamination from date field
                import re
                # Remove narrative text fragments
                date_str = re.sub(r'\s*written by:?\s*', ' ', date_str, flags=re.IGNORECASE)
                date_str = re.sub(r'\s*translate\s*', ' ', date_str, flags=re.IGNORECASE)
                # Extract date pattern (YYYY-MM-DD or DD/MM/YYYY)
                date_pattern = r'(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})'
                date_match = re.search(date_pattern, date_str)
                if date_match:
                    clean_date = date_match.group(1)
                    # Convert DD/MM/YYYY to YYYY-MM-DD
                    if '/' in clean_date:
                        parts = clean_date.split('/')
                        if len(parts) == 3:
                            clean_date = f"{parts[2]}-{parts[1]}-{parts[0]}"
                    date_str = clean_date
                    logger.info(f"✅ Extracted date (cleaned): {date_str}")
                else:
                    logger.warning(f"⚠️ Could not parse date: {date_str}")
                
                result["receipt_date"] = {
                    "date": date_str,
                    "raw_text": date_str, 
                    "confidence": 0.95
                }
        
        # Totals (check both top level and nested in receipt)
        totals = response.get("totals")
        if not totals and isinstance(receipt, dict):
            totals = receipt.get("totals")
        
        if totals and isinstance(totals, dict):
            result["totals"] = totals
            logger.info(f"✅ Extracted totals: {totals}")
        
        # Items (check both top level and nested in receipt - Phi-3 often nests incorrectly)
        items_raw = response.get("items", [])
        if not items_raw and isinstance(receipt, dict):
            items_raw = receipt.get("items", [])
            if items_raw:
                logger.warning("⚠️ Items were nested inside 'receipt' - extracting anyway")
        
        if isinstance(items_raw, list) and items_raw:
            result["item_list"] = items_raw
            logger.info(f"✅ Extracted {len(items_raw)} items")
        
        # VAT (check both top level and nested in receipt)
        vat_info = response.get("vat_information")
        if not vat_info and isinstance(receipt, dict):
            vat_info = receipt.get("vat_information")
        
        if vat_info:
            result["vat_information"] = vat_info
            if isinstance(vat_info, list):
                logger.info(f"✅ Extracted {len(vat_info)} VAT entries")
            else:
                logger.info(f"✅ Extracted VAT information")
        
        # Payments (check both top level and nested in receipt)
        payments = response.get("payments", [])
        if not payments and isinstance(receipt, dict):
            payments = receipt.get("payments", [])
        
        if isinstance(payments, list) and payments:
            result["payment_methods"] = [{"method": p.get("method"), "amount": p.get("amount"), "confidence": p.get("confidence", 0.9)} for p in payments]
            logger.info(f"✅ Extracted {len(payments)} payment methods")
            
            # Extract card details
            card_details = []
            for p in payments:
                if p.get("card_brand"):
                    card_details.append({
                        "card_brand": p["card_brand"],
                        "confidence": p.get("confidence", 0.9)
                    })
            if card_details:
                result["card_details"] = card_details
        
        # Discounts (check both top level and nested in receipt)
        discounts = response.get("discounts", [])
        if not discounts and isinstance(receipt, dict):
            discounts = receipt.get("discounts", [])
        
        if isinstance(discounts, list) and discounts:
            result["discount_items"] = discounts
            logger.info(f"✅ Extracted {len(discounts)} discounts")
        
        logger.info(f"📦 Normalized result: supplier={result['supplier_name']['value']}, items={len(result['item_list'])}, date={result['receipt_date']['date']}")
        
        return result


# Singleton instance for API integration
_phi_item_extractor_instance = None


def get_phi_item_extractor(model_path: Optional[str] = None) -> PhiItemExtractor:
    """Get or create singleton Phi-3 Item Extractor instance."""
    global _phi_item_extractor_instance
    
    if _phi_item_extractor_instance is None:
        _phi_item_extractor_instance = PhiItemExtractor(model_path=model_path)
    
    return _phi_item_extractor_instance
