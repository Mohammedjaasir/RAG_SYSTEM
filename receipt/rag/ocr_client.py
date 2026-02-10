#!/usr/bin/env python3
"""
OCR Client - Receipt Image Processing via Microservice v1.0.0

Connects to the OCR microservice (doctr) running on port 8001 to:
- Process receipt images (JPG, PNG, PDF)
- Extract raw text with bounding box coordinates
- Return structured {text, layout} for RAG pipeline

Usage:
    from app.services.extraction.receipt.rag.ocr_client import get_ocr_client
    
    client = get_ocr_client()
    result = client.process_image(image_bytes, "receipt.jpg")
    # result = {"text": "...", "layout": {...}, "ocr_confidence": 0.95}
"""

import os
import logging
import requests
import base64
import yaml
import re
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Structured OCR result with text and layout."""
    text: str
    layout: Dict[str, Any]
    raw_response: Dict[str, Any]
    ocr_confidence: float
    word_count: int
    line_count: int


class ReceiptOCRClient:
    """
    OCR Client for receipt image processing.
    
    Connects to doctr microservice on configurable port (default 8001).
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 120
    ):
        """
        Initialize OCR client.
        
        Args:
            base_url: Base URL of OCR service (default from config or env)
            timeout: Request timeout in seconds
        """
        # Try to get URL from environment or config
        if base_url is None:
            base_url = self._get_ocr_url()
        
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.process_endpoint = "/process"
        self.health_endpoint = "/health"
        
        logger.info(f"✅ ReceiptOCRClient initialized: {self.base_url}")
    
    def _get_ocr_url(self) -> str:
        """Get OCR URL from environment or config file."""
        # Try environment variable first
        url = os.getenv('RECEIPT_OCR_URL') or os.getenv('OCR_SERVICE_URL')
        if url:
            return url
        
        # Try to load from config file
        try:
            from config.ocr_services_manager import get_ocr_services_config
            config = get_ocr_services_config()
            return config.receipt_ocr_url
        except Exception as e:
            logger.warning(f"Could not load OCR config: {e}")
        
        # Default fallback - external receipt OCR microservice
        return "http://149.102.159.66:8001"
    
    def health_check(self) -> bool:
        """Check if OCR service is healthy."""
        try:
            resp = requests.get(
                f"{self.base_url}{self.health_endpoint}",
                timeout=10
            )
            return resp.status_code == 200
        except Exception as e:
            logger.warning(f"OCR health check failed: {e}")
            return False
    
    def process_image(
        self,
        image_data: Union[bytes, str, Path],
        filename: str = "receipt.png",
        return_layout: bool = True
    ) -> OCRResult:
        """
        Process a receipt image through OCR.
        
        Args:
            image_data: Image as bytes, base64 string, or file path
            filename: Filename for the image
            return_layout: Whether to include layout info in response
            
        Returns:
            OCRResult with text, layout, and confidence
        """
        # Handle different input types
        if isinstance(image_data, (str, Path)):
            if Path(image_data).exists():
                with open(image_data, 'rb') as f:
                    image_bytes = f.read()
                filename = Path(image_data).name
            else:
                # Assume base64
                image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        # Prepare multipart form data
        files = {
            'file': (filename, image_bytes, self._get_mime_type(filename))
        }
        
        try:
            logger.info(f"📤 Sending image to OCR: {filename} ({len(image_bytes)} bytes)")
            
            resp = requests.post(
                f"{self.base_url}{self.process_endpoint}",
                files=files,
                timeout=self.timeout
            )
            resp.raise_for_status()
            
            # Log response details for debugging
            logger.debug(f"OCR Response Status: {resp.status_code}")
            logger.debug(f"OCR Response Headers: {dict(resp.headers)}")
            logger.debug(f"OCR Response Length: {len(resp.text)} bytes")
            
            # Save response to file for debugging
            debug_response_path = Path("/tmp/ocr_response_debug.yaml")
            debug_response_path.write_text(resp.text)
            logger.info(f"Saved raw OCR response to: {debug_response_path}")
            
            # Pre-process YAML to fix common issues (like unquoted asterisks)
            yaml_text = self._fix_malformed_yaml(resp.text)
            
            # Parse YAML response (the OCR service returns YAML format)
            result = None
            parse_error = None
            
            try:
                # Try safe_load first
                result = yaml.safe_load(yaml_text)
                logger.debug("Successfully parsed response as YAML with safe_load")
            except yaml.YAMLError as yaml_err:
                logger.warning(f"safe_load failed, trying with FullLoader: {yaml_err}")
                # Try with FullLoader which is more permissive
                try:
                    result = yaml.load(yaml_text, Loader=yaml.FullLoader)
                    logger.debug("Successfully parsed response as YAML with FullLoader")
                except yaml.YAMLError as yaml_err2:
                    logger.warning(f"FullLoader also failed: {yaml_err2}")
                    parse_error = yaml_err2
                    # Fallback to JSON if YAML fails
                    try:
                        result = resp.json()
                        logger.debug("Successfully parsed response as JSON")
                    except Exception as json_err:
                        logger.error(f"Failed to parse response as YAML or JSON")
                        logger.error(f"YAML error (safe_load): {yaml_err}")
                        logger.error(f"YAML error (FullLoader): {yaml_err2}")
                        logger.error(f"JSON error: {json_err}")
                        logger.error(f"Response status: {resp.status_code}")
                        logger.error(f"Full response saved to: {debug_response_path}")
                        raise RuntimeError(
                            f"OCR service returned invalid response format. "
                            f"Status: {resp.status_code}. "
                            f"Check {debug_response_path} for full response. "
                            f"YAML Error: {str(yaml_err)[:200]}"
                        )
            
            if result is None:
                raise RuntimeError(f"Failed to parse OCR response")
            
            # Extract text and layout from response
            ocr_result = self._parse_ocr_response(result)
            
            logger.info(f"✅ OCR complete: {ocr_result.word_count} words, {ocr_result.line_count} lines")
            return ocr_result
            
        except requests.Timeout:
            logger.error("OCR request timed out")
            raise TimeoutError("OCR service timeout")
        except requests.RequestException as e:
            logger.error(f"OCR request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response text: {e.response.text[:500]}")
            raise RuntimeError(f"OCR service error: {e}")
    
    def _fix_malformed_yaml(self, yaml_text: str) -> str:
        """
        Fix common YAML formatting issues from OCR service response.
        
        Issues fixed:
        - Unquoted strings starting with special YAML characters: *, @, `, |, >, &, !, %, #, etc.
        - Single special characters that need quoting
        
        Args:
            yaml_text: Raw YAML text from OCR service
            
        Returns:
            Fixed YAML text that can be safely parsed
        """
        lines = yaml_text.split('\n')
        fixed_lines = []
        
        # Special characters that cannot start unquoted strings in YAML
        yaml_special_chars = ['*', '@', '`', '|', '>', '&', '!', '%', '#', '{', '}', '[', ']', ',', '?', ':']
        
        for line in lines:
            # Pattern: '  - text: @' or '  - text: *****' or '  text: @something'
            # Match lines with text values that are unquoted and contain special chars
            match = re.match(r'^(\s*-?\s*text:\s*)([^"\'\n]+?)(\s*)$', line)
            if match:
                indent = match.group(1)
                value = match.group(2).strip()
                trailing = match.group(3)
                
                # Check if value needs quoting (starts with special char or is only special chars)
                needs_quoting = False
                
                if value:
                    # Check if starts with a special character
                    if value[0] in yaml_special_chars:
                        needs_quoting = True
                    # Check if it's only special characters (like ******* or @@@)
                    elif all(c in yaml_special_chars for c in value):
                        needs_quoting = True
                    # Check if it contains unescaped quotes or colons in problematic positions
                    elif ':' in value and not value.startswith('"'):
                        needs_quoting = True
                
                if needs_quoting:
                    # Escape any quotes in the value
                    escaped_value = value.replace('"', '\\"')
                    fixed_line = f'{indent}"{escaped_value}"{trailing}'
                    logger.debug(f"Fixed YAML line: '{line}' -> '{fixed_line}'")
                    fixed_lines.append(fixed_line)
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def _parse_ocr_response(self, response: Dict[str, Any]) -> OCRResult:
        """
        Parse OCR service response into structured format.
        
        Actual OCR service response format:
        {
            "text": {"raw": "...", "normalized": "...", "high_confidence_only": "..."},
            "confidence_stats": {"mean": 0.96, "min": 0.2, "max": 0.99},
            "word_count": {"total": 128, "high_confidence": 128},
            "words": [{"text": "...", "confidence": 0.99, "bbox": [...]}]
        }
        
        Output layout shape:
        {
            "lines": [{"text": "...", "bbox": [...], "word_count": int}],
            "words": [{"text": "...", "confidence": float, "bbox": [...]}],
            "tables": []  # Always present, empty if no tables
        }
        """
        # Handle string response
        if isinstance(response, str):
            return OCRResult(
                text=response,
                layout={"lines": [], "words": [], "tables": []},
                raw_response={"text": response},
                ocr_confidence=0.8,
                word_count=len(response.split()),
                line_count=len(response.split('\n'))
            )
        
        # Check if response has pages structure (use _extract_layout)
        if response.get('pages'):
            layout = self._extract_layout(response)
            text = self._reconstruct_text(response)
            confidence = self._calculate_confidence(response)
            word_count = len([w for p in response.get('pages', []) for w in p.get('words', [])])
            line_count = len(layout.get('lines', []))
            
            return OCRResult(
                text=text,
                layout=layout,
                raw_response=response,
                ocr_confidence=confidence,
                word_count=word_count,
                line_count=line_count
            )
        
        # Extract text from the word-based format
        text = ""
        text_obj = response.get('text', {})
        
        if isinstance(text_obj, dict):
            text = text_obj.get('raw', '') or text_obj.get('high_confidence_only', '')
        elif isinstance(text_obj, str):
            text = text_obj
        
        # Extract words array
        words_data = response.get('words', [])
        
        # If no text, reconstruct from words
        if not text and words_data:
            text = ' '.join(w.get('text', '') for w in words_data if isinstance(w, dict))
        
        # Build layout with words
        layout_words = [
            {
                "text": w.get('text', ''),
                "confidence": w.get('confidence', 0),
                "bbox": w.get('bbox', w.get('bbox_normalized', []))
            }
            for w in words_data if isinstance(w, dict)
        ]
        
        # Build lines by grouping words by Y-coordinate
        lines = self._group_words_into_lines(layout_words)
        
        layout = {
            "words": layout_words,
            "lines": lines,
            "tables": []  # Always include tables (empty if none)
        }
        
        # Calculate confidence from confidence_stats
        confidence_stats = response.get('confidence_stats', {})
        if isinstance(confidence_stats, dict):
            confidence = confidence_stats.get('mean', 0.8)
        else:
            confidence = 0.8
        
        # Get word count
        word_count_obj = response.get('word_count', {})
        if isinstance(word_count_obj, dict):
            word_count = word_count_obj.get('total', 0)
        else:
            word_count = len(text.split()) if text else 0
        
        line_count = len(lines) if lines else len(text.split('\n')) if text else 0
        
        return OCRResult(
            text=text,
            layout=layout,
            raw_response=response,
            ocr_confidence=confidence,
            word_count=word_count,
            line_count=line_count
        )
    
    def _group_words_into_lines(self, words: List[Dict[str, Any]], y_tolerance: float = 0.02) -> List[Dict[str, Any]]:
        """
        Group words into lines based on Y-coordinate proximity.
        
        Args:
            words: List of word dicts with bbox [x1, y1, x2, y2] or [[x1,y1], [x2,y2]]
            y_tolerance: Maximum Y difference to consider words on same line (normalized 0-1)
            
        Returns:
            List of line dicts with {text, bbox, word_count}
        """
        if not words:
            return []
        
        # Extract Y-center for each word
        word_with_y = []
        for word in words:
            bbox = word.get('bbox', [])
            y_center = self._get_bbox_y_center(bbox)
            if y_center is not None:
                word_with_y.append({
                    'word': word,
                    'y_center': y_center,
                    'x_min': self._get_bbox_x_min(bbox) or 0
                })
        
        if not word_with_y:
            # Fallback: no valid bboxes, create single line from all words
            return [{
                "text": ' '.join(w.get('text', '') for w in words),
                "bbox": [],
                "word_count": len(words)
            }]
        
        # Sort by Y-center, then by X for left-to-right ordering
        word_with_y.sort(key=lambda w: (w['y_center'], w['x_min']))
        
        # Group words into lines
        lines = []
        current_line_words = [word_with_y[0]]
        current_y = word_with_y[0]['y_center']
        
        for word_data in word_with_y[1:]:
            if abs(word_data['y_center'] - current_y) <= y_tolerance:
                # Same line
                current_line_words.append(word_data)
            else:
                # New line - finalize current line
                lines.append(self._finalize_line(current_line_words))
                current_line_words = [word_data]
                current_y = word_data['y_center']
        
        # Don't forget the last line
        if current_line_words:
            lines.append(self._finalize_line(current_line_words))
        
        return lines
    
    def _finalize_line(self, word_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Finalize a line from grouped word data."""
        # Sort by X position for proper text ordering
        word_data_list.sort(key=lambda w: w['x_min'])
        
        # Combine word texts
        text = ' '.join(w['word'].get('text', '') for w in word_data_list)
        
        # Combine bounding boxes
        all_bboxes = [w['word'].get('bbox', []) for w in word_data_list]
        combined_bbox = self._combine_bboxes(all_bboxes)
        
        return {
            "text": text,
            "bbox": combined_bbox,
            "word_count": len(word_data_list)
        }
    
    def _get_bbox_y_center(self, bbox: Any) -> Optional[float]:
        """Get Y-center from bbox (handles various formats)."""
        if not bbox:
            return None
        
        try:
            # Format: [x1, y1, x2, y2]
            if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
                return (bbox[1] + bbox[3]) / 2
            # Format: [[x1, y1], [x2, y2]]
            if len(bbox) == 2 and isinstance(bbox[0], (list, tuple)):
                return (bbox[0][1] + bbox[1][1]) / 2
            # Format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]] (quad)
            if len(bbox) == 4 and isinstance(bbox[0], (list, tuple)):
                y_vals = [p[1] for p in bbox]
                return sum(y_vals) / len(y_vals)
        except (IndexError, TypeError):
            pass
        return None
    
    def _get_bbox_x_min(self, bbox: Any) -> Optional[float]:
        """Get minimum X from bbox."""
        if not bbox:
            return None
        
        try:
            if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
                return bbox[0]
            if len(bbox) >= 2 and isinstance(bbox[0], (list, tuple)):
                return min(p[0] for p in bbox)
        except (IndexError, TypeError):
            pass
        return None
    
    def _combine_bboxes(self, bboxes: List[Any]) -> List[float]:
        """Combine multiple bboxes into a single enclosing bbox."""
        if not bboxes:
            return []
        
        x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
        
        for bbox in bboxes:
            if not bbox:
                continue
            try:
                if len(bbox) == 4 and isinstance(bbox[0], (int, float)):
                    x_mins.append(bbox[0])
                    y_mins.append(bbox[1])
                    x_maxs.append(bbox[2])
                    y_maxs.append(bbox[3])
                elif len(bbox) >= 2 and isinstance(bbox[0], (list, tuple)):
                    x_vals = [p[0] for p in bbox]
                    y_vals = [p[1] for p in bbox]
                    x_mins.append(min(x_vals))
                    y_mins.append(min(y_vals))
                    x_maxs.append(max(x_vals))
                    y_maxs.append(max(y_vals))
            except (IndexError, TypeError):
                continue
        
        if not x_mins:
            return []
        
        return [min(x_mins), min(y_mins), max(x_maxs), max(y_maxs)]
    
    def _reconstruct_text(self, response: Dict[str, Any]) -> str:
        """Reconstruct text from structured response."""
        pages = response.get('pages', [])
        if not pages:
            return response.get('text', '')
        
        text_parts = []
        for page in pages:
            lines = page.get('lines', [])
            for line in lines:
                words = line.get('words', [])
                line_text = ' '.join(w.get('value', '') for w in words)
                if line_text.strip():
                    text_parts.append(line_text)
        
        return '\n'.join(text_parts)
    
    def _extract_layout(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract layout information from OCR response."""
        layout = {
            "lines": [],
            "words": [],
            "tables": []
        }
        
        pages = response.get('pages', [])
        for page_idx, page in enumerate(pages):
            # Extract lines
            for line_idx, line in enumerate(page.get('lines', [])):
                geometry = line.get('geometry', [[0, 0], [1, 1]])
                words = line.get('words', [])
                line_text = ' '.join(w.get('value', '') for w in words)
                
                layout["lines"].append({
                    "text": line_text,
                    "page": page_idx,
                    "line_index": line_idx,
                    "bbox": geometry,
                    "word_count": len(words)
                })
            
            # Extract words
            for word in page.get('words', []):
                layout["words"].append({
                    "value": word.get('value', ''),
                    "confidence": word.get('confidence', 1.0),
                    "bbox": word.get('geometry', [[0, 0], [1, 1]])
                })
            
            # Extract tables (if present in OCR response)
            # Some OCR engines provide structured table data
            for table_idx, table in enumerate(page.get('tables', [])):
                table_data = {
                    "page": page_idx,
                    "table_index": table_idx,
                    "bbox": table.get('geometry', table.get('bbox', [])),
                    "rows": [],
                    "num_rows": 0,
                    "num_cols": 0
                }
                
                # Extract table rows and cells
                rows = table.get('rows', [])
                for row_idx, row in enumerate(rows):
                    row_data = {
                        "row_index": row_idx,
                        "cells": []
                    }
                    
                    cells = row.get('cells', [])
                    for cell_idx, cell in enumerate(cells):
                        cell_text = cell.get('text', cell.get('value', ''))
                        row_data["cells"].append({
                            "cell_index": cell_idx,
                            "text": cell_text,
                            "bbox": cell.get('geometry', cell.get('bbox', [])),
                            "confidence": cell.get('confidence', 1.0)
                        })
                    
                    table_data["rows"].append(row_data)
                
                table_data["num_rows"] = len(table_data["rows"])
                table_data["num_cols"] = max((len(r["cells"]) for r in table_data["rows"]), default=0)
                
                if table_data["num_rows"] > 0:
                    layout["tables"].append(table_data)
        
        return layout
    
    def _calculate_confidence(self, response: Dict[str, Any]) -> float:
        """Calculate overall OCR confidence from word confidences."""
        confidences = []
        
        pages = response.get('pages', [])
        for page in pages:
            for word in page.get('words', []):
                conf = word.get('confidence', 1.0)
                confidences.append(conf)
        
        if confidences:
            return sum(confidences) / len(confidences)
        
        return 0.8  # Default if no word confidences
    
    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type from filename."""
        ext = Path(filename).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.pdf': 'application/pdf',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.webp': 'image/webp'
        }
        return mime_types.get(ext, 'application/octet-stream')


# Singleton instance
_ocr_client_instance = None


def get_ocr_client(base_url: Optional[str] = None) -> ReceiptOCRClient:
    """Get or create singleton OCR client instance."""
    global _ocr_client_instance
    
    if _ocr_client_instance is None:
        _ocr_client_instance = ReceiptOCRClient(base_url=base_url)
    
    return _ocr_client_instance
