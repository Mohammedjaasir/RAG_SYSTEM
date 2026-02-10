#!/usr/bin/env python3
"""
Pipeline Orchestrator - End-to-End Receipt Processing v1.0.0

Complete pipeline from image upload to structured JSON output:
1. Image input (bytes, file path, or base64)
2. OCR processing via microservice
3. Text normalization (OCR error correction)
4. RAG retrieval (receipt intelligence)
5. Phi-3 extraction with RAG context
6. Confidence scoring and validation
7. Review flagging and structured output

Usage:
    from app.services.extraction.receipt.pipeline_orchestrator import get_pipeline_orchestrator
    
    orchestrator = get_pipeline_orchestrator()
    result = orchestrator.process_image(image_bytes, filename="receipt.jpg")
"""

import time
import logging
import base64
import json
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import dataclass, asdict

from config.config_loader import get_confidence_thresholds_cached, get_receipt_extraction_config_cached


logger = logging.getLogger(__name__)


@dataclass
@dataclass
class PipelineResult:
    """Structured result from the full extraction pipeline."""
    request_id: str
    success: bool
    
    # Extracted data
    items: List[Dict[str, Any]]
    item_count: int
    vendor_name: Optional[str]
    invoice_number: Optional[str]
    invoice_date: Optional[str]
    subtotal: Optional[float]
    tax_amount: Optional[float]
    total_amount: Optional[float]
    payment_method: Optional[str]
    
    # Confidence and review
    confidence: Dict[str, Any]
    needs_review: bool
    auto_approved: bool
    
    # Metadata
    ocr_text: str
    normalized_text: str
    country_detected: Optional[str]
    vendor_type: Optional[str]
    layout: Optional[Dict[str, Any]]
    
    # Processing info
    processing_time_ms: float
    pipeline_version: str
    rag_docs_used: List[str]
    
    # Error handling
    error: Optional[str]
    warnings: List[str]
    
    # Full detailed extraction from Phi-3 (includes VAT, address, phone, etc.)
    # This field has a default, so it must come last
    full_extraction: Optional[Dict[str, Any]] = None
    error: Optional[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PipelineOrchestrator:
    """
    End-to-end receipt processing pipeline.
    
    Orchestrates the full flow:
    Image → OCR → Normalize → RAG → Phi-3 → Score → Output
    """
    
    PIPELINE_VERSION = "1.0.0"
    
    def __init__(self):
        """Initialize pipeline orchestrator."""
        self.thresholds = get_confidence_thresholds_cached()
        self.extraction_config = get_receipt_extraction_config_cached()
        self._ocr_client = None
        self._text_normalizer = None
        self._vendor_classifier = None
        self._rag_retriever = None
        self._phi_extractor = None
        self._confidence_scorer = None
        self._pattern_learner = None
        self._initialized = False
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Lazy-load pipeline components."""
        try:
            # OCR Client
            from .rag.ocr_client import get_ocr_client
            self._ocr_client = get_ocr_client()
            logger.info("✅ OCR client loaded")
        except Exception as e:
            logger.warning(f"OCR client not available: {e}")
        
        try:
            # Text Normalizer
            from .support.text_normalizer import get_text_normalizer
            self._text_normalizer = get_text_normalizer()
            logger.info("✅ Text normalizer loaded")
        except Exception as e:
            logger.warning(f"Text normalizer not available: {e}")
        
        try:
            # Vendor Classifier
            from .rag.vendor_classifier import get_vendor_classifier
            self._vendor_classifier = get_vendor_classifier()
            logger.info("✅ Vendor classifier loaded")
        except Exception as e:
            logger.warning(f"Vendor classifier not available: {e}")
        
        try:
            # RAG Retriever
            from .rag import get_rag_retriever, initialize_knowledge_base
            # Initialize knowledge base on first load
            initialize_knowledge_base(force_reload=False)
            self._rag_retriever = get_rag_retriever()
            logger.info("✅ RAG retriever loaded")
        except Exception as e:
            logger.warning(f"RAG retriever not available: {e}")
        
        try:
            # Phi-3 Extractor
            from .extraction.phi_item_extractor import get_phi_item_extractor
            self._phi_extractor = get_phi_item_extractor()
            logger.info(f"✅ Phi-3 extractor loaded (model: {self._phi_extractor.model_loaded})")
        except Exception as e:
            logger.warning(f"Phi-3 extractor not available: {e}")
        
        try:
            # Confidence Scorer
            from .support.confidence_scorer import get_confidence_scorer
            self._confidence_scorer = get_confidence_scorer()
            logger.info("✅ Confidence scorer loaded")
        except Exception as e:
            logger.warning(f"Confidence scorer not available: {e}")
        
        try:
            # Pattern Learner
            from .rag.pattern_learner import get_pattern_learner
            self._pattern_learner = get_pattern_learner()
            logger.info("✅ Pattern learner loaded")
        except Exception as e:
            logger.warning(f"Pattern learner not available: {e}")
        
        self._initialized = True
        logger.info("PipelineOrchestrator initialized")
    
    def process_image(
        self,
        image_data: Union[bytes, str, Path],
        filename: str = "receipt.png",
        country_hint: Optional[str] = None,
        vendor_type_hint: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> PipelineResult:
        """
        Process a receipt image through the full extraction pipeline.
        
        Args:
            image_data: Image as bytes, base64 string, or file path
            filename: Filename for the image
            country_hint: Optional country code hint (IN, UK, US, etc.)
            vendor_type_hint: Optional vendor type hint
            request_id: Optional request tracking ID
            
        Returns:
            PipelineResult with extracted data and confidence
        """
        start_time = time.time()
        request_id = request_id or f"pipe_{int(time.time() * 1000)}"
        warnings = []
        
        # Initialize result structure
        items = []
        ocr_text = ""
        normalized_text = ""
        layout = None
        ocr_confidence = 0.8
        detected_country = country_hint
        rag_context = ""
        rag_docs_used = []
        
        try:
            # ===== STEP 1: OCR Processing =====
            logger.info(f"[{request_id}] Step 1: OCR processing...")
            
            if self._ocr_client:
                try:
                    ocr_result = self._ocr_client.process_image(image_data, filename)
                    ocr_text = ocr_result.text
                    layout = ocr_result.layout
                    ocr_confidence = ocr_result.ocr_confidence
                    logger.info(f"[{request_id}] OCR: {ocr_result.word_count} words, confidence: {ocr_confidence:.2f}")
                except Exception as e:
                    logger.error(f"[{request_id}] OCR failed: {e}")
                    warnings.append(f"OCR error: {str(e)}")
                    
                    # Try to extract base64 text if image processing fails
                    if isinstance(image_data, str) and not Path(image_data).exists():
                        # Might be base64 - can't process without OCR
                        return self._error_result(
                            request_id, start_time,
                            f"OCR service unavailable: {e}"
                        )
            else:
                # Fallback: treat input as text if OCR not available
                if isinstance(image_data, (str, Path)) and Path(image_data).exists():
                    with open(image_data, 'rb') as f:
                        content = f.read()
                    try:
                        ocr_text = content.decode('utf-8')
                        warnings.append("OCR not available, treating input as text")
                    except:
                        return self._error_result(
                            request_id, start_time,
                            "OCR client not initialized and input is binary"
                        )
                elif isinstance(image_data, str):
                    ocr_text = image_data
                    warnings.append("OCR not available, treating input as text")
                else:
                    return self._error_result(
                        request_id, start_time,
                        "OCR client not initialized"
                    )
            
            if not ocr_text or not ocr_text.strip():
                return self._error_result(
                    request_id, start_time,
                    "No text extracted from image"
                )
            
            # ===== STEP 2: Text Normalization =====
            logger.info(f"[{request_id}] Step 2: Text normalization...")
            
            if self._text_normalizer:
                norm_result = self._text_normalizer.normalize(ocr_text, country_hint)
                normalized_text = norm_result.normalized_text
                detected_country = norm_result.detected_country or country_hint
                logger.info(f"[{request_id}] Normalized: {len(norm_result.corrections_made)} corrections, country: {detected_country}")
            else:
                normalized_text = ocr_text
                warnings.append("Text normalizer not available")
            
            # ===== STEP 2.5: Vendor Classification =====
            logger.info(f"[{request_id}] Step 2.5: Vendor classification...")
            
            vendor_classification = None
            if self._vendor_classifier:
                try:
                    vendor_classification = self._vendor_classifier.classify(normalized_text, detected_country)
                    
                    # Update hints from classification
                    if vendor_classification.country and not detected_country:
                        detected_country = vendor_classification.country
                    
                    if vendor_classification.vendor_type and not vendor_type_hint:
                        vendor_type_hint = vendor_classification.vendor_type
                    
                    logger.info(
                        f"[{request_id}] Classified: {vendor_classification.vendor_name or 'Unknown'} "
                        f"({vendor_classification.vendor_type}, {vendor_classification.country}) "
                        f"confidence={vendor_classification.confidence:.2f}"
                    )
                except Exception as e:
                    logger.warning(f"[{request_id}] Vendor classification failed: {e}")
                    warnings.append(f"Vendor classification error: {str(e)}")
            else:
                warnings.append("Vendor classifier not available")
            
            # ===== STEP 3: RAG Retrieval =====
            logger.info(f"[{request_id}] Step 3: RAG knowledge retrieval...")
            
            pattern_match_confidence = 0.5  # Default
            if self._rag_retriever:
                try:
                    rag_context, pattern_match_confidence = self._rag_retriever.retrieve_with_confidence(
                        ocr_text=normalized_text,
                        country=detected_country,
                        vendor_type=vendor_type_hint
                    )
                    rag_docs_used = ["field_definitions", "tax_rules", "layout_patterns"]
                    logger.info(f"[{request_id}] RAG context: {len(rag_context)} chars, pattern confidence: {pattern_match_confidence:.3f}")
                except Exception as e:
                    logger.warning(f"[{request_id}] RAG retrieval failed: {e}")
                    warnings.append(f"RAG retrieval error: {str(e)}")
                    # Fallback to simple retrieval
                    try:
                        rag_context = self._rag_retriever.retrieve_for_extraction(
                            ocr_text=normalized_text,
                            country=detected_country,
                            vendor_type=vendor_type_hint
                        )
                    except:
                        pass
            else:
                warnings.append("RAG retriever not available")
            
            # ===== STEP 4: Phi-3 Full Receipt Extraction =====
            logger.info(f"[{request_id}] Step 4: Phi-3 full receipt extraction...")
            
            full_extraction = None
            if self._phi_extractor and self._phi_extractor.model_loaded:
                try:
                    # Build layout JSON for prompt
                    layout_json = json.dumps(layout) if layout else None
                    
                    # Call full receipt extraction with layout and RAG context
                    full_extraction = self._phi_extractor.extract_full_receipt(
                        normalized_text=normalized_text,
                        layout_json=layout_json,
                        rag_context=rag_context
                    )
                    
                    # Debug: Log what we got
                    logger.info(f"[{request_id}] Full extraction result keys: {list(full_extraction.keys()) if full_extraction else 'None'}")
                    if full_extraction:
                        logger.info(f"[{request_id}] Has supplier_name: {full_extraction.get('supplier_name')}")
                        logger.info(f"[{request_id}] Has item_list: {len(full_extraction.get('item_list', []))} items")
                    
                    # Extract items from full result
                    items = full_extraction.get('item_list', [])
                    logger.info(f"[{request_id}] Full extraction: {len(items)} items")
                except Exception as e:
                    logger.error(f"[{request_id}] Full extraction failed: {e}")
                    warnings.append(f"Extraction error: {str(e)}")
                    # Fallback to items-only extraction
                    try:
                        extraction_text = f"[REFERENCE_KNOWLEDGE]\n{rag_context}\n\n[RECEIPT_TEXT]\n{normalized_text}" if rag_context else normalized_text
                        items = self._phi_extractor.extract_items(extraction_text)
                    except:
                        pass
            else:
                warnings.append("Phi-3 model not loaded")
            
            # ===== STEP 5: Confidence Scoring =====
            logger.info(f"[{request_id}] Step 5: Confidence scoring...")
            
            confidence_result = None
            per_field_confidences = None
            
            # Extract per-field confidences from full extraction
            if full_extraction and isinstance(full_extraction, dict):
                per_field_confidences = {}
                for field in ['supplier_name', 'receipt_number', 'receipt_date']:
                    field_data = full_extraction.get(field, {})
                    if isinstance(field_data, dict) and 'confidence' in field_data:
                        per_field_confidences[field] = field_data['confidence']
                
                # Add totals confidence
                totals = full_extraction.get('totals', {})
                for total_key in ['subtotal', 'final_total']:
                    if isinstance(totals.get(total_key), dict):
                        per_field_confidences[f'totals.{total_key}'] = totals[total_key].get('confidence', 0.0)
            
            if self._confidence_scorer:
                confidence_result = self._confidence_scorer.calculate_from_items(
                    items=items,
                    ocr_confidence=ocr_confidence,
                    pattern_match_confidence=pattern_match_confidence,
                    per_field_confidences=per_field_confidences
                )
                
                if confidence_result.validation_errors:
                    warnings.extend(confidence_result.validation_errors)
            else:
                # Fallback confidence calculation
                avg_item_conf = sum(
                    item.get('extraction_confidence', 0.5) for item in items
                ) / len(items) if items else 0.5
                
                confidence_result = type('ConfidenceResult', (), {
                    'overall_confidence': avg_item_conf,
                    'needs_review': avg_item_conf < 0.7,
                    'auto_approved': avg_item_conf >= 0.85,
                    'to_dict': lambda: {'overall_confidence': avg_item_conf}
                })()
            
            # ===== STEP 5.5: Config-Based Fallback for Low-Confidence Fields =====
            logger.info(f"[{request_id}] Step 5.5: Config-based fallback for low-confidence fields...")
            
            if self._should_fallback_to_config(per_field_confidences or {}):
                logger.info(f"[{request_id}] Fields detected below threshold, applying config-based fallback")
                full_extraction = self._apply_config_fallback_to_full_extraction(
                    full_extraction=full_extraction,
                    normalized_text=normalized_text,
                    request_id=request_id,
                    warnings=warnings
                )
            else:
                logger.debug(f"[{request_id}] All fields above confidence threshold, no fallback needed")
            
            # ===== STEP 6: Pattern Learning (Auto-learn from per-field confidence) =====
            if self._pattern_learner and full_extraction:
                try:
                    overall_conf = confidence_result.overall_confidence if hasattr(confidence_result, 'overall_confidence') else 0.0
                    validation_ok = confidence_result.validation_passed if hasattr(confidence_result, 'validation_passed') else True
                    
                    # Always attempt learning - per-field logic will filter by individual confidence
                    if self._pattern_learner.should_learn(overall_conf, validation_ok):
                        logger.info(f"[{request_id}] Attempting per-field pattern learning (overall confidence={overall_conf:.2%})...")
                        
                        learned_patterns = self._pattern_learner.learn_from_success(
                            request_id=request_id,
                            full_extraction=full_extraction,
                            ocr_text=ocr_text,
                            normalized_text=normalized_text,
                            overall_confidence=overall_conf,
                            country_detected=detected_country,
                            vendor_classification=vendor_classification.to_dict() if vendor_classification else None
                        )
                        
                        if learned_patterns:
                            # Count by learning mode
                            auto_approved = sum(1 for p in learned_patterns if not getattr(p, 'needs_review', False))
                            needs_review = len(learned_patterns) - auto_approved
                            
                            if auto_approved > 0 and needs_review > 0:
                                logger.info(f"[{request_id}] ✅ Learned {len(learned_patterns)} patterns: {auto_approved} auto-approved, {needs_review} needs review")
                            elif auto_approved > 0:
                                logger.info(f"[{request_id}] ✅ Learned {auto_approved} patterns (auto-approved)")
                            elif needs_review > 0:
                                logger.info(f"[{request_id}] 📝 Learned {needs_review} patterns (needs review)")
                        else:
                            logger.debug(f"[{request_id}] No patterns learned (all fields below 70% confidence threshold)")
                    else:
                        logger.debug(f"[{request_id}] Skipping pattern learning (rate limit or validation failed)")
                        
                except Exception as e:
                    logger.warning(f"[{request_id}] Pattern learning failed: {e}")
                    warnings.append(f"Pattern learning error: {str(e)}")
            
            # ===== STEP 7: Build Result =====
            processing_time = (time.time() - start_time) * 1000
            
            # Extract header fields from full extraction
            vendor_name = None
            invoice_number = None
            invoice_date = None
            subtotal = None
            tax_amount = None
            total_amount = None
            payment_method = None
            
            if full_extraction:
                # Supplier/vendor name
                sn = full_extraction.get('supplier_name', {})
                vendor_name = sn.get('value') if isinstance(sn, dict) else sn
                
                # Receipt/invoice number
                rn = full_extraction.get('receipt_number', {})
                invoice_number = rn.get('value') if isinstance(rn, dict) else rn
                
                # Receipt date
                rd = full_extraction.get('receipt_date', {})
                invoice_date = rd.get('date') if isinstance(rd, dict) else rd
                
                # Totals
                totals = full_extraction.get('totals', {})
                if isinstance(totals, dict):
                    if isinstance(totals.get('subtotal'), dict):
                        subtotal = totals['subtotal'].get('amount')
                    if isinstance(totals.get('final_total'), dict):
                        total_amount = totals['final_total'].get('amount')
                
                # VAT/Tax
                vat_info = full_extraction.get('vat_information', {})
                if isinstance(vat_info, dict):
                    vat_entries = vat_info.get('vat_data_entries', [])
                    if vat_entries:
                        tax_amount = sum(e.get('vat_amount', 0) for e in vat_entries if isinstance(e, dict))
                
                # Payment method
                payments = full_extraction.get('payment_methods', [])
                if payments and isinstance(payments, list) and isinstance(payments[0], dict):
                    payment_method = payments[0].get('method')
            
            return PipelineResult(
                request_id=request_id,
                success=True,
                items=items,
                item_count=len(items),
                vendor_name=vendor_name,
                invoice_number=invoice_number,
                invoice_date=invoice_date,
                subtotal=subtotal,
                tax_amount=tax_amount,
                total_amount=total_amount,
                payment_method=payment_method,
                full_extraction=full_extraction,  # Include full Phi-3 extraction
                confidence=confidence_result.to_dict() if hasattr(confidence_result, 'to_dict') else {},
                needs_review=confidence_result.needs_review,
                auto_approved=confidence_result.auto_approved,
                ocr_text=ocr_text[:1000] + "..." if len(ocr_text) > 1000 else ocr_text,
                normalized_text=normalized_text[:1000] + "..." if len(normalized_text) > 1000 else normalized_text,
                country_detected=detected_country,
                vendor_type=vendor_type_hint,
                layout=layout,
                processing_time_ms=round(processing_time, 2),
                pipeline_version=self.PIPELINE_VERSION,
                rag_docs_used=rag_docs_used,
                error=None,
                warnings=warnings
            )
            
        except Exception as e:
            import traceback
            logger.error(f"[{request_id}] Pipeline error: {e}")
            logger.error(f"[{request_id}] Traceback: {traceback.format_exc()}")
            return self._error_result(request_id, start_time, str(e), warnings)
    
    def process_text(
        self,
        ocr_text: str,
        country_hint: Optional[str] = None,
        vendor_type_hint: Optional[str] = None,
        ocr_confidence: float = 0.8,
        request_id: Optional[str] = None
    ) -> PipelineResult:
        """
        Process pre-extracted OCR text through the pipeline.
        
        Skips OCR step, useful when text is already available.
        """
        start_time = time.time()
        request_id = request_id or f"text_{int(time.time() * 1000)}"
        warnings = []
        
        # Skip to normalization
        normalized_text = ocr_text
        detected_country = country_hint
        rag_context = ""
        rag_docs_used = []
        items = []
        
        try:
            # Normalize
            if self._text_normalizer:
                norm_result = self._text_normalizer.normalize(ocr_text, country_hint)
                normalized_text = norm_result.normalized_text
                detected_country = norm_result.detected_country or country_hint
            
            # RAG
            if self._rag_retriever:
                rag_context = self._rag_retriever.retrieve_for_extraction(
                    ocr_text=normalized_text,
                    country=detected_country,
                    vendor_type=vendor_type_hint
                )
                rag_docs_used = ["field_definitions", "tax_rules", "layout_patterns"]
            
            # Extract
            if self._phi_extractor and self._phi_extractor.model_loaded:
                extraction_text = normalized_text
                if rag_context:
                    extraction_text = f"[REFERENCE_KNOWLEDGE]\n{rag_context}\n\n[RECEIPT_TEXT]\n{normalized_text}"
                items = self._phi_extractor.extract_items(extraction_text)
            
            # Score
            confidence_result = None
            if self._confidence_scorer:
                confidence_result = self._confidence_scorer.calculate_from_items(
                    items=items,
                    ocr_confidence=ocr_confidence,
                    rag_context_available=bool(rag_context)
                )
            else:
                avg_conf = sum(
                    item.get('extraction_confidence', 0.5) for item in items
                ) / len(items) if items else 0.5
                confidence_result = type('ConfidenceResult', (), {
                    'overall_confidence': avg_conf,
                    'needs_review': avg_conf < 0.7,
                    'auto_approved': avg_conf >= 0.85,
                    'to_dict': lambda: {'overall_confidence': avg_conf}
                })()
            
            processing_time = (time.time() - start_time) * 1000
            
            return PipelineResult(
                request_id=request_id,
                success=True,
                items=items,
                item_count=len(items),
                vendor_name=None,
                invoice_number=None,
                invoice_date=None,
                subtotal=None,
                tax_amount=None,
                total_amount=None,
                payment_method=None,
                confidence=confidence_result.to_dict() if hasattr(confidence_result, 'to_dict') else {},
                needs_review=confidence_result.needs_review,
                auto_approved=confidence_result.auto_approved,
                ocr_text=ocr_text[:1000] + "..." if len(ocr_text) > 1000 else ocr_text,
                normalized_text=normalized_text[:1000] + "..." if len(normalized_text) > 1000 else normalized_text,
                country_detected=detected_country,
                vendor_type=vendor_type_hint,
                layout=None,
                processing_time_ms=round(processing_time, 2),
                pipeline_version=self.PIPELINE_VERSION,
                rag_docs_used=rag_docs_used,
                error=None,
                warnings=warnings
            )
            
        except Exception as e:
            return self._error_result(request_id, start_time, str(e), warnings)
    
    def _should_fallback_to_config(self, per_field_confidences: Dict[str, float]) -> bool:
        """Check if any field is below threshold."""
        thresholds = self.thresholds['field_thresholds']
        default_threshold = self.thresholds['global']['default_threshold']
        
        for field_name, confidence in per_field_confidences.items():
            field_threshold = thresholds.get(field_name, {}).get('threshold', default_threshold)
            if confidence < field_threshold:
                return True
        
        return False
    
    def _get_field_threshold(self, field_name: str) -> float:
        """Get threshold for a specific field."""
        thresholds = self.thresholds['field_thresholds']
        default_threshold = self.thresholds['global']['default_threshold']
        
        field_config = thresholds.get(field_name, {})
        if isinstance(field_config, dict):
            return field_config.get('threshold', default_threshold)
        return default_threshold
    
    def _extract_date_fallback(self, normalized_text: str, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Fallback date extraction using config-based patterns.
        
        Args:
            normalized_text: Normalized receipt text
            request_id: Request tracking ID
            
        Returns:
            Date extraction result or None if extraction fails
        """
        try:
            from app.services.extraction.receipt.extraction.config_based_extraction.date_extractor import extract_date
            
            # Get date config from extraction config
            date_config = self.extraction_config.get('date_config', {})
            
            if not date_config:
                logger.warning(f"[{request_id}] Date config not available for fallback extraction")
                return None
            
            # Call config-based date extractor
            extracted_date = extract_date(normalized_text, date_config)
            
            if extracted_date:
                logger.info(f"[{request_id}] ✅ Config-based date extraction succeeded: {extracted_date}")
                
                # Return in same format as RAG extraction
                return {
                    "date": extracted_date,
                    "raw_text": extracted_date,
                    "confidence": 0.75,  # Config-based extraction confidence
                    "extraction_method": "config"
                }
            else:
                logger.debug(f"[{request_id}] Config-based date extraction returned no result")
                return None
                
        except Exception as e:
            logger.error(f"[{request_id}] Config-based date extraction failed: {e}", exc_info=True)
            return None
    
    def _extract_supplier_name_fallback(self, normalized_text: str, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Fallback supplier name extraction using config-based patterns.
        
        Args:
            normalized_text: Normalized receipt text
            request_id: Request tracking ID
            
        Returns:
            Supplier name extraction result or None if extraction fails
        """
        try:
            from app.services.extraction.receipt.extraction.config_based_extraction.supplier_name_extractor import extract_supplier
            
            # Get supplier config from extraction config
            supplier_config = self.extraction_config.get('supplier_config', {})
            
            if not supplier_config:
                logger.warning(f"[{request_id}] Supplier config not available for fallback extraction")
                return None
            
            # Split text into lines for line-by-line processing
            lines = normalized_text.split('\n')
            
            # Call config-based supplier extractor
            extracted_supplier = extract_supplier(lines, supplier_config)
            
            if extracted_supplier:
                logger.info(f"[{request_id}] ✅ Config-based supplier extraction succeeded: {extracted_supplier}")
                
                # Return in same format as RAG extraction
                return {
                    "value": extracted_supplier,
                    "raw_text": extracted_supplier,
                    "confidence": 0.75,  # Config-based extraction confidence
                    "extraction_method": "config"
                }
            else:
                logger.debug(f"[{request_id}] Config-based supplier extraction returned no result")
                return None
                
        except Exception as e:
            logger.error(f"[{request_id}] Config-based supplier extraction failed: {e}", exc_info=True)
            return None
    
    def _extract_total_amount_fallback(self, normalized_text: str, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Fallback total amount extraction using config-based patterns.
        
        Args:
            normalized_text: Normalized receipt text
            request_id: Request tracking ID
            
        Returns:
            Total amount extraction result or None if extraction fails
        """
        try:
            from app.services.extraction.receipt.extraction.config_based_extraction.total_amount_extractor import extract_total
            
            # Get total amount config from extraction config
            total_config = self.extraction_config.get('total_amount_config', {})
            
            if not total_config:
                logger.warning(f"[{request_id}] Total amount config not available for fallback extraction")
                return None
            
            # Split text into lines for line-by-line processing
            lines = normalized_text.split('\n')
            
            # Call config-based total amount extractor
            extracted_total = extract_total(lines, total_config)
            
            if extracted_total:
                logger.info(f"[{request_id}] ✅ Config-based total amount extraction succeeded: {extracted_total}")
                
                # Return in same format as RAG extraction
                return {
                    "value": float(extracted_total),
                    "raw_text": extracted_total,
                    "confidence": 0.75,  # Config-based extraction confidence
                    "extraction_method": "config"
                }
            else:
                logger.debug(f"[{request_id}] Config-based total amount extraction returned no result")
                return None
                
        except Exception as e:
            logger.error(f"[{request_id}] Config-based total amount extraction failed: {e}", exc_info=True)
            return None
    
    def _extract_net_amount_fallback(self, normalized_text: str, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Fallback net amount extraction using config-based patterns.
        
        Args:
            normalized_text: Normalized receipt text
            request_id: Request tracking ID
            
        Returns:
            Net amount extraction result or None if extraction fails
        """
        try:
            from app.services.extraction.receipt.extraction.config_based_extraction.net_amount_extractor import extract_net
            
            # Get net amount config from extraction config
            net_config = self.extraction_config.get('net_amount_config', {})
            
            if not net_config:
                logger.warning(f"[{request_id}] Net amount config not available for fallback extraction")
                return None
            
            # Split text into lines for line-by-line processing
            lines = normalized_text.split('\n')
            
            # Get strategies from net config
            strategies = net_config.get('strategies', []) if isinstance(net_config, dict) else []
            
            # Call config-based net amount extractor
            extracted_net = extract_net(lines, strategies)
            
            if extracted_net:
                logger.info(f"[{request_id}] ✅ Config-based net amount extraction succeeded: {extracted_net}")
                
                # Return in same format as RAG extraction
                return {
                    "value": float(extracted_net),
                    "raw_text": extracted_net,
                    "confidence": 0.75,  # Config-based extraction confidence
                    "extraction_method": "config"
                }
            else:
                logger.debug(f"[{request_id}] Config-based net amount extraction returned no result")
                return None
                
        except Exception as e:
            logger.error(f"[{request_id}] Config-based net amount extraction failed: {e}", exc_info=True)
            return None
    
    def _apply_config_fallback_to_full_extraction(
        self,
        full_extraction: Dict[str, Any],
        normalized_text: str,
        request_id: str,
        warnings: List[str]
    ) -> Dict[str, Any]:
        """
        Apply config-based fallback extraction for fields below confidence threshold.
        
        Supports fallback extraction for:
        - receipt_date: Using date_extractor
        - supplier_name: Using supplier_name_extractor
        - totals.total_amount: Using total_amount_extractor
        - totals.net_amount: Using net_amount_extractor
        
        Args:
            full_extraction: RAG-based extraction result
            normalized_text: Normalized receipt text
            request_id: Request tracking ID
            warnings: Warnings list to append to
            
        Returns:
            Modified full_extraction with fallback results
        """
        if not full_extraction:
            return full_extraction
        
        # ===== FIELD 1: receipt_date =====
        receipt_date_field = full_extraction.get('receipt_date', {})
        receipt_date_confidence = 0.0
        
        if isinstance(receipt_date_field, dict):
            receipt_date_confidence = receipt_date_field.get('confidence', 0.0)
        
        # Get threshold for receipt_date
        date_threshold = self._get_field_threshold('receipt_date')
        
        logger.info(f"[{request_id}] Receipt date confidence: {receipt_date_confidence:.3f}, threshold: {date_threshold:.3f}")
        
        # If below threshold, try config-based extraction
        if receipt_date_confidence < date_threshold:
            logger.info(f"[{request_id}] Receipt date confidence below threshold, attempting config-based fallback")
            
            config_date_result = self._extract_date_fallback(normalized_text, request_id)
            
            if config_date_result:
                config_date_confidence = config_date_result.get('confidence', 0.0)
                
                # Use config result if it has better confidence
                if config_date_confidence > receipt_date_confidence:
                    logger.info(
                        f"[{request_id}] Using config-based date (confidence: {config_date_confidence:.3f}) "
                        f"over RAG result (confidence: {receipt_date_confidence:.3f})"
                    )
                    full_extraction['receipt_date'] = config_date_result
                    warnings.append(f"receipt_date: Used config-based extraction (RAG confidence: {receipt_date_confidence:.2f} → config confidence: {config_date_confidence:.2f})")
                else:
                    logger.info(
                        f"[{request_id}] Keeping RAG date result (confidence: {receipt_date_confidence:.3f}) "
                        f"over config result (confidence: {config_date_confidence:.3f})"
                    )
            else:
                logger.warning(f"[{request_id}] Config-based date extraction failed, keeping RAG result")
        
        # ===== FIELD 2: supplier_name =====
        supplier_name_field = full_extraction.get('supplier_name', {})
        supplier_name_confidence = 0.0
        
        if isinstance(supplier_name_field, dict):
            supplier_name_confidence = supplier_name_field.get('confidence', 0.0)
        
        # Get threshold for supplier_name
        supplier_threshold = self._get_field_threshold('supplier_name')
        
        logger.info(f"[{request_id}] Supplier name confidence: {supplier_name_confidence:.3f}, threshold: {supplier_threshold:.3f}")
        
        # If below threshold, try config-based extraction
        if supplier_name_confidence < supplier_threshold:
            logger.info(f"[{request_id}] Supplier name confidence below threshold, attempting config-based fallback")
            
            config_supplier_result = self._extract_supplier_name_fallback(normalized_text, request_id)
            
            if config_supplier_result:
                config_supplier_confidence = config_supplier_result.get('confidence', 0.0)
                
                # Use config result if it has better confidence
                if config_supplier_confidence > supplier_name_confidence:
                    logger.info(
                        f"[{request_id}] Using config-based supplier (confidence: {config_supplier_confidence:.3f}) "
                        f"over RAG result (confidence: {supplier_name_confidence:.3f})"
                    )
                    full_extraction['supplier_name'] = config_supplier_result
                    warnings.append(f"supplier_name: Used config-based extraction (RAG confidence: {supplier_name_confidence:.2f} → config confidence: {config_supplier_confidence:.2f})")
                else:
                    logger.info(
                        f"[{request_id}] Keeping RAG supplier result (confidence: {supplier_name_confidence:.3f}) "
                        f"over config result (confidence: {config_supplier_confidence:.3f})"
                    )
            else:
                logger.warning(f"[{request_id}] Config-based supplier extraction failed, keeping RAG result")
        
        # ===== FIELD 3: total_amount (in totals object) =====
        totals_field = full_extraction.get('totals', {})
        total_amount_confidence = 0.0
        
        if isinstance(totals_field, dict):
            total_amount_data = totals_field.get('total_amount', {})
            if isinstance(total_amount_data, dict):
                total_amount_confidence = total_amount_data.get('confidence', 0.0)
        
        # Get threshold for total_amount
        total_threshold = self._get_field_threshold('total_amount')
        
        logger.info(f"[{request_id}] Total amount confidence: {total_amount_confidence:.3f}, threshold: {total_threshold:.3f}")
        
        # If below threshold, try config-based extraction
        if total_amount_confidence < total_threshold:
            logger.info(f"[{request_id}] Total amount confidence below threshold, attempting config-based fallback")
            
            config_total_result = self._extract_total_amount_fallback(normalized_text, request_id)
            
            if config_total_result:
                config_total_confidence = config_total_result.get('confidence', 0.0)
                
                # Use config result if it has better confidence
                if config_total_confidence > total_amount_confidence:
                    logger.info(
                        f"[{request_id}] Using config-based total (confidence: {config_total_confidence:.3f}) "
                        f"over RAG result (confidence: {total_amount_confidence:.3f})"
                    )
                    if not isinstance(totals_field, dict):
                        totals_field = {}
                    totals_field['total_amount'] = config_total_result
                    full_extraction['totals'] = totals_field
                    warnings.append(f"total_amount: Used config-based extraction (RAG confidence: {total_amount_confidence:.2f} → config confidence: {config_total_confidence:.2f})")
                else:
                    logger.info(
                        f"[{request_id}] Keeping RAG total result (confidence: {total_amount_confidence:.3f}) "
                        f"over config result (confidence: {config_total_confidence:.3f})"
                    )
            else:
                logger.warning(f"[{request_id}] Config-based total amount extraction failed, keeping RAG result")
        
        # ===== FIELD 4: net_amount (in totals object) =====
        net_amount_confidence = 0.0
        
        if isinstance(totals_field, dict):
            net_amount_data = totals_field.get('net_amount', {})
            if isinstance(net_amount_data, dict):
                net_amount_confidence = net_amount_data.get('confidence', 0.0)
        
        # Get threshold for net_amount
        net_threshold = self._get_field_threshold('net_amount')
        
        logger.info(f"[{request_id}] Net amount confidence: {net_amount_confidence:.3f}, threshold: {net_threshold:.3f}")
        
        # If below threshold, try config-based extraction
        if net_amount_confidence < net_threshold:
            logger.info(f"[{request_id}] Net amount confidence below threshold, attempting config-based fallback")
            
            config_net_result = self._extract_net_amount_fallback(normalized_text, request_id)
            
            if config_net_result:
                config_net_confidence = config_net_result.get('confidence', 0.0)
                
                # Use config result if it has better confidence
                if config_net_confidence > net_amount_confidence:
                    logger.info(
                        f"[{request_id}] Using config-based net (confidence: {config_net_confidence:.3f}) "
                        f"over RAG result (confidence: {net_amount_confidence:.3f})"
                    )
                    if not isinstance(totals_field, dict):
                        totals_field = {}
                    totals_field['net_amount'] = config_net_result
                    full_extraction['totals'] = totals_field
                    warnings.append(f"net_amount: Used config-based extraction (RAG confidence: {net_amount_confidence:.2f} → config confidence: {config_net_confidence:.2f})")
                else:
                    logger.info(
                        f"[{request_id}] Keeping RAG net result (confidence: {net_amount_confidence:.3f}) "
                        f"over config result (confidence: {config_net_confidence:.3f})"
                    )
            else:
                logger.warning(f"[{request_id}] Config-based net amount extraction failed, keeping RAG result")
        
        return full_extraction

    def _error_result(
        self,
        request_id: str,
        start_time: float,
        error: str,
        warnings: Optional[List[str]] = None
    ) -> PipelineResult:
        """Create error result."""
        processing_time = (time.time() - start_time) * 1000
        
        return PipelineResult(
            request_id=request_id,
            success=False,
            items=[],
            item_count=0,
            vendor_name=None,
            invoice_number=None,
            invoice_date=None,
            subtotal=None,
            tax_amount=None,
            total_amount=None,
            payment_method=None,
            confidence={},
            needs_review=True,
            auto_approved=False,
            ocr_text="",
            normalized_text="",
            country_detected=None,
            vendor_type=None,
            layout=None,
            processing_time_ms=round(processing_time, 2),
            pipeline_version=self.PIPELINE_VERSION,
            rag_docs_used=[],
            error=error,
            warnings=warnings or []
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get pipeline component status."""
        return {
            "initialized": self._initialized,
            "ocr_client_available": self._ocr_client is not None,
            "ocr_healthy": self._ocr_client.health_check() if self._ocr_client else False,
            "text_normalizer_available": self._text_normalizer is not None,
            "rag_retriever_available": self._rag_retriever is not None,
            "phi_extractor_available": self._phi_extractor is not None,
            "phi_model_loaded": self._phi_extractor.model_loaded if self._phi_extractor else False,
            "confidence_scorer_available": self._confidence_scorer is not None,
            "pipeline_version": self.PIPELINE_VERSION
        }
    
    def refresh_rag_knowledge(self, force: bool = False) -> Dict[str, Any]:
        """
        Perform hard refresh of RAG knowledge base.
        
        This re-embeds all documents in the knowledge base. Use when:
        - Embedding model is upgraded
        - Chunking strategy changes
        - Major schema updates needed
        
        Args:
            force: Skip confirmation (use with caution in production)
            
        Returns:
            Refresh result with statistics
        """
        try:
            from .rag import get_rag_refresh_manager
            
            refresh_manager = get_rag_refresh_manager()
            result = refresh_manager.hard_refresh(force=force)
            
            logger.info(f"✅ RAG hard refresh completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"❌ RAG hard refresh failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG knowledge base statistics."""
        try:
            from .rag import get_vector_store, get_rag_refresh_manager
            
            vector_store = get_vector_store()
            refresh_manager = get_rag_refresh_manager()
            
            return {
                "vector_store": vector_store.get_stats(),
                "refresh_history": refresh_manager.get_refresh_history(),
                "refresh_manager_status": refresh_manager.get_status()
            }
        except Exception as e:
            logger.error(f"❌ Failed to get RAG stats: {e}")
            return {"error": str(e)}


# Singleton instance
_pipeline_orchestrator_instance = None


def get_pipeline_orchestrator() -> PipelineOrchestrator:
    """Get or create singleton PipelineOrchestrator instance."""
    global _pipeline_orchestrator_instance
    
    if _pipeline_orchestrator_instance is None:
        _pipeline_orchestrator_instance = PipelineOrchestrator()
    
    return _pipeline_orchestrator_instance
