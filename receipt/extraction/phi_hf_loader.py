#!/usr/bin/env python3
"""
Phi-3 Hugging Face Loader - Load and Run Phi-3 from HF v1.0.0

Provides Hugging Face backend for Phi-3 model to avoid manual GGUF downloads.
Automatically downloads model from HF on first run and caches locally.

Features:
- Auto-download from Hugging Face Hub
- Local caching for fast subsequent loads
- Optional 4-bit quantization (bitsandbytes)
- Fallback to llama-cpp if HF backend disabled

Environment Variables:
- PHI_BACKEND: Set to "hf" to use HF backend (default: "llama_cpp")
- PHI_HF_MODEL: HF model ID (default: "microsoft/Phi-3-mini-4k-instruct")
- PHI_CACHE_DIR: Local cache directory for model

Usage:
    from app.services.extraction.receipt.extraction.phi_hf_loader import get_phi3_hf_loader
    
    loader = get_phi3_hf_loader()
    response = loader.generate(prompt, max_tokens=1500)
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_CACHE_DIR = Path(__file__).parent.parent.parent / "models" / "receipts" / "phi3_hf"


class Phi3HFLoader:
    """
    Hugging Face Phi-3 model loader and inference engine.
    
    Downloads and caches Phi-3 from Hugging Face, provides generate()
    method compatible with existing extraction pipeline.
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_quantization: bool = False,
        device: str = "auto"
    ):
        """
        Initialize Phi-3 HF Loader.
        
        Args:
            model_id: HF model identifier (e.g., "microsoft/Phi-3-mini-4k-instruct")
            cache_dir: Local directory to cache the model
            use_quantization: Enable 4-bit quantization (requires bitsandbytes)
            device: Device to load model on ("auto", "cpu", "cuda")
        """
        self.model_id = model_id or os.getenv("PHI_HF_MODEL", DEFAULT_MODEL_ID)
        self.cache_dir = Path(cache_dir or os.getenv("PHI_CACHE_DIR", str(DEFAULT_CACHE_DIR)))
        self.use_quantization = use_quantization
        self.device = device
        
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        logger.info(f"Phi3HFLoader initialized (model: {self.model_id})")
    
    def load_model(self) -> bool:
        """
        Load or download Phi-3 model from Hugging Face.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if self.model_loaded:
            return True
        
        try:
            # Import transformers (lazy import)
            from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
            import torch
            
            logger.info(f"Loading Phi-3 from HF: {self.model_id}")
            
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine device
            if self.device == "auto":
                device_map = "auto" if torch.cuda.is_available() else "cpu"
            else:
                device_map = self.device
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            
            # Load and patch config first (fixes rope_scaling bug)
            logger.info("Loading model config...")
            config = AutoConfig.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True
            )
            
            # Fix rope_scaling config if malformed
            if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
                if not isinstance(config.rope_scaling, dict) or 'type' not in config.rope_scaling:
                    logger.warning("Fixing malformed rope_scaling config...")
                    config.rope_scaling = None
            
            # Force eager attention to avoid flash-attention issues
            config._attn_implementation = "eager"
            
            # Model loading kwargs
            model_kwargs = {
                "config": config,
                "cache_dir": str(self.cache_dir),
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "device_map": device_map,
            }
            
            # Add quantization config if enabled
            if self.use_quantization and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                    logger.info("4-bit quantization enabled")
                except ImportError:
                    logger.warning("bitsandbytes not available, loading without quantization")
            
            # Load model with patched config
            logger.info("Loading model (this may take a moment on first run)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            
            self.model_loaded = True
            logger.info(f"✅ Phi-3 loaded from HF: {self.model_id}")
            return True
            
        except ImportError as e:
            logger.error(f"Missing dependency for HF backend: {e}")
            logger.error("Install with: pip install transformers torch accelerate")
            return False
        except Exception as e:
            logger.error(f"Failed to load Phi-3 from HF: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1500,
        temperature: float = 0.0,
        stop_sequences: Optional[list] = None
    ) -> str:
        """
        Generate text using Phi-3.
        
        Args:
            prompt: Input prompt (formatted with system/user tags)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 for deterministic)
            stop_sequences: Sequences to stop generation
            
        Returns:
            Generated text (assistant response)
        """
        if not self.model_loaded:
            if not self.load_model():
                raise RuntimeError("Phi-3 model not loaded")
        
        try:
            import torch
            
            # Default stop sequences for Phi-3
            stop_sequences = stop_sequences or ["<|end|>", "</s>"]
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096 - max_tokens
            )
            
            # Move to device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response (exclude input tokens)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Clean up stop sequences from response
            for stop in stop_sequences:
                if stop in response:
                    response = response.split(stop)[0]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        return {
            "backend": "huggingface",
            "model_id": self.model_id,
            "model_loaded": self.model_loaded,
            "cache_dir": str(self.cache_dir),
            "quantization": self.use_quantization,
            "device": self.device
        }


def should_use_hf_backend() -> bool:
    """Check if HF backend should be used based on environment."""
    backend = os.getenv("PHI_BACKEND", "llama_cpp").lower()
    return backend == "hf" or backend == "huggingface"


# Singleton instance
_phi3_hf_loader_instance = None


def get_phi3_hf_loader() -> Phi3HFLoader:
    """Get or create singleton Phi3HFLoader instance."""
    global _phi3_hf_loader_instance
    
    if _phi3_hf_loader_instance is None:
        _phi3_hf_loader_instance = Phi3HFLoader()
    
    return _phi3_hf_loader_instance
