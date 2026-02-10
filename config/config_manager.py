"""
Configuration Manager for Receipt Extraction System
"""
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class OCRConfig:
    """Configuration class for OCR and Line Classification."""
    
    # Path configuration
    config_path: str = ""
    
    # Line Reconstruction parameters
    line_tolerance: float = 0.5
    debug_output: bool = True
    
    # Directory configuration (relative to project root)
    line_input_base_dir: str = "data/ocr_output"
    line_output_base_dir: str = "data/line_classification"
    categorized_ocr_dir: str = "data/ocr_output"
    reconstructed_output_dir: str = "data/reconstructed_text"
    
    # Document type processing flags
    process_receipts: bool = True
    process_invoices: bool = True
    process_others: bool = True
    
    # Specific directory names
    receipts_dir: str = "receipts"
    invoices_dir: str = "invoices"
    others_dir: str = "others"
    
    # Confidence categories
    confidence_levels: List[str] = field(default_factory=lambda: [
        'best_confidence', 
        'medium_confidence', 
        'low_confidence'
    ])
    
    # Statistics filenames
    line_stats_filename: str = "line_classification_stats.json"
    
    def __post_init__(self):
        """Initialize paths and load from JSON if available."""
        # Find project root (where this config folder is)
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        
        if not self.config_path:
            self.config_path = str(project_root / "config" / "receipt_extraction_config.json")
        
        # Ensure directories are absolute if they don't exist
        for attr in ['line_input_base_dir', 'line_output_base_dir', 
                    'categorized_ocr_dir', 'reconstructed_output_dir']:
            val = getattr(self, attr)
            if not os.path.isabs(val):
                setattr(self, attr, str(project_root / val))
                
        # Try to load existing config if it exists
        self.load_config()

    def load_config(self):
        """Load configuration from JSON if file exists."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    # Update fields from JSON
                    for key, value in data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")

class ConfigManager:
    """Alias/Wrapper for OCRConfig for backward compatibility if needed."""
    def __init__(self, config_path=None):
        self.ocr_config = OCRConfig(config_path=config_path) if config_path else OCRConfig()
        self.config = self._to_dict()

    def _to_dict(self):
        """Convert config to dictionary format for generic use."""
        return {
            'line_tolerance': self.ocr_config.line_tolerance,
            'debug_output': self.ocr_config.debug_output,
            'directories': {
                'input': self.ocr_config.line_input_base_dir,
                'output': self.ocr_config.line_output_base_dir
            }
        }
    
    def get_patterns(self, pattern_key):
        """Placeholder for pattern retrieval."""
        return []
    
    def get_rules(self, rules_key):
        """Placeholder for rule retrieval."""
        return {}
