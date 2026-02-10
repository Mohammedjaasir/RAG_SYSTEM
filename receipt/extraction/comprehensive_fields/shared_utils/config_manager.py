"""
Configuration Manager - Handles config loading and management
"""
import json
import re
import os
from pathlib import Path


class ConfigManager:
    """Manages configuration loading and pattern updates."""
    
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = self._get_default_config_path()
        
        self.config_path = Path(config_path)
        self.config = self.load_configuration()
        self.optional_colon_patterns_updated = False
        self._update_patterns_for_optional_colons()
    
    def _get_default_config_path(self):
        """Get default config path."""
        return os.path.join(os.path.dirname(__file__), '..','..','..', '..', '..', '..', '..', 'config', 'receipt_extraction_config.json')
    
    def load_configuration(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Provide default configuration if config file is not available."""
        return {
            'suppliers': {
                'all_suppliers': [
                    'Morrison', 'MORRISON', 'Morrisons', 'MORRISONS',
                    'TESCO', 'Tesco', 'tesco', 'Tesco Express',
                    'ASDA', 'Asda', 'asda',
                    'Sainsbury', 'Sainsburys', 'SAINSBURY', 'SAINSBURYS',
                    'ALDI', 'Aldi', 'aldi',
                    'LIDL', 'Lidl', 'lidl'
                ]
            },
            'extraction_patterns': {
                'date_patterns': [
                    {'pattern': r'\\b\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}\\b', 'description': 'DD/MM/YYYY format'}
                ]
            },
            'extraction_rules': {
                'supplier_extraction': {'target_classes': ['HEADER']},
                'date_extraction': {'target_classes': ['HEADER', 'FOOTER', 'IGNORE']},
                'totals_extraction': {'target_classes': ['IGNORE', 'SUMMARY_KEY_VALUE']}
            },
            'confidence_scoring': {
                'scoring_factors': {
                    'pattern_match_exact': 1.0,
                    'fallback_method': 0.3
                }
            }
        }
    
    def _update_patterns_for_optional_colons(self):
        """Update all extraction patterns to make colons optional in key-value pairs."""
        if not self.config or 'extraction_patterns' not in self.config:
            print("⚠️  No extraction patterns found in config")
            self.optional_colon_patterns_updated = False
            return
        
        pattern_groups = [
            'vat_patterns', 'receipt_number_patterns', 'invoice_number_patterns',
            'transaction_number_patterns', 'reference_number_patterns', 
            'auth_code_patterns', 'total_patterns', 'subtotal_patterns'
        ]
        
        updated_count = 0
        for pattern_group in pattern_groups:
            if pattern_group in self.config['extraction_patterns']:
                patterns = self.config['extraction_patterns'][pattern_group]
                for pattern_item in patterns:
                    if 'pattern' in pattern_item:
                        old_pattern = pattern_item['pattern']
                        new_pattern = old_pattern
                        
                        # Replace literal colons in patterns
                        new_pattern = re.sub(r'(\\s\*):(?!\\s\*)', r'\1:?', new_pattern)
                        new_pattern = re.sub(r':(?=\\s)', r':?', new_pattern)
                        new_pattern = re.sub(r':?\?', ':?', new_pattern)
                        new_pattern = re.sub(r'\(:?\?i\)', '(?i)', new_pattern)
                        
                        if old_pattern != new_pattern:
                            pattern_item['pattern'] = new_pattern
                            updated_count += 1
        
        self.optional_colon_patterns_updated = updated_count > 0
        if updated_count > 0:
            print(f"✅ Updated {updated_count} patterns to make colons optional")
    
    def get_patterns(self, pattern_key):
        """Get patterns by key."""
        return self.config.get('extraction_patterns', {}).get(pattern_key, [])
    
    def get_rules(self, rules_key):
        """Get extraction rules by key."""
        return self.config.get('extraction_rules', {}).get(rules_key, {})
    
    def get_suppliers(self):
        """Get supplier list."""
        return self.config.get('suppliers', {}).get('all_suppliers', [])
    
    def get_keyword_categories(self):
        """Get keyword categories."""
        return self.config.get('keyword_categories', {}).get('categories', {})