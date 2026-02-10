"""
Totals Extractor - Extracts subtotal, total, and net after discount amounts
"""
import re
import pandas as pd
from typing import Dict, Optional, Any, List
from ..shared_utils.config_manager import ConfigManager
from ..shared_utils.confidence_scorer import ConfidenceScorer
from ..support_modules.payable_detector import PayableDetector
from ..models.extraction_models import TotalExtractionResult


class TotalsExtractor:
    """Extracts totals with enhanced strategy-based extraction and cross-field validation."""
    
    def __init__(self, config_manager: ConfigManager, confidence_scorer: ConfidenceScorer):
        self.config_manager = config_manager
        self.confidence_scorer = confidence_scorer
        self.extraction_rules = config_manager.config.get('extraction_rules', {})
        self.subtotal_patterns = config_manager.get_patterns('subtotal_patterns')
        self.total_patterns = config_manager.get_patterns('total_patterns')
        self.payable_detector = PayableDetector()
    
    def extract_totals(self, df, discount_context=None, structure_analysis=None) -> Dict[str, Any]:
        """
        Priority 3A & 3B: Enhanced Strategy-Based Totals Extraction with Cross-Field Validation
        """
        print("🔍 Starting extract_totals method")
        
        if 'totals_extraction' not in self.extraction_rules:
            print("🔍 DEBUG: No totals_extraction rules found - early return")
            return {'subtotal': None, 'net_after_discount': None, 'final_total': None, 'items_subtotal': None}
        
        print("🔍 DEBUG: Extraction rules found, proceeding")
        
        rules = self.extraction_rules['totals_extraction']
        target_classes = rules.get('target_classes', ['IGNORE', 'SUMMARY_KEY_VALUE'])
        
        # Handle different column name formats
        class_column = 'line_type' if 'line_type' in df.columns else 'predicted_class'
        
        # Start with target classes from primary classification
        summary_lines = df[df[class_column].isin(target_classes)].copy()
        
        # Also check the other classification column if it exists
        other_class_column = 'predicted_class' if class_column == 'line_type' else 'line_type'
        if other_class_column in df.columns:
            additional_lines = df[df[other_class_column].isin(target_classes)].copy()
            summary_lines = pd.concat([summary_lines, additional_lines]).drop_duplicates()
        
        # Include lines with total-related keywords
        total_keywords = ['total', 'subtotal', 'sub total', 'net total', 'grand total', 'balance due', 
                         'amount due', 'total due', 'final total', 'discount', 'vat total']
        
        text_column = 'cleaned_text' if 'cleaned_text' in df.columns else ('text' if 'text' in df.columns else 'original_text')
        
        keyword_lines = pd.DataFrame()
        for keyword in total_keywords:
            matching_lines = df[df[text_column].str.contains(keyword, case=False, na=False)].copy()
            if not matching_lines.empty:
                keyword_lines = pd.concat([keyword_lines, matching_lines])
        
        if not keyword_lines.empty:
            keyword_lines = keyword_lines.drop_duplicates()
            summary_lines = pd.concat([summary_lines, keyword_lines]).drop_duplicates()
            print(f"   🔍 Added {len(keyword_lines)} lines containing total-related keywords")
        
        # Determine extraction strategy
        if isinstance(discount_context, dict):
            has_discounts = discount_context.get('has_discounts', False)
        elif hasattr(discount_context, 'has_discounts'):
            has_discounts = discount_context.has_discounts
        else:
            has_discounts = False
        extraction_strategy = self._determine_extraction_strategy(df, has_discounts, discount_context, structure_analysis)
        
        # Filter patterns based on strategy
        filtered_subtotal_patterns = self._filter_patterns_by_strategy(
            self.subtotal_patterns, extraction_strategy.get('subtotal_rules', {}), 'subtotal'
        )
        filtered_total_patterns = self._filter_patterns_by_strategy(
            self.total_patterns, extraction_strategy.get('total_rules', {}), 'total'
        )
        
        print(f"🧮 Extracting totals from {len(summary_lines)} summary lines")
        
        results = {
            'subtotal': None,
            'net_after_discount': None,
            'items_subtotal': None,
            'final_total': None
        }
        
        # Search for subtotals
        print("🔍 Searching for SUBTOTAL patterns...")
        subtotal_highest_conf = 0.0
        
        for _, row in summary_lines.iterrows():
            text = ""
            for col_name in ['cleaned_text', 'text', 'original_text']:
                if col_name in row and pd.notna(row[col_name]):
                    text = str(row[col_name]).strip()
                    break
            
            for pattern_config in filtered_subtotal_patterns:
                pattern = pattern_config['pattern']
                match = re.search(pattern, text)
                if match:
                    text_lower = text.lower()
                    
                    # For UK receipts, "re-discount Subtotal" means the subtotal BEFORE discounts
                    is_prediscount_subtotal = 're-discount' in text_lower and 'subtotal' in text_lower
                    
                    subtotal_exclusions = ['discount subtotal', 'rediscount', 'pre-discount', 
                                           'prediscount', 'before discount', 'discount sub', 'subtotal -', '- subtotal']
                    
                    matching_exclusions = [kw for kw in subtotal_exclusions if kw in text_lower]
                    if matching_exclusions and not is_prediscount_subtotal:
                        continue
                    
                    amount_str = match.group(1).replace(',', '.')
                    try:
                        amount_float = float(amount_str)
                        
                        # Skip negative amounts for subtotals
                        if amount_float < 0:
                            continue
                            
                        # Handle confidence
                        ml_confidence = 0.5
                        for conf_col in ['confidence', 'confidence_score', 'primary_confidence']:
                            if conf_col in row and pd.notna(row[conf_col]):
                                ml_confidence = self.confidence_scorer.safe_convert(row[conf_col])
                                break
                        
                        # Pattern bonus
                        pattern_bonus = 0.0
                        if any(keyword in pattern_config['description'].lower() for keyword in ['subtotal']):
                            pattern_bonus = 0.4
                        elif any(keyword in pattern_config['description'].lower() for keyword in ['sub total', 'item total']):
                            pattern_bonus = 0.35
                        elif any(keyword in pattern_config['description'].lower() for keyword in ['net total', 'net ttl']):
                            pattern_bonus = 0.1
                        elif 'excl vat' in pattern_config['description'].lower():
                            pattern_bonus = 0.25
                        
                        # Rule priority
                        rule_priority = 0
                        pattern_desc = pattern_config['description'].lower()
                        
                        if 'subtotal' in pattern_desc and 'discount' not in pattern_desc:
                            rule_priority = 100
                        elif any(keyword in pattern_desc for keyword in ['sub total', 'item total']) and 'discount' not in pattern_desc:
                            rule_priority = 90
                        elif 'excl vat' in pattern_desc:
                            rule_priority = 80
                        elif 'net ttl (net total abbreviation)' in pattern_desc or 'net total' in pattern_desc:
                            rule_priority = 30
                        elif 'discount' in pattern_desc:
                            rule_priority = 10
                        
                        combined_confidence = min((ml_confidence + 0.5 + pattern_bonus) / 1.7, 1.0)
                        
                        # Skip "net" amounts for subtotal extraction
                        if 'net' in text.lower() and 'subtotal' not in text.lower():
                            continue
                            
                        if rule_priority > (results.get('subtotal', {}).get('rule_priority', -1) if results.get('subtotal') else -1):
                            subtotal_data = TotalExtractionResult(
                                amount=amount_float,
                                raw_text=text,
                                confidence=combined_confidence,
                                line_number=self.confidence_scorer.safe_convert(row.get('line_number', 0), int),
                                extraction_method='config_regex_pattern',
                                rule_priority=rule_priority,
                                pattern_used=pattern_config['description'],
                                field_type='subtotal_items_before_discounts'
                            )
                            
                            results['subtotal'] = subtotal_data
                            results['items_subtotal'] = subtotal_data
                            print(f"   ✅ New best subtotal: £{amount_float} from '{text}'")
                            
                    except ValueError:
                        continue
        
        # Try enhanced subtotal detection if no subtotal found
        if not results.get('subtotal'):
            print("🔍 No subtotal found with regular patterns. Trying enhanced detection...")
            all_text_lines = []
            for _, row in df.iterrows():
                text = ""
                for col_name in ['cleaned_text', 'text', 'original_text']:
                    if col_name in row and pd.notna(row[col_name]):
                        text = str(row[col_name]).strip()
                        break
                if text:
                    all_text_lines.append(text)
            
            receipt_text = '\n'.join(all_text_lines)
            simple_subtotal = self.payable_detector.detect_subtotal_simple_priority(receipt_text)
            
            if simple_subtotal:
                subtotal_data = TotalExtractionResult(
                    amount=simple_subtotal['amount'],
                    raw_text=simple_subtotal['raw_text'],
                    confidence=simple_subtotal['confidence'],
                    line_number=simple_subtotal.get('line_number', 0),
                    extraction_method='simple_priority_detection',
                    rule_priority=110,
                    pattern_used=f"Simple priority: {simple_subtotal['source']} - {simple_subtotal['pattern_description']}",
                    field_type='subtotal_items_before_discounts'
                )
                
                results['subtotal'] = subtotal_data
                results['items_subtotal'] = subtotal_data
                print(f"🎯 Simple priority detection SUCCESS")
        
        # Search for totals using simple priority detection first
        print("🎯 Priority 1: Applying simple priority-based net after discount detection...")
        simple_total_result = self.payable_detector.extract_net_after_discount_simple_priority(df)
        
        if simple_total_result:
            print(f"✅ Simple priority detector found net after discount: £{simple_total_result['amount']}")
            simple_result = TotalExtractionResult(
                amount=simple_total_result['amount'],
                raw_text=simple_total_result['raw_text'],
                confidence=simple_total_result['confidence'],
                rule_priority=110,
                line_number=simple_total_result.get('line_number', 0),
                extraction_method='simple_priority_context_aware',
                pattern_used=f"Simple priority detection - method: {simple_total_result['extraction_method']}",
                field_type='net_after_discount_primary'
            )
            
            results['final_total'] = simple_result
            results['net_after_discount'] = simple_result
        
        # Fallback to traditional regex patterns
        if not results.get('final_total'):
            print("🔍 Priority 3: Final fallback to traditional regex pattern matching...")
            for _, row in summary_lines.iterrows():
                current_line = self.confidence_scorer.safe_convert(row.get('line_number', 0), int)
                if results.get('subtotal') and results['subtotal'].line_number == current_line:
                    continue
                    
                text = ""
                for col_name in ['cleaned_text', 'text', 'original_text']:
                    if col_name in row and pd.notna(row[col_name]):
                        text = str(row[col_name]).strip()
                        break
                
                for pattern_config in filtered_total_patterns:
                    pattern = pattern_config['pattern']
                    match = re.search(pattern, text)
                    if match:
                        text_lower = text.lower()
                        total_exclusions = ['discount total', 'discount amount', 'discount due', 'total discount',
                                            'savings total', 'coupon total', 'promo total', 'off total',
                                            'total -', '- total', 'discount -', '- discount']
                        
                        subtotal_exclusions = ['discount subtotal', 're-discount', 'rediscount', 'pre-discount', 
                                               'prediscount', 'before discount', 'discount sub', 'subtotal -', '- subtotal']
                        
                        all_exclusions = total_exclusions + subtotal_exclusions
                        matching_exclusions = [kw for kw in all_exclusions if kw in text_lower]
                        if matching_exclusions:
                            continue
                        
                        amount_str = match.group(1).replace(',', '.')
                        try:
                            amount_float = float(amount_str)
                            
                            # Handle confidence
                            ml_confidence = 0.5
                            for conf_col in ['confidence', 'confidence_score', 'primary_confidence']:
                                if conf_col in row and pd.notna(row[conf_col]):
                                    ml_confidence = self.confidence_scorer.safe_convert(row[conf_col])
                                    break
                            
                            # Pattern bonus
                            pattern_bonus = 0.0
                            if any(keyword in pattern_config['description'].lower() for keyword in ['total due', 'amount due', 'balance due', 'grand total', 'total payable']):
                                pattern_bonus = 0.4
                            elif any(keyword in pattern_config['description'].lower() for keyword in ['total to pay', 'amount payable', 'incl vat', 'vat ttl']):
                                pattern_bonus = 0.3
                            elif any(keyword in pattern_config['description'].lower() for keyword in ['basic total', 'ttl abbreviation']):
                                pattern_bonus = 0.2
                            elif any(keyword in pattern_config['description'].lower() for keyword in ['net total', 'net ttl']):
                                pattern_bonus = 0.5
                            elif any(keyword in pattern_config['description'].lower() for keyword in ['vat ttl', 'vat total']):
                                pattern_bonus = 0.2
                            elif 'subtotal' in pattern_config['description'].lower():
                                pattern_bonus = 0.1
                            
                            # Rule priority
                            rule_priority = 0
                            pattern_desc = pattern_config['description'].lower()
                            
                            if any(keyword in pattern_desc for keyword in ['total due', 'amount due', 'balance due', 'grand total', 'total payable']):
                                rule_priority = 100
                            elif any(keyword in pattern_desc for keyword in ['basic total', 'total to pay', 'amount payable']):
                                rule_priority = 90
                            elif any(keyword in pattern_desc for keyword in ['total incl vat', 'amount incl vat']):
                                rule_priority = 80
                            elif 'ttl abbreviation' in pattern_desc and 'vat' not in pattern_desc:
                                rule_priority = 70
                            elif 'subtotal as final amount' in pattern_desc:
                                rule_priority = 60
                            elif ('net ttl' in pattern_desc or 'net total' in pattern_desc):
                                rule_priority = 50
                            elif 'vat ttl' in pattern_desc or 'vat total' in pattern_desc:
                                rule_priority = 30
                            elif 'subtotal' in pattern_desc and 'as final amount' not in pattern_desc:
                                rule_priority = 20
                            else:
                                rule_priority = 40
                            
                            combined_confidence = min((ml_confidence + 0.5 + pattern_bonus) / 1.8, 1.0)
                            
                            current_total = results.get('net_after_discount')
                            current_priority = current_total.rule_priority if current_total else -1
                            
                            if rule_priority > current_priority:
                                total_data = TotalExtractionResult(
                                    amount=amount_float,
                                    raw_text=text,
                                    confidence=combined_confidence,
                                    rule_priority=rule_priority,
                                    line_number=self.confidence_scorer.safe_convert(row.get('line_number', 0), int),
                                    extraction_method='config_regex_pattern',
                                    pattern_used=pattern_config['description'],
                                    field_type='final_payable_after_discounts' if has_discounts else 'final_total'
                                )
                                
                                if has_discounts:
                                    results['net_after_discount'] = total_data
                                    results['final_total'] = total_data
                                    print(f"   ✅ New best net after discount: £{amount_float} from '{text}'")
                                else:
                                    results['final_total'] = total_data
                                    results['net_after_discount'] = total_data
                                    print(f"   ✅ New best final total: £{amount_float} from '{text}'")
                                
                        except ValueError:
                            continue
        
        # Cross-field validation
        print("🔬 Running cross-field validation...")
        validation_results = self._perform_cross_field_validation(
            results, extraction_strategy, has_discounts, discount_context
        )
        
        # Apply validation corrections if needed
        if validation_results.get('corrections'):
            for correction in validation_results['corrections']:
                if correction['action'] == 'swap_fields':
                    temp = results['subtotal']
                    results['subtotal'] = results['net_after_discount']
                    results['net_after_discount'] = temp
                    print(f"   ↔️ Swapped subtotal and total based on validation")
        
        # Print final results (handle both dataclass objects and dictionaries)
        final_subtotal = results['items_subtotal']
        subtotal_str = 'N/A'
        if final_subtotal:
            subtotal_str = str(final_subtotal.amount if hasattr(final_subtotal, 'amount') else final_subtotal.get('amount'))
        
        final_total = results['final_total']
        total_str = 'N/A'
        if final_total:
            total_str = str(final_total.amount if hasattr(final_total, 'amount') else final_total.get('amount'))
        
        print(f"✅ Final ITEMS SUBTOTAL: £{subtotal_str}")
        print(f"✅ Final TOTAL: £{total_str}")
        
        # Convert TotalExtractionResult objects to dictionaries for compatibility
        converted_results = {}
        for key, value in results.items():
            if value is None:
                converted_results[key] = None
            elif hasattr(value, '__dataclass_fields__'):
                # Convert dataclass to dictionary
                converted_results[key] = {
                    'amount': value.amount,
                    'raw_text': value.raw_text,
                    'confidence': value.confidence,
                    'line_number': value.line_number,
                    'extraction_method': value.extraction_method,
                    'rule_priority': getattr(value, 'rule_priority', 0),
                    'pattern_used': getattr(value, 'pattern_used', None),
                    'field_type': getattr(value, 'field_type', None)
                }
            else:
                converted_results[key] = value
        
        return converted_results
    
    def _determine_extraction_strategy(self, df, has_discounts, discount_context, structure_analysis=None):
        """Determine extraction strategy based on receipt structure and context."""
        strategy = {
            'strategy_name': 'STANDARD',
            'receipt_type': 'UNKNOWN',
            'field_mapping': 'Standard',
            'subtotal_rules': {},
            'total_rules': {},
            'validation_rules': {}
        }
        
        if structure_analysis:
            if hasattr(structure_analysis, 'receipt_type'):
                receipt_type = structure_analysis.receipt_type
            else:
                receipt_type = 'UNKNOWN'
            strategy['receipt_type'] = receipt_type
            
            if receipt_type == 'UK_VAT':
                strategy.update(self._get_uk_vat_strategy(has_discounts, discount_context))
            elif receipt_type == 'US_SALES_TAX':
                strategy.update(self._get_us_tax_strategy(has_discounts, discount_context))
            elif receipt_type == 'EU_VAT':
                strategy.update(self._get_eu_vat_strategy(has_discounts, discount_context))
            else:
                strategy.update(self._get_standard_strategy(has_discounts, discount_context))
        else:
            strategy.update(self._get_heuristic_strategy(df, has_discounts, discount_context))
        
        return strategy
    
    def _get_uk_vat_strategy(self, has_discounts, discount_context):
        """UK VAT-specific extraction strategy."""
        return {
            'strategy_name': 'UK_VAT_STRATEGY',
            'field_mapping': 'UK VAT Context-Aware',
            'subtotal_rules': {
                'prefer_patterns': ['subtotal', 'sub total', 'items total'],
                'avoid_patterns': ['net total', 'net amount'],
                'context_awareness': True,
                'vat_context': True
            },
            'total_rules': {
                'prefer_patterns': ['total', 'final total', 'amount due'],
                'discount_aware': has_discounts,
                'net_after_discount_logic': has_discounts
            },
            'validation_rules': {
                'expect_vat': True,
                'vat_rate_range': [0, 5, 20],
                'discount_validation': has_discounts
            }
        }
    
    def _get_us_tax_strategy(self, has_discounts, discount_context):
        """US Sales Tax-specific extraction strategy."""
        return {
            'strategy_name': 'US_SALES_TAX_STRATEGY',
            'field_mapping': 'US Tax Context-Aware',
            'subtotal_rules': {
                'prefer_patterns': ['subtotal', 'sub total', 'before tax'],
                'avoid_patterns': ['net'],
                'context_awareness': True,
                'tax_context': True
            },
            'total_rules': {
                'prefer_patterns': ['total', 'grand total', 'final total'],
                'discount_aware': has_discounts,
                'net_after_discount_logic': has_discounts
            },
            'validation_rules': {
                'expect_sales_tax': True,
                'tax_rate_range': [0, 15],
                'discount_validation': has_discounts
            }
        }
    
    def _get_eu_vat_strategy(self, has_discounts, discount_context):
        """EU VAT-specific extraction strategy."""
        return {
            'strategy_name': 'EU_VAT_STRATEGY',
            'field_mapping': 'EU VAT Context-Aware',
            'subtotal_rules': {
                'prefer_patterns': ['subtotal', 'netto', 'before vat'],
                'context_awareness': True,
                'vat_context': True
            },
            'total_rules': {
                'prefer_patterns': ['total', 'brutto', 'final amount'],
                'discount_aware': has_discounts,
                'net_after_discount_logic': has_discounts
            },
            'validation_rules': {
                'expect_vat': True,
                'vat_rate_range': [0, 10, 21, 25],
                'discount_validation': has_discounts
            }
        }
    
    def _get_standard_strategy(self, has_discounts, discount_context):
        """Standard fallback extraction strategy."""
        return {
            'strategy_name': 'STANDARD_STRATEGY',
            'field_mapping': 'Standard',
            'subtotal_rules': {
                'prefer_patterns': ['subtotal', 'sub total'],
                'context_awareness': False
            },
            'total_rules': {
                'prefer_patterns': ['total'],
                'discount_aware': has_discounts,
                'net_after_discount_logic': has_discounts
            },
            'validation_rules': {
                'discount_validation': has_discounts
            }
        }
    
    def _get_heuristic_strategy(self, df, has_discounts, discount_context):
        """Heuristic-based strategy when structure analysis unavailable."""
        text_column = 'cleaned_text' if 'cleaned_text' in df.columns else ('text' if 'text' in df.columns else 'original_text')
        all_text = ' '.join(df[text_column].fillna('').astype(str)).lower()
        
        if any(indicator in all_text for indicator in ['vat', '£', 'hmrc']):
            return self._get_uk_vat_strategy(has_discounts, discount_context)
        elif any(indicator in all_text for indicator in ['sales tax', '$', 'state tax']):
            return self._get_us_tax_strategy(has_discounts, discount_context)
        elif any(indicator in all_text for indicator in ['mwst', '€', 'iva']):
            return self._get_eu_vat_strategy(has_discounts, discount_context)
        else:
            return self._get_standard_strategy(has_discounts, discount_context)
    
    def _filter_patterns_by_strategy(self, patterns, strategy_rules, pattern_type):
        """Filter and prioritize patterns based on extraction strategy."""
        if not strategy_rules or not patterns:
            return patterns
        
        prefer_patterns = strategy_rules.get('prefer_patterns', [])
        avoid_patterns = strategy_rules.get('avoid_patterns', [])
        
        if not prefer_patterns and not avoid_patterns:
            return patterns
        
        filtered_patterns = []
        preferred_patterns = []
        
        for pattern_config in patterns:
            pattern_text = pattern_config.get('pattern', '').lower()
            
            # Check if this pattern should be avoided
            should_avoid = any(avoid in pattern_text for avoid in avoid_patterns)
            if should_avoid:
                continue
            
            # Check if this pattern is preferred
            is_preferred = any(prefer in pattern_text for prefer in prefer_patterns)
            if is_preferred:
                preferred_patterns.append(pattern_config)
            else:
                filtered_patterns.append(pattern_config)
        
        # Return preferred patterns first, then others
        return preferred_patterns + filtered_patterns
    
    def _perform_cross_field_validation(self, results, extraction_strategy, has_discounts, discount_context):
        """Perform cross-field validation to ensure mathematical consistency."""
        validation_results = {
            'corrections': [],
            'warnings': [],
            'confidence_adjustments': False
        }
        
        # Extract amounts for validation (handle both dataclass objects and dictionaries)
        subtotal_val = results.get('subtotal')
        total_val = results.get('final_total')
        
        subtotal_amt = None
        if subtotal_val:
            subtotal_amt = subtotal_val.amount if hasattr(subtotal_val, 'amount') else subtotal_val.get('amount')
        
        total_amt = None
        if total_val:
            total_amt = total_val.amount if hasattr(total_val, 'amount') else total_val.get('amount')
        
        # Basic mathematical consistency check
        if subtotal_amt is not None and total_amt is not None:
            if extraction_strategy.get('receipt_type') == 'UK_VAT' and has_discounts:
                if subtotal_amt > total_amt:
                    validation_results['corrections'].append({
                        'type': 'FIELD_SWAP',
                        'description': f'UK VAT context: Subtotal £{subtotal_amt} > Total £{total_amt}',
                        'action': 'swap_fields',
                        'confidence': 'high'
                    })
            elif not has_discounts:
                if subtotal_amt > total_amt * 1.3:
                    validation_results['warnings'].append(
                        f'No discounts but subtotal £{subtotal_amt} much higher than total £{total_amt}'
                    )
        
        return validation_results
    
    def extract_totals_with_enhanced_payable_fallback(self, df):
        """Enhanced totals extraction with payable_detector.py fallback integration."""
        results = self.extract_totals(df)
        
        # Check if we need fallback for final_total (handle both dataclass and dict)
        final_total_val = results.get('final_total')
        needs_fallback = not final_total_val
        
        if not needs_fallback and final_total_val:
            confidence = final_total_val.confidence if hasattr(final_total_val, 'confidence') else final_total_val.get('confidence', 1.0)
            if confidence < 0.5:
                needs_fallback = True
        
        if needs_fallback:
            print("🔄 Applying enhanced payable detection fallback...")
            text_column = None
            for col_name in ['cleaned_text', 'text', 'original_text']:
                if col_name in df.columns:
                    text_column = col_name
                    break
            
            if text_column:
                all_text = '\n'.join(df[text_column].fillna('').astype(str))
                enhanced_payable = self.payable_detector.find_payable_amount_enhanced(all_text)
                
                if enhanced_payable:
                    results['final_total'] = {
                        'amount': enhanced_payable['amount'],
                        'raw_text': enhanced_payable['raw_text'],
                        'confidence': enhanced_payable['confidence'],
                        'extraction_method': 'enhanced_payable_fallback',
                        'rule_priority': 100,
                        'pattern_used': f"Enhanced detection - keyword: {enhanced_payable['matched_keyword']}",
                        'field_type': 'final_payable_amount'
                    }
                    print(f"✅ Enhanced payable detector found: {enhanced_payable['amount']}")
        
        return results