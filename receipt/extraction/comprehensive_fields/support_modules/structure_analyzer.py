"""
Structure Analyzer - Analyzes receipt structure for extraction strategy
"""
import re
import pandas as pd
from typing import Dict, Any, List
from ..models.extraction_models import ReceiptStructureAnalysis


class StructureAnalyzer:
    """Analyzes receipt structure for extraction strategy."""
    
    def analyze_receipt_structure(self, df) -> ReceiptStructureAnalysis:
        """
        Priority 2A: Comprehensive Receipt Structure Analysis
        """
        print("🔍 Priority 2A: Analyzing Receipt Structure...")
        
        # Combine all text for comprehensive analysis
        text_column = None
        for col_name in ['cleaned_text', 'text', 'original_text', 'line_text']:
            if col_name in df.columns:
                text_column = col_name
                break
        
        if text_column is None:
            raise ValueError(f"No text column found. Available columns: {list(df.columns)}")
        
        print(f"   Using text column: '{text_column}'")
        all_text = ' '.join(df[text_column].fillna('').astype(str)).lower()
        all_lines = df[text_column].fillna('').astype(str).tolist()
        
        structure_analysis = ReceiptStructureAnalysis()
        
        # 1. CURRENCY AND LOCALE DETECTION
        print("   💰 Analyzing currency and locale...")
        currency_patterns = {
            'GBP': [r'£', r'gbp', r'pounds?', r'pence'],
            'EUR': [r'€', r'eur', r'euros?'],
            'USD': [r'\$', r'usd', r'dollars?', r'cents?']
        }
        
        detected_currencies = {}
        for currency, patterns in currency_patterns.items():
            count = sum(1 for pattern in patterns if re.search(pattern, all_text))
            if count > 0:
                detected_currencies[currency] = count
        
        if detected_currencies:
            structure_analysis.currency_detected = max(detected_currencies, key=detected_currencies.get)
            print(f"      ✅ Currency detected: {structure_analysis.currency_detected}")
        
        # 2. RECEIPT TYPE AND TAX SYSTEM DETECTION
        print("   🏛️  Analyzing tax system...")
        
        # UK VAT indicators
        uk_vat_indicators = [
            r'vat\s+(?:reg|registration|no|number)', r'vat\s+@\s*\d+(?:\.\d+)?%',
            r'vat\s+\d+(?:\.\d+)?%', r'hmrc', r'customs\s+&\s+excise',
            r'vat(?:\s+inc)?(?:lusive)?', r'vat\s+rate'
        ]
        
        # US Sales Tax indicators  
        us_tax_indicators = [
            r'sales?\s+tax', r'state\s+tax', r'local\s+tax', 
            r'tax\s+\d+(?:\.\d+)?%', r'tax\s+@\s*\d+(?:\.\d+)?%',
            r'(?<!v)(?<!va)tax(?:\s+inc)?(?:lusive)?'
        ]
        
        # EU VAT indicators
        eu_vat_indicators = [
            r'mwst', r'iva', r'tva', r'btw', r'dph', r'vat\s+id',
            r'eu\s+vat', r'vies'
        ]
        
        uk_score = sum(1 for pattern in uk_vat_indicators if re.search(pattern, all_text))
        us_score = sum(1 for pattern in us_tax_indicators if re.search(pattern, all_text))
        eu_score = sum(1 for pattern in eu_vat_indicators if re.search(pattern, all_text))
        
        # Combine with currency detection for better accuracy
        if structure_analysis.currency_detected == 'GBP':
            uk_score += 2
        elif structure_analysis.currency_detected == 'USD':
            us_score += 2
        elif structure_analysis.currency_detected == 'EUR':
            eu_score += 2
        
        if uk_score >= us_score and uk_score >= eu_score and uk_score > 0:
            structure_analysis.receipt_type = 'UK_VAT'
            structure_analysis.tax_system = 'VAT'
            structure_analysis.tax_structure['has_vat'] = True
            print("      🇬🇧 UK VAT receipt detected")
        elif us_score > uk_score and us_score >= eu_score and us_score > 0:
            structure_analysis.receipt_type = 'US_SALES_TAX'
            structure_analysis.tax_system = 'SALES_TAX'
            structure_analysis.tax_structure['has_sales_tax'] = True
            print("      🇺🇸 US Sales Tax receipt detected")
        elif eu_score > 0:
            structure_analysis.receipt_type = 'EU_VAT'
            structure_analysis.tax_system = 'VAT'
            structure_analysis.tax_structure['has_vat'] = True
            print("      🇪🇺 EU VAT receipt detected")
        else:
            print("      ❓ Tax system not clearly identified")
        
        # 3. DISCOUNT ANALYSIS
        print("   🏷️  Analyzing discount structure...")
        
        discount_context = self._analyze_discounts(all_text, all_lines)
        structure_analysis.discount_analysis = discount_context
        
        # 4. TAX RATE DETECTION
        print("   📊 Analyzing tax rates...")
        tax_rate_patterns = [
            r'(?:vat|tax)\s*@?\s*(\d+(?:\.\d+)?)%',
            r'(\d+(?:\.\d+)?)%\s*(?:vat|tax)',
            r'rate\s*:?\s*(\d+(?:\.\d+)?)%'
        ]
        
        detected_rates = []
        for pattern in tax_rate_patterns:
            matches = re.findall(pattern, all_text)
            for match in matches:
                try:
                    rate = float(match)
                    if 5 <= rate <= 30:  # Reasonable tax rate range
                        detected_rates.append(rate)
                except ValueError:
                    continue
        
        if detected_rates:
            structure_analysis.tax_structure['tax_rate_detected'] = max(set(detected_rates), key=detected_rates.count)
            print(f"      📈 Tax rate detected: {structure_analysis.tax_structure['tax_rate_detected']}%")
        
        # 5. FORMAT COMPLEXITY ANALYSIS
        print("   📋 Analyzing receipt complexity...")
        
        complexity_indicators = {
            'multiple_tax_rates': len(set(detected_rates)) > 1,
            'multiple_discounts': len(structure_analysis.discount_analysis['discount_types']) > 1,
            'itemized_receipt': len(df) > 5,
            'detailed_breakdown': any('breakdown' in line.lower() for line in all_lines),
            'multiple_totals': sum(1 for line in all_lines if re.search(r'\btotal\b', line.lower())) > 1
        }
        
        complexity_score = sum(complexity_indicators.values())
        if complexity_score >= 3:
            structure_analysis.format_complexity = 'complex'
        elif complexity_score >= 1:
            structure_analysis.format_complexity = 'moderate'
        else:
            structure_analysis.format_complexity = 'simple'
        
        structure_analysis.structural_elements = [k for k, v in complexity_indicators.items() if v]
        
        print(f"      📊 Receipt complexity: {structure_analysis.format_complexity}")
        
        # 6. CONFIDENCE SCORING
        structure_analysis.confidence_scores = {
            'receipt_type_confidence': max(uk_score, us_score, eu_score) / 5.0 if max(uk_score, us_score, eu_score) > 0 else 0.0,
            'tax_system_confidence': 0.9 if structure_analysis.tax_system != 'unknown' else 0.0,
            'currency_confidence': 0.95 if structure_analysis.currency_detected else 0.0,
            'discount_confidence': structure_analysis.discount_analysis['confidence'],
            'overall_confidence': 0.0
        }
        
        # Calculate overall confidence
        confidence_values = [v for v in structure_analysis.confidence_scores.values() if v > 0]
        if confidence_values:
            structure_analysis.confidence_scores['overall_confidence'] = sum(confidence_values) / len(confidence_values)
        
        print(f"   ✅ Structure analysis complete (confidence: {structure_analysis.confidence_scores['overall_confidence']:.2f})")
        print(f"      Receipt Type: {structure_analysis.receipt_type}")
        print(f"      Tax System: {structure_analysis.tax_system}")
        print(f"      Has Discounts: {structure_analysis.discount_analysis['has_discounts']}")
        print(f"      Complexity: {structure_analysis.format_complexity}")
        
        return structure_analysis
    
    def _analyze_discounts(self, all_text: str, all_lines: List[str]) -> Dict[str, Any]:
        """Analyze discount structure from text."""
        # Exclusions - text that should NOT be considered discount-related
        discount_exclusions = [
            'no loyalty card', 'loyalty card presented', 'register for loyalty',
            'rewards on', 'visit www', 'download the', 'use the website',
            'for more information', 'member benefits', 'enjoy', 'fuelsave',
            'total net', 'net total', 'vat', 'tax'
        ]
        
        # Check for exclusions
        potential_exclusions = [exc for exc in discount_exclusions if exc in all_text]
        exclusions_found = []
        
        # Look for explicit discount indicators
        discount_indicators = [
            r'\bdiscount\b', r'\bcoupon\s+(?:used|applied|redeemed)\b',
            r'\bvoucher\s+(?:used|applied|redeemed)\b', r'\bpromo\s+code\b.*applied',
            r'\boffer\b', r'\bdeal\b', r'\breduction\b', r'\bmarkdown\b',
            r'\bclearance\b', r'\bmember\s+discount\b', r'\bloyalty\s+discount\b',
            r'\bstaff\s+discount\b', r'\bemployee\s+discount\b',
            r'save\s*£?\$?\d+', r'\d+%\s*off', r'buy\s+\d+\s+get\s+\d+',
            r'bogof', r'two\s+for\s+one', r'coupon\s+applied',
            r'member\s+(?:discount|savings)', r'loyalty\s+(?:discount|savings)',
            r'(?:discount|coupon|voucher|reduction)\s*-\s*£?\$?\d+',
            r'-\s*£?\$?\d+.*(?:discount|coupon|voucher|reduction)'
        ]
        
        found_discount_indicators = []
        discount_types = []
        
        for indicator in discount_indicators:
            matches = re.findall(indicator, all_text)
            if matches:
                found_discount_indicators.extend(matches)
                if 'coupon' in indicator or 'voucher' in indicator:
                    discount_types.append('COUPON')
                elif 'member' in indicator or 'loyalty' in indicator:
                    discount_types.append('LOYALTY')
                elif 'staff' in indicator or 'employee' in indicator:
                    discount_types.append('EMPLOYEE')
                elif '%' in str(matches[0]) or 'off' in indicator:
                    discount_types.append('PERCENTAGE')
                elif 'buy' in indicator or 'bogof' in indicator:
                    discount_types.append('PROMOTIONAL')
                else:
                    discount_types.append('GENERAL')
        
        # Determine final exclusions
        if potential_exclusions and not found_discount_indicators:
            exclusions_found = potential_exclusions
        
        # Calculate discount confidence
        discount_confidence = 0.0
        if found_discount_indicators:
            discount_confidence = min(len(found_discount_indicators) * 0.4, 1.0)
            if exclusions_found:
                discount_confidence = max(discount_confidence * 0.6, 0.3)
        
        return {
            'has_discounts': len(found_discount_indicators) > 0 and discount_confidence > 0.3,
            'discount_types': list(set(discount_types)),
            'discount_indicators': found_discount_indicators,
            'exclusions_found': exclusions_found,
            'confidence': discount_confidence
        }