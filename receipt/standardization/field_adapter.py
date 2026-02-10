#!/usr/bin/env python3
"""
Field Standardization Adapter v1.4.0
Converts between legacy field formats and standardized schema.
Enhanced with item_details support for comprehensive extraction.
v1.4.0: Enhanced item extraction support with sophisticated pattern recognition.
Provides backward compatibility while migrating to standardized naming.

Version 1.2.0: Enhanced Address Field Conversion Support
- Added enhanced address field structure conversion for contextual address extraction
- Improved contact information field mapping (address, phone, email, website, fax)
- Enhanced field value extraction from multiple address data structures
- Added support for supplier-context aware address field conversion
"""

import copy
from typing import Dict, List, Any, Optional, Union
from .standardized_schema import (
    AmountField, TextField, DateField, ExtractionMetadata,
    FieldMigrationMapping, StandardizedReceiptSchema,
    migrate_field_name, get_ui_display_name,
    create_amount_field, create_text_field,
    validate_standardized_output
)

class FieldStandardizationAdapter:
    """
    Field Standardization Adapter v1.4.0

    Enhanced adapter class to convert between legacy and standardized field formats.
    Now supports item_details structure for comprehensive item extraction.
    v1.4.0: Enhanced with sophisticated item extraction patterns and better classification handling.    Key Functions:
    1. Convert legacy extractor outputs to standardized format
    2. Enhanced contact information field conversion (address, phone, email, website)
    3. Provide backward compatibility for existing APIs
    4. Validate field structures and naming
    5. Support for contextual address extraction field conversion
    6. Generate migration reports
    
    Version 1.2.0 Enhancements:
    - Enhanced address field conversion for contextual extraction results
    - Improved contact information field mapping and validation
    - Added support for multiple address data structure formats
    """
    
    def __init__(self):
        self.conversion_stats = {
            'fields_migrated': 0,
            'fields_standardized': 0,
            'deprecated_fields_found': [],
            'validation_errors': [],
            'validation_warnings': []
        }
    
    def standardize_extractor_output(self, raw_output: Dict[str, Any], 
                                   extractor_name: str) -> Dict[str, Any]:
        """
        Convert raw extractor output to standardized format.
        
        Args:
            raw_output: Original extractor output with legacy field names
            extractor_name: Name of the extractor for metadata tracking
            
        Returns:
            Standardized output with consistent field names and structures
        """
        
        print(f"🔄 Standardizing output from {extractor_name}...")
        
        standardized = {
            'extraction_metadata': {
                'extractor_name': extractor_name,
                'standardization_timestamp': self._get_timestamp(),
                'original_field_count': len(raw_output),
                'migration_applied': True
            }
        }
        
        # Process each section of the raw output
        standardized.update(self._standardize_totals_section(raw_output))
        standardized.update(self._standardize_supplier_section(raw_output))
        standardized.update(self._standardize_receipt_id_section(raw_output))
        standardized.update(self._standardize_discount_section(raw_output))
        standardized.update(self._standardize_vat_section(raw_output))
        standardized.update(self._standardize_payment_section(raw_output))
        standardized.update(self._standardize_item_section(raw_output))
        
        # Validate standardized output
        validation_results = validate_standardized_output(standardized)
        standardized['extraction_metadata']['validation_results'] = validation_results
        
        # Update conversion stats
        self._update_conversion_stats(validation_results, len(standardized))
        
        print(f"   ✅ Standardized {len(standardized)} fields")
        if validation_results['deprecated_fields']:
            print(f"   ⚠️  Found {len(validation_results['deprecated_fields'])} deprecated fields")
        
        # Convert field objects to dictionaries for JSON serialization
        standardized = self._convert_field_objects_to_dicts(standardized)
        
        return standardized
    
    def _convert_field_objects_to_dicts(self, data: Any) -> Any:
        """
        Recursively convert AmountField, TextField, and DateField objects to dictionaries
        for JSON serialization compatibility.
        """
        from .standardized_schema import AmountField, TextField, DateField
        
        if isinstance(data, (AmountField, TextField, DateField)):
            return data.to_dict()
        elif isinstance(data, dict):
            return {key: self._convert_field_objects_to_dicts(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._convert_field_objects_to_dicts(item) for item in data]
        else:
            return data
    
    def _standardize_totals_section(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert totals-related fields to standardized format."""
        
        totals_section = {}
        
        # Handle legacy totals structure from comprehensive_receipt_extractor
        if 'totals' in raw_output:
            totals_data = raw_output['totals']
            
            # Map legacy total -> final_total
            if 'total' in totals_data and totals_data['total']:
                totals_section['final_total'] = self._convert_to_amount_field(
                    totals_data['total'], 'final_total'
                )
            
            # Map legacy subtotal -> items_subtotal  
            if 'subtotal' in totals_data and totals_data['subtotal']:
                totals_section['items_subtotal'] = self._convert_to_amount_field(
                    totals_data['subtotal'], 'items_subtotal'
                )
                
            # Map legacy net_after_discount -> final_total (if not already set)
            if 'net_after_discount' in totals_data and totals_data['net_after_discount']:
                if 'final_total' not in totals_section:
                    totals_section['final_total'] = self._convert_to_amount_field(
                        totals_data['net_after_discount'], 'final_total'
                    )
        
        # Handle direct total fields in root
        direct_total_mappings = {
            'total_amount': 'final_total',
            'subtotal_amount': 'items_subtotal', 
            'grand_total': 'final_total',
            'net_total': 'final_total'
        }
        
        for old_field, new_field in direct_total_mappings.items():
            if old_field in raw_output and raw_output[old_field] is not None:
                # Only set if not already present (avoid overwriting)
                if new_field not in totals_section:
                    totals_section[new_field] = self._convert_simple_amount_to_field(
                        raw_output[old_field], old_field, new_field
                    )
        
        return totals_section
    
    def _standardize_supplier_section(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert supplier-related fields to standardized format."""
        
        supplier_section = {}
        
        # Handle supplier name mappings
        supplier_name_fields = ['supplier_name', 'supplier', 'merchant', 'business_name']
        for field in supplier_name_fields:
            if field in raw_output and raw_output[field]:
                supplier_section['supplier_name'] = self._convert_to_text_field(
                    raw_output[field], 'supplier_name'
                )
                break
        
        # Handle supplier contact details
        if 'contact_details' in raw_output:
            contact_data = raw_output['contact_details']
            
            if 'phone_numbers' in contact_data:
                supplier_section['supplier_phone'] = [
                    self._convert_to_text_field(phone, 'supplier_phone') 
                    for phone in contact_data['phone_numbers']
                ]
            
            if 'email_addresses' in contact_data:
                supplier_section['supplier_email'] = [
                    self._convert_to_text_field(email, 'supplier_email')
                    for email in contact_data['email_addresses']  
                ]
            
            if 'addresses' in contact_data:
                supplier_section['supplier_address'] = [
                    self._convert_to_text_field(addr, 'supplier_address')
                    for addr in contact_data['addresses']
                ]
            
            if 'websites' in contact_data:
                supplier_section['supplier_website'] = [
                    self._convert_to_text_field(web, 'supplier_website')
                    for web in contact_data['websites']
                ]
        
        # Handle VAT number
        vat_fields = ['vat_number', 'supplier_vat_number', 'tax_id']
        for field in vat_fields:
            if field in raw_output and raw_output[field]:
                supplier_section['supplier_vat_number'] = self._convert_to_text_field(
                    raw_output[field], 'supplier_vat_number'
                )
                break
        
        return supplier_section
    
    def _standardize_receipt_id_section(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert receipt identification fields."""
        
        receipt_id_section = {}
        
        # Date handling
        date_fields = ['receipt_date', 'date', 'transaction_date']
        for field in date_fields:
            if field in raw_output and raw_output[field]:
                receipt_id_section['receipt_date'] = self._convert_to_date_field(
                    raw_output[field], 'receipt_date'
                )
                break
        
        # Receipt number handling
        receipt_num_fields = ['receipt_number', 'receipt_id', 'transaction_id', 'invoice_number']
        for field in receipt_num_fields:
            if field in raw_output and raw_output[field]:
                receipt_id_section['receipt_number'] = self._convert_to_text_field(
                    raw_output[field], 'receipt_number'
                )
                break
        
        return receipt_id_section
    
    def _standardize_discount_section(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert discount-related fields."""
        
        discount_section = {}
        
        if 'discount_details' in raw_output:
            discount_data = raw_output['discount_details']
            
            # Convert discount items
            if 'discounts' in discount_data:
                discount_section['discount_items'] = discount_data['discounts']
            
            # Convert coupon items  
            if 'coupons' in discount_data:
                discount_section['coupon_items'] = discount_data['coupons']
            
            # Convert savings to loyalty_savings
            if 'savings' in discount_data:
                discount_section['loyalty_savings'] = discount_data['savings']
        
        # Handle direct discount fields
        if 'discount' in raw_output:
            if isinstance(raw_output['discount'], list):
                discount_section['discount_items'] = raw_output['discount']
            elif raw_output['discount']:
                discount_section['discount_items'] = [raw_output['discount']]
        
        return discount_section
    
    def _standardize_vat_section(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert VAT-related fields with enhanced VAT details support."""
        
        vat_section = {}
        
        # Handle comprehensive VAT details from comprehensive integrated extractor
        if 'vat_details' in raw_output:
            vat_details = raw_output['vat_details']
            
            # Pass through the complete VAT details structure for web interface
            if isinstance(vat_details, dict):
                # Create vat_information section for web interface display
                vat_section['vat_information'] = {
                    'vat_data_entries': vat_details.get('vat_data_entries', []),
                    'vat_headers': vat_details.get('vat_headers', []),
                    'vat_summary': vat_details.get('vat_summary', {}),
                    'extraction_metadata': vat_details.get('extraction_metadata', {})
                }
                
                # Also maintain original structure
                vat_section['vat_details'] = vat_details
        
        if 'vat_data' in raw_output:
            vat_data = raw_output['vat_data']
            
            # Convert VAT items
            if 'vat_details' in vat_data:
                vat_section['vat_items'] = vat_data['vat_details']
            
            # Convert total VAT
            if 'total_vat' in vat_data:
                vat_section['vat_total'] = self._convert_to_amount_field(
                    vat_data['total_vat'], 'vat_total'
                )
        
        # Handle direct VAT fields
        vat_amount_fields = ['vat_amount', 'vat', 'tax', 'tax_amount']
        for field in vat_amount_fields:
            if field in raw_output and raw_output[field] is not None:
                if 'vat_total' not in vat_section:
                    vat_section['vat_total'] = self._convert_simple_amount_to_field(
                        raw_output[field], field, 'vat_total'
                    )
                break
        
        return vat_section
    
    def _standardize_payment_section(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert payment-related fields."""
        
        payment_section = {}
        
        if 'payment_details' in raw_output:
            payment_data = raw_output['payment_details']
            
            # Convert payment methods
            if 'payment_methods' in payment_data:
                payment_section['payment_methods'] = payment_data['payment_methods']
            
            # Convert payment amounts
            if 'payment_amounts' in payment_data:
                payment_section['payment_amounts'] = [
                    self._convert_to_amount_field(amt, 'payment_amounts')
                    for amt in payment_data['payment_amounts']
                ]
            
            # Convert change details
            if 'change_details' in payment_data:
                change_details = payment_data['change_details']
                if change_details:
                    payment_section['change_given'] = self._convert_to_amount_field(
                        change_details[0] if isinstance(change_details, list) else change_details,
                        'change_given'
                    )
            
            # Convert card details
            if 'card_details' in payment_data:
                payment_section['card_details'] = payment_data['card_details']
        
        return payment_section
    
    def _standardize_item_section(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Convert item-related fields."""
        
        item_section = {}
        
        # Handle item_details structure from comprehensive extractor (v1.3.0)
        if 'item_details' in raw_output:
            item_details = raw_output['item_details']
            if isinstance(item_details, dict):
                # Pass through the complete item_details structure
                item_section['item_details'] = item_details
                
                # Extract item_data for count and backward compatibility
                item_data = item_details.get('item_data', [])
                item_section['item_count'] = len(item_data) if item_data else 0
                
                # Legacy compatibility
                if item_data:
                    item_section['item_list'] = item_data
        
        # Handle legacy 'items' field
        elif 'items' in raw_output:
            item_section['item_list'] = raw_output['items']
            item_section['item_count'] = len(raw_output['items']) if raw_output['items'] else 0
        
        return item_section
    
    def _convert_to_amount_field(self, value: Any, field_type: str) -> AmountField:
        """Convert various amount formats to standardized AmountField."""
        
        if isinstance(value, dict):
            return create_amount_field(
                amount=value.get('amount'),
                raw_text=value.get('raw_text', ''),
                confidence=value.get('confidence', 0.0),
                line_number=value.get('line_number', 0),
                extraction_method=value.get('extraction_method', 'legacy_conversion'),
                field_type=field_type,
                pattern_used=value.get('pattern_used')
            )
        else:
            # Simple numeric value
            return create_amount_field(
                amount=float(value) if value is not None else None,
                raw_text=str(value) if value is not None else '',
                confidence=0.8,  # Default confidence for simple conversions
                line_number=0,
                extraction_method='legacy_conversion',
                field_type=field_type
            )
    
    def _convert_simple_amount_to_field(self, value: Any, old_field: str, new_field: str) -> AmountField:
        """Convert simple amount value to AmountField."""
        
        return create_amount_field(
            amount=float(value) if value is not None else None,
            raw_text=str(value) if value is not None else '',
            confidence=0.7,  # Lower confidence for simple conversions  
            line_number=0,
            extraction_method=f'migrated_from_{old_field}',
            field_type=new_field
        )
    
    def _convert_to_text_field(self, value: Any, field_type: str) -> TextField:
        """Convert various text formats to standardized TextField."""
        
        if isinstance(value, dict):
            # Enhanced handling for different field structures
            main_value = (
                value.get('value') or 
                value.get('address') or 
                value.get('full_address') or
                value.get('phone_number') or
                value.get('email') or
                value.get('website') or
                value.get('fax_number') or
                value.get('raw_text', '')
            )
            
            return create_text_field(
                value=main_value,
                raw_text=value.get('raw_text', main_value),
                confidence=value.get('confidence', 0.0),
                line_number=value.get('line_number', 0),
                extraction_method=value.get('extraction_method', 'supplier_context_aware'),
                field_type=field_type,
                pattern_used=value.get('pattern_used')
            )
        else:
            # Simple string value
            return create_text_field(
                value=str(value) if value is not None else '',
                raw_text=str(value) if value is not None else '',
                confidence=0.8,  # Default confidence
                line_number=0,
                extraction_method='legacy_conversion',
                field_type=field_type
            )
    
    def _convert_to_date_field(self, value: Any, field_type: str) -> DateField:
        """Convert date formats to standardized DateField."""
        
        if isinstance(value, dict):
            return DateField(
                date=value.get('date', value.get('value')),
                raw_text=value.get('raw_text', ''),
                confidence=value.get('confidence', 0.0),
                line_number=value.get('line_number', 0),
                extraction_method=value.get('extraction_method', 'legacy_conversion'),
                original_format=value.get('original_format'),
                parsed_components=value.get('parsed_components')
            )
        else:
            # Simple date string
            return DateField(
                date=str(value) if value is not None else None,
                raw_text=str(value) if value is not None else '',
                confidence=0.8,
                line_number=0,
                extraction_method='legacy_conversion'
            )
    
    def generate_backward_compatible_output(self, standardized_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate backward-compatible output from standardized data.
        Ensures existing APIs continue to work during migration.
        """
        
        print("🔄 Generating backward-compatible output...")
        
        compatible_output = copy.deepcopy(standardized_data)
        
        # Add legacy field aliases
        legacy_mappings = {
            'final_total': ['total', 'total_amount', 'net_after_discount'],
            'items_subtotal': ['subtotal', 'subtotal_amount'],
            'vat_total': ['vat_amount', 'vat'],
            'supplier_name': ['supplier', 'merchant'],
            'receipt_date': ['date'],
            'receipt_number': ['receipt_id']
        }
        
        for new_field, old_fields in legacy_mappings.items():
            if new_field in standardized_data:
                field_data = standardized_data[new_field]
                for old_field in old_fields:
                    compatible_output[old_field] = field_data
        
        print(f"   ✅ Added {sum(len(aliases) for aliases in legacy_mappings.values())} legacy aliases")
        
        return compatible_output
    
    def _update_conversion_stats(self, validation_results: Dict[str, Any], field_count: int):
        """Update internal conversion statistics."""
        
        self.conversion_stats['fields_standardized'] += field_count
        self.conversion_stats['fields_migrated'] += len(validation_results.get('deprecated_fields', []))
        self.conversion_stats['deprecated_fields_found'].extend(validation_results.get('deprecated_fields', []))
        self.conversion_stats['validation_errors'].extend(validation_results.get('errors', []))
        self.conversion_stats['validation_warnings'].extend(validation_results.get('warnings', []))
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for metadata."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_conversion_report(self) -> Dict[str, Any]:
        """Generate a report of the field standardization process."""
        
        return {
            'conversion_summary': self.conversion_stats,
            'migration_status': 'in_progress' if self.conversion_stats['deprecated_fields_found'] else 'complete',
            'recommendations': self._generate_migration_recommendations()
        }
    
    def _generate_migration_recommendations(self) -> List[str]:
        """Generate recommendations for completing the migration."""
        
        recommendations = []
        
        if self.conversion_stats['deprecated_fields_found']:
            recommendations.append(
                "Update extractor outputs to use standardized field names"
            )
        
        if self.conversion_stats['validation_errors']:
            recommendations.append(
                "Fix validation errors in field structures"
            )
        
        if self.conversion_stats['fields_migrated'] > 0:
            recommendations.append(
                "Consider updating client code to use new field names"
            )
        
        return recommendations
