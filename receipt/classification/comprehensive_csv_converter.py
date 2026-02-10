"""
Comprehensive CSV Converter for Line Classification Results

Converts all line classification JSON files to CSV format while preserving
the folder structure (receipts, invoices, others with confidence levels).
"""

import json
import sys
from pathlib import Path
from typing import List

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config_manager import OCRConfig

# ImprovedCSVConverter import - commented out since module is not available
# from app.services.line_classifier_service.improved_csv_converter import ImprovedCSVConverter


class ComprehensiveCSVConverter:
    """Converter that processes all line classification results maintaining folder structure."""
    
    def __init__(self, config: OCRConfig = None):
        """Initialize the comprehensive converter."""
        self.config = config or OCRConfig()
        
        # Setup directories
        self.input_base_dir = Path(self.config.line_output_base_dir)
        
        # Statistics
        self.global_stats = {
            'total_files_processed': 0,
            'total_csv_files_created': 0,
            'total_rows_generated': 0,
            'document_types_processed': {},
            'errors': []
        }
    
    def process_all_folders(self):
        """Process all document type folders while preserving structure."""
        print("🚀 Comprehensive CSV Conversion for Line Classification Results")
        print("=" * 80)
        
        print(f"📁 Input directory: {self.input_base_dir}")
        
        # Process each document type that was configured
        document_types = []
        if self.config.process_receipts:
            document_types.append(('receipts', self.config.receipts_dir))
        if self.config.process_invoices:
            document_types.append(('invoices', self.config.invoices_dir))
        if self.config.process_others:
            document_types.append(('others', self.config.others_dir))
        
        print(f"🎯 Processing document types: {[dt[0] for dt in document_types]}")
        
        for doc_type_name, doc_type_dir in document_types:
            print(f"\n📋 Processing {doc_type_name.upper()}...")
            self._process_document_type(doc_type_name, doc_type_dir)
        
        # Create global combined CSV
        self._create_global_combined_csv()
        
        # Print final summary
        self._print_global_summary()
    
    def _process_document_type(self, doc_type_name: str, doc_type_dir: str):
        """Process all confidence levels for a document type."""
        input_doc_dir = self.input_base_dir / doc_type_dir
        
        if not input_doc_dir.exists():
            print(f"  ⚠️  Directory not found: {input_doc_dir}")
            return
        
        # Initialize stats for this document type
        doc_stats = {
            'files_processed': 0,
            'csv_files_created': 0,
            'total_rows': 0,
            'confidence_levels': {}
        }
        
        # Process each confidence level
        for confidence_level in self.config.confidence_levels:
            confidence_dir = input_doc_dir / confidence_level
            
            if not confidence_dir.exists():
                print(f"    ⚠️  Confidence directory not found: {confidence_dir}")
                continue
            
            # Get JSON files in this confidence level
            json_files = list(confidence_dir.glob("*_line_classified.json"))
            
            if not json_files:
                print(f"    📊 {confidence_level}: No JSON files found")
                continue
            
            print(f"    📊 {confidence_level}: {len(json_files)} JSON files")
            
            # Create converter for this specific directory
            converter = ImprovedCSVConverter(
                input_dir=str(confidence_dir),
                output_dir=str(confidence_dir)  # Output to same directory
            )
            
            # Process files in this confidence level
            success_count = 0
            total_rows = 0
            
            for json_file in json_files:
                csv_rows = converter.process_json_file(json_file)
                
                if csv_rows:
                    # Create CSV file in same directory
                    csv_filename = json_file.stem.replace('_line_classified', '_ml_training') + '.csv'
                    csv_file = confidence_dir / csv_filename
                    
                    # Write CSV
                    import csv
                    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=converter.csv_headers)
                        writer.writeheader()
                        writer.writerows(csv_rows)
                    
                    success_count += 1
                    total_rows += len(csv_rows)
                    
                    if len(json_files) <= 10:  # Only show individual files for small batches
                        print(f"      ✅ {json_file.name} → {csv_filename} ({len(csv_rows)} rows)")
            
            # Update confidence level stats
            confidence_stats = {
                'json_files_found': len(json_files),
                'csv_files_created': success_count,
                'total_rows': total_rows
            }
            doc_stats['confidence_levels'][confidence_level] = confidence_stats
            doc_stats['files_processed'] += len(json_files)
            doc_stats['csv_files_created'] += success_count
            doc_stats['total_rows'] += total_rows
            
            print(f"      📈 Created {success_count}/{len(json_files)} CSV files ({total_rows:,} rows)")
        
        # Store document type stats
        self.global_stats['document_types_processed'][doc_type_name] = doc_stats
        self.global_stats['total_files_processed'] += doc_stats['files_processed']
        self.global_stats['total_csv_files_created'] += doc_stats['csv_files_created']
        self.global_stats['total_rows_generated'] += doc_stats['total_rows']
        
        # Print document type summary
        print(f"  📈 {doc_type_name.upper()} Summary: {doc_stats['csv_files_created']}/{doc_stats['files_processed']} CSV files created ({doc_stats['total_rows']:,} rows)")
    
    def _create_global_combined_csv(self):
        """Create a global combined CSV with all data."""
        print(f"\n🔄 Creating global combined CSV...")
        
        combined_csv = self.input_base_dir / "all_line_classifications_ml_data.csv"
        all_rows = []
        
        # Import the converter to get headers
        from line_classifier_service.improved_csv_converter import ImprovedCSVConverter
        temp_converter = ImprovedCSVConverter(".", ".")
        
        # Collect all JSON files
        all_json_files = list(self.input_base_dir.rglob("*_line_classified.json"))
        print(f"  📊 Found {len(all_json_files)} total JSON files")
        
        for json_file in all_json_files:
            try:
                csv_rows = temp_converter.process_json_file(json_file)
                
                # Add source information to each row
                for row in csv_rows:
                    # Extract document type and confidence level from path
                    path_parts = json_file.relative_to(self.input_base_dir).parts
                    if len(path_parts) >= 2:
                        row['document_type'] = path_parts[0]  # receipts, invoices, others
                        row['confidence_level'] = path_parts[1]  # best_confidence, medium_confidence, etc.
                    row['source_file'] = json_file.stem.replace('_line_classified', '')
                
                all_rows.extend(csv_rows)
                
            except Exception as e:
                error_msg = f"Error processing {json_file}: {e}"
                self.global_stats['errors'].append(error_msg)
                print(f"    ❌ {error_msg}")
        
        # Update headers to include source information
        extended_headers = ['document_type', 'confidence_level', 'source_file'] + temp_converter.csv_headers
        
        # Write combined CSV
        import csv
        with open(combined_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=extended_headers)
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"  ✅ Global combined CSV created: {combined_csv}")
        print(f"  📊 Total rows: {len(all_rows):,}")
        print(f"  📊 Total columns: {len(extended_headers)}")
        
        return len(all_rows)
    
    def _print_global_summary(self):
        """Print comprehensive summary."""
        print(f"\n🎯 COMPREHENSIVE CSV CONVERSION RESULTS")
        print("=" * 80)
        
        print(f"\n📊 Processing Summary:")
        print(f"   JSON files processed: {self.global_stats['total_files_processed']:,}")
        print(f"   CSV files created: {self.global_stats['total_csv_files_created']:,}")
        print(f"   Total rows generated: {self.global_stats['total_rows_generated']:,}")
        
        print(f"\n📋 Document Types Processed:")
        for doc_type, stats in self.global_stats['document_types_processed'].items():
            print(f"   {doc_type.upper():12s}: {stats['csv_files_created']:6,} CSV files, {stats['total_rows']:8,} rows")
            
            # Show confidence level breakdown
            for conf_level, conf_stats in stats['confidence_levels'].items():
                if conf_stats['csv_files_created'] > 0:
                    print(f"     {conf_level:20s}: {conf_stats['csv_files_created']:3,} files, {conf_stats['total_rows']:6,} rows")
        
        if self.global_stats['errors']:
            print(f"\n❌ Errors ({len(self.global_stats['errors'])}):")
            for error in self.global_stats['errors'][:5]:  # Show first 5 errors
                print(f"   {error}")
            if len(self.global_stats['errors']) > 5:
                print(f"   ... and {len(self.global_stats['errors']) - 5} more errors")
        else:
            print(f"\n✅ No errors encountered!")
        
        print(f"\n📁 CSV files saved alongside JSON files in: {self.input_base_dir}")
        print(f"📄 Global combined CSV: {self.input_base_dir}/all_line_classifications_ml_data.csv")


def main():
    """Main function to run the comprehensive CSV converter."""
    try:
        # Load configuration
        config = OCRConfig()
        print(f"✅ Configuration loaded from: {config.config_path}")
        
        # Create and run converter
        converter = ComprehensiveCSVConverter(config)
        converter.process_all_folders()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
