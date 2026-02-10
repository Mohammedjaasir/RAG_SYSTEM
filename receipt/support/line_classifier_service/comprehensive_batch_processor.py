"""
Comprehensive Line Classifier Batch Processor

This processor handles line classification for all document types while preserving
the folder structure from the classified documents.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config_manager import OCRConfig
from .advanced_rule_engine_complete import AdvancedRuleEngine


class ComprehensiveBatchProcessor:
    """Comprehensive batch processor for line classification."""
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """Initialize the processor with configuration."""
        self.config = config or OCRConfig()
        
        # Setup directories
        self.input_base_dir = Path(self.config.line_input_base_dir)
        self.output_base_dir = Path(self.config.line_output_base_dir)
        
        # Initialize classifier
        self.classifier = AdvancedRuleEngine()
        
        # Processing statistics
        self.global_stats = {
            "processing_started": datetime.now().isoformat(),
            "document_types_processed": {},
            "total_files": 0,
            "total_lines": 0,
            "successfully_processed": 0,
            "failed_files": 0,
            "processing_time": 0.0,
            "classification_counts": {},
            "confidence_distribution": {"very_high": 0, "high": 0, "medium": 0, "low": 0},
            "rule_usage": {},
            "errors": []
        }
    
    def process_all_documents(self):
        """Process all document types according to configuration."""
        print("🚀 Comprehensive Line Classification Processing")
        print("=" * 80)
        
        start_time = time.time()
        
        # Create output base directory
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each document type
        document_types = []
        if self.config.process_receipts:
            document_types.append(('receipts', self.config.receipts_dir))
        if self.config.process_invoices:
            document_types.append(('invoices', self.config.invoices_dir))
        if self.config.process_others:
            document_types.append(('others', self.config.others_dir))
        
        print(f"📁 Input directory: {self.input_base_dir}")
        print(f"📁 Output directory: {self.output_base_dir}")
        print(f"🎯 Processing document types: {[dt[0] for dt in document_types]}")
        
        for doc_type_name, doc_type_dir in document_types:
            print(f"\n📋 Processing {doc_type_name.upper()}...")
            self._process_document_type(doc_type_name, doc_type_dir)
        
        # Finalize processing
        end_time = time.time()
        self.global_stats["processing_time"] = round(end_time - start_time, 2)
        self.global_stats["processing_completed"] = datetime.now().isoformat()
        
        # Save final statistics
        self._save_global_statistics()
        self._print_final_summary()
    
    def _process_document_type(self, doc_type_name: str, doc_type_dir: str):
        """Process all files for a specific document type."""
        input_doc_dir = self.input_base_dir / doc_type_dir
        output_doc_dir = self.output_base_dir / doc_type_dir
        
        if not input_doc_dir.exists():
            print(f"  ⚠️  Directory not found: {input_doc_dir}")
            return
        
        # Initialize stats for this document type
        doc_stats = {
            "files_processed": 0,
            "files_failed": 0,
            "confidence_levels": {}
        }
        
        # Process each confidence level
        for confidence_level in self.config.confidence_levels:
            confidence_dir = input_doc_dir / confidence_level
            
            if not confidence_dir.exists():
                print(f"    ⚠️  Confidence directory not found: {confidence_dir}")
                continue
            
            # Create output confidence directory
            output_confidence_dir = output_doc_dir / confidence_level
            output_confidence_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all text files in this confidence level
            txt_files = list(confidence_dir.glob("*.txt"))
            print(f"    📊 {confidence_level}: {len(txt_files)} files")
            
            confidence_stats = {
                "files_found": len(txt_files),
                "files_processed": 0,
                "files_failed": 0
            }
            
            for txt_file in txt_files:
                try:
                    result = self._process_single_file(
                        txt_file, 
                        output_confidence_dir, 
                        doc_type_name, 
                        confidence_level
                    )
                    
                    if result['success']:
                        confidence_stats["files_processed"] += 1
                        doc_stats["files_processed"] += 1
                        self.global_stats["successfully_processed"] += 1
                        
                        # Update global statistics
                        self._update_global_stats(result['classification_results'])
                    else:
                        confidence_stats["files_failed"] += 1
                        doc_stats["files_failed"] += 1
                        self.global_stats["failed_files"] += 1
                        self.global_stats["errors"].append(f"{txt_file.name}: {result['error']}")
                
                except Exception as e:
                    error_msg = f"Unexpected error processing {txt_file.name}: {e}"
                    print(f"      ❌ {error_msg}")
                    confidence_stats["files_failed"] += 1
                    doc_stats["files_failed"] += 1
                    self.global_stats["failed_files"] += 1
                    self.global_stats["errors"].append(error_msg)
            
            doc_stats["confidence_levels"][confidence_level] = confidence_stats
            
            # Print confidence level summary
            if confidence_stats["files_processed"] > 0:
                print(f"      ✅ Processed: {confidence_stats['files_processed']}")
            if confidence_stats["files_failed"] > 0:
                print(f"      ❌ Failed: {confidence_stats['files_failed']}")
        
        # Store document type stats
        self.global_stats["document_types_processed"][doc_type_name] = doc_stats
        
        # Print document type summary
        total_processed = doc_stats["files_processed"]
        total_failed = doc_stats["files_failed"]
        print(f"  📈 {doc_type_name.upper()} Summary: {total_processed} processed, {total_failed} failed")
    
    def _process_single_file(self, input_file: Path, output_dir: Path, 
                           doc_type: str, confidence_level: str) -> Dict:
        """Process a single file and return results."""
        try:
            # Read the file
            with open(input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Clean lines
            lines = [line.rstrip('\n\r') for line in lines]
            
            if not lines:
                return {
                    'success': False,
                    'error': 'Empty file',
                    'classification_results': []
                }
            
            # Classify lines
            classification_results = self.classifier.classify_document(lines, input_file.name)
            
            # Create output data
            output_data = {
                "metadata": {
                    "source_file": input_file.name,
                    "document_type": doc_type,
                    "confidence_level": confidence_level,
                    "processed_at": datetime.now().isoformat(),
                    "total_lines": len(lines),
                    "non_empty_lines": len([line for line in lines if line.strip()]),
                    "classifier_type": "advanced_feature_based",
                    "source_path": str(input_file),
                },
                "classification_summary": self._build_classification_summary(classification_results),
                "line_classifications": []
            }
            
            # Add detailed line results
            for i, (line, result) in enumerate(zip(lines, classification_results)):
                line_data = {
                    "line_number": i + 1,
                    "original_text": line,
                    "cleaned_text": line.strip(),
                    "line_type": result.line_type.value,
                    "confidence": round(result.confidence, 3),
                    "primary_confidence": round(result.primary_confidence, 3),
                    "secondary_confidence": round(result.secondary_confidence, 3),
                    "evidence_score": round(result.evidence_score, 3),
                    "context_bonus": round(result.context_bonus, 3),
                    "rule_triggered": result.rule_triggered,
                    "rule_priority": result.rule_priority,
                    "reasons": result.reasons,
                    "features_used": result.features_used,
                    "is_empty": len(line.strip()) == 0,
                    "line_length": len(line),
                    "position_ratio": round(i / len(lines), 3) if len(lines) > 1 else 0.0
                }
                output_data["line_classifications"].append(line_data)
            
            # Save output file
            output_file = output_dir / f"{input_file.stem}_line_classified.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            return {
                'success': True,
                'output_file': str(output_file),
                'classification_results': classification_results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'classification_results': []
            }
    
    def _build_classification_summary(self, results: List) -> Dict:
        """Build comprehensive classification summary."""
        summary = {
            "type_counts": {},
            "confidence_stats": {
                "mean": 0.0,
                "primary_mean": 0.0,
                "evidence_mean": 0.0,
                "min": 1.0,
                "max": 0.0,
                "very_high_count": 0,  # >= 0.9
                "high_count": 0,       # 0.75-0.89
                "medium_count": 0,     # 0.6-0.74
                "low_count": 0         # < 0.6
            },
            "rule_analysis": {
                "rule_usage": {},
                "priority_distribution": {},
                "context_bonus_applied": 0
            },
            "advanced_metrics": {
                "avg_evidence_score": 0.0,
                "avg_context_bonus": 0.0,
                "high_evidence_lines": 0,  # evidence > 0.8
                "context_enhanced_lines": 0  # context_bonus > 0
            },
            "structure_analysis": {
                "has_header": False,
                "has_footer": False,
                "has_items": False,
                "has_totals": False,
                "has_vat": False,
                "has_item_headers": False
            }
        }
        
        if not results:
            return summary
        
        confidences = []
        primary_confidences = []
        evidence_scores = []
        context_bonuses = []
        
        for result in results:
            # Count types
            line_type = result.line_type.value
            summary["type_counts"][line_type] = summary["type_counts"].get(line_type, 0) + 1
            
            # Track confidence metrics
            conf = result.confidence
            confidences.append(conf)
            primary_confidences.append(result.primary_confidence)
            evidence_scores.append(result.evidence_score)
            context_bonuses.append(result.context_bonus)
            
            summary["confidence_stats"]["min"] = min(summary["confidence_stats"]["min"], conf)
            summary["confidence_stats"]["max"] = max(summary["confidence_stats"]["max"], conf)
            
            # Confidence distribution
            if conf >= 0.9:
                summary["confidence_stats"]["very_high_count"] += 1
            elif conf >= 0.75:
                summary["confidence_stats"]["high_count"] += 1
            elif conf >= 0.6:
                summary["confidence_stats"]["medium_count"] += 1
            else:
                summary["confidence_stats"]["low_count"] += 1
            
            # Rule analysis
            rule = result.rule_triggered
            summary["rule_analysis"]["rule_usage"][rule] = summary["rule_analysis"]["rule_usage"].get(rule, 0) + 1
            
            priority = result.rule_priority
            summary["rule_analysis"]["priority_distribution"][priority] = summary["rule_analysis"]["priority_distribution"].get(priority, 0) + 1
            
            if result.context_bonus > 0:
                summary["rule_analysis"]["context_bonus_applied"] += 1
            
            # Advanced metrics
            if result.evidence_score > 0.8:
                summary["advanced_metrics"]["high_evidence_lines"] += 1
            if result.context_bonus > 0:
                summary["advanced_metrics"]["context_enhanced_lines"] += 1
            
            # Structure analysis
            if line_type == "HEADER":
                summary["structure_analysis"]["has_header"] = True
            elif line_type == "FOOTER":
                summary["structure_analysis"]["has_footer"] = True
            elif line_type in ["ITEM_DATA"]:
                summary["structure_analysis"]["has_items"] = True
            elif line_type == "ITEM_HEADER":
                summary["structure_analysis"]["has_item_headers"] = True
            elif line_type == "SUMMARY_KEY_VALUE":
                summary["structure_analysis"]["has_totals"] = True
            elif line_type in ["VAT_DATA", "VAT_HEADER"]:
                summary["structure_analysis"]["has_vat"] = True
        
        # Calculate averages
        if confidences:
            summary["confidence_stats"]["mean"] = round(sum(confidences) / len(confidences), 3)
            summary["confidence_stats"]["primary_mean"] = round(sum(primary_confidences) / len(primary_confidences), 3)
            summary["confidence_stats"]["evidence_mean"] = round(sum(evidence_scores) / len(evidence_scores), 3)
            summary["advanced_metrics"]["avg_evidence_score"] = round(sum(evidence_scores) / len(evidence_scores), 3)
            summary["advanced_metrics"]["avg_context_bonus"] = round(sum(context_bonuses) / len(context_bonuses), 3)
        
        return summary
    
    def _update_global_stats(self, results: List):
        """Update global processing statistics."""
        self.global_stats["total_files"] += 1
        self.global_stats["total_lines"] += len(results)
        
        for result in results:
            # Count line types
            line_type = result.line_type.value
            self.global_stats["classification_counts"][line_type] = self.global_stats["classification_counts"].get(line_type, 0) + 1
            
            # Confidence distribution
            conf = result.confidence
            if conf >= 0.9:
                self.global_stats["confidence_distribution"]["very_high"] += 1
            elif conf >= 0.75:
                self.global_stats["confidence_distribution"]["high"] += 1
            elif conf >= 0.6:
                self.global_stats["confidence_distribution"]["medium"] += 1
            else:
                self.global_stats["confidence_distribution"]["low"] += 1
            
            # Rule usage
            rule = result.rule_triggered
            self.global_stats["rule_usage"][rule] = self.global_stats["rule_usage"].get(rule, 0) + 1
    
    def _save_global_statistics(self):
        """Save comprehensive global statistics."""
        stats_file = self.output_base_dir / self.config.line_stats_filename
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.global_stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Global statistics saved to: {stats_file}")
    
    def _print_final_summary(self):
        """Print comprehensive final summary."""
        print(f"\n🎯 LINE CLASSIFICATION RESULTS")
        print("=" * 80)
        
        print(f"\n📊 Processing Summary:")
        print(f"   Total files processed: {self.global_stats['successfully_processed']:,}")
        print(f"   Total lines classified: {self.global_stats['total_lines']:,}")
        print(f"   Failed files: {self.global_stats['failed_files']:,}")
        print(f"   Processing time: {self.global_stats['processing_time']:.2f} seconds")
        
        if self.global_stats['processing_time'] > 0:
            rate = self.global_stats['successfully_processed'] / self.global_stats['processing_time']
            print(f"   Processing rate: {rate:.1f} files/second")
        
        print(f"\n📋 Document Types Processed:")
        for doc_type, stats in self.global_stats["document_types_processed"].items():
            print(f"   {doc_type.upper():12s}: {stats['files_processed']:6,} files processed, {stats['files_failed']:6,} failed")
        
        print(f"\n🏷️ Line Classification Distribution:")
        total_lines = self.global_stats["total_lines"]
        for line_type, count in sorted(self.global_stats["classification_counts"].items()):
            percentage = (count / total_lines) * 100 if total_lines > 0 else 0
            print(f"   {line_type:20s}: {count:8,} lines ({percentage:5.1f}%)")
        
        print(f"\n🎯 Confidence Analysis:")
        conf_dist = self.global_stats["confidence_distribution"]
        for level, count in conf_dist.items():
            percentage = (count / total_lines) * 100 if total_lines > 0 else 0
            print(f"   {level.replace('_', ' ').title():12s}: {count:8,} lines ({percentage:5.1f}%)")
        
        if self.global_stats["errors"]:
            print(f"\n❌ Errors ({len(self.global_stats['errors'])}):")
            for error in self.global_stats["errors"][:10]:  # Show first 10 errors
                print(f"   {error}")
            if len(self.global_stats["errors"]) > 10:
                print(f"   ... and {len(self.global_stats['errors']) - 10} more errors")


def main():
    """Main function to run the comprehensive batch processor."""
    try:
        # Load configuration
        config = OCRConfig()
        print(f"✅ Configuration loaded from: {config.config_path}")
        
        # Create and run processor
        processor = ComprehensiveBatchProcessor(config)
        processor.process_all_documents()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
