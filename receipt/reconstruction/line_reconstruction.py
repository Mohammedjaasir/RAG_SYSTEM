#!/usr/bin/env python3
"""
Line reconstruction using the line_reconstructor algorithm with DocTR text format.

This script processes OCR results from categorized confidence folders and applies
advanced line reconstruction using vertical overlap detection and intelligent grouping.
It works with DocTR format output where each line contains: x0 y0 x1 y1 x2 y2 x3 y3 word confidence
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config_manager import OCRConfig
from ..support.line_reconstruction_service.sort_ocr_text import sort_ocr_text


def load_ocr_data(file_path: Path) -> Optional[str]:
    """Load DocTR OCR data and return reconstructed text directly."""
    try:
        # Use the original sort_ocr_text function directly on the file path
        return sort_ocr_text(str(file_path), line_tolerance=0.5)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def process_single_file(
    txt_file: Path,
    config: OCRConfig,
    output_dir: Path,
    category: str
) -> Dict[str, Any]:
    """
    Process a single DocTR OCR text file using line reconstruction algorithm.
    
    Args:
        txt_file: Path to the input text file (DocTR format)
        config: Configuration manager instance
        output_dir: Output directory for results
        category: Confidence category (best/medium/low)
    
    Returns:
        Dictionary with processing statistics
    """
    # Create category-specific output directory
    category_output_dir = output_dir / category
    category_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filename without extension
    base_name = txt_file.stem
    
    try:
        # Apply line reconstruction using the original sort_ocr_text function
        reconstructed_text = sort_ocr_text(
            str(txt_file),
            line_tolerance=config.line_tolerance,
            debug=config.debug_output
        )
        
        if not reconstructed_text:
            return {"success": False, "error": "No text reconstructed"}
        
        # Count lines
        lines = reconstructed_text.strip().split('\n') if reconstructed_text.strip() else []
        line_count = len(lines)
        
        # Prepare output data
        output_data = {
            "source_file": str(txt_file),
            "category": category,
            "algorithm": "line_reconstructor",
            "parameters": {
                "line_tolerance": config.line_tolerance,
                "debug_output": config.debug_output
            },
            "statistics": {
                "total_lines": line_count,
                "non_empty_lines": len([line for line in lines if line.strip()])
            },
            "reconstructed_text": reconstructed_text,
            "lines": lines
        }
        
        # Write JSON output
        json_output_path = category_output_dir / f"{base_name}.json"
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Write plain text output
        txt_output_path = category_output_dir / f"{base_name}.txt"
        with open(txt_output_path, 'w', encoding='utf-8') as f:
            f.write(reconstructed_text)
        
        if config.debug_output:
            print(f"✓ Processed {txt_file.name}: {line_count} lines → {category_output_dir}")
        
        return {
            "success": True,
            "line_count": line_count,
            "output_json": str(json_output_path),
            "output_txt": str(txt_output_path)
        }
        
    except Exception as e:
        error_msg = f"Error processing {txt_file}: {e}"
        print(f"✗ {error_msg}")
        return {"success": False, "error": error_msg}


def process_category_folder(
    category_dir: Path,
    config: OCRConfig,
    output_dir: Path,
    category: str
) -> Dict[str, Any]:
    """
    Process all text files in a category folder.
    
    Args:
        category_dir: Path to the category directory
        config: Configuration manager instance
        output_dir: Output directory for results
        category: Category name
    
    Returns:
        Dictionary with processing statistics for the category
    """
    txt_files = list(category_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"No text files found in {category_dir}")
        return {"category": category, "files_processed": 0, "files_failed": 0, "total_lines": 0, "files": []}
    
    print(f"\nProcessing {category} confidence category ({len(txt_files)} files)...")
    
    category_stats = {
        "category": category,
        "files_processed": 0,
        "files_failed": 0,
        "total_lines": 0,
        "files": []
    }
    
    for txt_file in txt_files:
        result = process_single_file(txt_file, config, output_dir, category)
        
        file_info = {
            "filename": txt_file.name,
            "success": result["success"]
        }
        
        if result["success"]:
            category_stats["files_processed"] += 1
            category_stats["total_lines"] += result["line_count"]
            file_info.update({
                "line_count": result["line_count"],
                "output_json": result["output_json"],
                "output_txt": result["output_txt"]
            })
        else:
            category_stats["files_failed"] += 1
            file_info["error"] = result.get("error", "Unknown error")
        
        category_stats["files"].append(file_info)
    
    return category_stats


def generate_summary_report(
    all_stats: List[Dict[str, Any]],
    config: OCRConfig,
    output_dir: Path
) -> None:
    """Generate summary report of line reconstruction process."""
    
    # Calculate overall statistics
    total_files = sum(stats["files_processed"] + stats["files_failed"] for stats in all_stats)
    total_processed = sum(stats["files_processed"] for stats in all_stats)
    total_failed = sum(stats["files_failed"] for stats in all_stats)
    total_lines = sum(stats["total_lines"] for stats in all_stats)
    
    summary = {
        "algorithm": "line_reconstructor",
        "timestamp": str(Path().cwd()),  # Could use datetime if needed
        "configuration": {
            "input_directory": config.categorized_ocr_dir,
            "output_directory": config.reconstructed_output_dir,
            "line_tolerance": config.line_tolerance,
            "debug_output": config.debug_output
        },
        "overall_statistics": {
            "total_files": total_files,
            "files_processed": total_processed,
            "files_failed": total_failed,
            "success_rate": f"{(total_processed/total_files*100):.1f}%" if total_files > 0 else "0%",
            "total_lines_reconstructed": total_lines
        },
        "category_statistics": all_stats
    }
    
    # Write summary report
    summary_path = output_dir / "line_reconstruction_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("LINE RECONSTRUCTION SUMMARY")
    print(f"{'='*60}")
    print(f"Algorithm: line_reconstructor")
    print(f"Total files: {total_files}")
    print(f"Successfully processed: {total_processed}")
    print(f"Failed: {total_failed}")
    print(f"Success rate: {summary['overall_statistics']['success_rate']}")
    print(f"Total lines reconstructed: {total_lines:,}")
    print(f"\nCategory breakdown:")
    for stats in all_stats:
        print(f"  {stats['category']}: {stats['files_processed']} files, {stats['total_lines']:,} lines")
    print(f"\nSummary report saved to: {summary_path}")


def main():
    """Main function to orchestrate the line reconstruction process."""
    print("Line Reconstruction using Line Algorithm (DocTR Format)")
    print("=" * 60)
    
    # Load configuration
    try:
        config = OCRConfig()
        print(f"✓ Configuration loaded from: {config.config_path}")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Setup paths
    base_input_dir = Path(config.categorized_ocr_dir)
    output_dir = Path(config.reconstructed_output_dir)
    
    # Validate input directory
    if not base_input_dir.exists():
        print(f"✗ Input directory not found: {base_input_dir}")
        sys.exit(1)
    
    print(f"✓ Input directory: {base_input_dir}")
    print(f"✓ Output directory: {output_dir}")
    print(f"✓ Line tolerance: {config.line_tolerance}")
    print(f"✓ Debug output: {config.debug_output}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each confidence category
    categories = ['best_confidence', 'medium_confidence', 'low_confidence']
    all_stats = []
    
    for category in categories:
        category_dir = base_input_dir / category
        
        if not category_dir.exists():
            print(f"⚠ Category directory not found: {category_dir}")
            continue
        
        stats = process_category_folder(category_dir, config, output_dir, category)
        all_stats.append(stats)
    
    # Generate summary report
    if all_stats:
        generate_summary_report(all_stats, config, output_dir)
    else:
        print("✗ No categories were processed successfully")
        sys.exit(1)


if __name__ == "__main__":
    main()
