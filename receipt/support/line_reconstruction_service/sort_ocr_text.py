#!/usr/bin/env python3
"""
Sort OCR text with bounding boxes in proper reading order (top-to-bottom, left-to-right).
Input format: x0 y0 x1 y1 x2 y2 x3 y3 : text
"""

import re
from typing import List, Tuple

class OCRTextBox:
    """Represents a text box from OCR output with bounding coordinates."""
    
    def __init__(self, line: str):
        """
        Parse a line in format: x0 y0 x1 y1 x2 y2 x3 y3 text
        """
        parts = line.strip().split()
        
        if len(parts) < 8:
            raise ValueError(f"Expected at least 8 coordinates, got {len(parts)}: {line.strip()}")
        
        # Parse coordinates as integers
        try:
            self.x0, self.y0, self.x1, self.y1 = map(int, parts[:4])
            self.x2, self.y2, self.x3, self.y3 = map(int, parts[4:8])
        except ValueError as e:
            raise ValueError(f"Invalid coordinates: {parts[:8]}") from e
        
        # Extract text (everything after the 8 coordinates, excluding confidence score)
        if len(parts) > 8:
            # The last part is typically the confidence score (float)
            # Try to identify and exclude it
            text_parts = parts[8:]
            if text_parts:
                # Check if the last part is a confidence score (float)
                try:
                    float(text_parts[-1])
                    # If it's a valid float, it's likely a confidence score - exclude it
                    self.text = ' '.join(text_parts[:-1])
                except ValueError:
                    # If it's not a float, keep all parts as text
                    self.text = ' '.join(text_parts)
            else:
                self.text = ""
        else:
            self.text = ""
        
        # Calculate bounding box properties for sorting
        self.left = min(self.x0, self.x1, self.x2, self.x3)
        self.right = max(self.x0, self.x1, self.x2, self.x3)
        self.top = min(self.y0, self.y1, self.y2, self.y3)
        self.bottom = max(self.y0, self.y1, self.y2, self.y3)
        
        # Center coordinates for sorting
        self.center_x = (self.left + self.right) / 2
        self.center_y = (self.top + self.bottom) / 2
        
        # Width and height
        self.width = self.right - self.left
        self.height = self.bottom - self.top
    
    def __repr__(self):
        return f"OCRTextBox('{self.text}', center=({self.center_x:.1f}, {self.center_y:.1f}))"
    
    def overlaps_vertically(self, other: 'OCRTextBox', tolerance: float = 0.5) -> bool:
        """Check if this box overlaps vertically with another box."""
        # Calculate overlap
        overlap_top = max(self.top, other.top)
        overlap_bottom = min(self.bottom, other.bottom)
        overlap_height = max(0, overlap_bottom - overlap_top)
        
        # Check if overlap is significant relative to both boxes
        min_height = min(self.height, other.height)
        return overlap_height >= tolerance * min_height

class OCRTextLine:
    """Represents a line of text composed of multiple OCR text boxes."""
    
    def __init__(self, first_box: OCRTextBox):
        self.boxes = [first_box]
        self.top = first_box.top
        self.bottom = first_box.bottom
        self.left = first_box.left
        self.right = first_box.right
    
    def can_add_box(self, box: OCRTextBox, tolerance: float = 0.5) -> bool:
        """Check if a box can be added to this line (vertically overlapping)."""
        return any(existing_box.overlaps_vertically(box, tolerance) for existing_box in self.boxes)
    
    def add_box(self, box: OCRTextBox):
        """Add a box to this line, maintaining left-to-right order."""
        # Find the correct position to insert the box
        insert_pos = 0
        for i, existing_box in enumerate(self.boxes):
            if box.center_x < existing_box.center_x:
                insert_pos = i
                break
            insert_pos = i + 1
        
        self.boxes.insert(insert_pos, box)
        
        # Update line boundaries
        self.top = min(self.top, box.top)
        self.bottom = max(self.bottom, box.bottom)
        self.left = min(self.left, box.left)
        self.right = max(self.right, box.right)
    
    def get_text(self, separator: str = " ") -> str:
        """Get the text of this line with boxes separated by the given separator."""
        return separator.join(box.text for box in self.boxes if box.text.strip())
    
    @property
    def center_y(self) -> float:
        """Get the vertical center of this line."""
        return (self.top + self.bottom) / 2

def sort_ocr_text(file_path: str, line_tolerance: float = 0.5, debug: bool = False) -> str:
    """
    Sort OCR text in proper reading order (top-to-bottom, left-to-right).
    
    Args:
        file_path: Path to the OCR text file
        line_tolerance: Tolerance for grouping boxes into lines (0.0 to 1.0)
        debug: Print debug information
    
    Returns:
        Sorted text with one line per row
    """
    # Read and parse all text boxes
    boxes = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                box = OCRTextBox(line)
                if box.text.strip():  # Only keep boxes with non-empty text
                    boxes.append(box)
            except ValueError as e:
                if debug:
                    print(f"Warning: Skipping line {line_num}: {e}")
                continue
    
    if not boxes:
        return ""
    
    if debug:
        print(f"Parsed {len(boxes)} text boxes")
    
    # Sort boxes by vertical position first (top to bottom)
    boxes.sort(key=lambda box: box.center_y)
    
    # Create lines using a more intelligent approach
    lines = []
    
    for box in boxes:
        # Find the best line to add this box to
        best_line = None
        best_score = float('inf')
        
        for line in lines:
            # Check if this box can reasonably belong to this line
            # Calculate vertical distance from box to line center
            vertical_distance = abs(box.center_y - line.center_y)
            
            # Calculate average height of boxes in the line
            avg_height = sum(b.height for b in line.boxes) / len(line.boxes)
            
            # A box can belong to a line if:
            # 1. The vertical distance is small relative to text height
            # 2. There's some vertical overlap or very close proximity
            max_allowed_distance = avg_height * line_tolerance
            
            if vertical_distance <= max_allowed_distance:
                # Use vertical distance as score (closer is better)
                score = vertical_distance
                if score < best_score:
                    best_score = score
                    best_line = line
        
        if best_line is not None:
            best_line.add_box(box)
        else:
            # Create a new line
            lines.append(OCRTextLine(box))
    
    # Sort lines by their vertical position
    lines.sort(key=lambda line: line.center_y)
    
    if debug:
        print(f"Grouped into {len(lines)} lines")
        for i, line in enumerate(lines):
            print(f"Line {i+1}: {line.get_text()}")
    
    # Generate final text
    return '\n'.join(line.get_text() for line in lines)

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sort OCR text in reading order")
    parser.add_argument("file", help="Path to OCR text file")
    parser.add_argument("--tolerance", type=float, default=0.5, 
                       help="Line grouping tolerance (0.0-1.0, default: 0.5)")
    parser.add_argument("--debug", action="store_true", 
                       help="Print debug information")
    parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    try:
        sorted_text = sort_ocr_text(args.file, args.tolerance, args.debug)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(sorted_text)
            print(f"Sorted text written to {args.output}")
        else:
            print(sorted_text)
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
