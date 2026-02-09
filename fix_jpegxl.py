"""
Fix JPEG XL files by converting them using alternative methods.
"""

import os
import subprocess
from pathlib import Path

def convert_jpegxl_using_imagemagick(input_path, output_path):
    """Convert JPEG XL using ImageMagick."""
    try:
        result = subprocess.run(
            ['convert', input_path, output_path],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"Converted using ImageMagick: {input_path} -> {output_path}")
            return True
    except FileNotFoundError:
        print("ImageMagick not found. Trying alternative method...")
    except Exception as e:
        print(f"ImageMagick error: {e}")
    return False

def fix_jpegxl_file(file_path):
    """Fix a JPEG XL file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    # Try ImageMagick first
    output_path = file_path.with_suffix('.jpg')
    if convert_jpegxl_using_imagemagick(str(file_path), str(output_path)):
        # Remove original
        file_path.unlink()
        return True
    
    # If ImageMagick fails, try to download/replace or skip
    print(f"Could not convert {file_path}")
    print("Options:")
    print("1. Install ImageMagick: brew install imagemagick")
    print("2. Remove this file from test set")
    print("3. Replace with original image")
    
    return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        fix_jpegxl_file(sys.argv[1])
    else:
        # Fix the problematic file
        fix_jpegxl_file('data/images/test/9c102138-21_.jpg')
