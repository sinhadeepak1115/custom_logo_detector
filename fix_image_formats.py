"""
Fix unsupported image formats in dataset.
Converts JPEG XL, WebP, GIF to standard JPEG/PNG formats.
"""

import os
from pathlib import Path
from PIL import Image
import shutil

def convert_image(input_path, output_path=None, format='JPEG'):
    """
    Convert image to standard format.
    
    Args:
        input_path: Path to input image
        output_path: Path to save converted image (if None, overwrites)
        format: Output format ('JPEG' or 'PNG')
    """
    try:
        # Try to open with PIL
        img = Image.open(input_path)
        
        # Convert RGBA to RGB if needed (for JPEG)
        if format == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
            img = background
        elif img.mode not in ('RGB', 'L'):
            img = img.convert('RGB')
        
        # Determine output path
        if output_path is None:
            output_path = input_path
        
        # Save converted image
        img.save(output_path, format=format, quality=95)
        print(f"Converted: {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False


def fix_dataset_images(dataset_dir):
    """
    Fix unsupported image formats in dataset directory.
    
    Args:
        dataset_dir: Directory containing images (e.g., 'data/images/test')
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Directory not found: {dataset_dir}")
        return
    
    # Supported extensions
    supported_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Find all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']:
        image_files.extend(dataset_path.glob(f'*{ext}'))
        image_files.extend(dataset_path.glob(f'*{ext.upper()}'))
    
    converted_count = 0
    skipped_count = 0
    
    for img_path in image_files:
        # Check if file needs conversion
        ext = img_path.suffix.lower()
        
        if ext in supported_exts:
            # Check if it's actually readable
            try:
                img = Image.open(img_path)
                img.verify()
                skipped_count += 1
                continue
            except:
                # File exists but can't be read - needs conversion
                pass
        
        # Determine output format based on original extension
        if ext in ['.jpg', '.jpeg', '.webp']:
            output_format = 'JPEG'
            new_ext = '.jpg'
        else:
            output_format = 'PNG'
            new_ext = '.png'
        
        # Create backup and convert
        backup_path = img_path.with_suffix(img_path.suffix + '.backup')
        
        # Only backup if not already backed up
        if not backup_path.exists():
            shutil.copy2(img_path, backup_path)
        
        # Convert
        new_path = img_path.with_suffix(new_ext)
        if convert_image(img_path, new_path, output_format):
            # Remove old file if extension changed
            if img_path != new_path:
                img_path.unlink()
            converted_count += 1
    
    print(f"\nSummary:")
    print(f"  Converted: {converted_count} images")
    print(f"  Skipped (already OK): {skipped_count} images")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fix unsupported image formats in dataset')
    parser.add_argument('--dir', type=str, required=True,
                        help='Directory containing images to fix')
    
    args = parser.parse_args()
    
    fix_dataset_images(args.dir)
