"""
Quick start script to process existing Label Studio output.
This script adapts to the existing project structure.
"""

import os
import sys
import json
from convert_coco_to_yolo import convert_coco_to_yolo
from split_dataset import split_dataset


def fix_label_studio_paths(coco_json_path, images_dir):
    """
    Fix Label Studio paths in COCO JSON to point to actual image locations.
    
    Args:
        coco_json_path: Path to COCO JSON file
        images_dir: Directory containing actual images
    """
    # Load JSON
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    
    # Get list of actual image files
    actual_images = {os.path.basename(f): f for f in os.listdir(images_dir) 
                     if os.path.isfile(os.path.join(images_dir, f))}
    
    # Update file paths in JSON
    updated_count = 0
    for img in data['images']:
        original_path = img['file_name']
        basename = os.path.basename(original_path)
        
        # Check if image exists in the images directory
        if basename in actual_images:
            # Update to relative path from project root
            img['file_name'] = os.path.join(images_dir, basename)
            updated_count += 1
    
    # Save updated JSON
    fixed_json_path = coco_json_path.replace('.json', '_fixed.json')
    with open(fixed_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Fixed {updated_count} image paths")
    print(f"Saved fixed JSON to: {fixed_json_path}")
    
    return fixed_json_path


def quick_start(label_studio_output_dir, output_base='data'):
    """
    Quick start function to process Label Studio output.
    
    Args:
        label_studio_output_dir: Directory containing Label Studio output
        output_base: Base directory for processed output
    """
    # Find result.json
    result_json = os.path.join(label_studio_output_dir, 'result.json')
    if not os.path.exists(result_json):
        print(f"Error: result.json not found in {label_studio_output_dir}")
        return False
    
    # Find images directory
    images_dir = os.path.join(label_studio_output_dir, 'images')
    if not os.path.exists(images_dir):
        print(f"Error: images directory not found in {label_studio_output_dir}")
        return False
    
    print("=" * 60)
    print("Quick Start: Processing Label Studio Output")
    print("=" * 60)
    
    # Fix paths in JSON if needed
    print("\nStep 1: Checking and fixing image paths...")
    fixed_json = fix_label_studio_paths(result_json, images_dir)
    
    # Convert COCO to YOLO
    print("\nStep 2: Converting COCO to YOLO format...")
    yolo_annotations_dir = os.path.join(label_studio_output_dir, 'yolo_annotations')
    convert_coco_to_yolo(fixed_json, images_dir, yolo_annotations_dir)
    
    # Split dataset
    print("\nStep 3: Splitting dataset...")
    split_dataset(
        label_studio_output_dir,
        output_base,
        test_size=0.2,
        val_size=0.5,
        random_state=1
    )
    
    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Update yolov5/data/custom_logo.yaml with paths:")
    print(f"   train: {output_base}/suzuki_logo_detection/images/train")
    print(f"   val: {output_base}/suzuki_logo_detection/images/val")
    print(f"   test: {output_base}/suzuki_logo_detection/images/test")
    print(f"\n2. Train your model:")
    print(f"   python yolov5/train.py --img 640 --cfg yolov5s.yaml --hyp hyp.scratch-low.yaml \\")
    print(f"       --batch 16 --epochs 200 --data custom_logo.yaml --weights yolov5s.pt \\")
    print(f"       --workers 1 --name yolo_logo_detection")
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick start: process Label Studio output')
    parser.add_argument('--label_studio_dir', type=str, 
                        default='project-5-at-2026-02-06-06-49-4f3e5bf6',
                        help='Label Studio output directory')
    parser.add_argument('--output_base', type=str, default='data',
                        help='Base directory for processed output')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.label_studio_dir):
        print(f"Error: Directory not found: {args.label_studio_dir}")
        sys.exit(1)
    
    quick_start(args.label_studio_dir, args.output_base)
