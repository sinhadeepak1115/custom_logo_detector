"""
Main script to prepare dataset: convert COCO to YOLO format and split dataset.
This script automates the entire dataset preparation pipeline.
"""

import os
import sys
import argparse
from convert_coco_to_yolo import convert_coco_to_yolo
from split_dataset import split_dataset


def prepare_dataset(coco_json_path, images_dir, output_base, test_size=0.2, val_size=0.5, random_state=1):
    """
    Complete dataset preparation pipeline.
    
    Args:
        coco_json_path: Path to COCO JSON file
        images_dir: Directory containing images
        output_base: Base directory for output (will create Label_Studio_Output structure)
        test_size: Proportion of dataset for test set
        val_size: Proportion of test set for validation
        random_state: Random seed for reproducibility
    """
    # Set up paths
    base_path = os.path.join(output_base, 'Label_Studio_Output')
    yolo_annotations_dir = os.path.join(base_path, 'yolo_annotations')
    
    # Step 1: Convert COCO to YOLO format
    print("=" * 60)
    print("Step 1: Converting COCO annotations to YOLO format")
    print("=" * 60)
    convert_coco_to_yolo(coco_json_path, images_dir, yolo_annotations_dir)
    
    # Step 2: Split dataset
    print("\n" + "=" * 60)
    print("Step 2: Splitting dataset into train/val/test")
    print("=" * 60)
    split_dataset(
        base_path,
        output_base,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"\nDataset structure:")
    print(f"  {output_base}/")
    print(f"    images/")
    print(f"      train/")
    print(f"      val/")
    print(f"      test/")
    print(f"    labels/")
    print(f"      train/")
    print(f"      val/")
    print(f"      test/")
    
    # Update data config file
    data_config_path = "data_config.yaml"
    if os.path.exists(data_config_path):
        print(f"\nDon't forget to update {data_config_path} with the correct paths:")
        print(f"  train: {output_base}/images/train")
        print(f"  val: {output_base}/images/val")
        print(f"  test: {output_base}/images/test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset: convert COCO to YOLO and split')
    parser.add_argument('--coco_json', type=str, required=True,
                        help='Path to COCO JSON file')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output_base', type=str, default='data',
                        help='Base directory for output (default: data)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of dataset for test set (default: 0.2)')
    parser.add_argument('--val_size', type=float, default=0.5,
                        help='Proportion of test set for validation (default: 0.5)')
    parser.add_argument('--random_state', type=int, default=1,
                        help='Random seed for reproducibility (default: 1)')
    
    args = parser.parse_args()
    
    prepare_dataset(
        args.coco_json,
        args.images_dir,
        args.output_base,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
