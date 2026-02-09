"""
Split dataset into train, validation, and test sets.
This script splits images and their corresponding YOLO annotation files.
"""

import os
import shutil
from sklearn.model_selection import train_test_split


def copy_files(list_img, list_annot, split, output_base):
    """
    Copy image and annotation files to split directories.
    
    Args:
        list_img: List of image file paths
        list_annot: List of annotation file paths
        split: Split name ('train', 'val', or 'test')
        output_base: Base output directory
    """
    # Copy the images over
    img_folder = os.path.join(output_base, 'images', split)
    if not os.path.isdir(img_folder):
        os.makedirs(img_folder)

    for x in list_img:
        shutil.copy(x, img_folder)

    # Copy the annotation files over
    annot_folder = os.path.join(output_base, 'labels', split)
    if not os.path.isdir(annot_folder):
        os.makedirs(annot_folder)

    for x in list_annot:
        shutil.copy(x, annot_folder)
    
    print(f"Copied {len(list_img)} images and {len(list_annot)} annotations to {split} split")
    
    return


def split_dataset(base_path, output_base, test_size=0.2, val_size=0.5, random_state=1):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        base_path: Base path containing images and yolo_annotations folders
        output_base: Output base directory for split datasets
        test_size: Proportion of dataset to use for test set
        val_size: Proportion of test set to use for validation (after test split)
        random_state: Random seed for reproducibility
    """
    # Get lists of images and annotations
    images_dir = os.path.join(base_path, 'images')
    annots_dir = os.path.join(base_path, 'yolo_annotations')
    
    if not os.path.isdir(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    if not os.path.isdir(annots_dir):
        raise ValueError(f"Annotations directory not found: {annots_dir}")
    
    image_list = [os.path.join(images_dir, x) for x in os.listdir(images_dir) 
                  if os.path.isfile(os.path.join(images_dir, x))]
    annot_list = [os.path.join(annots_dir, os.path.splitext(os.path.basename(img))[0] + '.txt')
                  for img in image_list]
    
    # Filter to only include files that exist
    valid_pairs = [(img, annot) for img, annot in zip(image_list, annot_list) 
                   if os.path.exists(annot)]
    image_list = [pair[0] for pair in valid_pairs]
    annot_list = [pair[1] for pair in valid_pairs]
    
    # Sort to ensure matching order
    image_list.sort()
    annot_list.sort()
    
    print(f"Found {len(image_list)} image-annotation pairs")
    
    # Split into train and test
    img_train, img_test, annot_train, annot_test = train_test_split(
        image_list, annot_list, test_size=test_size, random_state=random_state
    )
    
    # Split test into validation and test
    img_val, img_test, annot_val, annot_test = train_test_split(
        img_test, annot_test, test_size=val_size, random_state=random_state
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(img_train)} images")
    print(f"  Validation: {len(img_val)} images")
    print(f"  Test: {len(img_test)} images")
    
    # Copy files to new folders
    copy_files(img_train, annot_train, 'train', output_base)
    copy_files(img_val, annot_val, 'val', output_base)
    copy_files(img_test, annot_test, 'test', output_base)
    
    print(f"\nDataset split complete!")
    print(f"Output directory: {output_base}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test sets')
    parser.add_argument('--base_path', type=str, required=True,
                        help='Base path containing images and yolo_annotations folders')
    parser.add_argument('--output_base', type=str, required=True,
                        help='Output base directory for split datasets')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of dataset for test set (default: 0.2)')
    parser.add_argument('--val_size', type=float, default=0.5,
                        help='Proportion of test set for validation (default: 0.5)')
    parser.add_argument('--random_state', type=int, default=1,
                        help='Random seed for reproducibility (default: 1)')
    
    args = parser.parse_args()
    
    split_dataset(
        args.base_path,
        args.output_base,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )
