"""
Convert COCO format annotations to YOLO format.
This script processes Label Studio COCO JSON exports and converts them to YOLO format.
"""

import cv2
import json
import os
import pandas as pd


def expand_bbox_coords(bbox):
    """Expand COCO bbox format [x, y, width, height] to [xmin, ymin, xmax, ymax]."""
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[0] + bbox[2]
    ymax = bbox[1] + bbox[3]
    return (xmin, ymin, xmax, ymax)


def coco_to_dict(img, annots_df):
    """
    Convert COCO image and annotations to dictionary format.
    
    Args:
        img: Image row from images dataframe
        annots_df: Annotations dataframe
        
    Returns:
        Dictionary with image info and bboxes
    """
    # Obtain relevant image data
    img_name = os.path.basename(img['file_name'])
    img_size = (img['width'], img['height'], 3)
    
    # Cut to relevant bbox annotations
    img_id = img['id']
    tags = annots_df[annots_df['image_id'] == img_id]

    # Iterate through bbox annotations
    bboxes = []
    for _, tag in tags.iterrows():
        bbox_annot = {
            'label': tag['category_id'],
            'xmin': tag['xmin'],
            'ymin': tag['ymin'],
            'xmax': tag['xmax'],
            'ymax': tag['ymax']
        }
        bboxes.append(bbox_annot)
       
    img_dict = {
        'bboxes': bboxes,
        'image_name': img_name,
        'image_size': img_size
    }
    
    return img_dict


def dict_to_yolo(img_dict):
    """
    Convert image dictionary to YOLO format annotation.
    
    Args:
        img_dict: Dictionary with image info and bboxes
        
    Returns:
        Tuple of (annotation_filename, annotation_lines)
    """
    img_name = img_dict['image_name']
    img_width, img_height, img_depth = img_dict['image_size']

    annot_txt = []
    for box in img_dict['bboxes']:
        # Extract abs bbox info
        lbl = box['label']
        x_centre = (box['xmin'] + box['xmax']) / 2
        y_centre = (box['ymin'] + box['ymax']) / 2
        width = box['xmax'] - box['xmin']
        height = box['ymax'] - box['ymin']

        # Convert bbox info to relative [0,1]
        x_centre = round(x_centre / img_width, 6)
        y_centre = round(y_centre / img_height, 6)
        width = round(width / img_width, 6)
        height = round(height / img_height, 6)

        annot_txt.append(" ".join([
            str(lbl), str(x_centre), str(y_centre), str(width), str(height)
        ]))

    annot_name = os.path.splitext(img_name)[0] + '.txt'

    return annot_name, annot_txt


def convert_coco_to_yolo(coco_json_path, images_dir, output_dir):
    """
    Main function to convert COCO JSON to YOLO format.
    
    Args:
        coco_json_path: Path to COCO JSON file
        images_dir: Directory containing images
        output_dir: Output directory for YOLO annotation files
    """
    # Create output directory if it doesn't exist
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Load the annotation set
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # Convert annotated images json to dataframe
    images = pd.DataFrame(data['images'])
    
    # Convert annotated labels json to dataframe
    annots = pd.DataFrame(data['annotations'])
    annots[['xmin', 'ymin', 'xmax', 'ymax']] = annots.apply(
        lambda x: expand_bbox_coords(x['bbox']), axis=1, result_type='expand'
    )

    # Convert categories json to dataframe
    labels = pd.DataFrame(data['categories'])

    print(f"Found {len(images)} images")
    print(f"Found {len(annots)} annotations")
    print(f"Found {len(labels)} categories:")
    for _, cat in labels.iterrows():
        print(f"  - ID {cat['id']}: {cat['name']}")

    # Convert COCO to YOLO format
    converted_count = 0
    for _, image in images.iterrows():
        # Extract COCO annotations to YOLO format
        image_dict = coco_to_dict(image, annots)
        file_name, file_txt = dict_to_yolo(image_dict)

        # Save the file
        output_path = os.path.join(output_dir, file_name)
        with open(output_path, 'w') as f:
            for entry in file_txt:
                f.write(f"{entry}\n")
        
        converted_count += 1

    print(f"\nConverted {converted_count} annotation files to YOLO format")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert COCO annotations to YOLO format')
    parser.add_argument('--coco_json', type=str, required=True,
                        help='Path to COCO JSON file')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for YOLO annotation files')
    
    args = parser.parse_args()
    
    convert_coco_to_yolo(args.coco_json, args.images_dir, args.output_dir)
