"""
Visualize annotations on images.
This script displays images with bounding boxes and labels overlaid.
"""

import cv2
import json
import os
import pandas as pd
import matplotlib.pyplot as plt


def expand_bbox_coords(bbox):
    """Expand COCO bbox format [x, y, width, height] to [xmin, ymin, xmax, ymax]."""
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[0] + bbox[2]
    ymax = bbox[1] + bbox[3]
    return (xmin, ymin, xmax, ymax)


def obtain_bbox_label(categories, bbox_tag):
    """Get label name from category ID."""
    label = categories[categories['id'] == bbox_tag['category_id']]['name'].item()
    return str(label)


def visualize_annotation(coco_json_path, images_dir, img_id=0, save_path=None):
    """
    Visualize annotations for a specific image.
    
    Args:
        coco_json_path: Path to COCO JSON file
        images_dir: Directory containing images
        img_id: Image ID to visualize (default: 0)
        save_path: Optional path to save the visualization
    """
    # Load the annotation set
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # Convert to dataframes
    images = pd.DataFrame(data['images'])
    annots = pd.DataFrame(data['annotations'])
    annots[['xmin', 'ymin', 'xmax', 'ymax']] = annots.apply(
        lambda x: expand_bbox_coords(x['bbox']), axis=1, result_type='expand'
    )
    labels = pd.DataFrame(data['categories'])

    # Get the entry for the relevant image id
    test_img = images[images['id'] == img_id]
    
    if len(test_img) == 0:
        print(f"Image ID {img_id} not found in dataset")
        return

    # Load image
    img_file = os.path.basename(test_img['file_name'].iloc[0])
    # Handle relative paths
    if not os.path.isabs(test_img['file_name'].iloc[0]):
        path = os.path.join(images_dir, img_file)
    else:
        path = test_img['file_name'].iloc[0]
    
    if not os.path.exists(path):
        # Try to find the image in the images directory
        path = os.path.join(images_dir, img_file)
        if not os.path.exists(path):
            print(f"Image not found: {path}")
            return

    image = cv2.imread(path)
    if image is None:
        print(f"Could not load image: {path}")
        return

    # Ensure we are using the correct colour spectrum when displaying
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Overlay relevant bounding boxes
    relevant_annots = annots[annots['image_id'] == img_id]
    for index, tag in relevant_annots.iterrows():
        # Display bbox
        cv2.rectangle(
            image,
            (int(tag.xmin), int(tag.ymin)),
            (int(tag.xmax), int(tag.ymax)),
            (255, 0, 0), 2
        )
        
        # Display text label
        text = obtain_bbox_label(labels, tag)
        cv2.putText(
            image, text, (int(tag.xmin), int(tag.ymin) - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (255, 0, 0), 2
        )

    # Display
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Image ID: {img_id} - {img_file}")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize annotations on images')
    parser.add_argument('--coco_json', type=str, required=True,
                        help='Path to COCO JSON file')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--img_id', type=int, default=0,
                        help='Image ID to visualize (default: 0)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Optional path to save the visualization')
    
    args = parser.parse_args()
    
    visualize_annotation(
        args.coco_json,
        args.images_dir,
        img_id=args.img_id,
        save_path=args.save_path
    )
